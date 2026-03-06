#!/usr/bin/env python3
"""RTL Similarity Analyzer — CLI entry point.

Usage examples
--------------
# Analyse projects that are already in verilog_proj/
python run.py

# Discover & clone from GitHub, then analyse
python run.py --github

# Specify a custom project directory
python run.py --project-dir /path/to/rtl_projects

# Only specific GitHub repos (comma-separated search query)
python run.py --github --github-query "topic:risc-v language:verilog stars:>200"

# Skip anomaly detection
python run.py --no-anomaly

# Verbose logging
python run.py -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import AnalysisConfig
from src.cache import AnalysisCache, project_content_hash, similarity_config_hash
from src.discovery import ProjectDiscovery
from src.iverilog_analyzer import IVerilogAnalyzer
from src.similarity import SimilarityEngine
from src.anomaly import AnomalyDetector
from src.surprise import SurpriseAnalyzer, SurpriseReport
from src.interface_analyzer import InterfaceAnalyzer, InterfaceReport
from src.pha_analyzer import PHAAnalyzer, PHAReport
from src.report import ReportGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTL gate-level similarity analysis with iverilog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--project-dir", default="verilog_proj",
        help="Directory containing RTL sub-projects (default: verilog_proj/)",
    )
    p.add_argument(
        "--output-dir", default="output",
        help="Base directory for reports and figures (default: output/). "
             "Each run creates a timestamped subfolder, e.g. output/run_2026-02-20_08-15-00.",
    )
    p.add_argument(
        "--github", action="store_true",
        help="Discover and clone Verilog projects from GitHub before analysis",
    )
    p.add_argument(
        "--github-query", default=None,
        help="Custom GitHub search query (overrides default)",
    )
    p.add_argument(
        "--github-token", default=None,
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    p.add_argument(
        "--github-max", type=int, default=30,
        help="Maximum number of NEW repos to clone per run (default: 30). "
             "Previously discovered repos are queued and cloned on future runs.",
    )
    p.add_argument(
        "--no-anomaly", action="store_true",
        help="Skip anomaly detection",
    )
    p.add_argument(
        "--no-surprise", action="store_true",
        help="Skip surprise / cross-domain analysis",
    )
    p.add_argument(
        "--min-surprise", type=float, default=0.15,
        help="Minimum surprise score to report a pair (default: 0.15)",
    )
    p.add_argument(
        "--no-interface", action="store_true",
        help="Skip interface / port-compatibility analysis",
    )
    p.add_argument(
        "--min-compat", type=float, default=0.40,
        help="Minimum normalised port-match score to report a pair (default: 0.40)",
    )
    p.add_argument(
        "--no-pha", action="store_true",
        help="Skip Polymorphic Heterogeneous Architecture (PHA) synthesis analysis",
    )
    p.add_argument(
        "--pha-threshold", type=float, default=0.35,
        help="Minimum pairwise similarity to cluster DSAs for PHA merging (default: 0.35)",
    )
    p.add_argument(
        "--pha-merge-jaccard", type=float, default=0.60,
        help="Minimum n-gram Jaccard to merge components in a PHA (default: 0.60)",
    )
    p.add_argument(
        "--diagram-strings", action="store_true",
        help="Enable diagram-string serialisation for PHA clusters.  Each project's "
             "architecture is converted to a structured text representation and compared "
             "using LCS, sequence alignment, and substring matching.",
    )
    p.add_argument(
        "--llm-api-key", default=None,
        help="OpenAI-compatible API key for LLM-assisted PHA analysis.  "
             "Alternatively set OPENAI_API_KEY env var.",
    )
    p.add_argument(
        "--llm-model", default="gpt-4o-mini",
        help="LLM model name for PHA analysis (default: gpt-4o-mini)",
    )
    p.add_argument(
        "--llm-base-url", default=None,
        help="Custom base URL for an OpenAI-compatible API endpoint",
    )
    p.add_argument(
        "--gate-weight", type=float, default=0.50,
        help="Weight for gate-type cosine similarity (default: 0.50)",
    )
    p.add_argument(
        "--struct-weight", type=float, default=0.25,
        help="Weight for structural graph similarity (default: 0.25)",
    )
    p.add_argument(
        "--partial-weight", type=float, default=0.25,
        help="Weight for partial gate-pattern n-gram Jaccard (default: 0.25)",
    )
    p.add_argument(
        "--ngram-n", type=int, default=3,
        help="Gate-sequence length for n-gram partial matching (default: 3)",
    )
    p.add_argument(
        "--zscore-threshold", type=float, default=2.0,
        help="Robust Z-score threshold for anomaly flagging (default: 2.0)",
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="Figure resolution in DPI (default: 300)",
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Disable caching — recompute everything from scratch",
    )
    p.add_argument(
        "--clear-cache", action="store_true",
        help="Delete the cache directory before running",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    return p.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    # Quieten noisy third-party loggers
    for lib in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    # ── Build config (YAML base → CLI overrides) ─────────────────────
    yaml_path = Path(args.config)
    try:
        cfg = AnalysisConfig.from_yaml(yaml_path)
        log.info("Loaded configuration from %s", yaml_path)
    except Exception:
        cfg = AnalysisConfig()

    # Apply CLI overrides (only explicitly-provided flags)
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.output_dir = Path(args.output_dir) / f"run_{run_stamp}"
    cfg.project_dir = Path(args.project_dir)
    cfg.github_max_repos = args.github_max
    cfg.gate_type_weight = args.gate_weight
    cfg.structural_weight = args.struct_weight
    cfg.partial_weight = args.partial_weight
    cfg.ngram_n = args.ngram_n
    cfg.anomaly_zscore_threshold = args.zscore_threshold
    cfg.figure_dpi = args.dpi
    cfg.verbose = args.verbose
    cfg.pha_threshold = args.pha_threshold
    cfg.pha_merge_jaccard = args.pha_merge_jaccard
    cfg.min_surprise = args.min_surprise
    cfg.min_compat = args.min_compat
    cfg.use_diagram_strings = args.diagram_strings
    cfg.llm_model = args.llm_model
    cfg.skip_anomaly = args.no_anomaly
    cfg.skip_surprise = args.no_surprise
    cfg.skip_interface = args.no_interface
    cfg.skip_pha = args.no_pha
    cfg.use_cache = not args.no_cache
    cfg.clear_cache = args.clear_cache
    if args.github_token:
        cfg.github_token = args.github_token
    if args.llm_api_key:
        cfg.llm_api_key = args.llm_api_key
    if args.llm_base_url:
        cfg.llm_base_url = args.llm_base_url
    cfg.__post_init__()

    # ── Discovery ─────────────────────────────────────────────────────
    discovery = ProjectDiscovery(cfg)

    if args.github:
        log.info("Discovering projects on GitHub …")
        discovery.discover_github(query=args.github_query, clone=True)

    project_paths = discovery.scan_local()
    if not project_paths:
        log.error(
            "No Verilog projects found in '%s'. "
            "Add projects there or use --github to clone some.",
            cfg.project_dir,
        )
        return 1

    log.info("Analysing %d project(s) …", len(project_paths))

    # ── Cache setup ───────────────────────────────────────────────────
    cache = AnalysisCache(cfg.cache_dir, enabled=cfg.use_cache)
    if cfg.clear_cache:
        removed = AnalysisCache.clear(cfg.cache_dir)
        log.info("Cleared cache (%d file(s) removed).", removed)
        cache = AnalysisCache(cfg.cache_dir, enabled=cfg.use_cache)

    # ── Gate-level analysis (with caching) ────────────────────────────
    from collections import Counter as _Counter
    from src.iverilog_analyzer import ProjectAnalysis as _PA
    analyzer = IVerilogAnalyzer(cfg)
    projects = []
    content_hashes: dict = {}  # project_name → content SHA
    cache_hits = 0

    for i, path in enumerate(project_paths, 1):
        log.info("[%d/%d] %s", i, len(project_paths), path.name)

        # Compute content hash for cache key
        source_files = analyzer._collect_sources(path)
        if not source_files:
            pa = _PA(name=path.name, path=path, error="No Verilog source files found")
            projects.append(pa)
            log.warning("  Skipped (no Verilog sources)")
            continue

        c_hash = project_content_hash(source_files)
        content_hashes[path.name] = c_hash

        # Try cache
        cached = None
        if cache.enabled:
            cached = cache.gates.load(path.name, c_hash)

        if cached is not None:
            pa = _PA(
                name=path.name,
                path=path,
                source_files=source_files,
                gate_counts=_Counter(cached["gate_counts"]),
                graph=cache.gates.to_graph(cached),
                modules=cached.get("modules", []),
                line_count=cached.get("line_count", 0),
                used_iverilog=cached.get("used_iverilog", False),
            )
            cache_hits += 1
            log.info(
                "  %s: %d gates, %d nodes, %d edges [cached]",
                pa.name, pa.total_gates,
                pa.graph.number_of_nodes(),
                pa.graph.number_of_edges(),
            )
        else:
            pa = analyzer.analyze_project(path)
            # Save to cache
            if cache.enabled and not pa.error:
                cache.gates.save(
                    pa.name, c_hash,
                    pa.gate_counts, pa.graph, pa.modules,
                    pa.line_count, pa.used_iverilog,
                )

        projects.append(pa)
        if pa.error:
            log.warning("  Skipped (%s)", pa.error)

    if cache_hits > 0:
        log.info("Cache: %d/%d project(s) loaded from cache.", cache_hits, len(project_paths))

    analysed = [p for p in projects if p.total_gates > 0 or p.graph.number_of_nodes() > 0]
    if len(analysed) < 2:
        log.error(
            "Need at least 2 projects with extractable gate information for comparison. "
            "Check that your Verilog files are present and parseable."
        )
        return 1

    # ── Similarity (with caching) ─────────────────────────────────────
    sim_engine = SimilarityEngine(cfg)
    sim_cache_key = similarity_config_hash(
        content_hashes,
        cfg.gate_type_weight, cfg.structural_weight,
        getattr(cfg, "partial_weight", 0.25), cfg.ngram_n,
    )

    cached_sim = None
    if cache.enabled:
        cached_sim = cache.similarity.load(sim_cache_key)

    if cached_sim is not None:
        df_combined, df_gate, df_struct, df_ngram = cached_sim
        log.info("Loaded similarity matrices from cache.")
    else:
        log.info("Computing pairwise similarity …")
        df_combined, df_gate, df_struct, df_ngram = sim_engine.compute_matrix(
            analysed, ngram_n=cfg.ngram_n
        )
        if cache.enabled:
            cache.similarity.save(
                sim_cache_key, df_combined, df_gate, df_struct, df_ngram,
            )

    log.info("Extracting partial gate-pattern matches …")
    partial_matches = sim_engine.partial_matches_table(
        analysed, ngram_n=cfg.ngram_n, top_pairs=15, top_patterns=8
    )

    # ── Surprise / cross-domain analysis ────────────────────────────
    surprise_report: Optional[SurpriseReport] = None
    if not cfg.skip_surprise and len(analysed) >= 3:
        log.info("Running cross-domain surprise analysis …")
        surprise_report = SurpriseAnalyzer(
            min_surprise=cfg.min_surprise,
            ngram_n=cfg.ngram_n,
        ).analyse(analysed, df_combined)
        if surprise_report.pairs:
            log.info(
                "Found %d surprising cross-domain pair(s) (top: %s ↔ %s, score=%.3f)",
                len(surprise_report.pairs),
                surprise_report.pairs[0].project_a,
                surprise_report.pairs[0].project_b,
                surprise_report.pairs[0].surprise_score,
            )
        else:
            log.info("No surprising pairs found above threshold %.2f.", cfg.min_surprise)

    # ── Interface compatibility analysis ────────────────────────────
    interface_report: Optional[InterfaceReport] = None
    if not cfg.skip_interface and len(analysed) >= 2:
        log.info("Running interface compatibility analysis …")
        interface_report = InterfaceAnalyzer(
            min_compatibility=cfg.min_compat,
            max_workers=cfg.max_workers,
            max_modules_per_project=cfg.max_modules_per_project,
        ).analyse(analysed)
        if interface_report.compatible_pairs:
            top = interface_report.compatible_pairs[0]
            log.info(
                "Found %d compatible module pairs (top: %s::%s ↔ %s::%s, "
                "norm=%.2f, effort=%s)",
                len(interface_report.compatible_pairs),
                top.module_a.project, top.module_a.name,
                top.module_b.project, top.module_b.name,
                top.name_match_score, top.effort,
            )
        else:
            log.info("No compatible module pairs found above threshold %.0f%%.",
                     cfg.min_compat * 100)

    # ── PHA synthesis analysis ─────────────────────────────────────────
    pha_report: Optional[PHAReport] = None
    if not cfg.skip_pha and len(analysed) >= 2:
        log.info("Running Polymorphic Heterogeneous Architecture (PHA) synthesis …")
        pha_report = PHAAnalyzer(
            cluster_threshold=cfg.pha_threshold,
            merge_jaccard=cfg.pha_merge_jaccard,
            ngram_n=cfg.ngram_n,
            use_diagram_strings=cfg.use_diagram_strings,
            max_diagram_cluster_size=cfg.max_diagram_cluster_size,
            llm_api_key=cfg.llm_api_key or "",
            llm_model=cfg.llm_model,
            llm_base_url=cfg.llm_base_url or "",
        ).analyse(analysed, df_combined, interface_report)
        if pha_report.clusters:
            top = pha_report.clusters[0]
            log.info(
                "Proposed %d PHA cluster(s) (top: %d DSAs, "
                "%.1f%% area savings)",
                len(pha_report.clusters),
                len(top.member_projects),
                top.area_savings_pct,
            )
        else:
            log.info("No PHA clusters found above threshold %.2f.",
                     cfg.pha_threshold)

    # ── Anomaly detection ─────────────────────────────────────────────
    anomalies = []
    if not cfg.skip_anomaly and len(analysed) >= 3:
        log.info("Running anomaly detection …")
        anomalies = AnomalyDetector(cfg).detect(analysed)
        if anomalies:
            log.info(
                "Flagged %d anomalous project(s): %s",
                len(anomalies),
                ", ".join(a.project_name for a in anomalies),
            )
        else:
            log.info("No anomalies detected.")

    # ── Reports ───────────────────────────────────────────────────────
    log.info("Generating reports in '%s' …", cfg.output_dir)
    reporter = ReportGenerator(cfg)
    reporter.generate(
        analysed, df_combined, df_gate, df_struct, df_ngram,
        partial_matches, anomalies, surprise_report, interface_report,
        pha_report,
    )

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
