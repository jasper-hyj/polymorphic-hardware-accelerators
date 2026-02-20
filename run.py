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
from src.discovery import ProjectDiscovery
from src.iverilog_analyzer import IVerilogAnalyzer
from src.similarity import SimilarityEngine
from src.anomaly import AnomalyDetector
from src.surprise import SurpriseAnalyzer, SurpriseReport
from src.interface_analyzer import InterfaceAnalyzer, InterfaceReport
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
        help="Maximum number of GitHub repos to clone (default: 30)",
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

    # ── Build config ──────────────────────────────────────────────────
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = Path(args.output_dir) / f"run_{run_stamp}"

    cfg = AnalysisConfig(
        project_dir=Path(args.project_dir),
        output_dir=run_output_dir,
        github_max_repos=args.github_max,
        gate_type_weight=args.gate_weight,
        structural_weight=args.struct_weight,
        partial_weight=args.partial_weight,
        ngram_n=args.ngram_n,
        anomaly_zscore_threshold=args.zscore_threshold,
        figure_dpi=args.dpi,
        verbose=args.verbose,
    )
    if args.github_token:
        cfg.github_token = args.github_token

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

    # ── Gate-level analysis ───────────────────────────────────────────
    analyzer = IVerilogAnalyzer(cfg)
    projects = []
    for i, path in enumerate(project_paths, 1):
        log.info("[%d/%d] %s", i, len(project_paths), path.name)
        pa = analyzer.analyze_project(path)
        projects.append(pa)
        if pa.error:
            log.warning("  Skipped (%s)", pa.error)

    analysed = [p for p in projects if p.total_gates > 0 or p.graph.number_of_nodes() > 0]
    if len(analysed) < 2:
        log.error(
            "Need at least 2 projects with extractable gate information for comparison. "
            "Check that your Verilog files are present and parseable."
        )
        return 1

    # ── Similarity ────────────────────────────────────────────────────
    log.info("Computing pairwise similarity …")
    sim_engine = SimilarityEngine(cfg)
    df_combined, df_gate, df_struct, df_ngram = sim_engine.compute_matrix(
        analysed, ngram_n=cfg.ngram_n
    )
    log.info("Extracting partial gate-pattern matches …")
    partial_matches = sim_engine.partial_matches_table(
        analysed, ngram_n=cfg.ngram_n, top_pairs=15, top_patterns=8
    )

    # ── Surprise / cross-domain analysis ────────────────────────────
    surprise_report: Optional[SurpriseReport] = None
    if not args.no_surprise and len(analysed) >= 3:
        log.info("Running cross-domain surprise analysis …")
        surprise_report = SurpriseAnalyzer(
            min_surprise=args.min_surprise,
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
            log.info("No surprising pairs found above threshold %.2f.", args.min_surprise)

    # ── Interface compatibility analysis ────────────────────────────
    interface_report: Optional[InterfaceReport] = None
    if not args.no_interface and len(analysed) >= 2:
        log.info("Running interface compatibility analysis …")
        interface_report = InterfaceAnalyzer(
            min_compatibility=args.min_compat,
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
                     args.min_compat * 100)

    # ── Anomaly detection ─────────────────────────────────────────────
    anomalies = []
    if not args.no_anomaly and len(analysed) >= 3:
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
    )

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
