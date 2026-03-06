"""Verbal (text) report generation for the PHA RTL similarity analysis.

Produces a single ``report.txt`` containing numbered sections covering
dataset summary, similarity results, anomalies, surprise findings,
interface compatibility, PHA synthesis, diagram-string analysis, and
methodology notes.
"""

from __future__ import annotations

import datetime
import logging
import textwrap
from typing import List, Optional

import numpy as np
import pandas as pd

from .anomaly import Anomaly
from .config import AnalysisConfig
from .interface_analyzer import InterfaceReport
from .iverilog_analyzer import GATE_TYPES, ProjectAnalysis
from .pha_analyzer import PHAReport
from .similarity import SimilarityEngine
from .surprise import SurpriseReport

log = logging.getLogger(__name__)

_SECTION = "=" * 72
_SUBSEC  = "-" * 72


# ── Public entry point ────────────────────────────────────────────────

def write_verbal(
    cfg: AnalysisConfig,
    projects: List[ProjectAnalysis],
    df_combined: pd.DataFrame,
    df_ngram: pd.DataFrame,
    partial_matches: pd.DataFrame,
    top_pairs: pd.DataFrame,
    anomalies: List[Anomaly],
    surprise_report: Optional[SurpriseReport] = None,
    interface_report: Optional[InterfaceReport] = None,
    pha_report: Optional[PHAReport] = None,
) -> None:
    """Write ``report.txt`` to *cfg.output_dir*."""
    lines: List[str] = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    def h1(title: str) -> None:
        lines.extend(["", _SECTION, f"  {title}", _SECTION])

    def h2(title: str) -> None:
        lines.extend(["", _SUBSEC, f"  {title}", _SUBSEC])

    def para(text: str, width: int = 70) -> None:
        lines.append(textwrap.fill(text, width=width))

    # ── Header ────────────────────────────────────────────────────
    lines.append("POLYMORPHIC HARDWARE ACCELERATOR RTL SIMILARITY ANALYSIS")
    lines.append(f"Generated: {ts}")
    lines.append(f"Projects analysed: {len(projects)}")
    lines.append(
        f"Similarity weights: gate-type={cfg.gate_type_weight:.0%}, "
        f"structural={cfg.structural_weight:.0%}, "
        f"n-gram={cfg.partial_weight:.0%}"
    )

    # ── 0. Executive summary ──────────────────────────────────────
    _section_executive_summary(
        lines, h1, h2, para, cfg, projects,
        df_combined, top_pairs, anomalies,
        surprise_report, interface_report, pha_report,
    )

    # ── 1. Dataset summary ────────────────────────────────────────
    _section_dataset(lines, h1, h2, para, projects)

    # ── 2. Similarity results ─────────────────────────────────────
    _section_similarity(lines, h1, h2, para, cfg, df_combined, top_pairs)

    # ── 3. Partial gate-pattern matching ──────────────────────────
    _section_partial(lines, h1, h2, para, partial_matches, df_ngram)

    # ── 4. Gate-type distribution ─────────────────────────────────
    _section_gate_distribution(lines, h1, projects)

    # ── 5. Anomalies ──────────────────────────────────────────────
    _section_anomalies(lines, h1, para, cfg, anomalies)

    # ── 6. Surprise findings ──────────────────────────────────────
    _section_surprise(lines, h1, h2, para, surprise_report)

    # ── 7. Interface compatibility ────────────────────────────────
    _section_interface(lines, h1, h2, para, projects, interface_report)

    # ── 8. PHA synthesis ──────────────────────────────────────────
    _section_pha(lines, h1, h2, para, pha_report)

    # ── 8b. Diagram-string analysis ───────────────────────────────
    _section_diagram_strings(lines, h1, h2, para, pha_report)

    # ── 9. Methodology ────────────────────────────────────────────
    _section_methodology(lines, h1, para)

    lines.extend(["", _SECTION])

    report_path = cfg.output_dir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Verbal report → %s", report_path)

    # Also write a concise executive_summary.txt for quick reading
    _write_executive_summary_file(
        cfg, projects, df_combined, top_pairs, anomalies,
        surprise_report, interface_report, pha_report,
    )


# ── Section helpers ───────────────────────────────────────────────────

def _section_executive_summary(
    lines, h1, h2, para, cfg, projects,
    df_combined, top_pairs, anomalies,
    surprise_report, interface_report, pha_report,
):
    """Section 0: one-page executive summary covering all major findings."""
    h1("EXECUTIVE SUMMARY")
    para(
        "This section provides a concise overview of all analysis results "
        "so you can quickly understand the key findings without reading the "
        "full report."
    )

    total_gates = sum(p.total_gates for p in projects)
    total_lines = sum(p.line_count for p in projects)
    iverilog_cnt = sum(1 for p in projects if p.used_iverilog)

    h2("Dataset at a Glance")
    lines.append(f"  Projects analysed     : {len(projects)}")
    lines.append(f"  Compiled with iverilog: {iverilog_cnt} / {len(projects)}")
    lines.append(f"  Total source lines    : {total_lines:,}")
    lines.append(f"  Total gates detected  : {total_gates:,}")

    # Similarity highlights
    if not top_pairs.empty:
        h2("Similarity Highlights")
        mean_sim = float(df_combined.values[np.triu_indices_from(df_combined.values, k=1)].mean())
        top_row = top_pairs.iloc[0]
        lines.append(f"  Mean pairwise similarity : {mean_sim:.3f}")
        lines.append(
            f"  Most similar pair        : {top_row.get('project_a', '?')} "
            f"↔ {top_row.get('project_b', '?')} "
            f"(score {top_row.get('similarity', 0):.4f})"
        )

    # Anomalies
    h2("Anomaly Detection")
    if anomalies:
        lines.append(f"  {len(anomalies)} project(s) flagged as statistical outliers:")
        for a in anomalies[:5]:
            lines.append(f"    • {a.project_name} (Z={a.zscore:.2f}, driver: {list(a.flagged_features.keys())[0]})")
    else:
        lines.append("  No anomalies detected.")

    # Surprise
    h2("Cross-Domain Surprise")
    if surprise_report and surprise_report.pairs:
        top_s = surprise_report.pairs[0]
        lines.append(f"  {len(surprise_report.pairs)} surprising cross-domain pair(s).")
        lines.append(
            f"  Top surprise: {top_s.project_a} ↔ {top_s.project_b} "
            f"(score {top_s.surprise_score:.3f})"
        )
    else:
        lines.append("  No surprising cross-domain pairs found (or skipped).")

    # Interface compatibility
    h2("Interface Compatibility")
    if interface_report and interface_report.compatible_pairs:
        top_i = interface_report.compatible_pairs[0]
        lines.append(f"  {len(interface_report.compatible_pairs)} compatible module pair(s).")
        lines.append(
            f"  Best match: {top_i.module_a.project}::{top_i.module_a.name} ↔ "
            f"{top_i.module_b.project}::{top_i.module_b.name} "
            f"(norm={top_i.name_match_score:.2f}, effort={top_i.effort})"
        )
    else:
        lines.append("  No compatible pairs found (or skipped).")

    # PHA synthesis
    h2("PHA Synthesis")
    if pha_report and pha_report.clusters:
        lines.append(f"  {len(pha_report.clusters)} PHA cluster(s) proposed:")
        for c in pha_report.clusters[:3]:
            lines.append(
                f"    • Cluster {c.cluster_id}: {len(c.member_projects)} DSAs, "
                f"{len(c.shared_components)} shared + "
                f"{len(c.merged_components)} merged components, "
                f"{c.area_savings_pct:.1f}% area savings"
            )
    else:
        lines.append("  No PHA clusters proposed (or skipped).")


def _write_executive_summary_file(
    cfg, projects, df_combined, top_pairs, anomalies,
    surprise_report, interface_report, pha_report,
):
    """Write a standalone executive_summary.txt for quick reading."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    total_gates = sum(p.total_gates for p in projects)
    total_lines = sum(p.line_count for p in projects)
    iverilog_cnt = sum(1 for p in projects if p.used_iverilog)

    s = []
    s.append("=" * 72)
    s.append("  POLYMORPHIC HARDWARE ACCELERATOR — EXECUTIVE SUMMARY")
    s.append("=" * 72)
    s.append(f"Generated: {ts}")
    s.append("")
    s.append(f"Projects: {len(projects)}  |  Lines: {total_lines:,}  |  "
             f"Gates: {total_gates:,}  |  iverilog: {iverilog_cnt}/{len(projects)}")
    s.append("")

    if not top_pairs.empty:
        mean_sim = float(df_combined.values[np.triu_indices_from(df_combined.values, k=1)].mean())
        s.append(f"Mean similarity: {mean_sim:.3f}")
        tr = top_pairs.iloc[0]
        s.append(f"Most similar:    {tr.get('project_a', '?')} ↔ "
                 f"{tr.get('project_b', '?')} ({tr.get('similarity', 0):.4f})")
    s.append("")

    if anomalies:
        s.append(f"Anomalies: {len(anomalies)} project(s)")
        for a in anomalies[:3]:
            s.append(f"  • {a.project_name} (Z={a.zscore:.2f})")
    else:
        s.append("Anomalies: none")

    if surprise_report and surprise_report.pairs:
        ts_ = surprise_report.pairs[0]
        s.append(f"\nSurprise pairs: {len(surprise_report.pairs)}")
        s.append(f"  Top: {ts_.project_a} ↔ {ts_.project_b} ({ts_.surprise_score:.3f})")
    else:
        s.append("\nSurprise: none/skipped")

    if interface_report and interface_report.compatible_pairs:
        ti = interface_report.compatible_pairs[0]
        s.append(f"\nInterface matches: {len(interface_report.compatible_pairs)}")
        s.append(f"  Best: {ti.module_a.project}::{ti.module_a.name} ↔ "
                 f"{ti.module_b.project}::{ti.module_b.name} "
                 f"(norm={ti.name_match_score:.2f})")
    else:
        s.append("\nInterface: none/skipped")

    if pha_report and pha_report.clusters:
        s.append(f"\nPHA clusters: {len(pha_report.clusters)}")
        for c in pha_report.clusters[:3]:
            s.append(
                f"  Cluster {c.cluster_id}: {len(c.member_projects)} DSAs, "
                f"{c.area_savings_pct:.1f}% savings"
            )
    else:
        s.append("\nPHA: none/skipped")

    s.append("")
    s.append("=" * 72)
    s.append(f"Full report: report.txt   |   Config: config.yaml")
    s.append("=" * 72)

    path = cfg.output_dir / "executive_summary.txt"
    path.write_text("\n".join(s), encoding="utf-8")
    log.info("Executive summary → %s", path)

def _section_dataset(lines, h1, h2, para, projects):
    h1("1. DATASET SUMMARY")
    iverilog_used = sum(1 for p in projects if p.used_iverilog)
    para(
        f"The dataset comprises {len(projects)} RTL projects harvested from "
        f"local directories. Gate-level netlist extraction via Icarus Verilog "
        f"(iverilog) succeeded for {iverilog_used} out of {len(projects)} "
        f"projects; the remainder were analysed using a regex-based RTL scanner."
    )

    total_gates = sum(p.total_gates for p in projects)
    total_lines = sum(p.line_count for p in projects)
    lines.append("")
    lines.append(f"  Total source lines : {total_lines:,}")
    lines.append(f"  Total gate count   : {total_gates:,}")
    lines.append(f"  Avg gates/project  : {total_gates / max(1, len(projects)):,.0f}")

    h2("2a. Per-Project Statistics")
    col_w = max(len(p.name) for p in projects) + 2
    header = (
        f"{'Project':<{col_w}} {'Files':>6} {'Lines':>8} "
        f"{'Gates':>8} {'DFF':>6} {'AND':>6} {'Nodes':>7} {'Edges':>7} "
        f"{'Backend':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for p in sorted(projects, key=lambda x: x.total_gates, reverse=True):
        be = "iverilog" if p.used_iverilog else "regex"
        lines.append(
            f"{p.name:<{col_w}} {len(p.source_files):>6} {p.line_count:>8,} "
            f"{p.total_gates:>8,} {p.gate_counts.get('DFF', 0):>6} "
            f"{p.gate_counts.get('AND', 0):>6} "
            f"{p.graph.number_of_nodes():>7} {p.graph.number_of_edges():>7} "
            f"{be:>10}"
        )


def _section_similarity(lines, h1, h2, para, cfg, df_combined, top_pairs):
    h1("2. SIMILARITY ANALYSIS")
    para(
        "Pairwise combined similarity is computed as a weighted combination: "
        f"{cfg.gate_type_weight:.0%} gate-type cosine similarity "
        f"(distribution of AND/OR/NOT/XOR/DFF/… gate types) and "
        f"{cfg.structural_weight:.0%} structural graph similarity "
        "(density, clustering, degree-entropy, component count)."
    )

    h2("Top-15 Most Similar Project Pairs")
    lines.append(top_pairs.to_string(float_format=lambda x: f"{x:.4f}"))

    vals = df_combined.values.copy()
    np.fill_diagonal(vals, np.nan)
    flat = vals[~np.isnan(vals)]
    lines.append("")
    lines.append("  Matrix statistics (off-diagonal):")
    lines.append(f"    Mean     : {np.nanmean(flat):.4f}")
    lines.append(f"    Median   : {np.nanmedian(flat):.4f}")
    lines.append(f"    Std dev  : {np.nanstd(flat):.4f}")
    lines.append(f"    Min      : {np.nanmin(flat):.4f}")
    lines.append(f"    Max      : {np.nanmax(flat):.4f}")


def _section_partial(lines, h1, h2, para, partial_matches, df_ngram):
    h1("3. PARTIAL GATE-PATTERN MATCHING")
    para(
        "Gate n-gram Jaccard similarity measures shared sub-circuit patterns "
        "between projects. N-grams are short sequences of gate types extracted "
        "from graph traversal (e.g. AND\u2192XOR\u2192DFF). The Jaccard score is "
        "intersection/union of weighted n-gram bags — a high score indicates "
        "common design idioms or potential IP reuse even when total gate counts differ."
    )

    h2("Top Shared Gate Patterns (Project Pairs)")
    if not partial_matches.empty:
        pair_summary = (
            partial_matches.groupby(["project_a", "project_b", "jaccard_similarity"])
            ["shared_gate_pattern"]
            .apply(lambda x: ", ".join(x.head(4)))
            .reset_index()
            .sort_values("jaccard_similarity", ascending=False)
        )
        lines.append(pair_summary.to_string(index=False))
        lines.append("")
        lines.append("Full shared-pattern breakdown: see partial_gate_patterns.csv")
    else:
        lines.append("  No shared gate patterns detected between any project pair.")

    h2("N-gram Jaccard Similarity — Top Pairs")
    lines.append(SimilarityEngine.top_pairs(df_ngram, n=10).to_string(
        float_format=lambda x: f"{x:.4f}"
    ))


def _section_gate_distribution(lines, h1, projects):
    h1("4. GATE-TYPE DISTRIBUTION")
    gate_totals = {
        g: sum(p.gate_counts.get(g, 0) for p in projects) for g in GATE_TYPES
    }
    total = sum(gate_totals.values()) or 1
    for g, cnt in sorted(gate_totals.items(), key=lambda x: -x[1]):
        bar = "#" * int(40 * cnt / total)
        lines.append(f"  {g:<12} {cnt:>8,}  ({100 * cnt / total:5.1f}%)  {bar}")


def _section_anomalies(lines, h1, para, cfg, anomalies):
    h1("5. ANOMALY DETECTION")
    if anomalies:
        para(
            f"Anomaly detection (robust Z-score threshold "
            f"{cfg.anomaly_zscore_threshold}) flagged "
            f"{len(anomalies)} project(s) as statistical outliers:"
        )
        for a in anomalies:
            lines.append(f"\n  [{a.project_name}]  Z-score: {a.zscore:.2f}")
            lines.append(f"    {a.description}")
    else:
        para(
            "No anomalies detected. All projects fall within "
            f"{cfg.anomaly_zscore_threshold} robust standard deviations "
            "of the feature-space median."
        )


def _section_surprise(lines, h1, h2, para, surprise_report):
    h1("6. SURPRISING SIMILARITIES (CROSS-DOMAIN ANALYSIS)")
    if surprise_report and surprise_report.pairs:
        para(
            "The following project pairs are structurally similar despite "
            "coming from different hardware domains.  The surprise score "
            "= combined_similarity × domain_dissimilarity; pairs near 1.0 "
            "are the most unexpected findings.  Shared gate-sequence patterns "
            "are mapped to known functional circuit blocks to explain WHY "
            "the similarity exists."
        )
        lines.append("")

        h2("Domain Classification")
        col_w = max(len(n) for n in surprise_report.domain_map) + 2
        lines.append(f"  {'Project':<{col_w}} Domain")
        lines.append(f"  {'-' * col_w} {'-' * 30}")
        for proj, domain in sorted(surprise_report.domain_map.items()):
            lines.append(f"  {proj:<{col_w}} {domain}")

        lines.append("")
        h2("Top Surprising Pairs (ranked by surprise score)")

        for rank, sp in enumerate(surprise_report.pairs, 1):
            lines.append(f"\n  [{rank}] {sp.project_a}  \u2194  {sp.project_b}")
            lines.append(f"      Domain A : {sp.domain_a}")
            lines.append(f"      Domain B : {sp.domain_b}")
            lines.append(
                f"      Similarity : {sp.combined_similarity:.4f}   "
                f"Domain distance : {sp.domain_dissimilarity:.2f}   "
                f"Surprise score : {sp.surprise_score:.4f}"
            )
            if sp.pattern_interpretations:
                lines.append("      Shared functional blocks identified:")
                seen: set = set()
                for pat, meaning in sp.pattern_interpretations[:5]:
                    if meaning in seen:
                        continue
                    seen.add(meaning)
                    lines.append(f"        \u2022 [{pat}]  {meaning}")
            lines.append("")
            for expl_line in sp.explanation.split("  "):
                if expl_line.strip():
                    lines.append(
                        "      " + textwrap.fill(
                            expl_line.strip(), width=68,
                            subsequent_indent="      "
                        )
                    )
            lines.append("")
    elif surprise_report:
        para(
            "No cross-domain pairs exceeded the surprise score threshold. "
            "This may indicate that all projects in the current dataset "
            "share the same hardware domain, or that no pair has both high "
            "structural similarity and high domain dissimilarity. "
            "Try adding projects from more diverse domains (e.g. mix CPUs, "
            "image codecs, crypto cores, and communication IPs)."
        )
    else:
        lines.append("  (Surprise analysis was not run.)")


def _section_interface(lines, h1, h2, para, projects, interface_report):
    h1("7. INTERFACE COMPATIBILITY & REUSABILITY ANALYSIS")
    if interface_report and interface_report.compatible_pairs:
        total_mods = len(interface_report.modules)
        total_pairs = len(interface_report.compatible_pairs)
        proto_projects = {
            proto: projs
            for proto, projs in interface_report.protocol_coverage.items()
            if projs
        }
        para(
            f"Port-level interface analysis parsed {total_mods} module "
            f"declarations across {len(projects)} projects and identified "
            f"{total_pairs} cross-project module pairs with \u226540\u202f% port "
            f"compatibility after signal-name normalisation.  These pairs "
            f"can potentially be reused in each other\u2019s designs by "
            f"changing only the top-level signal connections."
        )

        if proto_projects:
            lines.append("")
            h2("Standard Protocol Coverage")
            para(
                "Modules speaking documented bus protocols are directly "
                "interoperable on a shared bus without any adapter layer."
            )
            for proto, projs in sorted(
                proto_projects.items(), key=lambda kv: -len(kv[1])
            ):
                lines.append(
                    f"  {proto:<20} detected in {len(projs)} project(s): "
                    + ", ".join(projs[:6])
                    + ("\u2026" if len(projs) > 6 else "")
                )

        lines.append("")
        h2("Top Reusable Modules (by reusability score)")
        df_r = interface_report.reusability_scores
        top_mods = df_r.head(15)
        col_mod  = max(len(str(r["module"]))  for _, r in top_mods.iterrows()) + 2
        col_proj = max(len(str(r["project"])) for _, r in top_mods.iterrows()) + 2
        lines.append(
            f"  {'Module':<{col_mod}} {'Project':<{col_proj}} "
            f"{'Score':>6}  {'Ports':>5}  {'Protocol'}"
        )
        lines.append(f"  {'-' * (col_mod + col_proj + 30)}")
        for _, row in top_mods.iterrows():
            lines.append(
                f"  {str(row['module']):<{col_mod}} "
                f"{str(row['project']):<{col_proj}} "
                f"{row['reusability_score']:>6.3f}  "
                f"{int(row['port_count']):>5}  "
                f"{row['protocols'] or '(no standard protocol)'}"
            )

        lines.append("")
        h2("Top Compatible Module Pairs (ranked by port match score)")
        effort_order = {"zero": 0, "low": 1, "medium": 2, "high": 3}
        sorted_pairs = sorted(
            interface_report.compatible_pairs[:20],
            key=lambda cp: (effort_order[cp.effort], -cp.name_match_score)
        )
        for rank, cp in enumerate(sorted_pairs, 1):
            lines.append(
                f"\n  [{rank}] {cp.module_a.project}::{cp.module_a.name}  "
                f"\u2194  {cp.module_b.project}::{cp.module_b.name}"
            )
            lines.append(
                f"      Compatibility : {cp.name_match_score:.1%}  "
                f"(exact={cp.port_match_score:.1%})   "
                f"Effort : {cp.effort.upper()}"
            )
            if cp.shared_protocols:
                lines.append(
                    f"      Shared protocol: {', '.join(cp.shared_protocols)}"
                )
            if cp.rename_recipe:
                top_r = cp.rename_recipe[:4]
                lines.append(
                    f"      Rename recipe ({len(cp.rename_recipe)} signal(s)): "
                    + ", ".join(f"'{a}'\u2192'{b}'" for a, b in top_r)
                    + ("\u2026" if len(cp.rename_recipe) > 4 else "")
                )
            if cp.resize_recipe:
                top_w = cp.resize_recipe[:3]
                lines.append(
                    f"      Width changes  ({len(cp.resize_recipe)} signal(s)): "
                    + ", ".join(f"'{n}' {wa}b\u2192{wb}b" for n, wa, wb in top_w)
                )
            lines.append("")
            for expl_line in cp.explanation.split("  "):
                if expl_line.strip():
                    lines.append(
                        "      " + textwrap.fill(
                            expl_line.strip(), width=68,
                            subsequent_indent="      ",
                        )
                    )
            lines.append("")
    elif interface_report:
        para(
            "No cross-project module pairs exceeded the compatibility threshold. "
            "All modules may have idiosyncratic port naming conventions or "
            "very different port counts.  Consider lowering --min-compat."
        )
    else:
        lines.append("  (Interface analysis was not run.)")


def _section_pha(lines, h1, h2, para, pha_report):
    h1("8. POLYMORPHIC HETEROGENEOUS ARCHITECTURE (PHA) SYNTHESIS")
    if pha_report and pha_report.clusters:
        para(
            f"The PHA synthesis analysis clustered {pha_report.total_projects} "
            f"Domain-Specific Accelerators into {len(pha_report.clusters)} "
            f"candidate Polymorphic Heterogeneous Architecture(s).  For each "
            f"cluster, shared gate-pattern components are instantiated once, "
            f"similar components are merged into configurable blocks with "
            f"mode-select multiplexers, and unique components are retained "
            f"behind a configuration layer.  Per-DSA interconnect interfaces "
            f"are unified into a single polymorphic port block."
        )
        lines.append(f"  Clustering threshold : {pha_report.clustering_threshold:.2f}")
        lines.append(f"  Merge Jaccard threshold : {pha_report.merge_jaccard_threshold:.2f}")
        if pha_report.unclustered_projects:
            lines.append(
                f"  Unclustered projects ({len(pha_report.unclustered_projects)}): "
                + ", ".join(pha_report.unclustered_projects[:10])
                + ("\u2026" if len(pha_report.unclustered_projects) > 10 else "")
            )

        for c in pha_report.clusters:
            h2(f"PHA Cluster {c.cluster_id} \u2014 {len(c.member_projects)} DSAs")
            lines.append(f"  Members : {', '.join(c.member_projects)}")
            lines.append(
                f"  Domains : {', '.join(sorted(set(c.member_domains.values())))}"
            )
            lines.append(f"  Avg similarity : {c.avg_similarity:.4f}")
            lines.append("")

            lines.append("  Component Decomposition:")
            lines.append(
                f"    Shared (identical across all members) : "
                f"{len(c.shared_components)} block(s)"
            )
            if c.shared_components:
                for fc in c.shared_components[:8]:
                    label = f" \u2014 {fc.semantic_label}" if fc.semantic_label else ""
                    lines.append(
                        f"      \u2022 [{fc.pattern_str}]{label}  "
                        f"(\u00d7{sum(fc.occurrences.values())} total)"
                    )

            lines.append(
                f"    Merged (folded with mux) : "
                f"{len(c.merged_components)} pair(s)"
            )
            if c.merged_components:
                for mc in c.merged_components[:5]:
                    lines.append(
                        f"      \u2022 {mc.merged_name}  "
                        f"Jaccard={mc.jaccard:.3f}  "
                        f"mux overhead={mc.mux_overhead:.1%}  "
                        f"projects: {', '.join(mc.projects_involved)}"
                    )

            total_unique = sum(len(v) for v in c.unique_components.values())
            lines.append(
                f"    Unique (DSA-specific) : {total_unique} block(s) "
                f"across {len(c.unique_components)} project(s)"
            )
            for pname, comps in list(c.unique_components.items())[:3]:
                lines.append(f"      {pname}: {len(comps)} block(s)")
                for fc in comps[:3]:
                    label = f" \u2014 {fc.semantic_label}" if fc.semantic_label else ""
                    lines.append(f"        \u2022 [{fc.pattern_str}]{label}")

            if c.unified_interface:
                ui = c.unified_interface
                lines.append("")
                lines.append("  Unified Interface (J+K+L \u2192 J+K+L):")
                lines.append(
                    f"    Common signals (pass-through) : "
                    f"{len(ui.common_signals)}"
                )
                lines.append(
                    f"    Muxed signals (mode-dependent) : "
                    f"{len(ui.muxed_signals)}"
                )
                lines.append(f"    Total port count : {ui.total_port_count}")
                if ui.protocol:
                    lines.append(f"    Common protocol : {ui.protocol}")

            lines.append("")
            lines.append("  Resource Estimation:")
            lines.append(
                f"    Individual DSA total : {c.individual_gate_total:,} gates"
            )
            lines.append(
                f"    PHA estimated total  : {c.pha_gate_total:,} gates"
            )
            lines.append(
                f"    Area savings         : {c.area_savings_pct:.1f}%"
            )
            lines.append(
                f"    Breakdown \u2014 shared: {c.shared_gate_count:,}  "
                f"merged: {c.merged_gate_count:,}  "
                f"unique: {c.unique_gate_count:,}"
            )

            lines.append("")
            for expl_line in c.explanation.split("  "):
                if expl_line.strip():
                    lines.append(
                        "    " + textwrap.fill(
                            expl_line.strip(), width=68,
                            subsequent_indent="    "
                        )
                    )
            lines.append("")
    elif pha_report:
        para(
            "No PHA clusters could be formed.  No project pair exceeded "
            f"the clustering threshold of {pha_report.clustering_threshold:.2f}.  "
            "Consider lowering --pha-threshold or adding more structurally "
            "similar projects."
        )
    else:
        lines.append("  (PHA synthesis analysis was not run.)")


def _section_diagram_strings(lines, h1, h2, para, pha_report):
    has_diagrams = (
        pha_report and pha_report.clusters
        and any(c.diagram_strings for c in pha_report.clusters)
    )
    if not has_diagrams:
        return

    h1("8b. DIAGRAM-STRING PATTERN RECOGNITION")
    para(
        "Each project\u2019s gate-level architecture was serialised into a "
        "structured text representation (\u2018diagram string\u2019).  Pairwise "
        "comparisons use Longest Common Subsequence (LCS), contiguous "
        "substring matching, Needleman\u2013Wunsch global sequence alignment, "
        "and functional-block annotation to identify shared and mergeable "
        "components at the string level."
    )
    for c in pha_report.clusters:
        if not c.diagram_strings:
            continue
        h2(f"Cluster {c.cluster_id} \u2014 Diagram Strings")
        for pname, ds in c.diagram_strings.items():
            lines.append(f"  {pname}:")
            lines.append(f"    Domain            : {ds.domain}")
            lines.append(f"    Tokens            : {len(ds.tokens)}")
            lines.append(f"    Compact (head 60) : {ds.compact[:60]}")
            lines.append(f"    Blocks  (head 80) : {ds.block_sequence[:80]}")
            lines.append(f"    Interface         : {ds.interface_summary[:80]}")
            lines.append("")

        if c.diagram_comparisons:
            h2(f"Cluster {c.cluster_id} \u2014 String Comparisons")
            for cmp in c.diagram_comparisons:
                lines.append(f"  {cmp.project_a} \u2194 {cmp.project_b}:")
                lines.append(
                    f"    LCS ratio           : {cmp.lcs_ratio:.4f}  "
                    f"({cmp.lcs_length} tokens)"
                )
                lines.append(
                    f"    Alignment identity  : "
                    f"{cmp.alignment.identity_pct:.1f}%  "
                    f"(matches={cmp.alignment.matches}, "
                    f"mismatches={cmp.alignment.mismatches}, "
                    f"gaps={cmp.alignment.gaps})"
                )
                lines.append(
                    f"    String similarity   : "
                    f"{cmp.overall_string_similarity:.4f}"
                )
                lines.append(
                    f"    Longest common substr: "
                    f"'{cmp.longest_common_substr.pattern}' "
                    f"(len={cmp.longest_common_substr.length}"
                    + (f", label={cmp.longest_common_substr.semantic_label}"
                       if cmp.longest_common_substr.semantic_label else "")
                    + ")"
                )
                if cmp.shared_substrings:
                    lines.append(
                        f"    Shared substrings   : "
                        f"{len(cmp.shared_substrings)}"
                    )
                    for ss in cmp.shared_substrings[:5]:
                        label = (
                            f" ({ss.semantic_label})" if ss.semantic_label else ""
                        )
                        lines.append(
                            f"      \u2022 [{ss.compact}]{label} len={ss.length}"
                        )
                if cmp.shared_blocks:
                    lines.append(
                        f"    Named shared blocks : "
                        + ", ".join(cmp.shared_blocks[:8])
                    )
                if cmp.mergeable_candidates:
                    lines.append(
                        f"    Merge candidates    : "
                        f"{len(cmp.mergeable_candidates)}"
                    )
                    for sub_a, sub_b, sim in cmp.mergeable_candidates[:3]:
                        lines.append(
                            f"      \u2022 A=[{sub_a}]  B=[{sub_b}]  "
                            f"sim={sim:.3f}"
                        )
                lines.append("    Alignment (head 60 cols):")
                lines.append(
                    f"      A: {cmp.alignment.aligned_a[:120]}"
                )
                lines.append(
                    f"         {cmp.alignment.match_line[:60]}"
                )
                lines.append(
                    f"      B: {cmp.alignment.aligned_b[:120]}"
                )
                lines.append("")

        if c.llm_analysis:
            h2(f"Cluster {c.cluster_id} \u2014 LLM Analysis")
            for llm_line in c.llm_analysis.split("\n"):
                lines.append(f"    {llm_line}")
            lines.append("")


def _section_methodology(lines, h1, para):
    h1("9. METHODOLOGY")
    para(
        "Gate-level information was extracted by compiling each project "
        "with Icarus Verilog (iverilog -g2012) and parsing the resulting "
        "VVP intermediate representation.  Each `.functor` directive encodes "
        "one logic gate and its four input connections; these are collected "
        "into a directed NetworkX graph.  For projects that do not compile "
        "cleanly, a regex scan of the raw RTL source files provides a "
        "fallback gate-type count."
    )
    lines.append("")
    para(
        "Three similarity metrics are computed: (1) cosine similarity over "
        "gate-type count vectors, (2) structural L1 similarity over graph "
        "topology features (density, clustering, degree-entropy, component "
        "count), and (3) weighted Jaccard similarity over gate-type n-gram "
        "bags (default n=3) extracted via BFS traversal of the gate graph. "
        "The combined score is a weighted average of all three.  "
        "The surprise score multiplies combined similarity by domain "
        "dissimilarity (Jaccard distance of domain label sets) to surface "
        "structurally similar but architecturally unrelated project pairs."
    )
