"""Research-grade reporting: verbal, visual, and CSV outputs.

Orchestration module — delegates heavy lifting to:
  * :mod:`report_verbal`  — structured text report
  * :mod:`report_figures` — matplotlib/seaborn figures

Output directory layout
-----------------------
output/
  report.txt            — structured verbal report (human + LaTeX-ready)
  similarity_combined.csv
  similarity_gate_type.csv
  similarity_structural.csv
  gate_counts.csv
  anomalies.csv
  fig_heatmap_combined.png    — 300 DPI publication heatmap
  fig_heatmap_gate.png
  fig_gate_distribution.png   — stacked bar chart of gate-type mix
  fig_anomaly_scores.png      — Z-score bar chart (if anomalies exist)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")               # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from .anomaly import Anomaly, AnomalyDetector
from .config import AnalysisConfig
from .interface_analyzer import InterfaceAnalyzer, InterfaceReport
from .iverilog_analyzer import GATE_TYPES, ProjectAnalysis
from .pha_analyzer import PHAAnalyzer, PHAReport
from .similarity import SimilarityEngine
from .surprise import SurpriseAnalyzer, SurpriseReport

from . import report_figures
from . import report_verbal

log = logging.getLogger(__name__)


class ReportGenerator:
    """Write verbal, visual, and CSV reports to *cfg.output_dir*."""

    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(
            {
                "font.family":    cfg.font_family,
                "font.size":      cfg.font_size,
                "axes.titlesize": cfg.font_size + 2,
                "axes.labelsize": cfg.font_size,
                "xtick.labelsize": max(cfg.font_size - 2, 7),
                "ytick.labelsize": max(cfg.font_size - 2, 7),
                "figure.dpi":     cfg.figure_dpi,
                "savefig.dpi":    cfg.figure_dpi,
                "savefig.bbox":   "tight",
            }
        )

    # ── Entry point ───────────────────────────────────────────────────

    def generate(
        self,
        projects: List[ProjectAnalysis],
        df_combined: pd.DataFrame,
        df_gate: pd.DataFrame,
        df_struct: pd.DataFrame,
        df_ngram: pd.DataFrame,
        partial_matches: pd.DataFrame,
        anomalies: List[Anomaly],
        surprise_report: Optional[SurpriseReport] = None,
        interface_report: Optional[InterfaceReport] = None,
        pha_report: Optional[PHAReport] = None,
    ) -> None:
        if not projects:
            log.warning("No projects to report.")
            return

        top_pairs = SimilarityEngine.top_pairs(df_combined, n=15)

        # CSV tables
        self._write_csv(
            projects, df_combined, df_gate, df_struct, df_ngram,
            top_pairs, partial_matches, anomalies, surprise_report,
            interface_report, pha_report,
        )

        # Verbal text report
        report_verbal.write_verbal(
            self.cfg, projects, df_combined, df_ngram, partial_matches,
            top_pairs, anomalies, surprise_report, interface_report,
            pha_report,
        )

        # Figures
        report_figures.write_heatmaps(
            self.cfg, df_combined, df_gate, df_struct, df_ngram,
        )
        report_figures.write_gate_distribution(self.cfg, projects)
        report_figures.write_partial_match_chart(self.cfg, partial_matches)
        if surprise_report and surprise_report.pairs:
            report_figures.write_surprise_scatter(self.cfg, surprise_report)
        if interface_report and interface_report.compatible_pairs:
            report_figures.write_interface_chart(self.cfg, interface_report)
        if anomalies:
            report_figures.write_anomaly_chart(self.cfg, anomalies)
        if pha_report and pha_report.clusters:
            report_figures.write_pha_charts(self.cfg, pha_report)

        log.info("Reports written to %s", self.cfg.output_dir)

    # ── CSV outputs ───────────────────────────────────────────────────

    def _write_csv(
        self,
        projects: List[ProjectAnalysis],
        df_combined: pd.DataFrame,
        df_gate: pd.DataFrame,
        df_struct: pd.DataFrame,
        df_ngram: pd.DataFrame,
        top_pairs: pd.DataFrame,
        partial_matches: pd.DataFrame,
        anomalies: List[Anomaly],
        surprise_report: Optional[SurpriseReport] = None,
        interface_report: Optional[InterfaceReport] = None,
        pha_report: Optional[PHAReport] = None,
    ) -> None:
        out = self.cfg.output_dir

        df_combined.round(4).to_csv(out / "similarity_combined.csv")
        df_gate.round(4).to_csv(out / "similarity_gate_type.csv")
        df_struct.round(4).to_csv(out / "similarity_structural.csv")
        df_ngram.round(4).to_csv(out / "similarity_ngram_partial.csv")

        # Gate count table
        rows = []
        for p in projects:
            row = {"project": p.name, "total_gates": p.total_gates,
                   "source_files": len(p.source_files), "line_count": p.line_count,
                   "graph_nodes": p.graph.number_of_nodes(),
                   "graph_edges": p.graph.number_of_edges(),
                   "used_iverilog": p.used_iverilog}
            for g in GATE_TYPES:
                row[g] = p.gate_counts.get(g, 0)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "gate_counts.csv", index=False)

        top_pairs.to_csv(out / "top_similar_pairs.csv")

        if not partial_matches.empty:
            partial_matches.to_csv(out / "partial_gate_patterns.csv", index=False)

        if anomalies:
            AnomalyDetector(self.cfg).to_dataframe(anomalies).to_csv(
                out / "anomalies.csv", index=False
            )

        if surprise_report:
            analyzer = SurpriseAnalyzer()
            analyzer.to_dataframe(surprise_report).to_csv(
                out / "surprise_findings.csv", index=False
            )
            analyzer.domain_summary(surprise_report).to_csv(
                out / "domain_classification.csv", index=False
            )

        if interface_report:
            ia = InterfaceAnalyzer()
            ia.pairs_to_dataframe(interface_report).to_csv(
                out / "interface_compatibility.csv", index=False
            )
            interface_report.reusability_scores.to_csv(
                out / "module_reusability.csv", index=False
            )

        if pha_report and pha_report.clusters:
            PHAAnalyzer.clusters_to_dataframe(pha_report).to_csv(
                out / "pha_clusters.csv", index=False
            )
            df_comp = PHAAnalyzer.components_to_dataframe(pha_report)
            if not df_comp.empty:
                df_comp.to_csv(out / "pha_components.csv", index=False)
            df_iface = PHAAnalyzer.interfaces_to_dataframe(pha_report)
            if not df_iface.empty:
                df_iface.to_csv(out / "pha_unified_interfaces.csv", index=False)

            # ── Diagram-string CSV outputs ──
            self._write_diagram_string_csvs(pha_report, out)

    # ── Diagram-string CSV helper ─────────────────────────────────────

    def _write_diagram_string_csvs(
        self,
        pha_report: PHAReport,
        out: Path,
    ) -> None:
        """Write CSV files for diagram-string serialisation and comparisons.

        Produces up to two files:
          • diagram_strings.csv  — one row per project per cluster
          • diagram_comparisons.csv — one row per pairwise comparison
        """
        # ── Diagram strings table ──
        ds_rows = []
        for c in pha_report.clusters:
            for pname, ds in c.diagram_strings.items():
                ds_rows.append({
                    "cluster_id": c.cluster_id,
                    "project": pname,
                    "domain": ds.domain,
                    "total_gates": ds.total_gates,
                    "n_modules": ds.n_modules,
                    "line_count": ds.line_count,
                    "token_count": len(ds.tokens),
                    "compact_head": ds.compact[:120],
                    "block_sequence_head": ds.block_sequence[:200],
                    "interface_summary": ds.interface_summary[:200],
                })
        if ds_rows:
            pd.DataFrame(ds_rows).to_csv(
                out / "diagram_strings.csv", index=False,
            )

        # ── Diagram comparisons table ──
        cmp_rows = []
        for c in pha_report.clusters:
            for cmp in c.diagram_comparisons:
                cmp_rows.append({
                    "cluster_id": c.cluster_id,
                    "project_a": cmp.project_a,
                    "project_b": cmp.project_b,
                    "lcs_length": cmp.lcs_length,
                    "lcs_ratio": cmp.lcs_ratio,
                    "alignment_identity_pct": cmp.alignment.identity_pct,
                    "alignment_matches": cmp.alignment.matches,
                    "alignment_mismatches": cmp.alignment.mismatches,
                    "alignment_gaps": cmp.alignment.gaps,
                    "alignment_normalised_score": cmp.alignment.normalised_score,
                    "longest_common_substr_len": cmp.longest_common_substr.length,
                    "longest_common_substr_label": (
                        cmp.longest_common_substr.semantic_label or ""
                    ),
                    "shared_substrings_count": len(cmp.shared_substrings),
                    "shared_blocks": ", ".join(cmp.shared_blocks),
                    "mergeable_candidates_count": len(cmp.mergeable_candidates),
                    "overall_string_similarity": cmp.overall_string_similarity,
                })
        if cmp_rows:
            pd.DataFrame(cmp_rows).to_csv(
                out / "diagram_comparisons.csv", index=False,
            )

