"""Research-grade reporting: verbal, visual, and CSV outputs.

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

import datetime
import logging
import textwrap
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")               # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from .anomaly import Anomaly, AnomalyDetector
from .config import AnalysisConfig
from .iverilog_analyzer import GATE_TYPES, ProjectAnalysis
from .similarity import SimilarityEngine

log = logging.getLogger(__name__)

_SECTION = "=" * 72
_SUBSEC  = "-" * 72


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
    ) -> None:
        if not projects:
            log.warning("No projects to report.")
            return

        top_pairs = SimilarityEngine.top_pairs(df_combined, n=15)

        self._write_csv(
            projects, df_combined, df_gate, df_struct, df_ngram,
            top_pairs, partial_matches, anomalies,
        )
        self._write_verbal(projects, df_combined, df_ngram, partial_matches, top_pairs, anomalies)
        self._write_heatmaps(df_combined, df_gate, df_struct, df_ngram)
        self._write_gate_distribution(projects)
        self._write_partial_match_chart(partial_matches)
        if anomalies:
            self._write_anomaly_chart(anomalies)

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
    ) -> None:
        out = self.cfg.output_dir

        # Round matrices for readability
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

        # Top pairs
        top_pairs.to_csv(out / "top_similar_pairs.csv")

        # Partial gate-pattern matches
        if not partial_matches.empty:
            partial_matches.to_csv(out / "partial_gate_patterns.csv", index=False)

        # Anomalies
        if anomalies:
            AnomalyDetector(self.cfg).to_dataframe(anomalies).to_csv(
                out / "anomalies.csv", index=False
            )

    # ── Verbal report ─────────────────────────────────────────────────

    def _write_verbal(
        self,
        projects: List[ProjectAnalysis],
        df_combined: pd.DataFrame,
        df_ngram: pd.DataFrame,
        partial_matches: pd.DataFrame,
        top_pairs: pd.DataFrame,
        anomalies: List[Anomaly],
    ) -> None:
        lines: List[str] = []
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        def h1(title: str):
            lines.extend(["", _SECTION, f"  {title}", _SECTION])

        def h2(title: str):
            lines.extend(["", _SUBSEC, f"  {title}", _SUBSEC])

        def para(text: str, width: int = 70):
            lines.append(textwrap.fill(text, width=width))

        # ── Header ────────────────────────────────────────────────
        lines.append("POLYMORPHIC HARDWARE ACCELERATOR RTL SIMILARITY ANALYSIS")
        lines.append(f"Generated: {ts}")
        lines.append(f"Projects analysed: {len(projects)}")
        lines.append(
            f"Similarity weights: gate-type={self.cfg.gate_type_weight:.0%}, "
            f"structural={self.cfg.structural_weight:.0%}"
        )

        # ── 1. Dataset summary ────────────────────────────────────
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
        lines.append(f"  Avg gates/project  : {total_gates/max(1,len(projects)):,.0f}")

        # ── 2. Per-project table ───────────────────────────────────
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
                f"{p.total_gates:>8,} {p.gate_counts.get('DFF',0):>6} "
                f"{p.gate_counts.get('AND',0):>6} "
                f"{p.graph.number_of_nodes():>7} {p.graph.number_of_edges():>7} "
                f"{be:>10}"
            )

        # ── 3. Similarity results ──────────────────────────────────
        h1("2. SIMILARITY ANALYSIS")
        para(
            "Pairwise combined similarity is computed as a weighted combination: "
            f"{self.cfg.gate_type_weight:.0%} gate-type cosine similarity "
            f"(distribution of AND/OR/NOT/XOR/DFF/… gate types) and "
            f"{self.cfg.structural_weight:.0%} structural graph similarity "
            "(density, clustering, degree-entropy, component count)."
        )

        h2("Top-15 Most Similar Project Pairs")
        lines.append(top_pairs.to_string(float_format=lambda x: f"{x:.4f}"))

        # Overall matrix stats
        vals = df_combined.values.copy()
        np.fill_diagonal(vals, np.nan)
        flat = vals[~np.isnan(vals)]
        lines.append("")
        lines.append(f"  Matrix statistics (off-diagonal):")
        lines.append(f"    Mean     : {np.nanmean(flat):.4f}")
        lines.append(f"    Median   : {np.nanmedian(flat):.4f}")
        lines.append(f"    Std dev  : {np.nanstd(flat):.4f}")
        lines.append(f"    Min      : {np.nanmin(flat):.4f}")
        lines.append(f"    Max      : {np.nanmax(flat):.4f}")

        # ── 3. Partial gate-pattern matching ─────────────────────────
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
            # Pivot to show one row per pair with top patterns
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

        # Top-10 n-gram Jaccard pairs
        h2("N-gram Jaccard Similarity — Top Pairs")
        lines.append(SimilarityEngine.top_pairs(df_ngram, n=10).to_string(
            float_format=lambda x: f"{x:.4f}"
        ))

        # ── 4. Gate-type distribution ─────────────────────────────
        h1("4. GATE-TYPE DISTRIBUTION")
        gate_totals = {g: sum(p.gate_counts.get(g, 0) for p in projects) for g in GATE_TYPES}
        total = sum(gate_totals.values()) or 1
        for g, cnt in sorted(gate_totals.items(), key=lambda x: -x[1]):
            bar = "#" * int(40 * cnt / total)
            lines.append(f"  {g:<12} {cnt:>8,}  ({100*cnt/total:5.1f}%)  {bar}")

        # ── 5. Anomalies ───────────────────────────────────────────
        h1("5. ANOMALY DETECTION")
        if anomalies:
            para(
                f"Anomaly detection (robust Z-score threshold "
                f"{self.cfg.anomaly_zscore_threshold}) flagged "
                f"{len(anomalies)} project(s) as statistical outliers:"
            )
            for a in anomalies:
                lines.append(f"\n  [{a.project_name}]  Z-score: {a.zscore:.2f}")
                lines.append(f"    {a.description}")
        else:
            para(
                "No anomalies detected. All projects fall within "
                f"{self.cfg.anomaly_zscore_threshold} robust standard deviations "
                "of the feature-space median."
            )

        # ── 6. Methodology note ────────────────────────────────────
        h1("6. METHODOLOGY")
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
            "The combined score is a weighted average of all three."
        )
        lines.extend(["", _SECTION])

        # Write
        report_path = self.cfg.output_dir / "report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Verbal report → %s", report_path)

    # ── Figures ───────────────────────────────────────────────────────

    def _heatmap_fig(
        self,
        df: pd.DataFrame,
        title: str,
    ) -> plt.Figure:
        n = len(df)
        size = max(6, min(n * 0.55, 22))
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        mask = np.zeros_like(df.values, dtype=bool)
        np.fill_diagonal(mask, True)
        sns.heatmap(
            df,
            ax=ax,
            mask=mask,
            cmap=self.cfg.colormap,
            vmin=0, vmax=1,
            annot=(n <= 20),
            fmt=".2f",
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Similarity", "shrink": 0.8},
        )
        ax.set_title(title, pad=14, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        return fig

    def _write_heatmaps(
        self,
        df_combined: pd.DataFrame,
        df_gate: pd.DataFrame,
        df_struct: pd.DataFrame,
        df_ngram: pd.DataFrame,
    ) -> None:
        pairs = [
            (df_combined, "Combined Similarity (gate-type + structural + n-gram)",
             "fig_heatmap_combined"),
            (df_gate,    "Gate-Type Cosine Similarity",
             "fig_heatmap_gate"),
            (df_struct,  "Structural Graph Similarity",
             "fig_heatmap_structural"),
            (df_ngram,   "Partial Gate-Pattern N-gram Similarity (Jaccard)",
             "fig_heatmap_ngram"),
        ]
        for df, title, stem in pairs:
            fig = self._heatmap_fig(df, title)
            path = self.cfg.output_dir / f"{stem}.{self.cfg.figure_format}"
            fig.savefig(path)
            plt.close(fig)
            log.info("Figure → %s", path)

    def _write_gate_distribution(self, projects: List[ProjectAnalysis]) -> None:
        names = [p.name for p in projects]
        data = np.array(
            [[p.gate_counts.get(g, 0) for g in GATE_TYPES] for p in projects],
            dtype=float,
        )
        # Normalise rows to percentages
        row_totals = data.sum(axis=1, keepdims=True)
        row_totals[row_totals == 0] = 1
        data_pct = 100 * data / row_totals

        n_proj = len(projects)
        fig_h = max(5, n_proj * 0.4)
        fig, ax = plt.subplots(figsize=(12, fig_h))

        cmap = plt.get_cmap("tab20")
        colors = [cmap(i / len(GATE_TYPES)) for i in range(len(GATE_TYPES))]
        lefts = np.zeros(n_proj)
        for gi, (gate, color) in enumerate(zip(GATE_TYPES, colors)):
            vals = data_pct[:, gi]
            ax.barh(names, vals, left=lefts, height=0.72, color=color, label=gate)
            lefts += vals

        ax.set_xlabel("Gate-type composition (%)")
        ax.set_title("Gate-Type Distribution per Project", fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter())
        ax.legend(
            title="Gate Type",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=8,
        )
        ax.invert_yaxis()
        fig.tight_layout()
        path = self.cfg.output_dir / f"fig_gate_distribution.{self.cfg.figure_format}"
        fig.savefig(path)
        plt.close(fig)
        log.info("Figure → %s", path)

    def _write_partial_match_chart(self, partial_matches: pd.DataFrame) -> None:
        """Bubble chart: x=project_a, y=project_b, size=min_shared gate patterns."""
        if partial_matches.empty:
            return

        # One row per pair (aggregate: total shared patterns, mean jaccard)
        pair_df = (
            partial_matches.groupby(["project_a", "project_b", "jaccard_similarity"])
            .agg(total_patterns=("min_shared", "sum"), pattern_count=("shared_gate_pattern", "count"))
            .reset_index()
            .sort_values("jaccard_similarity", ascending=False)
            .head(30)
        )

        fig, ax = plt.subplots(figsize=(13, 6))
        scatter = ax.scatter(
            pair_df["project_a"],
            pair_df["project_b"],
            s=pair_df["pattern_count"] * 60 + 30,
            c=pair_df["jaccard_similarity"],
            cmap="YlOrRd",
            vmin=0, vmax=1,
            alpha=0.85,
            edgecolors="0.3",
            linewidths=0.5,
        )
        cb = fig.colorbar(scatter, ax=ax, pad=0.02)
        cb.set_label("Jaccard Similarity")
        ax.set_xlabel("Project A")
        ax.set_ylabel("Project B")
        ax.set_title(
            "Partial Gate-Pattern Matches\n(bubble size = number of shared n-gram types)",
            fontweight="bold",
        )
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        fig.tight_layout()
        path = self.cfg.output_dir / f"fig_partial_patterns.{self.cfg.figure_format}"
        fig.savefig(path)
        plt.close(fig)
        log.info("Figure → %s", path)

    def _write_anomaly_chart(self, anomalies: List[Anomaly]) -> None:
        names  = [a.project_name for a in anomalies]
        scores = [a.zscore for a in anomalies]

        fig, ax = plt.subplots(figsize=(max(6, len(anomalies) * 0.9), 5))
        bars = ax.bar(names, scores, color="tomato", edgecolor="white", linewidth=0.8)
        ax.axhline(
            self.cfg.anomaly_zscore_threshold,
            color="navy", linestyle="--", linewidth=1.2,
            label=f"Threshold = {self.cfg.anomaly_zscore_threshold}",
        )
        ax.set_ylabel("Max Robust Z-Score")
        ax.set_title("Anomaly Detection — Outlier Z-Scores", fontweight="bold")
        ax.legend()
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{score:.2f}",
                ha="center", va="bottom", fontsize=9,
            )
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        path = self.cfg.output_dir / f"fig_anomaly_scores.{self.cfg.figure_format}"
        fig.savefig(path)
        plt.close(fig)
        log.info("Figure → %s", path)
