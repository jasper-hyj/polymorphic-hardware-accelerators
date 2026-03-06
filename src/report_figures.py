"""Figure generation for the RTL similarity analysis report.

All matplotlib/seaborn figure methods are collected here so that
:mod:`report` stays focused on orchestration and CSV I/O.
"""

from __future__ import annotations

import logging
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from .anomaly import Anomaly
from .config import AnalysisConfig
from .iverilog_analyzer import GATE_TYPES, ProjectAnalysis

log = logging.getLogger(__name__)


# ── Helper ────────────────────────────────────────────────────────────

def _heatmap_fig(
    df: pd.DataFrame,
    title: str,
    colormap: str,
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
        cmap=colormap,
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


# ═══════════════════════════════════════════════════════════════════════
#  Public figure-writing functions
# ═══════════════════════════════════════════════════════════════════════

def write_heatmaps(
    cfg: AnalysisConfig,
    df_combined: pd.DataFrame,
    df_gate: pd.DataFrame,
    df_struct: pd.DataFrame,
    df_ngram: pd.DataFrame,
) -> None:
    pairs = [
        (df_combined, "Combined Similarity (gate-type + structural + n-gram)",
         "fig_heatmap_combined"),
        (df_gate, "Gate-Type Cosine Similarity",
         "fig_heatmap_gate"),
        (df_struct, "Structural Graph Similarity",
         "fig_heatmap_structural"),
        (df_ngram, "Partial Gate-Pattern N-gram Similarity (Jaccard)",
         "fig_heatmap_ngram"),
    ]
    for df, title, stem in pairs:
        fig = _heatmap_fig(df, title, cfg.colormap)
        path = cfg.output_dir / f"{stem}.{cfg.figure_format}"
        fig.savefig(path)
        plt.close(fig)
        log.info("Figure \u2192 %s", path)


def write_gate_distribution(
    cfg: AnalysisConfig,
    projects: List[ProjectAnalysis],
) -> None:
    names = [p.name for p in projects]
    data = np.array(
        [[p.gate_counts.get(g, 0) for g in GATE_TYPES] for p in projects],
        dtype=float,
    )
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
    path = cfg.output_dir / f"fig_gate_distribution.{cfg.figure_format}"
    fig.savefig(path)
    plt.close(fig)
    log.info("Figure \u2192 %s", path)


def write_partial_match_chart(
    cfg: AnalysisConfig,
    partial_matches: pd.DataFrame,
) -> None:
    """Bubble chart: x=project_a, y=project_b, size=shared pattern count."""
    if partial_matches.empty:
        return

    pair_df = (
        partial_matches.groupby(["project_a", "project_b", "jaccard_similarity"])
        .agg(
            total_patterns=("min_shared", "sum"),
            pattern_count=("shared_gate_pattern", "count"),
        )
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
        "Partial Gate-Pattern Matches\n"
        "(bubble size = number of shared n-gram types)",
        fontweight="bold",
    )
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    path = cfg.output_dir / f"fig_partial_patterns.{cfg.figure_format}"
    fig.savefig(path)
    plt.close(fig)
    log.info("Figure \u2192 %s", path)


def write_interface_chart(
    cfg: AnalysisConfig,
    interface_report: "InterfaceReport",
) -> None:
    """Two-panel figure: reusability scatter + protocol coverage bar."""
    from .interface_analyzer import InterfaceReport  # noqa: F811

    df = interface_report.reusability_scores
    if df.empty:
        return

    proto_cov = {
        p: projs
        for p, projs in interface_report.protocol_coverage.items()
        if projs
    }

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 12),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    all_protos = sorted(proto_cov.keys())
    cmap_vals = plt.cm.tab10(np.linspace(0, 0.9, max(len(all_protos), 1)))
    proto_color = {p: cmap_vals[i] for i, p in enumerate(all_protos)}
    proto_color["(none)"] = (0.7, 0.7, 0.7, 0.85)

    def _proto_label(row_protos: str) -> str:
        if not row_protos:
            return "(none)"
        return row_protos.split(",")[0].strip()

    df = df.copy()
    df["_proto_label"] = df["protocols"].apply(_proto_label)
    df["_color"] = df["_proto_label"].map(
        lambda p: proto_color.get(p, proto_color["(none)"])
    )
    sizes = 80 + 400 * (
        df["same_signature_modules"] /
        max(df["same_signature_modules"].max(), 1)
    )

    ax_top.scatter(
        df["port_count"], df["reusability_score"],
        s=sizes, c=list(df["_color"]),
        alpha=0.82, edgecolors="white", linewidths=0.5,
    )

    for _, row in df.head(15).iterrows():
        ax_top.annotate(
            f"{row['project'][:10]}::{row['module'][:14]}",
            (row["port_count"], row["reusability_score"]),
            fontsize=6.5, ha="left", va="bottom",
            xytext=(4, 3), textcoords="offset points",
        )

    ax_top.set_xlabel("Port count", fontsize=10)
    ax_top.set_ylabel("Reusability score", fontsize=10)
    ax_top.set_title(
        "Module Reusability Map  "
        "(upper-left = small interface + standard protocol = most reusable)",
        fontweight="bold",
    )
    ax_top.set_ylim(-0.05, 1.05)

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=proto_color.get(p, proto_color["(none)"]),
            markersize=8, label=p,
        )
        for p in all_protos + (
            ["(none)"] if "(none)" in df["_proto_label"].values else []
        )
    ]
    if legend_handles:
        ax_top.legend(
            handles=legend_handles, title="Protocol",
            fontsize=8, title_fontsize=8,
            loc="upper right", framealpha=0.85,
        )

    for sz_label, sz_val in [(1, 80), (5, 280), (10, 480)]:
        ax_top.scatter([], [], s=sz_val, color="grey", alpha=0.5,
                       label=f"{sz_label} sharing same port shape")
    ax_top.legend(
        *ax_top.get_legend_handles_labels(),
        fontsize=7, loc="lower right", framealpha=0.8,
    )

    if proto_cov:
        protos = sorted(proto_cov, key=lambda p: -len(proto_cov[p]))
        counts = [len(proto_cov[p]) for p in protos]
        colors = [proto_color.get(p, proto_color["(none)"]) for p in protos]
        bars = ax_bot.barh(protos, counts, color=colors,
                           edgecolor="white", linewidth=0.7)
        for bar, cnt in zip(bars, counts):
            ax_bot.text(
                bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=8,
            )
        ax_bot.set_xlabel("Number of projects with this protocol", fontsize=9)
        ax_bot.set_title("Standard Protocol Coverage", fontweight="bold")
        ax_bot.set_xlim(0, max(counts) + 2)
    else:
        ax_bot.text(0.5, 0.5, "No standard protocols detected",
                    ha="center", va="center", transform=ax_bot.transAxes)
        ax_bot.axis("off")

    fig.tight_layout(pad=3.0)
    path = cfg.output_dir / f"fig_interface_reusability.{cfg.figure_format}"
    fig.savefig(path)
    plt.close(fig)
    log.info("Figure \u2192 %s", path)


def write_surprise_scatter(
    cfg: AnalysisConfig,
    surprise_report: "SurpriseReport",
) -> None:
    """Scatter plot: similarity vs domain dissimilarity, bubble = surprise."""
    from .surprise import SurpriseReport  # noqa: F811

    pairs = surprise_report.pairs
    if not pairs:
        return

    xs = [p.combined_similarity for p in pairs]
    ys = [p.domain_dissimilarity for p in pairs]
    ss = [p.surprise_score for p in pairs]
    names = [f"{p.project_a[:10]}\u2194{p.project_b[:10]}" for p in pairs]

    ss_arr = np.array(ss)
    sizes = 200 + 1800 * (
        ss_arr / ss_arr.max() if ss_arr.max() > 0 else ss_arr
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        xs, ys, s=sizes, c=ss_arr,
        cmap="YlOrRd", edgecolors="grey", linewidths=0.6, alpha=0.85,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Surprise score", fontsize=9)

    for name, x, y in zip(names, xs, ys):
        ax.annotate(
            name, (x, y),
            fontsize=7.5, ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points",
        )

    ax.set_xlabel("Combined Similarity", fontsize=10)
    ax.set_ylabel("Domain Dissimilarity", fontsize=10)
    ax.set_title(
        "Surprise Map \u2014 High Surprise = upper-right corner",
        fontweight="bold",
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.axhline(0.5, color="steelblue", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(0.5, color="steelblue", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(0.97, 0.97, "High\nSurprise", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="firebrick", alpha=0.7)

    fig.tight_layout()
    path = cfg.output_dir / f"fig_surprise_map.{cfg.figure_format}"
    fig.savefig(path)
    plt.close(fig)
    log.info("Figure \u2192 %s", path)


def write_anomaly_chart(
    cfg: AnalysisConfig,
    anomalies: List[Anomaly],
) -> None:
    names = [a.project_name for a in anomalies]
    scores = [a.zscore for a in anomalies]

    fig, ax = plt.subplots(figsize=(max(6, len(anomalies) * 0.9), 5))
    bars = ax.bar(names, scores, color="tomato", edgecolor="white", linewidth=0.8)
    ax.axhline(
        cfg.anomaly_zscore_threshold,
        color="navy", linestyle="--", linewidth=1.2,
        label=f"Threshold = {cfg.anomaly_zscore_threshold}",
    )
    ax.set_ylabel("Max Robust Z-Score")
    ax.set_title("Anomaly Detection \u2014 Outlier Z-Scores", fontweight="bold")
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
    path = cfg.output_dir / f"fig_anomaly_scores.{cfg.figure_format}"
    fig.savefig(path)
    plt.close(fig)
    log.info("Figure \u2192 %s", path)


def write_pha_charts(
    cfg: AnalysisConfig,
    pha_report: "PHAReport",
) -> None:
    """Multi-panel PHA synthesis figure: component breakdown, area savings,
    unified interface composition."""
    from .pha_analyzer import PHAReport  # noqa: F811

    clusters = pha_report.clusters
    if not clusters:
        return

    n_panels = 2 + (1 if any(c.unified_interface for c in clusters) else 0)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(max(10, len(clusters) * 2.5), 5 * n_panels),
        gridspec_kw={
            "height_ratios": [2, 2] + ([1.5] if n_panels == 3 else []),
        },
    )
    if n_panels == 1:
        axes = [axes]

    labels = [
        f"PHA-{c.cluster_id}\n({len(c.member_projects)} DSAs)"
        for c in clusters
    ]
    shared_vals = [c.shared_gate_count for c in clusters]
    merged_vals = [c.merged_gate_count for c in clusters]
    unique_vals = [c.unique_gate_count for c in clusters]
    x = np.arange(len(clusters))
    w = 0.5

    # Panel 1: component breakdown
    ax = axes[0]
    ax.bar(x, shared_vals, w, label="Shared (1\u00d7 instance)", color="#4CAF50")
    ax.bar(x, merged_vals, w, bottom=shared_vals,
           label="Merged (configurable + mux)", color="#FF9800")
    bottoms_2 = [s + m for s, m in zip(shared_vals, merged_vals)]
    ax.bar(x, unique_vals, w, bottom=bottoms_2,
           label="Unique (DSA-specific)", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Estimated gate count")
    ax.set_title(
        "PHA Component Decomposition\n"
        "(Shared hardware counted once \u2192 area savings)",
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right")
    for i, c in enumerate(clusters):
        ax.text(
            i,
            shared_vals[i] + merged_vals[i] + unique_vals[i] + 5,
            f"{c.pha_gate_total:,}g",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    # Panel 2: individual vs PHA
    ax2 = axes[1]
    indiv = [c.individual_gate_total for c in clusters]
    pha = [c.pha_gate_total for c in clusters]
    w2 = 0.35
    ax2.bar(x - w2 / 2, indiv, w2, label="Individual DSAs (sum)",
            color="#78909C", edgecolor="white")
    ax2.bar(x + w2 / 2, pha, w2, label="PHA (merged)",
            color="#26A69A", edgecolor="white")
    for i, c in enumerate(clusters):
        ax2.annotate(
            f"-{c.area_savings_pct:.0f}%",
            xy=(i + w2 / 2, pha[i]),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold", color="#D32F2F",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Total gate count")
    ax2.set_title(
        "Area Efficiency: Individual DSAs vs. Polymorphic Architecture",
        fontweight="bold",
    )
    ax2.legend(fontsize=8)

    # Panel 3: unified interface
    if n_panels == 3:
        ax3 = axes[2]
        common_counts = []
        muxed_counts = []
        c_labels = []
        for c in clusters:
            if c.unified_interface:
                common_counts.append(len(c.unified_interface.common_signals))
                muxed_counts.append(len(c.unified_interface.muxed_signals))
                proto_str = (
                    f"\n({c.unified_interface.protocol})"
                    if c.unified_interface.protocol else ""
                )
                c_labels.append(f"PHA-{c.cluster_id}{proto_str}")
        if c_labels:
            x3 = np.arange(len(c_labels))
            ax3.bar(x3, common_counts, 0.5,
                    label="Common (pass-through)", color="#42A5F5")
            ax3.bar(x3, muxed_counts, 0.5, bottom=common_counts,
                    label="Muxed (mode-dependent)", color="#FFA726")
            ax3.set_xticks(x3)
            ax3.set_xticklabels(c_labels, fontsize=9)
            ax3.set_ylabel("Signal count")
            ax3.set_title(
                "Unified Interface Composition (J+K+L merged port block)",
                fontweight="bold",
            )
            ax3.legend(fontsize=8)

    fig.tight_layout(pad=3.0)
    path = cfg.output_dir / f"fig_pha_synthesis.{cfg.figure_format}"
    fig.savefig(path)
    plt.close(fig)
    log.info("Figure \u2192 %s", path)
