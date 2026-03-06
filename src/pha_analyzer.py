"""Polymorphic Heterogeneous Architecture (PHA) synthesis analyser.

This module implements the core contribution of the project: given a set of
independent Domain-Specific Accelerators (DSAs), it proposes how to merge
them into a unified Polymorphic Heterogeneous Architecture (PHA).

The heavy-lifting sub-algorithms (clustering, decomposition, interface
unification, resource estimation, explanation) are in :mod:`pha_components`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .iverilog_analyzer import GATE_TYPES, ProjectAnalysis
from .surprise import classify_domain
from .interface_analyzer import InterfaceReport
from .diagram_serializer import (
    DiagramSerializer,
    DiagramString,
    ComparisonResult as DiagramComparison,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FunctionalComponent:
    """A reusable functional block identified from gate n-gram patterns."""
    name: str
    pattern: Tuple[str, ...]
    pattern_str: str
    occurrences: Dict[str, int] = field(default_factory=dict)
    semantic_label: Optional[str] = None

    @property
    def project_count(self) -> int:
        return len(self.occurrences)


@dataclass
class MergedComponent:
    """Two similar-but-not-identical components proposed for merging."""
    component_a: FunctionalComponent
    component_b: FunctionalComponent
    jaccard: float
    merged_name: str
    mux_overhead: float
    projects_involved: List[str] = field(default_factory=list)


@dataclass
class UnifiedInterface:
    """A merged interface block combining per-DSA interconnects (J+K+L)."""
    common_signals: List[Tuple[str, str, int]]
    muxed_signals: List[Tuple[str, str, int, List[str]]]
    protocol: Optional[str] = None
    total_port_count: int = 0
    member_projects: List[str] = field(default_factory=list)


@dataclass
class PHACluster:
    """One candidate Polymorphic Heterogeneous Architecture."""
    cluster_id: int
    member_projects: List[str]
    member_domains: Dict[str, str]
    avg_similarity: float

    shared_components: List[FunctionalComponent]
    merged_components: List[MergedComponent]
    unique_components: Dict[str, List[FunctionalComponent]]

    unified_interface: Optional[UnifiedInterface] = None

    individual_gate_total: int = 0
    shared_gate_count: int = 0
    merged_gate_count: int = 0
    unique_gate_count: int = 0
    pha_gate_total: int = 0
    area_efficiency: float = 0.0
    area_savings_pct: float = 0.0

    explanation: str = ""

    diagram_strings: Dict[str, "DiagramString"] = field(default_factory=dict)
    diagram_comparisons: List["DiagramComparison"] = field(default_factory=list)
    llm_analysis: str = ""


@dataclass
class PHAReport:
    """Top-level result of the PHA synthesis analysis."""
    clusters: List[PHACluster]
    unclustered_projects: List[str]
    total_projects: int
    clustering_threshold: float
    merge_jaccard_threshold: float


# ═══════════════════════════════════════════════════════════════════════
#  Main analyser
# ═══════════════════════════════════════════════════════════════════════

class PHAAnalyzer:
    """Propose Polymorphic Heterogeneous Architectures from DSA analysis."""

    def __init__(
        self,
        cluster_threshold: float = 0.35,
        merge_jaccard: float = 0.60,
        ngram_n: int = 3,
        min_cluster_size: int = 2,
        use_diagram_strings: bool = False,
        max_diagram_cluster_size: int = 30,
        llm_api_key: str = "",
        llm_model: str = "gpt-4o-mini",
        llm_base_url: str = "",
    ):
        self.cluster_threshold = cluster_threshold
        self.merge_jaccard = merge_jaccard
        self.ngram_n = ngram_n
        self.min_cluster_size = min_cluster_size
        self.use_diagram_strings = use_diagram_strings
        self.max_diagram_cluster_size = max_diagram_cluster_size
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url or None

    def analyse(
        self,
        projects: List[ProjectAnalysis],
        df_combined: pd.DataFrame,
        interface_report: Optional[InterfaceReport] = None,
    ) -> PHAReport:
        """Run the full PHA synthesis analysis."""
        # Lazy import to break circular dependency
        from .pha_components import (
            agglomerative_clusters,
            decompose_components,
            unify_interfaces,
            estimate_resources,
            explain_cluster,
        )

        log.info("Clustering %d DSAs (threshold=%.2f) \u2026",
                 len(projects), self.cluster_threshold)
        clusters_raw = agglomerative_clusters(
            df_combined,
            threshold=self.cluster_threshold,
            min_cluster_size=self.min_cluster_size,
        )
        log.info("  Found %d candidate PHA cluster(s).", len(clusters_raw))

        name_to_proj = {p.name: p for p in projects}
        all_clustered: Set[str] = set()
        pha_clusters: List[PHACluster] = []

        for cid, member_names in enumerate(clusters_raw, 1):
            all_clustered.update(member_names)
            members = [name_to_proj[n] for n in member_names if n in name_to_proj]
            if len(members) < 2:
                continue

            log.info("  [Cluster %d] %d members: %s",
                     cid, len(members), ", ".join(m.name for m in members))

            sims = []
            for i, a in enumerate(member_names):
                for j, b in enumerate(member_names):
                    if j > i and a in df_combined.index and b in df_combined.columns:
                        sims.append(float(df_combined.loc[a, b]))
            avg_sim = sum(sims) / len(sims) if sims else 0.0

            domains = {p.name: classify_domain(p) for p in members}

            log.info("    Decomposing into shared / merged / unique components \u2026")
            shared, merged, unique = decompose_components(
                members, ngram_n=self.ngram_n, merge_threshold=self.merge_jaccard,
            )
            log.info("    %d shared, %d merged, %d unique-project group(s)",
                     len(shared), len(merged), len(unique))

            unified_iface = unify_interfaces(members, interface_report)
            if unified_iface:
                log.info("    Unified interface: %d common + %d muxed signals",
                         len(unified_iface.common_signals),
                         len(unified_iface.muxed_signals))

            (indiv_total, shared_g, merged_g, unique_g,
             pha_total, efficiency, savings) = estimate_resources(
                members, shared, merged, unique,
            )
            log.info("    Estimated area: individual=%d  PHA=%d  savings=%.1f%%",
                     indiv_total, pha_total, savings)

            explanation = explain_cluster(
                cid, members, domains, avg_sim,
                shared, merged, unique,
                unified_iface, savings,
            )

            # ── Diagram-string analysis (optional) ──
            ds_map: Dict[str, DiagramString] = {}
            ds_comparisons: List[DiagramComparison] = []
            llm_text = ""
            if self.use_diagram_strings:
                # Cap members for O(M²) string comparison phase
                ds_members = members
                if len(members) > self.max_diagram_cluster_size:
                    log.info("    Cluster has %d members; capping diagram-string "
                             "phase to %d highest-similarity members",
                             len(members), self.max_diagram_cluster_size)
                    # Pick members with highest average pairwise similarity
                    avg_sims_per_member = {}
                    for m in members:
                        s = []
                        for o in members:
                            if m.name != o.name and m.name in df_combined.index and o.name in df_combined.columns:
                                s.append(float(df_combined.loc[m.name, o.name]))
                        avg_sims_per_member[m.name] = sum(s) / len(s) if s else 0.0
                    top_names = sorted(avg_sims_per_member, key=avg_sims_per_member.get, reverse=True)[:self.max_diagram_cluster_size]
                    ds_members = [m for m in members if m.name in set(top_names)]
                log.info("    Serialising %d architectures to diagram strings …",
                         len(ds_members))
                serializer = DiagramSerializer(ngram_n=self.ngram_n)
                for p in ds_members:
                    ds_map[p.name] = serializer.serialize(p, interface_report)
                    log.info("      %s: %d tokens, compact=%s\u2026",
                             p.name, len(ds_map[p.name].tokens),
                             ds_map[p.name].compact[:40])
                member_ds = list(ds_map.values())
                for i in range(len(member_ds)):
                    for j in range(i + 1, len(member_ds)):
                        cmp = serializer.compare(member_ds[i], member_ds[j])
                        ds_comparisons.append(cmp)
                        log.info(
                            "      %s \u2194 %s: LCS ratio=%.3f, "
                            "alignment identity=%.1f%%, "
                            "string sim=%.3f, %d shared substrings",
                            cmp.project_a, cmp.project_b,
                            cmp.lcs_ratio,
                            cmp.alignment.identity_pct,
                            cmp.overall_string_similarity,
                            len(cmp.shared_substrings),
                        )
                if self.llm_api_key:
                    log.info("    Sending diagram strings to LLM (%s) \u2026",
                             self.llm_model)
                    llm_text = serializer.llm_analyze(
                        member_ds, ds_comparisons,
                        api_key=self.llm_api_key,
                        model=self.llm_model,
                        base_url=self.llm_base_url,
                    )
                    log.info("    LLM response: %d chars", len(llm_text))

            pha_clusters.append(PHACluster(
                cluster_id=cid,
                member_projects=member_names,
                member_domains=domains,
                avg_similarity=round(avg_sim, 4),
                shared_components=shared,
                merged_components=merged,
                unique_components=unique,
                unified_interface=unified_iface,
                individual_gate_total=indiv_total,
                shared_gate_count=shared_g,
                merged_gate_count=merged_g,
                unique_gate_count=unique_g,
                pha_gate_total=pha_total,
                area_efficiency=efficiency,
                area_savings_pct=savings,
                explanation=explanation,
                diagram_strings=ds_map,
                diagram_comparisons=ds_comparisons,
                llm_analysis=llm_text,
            ))

        unclustered = [p.name for p in projects if p.name not in all_clustered]
        if unclustered:
            log.info("  %d project(s) not part of any PHA cluster.", len(unclustered))

        return PHAReport(
            clusters=pha_clusters,
            unclustered_projects=unclustered,
            total_projects=len(projects),
            clustering_threshold=self.cluster_threshold,
            merge_jaccard_threshold=self.merge_jaccard,
        )

    # ── CSV helpers (static — no need to instantiate with params) ─────

    @staticmethod
    def clusters_to_dataframe(report: PHAReport) -> pd.DataFrame:
        """One row per cluster with summary statistics."""
        rows = []
        for c in report.clusters:
            rows.append({
                "cluster_id": c.cluster_id,
                "member_count": len(c.member_projects),
                "members": ", ".join(c.member_projects),
                "domains": ", ".join(sorted(set(c.member_domains.values()))),
                "avg_similarity": c.avg_similarity,
                "shared_components": len(c.shared_components),
                "merged_components": len(c.merged_components),
                "unique_component_groups": len(c.unique_components),
                "individual_gates": c.individual_gate_total,
                "pha_gates": c.pha_gate_total,
                "area_savings_pct": c.area_savings_pct,
                "unified_interface_ports": (
                    c.unified_interface.total_port_count
                    if c.unified_interface else 0
                ),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def components_to_dataframe(report: PHAReport) -> pd.DataFrame:
        """One row per component (shared / merged / unique) across all clusters."""
        rows = []
        for c in report.clusters:
            for fc in c.shared_components:
                rows.append({
                    "cluster_id": c.cluster_id,
                    "category": "shared",
                    "component_name": fc.name,
                    "gate_pattern": fc.pattern_str,
                    "semantic_label": fc.semantic_label or "",
                    "projects": ", ".join(fc.occurrences.keys()),
                    "total_occurrences": sum(fc.occurrences.values()),
                })
            for mc in c.merged_components:
                rows.append({
                    "cluster_id": c.cluster_id,
                    "category": "merged",
                    "component_name": mc.merged_name,
                    "gate_pattern": mc.component_a.pattern_str,
                    "semantic_label": mc.component_a.semantic_label or "",
                    "projects": ", ".join(mc.projects_involved),
                    "total_occurrences": (
                        sum(mc.component_a.occurrences.values())
                        + sum(mc.component_b.occurrences.values())
                    ),
                    "merge_jaccard": mc.jaccard,
                    "mux_overhead_pct": round(mc.mux_overhead * 100, 1),
                })
            for pname, comps in c.unique_components.items():
                for fc in comps:
                    rows.append({
                        "cluster_id": c.cluster_id,
                        "category": "unique",
                        "component_name": fc.name,
                        "gate_pattern": fc.pattern_str,
                        "semantic_label": fc.semantic_label or "",
                        "projects": pname,
                        "total_occurrences": sum(fc.occurrences.values()),
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def interfaces_to_dataframe(report: PHAReport) -> pd.DataFrame:
        """One row per signal in each cluster's unified interface."""
        rows = []
        for c in report.clusters:
            if c.unified_interface is None:
                continue
            ui = c.unified_interface
            for norm_name, direction, width in ui.common_signals:
                rows.append({
                    "cluster_id": c.cluster_id,
                    "signal": norm_name,
                    "direction": direction,
                    "width": width,
                    "category": "common",
                    "used_by": ", ".join(ui.member_projects),
                    "protocol": ui.protocol or "",
                })
            for norm_name, direction, width, projs in ui.muxed_signals:
                rows.append({
                    "cluster_id": c.cluster_id,
                    "signal": norm_name,
                    "direction": direction,
                    "width": width,
                    "category": "muxed",
                    "used_by": ", ".join(projs),
                    "protocol": ui.protocol or "",
                })
        return pd.DataFrame(rows)
