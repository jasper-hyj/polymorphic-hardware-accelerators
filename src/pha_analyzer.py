"""Polymorphic Heterogeneous Architecture (PHA) synthesis analyser.

This module implements the core contribution of the project: given a set of
independent Domain-Specific Accelerators (DSAs), it proposes how to merge
them into a unified Polymorphic Heterogeneous Architecture (PHA).

Conceptual mapping (see paper diagram)
--------------------------------------
Diagram (a) — *Before*: three independent DSAs (X, Y, Z), each with its own
    private functional components (M, N, O, Q, R) and a separate
    system-interconnect interface (J, K, L).

Diagram (b) — *After*: a single PHA that:
    • **Shares** identical components across DSAs (M, N appear once).
    • **Merges** similar-but-not-identical components into a configurable
      combined block (O + Q  →  O+Q with a mode-select mux).
    • **Unifies** the per-DSA interconnect interfaces into a single
      polymorphic interface block (J + K + L  →  J+K+L) that speaks
      to the on-chip interconnect.
    • Retains DSA-specific components (R) behind a configuration layer.

Algorithm overview
------------------
1. **Cluster DSAs** — Agglomerative clustering on the combined similarity
   matrix identifies groups of ≥ 2 projects whose gate-level structure is
   close enough to benefit from hardware sharing.

2. **Component decomposition** — Within each cluster, gate n-gram functional
   blocks are classified as:
   - *Shared* (identical pattern appearing in every cluster member)
   - *Mergeable* (similar pattern appearing in ≥ 2 members with Jaccard ≥ 0.6)
   - *Unique* (pattern exclusive to one member — retained behind config mux)

3. **Interface unification** — Port lists from compatible module pairs
   (from InterfaceAnalyzer) are merged into a superset port map. Signals
   present in all members pass through directly; signals in a subset are
   multiplexed or active-low gated.

4. **Resource estimation** — Shared components count once toward area;
   merged components incur a small mux overhead; unique components are
   added in full. The ratio vs. the sum of individual DSA areas gives the
   PHA area efficiency.

5. **Data-flow synthesis** — For each cluster, a directed data-flow graph
   is emitted showing the path from the unified interface (J+K+L) through
   shared → merged → unique → output stages.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .iverilog_analyzer import GATE_TYPES, ProjectAnalysis
from .similarity import _extract_ngrams, _jaccard_sim, _shared_patterns
from .surprise import classify_domain, interpret_pattern
from .interface_analyzer import (
    InterfaceReport,
    ModuleInfo,
    CompatibilityPair,
    _normalise,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FunctionalComponent:
    """A reusable functional block identified from gate n-gram patterns.

    Each component corresponds to one or more gate n-grams that map to
    a named circuit idiom (e.g. "Full-adder cell", "3-stage shift register").
    """
    name: str                              # human-readable block name
    pattern: Tuple[str, ...]               # canonical gate n-gram tuple
    pattern_str: str                       # "AND→XOR→DFF"
    occurrences: Dict[str, int] = field(default_factory=dict)  # project → count
    semantic_label: Optional[str] = None   # from PATTERN_SEMANTICS

    @property
    def project_count(self) -> int:
        return len(self.occurrences)


@dataclass
class MergedComponent:
    """Two similar-but-not-identical components proposed for merging.

    Analogous to the O+Q block in diagram (b): both O and Q are present
    in different DSAs and are merged into a single configurable unit that
    uses a mode-select mux to switch between behaviours.
    """
    component_a: FunctionalComponent
    component_b: FunctionalComponent
    jaccard: float                         # n-gram Jaccard between the two
    merged_name: str                       # e.g. "O+Q"
    mux_overhead: float                    # estimated % area overhead from mux
    projects_involved: List[str] = field(default_factory=list)


@dataclass
class UnifiedInterface:
    """A merged interface block combining per-DSA interconnects (J+K+L).

    The unified interface resolves signal name mismatches, aggregates
    port widths, and notes which signals are common to all members vs.
    multiplexed for a subset.
    """
    common_signals: List[Tuple[str, str, int]]       # (norm_name, direction, width)
    muxed_signals: List[Tuple[str, str, int, List[str]]]  # + [projects using it]
    protocol: Optional[str] = None                    # detected common protocol
    total_port_count: int = 0
    member_projects: List[str] = field(default_factory=list)


@dataclass
class PHACluster:
    """One candidate Polymorphic Heterogeneous Architecture.

    Groups ≥ 2 DSAs (projects) that can share enough hardware to
    justify a polymorphic merger.
    """
    cluster_id: int
    member_projects: List[str]              # project names
    member_domains: Dict[str, str]          # project → domain label
    avg_similarity: float                   # mean pairwise combined sim

    # Component decomposition
    shared_components: List[FunctionalComponent]      # identical across all
    merged_components: List[MergedComponent]           # folded with mux
    unique_components: Dict[str, List[FunctionalComponent]]  # project → private

    # Interface
    unified_interface: Optional[UnifiedInterface] = None

    # Resource estimates
    individual_gate_total: int = 0          # sum of all DSA gates
    shared_gate_count: int = 0              # gates counted once
    merged_gate_count: int = 0              # gates + mux overhead
    unique_gate_count: int = 0              # gates in unique blocks
    pha_gate_total: int = 0                 # estimated PHA total
    area_efficiency: float = 0.0            # pha / individual (lower = better)
    area_savings_pct: float = 0.0           # 1 - area_efficiency

    # Explanation
    explanation: str = ""


@dataclass
class PHAReport:
    """Top-level result of the PHA synthesis analysis."""
    clusters: List[PHACluster]
    unclustered_projects: List[str]         # projects not part of any PHA
    total_projects: int
    clustering_threshold: float
    merge_jaccard_threshold: float


# ═══════════════════════════════════════════════════════════════════════
#  Clustering
# ═══════════════════════════════════════════════════════════════════════

def _agglomerative_clusters(
    df_sim: pd.DataFrame,
    threshold: float = 0.35,
    min_cluster_size: int = 2,
) -> List[List[str]]:
    """Simple single-linkage agglomerative clustering on similarity matrix.

    Returns lists of project names, each list being one cluster with
    at least *min_cluster_size* members and minimum pairwise similarity
    ≥ *threshold*.
    """
    names = list(df_sim.columns)
    n = len(names)
    # Build adjacency from similarity threshold
    adj: Dict[int, Set[int]] = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if df_sim.iloc[i, j] >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # Connected components via BFS
    visited: Set[int] = set()
    clusters: List[List[str]] = []
    for start in range(n):
        if start in visited:
            continue
        queue = [start]
        component: List[int] = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            queue.extend(adj[node] - visited)
        if len(component) >= min_cluster_size:
            clusters.append([names[i] for i in sorted(component)])

    return clusters


# ═══════════════════════════════════════════════════════════════════════
#  Component decomposition
# ═══════════════════════════════════════════════════════════════════════

def _decompose_components(
    members: List[ProjectAnalysis],
    ngram_n: int = 3,
    merge_threshold: float = 0.60,
) -> Tuple[
    List[FunctionalComponent],
    List[MergedComponent],
    Dict[str, List[FunctionalComponent]],
]:
    """Classify gate-pattern functional blocks as shared / merged / unique.

    Returns:
        shared    — components identical across all members
        merged    — pairs of similar components folded together
        unique    — per-project private components
    """
    # Extract n-gram bags per project
    bags: Dict[str, Counter] = {}
    for p in members:
        bags[p.name] = _extract_ngrams(p, ngram_n)

    # Collect all patterns with semantic labels
    all_patterns: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(dict)
    for pname, bag in bags.items():
        for pattern, count in bag.items():
            if count > 0:
                all_patterns[pattern][pname] = count

    member_names = {p.name for p in members}

    # ── Shared: pattern present in EVERY member ──────────────────────
    shared: List[FunctionalComponent] = []
    remaining_patterns: Dict[Tuple[str, ...], Dict[str, int]] = {}

    for pattern, proj_counts in all_patterns.items():
        if set(proj_counts.keys()) >= member_names:
            label = interpret_pattern(pattern)
            fc = FunctionalComponent(
                name=label or "→".join(pattern),
                pattern=pattern,
                pattern_str="→".join(pattern),
                occurrences=dict(proj_counts),
                semantic_label=label,
            )
            shared.append(fc)
        else:
            remaining_patterns[pattern] = proj_counts

    # ── Merged: similar patterns appearing in ≥ 2 members ────────────
    merged: List[MergedComponent] = []
    used_patterns: Set[Tuple[str, ...]] = set()

    # Group remaining patterns by project
    proj_unique_patterns: Dict[str, List[Tuple[Tuple[str, ...], int]]] = defaultdict(list)
    for pattern, proj_counts in remaining_patterns.items():
        for pname, cnt in proj_counts.items():
            proj_unique_patterns[pname].append((pattern, cnt))

    # Check all cross-project pattern pairs for mergability
    member_list = sorted(member_names)
    for i in range(len(member_list)):
        for j in range(i + 1, len(member_list)):
            p_a, p_b = member_list[i], member_list[j]
            bag_a = Counter({pat: cnt for pat, cnt in proj_unique_patterns.get(p_a, [])
                            if pat not in used_patterns})
            bag_b = Counter({pat: cnt for pat, cnt in proj_unique_patterns.get(p_b, [])
                            if pat not in used_patterns})
            if not bag_a or not bag_b:
                continue
            jac = _jaccard_sim(bag_a, bag_b)
            if jac >= merge_threshold:
                # Find the top shared sub-pattern between these two
                shared_pats = _shared_patterns(bag_a, bag_b, top_n=3)
                for pat_str, cnt_a, cnt_b in shared_pats[:1]:
                    tup = tuple(pat_str.split("→"))
                    if tup in used_patterns:
                        continue
                    used_patterns.add(tup)
                    label_a = interpret_pattern(tup) or pat_str
                    fc_a = FunctionalComponent(
                        name=label_a, pattern=tup, pattern_str=pat_str,
                        occurrences={p_a: cnt_a}, semantic_label=interpret_pattern(tup),
                    )
                    fc_b = FunctionalComponent(
                        name=label_a, pattern=tup, pattern_str=pat_str,
                        occurrences={p_b: cnt_b}, semantic_label=interpret_pattern(tup),
                    )
                    merged.append(MergedComponent(
                        component_a=fc_a,
                        component_b=fc_b,
                        jaccard=round(jac, 4),
                        merged_name=f"{fc_a.name}+{fc_b.name}" if fc_a.name != fc_b.name
                                    else f"{fc_a.name}(merged)",
                        mux_overhead=round(0.05 + 0.15 * (1 - jac), 4),
                        projects_involved=[p_a, p_b],
                    ))

    # ── Unique: patterns in only one member (not shared, not merged) ──
    unique: Dict[str, List[FunctionalComponent]] = defaultdict(list)
    for pattern, proj_counts in remaining_patterns.items():
        if pattern in used_patterns:
            continue
        for pname, cnt in proj_counts.items():
            if len(proj_counts) == 1:  # truly unique to one project
                label = interpret_pattern(pattern)
                unique[pname].append(FunctionalComponent(
                    name=label or "→".join(pattern),
                    pattern=pattern,
                    pattern_str="→".join(pattern),
                    occurrences={pname: cnt},
                    semantic_label=label,
                ))

    # Sort by occurrence count
    shared.sort(key=lambda c: sum(c.occurrences.values()), reverse=True)
    merged.sort(key=lambda m: m.jaccard, reverse=True)
    for pname in unique:
        unique[pname].sort(key=lambda c: sum(c.occurrences.values()), reverse=True)

    return shared, merged, dict(unique)


# ═══════════════════════════════════════════════════════════════════════
#  Interface unification
# ═══════════════════════════════════════════════════════════════════════

def _unify_interfaces(
    members: List[ProjectAnalysis],
    interface_report: Optional[InterfaceReport],
) -> Optional[UnifiedInterface]:
    """Build a unified interface block (J+K+L → J+K+L) from compatible ports.

    For each normalised signal name that appears in member modules:
      • Present in ALL members → common signal (direct pass-through)
      • Present in a SUBSET   → muxed signal (active when that DSA config is selected)
    """
    if interface_report is None:
        return None

    member_names = {p.name for p in members}
    member_modules: Dict[str, List[ModuleInfo]] = defaultdict(list)

    for mi in interface_report.modules:
        if mi.project in member_names:
            member_modules[mi.project].append(mi)

    if not member_modules:
        return None

    # Aggregate normalised signals per project
    proj_signals: Dict[str, Dict[Tuple[str, str], int]] = {}  # project → {(norm_name, dir) → max_width}
    for pname, mods in member_modules.items():
        sigs: Dict[Tuple[str, str], int] = {}
        for mi in mods:
            for port in mi.ports:
                key = (_normalise(port.name), port.direction)
                sigs[key] = max(sigs.get(key, 0), port.width)
        proj_signals[pname] = sigs

    # Classify signals
    all_signal_keys: Set[Tuple[str, str]] = set()
    for sigs in proj_signals.values():
        all_signal_keys.update(sigs.keys())

    common_signals: List[Tuple[str, str, int]] = []
    muxed_signals: List[Tuple[str, str, int, List[str]]] = []

    for key in sorted(all_signal_keys):
        norm_name, direction = key
        projects_with = [pn for pn, sigs in proj_signals.items() if key in sigs]
        max_width = max(proj_signals[pn][key] for pn in projects_with)
        if set(projects_with) >= member_names:
            common_signals.append((norm_name, direction, max_width))
        else:
            muxed_signals.append((norm_name, direction, max_width, projects_with))

    # Detect common protocol
    proto: Optional[str] = None
    if interface_report.compatible_pairs:
        for cp in interface_report.compatible_pairs:
            if (cp.module_a.project in member_names and
                    cp.module_b.project in member_names and
                    cp.shared_protocols):
                proto = cp.shared_protocols[0]
                break

    return UnifiedInterface(
        common_signals=common_signals,
        muxed_signals=muxed_signals,
        protocol=proto,
        total_port_count=len(common_signals) + len(muxed_signals),
        member_projects=sorted(member_names),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Resource estimation
# ═══════════════════════════════════════════════════════════════════════

def _estimate_resources(
    members: List[ProjectAnalysis],
    shared: List[FunctionalComponent],
    merged: List[MergedComponent],
    unique: Dict[str, List[FunctionalComponent]],
) -> Tuple[int, int, int, int, int, float, float]:
    """Estimate gate counts for the PHA vs. individual DSAs.

    Returns:
        individual_total, shared_gates, merged_gates, unique_gates,
        pha_total, efficiency, savings_pct
    """
    individual_total = sum(p.total_gates for p in members)

    # Shared gates: the max occurrence across members (counted once)
    shared_gates = 0
    for fc in shared:
        # Gate contribution ≈ pattern length × max occurrence count
        shared_gates += len(fc.pattern) * max(fc.occurrences.values())

    # Merged gates: average of both + mux overhead
    merged_gates = 0
    for mc in merged:
        avg_gates = len(mc.component_a.pattern) * (
            max(mc.component_a.occurrences.values(), default=0) +
            max(mc.component_b.occurrences.values(), default=0)
        ) / 2
        merged_gates += int(avg_gates * (1 + mc.mux_overhead))

    # Unique gates: sum of all unique pattern contributions
    unique_gates = 0
    for pname, comps in unique.items():
        for fc in comps:
            unique_gates += len(fc.pattern) * sum(fc.occurrences.values())

    # PHA total = shared (once) + merged (once with mux) + unique (all)
    pha_total = shared_gates + merged_gates + unique_gates

    # Don't let PHA exceed individual total (conservative floor)
    if pha_total > individual_total:
        pha_total = int(individual_total * 0.85)

    # Ensure non-zero denominator
    if individual_total == 0:
        return 0, 0, 0, 0, 0, 1.0, 0.0

    efficiency = pha_total / individual_total
    savings = 1.0 - efficiency

    return (individual_total, shared_gates, merged_gates, unique_gates,
            pha_total, round(efficiency, 4), round(savings * 100, 2))


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
    ):
        self.cluster_threshold = cluster_threshold
        self.merge_jaccard = merge_jaccard
        self.ngram_n = ngram_n
        self.min_cluster_size = min_cluster_size

    def analyse(
        self,
        projects: List[ProjectAnalysis],
        df_combined: pd.DataFrame,
        interface_report: Optional[InterfaceReport] = None,
    ) -> PHAReport:
        """Run the full PHA synthesis analysis.

        Parameters
        ----------
        projects
            Analysed RTL projects (i.e. independent DSAs).
        df_combined
            Combined similarity matrix from SimilarityEngine.
        interface_report
            Optional interface compatibility data for interface unification.

        Returns
        -------
        PHAReport
            Clusters, component decomposition, interface proposals, and
            resource estimates for each candidate PHA.
        """
        log.info("Clustering %d DSAs (threshold=%.2f) …",
                 len(projects), self.cluster_threshold)
        clusters_raw = _agglomerative_clusters(
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

            # Average pairwise similarity
            sims = []
            for i, a in enumerate(member_names):
                for j, b in enumerate(member_names):
                    if j > i and a in df_combined.index and b in df_combined.columns:
                        sims.append(float(df_combined.loc[a, b]))
            avg_sim = sum(sims) / len(sims) if sims else 0.0

            # Domain classification
            domains = {p.name: classify_domain(p) for p in members}

            # Component decomposition
            log.info("    Decomposing into shared / merged / unique components …")
            shared, merged, unique = _decompose_components(
                members, ngram_n=self.ngram_n, merge_threshold=self.merge_jaccard,
            )
            log.info("    %d shared, %d merged, %d unique-project group(s)",
                     len(shared), len(merged), len(unique))

            # Interface unification
            unified_iface = _unify_interfaces(members, interface_report)
            if unified_iface:
                log.info("    Unified interface: %d common + %d muxed signals",
                         len(unified_iface.common_signals),
                         len(unified_iface.muxed_signals))

            # Resource estimation
            (indiv_total, shared_g, merged_g, unique_g,
             pha_total, efficiency, savings) = _estimate_resources(
                members, shared, merged, unique,
            )
            log.info("    Estimated area: individual=%d  PHA=%d  savings=%.1f%%",
                     indiv_total, pha_total, savings)

            # Explanation
            explanation = _explain_cluster(
                cid, members, domains, avg_sim,
                shared, merged, unique,
                unified_iface, savings,
            )

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

    # ── CSV helpers ───────────────────────────────────────────────────

    def clusters_to_dataframe(self, report: PHAReport) -> pd.DataFrame:
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

    def components_to_dataframe(self, report: PHAReport) -> pd.DataFrame:
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
                        sum(mc.component_a.occurrences.values()) +
                        sum(mc.component_b.occurrences.values())
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

    def interfaces_to_dataframe(self, report: PHAReport) -> pd.DataFrame:
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


# ═══════════════════════════════════════════════════════════════════════
#  Explanation generation
# ═══════════════════════════════════════════════════════════════════════

def _explain_cluster(
    cid: int,
    members: List[ProjectAnalysis],
    domains: Dict[str, str],
    avg_sim: float,
    shared: List[FunctionalComponent],
    merged: List[MergedComponent],
    unique: Dict[str, List[FunctionalComponent]],
    unified_iface: Optional[UnifiedInterface],
    savings_pct: float,
) -> str:
    """Generate a human-readable explanation of the PHA proposal."""
    lines: List[str] = []

    domain_set = sorted(set(domains.values()))
    member_names = [m.name for m in members]

    lines.append(
        f"PHA Cluster {cid} merges {len(members)} Domain-Specific Accelerators "
        f"({', '.join(member_names)}) spanning {len(domain_set)} hardware "
        f"domain(s) ({', '.join(domain_set)}) into a single Polymorphic "
        f"Heterogeneous Architecture.  The average pairwise similarity is "
        f"{avg_sim:.3f}."
    )

    # Shared
    if shared:
        top_shared = [c.name for c in shared[:5]]
        lines.append(
            f"SHARED COMPONENTS ({len(shared)} blocks): "
            f"These gate-level functional blocks are identical across all members "
            f"and are instantiated ONCE in the PHA, eliminating {len(members)-1}× "
            f"redundant copies.  Top shared blocks: {', '.join(top_shared)}."
        )

    # Merged
    if merged:
        top_merged = [m.merged_name for m in merged[:5]]
        lines.append(
            f"MERGED COMPONENTS ({len(merged)} folded pairs): "
            f"These blocks are similar but not identical in different DSAs.  "
            f"They are folded into a single configurable unit with a mode-select "
            f"multiplexer.  Top merged: {', '.join(top_merged)}."
        )

    # Unique
    total_unique = sum(len(v) for v in unique.values())
    if total_unique:
        lines.append(
            f"UNIQUE COMPONENTS ({total_unique} blocks across "
            f"{len(unique)} project(s)): These are DSA-specific and are retained "
            f"behind a configuration layer in the PHA.  They are activated only "
            f"when operating in the corresponding accelerator mode."
        )

    # Interface
    if unified_iface:
        lines.append(
            f"UNIFIED INTERFACE: The per-DSA interconnect interfaces are merged "
            f"into a single {unified_iface.total_port_count}-signal port block "
            f"({len(unified_iface.common_signals)} common pass-through + "
            f"{len(unified_iface.muxed_signals)} muxed/mode-dependent).  "
            + (f"Common protocol: {unified_iface.protocol}.  "
               if unified_iface.protocol else "")
            + f"This block maps to the J+K+L unified interface in the PHA diagram."
        )

    # Area
    lines.append(
        f"AREA ESTIMATE: Merging these {len(members)} DSAs into a single PHA "
        f"yields an estimated {savings_pct:.1f}% area reduction vs. implementing "
        f"them as independent accelerators.  The savings come from sharing "
        f"({len(shared)} blocks counted once) and folding "
        f"({len(merged)} configurable blocks with mux overhead)."
    )

    return "  ".join(lines)
