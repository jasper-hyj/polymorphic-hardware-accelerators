"""PHA sub-algorithms: clustering, decomposition, interface unification,
resource estimation, and explanation generation.

These are stateless functions extracted from :mod:`pha_analyzer` so each
concern lives in its own focused module.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .iverilog_analyzer import ProjectAnalysis
from .similarity import extract_ngrams, jaccard_sim, shared_patterns
from .surprise import classify_domain, interpret_pattern
from .interface_analyzer import (
    InterfaceReport,
    ModuleInfo,
    normalise_signal_name,
)

log = logging.getLogger(__name__)

# Import data-model classes (defined in pha_analyzer to avoid circulars)
from .pha_analyzer import (
    FunctionalComponent,
    MergedComponent,
    UnifiedInterface,
)


# ═══════════════════════════════════════════════════════════════════════
#  Clustering
# ═══════════════════════════════════════════════════════════════════════

def agglomerative_clusters(
    df_sim: pd.DataFrame,
    threshold: float = 0.55,
    min_cluster_size: int = 2,
) -> List[List[str]]:
    """Average-linkage agglomerative clustering on a similarity matrix.

    Starts with each project in its own cluster.  At each step, merges the
    pair of clusters whose *average* inter-cluster similarity is highest —
    but only if that similarity is ≥ ``threshold``.  Avoids the "chaining"
    artefact of single-linkage that collapses everything into one mega-cluster.

    Returns lists of project names, each list being one cluster with
    at least *min_cluster_size* members.
    """
    names = list(df_sim.columns)
    n = len(names)
    if n < min_cluster_size:
        return []

    # Pre-load similarity values into a fast numpy array
    sim = df_sim.values.copy()

    # Each cluster is a set of indices; track active clusters
    clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}

    # Cache of average inter-cluster similarity for every pair
    # Start: avg_sim[i][j] = sim[i][j] for singletons
    avg_sim: Dict[Tuple[int, int], float] = {}
    active = set(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            avg_sim[(i, j)] = float(sim[i, j])

    while True:
        # Find the best merge among active clusters
        best_pair: Optional[Tuple[int, int]] = None
        best_val = -1.0
        active_list = sorted(active)
        for ai in range(len(active_list)):
            for aj in range(ai + 1, len(active_list)):
                ci, cj = active_list[ai], active_list[aj]
                key = (min(ci, cj), max(ci, cj))
                val = avg_sim.get(key, -1.0)
                if val > best_val:
                    best_val = val
                    best_pair = key

        if best_pair is None or best_val < threshold:
            break

        # Merge cj into ci
        ci, cj = best_pair
        members_i = clusters[ci]
        members_j = clusters[cj]
        merged = members_i + members_j
        clusters[ci] = merged
        del clusters[cj]
        active.discard(cj)

        # Recompute average similarity from merged cluster to every other
        len_merged = len(merged)
        for ck in active:
            if ck == ci:
                continue
            key_old_i = (min(ci, ck), max(ci, ck))
            key_old_j = (min(cj, ck), max(cj, ck))
            members_k = clusters[ck]
            # Average sim = sum of all pairwise sims / (|merged| × |k|)
            total = 0.0
            for a in merged:
                for b in members_k:
                    total += float(sim[a, b])
            new_key = (min(ci, ck), max(ci, ck))
            avg_sim[new_key] = total / (len_merged * len(members_k))
            # Clean up stale key
            if key_old_j in avg_sim:
                del avg_sim[key_old_j]

    result = []
    for cid in sorted(active):
        if len(clusters[cid]) >= min_cluster_size:
            result.append([names[i] for i in sorted(clusters[cid])])
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Component decomposition
# ═══════════════════════════════════════════════════════════════════════

def decompose_components(
    members: List[ProjectAnalysis],
    ngram_n: int = 3,
    merge_threshold: float = 0.60,
) -> Tuple[
    List[FunctionalComponent],
    List[MergedComponent],
    Dict[str, List[FunctionalComponent]],
]:
    """Classify gate-pattern functional blocks as shared / merged / unique.

    Returns (shared, merged, unique_per_project).
    """
    bags: Dict[str, Counter] = {}
    for p in members:
        bags[p.name] = extract_ngrams(p, ngram_n)

    all_patterns: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(dict)
    for pname, bag in bags.items():
        for pattern, count in bag.items():
            if count > 0:
                all_patterns[pattern][pname] = count

    member_names = {p.name for p in members}

    # ── Shared: pattern present in EVERY member ──
    shared: List[FunctionalComponent] = []
    remaining: Dict[Tuple[str, ...], Dict[str, int]] = {}

    for pattern, proj_counts in all_patterns.items():
        if set(proj_counts.keys()) >= member_names:
            label = interpret_pattern(pattern)
            shared.append(FunctionalComponent(
                name=label or "\u2192".join(pattern),
                pattern=pattern,
                pattern_str="\u2192".join(pattern),
                occurrences=dict(proj_counts),
                semantic_label=label,
            ))
        else:
            remaining[pattern] = proj_counts

    # ── Merged: similar patterns appearing in >= 2 members ──
    merged: List[MergedComponent] = []
    used: Set[Tuple[str, ...]] = set()

    proj_unique: Dict[str, List[Tuple[Tuple[str, ...], int]]] = defaultdict(list)
    for pattern, proj_counts in remaining.items():
        for pname, cnt in proj_counts.items():
            proj_unique[pname].append((pattern, cnt))

    member_list = sorted(member_names)
    for i in range(len(member_list)):
        for j in range(i + 1, len(member_list)):
            p_a, p_b = member_list[i], member_list[j]
            bag_a = Counter({pat: cnt for pat, cnt in proj_unique.get(p_a, [])
                            if pat not in used})
            bag_b = Counter({pat: cnt for pat, cnt in proj_unique.get(p_b, [])
                            if pat not in used})
            if not bag_a or not bag_b:
                continue
            jac = jaccard_sim(bag_a, bag_b)
            if jac >= merge_threshold:
                shared_pats = shared_patterns(bag_a, bag_b, top_n=3)
                for pat_str, cnt_a, cnt_b in shared_pats[:1]:
                    tup = tuple(pat_str.split("\u2192"))
                    if tup in used:
                        continue
                    used.add(tup)
                    label_a = interpret_pattern(tup) or pat_str
                    fc_a = FunctionalComponent(
                        name=label_a, pattern=tup, pattern_str=pat_str,
                        occurrences={p_a: cnt_a},
                        semantic_label=interpret_pattern(tup),
                    )
                    fc_b = FunctionalComponent(
                        name=label_a, pattern=tup, pattern_str=pat_str,
                        occurrences={p_b: cnt_b},
                        semantic_label=interpret_pattern(tup),
                    )
                    merged.append(MergedComponent(
                        component_a=fc_a,
                        component_b=fc_b,
                        jaccard=round(jac, 4),
                        merged_name=(f"{fc_a.name}+{fc_b.name}"
                                     if fc_a.name != fc_b.name
                                     else f"{fc_a.name}(merged)"),
                        mux_overhead=round(0.05 + 0.15 * (1 - jac), 4),
                        projects_involved=[p_a, p_b],
                    ))

    # ── Unique: patterns in only one member ──
    unique: Dict[str, List[FunctionalComponent]] = defaultdict(list)
    for pattern, proj_counts in remaining.items():
        if pattern in used:
            continue
        for pname, cnt in proj_counts.items():
            if len(proj_counts) == 1:
                label = interpret_pattern(pattern)
                unique[pname].append(FunctionalComponent(
                    name=label or "\u2192".join(pattern),
                    pattern=pattern,
                    pattern_str="\u2192".join(pattern),
                    occurrences={pname: cnt},
                    semantic_label=label,
                ))

    shared.sort(key=lambda c: sum(c.occurrences.values()), reverse=True)
    merged.sort(key=lambda m: m.jaccard, reverse=True)
    for pname in unique:
        unique[pname].sort(key=lambda c: sum(c.occurrences.values()), reverse=True)

    return shared, merged, dict(unique)


# ═══════════════════════════════════════════════════════════════════════
#  Interface unification
# ═══════════════════════════════════════════════════════════════════════

def unify_interfaces(
    members: List[ProjectAnalysis],
    interface_report: Optional[InterfaceReport],
) -> Optional[UnifiedInterface]:
    """Build a unified interface block (J+K+L) from compatible ports."""
    if interface_report is None:
        return None

    member_names = {p.name for p in members}
    member_modules: Dict[str, List[ModuleInfo]] = defaultdict(list)
    for mi in interface_report.modules:
        if mi.project in member_names:
            member_modules[mi.project].append(mi)

    if not member_modules:
        return None

    proj_signals: Dict[str, Dict[Tuple[str, str], int]] = {}
    for pname, mods in member_modules.items():
        sigs: Dict[Tuple[str, str], int] = {}
        for mi in mods:
            for port in mi.ports:
                key = (normalise_signal_name(port.name), port.direction)
                sigs[key] = max(sigs.get(key, 0), port.width)
        proj_signals[pname] = sigs

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

    proto: Optional[str] = None
    if interface_report.compatible_pairs:
        for cp in interface_report.compatible_pairs:
            if (cp.module_a.project in member_names
                    and cp.module_b.project in member_names
                    and cp.shared_protocols):
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

def estimate_resources(
    members: List[ProjectAnalysis],
    shared: List[FunctionalComponent],
    merged: List[MergedComponent],
    unique: Dict[str, List[FunctionalComponent]],
) -> Tuple[int, int, int, int, int, float, float]:
    """Estimate gate counts for PHA vs. individual DSAs.

    Returns (individual_total, shared_gates, merged_gates, unique_gates,
             pha_total, efficiency, savings_pct).
    """
    individual_total = sum(p.total_gates for p in members)

    shared_gates = 0
    for fc in shared:
        shared_gates += len(fc.pattern) * max(fc.occurrences.values())

    merged_gates = 0
    for mc in merged:
        avg_gates = len(mc.component_a.pattern) * (
            max(mc.component_a.occurrences.values(), default=0)
            + max(mc.component_b.occurrences.values(), default=0)
        ) / 2
        merged_gates += int(avg_gates * (1 + mc.mux_overhead))

    unique_gates = 0
    for comps in unique.values():
        for fc in comps:
            unique_gates += len(fc.pattern) * sum(fc.occurrences.values())

    pha_total = shared_gates + merged_gates + unique_gates

    # If the PHA estimate exceeds the sum of individual DSAs the merge
    # is counter-productive.  Instead of silently clamping to 85% (which
    # hid real results), report the honest numbers but log a warning.
    if pha_total > individual_total and individual_total > 0:
        log.warning(
            "PHA gate estimate (%d) exceeds individual total (%d); "
            "merge overhead outweighs sharing — area savings negative.",
            pha_total, individual_total,
        )

    if individual_total == 0:
        return 0, 0, 0, 0, 0, 1.0, 0.0

    efficiency = pha_total / individual_total
    savings = 1.0 - efficiency

    return (individual_total, shared_gates, merged_gates, unique_gates,
            pha_total, round(efficiency, 4), round(savings * 100, 2))


# ═══════════════════════════════════════════════════════════════════════
#  Explanation generation
# ═══════════════════════════════════════════════════════════════════════

def explain_cluster(
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

    if shared:
        top = [c.name for c in shared[:5]]
        lines.append(
            f"SHARED COMPONENTS ({len(shared)} blocks): "
            f"These gate-level functional blocks are identical across all members "
            f"and are instantiated ONCE in the PHA, eliminating {len(members)-1}\u00d7 "
            f"redundant copies.  Top shared blocks: {', '.join(top)}."
        )

    if merged:
        top = [m.merged_name for m in merged[:5]]
        lines.append(
            f"MERGED COMPONENTS ({len(merged)} folded pairs): "
            f"These blocks are similar but not identical in different DSAs.  "
            f"They are folded into a single configurable unit with a mode-select "
            f"multiplexer.  Top merged: {', '.join(top)}."
        )

    total_unique = sum(len(v) for v in unique.values())
    if total_unique:
        lines.append(
            f"UNIQUE COMPONENTS ({total_unique} blocks across "
            f"{len(unique)} project(s)): These are DSA-specific and are retained "
            f"behind a configuration layer in the PHA.  They are activated only "
            f"when operating in the corresponding accelerator mode."
        )

    if unified_iface:
        lines.append(
            f"UNIFIED INTERFACE: The per-DSA interconnect interfaces are merged "
            f"into a single {unified_iface.total_port_count}-signal port block "
            f"({len(unified_iface.common_signals)} common pass-through + "
            f"{len(unified_iface.muxed_signals)} muxed/mode-dependent).  "
            + (f"Common protocol: {unified_iface.protocol}.  "
               if unified_iface.protocol else "")
            + "This block maps to the J+K+L unified interface in the PHA diagram."
        )

    lines.append(
        f"AREA ESTIMATE: Merging these {len(members)} DSAs into a single PHA "
        f"yields an estimated {savings_pct:.1f}% area reduction vs. implementing "
        f"them as independent accelerators.  The savings come from sharing "
        f"({len(shared)} blocks counted once) and folding "
        f"({len(merged)} configurable blocks with mux overhead)."
    )

    return "  ".join(lines)
