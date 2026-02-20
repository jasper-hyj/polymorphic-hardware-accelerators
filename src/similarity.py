"""Pairwise similarity computation between RTL projects.

Three complementary metrics are computed:

1. **Gate-type cosine similarity**
   Each project is described by a normalised count vector over the 12
   canonical gate types (AND, OR, NOT, XOR, …, DFF, MUX, …).  The cosine
   distance between two such vectors captures functional composition similarity.

2. **Structural graph similarity**
   Derived from the gate-level connectivity graph: graph density, average
   clustering coefficient, degree-sequence entropy, and connected-component
   count are compared via normalised L1 distance.

3. **Partial gate-pattern (n-gram) Jaccard similarity**
   Short gate-type sequences (n-grams) are extracted from graph traversal —
   e.g. "AND→XOR→DFF".  Jaccard overlap of n-gram sets measures how many
   sub-circuit patterns are shared between two projects, even when overall
   gate counts differ.  This detects IP reuse and common design idioms.

Weights for the combined score are configurable (default 0.50 / 0.25 / 0.25).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Dict, FrozenSet, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .iverilog_analyzer import GATE_TYPES, ProjectAnalysis

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────

def _gate_vector(pa: ProjectAnalysis) -> np.ndarray:
    """Return a 1-D float array of gate counts (one element per GATE_TYPES)."""
    vec = np.array([pa.gate_counts.get(g, 0) for g in GATE_TYPES], dtype=float)
    return vec


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1]; returns 0 for zero vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _graph_features(pa: ProjectAnalysis) -> np.ndarray:
    """Compact structural feature vector from the gate-level connectivity graph."""
    g = pa.graph
    n = g.number_of_nodes()
    e = g.number_of_edges()

    density = (e / (n * (n - 1))) if n > 1 else 0.0

    # Clustering (treat directed graph as undirected for clustering)
    try:
        import networkx as nx
        ug = g.to_undirected()
        clustering = nx.average_clustering(ug) if n > 0 else 0.0
    except Exception:
        clustering = 0.0

    # Degree-sequence entropy (captures structural complexity)
    if n > 0:
        degrees = [d for _, d in g.degree()]
        total_deg = sum(degrees) or 1
        probs = [d / total_deg for d in degrees if d > 0]
        entropy = -sum(p * math.log2(p) for p in probs) if probs else 0.0
    else:
        entropy = 0.0

    # Normalise entropy to [0, 1] using log2(n) as maximum
    max_entropy = math.log2(n) if n > 1 else 1.0
    norm_entropy = min(entropy / max_entropy, 1.0)

    # Number of weakly-connected components (normalised)
    try:
        import networkx as nx
        n_components = nx.number_weakly_connected_components(g) if n > 0 else 1
    except Exception:
        n_components = 1
    norm_components = 1.0 / n_components if n_components > 0 else 0.0

    return np.array([density, clustering, norm_entropy, norm_components], dtype=float)


def _structural_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised L1 distance converted to similarity in [0, 1]."""
    diff = np.abs(a - b)
    # Each feature is in [0,1]; max possible L1 = len(diff)
    raw = float(np.sum(diff)) / len(diff)
    return 1.0 - raw


# ── Partial gate-pattern (n-gram) extraction ──────────────────────────

def _extract_ngrams(pa: ProjectAnalysis, n: int = 3) -> Counter:
    """Extract gate-type n-grams via BFS on the connectivity graph.

    For each node in the graph, follow outgoing edges up to depth *n-1*
    and record the sequence of gate types along each path as a frozen tuple.
    When the graph has no edges (regex fallback), build n-grams from the
    sorted gate-type sequence weighted by count.

    Returns a Counter of (gate_type, …) tuples.
    """
    g = pa.graph
    gate_attr = nx.get_node_attributes(g, "gate_type")
    ngrams: Counter = Counter()

    if g.number_of_edges() > 0 and len(gate_attr) > 0:
        # Graph-based n-grams: DFS paths of length n
        for start in g.nodes():
            start_type = gate_attr.get(start, "UNK")
            # DFS up to depth n-1
            paths = [[start]]
            for _ in range(n - 1):
                next_paths = []
                for path in paths:
                    last = path[-1]
                    successors = list(g.successors(last))
                    if successors:
                        for s in successors[:4]:   # cap fan-out for speed
                            next_paths.append(path + [s])
                    else:
                        next_paths.append(path)    # keep dead-end paths
                paths = next_paths
            for path in paths:
                seq = tuple(gate_attr.get(node, "UNK") for node in path)
                if len(seq) >= 2:  # at least a pair
                    ngrams[seq] += 1
    else:
        # Sequence-based n-grams from gate type list (order by gate type)
        seq: List[str] = []
        for gate, count in sorted(pa.gate_counts.items()):
            seq.extend([gate] * count)
        for i in range(len(seq) - n + 1):
            ngrams[tuple(seq[i : i + n])] += 1
        # Also add all 2-grams for small projects
        for i in range(len(seq) - 1):
            ngrams[tuple(seq[i : i + 2])] += 1

    return ngrams


def _jaccard_sim(a: Counter, b: Counter) -> float:
    """Weighted Jaccard similarity: sum(min) / sum(max) over shared n-grams."""
    all_keys = set(a.keys()) | set(b.keys())
    if not all_keys:
        return 0.0
    numerator   = sum(min(a[k], b[k]) for k in all_keys)
    denominator = sum(max(a[k], b[k]) for k in all_keys)
    return numerator / denominator if denominator > 0 else 0.0


def _shared_patterns(
    a: Counter,
    b: Counter,
    top_n: int = 10,
) -> List[Tuple[str, int, int]]:
    """Return the top-n shared gate n-grams with counts in each project."""
    shared = set(k for k in a if a[k] > 0 and b[k] > 0)
    ranked = sorted(shared, key=lambda k: min(a[k], b[k]), reverse=True)[:top_n]
    return [
        ("→".join(k), a[k], b[k])
        for k in ranked
    ]


# ── Main class ────────────────────────────────────────────────────────

class SimilarityEngine:
    """Compute pairwise similarity between ProjectAnalysis objects."""

    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg

    def compute_matrix(
        self, projects: List[ProjectAnalysis], ngram_n: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (combined, gate_type, structural, ngram) similarity DataFrames.

        All four DataFrames are square, indexed and columned by project name.

        *ngram_n* controls the gate-sequence length used for partial matching (
        default 3: tri-grams like AND→XOR→DFF).
        """
        names = [p.name for p in projects]
        n = len(projects)

        gate_vecs   = [_gate_vector(p)       for p in projects]
        struct_vecs = [_graph_features(p)    for p in projects]
        ngram_bags  = [_extract_ngrams(p, ngram_n) for p in projects]

        gate_mat   = np.eye(n, dtype=float)
        struct_mat = np.eye(n, dtype=float)
        ngram_mat  = np.eye(n, dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                gs = _cosine_sim(gate_vecs[i], gate_vecs[j])
                ss = _structural_sim(struct_vecs[i], struct_vecs[j])
                ns = _jaccard_sim(ngram_bags[i], ngram_bags[j])
                gate_mat[i, j]   = gate_mat[j, i]   = gs
                struct_mat[i, j] = struct_mat[j, i] = ss
                ngram_mat[i, j]  = ngram_mat[j, i]  = ns

        # Combined: split weights 3 ways if partial_weight is defined,
        # otherwise fall back to gate + structural only.
        pw = getattr(self.cfg, "partial_weight", 0.25)
        gw = self.cfg.gate_type_weight * (1 - pw)
        sw = self.cfg.structural_weight * (1 - pw)
        combined_mat = gw * gate_mat + sw * struct_mat + pw * ngram_mat

        df_gate     = pd.DataFrame(gate_mat,     index=names, columns=names)
        df_struct   = pd.DataFrame(struct_mat,   index=names, columns=names)
        df_ngram    = pd.DataFrame(ngram_mat,    index=names, columns=names)
        df_combined = pd.DataFrame(combined_mat, index=names, columns=names)

        return df_combined, df_gate, df_struct, df_ngram

    def partial_matches_table(
        self,
        projects: List[ProjectAnalysis],
        ngram_n: int = 3,
        top_pairs: int = 10,
        top_patterns: int = 8,
    ) -> pd.DataFrame:
        """Return a tidy DataFrame of the top shared gate-pattern pairs.

        Columns: project_a, project_b, jaccard, shared_pattern, count_a, count_b
        """
        names = [p.name for p in projects]
        ngram_bags = [_extract_ngrams(p, ngram_n) for p in projects]

        pair_scores = []
        for i in range(len(projects)):
            for j in range(i + 1, len(projects)):
                score = _jaccard_sim(ngram_bags[i], ngram_bags[j])
                if score > 0:
                    pair_scores.append((score, i, j))

        pair_scores.sort(reverse=True)
        rows = []
        for score, i, j in pair_scores[:top_pairs]:
            shared = _shared_patterns(ngram_bags[i], ngram_bags[j], top_patterns)
            for pattern, ca, cb in shared:
                rows.append(
                    {
                        "project_a": names[i],
                        "project_b": names[j],
                        "jaccard_similarity": round(score, 4),
                        "shared_gate_pattern": pattern,
                        "count_in_a": ca,
                        "count_in_b": cb,
                        "min_shared": min(ca, cb),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def top_pairs(
        df: pd.DataFrame,
        n: int = 10,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """Return the top-n most similar pairs as a tidy DataFrame."""
        rows = []
        names = list(df.columns)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if exclude_self and i >= j:
                    continue
                rows.append({"project_a": a, "project_b": b, "similarity": df.iloc[i, j]})
        result = pd.DataFrame(rows).sort_values("similarity", ascending=False).head(n)
        result = result.reset_index(drop=True)
        result.index += 1  # 1-based rank
        return result
