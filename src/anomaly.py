"""Anomaly detection for RTL project feature vectors.

Uses robust Z-score (median ± MAD) so a single outlier does not inflate
the baseline.  Projects whose score exceeds the configured threshold are
flagged with a description of which features drive the anomaly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .iverilog_analyzer import GATE_TYPES, ProjectAnalysis

log = logging.getLogger(__name__)

FEATURE_NAMES = GATE_TYPES + [
    "total_gates",
    "graph_nodes",
    "graph_edges",
    "line_count",
]


@dataclass
class Anomaly:
    project_name: str
    zscore: float
    flagged_features: Dict[str, float] = field(default_factory=dict)
    description: str = ""


class AnomalyDetector:
    """Flag RTL projects that are statistical outliers in their feature space."""

    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg

    def detect(self, projects: List[ProjectAnalysis]) -> List[Anomaly]:
        """Return a (possibly empty) list of anomalous projects."""
        if len(projects) < 3:
            return []

        feature_matrix, names = self._build_matrix(projects)

        # Hardware metrics follow log-normal distributions (few massive
        # projects, many small ones).  Log-transform skew-sensitive
        # features before computing robust Z-scores.
        log_cols = []
        for j, fname in enumerate(FEATURE_NAMES):
            if fname != "graph_density":          # density is already [0,1]
                log_cols.append(j)
        for j in log_cols:
            feature_matrix[:, j] = np.log1p(feature_matrix[:, j])

        # Robust Z-score: (x - median) / (1.4826 * MAD)
        median = np.median(feature_matrix, axis=0)
        mad = np.median(np.abs(feature_matrix - median), axis=0)
        # Skip features with zero MAD (no variance) — they carry no
        # information for outlier detection.
        scale = 1.4826 * mad
        zero_var = (mad == 0)
        scale[zero_var] = 1.0   # placeholder; zeroed below
        z_matrix = np.abs((feature_matrix - median) / scale)
        z_matrix[:, zero_var] = 0.0  # zero-variance features → Z = 0
        row_scores = z_matrix.max(axis=1)

        anomalies: List[Anomaly] = []
        threshold = self.cfg.anomaly_zscore_threshold

        for i, score in enumerate(row_scores):
            if score >= threshold:
                flagged = {
                    FEATURE_NAMES[j]: float(z_matrix[i, j])
                    for j in range(len(FEATURE_NAMES))
                    if z_matrix[i, j] >= threshold
                }
                top_feat = max(flagged, key=flagged.get)
                feat_idx = FEATURE_NAMES.index(top_feat)
                # Show original value (expm1 reverses log1p)
                raw_val = np.expm1(feature_matrix[i, feat_idx]) \
                    if feat_idx in log_cols else feature_matrix[i, feat_idx]
                raw_med = np.expm1(median[feat_idx]) \
                    if feat_idx in log_cols else median[feat_idx]
                desc = (
                    f"Max Z-score {score:.2f} on feature '{top_feat}' "
                    f"(value={raw_val:.1f}, median={raw_med:.1f})"
                )
                anomalies.append(
                    Anomaly(
                        project_name=names[i],
                        zscore=float(score),
                        flagged_features=flagged,
                        description=desc,
                    )
                )

        anomalies.sort(key=lambda a: a.zscore, reverse=True)
        return anomalies

    def to_dataframe(self, anomalies: List[Anomaly]) -> pd.DataFrame:
        rows = [
            {
                "project": a.project_name,
                "max_zscore": round(a.zscore, 4),
                "flagged_features": ", ".join(a.flagged_features.keys()),
                "description": a.description,
            }
            for a in anomalies
        ]
        return pd.DataFrame(rows)

    # ── Private ───────────────────────────────────────────────────────

    @staticmethod
    def _graph_density(pa: ProjectAnalysis) -> float:
        n = pa.graph.number_of_nodes()
        e = pa.graph.number_of_edges()
        return (e / (n * (n - 1))) if n > 1 else 0.0

    def _build_matrix(
        self, projects: List[ProjectAnalysis]
    ):
        names = [p.name for p in projects]
        rows = []
        for p in projects:
            row = [float(p.gate_counts.get(g, 0)) for g in GATE_TYPES]
            row += [
                float(p.total_gates),
                float(p.graph.number_of_nodes()),
                float(p.graph.number_of_edges()),
                float(p.line_count),
            ]
            rows.append(row)
        return np.array(rows, dtype=float), names
