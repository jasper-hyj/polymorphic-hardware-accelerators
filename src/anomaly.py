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
    "graph_density",
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
        # Robust Z-score: (x - median) / (1.4826 * MAD)
        median = np.median(feature_matrix, axis=0)
        mad = np.median(np.abs(feature_matrix - median), axis=0)
        # Avoid division by zero
        scale = 1.4826 * np.where(mad == 0, 1e-9, mad)
        z_matrix = np.abs((feature_matrix - median) / scale)
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
                desc = (
                    f"Max Z-score {score:.2f} on feature '{top_feat}' "
                    f"(value={feature_matrix[i, FEATURE_NAMES.index(top_feat)]:.1f}, "
                    f"median={median[FEATURE_NAMES.index(top_feat)]:.1f})"
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
                self._graph_density(p),
                float(p.line_count),
            ]
            rows.append(row)
        return np.array(rows, dtype=float), names
