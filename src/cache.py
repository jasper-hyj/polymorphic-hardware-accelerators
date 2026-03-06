"""File-based cache for expensive analysis stages.

Two caching layers are provided:

1. **Gate-extraction cache** — keyed by a SHA-256 digest of all Verilog
   source files in a project.  Stores ``gate_counts``, graph edge-list,
   module names, line count, and whether iverilog was used.  This avoids
   re-running iverilog / regex parsing when source files haven't changed.

2. **Similarity-matrix cache** — keyed by the sorted set of
   (project_name, content_hash) tuples together with the similarity
   weights and n-gram configuration.  Stores the four pairwise similarity
   DataFrames (combined, gate-type, structural, n-gram) as CSV.

Both layers store plain-text artefacts under a configurable cache
directory (default ``.cache/``), safe to delete at any time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

log = logging.getLogger(__name__)


# ── Hashing helpers ───────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """SHA-256 hex digest of a single file (reads in 64 KiB chunks)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65_536), b""):
                h.update(chunk)
    except OSError:
        h.update(b"__missing__")
    return h.hexdigest()


def project_content_hash(source_files: List[Path]) -> str:
    """Stable SHA-256 digest of all source files in a project.

    The hash is computed over the sorted relative-path + individual file
    hashes so that file ordering on disk doesn't matter.
    """
    h = hashlib.sha256()
    entries: List[str] = []
    for sf in sorted(source_files, key=lambda p: str(p)):
        entries.append(f"{sf.name}:{_file_hash(sf)}")
    h.update("\n".join(entries).encode())
    return h.hexdigest()


def similarity_config_hash(
    project_hashes: Dict[str, str],
    gate_weight: float,
    struct_weight: float,
    partial_weight: float,
    ngram_n: int,
) -> str:
    """Hash the project set + similarity parameters into a cache key."""
    h = hashlib.sha256()
    # Sorted (name, hash) pairs ensure deterministic ordering
    for name in sorted(project_hashes):
        h.update(f"{name}:{project_hashes[name]}".encode())
    h.update(f"|gw={gate_weight}|sw={struct_weight}|pw={partial_weight}|n={ngram_n}".encode())
    return h.hexdigest()


# ── Gate-extraction cache ────────────────────────────────────────────

class GateCache:
    """Read/write per-project gate-extraction results from disk.

    Directory layout::

        <cache_dir>/gates/<project_name>/<content_hash>.json
    """

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir / "gates"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, project_name: str, content_hash: str) -> Path:
        safe = project_name.replace("/", "_").replace("\\", "_")
        return self._dir / safe / f"{content_hash}.json"

    # ── read ──

    def load(
        self, project_name: str, content_hash: str
    ) -> Optional[Dict]:
        """Return cached dict or None if miss."""
        p = self._path_for(project_name, content_hash)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            log.debug("  Cache HIT  %s (%s…)", project_name, content_hash[:12])
            return data
        except Exception as exc:
            log.debug("  Cache read error for %s: %s", project_name, exc)
            return None

    # ── write ──

    def save(
        self,
        project_name: str,
        content_hash: str,
        gate_counts: Counter,
        graph: nx.DiGraph,
        modules: List[str],
        line_count: int,
        used_iverilog: bool,
    ) -> None:
        p = self._path_for(project_name, content_hash)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Serialise graph as edge-list + node attributes
        edges = list(graph.edges())
        node_attrs = {
            str(n): graph.nodes[n].get("gate_type", "")
            for n in graph.nodes()
        }

        data = {
            "gate_counts": dict(gate_counts),
            "edges": edges,
            "node_attrs": node_attrs,
            "modules": modules,
            "line_count": line_count,
            "used_iverilog": used_iverilog,
        }
        try:
            p.write_text(json.dumps(data, indent=1), encoding="utf-8")
            log.debug("  Cache SAVE %s (%s…)", project_name, content_hash[:12])
        except OSError as exc:
            log.warning("  Cache write failed for %s: %s", project_name, exc)

    # ── reconstruct ──

    @staticmethod
    def to_graph(data: Dict) -> nx.DiGraph:
        """Reconstruct an ``nx.DiGraph`` from cached edge/node data."""
        g = nx.DiGraph()
        for node, gtype in data.get("node_attrs", {}).items():
            if gtype:
                g.add_node(node, gate_type=gtype)
            else:
                g.add_node(node)
        for u, v in data.get("edges", []):
            g.add_edge(u, v)
        return g


# ── Similarity-matrix cache ─────────────────────────────────────────

class SimilarityCache:
    """Read/write the four pairwise similarity DataFrames.

    Directory layout::

        <cache_dir>/similarity/<config_hash>/combined.csv
                                             gate_type.csv
                                             structural.csv
                                             ngram.csv
    """

    _NAMES = ("combined", "gate_type", "structural", "ngram")

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir / "similarity"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _sub(self, config_hash: str) -> Path:
        return self._dir / config_hash

    def load(
        self, config_hash: str
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        sub = self._sub(config_hash)
        frames: List[pd.DataFrame] = []
        for name in self._NAMES:
            csv_path = sub / f"{name}.csv"
            if not csv_path.exists():
                return None
            try:
                df = pd.read_csv(csv_path, index_col=0)
                frames.append(df)
            except Exception:
                return None
        log.info("  Similarity cache HIT (%s…)", config_hash[:12])
        return tuple(frames)  # type: ignore[return-value]

    def save(
        self,
        config_hash: str,
        combined: pd.DataFrame,
        gate_type: pd.DataFrame,
        structural: pd.DataFrame,
        ngram: pd.DataFrame,
    ) -> None:
        sub = self._sub(config_hash)
        sub.mkdir(parents=True, exist_ok=True)
        for name, df in zip(self._NAMES, [combined, gate_type, structural, ngram]):
            df.to_csv(sub / f"{name}.csv")
        log.info("  Similarity cache SAVE (%s…)", config_hash[:12])


# ── Cache manager (combines both) ───────────────────────────────────

class AnalysisCache:
    """Unified façade over gate-extraction and similarity caches."""

    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = cache_dir
        self.gates = GateCache(cache_dir)
        self.similarity = SimilarityCache(cache_dir)

    @staticmethod
    def clear(cache_dir: Path) -> int:
        """Delete all cached artefacts.  Returns number of files removed."""
        import shutil
        count = 0
        if cache_dir.exists():
            for item in sorted(cache_dir.iterdir()):
                if item.is_file():
                    item.unlink()
                    count += 1
                elif item.is_dir():
                    for f in item.rglob("*"):
                        if f.is_file():
                            count += 1
                    shutil.rmtree(item)
        return count
