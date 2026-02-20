"""Gate-level RTL analysis using Icarus Verilog (iverilog + vvp).

VVP netlist format reference
----------------------------
Each logic gate appears as a "functor" line:
    L_<addr> .functor <TYPE> <WIDTH>, <IN0>, <IN1>, <IN2>, <IN3>;
where IN* is either another functor address (v<addr>_<bit>) or a constant
(C4<HEX>).  Flip-flops appear typed as "D_FF" or "DFF8".

Strategy
--------
1. Compile every .v/.sv file in a project via `iverilog -o <vvp> <srcs>`.
2. Parse the resulting VVP text to extract:
   - gate type → gate-type count vector
   - functor-to-functor wiring → directed connectivity graph
3. Fall back to regex-based RTL parsing when iverilog fails.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .config import AnalysisConfig

log = logging.getLogger(__name__)

# ── Gate-type canonical names ──────────────────────────────────────────
GATE_TYPES = [
    "AND", "OR", "NOT", "XOR", "NAND", "NOR", "XNOR", "BUF",
    "DFF", "MUX", "LATCH", "TRISTATE",
]

# Map VVP functor keywords → canonical names
VVP_GATE_MAP: Dict[str, str] = {
    "AND":    "AND",   "NAND":   "NAND",
    "OR":     "OR",    "NOR":    "NOR",
    "XOR":    "XOR",   "XNOR":  "XNOR",
    "NOT":    "NOT",   "BUF":    "BUF",
    "BUFIF0": "TRISTATE", "BUFIF1": "TRISTATE",
    "NOTIF0": "TRISTATE", "NOTIF1": "TRISTATE",
    "D_FF":   "DFF",   "DFF8":   "DFF",
    "DFF16":  "DFF",   "D_FF_APOS": "DFF",
    "MUX":    "MUX",   "LATCH":  "LATCH",
}

# Regex patterns for RTL-level fallback (operates on raw .v text)
REGEX_GATE_MAP: Dict[str, str] = {
    r"\band\b":     "AND",   r"\bnand\b":  "NAND",
    r"\bor\b":      "OR",    r"\bnor\b":   "NOR",
    r"\bxor\b":     "XOR",   r"\bxnor\b":  "XNOR",
    r"\bnot\b":     "NOT",   r"\bbuf\b":   "BUF",
    r"\bbufif\d\b": "TRISTATE",
    r"\balways\b.*?@.*?posedge": "DFF",   # register inference
    r"\$dff\b":     "DFF",
    r"\bmux\b":     "MUX",
    r"\blatch\b":   "LATCH",
}

# ── Data containers ────────────────────────────────────────────────────

@dataclass
class ProjectAnalysis:
    """All analytical data extracted from one RTL project."""
    name: str
    path: Path
    source_files: List[Path] = field(default_factory=list)
    gate_counts: Counter = field(default_factory=Counter)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    modules: List[str] = field(default_factory=list)
    line_count: int = 0
    used_iverilog: bool = False
    error: Optional[str] = None

    # Derived
    @property
    def total_gates(self) -> int:
        return sum(self.gate_counts.values())

    @property
    def gate_vector(self) -> Dict[str, int]:
        return {g: self.gate_counts.get(g, 0) for g in GATE_TYPES}


# ── Main analyser class ────────────────────────────────────────────────

class IVerilogAnalyzer:
    """Compile Verilog projects with iverilog and parse VVP gate netlists."""

    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg
        self._iverilog_available = shutil.which(cfg.iverilog_bin) is not None
        if not self._iverilog_available:
            log.warning(
                "'%s' not found in PATH — falling back to regex analysis",
                cfg.iverilog_bin,
            )

    # ── Public API ────────────────────────────────────────────────────

    def analyze_project(self, project_path: Path) -> ProjectAnalysis:
        """Analyse one RTL project directory, return ProjectAnalysis."""
        pa = ProjectAnalysis(name=project_path.name, path=project_path)
        log.debug("  Collecting source files in %s …", project_path)
        pa.source_files = self._collect_sources(project_path)

        if not pa.source_files:
            pa.error = "No Verilog source files found"
            log.debug("  %s: no Verilog sources found, skipping.", project_path.name)
            return pa

        log.debug(
            "  %s: %d source file(s) found; counting lines …",
            project_path.name, len(pa.source_files),
        )
        pa.line_count = self._count_lines(pa.source_files)
        log.debug("  %s: %d lines of RTL.", project_path.name, pa.line_count)

        if self._iverilog_available:
            log.debug("  %s: running iverilog compilation …", project_path.name)
            self._iverilog_pass(pa)

        if not pa.used_iverilog:
            # Either iverilog unavailable or compilation failed
            log.debug(
                "  %s: falling back to regex gate analysis …",
                project_path.name,
            )
            self._regex_pass(pa)

        log.info(
            "  %s: %d gates, %d nodes, %d edges [%s]",
            pa.name, pa.total_gates,
            pa.graph.number_of_nodes(),
            pa.graph.number_of_edges(),
            "iverilog" if pa.used_iverilog else "regex",
        )
        return pa

    # ── iverilog pipeline ─────────────────────────────────────────────

    def _iverilog_pass(self, pa: ProjectAnalysis) -> None:
        """Compile to VVP then parse gate-level netlist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vvp_path = Path(tmpdir) / "netlist.vvp"
            success = self._compile(pa.source_files, vvp_path, pa.path)
            if not success:
                return
            try:
                vvp_text = vvp_path.read_text(errors="replace")
                gate_counts, graph, modules = self._parse_vvp(vvp_text)
                pa.gate_counts = gate_counts
                pa.graph = graph
                pa.modules = modules
                pa.used_iverilog = True
            except Exception as exc:
                log.debug("VVP parse error for %s: %s", pa.name, exc)

    def _compile(
        self,
        sources: List[Path],
        out_vvp: Path,
        project_root: Path,
    ) -> bool:
        """Run iverilog; return True on success."""
        # Build include-dir flags for every subdirectory containing headers
        inc_dirs = {"-I", str(project_root)}
        for src in sources:
            inc_dirs.update(["-I", str(src.parent)])

        cmd = [
            self.cfg.iverilog_bin,
            "-o", str(out_vvp),
            "-g2012",           # SystemVerilog 2012
            "-s", "__top__",    # if no top specified, iverilog picks first
            *sorted(inc_dirs),
            *[str(s) for s in sources[:200]],  # cap at 200 files
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.cfg.iverilog_timeout,
                cwd=str(project_root),
            )
            if result.returncode == 0:
                return True
            # Some projects compile partially — try without -s flag
            cmd2 = [c for c in cmd if c != "-s" and c != "__top__"]
            result2 = subprocess.run(
                cmd2, capture_output=True, text=True,
                timeout=self.cfg.iverilog_timeout,
                cwd=str(project_root),
            )
            return result2.returncode == 0
        except subprocess.TimeoutExpired:
            log.debug("iverilog timed out for %s", project_root.name)
        except FileNotFoundError:
            pass
        return False

    # ── VVP parser ────────────────────────────────────────────────────

    # Pattern: L_<hex_addr> .functor <TYPE> <WIDTH>, <inputs…>;
    _FUNCTOR_RE = re.compile(
        r"^(L_\w+)\s+\.functor\s+(\w+)\s+\d+,\s*"   # addr and type
        r"(v?\w+(?:_\d+)?)"                            # input 0
        r"(?:,\s*(v?\w+(?:_\d+)?))?"                  # input 1 (opt)
        r"(?:,\s*(v?\w+(?:_\d+)?))?"                  # input 2 (opt)
        r"(?:,\s*(v?\w+(?:_\d+)?))?"                  # input 3 (opt)
        r"\s*;",
        re.MULTILINE,
    )
    # .var  /  .net  lines give signal driver address
    _NET_RE = re.compile(
        r"^\s*\.(?:var|net)\s+\"(\w+)\".*?,\s*(L_\w+)\s*;",
        re.MULTILINE,
    )
    # .scope module declarations
    _SCOPE_RE = re.compile(r"^\s*\.scope\s+module\s+\"(\w+)\"", re.MULTILINE)

    def _parse_vvp(
        self, text: str
    ) -> Tuple[Counter, nx.DiGraph, List[str]]:
        """Parse VVP text → (gate_counts, directed_graph, module_names)."""
        gate_counts: Counter = Counter()
        graph = nx.DiGraph()

        # ── 1. Parse all functor lines ─────────────────────────────
        # Build a map: functor_addr → canonical_gate_type
        addr_to_type: Dict[str, str] = {}

        for m in self._FUNCTOR_RE.finditer(text):
            addr = m.group(1)
            raw_type = m.group(2).upper()
            gate_type = VVP_GATE_MAP.get(raw_type, "BUF")  # default BUF for unknown
            addr_to_type[addr] = gate_type
            gate_counts[gate_type] += 1
            graph.add_node(addr, gate_type=gate_type)

            # ── 2. Add EDGES from each input to this functor ───────
            # Inputs look like "v0x55a1b2c3d4_0" (v + functor addr + _bit)
            for group_idx in range(3, 7):
                inp = m.group(group_idx)
                if inp is None:
                    continue
                if inp.startswith("C"):   # constant — skip
                    continue
                # Extract the functor address from "v<addr>_<bit>"
                src_addr = inp if inp.startswith("L_") else "L_" + inp.lstrip("v").split("_")[0]
                if src_addr in addr_to_type or src_addr != addr:
                    graph.add_edge(src_addr, addr)

        # ── 3. Module names ────────────────────────────────────────
        modules = self._SCOPE_RE.findall(text)

        return gate_counts, graph, modules

    # ── Regex fallback ────────────────────────────────────────────────

    def _regex_pass(self, pa: ProjectAnalysis) -> None:
        """Count gates from raw RTL text (no iverilog required)."""
        combined = self._read_sources(pa.source_files)
        gate_counts: Counter = Counter()
        module_names: List[str] = []
        graph = nx.DiGraph()

        # Module names
        for m in re.finditer(r"\bmodule\s+(\w+)", combined):
            module_names.append(m.group(1))

        # Gate instance pattern:  and_gate U1 (.a(x), .y(z));
        inst_re = re.compile(
            r"\b(and|nand|or|nor|xor|xnor|not|buf|bufif\d|dff|latch|mux)\b",
            re.IGNORECASE,
        )
        for m in inst_re.finditer(combined):
            raw = m.group(1).upper()
            gtype = VVP_GATE_MAP.get(raw, raw)
            gate_counts[gtype] += 1

        # Sequential element heuristic: count always @(posedge …)
        dff_count = len(re.findall(r"always\s*@\s*\(\s*posedge", combined, re.IGNORECASE))
        gate_counts["DFF"] += dff_count

        # Build a simple intra-module signal graph (module→module connections)
        module_map = {n: i for i, n in enumerate(module_names)}
        port_re = re.compile(r"\.(\w+)\s*\(\s*(\w+)\s*\)")
        prev_mod: Optional[str] = None
        for line in combined.splitlines():
            mod_m = re.search(r"\bmodule\s+(\w+)", line)
            if mod_m:
                prev_mod = mod_m.group(1)
            inst_m = re.search(r"\b(\w+)\s+(\w+)\s*\(", line)
            if inst_m and prev_mod:
                inst_type = inst_m.group(1)
                if inst_type in module_map and inst_type != prev_mod:
                    graph.add_edge(prev_mod, inst_type)

        pa.gate_counts = gate_counts
        pa.graph = graph
        pa.modules = module_names

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _collect_sources(root: Path) -> List[Path]:
        srcs: List[Path] = []
        for ext in ("*.v", "*.sv"):
            srcs.extend(root.rglob(ext))
        # Exclude testbenches (common naming patterns)
        return [
            s for s in srcs
            if not any(
                x in s.name.lower()
                for x in ("tb_", "_tb.", "_test.", "testbench")
            )
        ]

    @staticmethod
    def _count_lines(sources: List[Path]) -> int:
        total = 0
        for src in sources:
            try:
                total += sum(1 for _ in src.open(errors="replace"))
            except OSError:
                pass
        return total

    @staticmethod
    def _read_sources(sources: List[Path], max_bytes: int = 5_000_000) -> str:
        parts: List[str] = []
        total = 0
        for src in sources:
            try:
                text = src.read_text(errors="replace")
                parts.append(text)
                total += len(text)
                if total > max_bytes:
                    break
            except OSError:
                pass
        return "\n".join(parts)
