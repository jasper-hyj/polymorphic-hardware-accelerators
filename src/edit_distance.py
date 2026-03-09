"""Hardware edit distance between two RTL projects.

Computes a multi-level edit distance that quantifies how many operations
are needed to transform one project's architecture into another, measuring
reusability potential.

Three levels of edits:

1. **Gate-level** — Levenshtein distance on per-module gate sequences.
   Operations: add gate, delete gate, substitute gate type (each cost 1).

2. **Module-level** — Optimal bipartite matching (Hungarian algorithm) of
   modules across two projects, augmented with module additions/deletions.

3. **Wiring-level** — Differences in the module interconnect topology.
   Each added/removed inter-module edge costs 1 rewire operation.

Usage::

    from src.edit_distance import EditDistanceAnalyzer
    analyzer = EditDistanceAnalyzer()
    result = analyzer.compare_projects("demo_projects/DSA_X", "demo_projects/DSA_Y")
    print(result.summary)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class InstanceInfo:
    """One sub-module instantiation with its port connections."""

    module_type: str  # e.g. "mac_unit_M"
    instance_name: str  # e.g. "u_M"
    # port_name → connected signal (e.g. {"a": "a_out", "acc": "m_acc"})
    port_connections: Dict[str, str] = field(default_factory=dict)


@dataclass
class VerilogModule:
    """One module extracted from a Verilog file."""

    name: str
    gate_sequence: List[str]  # ordered list of gate types
    ports: Dict[str, str]  # port_name → direction ("input"/"output")
    instantiates: List[Tuple[str, str]]  # (module_type, instance_name)
    instances: List[InstanceInfo] = field(default_factory=list)
    raw_text: str = ""

    @property
    def gate_count(self) -> int:
        return len(self.gate_sequence)


@dataclass
class ProjectArchitecture:
    """Parsed architecture of a Verilog project (module-level view)."""

    name: str
    path: Path
    modules: Dict[str, VerilogModule]  # module_name → VerilogModule
    top_module: Optional[str] = None
    # Module-level connectivity: (src_module, dst_module) edges
    wiring: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def module_names(self) -> List[str]:
        return sorted(self.modules.keys())


@dataclass
class EditOperation:
    """A single edit operation in the transformation."""

    op_type: str
    # "add_gate", "delete_gate", "change_gate",
    # "add_module", "delete_module",
    # "rewire"
    source: str  # what exists in project A (or "" for additions)
    target: str  # what exists in project B (or "" for deletions)
    cost: float
    detail: str

    def __repr__(self) -> str:
        return f"{self.op_type}: {self.detail} (cost={self.cost})"


@dataclass
class ModuleMatch:
    """Mapping of a module in project A to a module in project B."""

    module_a: Optional[str]  # None if this module only exists in B
    module_b: Optional[str]  # None if this module only exists in A
    gate_distance: int  # Levenshtein distance of gate sequences
    gate_operations: List[EditOperation]
    match_type: str  # "identical", "similar", "add", "delete"

    @property
    def label(self) -> str:
        if self.match_type == "identical":
            return f"{self.module_a} == {self.module_b} (identical)"
        elif self.match_type == "similar":
            return f"{self.module_a} -> {self.module_b} (edit dist={self.gate_distance})"
        elif self.match_type == "add":
            return f"+ {self.module_b} (add, cost={self.gate_distance})"
        elif self.match_type == "delete":
            return f"- {self.module_a} (delete, cost={self.gate_distance})"
        return f"{self.module_a} ? {self.module_b}"


# ── Reusability scale ──────────────────────────────────────────────

_REUSE_TIERS = [
    # (min_score, stars, label, short_description)
    (90, "★★★★★", "Drop-in Reusable",
     "Almost identical architectures — plug-and-play with minimal wiring tweaks."),
    (70, "★★★★☆", "Easy Adaptation",
     "Mostly shared modules — a handful of gate swaps and minor restructuring."),
    (40, "★★★☆☆", "Moderate Effort",
     "Significant overlap but meaningful gate or structural changes required."),
    (15, "★★☆☆☆", "Major Rework",
     "Limited overlap — substantial module redesign or addition needed."),
    (0,  "★☆☆☆☆", "Complete Redesign",
     "Architectures share almost nothing — better to build from scratch."),
]


def reusability_score(distance: int, total_gates_a: int, total_gates_b: int) -> float:
    """Convert an edit distance into a 0–100 reusability score.

    Normalises by the larger project's gate count so the metric is
    independent of absolute size.  A score of 100 means the two
    projects are identical; 0 means they share nothing.
    """
    normaliser = max(total_gates_a, total_gates_b, 1)
    raw = 1.0 - distance / normaliser
    return round(max(0.0, min(1.0, raw)) * 100, 1)


def reusability_tier(score: float) -> Tuple[str, str, str]:
    """Return (stars, label, description) for a reusability score."""
    for min_s, stars, label, desc in _REUSE_TIERS:
        if score >= min_s:
            return stars, label, desc
    return _REUSE_TIERS[-1][1:]


def _bar(score: float, width: int = 20) -> str:
    """Render a Unicode progress bar for a 0–100 score."""
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


# Maximum number of modules per project for the full Hungarian matching.
# Projects exceeding this threshold use a fast heuristic instead.
MAX_MODULES_FOR_HUNGARIAN = 500


@dataclass
class ProjectEditDistance:
    """Complete edit distance result between two projects."""

    project_a: str
    project_b: str
    module_matches: List[ModuleMatch]
    wiring_edits: List[EditOperation]

    # Aggregated costs
    gate_edit_cost: int = 0  # total gate adds + deletes + changes
    module_add_cost: int = 0  # gates in added modules
    module_delete_cost: int = 0  # gates in deleted modules
    rewire_cost: int = 0  # wiring changes
    total_distance: int = 0

    # Size context (set by the analyzer)
    total_gates_a: int = 0
    total_gates_b: int = 0

    # Breakdown by operation type
    operation_counts: Dict[str, int] = field(default_factory=dict)

    # Human-readable
    summary: str = ""
    step_by_step: str = ""

    @property
    def normalised_distance(self) -> float:
        """Distance normalised to [0, 1] by max possible edits."""
        normaliser = max(self.total_gates_a, self.total_gates_b, 1)
        return min(1.0, self.total_distance / normaliser)

    @property
    def reuse_score(self) -> float:
        """0–100 reusability score (100 = identical)."""
        return reusability_score(
            self.total_distance, self.total_gates_a, self.total_gates_b,
        )

    @property
    def reuse_tier(self) -> Tuple[str, str, str]:
        """(stars, label, description) for the reusability score."""
        return reusability_tier(self.reuse_score)


# ═══════════════════════════════════════════════════════════════════════
#  Verilog parser (module-level, gate-aware)
# ═══════════════════════════════════════════════════════════════════════

# Gate primitives recognised in structural Verilog
_GATE_KEYWORDS = {
    "and", "nand", "or", "nor", "xor", "xnor", "not", "buf",
    "bufif0", "bufif1", "notif0", "notif1",
}

_MODULE_RE = re.compile(
    r"\bmodule\s+(\w+)\s*"
    r"(?:#\s*\(.*?\))?\s*"  # optional parameter list
    r"\((.*?)\)\s*;",
    re.DOTALL,
)

_PORT_DIR_RE = re.compile(
    r"\b(input|output|inout)\s+(?:wire|reg)?\s*(?:\[\d+:\d+\])?\s*(\w+)",
)

_GATE_INST_RE = re.compile(
    r"\b(and|nand|or|nor|xor|xnor|not|buf|bufif\d|notif\d)\s+\w+\s*\(",
    re.IGNORECASE,
)

_MODULE_INST_RE = re.compile(
    r"\b(\w+)\s+(\w+)\s*\(\s*\.",
)

# Matches a full module instantiation with named port connections, e.g.:
#   mac_unit_M  u_M (.clk(clk), .rst(rst), .a(a_out), .b(coeff), .acc(m_acc));
_MODULE_INST_FULL_RE = re.compile(
    r"\b(\w+)\s+(\w+)\s*\((.*?)\)\s*;",
    re.DOTALL,
)

# Matches individual named port connections:  .port_name(signal_name)
# Handles bit-selects like .in(m_acc[3:0])
_PORT_CONN_RE = re.compile(
    r"\.(\w+)\s*\(\s*([\w]+)(?:\s*\[.*?\])?\s*\)",
)

_ALWAYS_DFF_RE = re.compile(
    r"always\s*@\s*\(\s*posedge",
    re.IGNORECASE,
)

_ASSIGN_MUX_RE = re.compile(
    r"assign\s+\w+(?:\[\d+\])?\s*=\s*\w+\s*\?",
)


def parse_verilog_modules(filepath: Path) -> Dict[str, VerilogModule]:
    """Parse a Verilog file and extract all modules with their gate sequences."""
    text = filepath.read_text(errors="replace")
    modules: Dict[str, VerilogModule] = {}

    # Split text into module blocks
    module_blocks = _split_modules(text)

    known = set(mb[0] for mb in module_blocks)
    for mod_name, mod_text in module_blocks:
        gate_seq = _extract_gate_sequence(mod_text)
        ports = _extract_ports(mod_text)
        instantiations, instances = _extract_instantiations(mod_text, known)

        modules[mod_name] = VerilogModule(
            name=mod_name,
            gate_sequence=gate_seq,
            ports=ports,
            instantiates=instantiations,
            instances=instances,
            raw_text=mod_text,
        )

    return modules


def _split_modules(text: str) -> List[Tuple[str, str]]:
    """Split Verilog text into (module_name, module_body) pairs."""
    blocks: List[Tuple[str, str]] = []
    # Find all module ... endmodule blocks
    pattern = re.compile(
        r"\bmodule\s+(\w+)\s*(?:#\s*\(.*?\))?\s*\(.*?\)\s*;(.*?)\bendmodule\b",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        blocks.append((m.group(1), m.group(2)))
    return blocks


def _extract_gate_sequence(mod_text: str) -> List[str]:
    """Extract ordered gate-type sequence from a module body."""
    gates: List[Tuple[int, str]] = []

    # Structural gate instantiations
    for m in _GATE_INST_RE.finditer(mod_text):
        gate_type = m.group(1).upper()
        # Normalise
        if gate_type.startswith("BUFIF") or gate_type.startswith("NOTIF"):
            gate_type = "TRISTATE"
        gates.append((m.start(), gate_type))

    # Sequential elements (always @(posedge ...))
    for m in _ALWAYS_DFF_RE.finditer(mod_text):
        gates.append((m.start(), "DFF"))

    # Ternary assigns → MUX
    for m in _ASSIGN_MUX_RE.finditer(mod_text):
        gates.append((m.start(), "MUX"))

    # Sort by position in source
    gates.sort(key=lambda x: x[0])
    return [g[1] for g in gates]


def _extract_ports(mod_text: str) -> Dict[str, str]:
    """Extract port names and directions."""
    ports: Dict[str, str] = {}
    for m in _PORT_DIR_RE.finditer(mod_text):
        ports[m.group(2)] = m.group(1)
    return ports


def _extract_instantiations(
    mod_text: str, known_modules: Set[str]
) -> Tuple[List[Tuple[str, str]], List[InstanceInfo]]:
    """Find module instantiations (type, instance_name) with port connections.

    Returns (legacy_list, instance_info_list).
    """
    _SKIP = _GATE_KEYWORDS | {
        "input", "output", "inout", "wire", "reg", "assign",
        "always", "initial", "module", "endmodule",
    }

    insts: List[Tuple[str, str]] = []
    instances: List[InstanceInfo] = []

    for m in _MODULE_INST_FULL_RE.finditer(mod_text):
        mod_type = m.group(1)
        inst_name = m.group(2)
        port_text = m.group(3)

        if mod_type.lower() in _SKIP or mod_type in _SKIP:
            continue

        # Only accept if it really has named-port syntax (contains ".")
        if "." not in port_text:
            continue

        # Parse individual port connections
        port_map: Dict[str, str] = {}
        for pc in _PORT_CONN_RE.finditer(port_text):
            port_name = pc.group(1)
            signal_name = pc.group(2)
            port_map[port_name] = signal_name

        if (mod_type, inst_name) not in insts:
            insts.append((mod_type, inst_name))
            instances.append(InstanceInfo(
                module_type=mod_type,
                instance_name=inst_name,
                port_connections=port_map,
            ))

    return insts, instances


def parse_project(project_path: Path) -> ProjectArchitecture:
    """Parse all Verilog files in a project directory."""
    verilog_files = list(project_path.rglob("*.v")) + list(project_path.rglob("*.sv"))
    # Exclude testbenches
    verilog_files = [
        f for f in verilog_files
        if not any(x in f.name.lower() for x in ("tb_", "_tb.", "_test.", "testbench"))
    ]

    all_modules: Dict[str, VerilogModule] = {}
    for vf in verilog_files:
        mods = parse_verilog_modules(vf)
        all_modules.update(mods)

    if not all_modules:
        log.warning("No modules found in %s", project_path)
        return ProjectArchitecture(
            name=project_path.name, path=project_path, modules={}
        )

    # Identify top module + wiring
    top_module, wiring = _infer_topology(all_modules)

    return ProjectArchitecture(
        name=project_path.name,
        path=project_path,
        modules=all_modules,
        top_module=top_module,
        wiring=wiring,
    )


def _infer_topology(
    modules: Dict[str, VerilogModule],
) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """Identify the top module and infer *signal-level* inter-module wiring.

    For the top module, we trace which signals connect the output ports
    of one sub-module instance to the input ports of another, producing
    edges like  (input_cond_A, mac_unit_M)  instead of flat
    (top, input_cond_A).  For non-top modules that instantiate
    sub-modules, we fall back to simple parent→child edges.
    """
    module_names = set(modules.keys())
    instantiated_modules: Set[str] = set()

    for mod_name, mod in modules.items():
        for inst_type, _inst_name in mod.instantiates:
            if inst_type in module_names and inst_type != mod_name:
                instantiated_modules.add(inst_type)

    # Top module = not instantiated by anyone else
    top_candidates = module_names - instantiated_modules
    top_module = None
    if len(top_candidates) == 1:
        top_module = next(iter(top_candidates))
    elif top_candidates:
        for c in top_candidates:
            if "top" in c.lower():
                top_module = c
                break
        if not top_module:
            top_module = max(top_candidates, key=lambda n: len(modules[n].instantiates))

    wiring = _trace_signal_wiring(modules, top_module)

    return top_module, wiring


def _trace_signal_wiring(
    modules: Dict[str, VerilogModule],
    top_module: Optional[str],
) -> List[Tuple[str, str]]:
    """Trace signal-level connections between sub-module instances.

    For each module that instantiates sub-modules, determine which
    sub-module's output port drives a signal that feeds into another
    sub-module's input port.  This produces accurate dataflow edges
    like  (mac_unit_M → data_sel_N)  instead of  (top → data_sel_N).
    """
    module_names = set(modules.keys())
    wiring: List[Tuple[str, str]] = []

    for parent_name, parent_mod in modules.items():
        if not parent_mod.instances:
            continue

        # Look up port directions of each instantiated child module
        # so we know which ports are outputs (drivers) and which are
        # inputs (consumers).

        # signal_name → list of (module_type, "output")
        signal_drivers: Dict[str, List[str]] = defaultdict(list)
        # signal_name → list of (module_type, "input")
        signal_consumers: Dict[str, List[str]] = defaultdict(list)

        for inst in parent_mod.instances:
            child_mod = modules.get(inst.module_type)
            if child_mod is None:
                continue

            for port_name, signal_name in inst.port_connections.items():
                direction = child_mod.ports.get(port_name)
                if direction == "output":
                    signal_drivers[signal_name].append(inst.module_type)
                elif direction == "input" or direction == "inout":
                    signal_consumers[signal_name].append(inst.module_type)
                elif direction is None:
                    # Port direction unknown — use heuristic: common
                    # output port names
                    if port_name in ("out", "o", "q", "y", "dout",
                                     "data_out", "result", "acc",
                                     "sum", "product", "carry"):
                        signal_drivers[signal_name].append(inst.module_type)
                    else:
                        signal_consumers[signal_name].append(inst.module_type)

        # Build edges: driver_module → consumer_module for each shared signal
        seen_edges: Set[Tuple[str, str]] = set()
        for signal, drivers in signal_drivers.items():
            consumers = signal_consumers.get(signal, [])
            for drv in drivers:
                for cons in consumers:
                    if drv != cons:
                        edge = (drv, cons)
                        if edge not in seen_edges:
                            seen_edges.add(edge)
                            wiring.append(edge)

        # If signal tracing produced no edges (e.g. port directions
        # not parsed), fall back to simple parent→child edges
        if not wiring and parent_mod.instances:
            for inst in parent_mod.instances:
                if inst.module_type in module_names:
                    edge = (parent_name, inst.module_type)
                    if edge not in set(wiring):
                        wiring.append(edge)

    return wiring


# ═══════════════════════════════════════════════════════════════════════
#  Gate-level Levenshtein edit distance
# ═══════════════════════════════════════════════════════════════════════

def gate_levenshtein(
    seq_a: List[str], seq_b: List[str],
) -> Tuple[int, List[EditOperation]]:
    """Compute Levenshtein edit distance on gate-type sequences.

    Returns (distance, list_of_operations).
    Operations: add_gate, delete_gate, change_gate.
    """
    n, m = len(seq_a), len(seq_b)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # delete from A
                    dp[i][j - 1],      # insert from B
                    dp[i - 1][j - 1],  # substitute
                )

    # Backtrace to recover operations
    ops: List[EditOperation] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq_a[i - 1] == seq_b[j - 1]:
            # Match — no edit needed
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            ops.append(EditOperation(
                op_type="change_gate",
                source=seq_a[i - 1],
                target=seq_b[j - 1],
                cost=1,
                detail=f"Change gate {seq_a[i - 1]} → {seq_b[j - 1]}",
            ))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion from A
            ops.append(EditOperation(
                op_type="delete_gate",
                source=seq_a[i - 1],
                target="",
                cost=1,
                detail=f"Delete gate {seq_a[i - 1]}",
            ))
            i -= 1
        else:
            # Insertion from B
            ops.append(EditOperation(
                op_type="add_gate",
                source="",
                target=seq_b[j - 1],
                cost=1,
                detail=f"Add gate {seq_b[j - 1]}",
            ))
            j -= 1

    ops.reverse()
    return dp[n][m], ops


# ═══════════════════════════════════════════════════════════════════════
#  Module-level Hungarian matching
# ═══════════════════════════════════════════════════════════════════════

def _hungarian_match(
    modules_a: Dict[str, VerilogModule],
    modules_b: Dict[str, VerilogModule],
    top_a: Optional[str] = None,
    top_b: Optional[str] = None,
) -> List[ModuleMatch]:
    """Find the optimal module-to-module matching using the Hungarian algorithm.

    Excludes top-level modules (which are structural wrappers) from matching.
    Costs are the gate-level Levenshtein distances.
    Unmatched modules become add/delete operations.
    """
    # Exclude top-level wrapper modules from matching (they just wire things)
    names_a = [n for n in sorted(modules_a) if n != top_a]
    names_b = [n for n in sorted(modules_b) if n != top_b]

    na, nb = len(names_a), len(names_b)
    if na == 0 and nb == 0:
        return []

    # Build cost matrix (na × nb) using gate Levenshtein distance
    # Also precompute gate operations
    dist_cache: Dict[Tuple[str, str], Tuple[int, List[EditOperation]]] = {}
    size = max(na, nb)
    cost_matrix = np.full((size, size), fill_value=1e6, dtype=float)

    for i, name_a in enumerate(names_a):
        for j, name_b in enumerate(names_b):
            dist, ops = gate_levenshtein(
                modules_a[name_a].gate_sequence,
                modules_b[name_b].gate_sequence,
            )
            dist_cache[(name_a, name_b)] = (dist, ops)
            cost_matrix[i, j] = dist

    # Fill dummy rows/cols with "add/delete" costs
    for i in range(na, size):
        for j in range(nb):
            # Adding module j from B (cost = number of gates)
            cost_matrix[i, j] = modules_b[names_b[j]].gate_count
    for j in range(nb, size):
        for i in range(na):
            # Deleting module i from A (cost = number of gates)
            cost_matrix[i, j] = modules_a[names_a[i]].gate_count

    # Solve assignment problem
    row_ind, col_ind = _solve_assignment(cost_matrix)

    matches: List[ModuleMatch] = []
    matched_a: Set[str] = set()
    matched_b: Set[str] = set()

    for r, c in zip(row_ind, col_ind):
        is_real_a = r < na
        is_real_b = c < nb

        if is_real_a and is_real_b:
            name_a = names_a[r]
            name_b = names_b[c]
            dist, ops = dist_cache[(name_a, name_b)]
            matched_a.add(name_a)
            matched_b.add(name_b)

            match_type = "identical" if dist == 0 else "similar"
            matches.append(ModuleMatch(
                module_a=name_a,
                module_b=name_b,
                gate_distance=dist,
                gate_operations=ops,
                match_type=match_type,
            ))
        elif is_real_a and not is_real_b:
            # Module in A has no match — delete
            name_a = names_a[r]
            matched_a.add(name_a)
            gate_count = modules_a[name_a].gate_count
            delete_ops = [
                EditOperation(
                    op_type="delete_gate", source=g, target="", cost=1,
                    detail=f"Delete gate {g} (from deleted module {name_a})",
                )
                for g in modules_a[name_a].gate_sequence
            ]
            matches.append(ModuleMatch(
                module_a=name_a,
                module_b=None,
                gate_distance=gate_count,
                gate_operations=delete_ops,
                match_type="delete",
            ))
        elif not is_real_a and is_real_b:
            # Module in B has no match — add
            name_b = names_b[c]
            matched_b.add(name_b)
            gate_count = modules_b[name_b].gate_count
            add_ops = [
                EditOperation(
                    op_type="add_gate", source="", target=g, cost=1,
                    detail=f"Add gate {g} (for new module {name_b})",
                )
                for g in modules_b[name_b].gate_sequence
            ]
            matches.append(ModuleMatch(
                module_a=None,
                module_b=name_b,
                gate_distance=gate_count,
                gate_operations=add_ops,
                match_type="add",
            ))

    # Catch any unmatched (shouldn't happen with Hungarian, but be safe)
    for name_a in names_a:
        if name_a not in matched_a:
            gc = modules_a[name_a].gate_count
            matches.append(ModuleMatch(
                module_a=name_a, module_b=None,
                gate_distance=gc,
                gate_operations=[
                    EditOperation("delete_gate", g, "", 1,
                                  f"Delete gate {g} (from deleted module {name_a})")
                    for g in modules_a[name_a].gate_sequence
                ],
                match_type="delete",
            ))
    for name_b in names_b:
        if name_b not in matched_b:
            gc = modules_b[name_b].gate_count
            matches.append(ModuleMatch(
                module_a=None, module_b=name_b,
                gate_distance=gc,
                gate_operations=[
                    EditOperation("add_gate", "", g, 1,
                                  f"Add gate {g} (for new module {name_b})")
                    for g in modules_b[name_b].gate_sequence
                ],
                match_type="add",
            ))

    # Sort: identical first, then similar, then add/delete
    type_order = {"identical": 0, "similar": 1, "add": 2, "delete": 3}
    matches.sort(key=lambda m: (type_order.get(m.match_type, 9), m.gate_distance))

    return matches


def _solve_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the linear assignment (Hungarian algorithm).

    Uses scipy if available, otherwise a simple greedy fallback.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost_matrix)
    except ImportError:
        log.debug("scipy not available; using greedy assignment fallback")
        return _greedy_assignment(cost_matrix)


def _greedy_assignment(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple greedy assignment as fallback when scipy is unavailable."""
    n, m = cost_matrix.shape
    used_cols: Set[int] = set()
    rows: List[int] = []
    cols: List[int] = []

    # Flatten and sort all (cost, row, col) entries
    entries = []
    for i in range(n):
        for j in range(m):
            entries.append((cost_matrix[i, j], i, j))
    entries.sort()

    used_rows: Set[int] = set()
    for _cost, i, j in entries:
        if i not in used_rows and j not in used_cols:
            rows.append(i)
            cols.append(j)
            used_rows.add(i)
            used_cols.add(j)
        if len(rows) == min(n, m):
            break

    return np.array(rows), np.array(cols)


# ═══════════════════════════════════════════════════════════════════════
#  Wiring edit distance
# ═══════════════════════════════════════════════════════════════════════

def _compute_wiring_edits(
    wiring_a: List[Tuple[str, str]],
    wiring_b: List[Tuple[str, str]],
    module_mapping: Dict[str, Optional[str]],  # A_name → B_name
) -> List[EditOperation]:
    """Compute wiring changes needed after module matching.

    Translates project A's wiring through the module mapping and compares
    with project B's wiring. Differences are rewire operations.
    """
    # Build reverse mapping too
    mapping_b_to_a: Dict[str, Optional[str]] = {}
    for a_name, b_name in module_mapping.items():
        if b_name:
            mapping_b_to_a[b_name] = a_name

    # Translate A's wiring into B's namespace
    translated_a: Set[Tuple[str, str]] = set()
    for src, dst in wiring_a:
        mapped_src = module_mapping.get(src)
        mapped_dst = module_mapping.get(dst)
        if mapped_src and mapped_dst:
            translated_a.add((mapped_src, mapped_dst))

    set_b = set((s, d) for s, d in wiring_b)

    edits: List[EditOperation] = []

    # Edges in A (translated) but not in B → need to remove
    for src, dst in translated_a - set_b:
        orig_src = mapping_b_to_a.get(src, src)
        orig_dst = mapping_b_to_a.get(dst, dst)
        edits.append(EditOperation(
            op_type="rewire",
            source=f"{orig_src}→{orig_dst}",
            target="(removed)",
            cost=1,
            detail=f"Remove connection {orig_src} → {orig_dst}",
        ))

    # Edges in B but not in translated A → need to add
    for src, dst in set_b - translated_a:
        edits.append(EditOperation(
            op_type="rewire",
            source="(new)",
            target=f"{src}→{dst}",
            cost=1,
            detail=f"Add connection {src} → {dst}",
        ))

    return edits


# ═══════════════════════════════════════════════════════════════════════
#  Main analyzer
# ═══════════════════════════════════════════════════════════════════════

class EditDistanceAnalyzer:
    """Compute the hardware edit distance between two RTL projects."""

    def compare_projects(
        self,
        path_a: Path,
        path_b: Path,
    ) -> ProjectEditDistance:
        """Parse both projects and compute the full edit distance."""
        arch_a = parse_project(path_a)
        arch_b = parse_project(path_b)
        return self.compare_architectures(arch_a, arch_b)

    def compare_architectures(
        self,
        arch_a: ProjectArchitecture,
        arch_b: ProjectArchitecture,
    ) -> ProjectEditDistance:
        """Compute edit distance between two parsed project architectures."""
        na = len(arch_a.modules)
        nb = len(arch_b.modules)

        # If either project is too large for Hungarian matching, use a
        # fast heuristic that avoids the O(n²·g + n³) cost.
        if max(na, nb) > MAX_MODULES_FOR_HUNGARIAN:
            log.info(
                "Fast heuristic for %s (%d modules) <-> %s (%d modules) "
                "[exceeds %d-module cap]",
                arch_a.name, na, arch_b.name, nb,
                MAX_MODULES_FOR_HUNGARIAN,
            )
            return self._compare_fast_heuristic(arch_a, arch_b)

        log.info(
            "Computing edit distance: %s (%d modules) <-> %s (%d modules)",
            arch_a.name, na, arch_b.name, nb,
        )

        # Step 1: Optimal module matching
        matches = _hungarian_match(
            arch_a.modules, arch_b.modules,
            top_a=arch_a.top_module,
            top_b=arch_b.top_module,
        )

        # Step 2: Build module mapping for wiring comparison
        module_mapping: Dict[str, Optional[str]] = {}
        for mm in matches:
            if mm.module_a and mm.module_b:
                module_mapping[mm.module_a] = mm.module_b
            elif mm.module_a:
                module_mapping[mm.module_a] = None
        # Also add top module mapping
        if arch_a.top_module and arch_b.top_module:
            module_mapping[arch_a.top_module] = arch_b.top_module

        # Step 3: Wiring edit distance
        wiring_edits = _compute_wiring_edits(
            arch_a.wiring, arch_b.wiring, module_mapping,
        )

        # Step 4: Aggregate costs
        gate_edit_cost = 0
        module_add_cost = 0
        module_delete_cost = 0
        op_counts: Dict[str, int] = defaultdict(int)

        for mm in matches:
            if mm.match_type == "similar":
                gate_edit_cost += mm.gate_distance
            elif mm.match_type == "add":
                module_add_cost += mm.gate_distance
            elif mm.match_type == "delete":
                module_delete_cost += mm.gate_distance

            for op in mm.gate_operations:
                op_counts[op.op_type] += 1

        rewire_cost = len(wiring_edits)
        op_counts["rewire"] = rewire_cost

        total = gate_edit_cost + module_add_cost + module_delete_cost + rewire_cost

        # Compute total gate counts for reusability scoring
        total_gates_a = sum(
            m.gate_count for m in arch_a.modules.values()
        )
        total_gates_b = sum(
            m.gate_count for m in arch_b.modules.values()
        )

        # Step 5: Generate human-readable report
        summary = self._generate_summary(
            arch_a, arch_b, matches, wiring_edits,
            gate_edit_cost, module_add_cost, module_delete_cost,
            rewire_cost, total, op_counts,
            total_gates_a, total_gates_b,
        )
        step_by_step = self._generate_steps(
            arch_a, arch_b, matches, wiring_edits,
        )

        return ProjectEditDistance(
            project_a=arch_a.name,
            project_b=arch_b.name,
            module_matches=matches,
            wiring_edits=wiring_edits,
            gate_edit_cost=gate_edit_cost,
            module_add_cost=module_add_cost,
            module_delete_cost=module_delete_cost,
            rewire_cost=rewire_cost,
            total_distance=total,
            total_gates_a=total_gates_a,
            total_gates_b=total_gates_b,
            operation_counts=dict(op_counts),
            summary=summary,
            step_by_step=step_by_step,
        )

    def _compare_fast_heuristic(
        self,
        arch_a: ProjectArchitecture,
        arch_b: ProjectArchitecture,
    ) -> ProjectEditDistance:
        """Fast approximate edit distance for oversized projects.

        Instead of O(n²·g + n³) Hungarian matching, we use:
        1. Gate-type histogram diff (symmetric difference of gate counts).
        2. Module-count diff penalty.
        3. Wiring topology diff (edge set symmetric difference).

        This gives a reasonable upper bound without touching the
        Levenshtein DP or the assignment solver.
        """
        from collections import Counter

        # Gate-type histograms across all modules
        hist_a: Counter = Counter()
        hist_b: Counter = Counter()
        total_gates_a = 0
        total_gates_b = 0
        for mod in arch_a.modules.values():
            hist_a.update(mod.gate_sequence)
            total_gates_a += mod.gate_count
        for mod in arch_b.modules.values():
            hist_b.update(mod.gate_sequence)
            total_gates_b += mod.gate_count

        # Symmetric distance on histograms: Σ |count_a[g] - count_b[g]|
        all_gates = set(hist_a) | set(hist_b)
        gate_diff = sum(abs(hist_a.get(g, 0) - hist_b.get(g, 0)) for g in all_gates)

        # Module-count difference (structural penalty)
        mod_diff = abs(len(arch_a.modules) - len(arch_b.modules))

        # Wiring topology diff
        set_a = set(arch_a.wiring)
        set_b = set(arch_b.wiring)
        wire_diff = len(set_a ^ set_b)

        total = gate_diff + mod_diff + wire_diff

        # Build a lightweight summary without individual module matches
        score = reusability_score(total, total_gates_a, total_gates_b)
        stars, tier_label, tier_desc = reusability_tier(score)
        bar = _bar(score)

        summary_lines = [
            f"{'='*64}",
            f"  {arch_a.name}  -->  {arch_b.name}  (fast heuristic)",
            f"{'='*64}",
            "",
            f"  Reusability Score   {score:5.1f} / 100   {stars}  {tier_label}",
            f"  {bar}",
            f"  {tier_desc}",
            "",
            f"  Edit distance       ~{total} operation(s)  [approximate]",
            f"  Project sizes       {arch_a.name}: {total_gates_a} gates, "
            f"{len(arch_a.modules)} modules  |  "
            f"{arch_b.name}: {total_gates_b} gates, {len(arch_b.modules)} modules",
            "",
            "  ── Cost Breakdown (heuristic) ─────────────────────────────",
            f"  Gate histogram difference ......................... {gate_diff:>6}",
            f"  Module count difference .......................... {mod_diff:>6}",
            f"  Wiring topology difference ....................... {wire_diff:>6}",
            f"                                            TOTAL   ~{total:>6}",
            "",
            f"  Note: full Hungarian matching skipped (project has "
            f"{max(len(arch_a.modules), len(arch_b.modules))} modules,",
            f"        cap = {MAX_MODULES_FOR_HUNGARIAN}).  Distance is an approximation.",
        ]

        return ProjectEditDistance(
            project_a=arch_a.name,
            project_b=arch_b.name,
            module_matches=[],
            wiring_edits=[],
            gate_edit_cost=gate_diff,
            module_add_cost=0,
            module_delete_cost=0,
            rewire_cost=wire_diff,
            total_distance=total,
            total_gates_a=total_gates_a,
            total_gates_b=total_gates_b,
            operation_counts={
                "gate_histogram_diff": gate_diff,
                "module_count_diff": mod_diff,
                "wire_topology_diff": wire_diff,
            },
            summary="\n".join(summary_lines),
            step_by_step=(
                f"  [approximate] Transform {arch_a.name} --> {arch_b.name}\n"
                f"  Gate type changes: ~{gate_diff}  |  "
                f"Module additions/removals: ~{mod_diff}  |  "
                f"Rewiring: ~{wire_diff}\n"
            ),
        )

    def compare_all_pairs(
        self, project_paths: List[Path],
    ) -> List[ProjectEditDistance]:
        """Compare all pairs of projects and return sorted results."""
        architectures = [parse_project(p) for p in project_paths]
        results: List[ProjectEditDistance] = []

        for i in range(len(architectures)):
            for j in range(i + 1, len(architectures)):
                result = self.compare_architectures(
                    architectures[i], architectures[j],
                )
                results.append(result)

        results.sort(key=lambda r: r.total_distance)
        return results

    # ── Report generation ─────────────────────────────────────────────

    @staticmethod
    def _generate_summary(
        arch_a: ProjectArchitecture,
        arch_b: ProjectArchitecture,
        matches: List[ModuleMatch],
        wiring_edits: List[EditOperation],
        gate_edit_cost: int,
        module_add_cost: int,
        module_delete_cost: int,
        rewire_cost: int,
        total: int,
        op_counts: Dict[str, int],
        total_gates_a: int = 0,
        total_gates_b: int = 0,
    ) -> str:
        # ── Reusability score ─────────────────────────────────────────
        score = reusability_score(total, total_gates_a, total_gates_b)
        stars, tier_label, tier_desc = reusability_tier(score)
        bar = _bar(score)

        identical = [m for m in matches if m.match_type == "identical"]
        similar = [m for m in matches if m.match_type == "similar"]
        added = [m for m in matches if m.match_type == "add"]
        deleted = [m for m in matches if m.match_type == "delete"]

        total_matched = len(identical) + len(similar)
        total_modules = total_matched + len(added) + len(deleted)
        reuse_pct = round(100 * len(identical) / max(total_modules, 1), 1)
        adaptable_pct = round(100 * total_matched / max(total_modules, 1), 1)

        lines = [
            f"{'='*64}",
            f"  {arch_a.name}  -->  {arch_b.name}",
            f"{'='*64}",
            "",
            f"  Reusability Score   {score:5.1f} / 100   {stars}  {tier_label}",
            f"  {bar}",
            f"  {tier_desc}",
            "",
            f"  Edit distance       {total} operation(s)",
            f"  Project sizes       {arch_a.name}: {total_gates_a} gates, "
            f"{len(arch_a.modules)} modules  |  "
            f"{arch_b.name}: {total_gates_b} gates, {len(arch_b.modules)} modules",
            "",
            "  ── Cost Breakdown ─────────────────────────────────────────",
            f"  Gate edits (swap/add/delete within matched modules) ... {gate_edit_cost:>4}",
            f"  New modules (gates added for entirely new modules) .... {module_add_cost:>4}",
            f"  Removed modules (gates in deleted modules) ............ {module_delete_cost:>4}",
            f"  Rewiring (inter-module connection changes) ............ {rewire_cost:>4}",
            f"                                                   TOTAL  {total:>4}",
            "",
            "  ── Operation Counts ───────────────────────────────────────",
        ]
        for op_name, label in [
            ("change_gate", "Gate type changes"),
            ("add_gate",    "Gate insertions"),
            ("delete_gate", "Gate deletions"),
            ("rewire",      "Wiring changes"),
        ]:
            cnt = op_counts.get(op_name, 0)
            if cnt > 0:
                lines.append(f"  {label:30s} {cnt:>4}")

        lines += [
            "",
            "  ── Module Reuse ───────────────────────────────────────────",
            f"  Modules directly reusable (identical) ..  {len(identical):>3}  "
            f"({reuse_pct:.0f}%)",
            f"  Modules adaptable (identical+similar) ..  {total_matched:>3}  "
            f"({adaptable_pct:.0f}%)",
            f"  New modules needed ....................... {len(added):>3}",
            f"  Modules to remove ........................ {len(deleted):>3}",
        ]

        if identical:
            lines.append("")
            lines.append("  Identical modules (reuse as-is):")
            for m in identical:
                lines.append(f"    = {m.module_a}  ==  {m.module_b}")
        if similar:
            lines.append("")
            lines.append("  Similar modules (need gate edits):")
            for m in similar:
                lines.append(
                    f"    ~ {m.module_a}  -->  {m.module_b}  "
                    f"({m.gate_distance} edit{'s' if m.gate_distance != 1 else ''})"
                )
        if added:
            lines.append("")
            lines.append("  New modules (must be created):")
            for m in added:
                lines.append(
                    f"    + {m.module_b}  ({m.gate_distance} gates)"
                )
        if deleted:
            lines.append("")
            lines.append("  Removed modules:")
            for m in deleted:
                lines.append(
                    f"    - {m.module_a}  ({m.gate_distance} gates)"
                )

        if wiring_edits:
            removes = [w for w in wiring_edits if "Remove" in w.detail]
            adds = [w for w in wiring_edits if "Add" in w.detail]
            lines += [
                "",
                f"  ── Wiring Changes ({len(wiring_edits)}) "
                "──────────────────────────────────────",
            ]
            if removes:
                lines.append("  Disconnect:")
                for we in removes:
                    lines.append(f"    x {we.detail.replace('Remove connection ', '')}")
            if adds:
                lines.append("  Connect:")
                for we in adds:
                    lines.append(f"    + {we.detail.replace('Add connection ', '')}")

        return "\n".join(lines)

    @staticmethod
    def _generate_steps(
        arch_a: ProjectArchitecture,
        arch_b: ProjectArchitecture,
        matches: List[ModuleMatch],
        wiring_edits: List[EditOperation],
    ) -> str:
        """Generate a numbered step-by-step transformation guide."""
        lines = [
            f"  TRANSFORMATION PLAN: {arch_a.name}  -->  {arch_b.name}",
            f"  {'─'*58}",
        ]
        step = 1

        # Phase 1: Reuse
        identical = [m for m in matches if m.match_type == "identical"]
        if identical:
            lines.append("  Phase 1 — Reuse (keep unchanged)")
            for m in identical:
                lines.append(
                    f"    [{step}]  KEEP   '{m.module_a}' "
                    f"(= '{m.module_b}', 0 edits)"
                )
                step += 1
            lines.append("")

        # Phase 2: Adapt
        similar = [m for m in matches if m.match_type == "similar"]
        if similar:
            lines.append("  Phase 2 — Adapt (modify existing modules)")
            for m in similar:
                lines.append(
                    f"    [{step}]  EDIT   '{m.module_a}'  -->  '{m.module_b}'  "
                    f"({m.gate_distance} gate change{'s' if m.gate_distance != 1 else ''}):"
                )
                for op in m.gate_operations:
                    lines.append(f"              {op.detail}")
                step += 1
            lines.append("")

        # Phase 3: Remove
        deleted = [m for m in matches if m.match_type == "delete"]
        if deleted:
            lines.append("  Phase 3 — Remove (delete unneeded modules)")
            for m in deleted:
                lines.append(
                    f"    [{step}]  DROP   '{m.module_a}'  "
                    f"({m.gate_distance} gates)"
                )
                step += 1
            lines.append("")

        # Phase 4: Build
        added = [m for m in matches if m.match_type == "add"]
        if added:
            lines.append("  Phase 4 — Build (create new modules)")
            for m in added:
                gates_str = ", ".join(
                    op.target for op in m.gate_operations if op.target
                )
                lines.append(
                    f"    [{step}]  NEW    '{m.module_b}'  "
                    f"({m.gate_distance} gates: {gates_str})"
                )
                step += 1
            lines.append("")

        # Phase 5: Rewire
        if wiring_edits:
            lines.append("  Phase 5 — Rewire (update connections)")
            for we in wiring_edits:
                tag = "x" if "Remove" in we.detail else "+"
                detail = (we.detail
                          .replace("Remove connection ", "")
                          .replace("Add connection ", ""))
                lines.append(f"    [{step}]  WIRE   {tag} {detail}")
                step += 1
            lines.append("")

        lines.append(f"  {'─'*58}")
        lines.append(f"  Total: {step - 1} transformation step(s)")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Pairwise matrix builder (for integration with the rest of the tool)
# ═══════════════════════════════════════════════════════════════════════

def edit_distance_matrix(
    project_paths: List[Path],
) -> Tuple[List[str], List[List[int]], List[ProjectEditDistance]]:
    """Build a pairwise edit-distance matrix for multiple projects.

    Returns (project_names, distance_matrix, all_results).
    """
    analyzer = EditDistanceAnalyzer()
    architectures = [parse_project(p) for p in project_paths]
    names = [a.name for a in architectures]
    n = len(architectures)
    matrix = [[0] * n for _ in range(n)]
    all_results: List[ProjectEditDistance] = []

    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            done += 1
            if total_pairs > 10 and done % max(1, total_pairs // 10) == 0:
                log.info("  edit-distance progress: %d / %d pairs", done, total_pairs)
            result = analyzer.compare_architectures(architectures[i], architectures[j])
            matrix[i][j] = result.total_distance
            matrix[j][i] = result.total_distance
            all_results.append(result)

    return names, matrix, all_results


def format_executive_summary(
    names: List[str],
    matrix: List[List[int]],
    results: List[ProjectEditDistance],
) -> str:
    """Generate a pretty executive summary with reusability scale.

    This is the top-level report printed before per-pair details.
    """
    W = 72  # report width
    lines: List[str] = []

    lines += [
        "=" * W,
        "  HARDWARE EDIT DISTANCE — EXECUTIVE SUMMARY".center(W),
        "=" * W,
        "",
    ]

    # ── Scale legend ──────────────────────────────────────────────────
    lines.append("  REUSABILITY SCALE")
    lines.append("  " + "─" * (W - 4))
    for min_s, stars, label, desc in _REUSE_TIERS:
        lines.append(f"  {stars}  {label:22s}  (score >= {min_s:>3})  {desc}")
    lines.append("")

    # ── Distance matrix ───────────────────────────────────────────────
    if len(names) <= 30:
        lines.append("  PAIRWISE EDIT DISTANCE MATRIX")
        lines.append("  " + "─" * (W - 4))
        col_w = max(len(n) for n in names) + 2
        header = " " * col_w + "".join(f"{n:>{col_w}}" for n in names)
        lines.append("  " + header)
        for i, name in enumerate(names):
            row = f"{name:<{col_w}}" + "".join(
                f"{matrix[i][j]:>{col_w}}" for j in range(len(names))
            )
            lines.append("  " + row)
        lines.append("")

    # ── Ranked pair table ─────────────────────────────────────────────
    ranked = sorted(results, key=lambda r: -r.reuse_score)

    lines.append("  ALL PAIRS — RANKED BY REUSABILITY (easiest adaptation first)")
    lines.append("  " + "─" * (W - 4))
    lines.append(
        f"  {'#':>3}  {'Project A':<20} {'Project B':<20} "
        f"{'Dist':>5}  {'Score':>6}  {'Rating'}"
    )
    lines.append(
        f"  {'':>3}  {'':─<20} {'':─<20} "
        f"{'':─>5}  {'':─>6}  {'':─<22}"
    )
    for rank, r in enumerate(ranked, 1):
        stars, tier_label, _ = r.reuse_tier
        lines.append(
            f"  {rank:>3}  {r.project_a:<20} {r.project_b:<20} "
            f"{r.total_distance:>5}  {r.reuse_score:>5.1f}%  "
            f"{stars} {tier_label}"
        )
    lines.append("")

    # ── Top-5 easiest adaptations ─────────────────────────────────────
    top_n = min(5, len(ranked))
    if top_n > 0:
        lines.append(f"  TOP {top_n} EASIEST PAIRS TO ADAPT")
        lines.append("  " + "─" * (W - 4))
        for r in ranked[:top_n]:
            stars, tier_label, _ = r.reuse_tier
            bar = _bar(r.reuse_score, 25)

            identical = sum(1 for m in r.module_matches if m.match_type == "identical")
            similar = sum(1 for m in r.module_matches if m.match_type == "similar")
            added = sum(1 for m in r.module_matches if m.match_type == "add")
            deleted = sum(1 for m in r.module_matches if m.match_type == "delete")

            lines += [
                f"  {r.project_a}  -->  {r.project_b}",
                f"    Score: {r.reuse_score:.1f}%  {bar}  {stars} {tier_label}",
                f"    Distance: {r.total_distance} ops  |  "
                f"Modules: {identical} reusable, {similar} adaptable, "
                f"{added} new, {deleted} removed",
                "",
            ]

    # ── Bottom-5 hardest ──────────────────────────────────────────────
    if len(ranked) > top_n:
        bottom_n = min(5, len(ranked))
        lines.append(f"  BOTTOM {bottom_n} HARDEST PAIRS TO ADAPT")
        lines.append("  " + "─" * (W - 4))
        for r in ranked[-bottom_n:]:
            stars, tier_label, _ = r.reuse_tier
            bar = _bar(r.reuse_score, 25)
            lines += [
                f"  {r.project_a}  -->  {r.project_b}",
                f"    Score: {r.reuse_score:.1f}%  {bar}  {stars} {tier_label}",
                f"    Distance: {r.total_distance} ops",
                "",
            ]

    return "\n".join(lines)
