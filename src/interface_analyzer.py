"""Interface compatibility analysis for hardware reusability.

For each Verilog module in the dataset this module:
  1. Parses every port declaration (name, direction, width).
  2. Detects standard bus protocols (AXI4, Wishbone, APB, SPI, …).
  3. Computes a cross-project compatibility score — after normalising
     signal names (stripping common prefixes/suffixes such as ``i_``,
     ``o_``, ``_n``) — together with a concrete rename / resize recipe.
  4. Assigns a composite reusability score per module.

The headline insight: if two modules from completely different hardware
domains share ≥ 70 % of their port shapes after name normalisation, the
only work required to reuse one inside the other project is a thin
``assign`` rename wrapper — no architectural changes needed.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .iverilog_analyzer import ProjectAnalysis

log = logging.getLogger(__name__)

# ── Bit-width helper ──────────────────────────────────────────────────


def _parse_width(range_str: Optional[str]) -> int:
    """Convert ``[msb:lsb]`` text to integer bit-width (1 if absent)."""
    if not range_str:
        return 1
    m = re.match(r"\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]", range_str)
    if m:
        return abs(int(m.group(1)) - int(m.group(2))) + 1
    return 1


# ── Protocol signatures ───────────────────────────────────────────────

_PROTOCOL_MIN_MATCH = 3   # keyword hits required to confirm a protocol

PROTOCOLS: Dict[str, List[str]] = {
    "AXI4-Full": [
        "awvalid", "awready", "awaddr", "wvalid", "wready", "wdata",
        "bvalid", "bready", "arvalid", "arready", "araddr",
        "rvalid", "rready", "rdata",
    ],
    "AXI4-Lite": [
        "awvalid", "awready", "awaddr", "wvalid", "wready",
        "arvalid", "arready", "araddr", "rvalid", "rready",
    ],
    "AXI4-Stream": ["tvalid", "tready", "tdata", "tlast"],
    "AHB": [
        "haddr", "hwrite", "hwdata", "hrdata", "hready", "hresp", "hsel",
    ],
    "APB": [
        "psel", "penable", "pwrite", "paddr", "pwdata", "prdata", "pready",
    ],
    "Wishbone": ["cyc", "stb", "ack", "we", "adr", "dat_i", "dat_o"],
    "SPI":  ["sclk", "mosi", "miso", "cs_n"],
    "I2C":  ["scl", "sda"],
    "UART": ["uart_tx", "uart_rx"],
    "PCIe": ["pcie", "tlp", "cfg_"],
}

# ── Name normalisation ────────────────────────────────────────────────

_STRIP_PREFIX = re.compile(
    r"^(i_|o_|s_|m_|in_|out_|io_|port_|sig_|ctl_|cfg_)", re.I
)
_STRIP_SUFFIX = re.compile(
    r"(_i|_o|_in|_out|_n|_p|_s|_m|_q|_d|_reg|_wire|_sig)$", re.I
)


def _normalise(name: str) -> str:
    name = _STRIP_PREFIX.sub("", name)
    name = _STRIP_SUFFIX.sub("", name)
    return name.lower()


# ── Data model ────────────────────────────────────────────────────────

@dataclass
class Port:
    name: str
    direction: str        # "input" | "output" | "inout"
    width: int = 1

    @property
    def normalised_name(self) -> str:
        return _normalise(self.name)


@dataclass
class ModuleInfo:
    project: str
    name: str
    source_file: str
    ports: List[Port] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)

    @property
    def port_count(self) -> int:
        return len(self.ports)

    @property
    def input_count(self) -> int:
        return sum(1 for p in self.ports if p.direction == "input")

    @property
    def output_count(self) -> int:
        return sum(1 for p in self.ports if p.direction == "output")

    def signature(self) -> Tuple[Tuple[str, int], ...]:
        """Sorted (direction, width) fingerprint — identical ⟹ plug-compatible."""
        return tuple(sorted((p.direction, p.width) for p in self.ports))


@dataclass
class CompatibilityPair:
    module_a: ModuleInfo
    module_b: ModuleInfo
    port_match_score: float            # exact name+dir+width match fraction
    name_match_score: float            # match after name normalisation
    rename_recipe: List[Tuple[str, str]]       # [(name_in_a, name_in_b)]
    resize_recipe: List[Tuple[str, int, int]]  # [(norm_name, w_a, w_b)]
    shared_protocols: List[str]
    effort: str                        # "zero" | "low" | "medium" | "high"
    explanation: str


@dataclass
class InterfaceReport:
    modules: List[ModuleInfo]
    compatible_pairs: List[CompatibilityPair]
    protocol_coverage: Dict[str, List[str]]    # protocol → [project, …]
    signature_clusters: Dict[str, List[ModuleInfo]]  # sig_hash → [mods]
    reusability_scores: pd.DataFrame            # one row per module


# ── Verilog port parser ───────────────────────────────────────────────

# ANSI-style: input/output [range] [type] name ,|;|)
_ANSI_PORT_RE = re.compile(
    r"\b(input|output|inout)\s+"
    r"(?:wire|reg|logic|signed|unsigned\s*)?"
    r"(\[[^\]]*\])?\s*"
    r"(?:wire|reg|logic|signed|unsigned\s*)?"
    r"(\w+)\s*[,;)]",
    re.IGNORECASE,
)

# Old-style: "input [range] name ;" on its own line
_OLD_PORT_RE = re.compile(
    r"^\s*(input|output|inout)\s+"
    r"(?:wire|reg|logic|signed|unsigned\s*)?"
    r"(\[[^\]]*\])?\s*"
    r"(\w+)\s*;",
    re.IGNORECASE | re.MULTILINE,
)

# module <name> [#(...)] (
_MODULE_RE = re.compile(r"\bmodule\s+(\w+)\s*[#(;]", re.IGNORECASE)

_RESERVED = frozenset(
    {"wire", "reg", "logic", "signed", "unsigned", "module", "endmodule",
     "input", "output", "inout", "parameter", "localparam"}
)


def _parse_verilog_modules(path: Path) -> List[ModuleInfo]:
    """Return all ModuleInfo objects found in a single .v / .sv file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    # Strip comments
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)

    modules: List[ModuleInfo] = []
    mod_matches = list(_MODULE_RE.finditer(text))

    for idx, mod_m in enumerate(mod_matches):
        mod_name = mod_m.group(1)
        start = mod_m.start()
        end = (
            mod_matches[idx + 1].start()
            if idx + 1 < len(mod_matches)
            else len(text)
        )
        chunk = text[start:end]
        mi = ModuleInfo(project="", name=mod_name, source_file=str(path))
        seen: set = set()

        for m in _ANSI_PORT_RE.finditer(chunk):
            direction = m.group(1).lower()
            width = _parse_width(m.group(2))
            name = m.group(3)
            if name.lower() in _RESERVED:
                continue
            key = (name.lower(), direction)
            if key not in seen:
                seen.add(key)
                mi.ports.append(Port(name=name, direction=direction, width=width))

        # Fall back to old-style if ANSI parser found nothing
        if not mi.ports:
            for m in _OLD_PORT_RE.finditer(chunk):
                direction = m.group(1).lower()
                width = _parse_width(m.group(2))
                name = m.group(3)
                if name.lower() in _RESERVED:
                    continue
                key = (name.lower(), direction)
                if key not in seen:
                    seen.add(key)
                    mi.ports.append(Port(name=name, direction=direction, width=width))

        if mi.ports:
            modules.append(mi)

    return modules


# ── Protocol detection ────────────────────────────────────────────────

def detect_protocols(mi: ModuleInfo) -> List[str]:
    """Return protocol names whose keywords are sufficiently present."""
    port_blob = " ".join(p.name.lower() for p in mi.ports)
    detected = []
    for proto, keywords in PROTOCOLS.items():
        hits = sum(1 for kw in keywords if kw in port_blob)
        if hits >= _PROTOCOL_MIN_MATCH:
            detected.append(proto)
    return detected


# ── Compatibility scoring ─────────────────────────────────────────────

def _match_ports(
    a_ports: List[Port],
    b_ports: List[Port],
) -> Tuple[float, float, List[Tuple[str, str]], List[Tuple[str, int, int]]]:
    """
    Returns (exact_score, norm_score, rename_recipe, resize_recipe).

    exact_score   Fraction of ports with matching name + direction + width.
    norm_score    Fraction matched after name normalisation (ignoring renames).
    rename_recipe [(a_name, b_name)] pairs that map to the same norm name.
    resize_recipe [(norm_name, w_a, w_b)] where widths differ.
    """
    if not a_ports or not b_ports:
        return 0.0, 0.0, [], []

    b_by_norm: Dict[Tuple[str, str], Port] = {
        (p.normalised_name, p.direction): p for p in b_ports
    }
    b_by_exact: Dict[Tuple[str, str], Port] = {
        (p.name.lower(), p.direction): p for p in b_ports
    }

    exact_hits = sum(
        1 for p in a_ports
        if (p.name.lower(), p.direction) in b_by_exact
        and b_by_exact[(p.name.lower(), p.direction)].width == p.width
    )

    norm_hits = 0
    rename_recipe: List[Tuple[str, str]] = []
    resize_recipe: List[Tuple[str, int, int]] = []

    for p in a_ports:
        key = (p.normalised_name, p.direction)
        if key not in b_by_norm:
            continue
        bp = b_by_norm[key]
        norm_hits += 1
        if p.name.lower() != bp.name.lower():
            rename_recipe.append((p.name, bp.name))
        if p.width != bp.width:
            resize_recipe.append((p.normalised_name, p.width, bp.width))

    denom = max(len(a_ports), len(b_ports))
    return exact_hits / denom, norm_hits / denom, rename_recipe, resize_recipe


def _effort(score: float, renames: int, resizes: int) -> str:
    if score >= 0.95 and renames == 0 and resizes == 0:
        return "zero"      # drop-in compatible — no wrapper
    if score >= 0.70 and renames <= 4 and resizes == 0:
        return "low"       # thin assign/alias rename wrapper
    if score >= 0.50:
        return "medium"    # adapter module needed
    return "high"          # significant interface rework


def _explain_pair(cp: CompatibilityPair) -> str:
    a, b = cp.module_a, cp.module_b
    parts = [
        f"Module '{a.name}' ({a.project}) and '{b.name}' ({b.project}) "
        f"share {cp.name_match_score:.0%} port compatibility after signal-name "
        f"normalisation (exact match: {cp.port_match_score:.0%}).  "
    ]
    if cp.shared_protocols:
        parts.append(
            f"Both implement the {', '.join(cp.shared_protocols)} protocol — "
            "they can share a bus without any adapter.  "
        )
    if cp.rename_recipe:
        top = cp.rename_recipe[:3]
        parts.append(
            f"Reuse effort: {cp.effort.upper()}.  "
            f"{len(cp.rename_recipe)} signal rename(s) needed: "
            + ", ".join(f"'{x}' → '{y}'" for x, y in top)
            + ("…" if len(cp.rename_recipe) > 3 else "")
            + ".  "
        )
    else:
        parts.append(f"Reuse effort: {cp.effort.upper()}.  No renames required.  ")
    if cp.resize_recipe:
        top = cp.resize_recipe[:2]
        parts.append(
            f"{len(cp.resize_recipe)} width mismatch(es): "
            + ", ".join(f"'{n}' {wa}b vs {wb}b" for n, wa, wb in top)
            + ".  "
        )
    if cp.effort == "zero":
        parts.append(
            "These modules are drop-in pin-compatible — "
            "swap one for the other in any top-level instantiation."
        )
    elif cp.effort == "low":
        parts.append(
            "A thin wrapper with only assign/alias renames is sufficient "
            "to make these modules substitutable in a top-level design."
        )
    elif cp.effort == "medium":
        parts.append(
            "A small adapter module (width conversion + signal bridging) "
            "is needed, but the internal logic of either module is reusable as-is."
        )
    return "".join(parts)


# ── Reusability scoring ───────────────────────────────────────────────

def _reusability_score(mi: ModuleInfo, sig_size: int, total_modules: int) -> float:
    """
    Composite score [0, 1]:
      40 % — protocol compliance (speaks a documented standard bus)
      30 % — port-shape prevalence (same signature appears in many projects)
      30 % — interface simplicity (fewer ports → easier to connect)
    """
    proto_score  = min(1.0, len(mi.protocols) * 0.5)
    prev_score   = min(1.0, (sig_size - 1) / max(total_modules * 0.08, 1))
    simplicity   = 1.0 / (1.0 + max(0, mi.port_count - 6) * 0.12)
    return round(0.40 * proto_score + 0.30 * prev_score + 0.30 * simplicity, 4)


# ── Main analyser ─────────────────────────────────────────────────────

class InterfaceAnalyzer:
    """Parse module port lists and identify cross-project reuse opportunities."""

    def __init__(
        self,
        min_ports: int = 2,
        min_compatibility: float = 0.40,
        max_pairs: int = 300,
    ):
        self.min_ports = min_ports
        self.min_compatibility = min_compatibility
        self.max_pairs = max_pairs

    def analyse(self, projects: List[ProjectAnalysis]) -> InterfaceReport:
        log.info("Parsing Verilog module port interfaces …")
        all_modules: List[ModuleInfo] = []

        for p_idx, pa in enumerate(projects, 1):
            verilog_srcs = [
                Path(src) for src in pa.source_files
                if Path(src).suffix.lower() in {".v", ".sv", ".vh", ".svh"}
            ]
            proj_modules_before = len(all_modules)
            log.debug(
                "  [%d/%d] Parsing %s (%d source file(s)) …",
                p_idx, len(projects), pa.name, len(verilog_srcs),
            )
            for src in verilog_srcs:
                path = Path(src)
                for mi in _parse_verilog_modules(path):
                    mi.project = pa.name
                    if mi.port_count >= self.min_ports:
                        mi.protocols = detect_protocols(mi)
                        all_modules.append(mi)
            proj_mods = len(all_modules) - proj_modules_before
            log.debug(
                "    → %d module interface(s) found in %s",
                proj_mods, pa.name,
            )

        log.info(
            "  %d parseable module interfaces across %d projects.",
            len(all_modules), len(projects),
        )

        # Protocol coverage
        proto_coverage: Dict[str, List[str]] = defaultdict(list)
        for mi in all_modules:
            for proto in mi.protocols:
                if mi.project not in proto_coverage[proto]:
                    proto_coverage[proto].append(mi.project)

        # Signature clusters
        sig_clusters: Dict[str, List[ModuleInfo]] = defaultdict(list)
        for mi in all_modules:
            sig_clusters[str(mi.signature())].append(mi)

        # Reusability scores
        total = max(len(all_modules), 1)
        reuse_rows = []
        for mi in all_modules:
            sig_key = str(mi.signature())
            score = _reusability_score(mi, len(sig_clusters[sig_key]), total)
            reuse_rows.append({
                "project":                mi.project,
                "module":                 mi.name,
                "source_file":            mi.source_file,
                "port_count":             mi.port_count,
                "input_count":            mi.input_count,
                "output_count":           mi.output_count,
                "protocols":              ", ".join(mi.protocols),
                "same_signature_modules": len(sig_clusters[sig_key]),
                "reusability_score":      score,
            })
        df_reuse = (
            pd.DataFrame(reuse_rows)
            .sort_values("reusability_score", ascending=False)
            .reset_index(drop=True)
        )

        # Cross-project compatibility pairs
        mods_by_proj: Dict[str, List[ModuleInfo]] = defaultdict(list)
        for mi in all_modules:
            mods_by_proj[mi.project].append(mi)

        proj_list = list(mods_by_proj.keys())
        total_proj_pairs = len(proj_list) * (len(proj_list) - 1) // 2
        total_mod_comparisons = sum(
            len(mods_by_proj[proj_list[i]]) * len(mods_by_proj[proj_list[j]])
            for i in range(len(proj_list))
            for j in range(i + 1, len(proj_list))
        )
        log.info(
            "  Cross-project compatibility: %d project-pair(s), "
            "~%d module-pair comparisons to run …",
            total_proj_pairs, total_mod_comparisons,
        )

        scored: list = []
        _LOG_EVERY = max(1, total_proj_pairs // 10)   # log ~10 progress lines
        for i in range(len(proj_list)):
            for j in range(i + 1, len(proj_list)):
                pair_idx = i * (len(proj_list) - 1) - i * (i - 1) // 2 + (j - i - 1)
                if pair_idx % _LOG_EVERY == 0:
                    log.info(
                        "    Comparing project pair %d/%d: %s vs %s "
                        "(%d × %d modules) …",
                        pair_idx + 1, total_proj_pairs,
                        proj_list[i], proj_list[j],
                        len(mods_by_proj[proj_list[i]]),
                        len(mods_by_proj[proj_list[j]]),
                    )
                hits_this_pair = 0
                for ma in mods_by_proj[proj_list[i]]:
                    for mb in mods_by_proj[proj_list[j]]:
                        exact, norm, renames, resizes = _match_ports(
                            ma.ports, mb.ports
                        )
                        if norm >= self.min_compatibility:
                            scored.append((norm, ma, mb, exact, norm, renames, resizes))
                            hits_this_pair += 1
                if hits_this_pair:
                    log.debug(
                        "      → %d compatible pair(s) found (%s vs %s)",
                        hits_this_pair, proj_list[i], proj_list[j],
                    )

        log.info(
            "  Sorting %d candidate compatible pair(s) …", len(scored)
        )
        scored.sort(key=lambda x: x[0], reverse=True)
        pairs: List[CompatibilityPair] = []
        for _, ma, mb, exact, norm, renames, resizes in scored[: self.max_pairs]:
            shared_proto = list(set(ma.protocols) & set(mb.protocols))
            effort = _effort(norm, len(renames), len(resizes))
            cp = CompatibilityPair(
                module_a=ma, module_b=mb,
                port_match_score=exact,
                name_match_score=norm,
                rename_recipe=renames,
                resize_recipe=resizes,
                shared_protocols=shared_proto,
                effort=effort,
                explanation="",
            )
            cp.explanation = _explain_pair(cp)
            pairs.append(cp)

        log.info(
            "  %d cross-project compatible module pairs "
            "(min compatibility %.0f%%).",
            len(pairs), self.min_compatibility * 100,
        )

        return InterfaceReport(
            modules=all_modules,
            compatible_pairs=pairs,
            protocol_coverage=dict(proto_coverage),
            signature_clusters=dict(sig_clusters),
            reusability_scores=df_reuse,
        )

    # ── CSV helpers ───────────────────────────────────────────────────

    def pairs_to_dataframe(self, report: InterfaceReport) -> pd.DataFrame:
        rows = []
        for cp in report.compatible_pairs:
            rows.append({
                "project_a":        cp.module_a.project,
                "module_a":         cp.module_a.name,
                "project_b":        cp.module_b.project,
                "module_b":         cp.module_b.name,
                "port_match_exact": round(cp.port_match_score, 4),
                "port_match_norm":  round(cp.name_match_score, 4),
                "shared_protocols": ", ".join(cp.shared_protocols),
                "rename_count":     len(cp.rename_recipe),
                "resize_count":     len(cp.resize_recipe),
                "effort":           cp.effort,
                "rename_recipe":    "; ".join(
                    f"{a}→{b}" for a, b in cp.rename_recipe[:6]
                ),
                "resize_recipe":    "; ".join(
                    f"{n}:{wa}b→{wb}b" for n, wa, wb in cp.resize_recipe[:6]
                ),
            })
        return pd.DataFrame(rows)
