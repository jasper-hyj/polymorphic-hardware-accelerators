"""Surprise analysis: find unexpectedly similar projects across different domains.

The core idea
-------------
Two projects that serve completely different purposes (e.g. a RISC-V CPU and a
neural-network accelerator) are *surprising* when their gate-level similarity is
high.  The surprise score is:

    surprise = combined_similarity × domain_dissimilarity

where domain_dissimilarity is 0 for same-domain pairs and up to 1.0 for pairs
from maximally-different domains.  High surprise scores pinpoint reusable
hardware idioms that transcend architectural boundaries.

Gate-pattern semantic interpretation
-------------------------------------
Common gate n-gram sequences are mapped to named functional blocks so the
report can explain *why* two projects share patterns (e.g. "both contain a
shift-register backbone" or "both implement a carry-propagate adder cell").
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

log = logging.getLogger(__name__)

from .iverilog_analyzer import ProjectAnalysis
from .similarity import _extract_ngrams, _jaccard_sim, _shared_patterns


# ── Domain taxonomy ───────────────────────────────────────────────────

DOMAINS: Dict[str, List[str]] = {
    "CPU / Processor":         ["risc", "cpu", "processor", "core", "mips",
                                 "arm", "riscv", "picorv", "neorv", "vex",
                                 "ibex", "rocket", "cva6", "serv", "zipcpu",
                                 "biriscv", "hazard", "kian"],
    "ML / Neural Accelerator": ["accelerator", "neural", "cnn", "tpu", "npu",
                                 "snn", "eyeriss", "levit", "systolic",
                                 "npusim", "neurospector"],
    "Signal / Image Processing":["image", "processing", "video", "jpeg",
                                  "filter", "fft", "dsp", "convol", "flower"],
    "Arithmetic Unit":          ["posit", "fpu", "arith", "float", "fixed",
                                  "divider", "adder", "mult", "alu"],
    "Memory / Cache":           ["memory", "cache", "dram", "sram", "fifo",
                                  "buffer", "ram", "rom"],
    "Communication / IO":       ["ethernet", "pcie", "usb", "uart", "spi",
                                  "i2c", "axi", "dma", "phy", "mac",
                                  "corundum", "verilog-eth", "verilog-axi",
                                  "verilog-pcie", "wireguard"],
    "FPGA Tool / Framework":    ["fpga", "yosys", "openroad", "openfpga",
                                  "litex", "filament", "nexus", "chisel",
                                  "fud", "primitives"],
    "GPU / Parallel":           ["gpu", "vortex", "miaow", "shader",
                                  "parallel", "simd"],
    "SoC / Platform":           ["soc", "platform", "opentitan", "pulp",
                                  "wujian", "openc910", "chipyard",
                                  "oh", "ultraemb"],
    "Misc / Unknown":           [],
}

# Adjacency: how related are two domains? 0=same, higher=more different
DOMAIN_DISTANCE: Dict[Tuple[str, str], float] = {
    # Same domain handled separately (→ 0.0)
    ("CPU / Processor",          "SoC / Platform"):           0.3,
    ("CPU / Processor",          "ML / Neural Accelerator"):  0.7,
    ("CPU / Processor",          "Arithmetic Unit"):          0.5,
    ("CPU / Processor",          "Memory / Cache"):           0.5,
    ("CPU / Processor",          "Communication / IO"):       0.7,
    ("CPU / Processor",          "Signal / Image Processing"):0.8,
    ("CPU / Processor",          "GPU / Parallel"):           0.6,
    ("CPU / Processor",          "FPGA Tool / Framework"):    0.8,
    ("ML / Neural Accelerator",  "Signal / Image Processing"):0.4,
    ("ML / Neural Accelerator",  "Arithmetic Unit"):          0.5,
    ("ML / Neural Accelerator",  "GPU / Parallel"):           0.5,
    ("ML / Neural Accelerator",  "Memory / Cache"):           0.6,
    ("ML / Neural Accelerator",  "Communication / IO"):       0.9,
    ("ML / Neural Accelerator",  "FPGA Tool / Framework"):    0.7,
    ("Signal / Image Processing","Arithmetic Unit"):          0.4,
    ("Signal / Image Processing","Communication / IO"):       0.7,
    ("Arithmetic Unit",          "Memory / Cache"):           0.7,
    ("Arithmetic Unit",          "Communication / IO"):       0.8,
    ("Communication / IO",       "FPGA Tool / Framework"):    0.6,
    ("SoC / Platform",           "Memory / Cache"):           0.4,
    ("SoC / Platform",           "Communication / IO"):       0.4,
}


def _domain_distance(d1: str, d2: str) -> float:
    """Return domain dissimilarity in [0, 1]; 0 = same domain."""
    if d1 == d2:
        return 0.0
    key = tuple(sorted([d1, d2]))
    return DOMAIN_DISTANCE.get(key, 1.0)   # type: ignore[arg-type]


def classify_domain(project: ProjectAnalysis) -> str:
    """Assign one domain label to a project from its name and module names."""
    text = " ".join(
        [project.name] + project.modules
    ).lower().replace("-", "").replace("_", "")

    scores: Dict[str, int] = {}
    for domain, keywords in DOMAINS.items():
        if domain == "Misc / Unknown":
            continue
        hit = sum(1 for kw in keywords if kw.replace("-", "").replace("_", "") in text)
        if hit:
            scores[domain] = hit

    if not scores:
        return "Misc / Unknown"
    return max(scores, key=scores.__getitem__)


# ── Gate pattern → functional block semantic map ──────────────────────
#
# Keys are tuples of canonical gate-type names (as extracted by iverilog_analyzer).
# The dict is ordered longest-first so more specific patterns take priority.

PATTERN_SEMANTICS: Dict[Tuple[str, ...], str] = {
    # Adder cells
    ("XOR", "AND"):                      "Half-adder cell (sum + carry)",
    ("XOR", "AND", "OR"):                "Full-adder cell (sum, carry-in, carry-out)",
    ("XOR", "XOR", "AND", "AND", "OR"):  "Full-adder (ripple carry variant)",
    ("AND", "XOR", "OR"):                "Carry-select / carry-propagate cell",

    # Registers & shift registers
    ("DFF", "DFF"):                      "2-stage pipeline / shift register",
    ("DFF", "DFF", "DFF"):               "3-stage shift register / pipeline",
    ("DFF", "DFF", "DFF", "DFF"):        "4-stage shift register / synchroniser",
    ("MUX", "DFF"):                      "Registered multiplexer (enable / load)",
    ("DFF", "MUX"):                      "State-feedback register",
    ("DFF", "MUX", "DFF"):               "State register with loopback (FSM element)",
    ("MUX", "DFF", "MUX"):               "Ping-pong / double-buffer cell",

    # Counters & accumulators
    ("AND", "XOR", "DFF"):               "Accumulator bit-cell (add + register)",
    ("XOR", "DFF"):                      "Toggle flip-flop / Gray-code counter bit",
    ("XOR", "DFF", "AND"):               "LFSR tap cell",
    ("XOR", "AND", "DFF"):               "Binary counter bit-cell",

    # Combinational logic primitives
    ("AND", "OR"):                       "Sum-of-products (SOP) expression",
    ("AND", "OR", "NOT"):                "AND-OR-INVERT (AOI) cell",
    ("OR", "AND", "NOT"):                "OR-AND-INVERT (OAI) / priority encoder cell",
    ("NOT", "AND"):                      "Inhibit / masked gate",
    ("NOT", "AND", "OR"):                "De Morgan equivalent expression",
    ("NAND", "NAND"):                    "NAND-based universal logic (inverter / buffer)",
    ("NOR", "NOR"):                      "NOR-based universal logic",

    # Arbitration / control
    ("OR", "DFF"):                       "Set-dominant latch / flag register",
    ("AND", "DFF"):                      "Gated register (clock / write enable)",
    ("AND", "NOT", "DFF"):               "Reset-priority register",
    ("MUX", "MUX"):                      "Priority multiplexer tree",
    ("DFF", "NOT", "AND"):               "Handshake / token-passing cell",

    # Memory + data-path
    ("TRISTATE", "DFF"):                 "Tri-state bus driver with register",
    ("MUX", "AND", "OR"):                "Bypass / forwarding path cell",
    ("DFF", "XOR"):                      "Parity tracking register",

    # Arithmetic
    ("NOT", "XOR"):                      "XNOR comparator bit",
    ("AND", "XOR", "OR", "DFF"):         "Multiply-accumulate (MAC) cell",
}


def interpret_pattern(ngram: Tuple[str, ...]) -> Optional[str]:
    """Return a human-readable functional description for a gate n-gram."""
    # Exact match
    if ngram in PATTERN_SEMANTICS:
        return PATTERN_SEMANTICS[ngram]
    # Substring match: look for any known pattern that appears as a sub-sequence
    for pat, desc in sorted(PATTERN_SEMANTICS.items(), key=lambda x: -len(x[0])):
        pat_len = len(pat)
        ngram_len = len(ngram)
        if pat_len <= ngram_len:
            for start in range(ngram_len - pat_len + 1):
                if ngram[start : start + pat_len] == pat:
                    return desc
    return None


# ── Surprise analysis ─────────────────────────────────────────────────

@dataclass
class SurprisingPair:
    project_a: str
    project_b: str
    domain_a: str
    domain_b: str
    combined_similarity: float
    domain_dissimilarity: float
    surprise_score: float
    shared_patterns: List[Tuple[str, int, int]]        # (pattern_str, cnt_a, cnt_b)
    pattern_interpretations: List[Tuple[str, str]]     # (pattern_str, meaning)
    explanation: str = ""


@dataclass
class SurpriseReport:
    domain_map: Dict[str, str]                          # project_name → domain
    pairs: List[SurprisingPair]
    total_analysed: int


class SurpriseAnalyzer:
    """Identify and explain unexpectedly similar cross-domain project pairs."""

    def __init__(self, min_similarity: float = 0.30, min_surprise: float = 0.20,
                 ngram_n: int = 3):
        self.min_similarity = min_similarity
        self.min_surprise = min_surprise
        self.ngram_n = ngram_n

    def analyse(
        self,
        projects: List[ProjectAnalysis],
        df_combined: pd.DataFrame,
    ) -> SurpriseReport:
        log.info("  Classifying %d project(s) into hardware domains …", len(projects))
        domain_map = {p.name: classify_domain(p) for p in projects}
        for pname, domain in domain_map.items():
            log.debug("    %s → %s", pname, domain)

        name_to_proj = {p.name: p for p in projects}
        names = list(df_combined.columns)
        total_pairs = len(names) * (len(names) - 1) // 2

        log.info(
            "  Extracting gate n-gram bags (n=%d) for %d project(s) …",
            self.ngram_n, len(projects),
        )
        ngram_bags = {}
        for idx, p in enumerate(projects, 1):
            bag = _extract_ngrams(p, self.ngram_n)
            ngram_bags[p.name] = bag
            log.debug(
                "    [%d/%d] %s: %d distinct n-gram type(s)",
                idx, len(projects), p.name, len(bag),
            )

        log.info(
            "  Scanning %d cross-domain project pair(s) for surprise …",
            total_pairs,
        )
        pairs: List[SurprisingPair] = []
        _LOG_EVERY = max(1, total_pairs // 10)
        done = 0
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if j <= i:
                    continue
                done += 1
                if done % _LOG_EVERY == 0:
                    log.info(
                        "    %d/%d pairs scanned (%.0f%%) …",
                        done, total_pairs, 100 * done / total_pairs,
                    )
                sim = float(df_combined.loc[a, b])
                if sim < self.min_similarity:
                    continue
                d_a = domain_map[a]
                d_b = domain_map[b]
                dd = _domain_distance(d_a, d_b)
                surprise = sim * dd
                if surprise < self.min_surprise:
                    continue

                shared = _shared_patterns(ngram_bags[a], ngram_bags[b], top_n=12)
                # Parse patterns back to tuples for semantic lookup
                interps: List[Tuple[str, str]] = []
                for pat_str, ca, cb in shared:
                    tup = tuple(pat_str.split("→"))
                    meaning = interpret_pattern(tup)
                    if meaning:
                        interps.append((pat_str, meaning))

                explanation = self._explain(
                    a, b, d_a, d_b, sim, dd, shared, interps
                )

                pairs.append(
                    SurprisingPair(
                        project_a=a,
                        project_b=b,
                        domain_a=d_a,
                        domain_b=d_b,
                        combined_similarity=round(sim, 4),
                        domain_dissimilarity=round(dd, 2),
                        surprise_score=round(surprise, 4),
                        shared_patterns=shared,
                        pattern_interpretations=interps,
                        explanation=explanation,
                    )
                )

        pairs.sort(key=lambda p: p.surprise_score, reverse=True)
        log.info(
            "  Surprise analysis complete: %d surprising pair(s) found "
            "(min_surprise=%.2f, min_similarity=%.2f).",
            len(pairs), self.min_surprise, self.min_similarity,
        )
        return SurpriseReport(
            domain_map=domain_map,
            pairs=pairs,
            total_analysed=len(names) * (len(names) - 1) // 2,
        )

    # ── Private ───────────────────────────────────────────────────────

    @staticmethod
    def _explain(
        a: str, b: str,
        d_a: str, d_b: str,
        sim: float, dd: float,
        shared: List[Tuple[str, int, int]],
        interps: List[Tuple[str, str]],
    ) -> str:
        lines = [
            f"'{a}' ({d_a}) and '{b}' ({d_b}) come from different hardware domains "
            f"(dissimilarity={dd:.2f}) yet share a combined similarity of {sim:.3f}, "
            f"yielding a surprise score of {sim*dd:.3f}.",
        ]

        if not shared:
            lines.append(
                "Despite overall metric similarity, no specific gate sequences are "
                "shared — the similarity is driven by a similar gate-type mix rather "
                "than identifiable sub-circuits."
            )
            return "  ".join(lines)

        # Most frequent shared patterns
        top3 = shared[:3]
        pattern_desc = "; ".join(
            f"'{p}' (×{min(ca,cb)} times in both)" for p, ca, cb in top3
        )
        lines.append(f"The strongest shared gate patterns are: {pattern_desc}.")

        if interps:
            func_names = list(dict.fromkeys(m for _, m in interps[:5]))
            lines.append(
                "These patterns are consistent with the following reusable "
                f"functional blocks: {', '.join(func_names)}."
            )
            lines.append(
                "This suggests that despite serving different architectural purposes, "
                "both projects draw on the same fundamental hardware idioms — "
                "likely because these primitives are the natural way to implement "
                "that functionality in synthesisable RTL regardless of domain."
            )
        else:
            lines.append(
                "The shared patterns are repetitive gate chains whose functional "
                "role could not be automatically identified, but their high repetition "
                "count suggests a common data-path structure (e.g. wide buses, "
                "replicated bit-slices, or parameterised arrays)."
            )

        return "  ".join(lines)

    # ── Output helpers ────────────────────────────────────────────────

    def to_dataframe(self, report: SurpriseReport) -> pd.DataFrame:
        rows = []
        for p in report.pairs:
            top_pats = "; ".join(f"{pat}(×{min(ca,cb)})" for pat, ca, cb in p.shared_patterns[:5])
            top_funcs = "; ".join(m for _, m in p.pattern_interpretations[:4])
            rows.append({
                "project_a":              p.project_a,
                "project_b":              p.project_b,
                "domain_a":               p.domain_a,
                "domain_b":               p.domain_b,
                "combined_similarity":    p.combined_similarity,
                "domain_dissimilarity":   p.domain_dissimilarity,
                "surprise_score":         p.surprise_score,
                "top_shared_patterns":    top_pats,
                "functional_blocks":      top_funcs,
            })
        return pd.DataFrame(rows)

    def domain_summary(self, report: SurpriseReport) -> pd.DataFrame:
        from collections import Counter
        counts = Counter(report.domain_map.values())
        return pd.DataFrame(
            [{"domain": d, "project_count": c} for d, c in counts.most_common()]
        )
