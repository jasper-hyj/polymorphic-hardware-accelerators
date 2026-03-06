"""Diagram-to-string serialiser & string-based comparison orchestrator.

Each project's architecture is converted into a structured text
representation ("diagram string") that encodes:

  - Gate-level topology (sorted BFS traversal of the gate connectivity graph)
  - Functional block annotations (from n-gram semantic labels)
  - Interface / port summary
  - Domain classification

Two diagram strings are then compared via the algorithms in
:mod:`string_algorithms` (LCS, substring matching, Needleman-Wunsch).

Optional LLM analysis is delegated to :mod:`llm_analyzer`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .iverilog_analyzer import GATE_TYPES, ProjectAnalysis
from .similarity import extract_ngrams
from .surprise import classify_domain, interpret_pattern
from .interface_analyzer import (
    InterfaceReport,
    ModuleInfo,
    normalise_signal_name,
    detect_protocols,
)
from .string_algorithms import (
    GATE_ABBREV,
    TOKEN_SEP,
    BLOCK_OPEN,
    BLOCK_CLOSE,
    SharedSubstring,
    AlignmentResult,
    lcs,
    longest_common_substring,
    all_common_substrings,
    needleman_wunsch,
    extract_uncovered,
    find_mergeable,
)
from .llm_analyzer import LLMGraphMatcher

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DiagramString:
    """Structured text representation of one DSA's architecture."""
    project_name: str
    domain: str

    # Core gate-sequence string  (e.g. "D→A→X→D→D→M→A→O")
    gate_sequence: str

    # Compact single-char form  (e.g. "DAXDDMAO")
    compact: str

    # Functional-block annotation string
    # e.g. "[FULL_ADDER]→[SHIFT_REG_3]→[MAC]→[FSM_ELEM]"
    block_sequence: str

    # Interface summary line
    # e.g. "IF{clk:I1,rst:I1,data:I32,valid:O1,ready:O1}[AXI4-Stream]"
    interface_summary: str

    # Raw token list (for alignment algorithms)
    tokens: List[str] = field(default_factory=list)

    # Block token list
    block_tokens: List[str] = field(default_factory=list)

    # Metadata
    total_gates: int = 0
    n_modules: int = 0
    line_count: int = 0

    # Full JSON representation (for LLM prompts)
    json_repr: str = ""


@dataclass
class ComparisonResult:
    """Full comparison between two diagram strings."""
    project_a: str
    project_b: str

    # LCS
    lcs_length: int
    lcs_ratio: float
    lcs_sequence: str

    # Longest common substring
    longest_common_substr: SharedSubstring

    # All significant shared substrings
    shared_substrings: List[SharedSubstring]

    # Global alignment
    alignment: AlignmentResult

    # Derived classification
    shared_blocks: List[str]
    mergeable_candidates: List[Tuple[str, str, float]]
    unique_to_a: List[str]
    unique_to_b: List[str]

    # Summary
    overall_string_similarity: float


# ═══════════════════════════════════════════════════════════════════════
#  Serialiser
# ═══════════════════════════════════════════════════════════════════════

class DiagramSerializer:
    """Convert ProjectAnalysis objects to diagram strings and compare them."""

    def __init__(
        self,
        ngram_n: int = 3,
        min_substr_len: int = 2,
        match_score: float = 2.0,
        mismatch_penalty: float = -1.0,
        gap_penalty: float = -0.5,
    ):
        self.ngram_n = ngram_n
        self.min_substr_len = min_substr_len
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

    # ── Serialise one project ─────────────────────────────────────────

    def serialize(
        self,
        pa: ProjectAnalysis,
        interface_report: Optional[InterfaceReport] = None,
    ) -> DiagramString:
        """Convert a ProjectAnalysis into a DiagramString."""
        domain = classify_domain(pa)

        # 1. Gate sequence via BFS on connectivity graph
        tokens = self._graph_to_sequence(pa)
        gate_seq = TOKEN_SEP.join(tokens) if tokens else "(empty)"
        compact = "".join(GATE_ABBREV.get(t, "?") for t in tokens)

        # 2. Functional-block sequence
        block_tokens, block_seq = self._tokens_to_blocks(tokens)

        # 3. Interface summary
        iface_summary = self._interface_summary(pa, interface_report)

        # 4. JSON representation (for LLM)
        json_repr = self._to_json(pa, domain, tokens, block_tokens, iface_summary)

        return DiagramString(
            project_name=pa.name,
            domain=domain,
            gate_sequence=gate_seq,
            compact=compact,
            block_sequence=block_seq,
            interface_summary=iface_summary,
            tokens=tokens,
            block_tokens=block_tokens,
            total_gates=pa.total_gates,
            n_modules=len(pa.modules),
            line_count=pa.line_count,
            json_repr=json_repr,
        )

    # ── Compare two diagram strings ──────────────────────────────────

    def compare(self, a: DiagramString, b: DiagramString) -> ComparisonResult:
        """Run the full string-based comparison pipeline."""
        # LCS
        lcs_tokens = lcs(a.tokens, b.tokens)
        max_len = max(len(a.tokens), len(b.tokens), 1)
        lcs_ratio = len(lcs_tokens) / max_len

        # Longest common substring
        lc_sub = longest_common_substring(a.tokens, b.tokens)

        # All shared substrings >= min length
        all_shared = all_common_substrings(
            a.tokens, b.tokens, min_len=self.min_substr_len,
        )

        # Global alignment
        alignment = needleman_wunsch(
            a.tokens, b.tokens,
            match=self.match_score,
            mismatch=self.mismatch_penalty,
            gap=self.gap_penalty,
        )

        # Derive shared blocks (map shared substrings to semantic labels)
        shared_blocks: List[str] = []
        for ss in all_shared:
            tup = tuple(ss.pattern.split(TOKEN_SEP))
            label = interpret_pattern(tup)
            ss.semantic_label = label
            if label and label not in shared_blocks:
                shared_blocks.append(label)

        # Unique segments
        covered_a: Set[int] = set()
        covered_b: Set[int] = set()
        for ss in all_shared:
            for i in range(ss.pos_a, ss.pos_a + ss.length):
                covered_a.add(i)
            for i in range(ss.pos_b, ss.pos_b + ss.length):
                covered_b.add(i)
        unique_a = extract_uncovered(a.tokens, covered_a)
        unique_b = extract_uncovered(b.tokens, covered_b)

        # Mergeable candidates
        mergeable = find_mergeable(alignment, a.tokens, b.tokens)

        # Overall similarity (weighted blend)
        # 0.35 LCS ratio + 0.30 alignment + 0.20 shared-substring coverage
        # + 0.15 Jaccard of block tokens
        shared_substr_cov = (
            sum(ss.length for ss in all_shared)
            / max(len(a.tokens) + len(b.tokens), 1)
        )
        block_jaccard = self._block_jaccard(a.block_tokens, b.block_tokens)
        overall = (
            0.35 * lcs_ratio
            + 0.30 * alignment.normalised_score
            + 0.20 * shared_substr_cov
            + 0.15 * block_jaccard
        )

        return ComparisonResult(
            project_a=a.project_name,
            project_b=b.project_name,
            lcs_length=len(lcs_tokens),
            lcs_ratio=round(lcs_ratio, 4),
            lcs_sequence=TOKEN_SEP.join(lcs_tokens),
            longest_common_substr=lc_sub,
            shared_substrings=all_shared,
            alignment=alignment,
            shared_blocks=shared_blocks,
            mergeable_candidates=mergeable,
            unique_to_a=unique_a,
            unique_to_b=unique_b,
            overall_string_similarity=round(overall, 4),
        )

    # ── LLM analysis (delegated to llm_analyzer) ─────────────────────

    def llm_analyze(
        self,
        diagrams: List[DiagramString],
        comparisons: Optional[List[ComparisonResult]] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> str:
        """Send diagram strings to an LLM for intelligent PHA analysis.

        Returns the LLM's analysis as a plain-text string.
        """
        matcher = LLMGraphMatcher(
            api_key=api_key or "",
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return matcher.analyze_text(diagrams, comparisons)

    # ── Private helpers ──────────────────────────────────────────────

    def _graph_to_sequence(self, pa: ProjectAnalysis) -> List[str]:
        """BFS traversal of gate graph -> ordered gate-type token list."""
        g = pa.graph
        gate_attr = nx.get_node_attributes(g, "gate_type")
        if not gate_attr:
            seq: List[str] = []
            for gate in GATE_TYPES:
                cnt = pa.gate_counts.get(gate, 0)
                seq.extend([gate] * min(cnt, 200))
            return seq

        roots = [n for n in g.nodes() if g.in_degree(n) == 0]
        if not roots:
            roots = sorted(g.nodes())[:1]

        visited: Set = set()
        sequence: List[str] = []
        queue = list(roots[:10])

        while queue and len(sequence) < 5000:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            gtype = gate_attr.get(node, "BUF")
            sequence.append(gtype)
            for succ in sorted(g.successors(node)):
                if succ not in visited:
                    queue.append(succ)

        return sequence

    def _tokens_to_blocks(
        self, tokens: List[str],
    ) -> Tuple[List[str], str]:
        """Map runs of gate tokens to named functional blocks where possible."""
        blocks: List[str] = []
        i = 0
        while i < len(tokens):
            matched = False
            for pat_len in range(min(5, len(tokens) - i), 1, -1):
                sub = tuple(tokens[i:i + pat_len])
                label = interpret_pattern(sub)
                if label:
                    tag = label.split("(")[0].strip().replace(" ", "_")
                    blocks.append(f"{BLOCK_OPEN}{tag}{BLOCK_CLOSE}")
                    i += pat_len
                    matched = True
                    break
            if not matched:
                blocks.append(tokens[i])
                i += 1

        block_seq = TOKEN_SEP.join(blocks)
        return blocks, block_seq

    @staticmethod
    def _block_jaccard(blocks_a: List[str], blocks_b: List[str]) -> float:
        """Weighted Jaccard similarity between two block-token lists."""
        from collections import Counter
        if not blocks_a and not blocks_b:
            return 0.0
        ca = Counter(blocks_a)
        cb = Counter(blocks_b)
        keys = set(ca) | set(cb)
        numer = sum(min(ca[k], cb[k]) for k in keys)
        denom = sum(max(ca[k], cb[k]) for k in keys)
        return numer / denom if denom else 0.0

    def _interface_summary(
        self,
        pa: ProjectAnalysis,
        interface_report: Optional[InterfaceReport],
    ) -> str:
        """One-line summary of the project's port interface."""
        parts: List[str] = []
        if interface_report:
            for mi in interface_report.modules:
                if mi.project == pa.name:
                    port_strs = []
                    for port in mi.ports[:12]:
                        d = "I" if port.direction == "input" else "O"
                        port_strs.append(
                            f"{normalise_signal_name(port.name)}:{d}{port.width}"
                        )
                    protos = detect_protocols(mi)
                    proto_str = f"[{','.join(protos)}]" if protos else ""
                    parts.append(
                        f"{mi.name}{{{',' .join(port_strs)}}}{proto_str}"
                    )
                    if len(parts) >= 5:
                        break
        else:
            for mod in pa.modules[:5]:
                parts.append(f"{mod}{{}}")

        return " | ".join(parts) if parts else "(no interface data)"

    def _to_json(
        self,
        pa: ProjectAnalysis,
        domain: str,
        tokens: List[str],
        block_tokens: List[str],
        iface: str,
    ) -> str:
        """Structured JSON for LLM consumption."""
        obj = {
            "project": pa.name,
            "domain": domain,
            "total_gates": pa.total_gates,
            "line_count": pa.line_count,
            "modules": pa.modules[:20],
            "gate_counts": {g: pa.gate_counts.get(g, 0) for g in GATE_TYPES},
            "gate_sequence_length": len(tokens),
            "gate_sequence_head": TOKEN_SEP.join(tokens[:80]),
            "functional_blocks": block_tokens[:40],
            "interface": iface,
            "graph_nodes": pa.graph.number_of_nodes(),
            "graph_edges": pa.graph.number_of_edges(),
        }
        return json.dumps(obj, indent=2)
