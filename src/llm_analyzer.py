"""LLM-based graph-matching analysis for Polymorphic Hardware Accelerators.

This module sends diagram-string representations of DSA architectures to
an LLM (OpenAI-compatible API) and asks it to:

  1. **Match** gate-level graphs across verilog projects by identifying
     shared, mergeable, and unique components.
  2. **Propose** a Polymorphic Heterogeneous Architecture (PHA) layout
     with concrete data-flow descriptions.
  3. **Estimate** area savings from sharing / merging.

Two transport back-ends are supported:

  - ``httpx``  (preferred, installed via ``pip install httpx``)
  - ``urllib`` (stdlib fallback — no extra dependency)

Usage
-----
>>> from src.llm_analyzer import LLMGraphMatcher
>>> matcher = LLMGraphMatcher(api_key="sk-...", model="gpt-4o-mini")
>>> result = matcher.match_graphs(diagrams, comparisons)
>>> print(result.raw_response)
>>> print(result.shared_components)   # parsed JSON list

The module can also be invoked from the CLI via ``--llm-api-key``.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LLMMatchResult:
    """Parsed result from the LLM graph-matching analysis."""

    raw_response: str = ""

    # Parsed fields (best-effort JSON extraction from the LLM reply)
    shared_components: List[Dict[str, Any]] = field(default_factory=list)
    merged_components: List[Dict[str, Any]] = field(default_factory=list)
    unique_components: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    unified_interface: Dict[str, Any] = field(default_factory=dict)
    area_savings_pct: Optional[float] = None
    data_flow: str = ""
    explanation: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.raw_response and not self.raw_response.startswith("("))


# ═══════════════════════════════════════════════════════════════════════
#  Prompt templates
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
You are a hardware architecture expert specialising in Polymorphic
Heterogeneous Architectures (PHA). You are given structured text
representations ("diagram strings") of multiple Domain-Specific
Accelerators (DSAs) and their pairwise comparison results.

Your task:
1. Identify components that are SHARED (identical) across DSAs.
2. Identify components that are MERGEABLE (similar enough to fold
   into a single configurable block with a mode-select multiplexer).
3. Identify components that are UNIQUE to each DSA and must be
   retained behind a configuration layer.
4. Propose a UNIFIED INTERFACE that merges the per-DSA interconnects
   into a single polymorphic port block.
5. Estimate the area savings (%) from sharing and merging.
6. Output a concrete data-flow description of the proposed PHA.

Respond in structured JSON with keys:
{
    "shared_components": [{"name": str, "gate_pattern": str, "description": str}],
    "merged_components": [{"name_a": str, "name_b": str, "merged_name": str,
                           "merge_strategy": str, "mux_overhead_pct": float}],
    "unique_components": {"project_name": [{"name": str, "description": str}]},
    "unified_interface": {"common_signals": [...], "muxed_signals": [...]},
    "area_savings_pct": float,
    "data_flow": str,
    "explanation": str
}
""")

GRAPH_MATCHING_SYSTEM_PROMPT = textwrap.dedent("""\
You are a hardware architecture expert specialising in gate-level
structural analysis of RTL designs.

You are given diagram-string representations of Verilog projects.
Each diagram string encodes a project's gate-level topology as a BFS
traversal of its connectivity graph, annotated with functional-block
labels and interface summaries.

Your task:
1. COMPARE the gate-level graph structures across all given projects.
2. For each pair, IDENTIFY which graph sub-structures (gate sequences)
   are structurally identical, similar, or unique.
3. EXPLAIN why the structural matches exist (shared design idioms,
   IP reuse, common arithmetic patterns, etc.).
4. RANK the project pairs by structural similarity and propose which
   ones would benefit most from hardware sharing in a Polymorphic
   Heterogeneous Architecture.
5. For the top matches, describe the SHARED DATAPATH that could be
   factored out into a single configurable hardware block.

Respond in structured JSON:
{
    "pair_rankings": [
        {
            "project_a": str,
            "project_b": str,
            "structural_similarity": float,
            "shared_subgraphs": [{"pattern": str, "semantic_label": str, "length": int}],
            "merge_potential": str,
            "explanation": str
        }
    ],
    "global_shared_patterns": [{"pattern": str, "description": str, "projects": [str]}],
    "pha_recommendation": str,
    "estimated_area_savings_pct": float
}
""")


# ═══════════════════════════════════════════════════════════════════════
#  Main class
# ═══════════════════════════════════════════════════════════════════════

class LLMGraphMatcher:
    """Use an LLM to match gate-level graphs across Verilog projects.

    Parameters
    ----------
    api_key
        OpenAI-compatible API key.  Falls back to ``OPENAI_API_KEY`` env var.
    model
        Model name (default ``gpt-4o-mini``).
    base_url
        Custom API endpoint URL (default OpenAI).
    max_tokens
        Max tokens in the LLM response.
    temperature
        Sampling temperature (lower = more deterministic).
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_tokens: int = 3000,
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ── Public API ────────────────────────────────────────────────────

    def match_graphs(
        self,
        diagrams: List[Any],
        comparisons: Optional[List[Any]] = None,
    ) -> LLMMatchResult:
        """Send diagram strings to the LLM for PHA graph-matching analysis.

        Parameters
        ----------
        diagrams
            List of ``DiagramString`` objects (from ``diagram_serializer``).
        comparisons
            Optional list of ``ComparisonResult`` objects.

        Returns
        -------
        LLMMatchResult
            Parsed result with shared/merged/unique components and
            area estimates.
        """
        if not self.api_key:
            return LLMMatchResult(
                raw_response=(
                    "(LLM analysis skipped \u2014 no API key. "
                    "Set OPENAI_API_KEY or pass --llm-api-key.)"
                ),
            )

        prompt = self._build_pha_prompt(diagrams, comparisons)
        raw = self._call_llm(SYSTEM_PROMPT, prompt)
        return self._parse_result(raw)

    def match_graph_structures(
        self,
        diagrams: List[Any],
        comparisons: Optional[List[Any]] = None,
    ) -> LLMMatchResult:
        """Focused graph-structure matching across projects.

        Uses a specialised prompt that asks the LLM to rank pairs by
        structural similarity and identify shared sub-graphs.
        """
        if not self.api_key:
            return LLMMatchResult(
                raw_response=(
                    "(LLM analysis skipped \u2014 no API key. "
                    "Set OPENAI_API_KEY or pass --llm-api-key.)"
                ),
            )

        prompt = self._build_graph_matching_prompt(diagrams, comparisons)
        raw = self._call_llm(GRAPH_MATCHING_SYSTEM_PROMPT, prompt)
        return self._parse_result(raw)

    def analyze_text(
        self,
        diagrams: List[Any],
        comparisons: Optional[List[Any]] = None,
    ) -> str:
        """Legacy interface: return the raw LLM response as a string.

        Used by ``DiagramSerializer.llm_analyze()`` and the PHA pipeline.
        """
        result = self.match_graphs(diagrams, comparisons)
        return result.raw_response

    # ── Prompt builders ───────────────────────────────────────────────

    def _build_pha_prompt(
        self,
        diagrams: List[Any],
        comparisons: Optional[List[Any]],
    ) -> str:
        """Build the user prompt for PHA analysis."""
        parts: List[str] = ["# DSA Diagram Strings\n"]

        for ds in diagrams:
            parts.append(f"## {ds.project_name} ({ds.domain})")
            parts.append(f"Gates: {ds.total_gates}  Modules: {ds.n_modules}")
            parts.append(
                f"Gate sequence (head 120 tokens): {ds.gate_sequence[:500]}"
            )
            parts.append(f"Block sequence: {ds.block_sequence[:400]}")
            parts.append(f"Interface: {ds.interface_summary}")
            parts.append("")

        if comparisons:
            parts.append("# Pairwise Comparisons\n")
            for cmp in comparisons:
                parts.append(f"## {cmp.project_a} \u2194 {cmp.project_b}")
                parts.append(
                    f"LCS ratio: {cmp.lcs_ratio:.3f}  "
                    f"Alignment identity: {cmp.alignment.identity_pct:.1f}%  "
                    f"Overall string similarity: "
                    f"{cmp.overall_string_similarity:.3f}"
                )
                parts.append(
                    f"Shared blocks: "
                    f"{', '.join(cmp.shared_blocks[:8]) or 'none identified'}"
                )
                parts.append(
                    f"Alignment:\n  A: {cmp.alignment.aligned_a[:200]}\n"
                    f"     {cmp.alignment.match_line[:200]}\n"
                    f"  B: {cmp.alignment.aligned_b[:200]}"
                )
                parts.append("")

        parts.append(
            "Analyse these DSAs and propose a Polymorphic Heterogeneous "
            "Architecture. Respond in the JSON format described above."
        )
        return "\n".join(parts)

    def _build_graph_matching_prompt(
        self,
        diagrams: List[Any],
        comparisons: Optional[List[Any]],
    ) -> str:
        """Build a prompt focused on graph-structure matching."""
        parts: List[str] = ["# Verilog Project Graph Structures\n"]

        for ds in diagrams:
            parts.append(f"## {ds.project_name} ({ds.domain})")
            parts.append(f"Total gates: {ds.total_gates}  Modules: {ds.n_modules}")
            parts.append(f"Lines of code: {ds.line_count}")
            parts.append(
                f"Gate sequence (BFS traversal, head 150 tokens): "
                f"{ds.gate_sequence[:600]}"
            )
            parts.append(f"Functional blocks: {ds.block_sequence[:500]}")
            parts.append(f"Interface: {ds.interface_summary}")
            parts.append("")

        if comparisons:
            parts.append("# Pairwise Structural Comparisons\n")
            for cmp in comparisons:
                parts.append(f"## {cmp.project_a} \u2194 {cmp.project_b}")
                parts.append(
                    f"LCS ratio: {cmp.lcs_ratio:.3f}  "
                    f"LCS length: {cmp.lcs_length} tokens"
                )
                parts.append(
                    f"Alignment identity: {cmp.alignment.identity_pct:.1f}%  "
                    f"(matches={cmp.alignment.matches}, "
                    f"mismatches={cmp.alignment.mismatches}, "
                    f"gaps={cmp.alignment.gaps})"
                )
                parts.append(
                    f"Overall string similarity: "
                    f"{cmp.overall_string_similarity:.3f}"
                )
                if cmp.shared_substrings:
                    top_ss = cmp.shared_substrings[:5]
                    ss_strs = [
                        f"'{s.compact}'(len={s.length}"
                        + (f",{s.semantic_label}" if s.semantic_label else "")
                        + ")"
                        for s in top_ss
                    ]
                    parts.append(f"Top shared substrings: {', '.join(ss_strs)}")
                if cmp.shared_blocks:
                    parts.append(
                        f"Named shared blocks: {', '.join(cmp.shared_blocks[:8])}"
                    )
                if cmp.mergeable_candidates:
                    parts.append(
                        f"Mergeable candidates: {len(cmp.mergeable_candidates)}"
                    )
                parts.append("")

        parts.append(
            "Analyse the structural similarity of these Verilog projects' "
            "gate-level graphs. Rank pairs by similarity and identify "
            "shared sub-graph patterns. Respond in the JSON format described."
        )
        return "\n".join(parts)

    # ── LLM transport ─────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str) -> str:
        """Send a chat completion request to the LLM."""
        try:
            import httpx
            return self._call_httpx(system, user)
        except ImportError:
            return self._call_urllib(system, user)

    def _call_httpx(self, system: str, user: str) -> str:
        import httpx

        url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            log.warning("LLM API call failed: %s", exc)
            return f"(LLM call failed: {exc})"

    def _call_urllib(self, system: str, user: str) -> str:
        import urllib.request

        url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }).encode()

        req = urllib.request.Request(
            url, data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            log.warning("LLM API call failed: %s", exc)
            return f"(LLM call failed: {exc})"

    # ── Response parsing ──────────────────────────────────────────────

    @staticmethod
    def _parse_result(raw: str) -> LLMMatchResult:
        """Best-effort JSON extraction from the LLM response."""
        result = LLMMatchResult(raw_response=raw)

        # Try to find a JSON block in the response
        json_str = raw.strip()

        # Strip markdown code fences if present
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            # Remove first and last fence lines
            start = 1
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            json_str = "\n".join(lines[start:end])

        # Try to find JSON object boundaries
        brace_start = json_str.find("{")
        brace_end = json_str.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_str[brace_start:brace_end + 1]

        try:
            data = json.loads(json_str)
            result.shared_components = data.get("shared_components", [])
            result.merged_components = data.get("merged_components", [])
            result.unique_components = data.get("unique_components", {})
            result.unified_interface = data.get("unified_interface", {})
            result.area_savings_pct = data.get(
                "area_savings_pct",
                data.get("estimated_area_savings_pct"),
            )
            result.data_flow = data.get("data_flow", "")
            result.explanation = data.get(
                "explanation",
                data.get("pha_recommendation", ""),
            )
        except (json.JSONDecodeError, AttributeError):
            log.debug("Could not parse LLM response as JSON; storing raw text.")
            result.explanation = raw

        return result
