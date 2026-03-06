"""String-based pattern recognition algorithms for diagram comparison.

This module contains the core string algorithms used to compare
gate-sequence diagram strings:

  - Longest Common Subsequence (LCS)
  - Longest Common Substring
  - All Common Substrings (suffix-array-inspired)
  - Needleman-Wunsch global sequence alignment
  - Uncovered-segment extraction
  - Mergeable-candidate detection

These are pure functions with no side-effects, operating on lists of
gate-type tokens (e.g. ``["AND", "XOR", "DFF"]``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .surprise import interpret_pattern

# ── Constants (shared with diagram_serializer) ────────────────────────

GATE_ABBREV: Dict[str, str] = {
    "AND": "A", "OR": "O", "NOT": "N", "XOR": "X",
    "NAND": "a", "NOR": "o", "XNOR": "x", "BUF": "B",
    "DFF": "D", "MUX": "M", "LATCH": "L", "TRISTATE": "T",
}
ABBREV_GATE: Dict[str, str] = {v: k for k, v in GATE_ABBREV.items()}

TOKEN_SEP = "\u2192"          # → separator used in diagram strings
BLOCK_OPEN = "["
BLOCK_CLOSE = "]"


# ═══════════════════════════════════════════════════════════════════════
#  Data structures (lightweight, no external deps beyond dataclasses)
# ═══════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field


@dataclass
class SharedSubstring:
    """One contiguous shared gate-sequence between two diagrams."""
    pattern: str               # e.g. "D→A→X"
    compact: str               # e.g. "DAX"
    length: int                # number of tokens
    pos_a: int                 # start index in diagram A
    pos_b: int                 # start index in diagram B
    semantic_label: Optional[str] = None


@dataclass
class AlignmentResult:
    """Result of Needleman-Wunsch global sequence alignment."""
    aligned_a: str             # aligned sequence with gaps  "D-AX-D"
    aligned_b: str             # aligned sequence with gaps  "DMA-XD"
    match_line: str            # "=×==×=" annotation
    score: float               # raw alignment score
    normalised_score: float    # score / max_possible ∈ [0,1]
    matches: int
    mismatches: int
    gaps: int
    identity_pct: float        # matches / alignment_length × 100


# ═══════════════════════════════════════════════════════════════════════
#  Algorithms
# ═══════════════════════════════════════════════════════════════════════

def lcs(a: List[str], b: List[str]) -> List[str]:
    """Longest Common Subsequence (allows gaps).

    Returns the actual subsequence tokens (not just the length).
    O(n × m) dynamic programming.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []

    # Cap for memory safety
    if n * m > 25_000_000:
        a = a[:5000]
        b = b[:5000]
        n, m = len(a), len(b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to recover the subsequence
    result: List[str] = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return list(reversed(result))


def longest_common_substring(
    a: List[str], b: List[str],
) -> SharedSubstring:
    """Find the longest contiguous shared sub-sequence.

    O(n × m) DP with rolling-row optimisation.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return SharedSubstring("", "", 0, 0, 0)

    if n * m > 25_000_000:
        a = a[:5000]
        b = b[:5000]
        n, m = len(a), len(b)

    best_len = 0
    best_i = 0
    best_j = 0

    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_i = i - best_len
                    best_j = j - best_len
        prev = curr

    sub_tokens = a[best_i:best_i + best_len]
    pattern = TOKEN_SEP.join(sub_tokens)
    compact = "".join(GATE_ABBREV.get(t, "?") for t in sub_tokens)
    label = interpret_pattern(tuple(sub_tokens)) if sub_tokens else None

    return SharedSubstring(
        pattern=pattern,
        compact=compact,
        length=best_len,
        pos_a=best_i,
        pos_b=best_j,
        semantic_label=label,
    )


def all_common_substrings(
    a: List[str],
    b: List[str],
    min_len: int = 2,
    max_results: int = 50,
) -> List[SharedSubstring]:
    """Extract all contiguous shared sub-sequences of length >= *min_len*.

    Uses a suffix-array-inspired approach: build a hash set of all
    substrings of *b*, then scan *a* for matches.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []

    a = a[:3000]
    b = b[:3000]
    n, m = len(a), len(b)

    max_sub_len = min(8, m)
    b_subs: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for j in range(m):
        for L in range(min_len, min(max_sub_len + 1, m - j + 1)):
            key = tuple(b[j:j + L])
            b_subs[key].append(j)

    results: List[SharedSubstring] = []
    seen_patterns: Set[Tuple[str, ...]] = set()

    for i in range(n):
        for L in range(max_sub_len, min_len - 1, -1):
            if i + L > n:
                continue
            key = tuple(a[i:i + L])
            if key in seen_patterns:
                continue
            if key in b_subs:
                seen_patterns.add(key)
                j = b_subs[key][0]
                pattern = TOKEN_SEP.join(key)
                compact = "".join(GATE_ABBREV.get(t, "?") for t in key)
                label = interpret_pattern(key)
                results.append(SharedSubstring(
                    pattern=pattern,
                    compact=compact,
                    length=L,
                    pos_a=i,
                    pos_b=j,
                    semantic_label=label,
                ))
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break

    results.sort(key=lambda s: (-s.length, s.pos_a))
    return results


def needleman_wunsch(
    a: List[str],
    b: List[str],
    match: float = 2.0,
    mismatch: float = -1.0,
    gap: float = -0.5,
) -> AlignmentResult:
    """Needleman-Wunsch global sequence alignment.

    Returns aligned sequences with gap markers, match annotation,
    and summary statistics.
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return AlignmentResult("", "", "", 0, 0, 0, 0, 0, 0.0)

    if n * m > 10_000_000:
        a = a[:3000]
        b = b[:3000]
        n, m = len(a), len(b)

    dp = np.zeros((n + 1, m + 1), dtype=float)
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score_val = match if a[i - 1] == b[j - 1] else mismatch
            dp[i, j] = max(
                dp[i - 1, j - 1] + score_val,
                dp[i - 1, j] + gap,
                dp[i, j - 1] + gap,
            )

    # Backtrack
    aligned_a: List[str] = []
    aligned_b: List[str] = []
    match_line: List[str] = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score_val = match if a[i - 1] == b[j - 1] else mismatch
            if dp[i, j] == dp[i - 1, j - 1] + score_val:
                aligned_a.append(a[i - 1])
                aligned_b.append(b[j - 1])
                match_line.append("=" if a[i - 1] == b[j - 1] else "\u00d7")
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + gap:
            aligned_a.append(a[i - 1])
            aligned_b.append("\u2013")
            match_line.append("\u2013")
            i -= 1
        else:
            aligned_a.append("\u2013")
            aligned_b.append(b[j - 1])
            match_line.append("\u2013")
            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()
    match_line.reverse()

    n_match = match_line.count("=")
    n_mismatch = match_line.count("\u00d7")
    n_gap = match_line.count("\u2013")
    aln_len = len(match_line) or 1
    max_possible = max(n, m) * match if max(n, m) > 0 else 1
    raw_score = float(dp[n, m]) if n > 0 or m > 0 else 0

    return AlignmentResult(
        aligned_a=TOKEN_SEP.join(aligned_a),
        aligned_b=TOKEN_SEP.join(aligned_b),
        match_line="".join(match_line),
        score=raw_score,
        normalised_score=max(0.0, round(raw_score / max(max_possible, 1), 4)),
        matches=n_match,
        mismatches=n_mismatch,
        gaps=n_gap,
        identity_pct=round(100 * n_match / aln_len, 2),
    )


def extract_uncovered(tokens: List[str], covered: Set[int]) -> List[str]:
    """Extract contiguous runs of tokens not in the *covered* index set."""
    runs: List[str] = []
    current: List[str] = []
    for i, tok in enumerate(tokens):
        if i not in covered:
            current.append(tok)
        else:
            if current:
                runs.append(TOKEN_SEP.join(current))
                current = []
    if current:
        runs.append(TOKEN_SEP.join(current))
    return runs


def find_mergeable(
    alignment: AlignmentResult,
    a_tokens: List[str],
    b_tokens: List[str],
) -> List[Tuple[str, str, float]]:
    """Identify substitution regions in the alignment as merge candidates.

    A mismatch flanked by matches on both sides suggests a configurable
    block (mode-select mux between the two variants).
    """
    ml = alignment.match_line
    if not ml:
        return []

    candidates: List[Tuple[str, str, float]] = []
    a_parts = alignment.aligned_a.split(TOKEN_SEP)
    b_parts = alignment.aligned_b.split(TOKEN_SEP)

    i = 0
    while i < len(ml):
        if ml[i] == "\u00d7":
            start = i
            while i < len(ml) and ml[i] == "\u00d7":
                i += 1
            end = i
            left_match = start > 0 and ml[start - 1] == "="
            right_match = end < len(ml) and ml[end] == "="
            if left_match or right_match:
                sub_a = TOKEN_SEP.join(a_parts[start:end])
                sub_b = TOKEN_SEP.join(b_parts[start:end])
                run_len = end - start
                sim = 1.0 - (run_len / max(len(ml), 1))
                candidates.append((sub_a, sub_b, round(sim, 3)))
        else:
            i += 1

    return candidates[:20]
