"""Configuration for RTL analysis pipeline.

Settings can be loaded from a YAML file (``config.yaml`` at repo root) or
overridden via CLI flags.  The YAML file makes it easy to tweak parameters
between runs without long command lines.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os


@dataclass
class AnalysisConfig:
    """All tuneable settings for the analysis pipeline.

    Every field can also be set from ``config.yaml`` using its name as key.
    CLI flags always take precedence over the YAML file.
    """

    # --- Paths ----------------------------------------------------------
    project_dir: Path = field(default_factory=lambda: Path("verilog_proj"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # --- iverilog -------------------------------------------------------
    iverilog_bin: str = "iverilog"
    vvp_bin: str = "vvp"
    iverilog_timeout: int = 120  # seconds per compilation

    # --- GitHub discovery -----------------------------------------------
    github_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("GITHUB_TOKEN")
    )
    github_min_stars: int = 50
    github_max_repos: int = 30
    clone_depth: int = 1

    # --- Similarity weights ---------------------------------------------
    gate_type_weight: float = 0.50   # portion of combined score
    structural_weight: float = 0.25  # portion of combined score
    partial_weight: float = 0.25     # n-gram Jaccard partial-match portion
    ngram_n: int = 3                 # gate-sequence length for n-gram extraction

    # --- Anomaly detection ----------------------------------------------
    anomaly_zscore_threshold: float = 3.0

    # --- PHA synthesis --------------------------------------------------
    pha_threshold: float = 0.80      # min pairwise similarity for clustering
    pha_merge_jaccard: float = 0.60  # min n-gram Jaccard to merge components
    max_diagram_cluster_size: int = 30  # cap cluster members for diagram-string phase

    # --- Surprise / interface -------------------------------------------
    min_surprise: float = 0.30
    min_compat: float = 0.40
    max_modules_per_project: int = 150  # cap large projects to limit O(M²)

    # --- Report settings ------------------------------------------------
    figure_dpi: int = 300
    figure_format: str = "png"
    font_family: str = "serif"
    font_size: int = 11
    colormap: str = "YlOrRd"

    # --- Diagram strings / LLM -----------------------------------------
    use_diagram_strings: bool = False
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = ""

    # --- Analysis flags -------------------------------------------------
    run_similarity: bool = True
    run_anomaly: bool = True
    run_surprise: bool = True
    run_interface: bool = True
    run_pha: bool = True
    run_edit_distance: bool = False

    # --- Cache ----------------------------------------------------------
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    use_cache: bool = True
    clear_cache: bool = False

    # --- Misc -----------------------------------------------------------
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    verbose: bool = False

    def __post_init__(self):
        self.project_dir = Path(self.project_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(self.cache_dir)

    # ── YAML loading ──────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> "AnalysisConfig":
        """Load config from a YAML file.  Missing keys use defaults."""
        import yaml  # optional dependency; gracefully absent

        data: Dict[str, Any] = {}
        if path.exists():
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def merge_cli(self, ns: Any) -> "AnalysisConfig":
        """Merge argparse Namespace onto self (CLI takes precedence).

        Only non-default CLI values overwrite the YAML-loaded config.
        """
        for key in self.__dataclass_fields__:
            cli_val = getattr(ns, key, None)
            if cli_val is not None:
                setattr(self, key, cli_val)
        self.__post_init__()
        return self
