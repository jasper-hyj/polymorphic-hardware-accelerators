"""Configuration for RTL analysis pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class AnalysisConfig:
    """All tuneable settings for the analysis pipeline."""

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
    anomaly_zscore_threshold: float = 2.0

    # --- Report settings ------------------------------------------------
    figure_dpi: int = 300
    figure_format: str = "png"
    font_family: str = "serif"
    font_size: int = 11
    colormap: str = "YlOrRd"

    # --- Misc -----------------------------------------------------------
    max_workers: int = 4
    verbose: bool = False

    def __post_init__(self):
        self.project_dir = Path(self.project_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
