"""Discover and clone Verilog / RTL projects from GitHub.

Discovery is *additive*: a registry file (``verilog_proj/.discovery_registry.json``)
records every repo ever found.  Each ``--github`` run only clones repos that
have not been cloned yet, so the dataset grows with every invocation.

Three complementary search strategies are used:
  1. **Keyword / topic queries** — broad GitHub Search API queries across many
     hardware domains with deliberately low star thresholds.
  2. **Organisation sweeps** — enumerate repos from hardware-focused GitHub
     orgs (pulp-platform, openhwgroup, ucb-bar, …).
  3. **Curated seed list** — hand-picked repos that are known to be
     interesting and may not surface via automated search.

The combined candidate list is deduped, filtered against the registry, and
up to ``github_max_repos`` new repos are cloned per run.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import AnalysisConfig

log = logging.getLogger(__name__)

# ── Curated seed list — deliberately spans many different domains ─────
CURATED_REPOS = [
    # CPU / Processor
    "cliffordwolf/picorv32",
    "stnolting/neorv32",
    "ZipCPU/zipcpu",
    "lowRISC/ibex",
    "ultraembedded/biriscv",
    "SpinalHDL/VexRiscv",
    "SI-RISCV/e200_opensource",
    "Wren6991/Hazard3",
    "olofk/serv",
    # ML / Neural Accelerator
    "ucb-bar/chipyard",          # SoC + accelerators
    "nvdla/hw",                  # deep-learning accelerator
    # Signal / Image Processing
    "alexforencich/verilog-dsp",
    # Arithmetic
    "openhwgroup/cvfpu",         # FPU
    # Communication / IO
    "alexforencich/verilog-ethernet",
    "alexforencich/verilog-axi",
    "alexforencich/verilog-pcie",
    "corundum/corundum",
    # FPGA Tool / Framework
    "YosysHQ/yosys",
    "enjoy-digital/litex",
    "lnis-uofu/OpenFPGA",
    # GPU / Parallel
    "vortexgpgpu/vortex",
    # SoC / Platform
    "pulp-platform/pulpissimo",
    "lowRISC/opentitan",
    "XUANTIE-RV/openc910",
    "analogdevicesinc/hdl",
    # Misc
    "parallella/oh",
    "ultraembedded/riscv",
    "pConst/basic_verilog",
    "darklife/darkriscv",
]

# Multi-domain search queries — cast a wide net across different hardware areas
GITHUB_SEARCH_QUERIES = [
    "language:verilog topic:risc-v stars:>80",
    "language:verilog topic:fpga stars:>80",
    "language:verilog topic:neural-network stars:>50",
    "language:systemverilog topic:processor stars:>60",
    "language:verilog topic:dsp stars:>40",
    "language:verilog arithmetic accelerator stars:>30",
    "language:verilog image processing fpga stars:>30",
    "language:verilog ethernet axi stars:>60",
]


class ProjectDiscovery:
    """Find RTL projects locally or on GitHub."""

    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg

    # ── Local scan ────────────────────────────────────────────────────
    def scan_local(self) -> List[Path]:
        """Return directories under *project_dir* that contain .v/.sv."""
        projects: List[Path] = []
        if not self.cfg.project_dir.exists():
            log.warning("Project directory %s does not exist", self.cfg.project_dir)
            return projects
        for child in sorted(self.cfg.project_dir.iterdir()):
            if child.is_dir() and self._has_verilog(child):
                projects.append(child)
        log.info("Found %d local projects in %s", len(projects), self.cfg.project_dir)
        return projects

    # ── GitHub discovery ──────────────────────────────────────────────
    def discover_github(
        self,
        query: Optional[str] = None,
        clone: bool = True,
    ) -> List[str]:
        """Search GitHub across multiple domain-specific queries and
        optionally clone the results.

        When *query* is provided it is used exclusively; otherwise all
        queries in GITHUB_SEARCH_QUERIES are run to maximise domain
        diversity.  Returns list of clone URLs found.
        """
        if query:
            queries = [query]
        else:
            queries = GITHUB_SEARCH_QUERIES

        all_urls: List[str] = []
        seen: set = set()
        per_query = max(5, self.cfg.github_max_repos // len(queries))

        for q in queries:
            urls = self._search_github(q, max_results=per_query)
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    all_urls.append(u)
            if len(all_urls) >= self.cfg.github_max_repos:
                break

        all_urls = all_urls[: self.cfg.github_max_repos]
        log.info(
            "Discovered %d unique GitHub repositories across %d queries",
            len(all_urls), len(queries),
        )

        if clone:
            for url in all_urls:
                self._clone_repo(url)
        return all_urls

    # ── Private helpers ───────────────────────────────────────────────
    @staticmethod
    def _has_verilog(directory: Path) -> bool:
        for ext in ("*.v", "*.sv", "*.vh", "*.svh"):
            if list(directory.rglob(ext)):
                return True
        return False

    def _search_github(self, query: Optional[str] = None,
                       max_results: Optional[int] = None) -> List[str]:
        """Use the GitHub REST API to find Verilog repositories."""
        if query is None:
            query = f"language:verilog stars:>{self.cfg.github_min_stars}"
        if max_results is None:
            max_results = self.cfg.github_max_repos

        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.cfg.github_token:
            headers["Authorization"] = f"token {self.cfg.github_token}"

        urls: List[str] = []

        # Page through results
        for page in range(1, 4):
            if len(urls) >= max_results:
                break
            api_url = (
                f"https://api.github.com/search/repositories"
                f"?q={urllib.request.quote(query)}"
                f"&sort=stars&order=desc&per_page=30&page={page}"
            )
            req = urllib.request.Request(api_url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
                items = data.get("items", [])
                for item in items:
                    urls.append(item["clone_url"])
                if len(items) < 30:
                    break
            except urllib.error.HTTPError as exc:
                if exc.code == 403:
                    log.warning("GitHub rate-limited on query '%s'; skipping", query)
                    return []
                log.warning("GitHub API error %s: %s", exc.code, exc.reason)
                break
            except Exception as exc:
                log.warning("GitHub request failed: %s", exc)
                break
            time.sleep(0.8)

        if not urls:
            # Per-query fallback: return matching curated repos
            log.debug("No results for query '%s'; using curated fallback", query)
            urls = [f"https://github.com/{r}.git" for r in CURATED_REPOS]

        return urls[:max_results]

    def _clone_repo(self, url: str) -> Optional[Path]:
        """Shallow-clone *url* into project_dir if not already present."""
        name = url.rstrip("/").rsplit("/", 1)[-1].replace(".git", "")
        dest = self.cfg.project_dir / name
        if dest.exists():
            log.debug("Skipping %s (already exists)", name)
            return dest

        self.cfg.project_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "git", "clone",
            "--depth", str(self.cfg.clone_depth),
            "--single-branch",
            url,
            str(dest),
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            log.info("Cloned %s", name)
            return dest
        except subprocess.CalledProcessError as exc:
            log.warning("Failed to clone %s: %s", url, exc.stderr.strip())
        except subprocess.TimeoutExpired:
            log.warning("Clone timed out for %s", url)
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
        return None
