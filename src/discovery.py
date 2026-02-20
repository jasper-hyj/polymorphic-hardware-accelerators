"""Discover and clone Verilog / RTL projects from GitHub."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional

from .config import AnalysisConfig

log = logging.getLogger(__name__)

# ── Curated seed list of well-known open-source RTL projects ──────────
CURATED_REPOS = [
    "YosysHQ/yosys",
    "ZipCPU/zipcpu",
    "openhwgroup/cva6",
    "chipsalliance/rocket-chip",
    "lowRISC/ibex",
    "lowRISC/opentitan",
    "SpinalHDL/VexRiscv",
    "ultraembedded/biriscv",
    "steveicarus/iverilog",
    "olofk/serv",
    "enjoy-digital/litex",
    "freechipsproject/chisel3",
    "parallella/oh",
    "alexforencich/verilog-ethernet",
    "alexforencich/verilog-axi",
    "stnolting/neorv32",
    "cliffordwolf/picorv32",
    "darklife/darern",
    "ucb-bar/chipyard",
    "pulp-platform/pulpissimo",
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
        """Search GitHub for Verilog repos and optionally clone them.

        Returns list of clone URLs that were found / cloned.
        """
        repos = self._search_github(query)
        if clone:
            for url in repos:
                self._clone_repo(url)
        return repos

    # ── Private helpers ───────────────────────────────────────────────
    @staticmethod
    def _has_verilog(directory: Path) -> bool:
        for ext in ("*.v", "*.sv", "*.vh", "*.svh"):
            if list(directory.rglob(ext)):
                return True
        return False

    def _search_github(self, query: Optional[str] = None) -> List[str]:
        """Use the GitHub REST API to find Verilog repositories."""
        if query is None:
            query = f"language:verilog stars:>{self.cfg.github_min_stars}"

        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.cfg.github_token:
            headers["Authorization"] = f"token {self.cfg.github_token}"

        urls: List[str] = []

        # Page through results (max 3 pages → ~90 repos)
        for page in range(1, 4):
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
                    log.warning("GitHub rate-limited; using curated list instead")
                    return [
                        f"https://github.com/{r}.git" for r in CURATED_REPOS
                    ]
                log.warning("GitHub API error %s: %s", exc.code, exc.reason)
                break
            except Exception as exc:
                log.warning("GitHub request failed: %s", exc)
                break
            time.sleep(1.0)  # be polite

        if not urls:
            log.info("Falling back to curated repo list (%d repos)", len(CURATED_REPOS))
            urls = [f"https://github.com/{r}.git" for r in CURATED_REPOS]

        # Limit
        urls = urls[: self.cfg.github_max_repos]
        log.info("Discovered %d GitHub repositories", len(urls))
        return urls

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
