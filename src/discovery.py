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

REGISTRY_FILE = ".discovery_registry.json"   # stored inside project_dir

# ── Curated seed list ─────────────────────────────────────────────────
CURATED_REPOS = [
    # CPU / RISC-V
    "cliffordwolf/picorv32",
    "stnolting/neorv32",
    "ZipCPU/zipcpu",
    "lowRISC/ibex",
    "ultraembedded/biriscv",
    "SI-RISCV/e200_opensource",
    "Wren6991/Hazard3",
    "olofk/serv",
    "ultraembedded/riscv",
    "darklife/darkriscv",
    "freechipsproject/rocket-chip",
    "chipsalliance/Cores-SweRV",
    "openhwgroup/cva6",
    "openhwgroup/cv32e40p",
    "syntacore/scr1",
    # ML / Neural Accelerator
    "nvdla/hw",
    "ucb-bar/gemmini",
    "ucb-bar/chipyard",
    # Signal / DSP / Image
    "alexforencich/verilog-dsp",
    "dawsonjon/fpu",
    # Arithmetic / FPU
    "openhwgroup/cvfpu",
    # Communication / IO
    "alexforencich/verilog-ethernet",
    "alexforencich/verilog-axi",
    "alexforencich/verilog-pcie",
    "corundum/corundum",
    "alexforencich/verilog-uart",
    "alexforencich/verilog-i2c",
    "ben-marshall/uart",
    # Memory / Cache
    "pulp-platform/axi",
    # FPGA Tool / Framework
    "YosysHQ/yosys",
    "enjoy-digital/litex",
    "lnis-uofu/OpenFPGA",
    # GPU / Parallel
    "vortexgpgpu/vortex",
    "miaow-gpu/miaow",
    # SoC / Platform
    "pulp-platform/pulpissimo",
    "XUANTIE-RV/openc910",
    "analogdevicesinc/hdl",
    "parallella/oh",
    "pConst/basic_verilog",
    # Crypto / Security
    "secworks/aes",
    "secworks/sha256",
    "secworks/sha512",
    "secworks/chacha",
    # NoC / Interconnect
    "bespoke-silicon-group/basejump_stl",
    # Misc / Educational
    "BrunoLevy/learn-fpga",
    "rsd-devel/RSD",
]

# ── Organisation sweeps ───────────────────────────────────────────────
# Repos from these orgs whose language is Verilog/SystemVerilog are added.
HARDWARE_ORGS = [
    "pulp-platform",
    "openhwgroup",
    "ucb-bar",
    "chipsalliance",
    "riscv",
    "lowRISC",
    "enjoy-digital",
    "alexforencich",
    "secworks",
    "olofk",
    "freechipsproject",
    "bespoke-silicon-group",
    "analogdevicesinc",
    "YosysHQ",
    "nvdla",
    "vortexgpgpu",
    "ZipCPU",
    "corundum",
    "lnis-uofu",
    "syntacore",
]

# ── Keyword / topic search queries ────────────────────────────────────
# Each entry: (label, query_string).  Star thresholds are intentionally
# low so new and niche projects are captured.  Language covers both
# Verilog and SystemVerilog.
GITHUB_SEARCH_QUERIES: List[Tuple[str, str]] = [
    # ── CPU / Processor ──────────────────────────────────────────────
    ("cpu",         "language:verilog topic:cpu stars:>5"),
    ("cpu-sv",      "language:systemverilog topic:cpu stars:>5"),
    ("riscv",       "language:verilog topic:risc-v stars:>5"),
    ("riscv-sv",    "language:systemverilog topic:risc-v stars:>5"),
    ("mips",        "language:verilog mips processor stars:>5"),
    ("arm",         "language:verilog arm processor stars:>5"),
    ("pipeline",    "language:verilog pipeline processor stars:>5"),
    # ── ML / Accelerator ─────────────────────────────────────────────
    ("nn-acc",      "language:verilog topic:neural-network stars:>3"),
    ("systolic",    "language:verilog systolic accelerator stars:>3"),
    ("tpu",         "language:verilog tpu accelerator stars:>3"),
    ("cnn-fpga",    "language:verilog cnn fpga stars:>3"),
    ("ml-sv",       "language:systemverilog neural accelerator stars:>3"),
    # ── DSP / Signal Processing ───────────────────────────────────────
    ("dsp",         "language:verilog topic:dsp stars:>5"),
    ("fft",         "language:verilog fft processor stars:>3"),
    ("fir",         "language:verilog fir filter stars:>3"),
    ("iir",         "language:verilog iir filter stars:>3"),
    ("cordic",      "language:verilog cordic stars:>3"),
    ("image",       "language:verilog image processing fpga stars:>3"),
    # ── Arithmetic / FPU ─────────────────────────────────────────────
    ("fpu",         "language:verilog fpu floating point stars:>5"),
    ("fpu-sv",      "language:systemverilog floating point unit stars:>5"),
    ("alu",         "language:verilog alu arithmetic stars:>3"),
    ("posit",       "language:verilog posit arithmetic stars:>1"),
    ("divider",     "language:verilog integer divider stars:>3"),
    ("multiplier",  "language:verilog multiplier booth stars:>3"),
    # ── Memory / Cache ────────────────────────────────────────────────
    ("cache",       "language:verilog cache controller stars:>5"),
    ("sdram",       "language:verilog sdram controller stars:>3"),
    ("ddr",         "language:verilog ddr memory controller stars:>5"),
    ("fifo",        "language:verilog fifo stars:>5"),
    ("sram",        "language:verilog sram stars:>3"),
    # ── Communication / IO ────────────────────────────────────────────
    ("axi",         "language:verilog topic:axi stars:>10"),
    ("ethernet",    "language:verilog ethernet stars:>10"),
    ("pcie",        "language:verilog pcie stars:>10"),
    ("uart",        "language:verilog uart stars:>5"),
    ("spi",         "language:verilog spi controller stars:>5"),
    ("i2c",         "language:verilog i2c controller stars:>5"),
    ("usb",         "language:verilog usb controller stars:>5"),
    ("can",         "language:verilog can bus controller stars:>3"),
    ("wishbone",    "language:verilog wishbone bus stars:>5"),
    ("apb",         "language:verilog apb bus stars:>3"),
    ("ahb",         "language:verilog ahb bus stars:>3"),
    # ── Crypto / Security ─────────────────────────────────────────────
    ("aes",         "language:verilog aes encryption stars:>5"),
    ("sha",         "language:verilog sha hash stars:>3"),
    ("rsa",         "language:verilog rsa cryptography stars:>3"),
    ("chacha",      "language:verilog chacha cipher stars:>1"),
    ("ecc",         "language:verilog elliptic curve stars:>3"),
    # ── GPU / Parallel ────────────────────────────────────────────────
    ("gpu",         "language:verilog gpu shader stars:>5"),
    ("gpu-sv",      "language:systemverilog gpu parallel stars:>5"),
    # ── NoC / Interconnect ────────────────────────────────────────────
    ("noc",         "language:verilog network on chip stars:>3"),
    ("crossbar",    "language:verilog crossbar switch stars:>3"),
    ("router",      "language:verilog router arbiter stars:>3"),
    # ── SoC / Platform ────────────────────────────────────────────────
    ("soc",         "language:verilog topic:soc stars:>10"),
    ("fpga-soc",    "language:verilog soc fpga stars:>5"),
    ("embedded",    "language:verilog embedded system stars:>5"),
    # ── FPGA / EDA Tools ─────────────────────────────────────────────
    ("fpga",        "language:verilog topic:fpga stars:>10"),
    ("fpga-sv",     "language:systemverilog topic:fpga stars:>10"),
    ("rtl",         "language:verilog rtl design stars:>5"),
    # ── Video / Display ───────────────────────────────────────────────
    ("hdmi",        "language:verilog hdmi transmitter stars:>3"),
    ("vga",         "language:verilog vga controller stars:>3"),
    # ── RF / Analog Interface ─────────────────────────────────────────
    ("adc",         "language:verilog adc interface stars:>3"),
    ("dac",         "language:verilog dac interface stars:>3"),
    # ── Catch-all broad sweeps ────────────────────────────────────────
    ("verilog-top", "language:verilog stars:>50"),
    ("sv-top",      "language:systemverilog stars:>50"),
]


class ProjectDiscovery:
    """Find RTL projects locally or on GitHub (additive across runs)."""

    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg
        self._registry_path = cfg.project_dir / REGISTRY_FILE

    # ── Registry ─────────────────────────────────────────────────────

    def _load_registry(self) -> Dict[str, dict]:
        if self._registry_path.exists():
            try:
                return json.loads(self._registry_path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("Could not read registry; starting fresh.")
        return {}

    def _save_registry(self, registry: Dict[str, dict]) -> None:
        self.cfg.project_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(
            json.dumps(registry, indent=2), encoding="utf-8"
        )

    def registry_stats(self) -> dict:
        reg = self._load_registry()
        cloned   = sum(1 for v in reg.values() if v.get("cloned"))
        return {"total_discovered": len(reg), "cloned": cloned,
                "pending": len(reg) - cloned}

    # ── Local scan ────────────────────────────────────────────────────

    def scan_local(self) -> List[Path]:
        projects: List[Path] = []
        if not self.cfg.project_dir.exists():
            log.warning("Project directory %s does not exist", self.cfg.project_dir)
            return projects
        for child in sorted(self.cfg.project_dir.iterdir()):
            if child.is_dir() and not child.name.startswith(".") \
                    and self._has_verilog(child):
                projects.append(child)
        log.info("Found %d local projects in %s", len(projects), self.cfg.project_dir)
        return projects

    # ── GitHub discovery ──────────────────────────────────────────────

    def discover_github(
        self,
        query: Optional[str] = None,
        clone: bool = True,
    ) -> List[str]:
        """Discover new repos from GitHub and clone any not yet present.

        Each run works cumulatively: previously cloned repos are skipped,
        previously discovered-but-not-cloned repos are cloned first, and
        new repos found this run are added to the registry for future runs.
        """
        registry = self._load_registry()
        stats_before = sum(1 for v in registry.values() if v.get("cloned"))

        # ── Collect candidates ────────────────────────────────────────
        candidates: List[Tuple[str, str]] = []   # (clone_url, source_label)

        if query:
            log.info("Running custom GitHub query: %s", query)
            for url in self._search_github(query, max_results=self.cfg.github_max_repos):
                candidates.append((url, "custom"))
        else:
            # Strategy 1: keyword/topic queries
            log.info("Running %d GitHub search queries …", len(GITHUB_SEARCH_QUERIES))
            per_q = max(10, self.cfg.github_max_repos // max(len(GITHUB_SEARCH_QUERIES), 1))
            for label, q in GITHUB_SEARCH_QUERIES:
                for url in self._search_github(q, max_results=per_q):
                    candidates.append((url, label))
                time.sleep(1.0)  # inter-query delay to avoid rate-limiting

            # Strategy 2: org sweeps
            log.info("Sweeping %d hardware organisations …", len(HARDWARE_ORGS))
            for org in HARDWARE_ORGS:
                for url in self._org_repos(org, max_results=30):
                    candidates.append((url, f"org:{org}"))

            # Strategy 3: curated seeds
            for slug in CURATED_REPOS:
                url = f"https://github.com/{slug}.git"
                candidates.append((url, "curated"))

        # ── Register new discoveries ──────────────────────────────────
        seen_urls: set = set(registry.keys())
        new_count = 0
        for url, source in candidates:
            if url not in seen_urls:
                seen_urls.add(url)
                name = url.rstrip("/").rsplit("/", 1)[-1].replace(".git", "")
                registry[url] = {
                    "name":          name,
                    "source":        source,
                    "cloned":        False,
                    "discovered_at": datetime.now(timezone.utc).isoformat(),
                }
                new_count += 1

        log.info(
            "Registry: %d total (+%d new this run), %d already cloned, %d pending.",
            len(registry), new_count,
            sum(1 for v in registry.values() if v.get("cloned")),
            sum(1 for v in registry.values() if not v.get("cloned")),
        )

        # ── Clone pending repos ───────────────────────────────────────
        if clone:
            pending = [
                (url, meta) for url, meta in registry.items()
                if not meta.get("cloned")
            ]
            # Curated first, then by discovery time (oldest first)
            pending.sort(key=lambda x: (x[1]["source"] != "curated",
                                        x[1]["discovered_at"]))
            to_clone = pending[: self.cfg.github_max_repos]

            log.info(
                "Cloning %d repo(s) this run (%d more queued for future runs) …",
                len(to_clone), max(0, len(pending) - len(to_clone)),
            )
            for url, _meta in to_clone:
                dest = self._clone_repo(url)
                if dest is not None:
                    registry[url]["cloned"] = True
                    registry[url]["cloned_at"] = datetime.now(timezone.utc).isoformat()

            stats_after = sum(1 for v in registry.values() if v.get("cloned"))
            log.info(
                "Done. Total cloned: %d (+%d this run). "
                "Still pending: %d (run --github again to clone more).",
                stats_after, stats_after - stats_before,
                sum(1 for v in registry.values() if not v.get("cloned")),
            )

        self._save_registry(registry)
        return [url for url, meta in registry.items() if meta.get("cloned")]

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _has_verilog(directory: Path) -> bool:
        for ext in ("*.v", "*.sv", "*.vh", "*.svh"):
            if list(directory.rglob(ext)):
                return True
        return False

    def _github_headers(self) -> dict:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.cfg.github_token:
            headers["Authorization"] = f"token {self.cfg.github_token}"
        return headers

    def _search_github(self, query: str, max_results: int = 30) -> List[str]:
        """GitHub Search API → clone URLs."""
        headers = self._github_headers()
        urls: List[str] = []

        for page in range(1, 5):   # up to 4 pages = 120 results per query
            if len(urls) >= max_results:
                break
            api_url = (
                "https://api.github.com/search/repositories"
                f"?q={urllib.request.quote(query)}"
                f"&sort=stars&order=desc&per_page=30&page={page}"
            )
            req = urllib.request.Request(api_url, headers=headers)
            last_page = False
            retry = 0
            while retry <= 3:
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        # Proactively slow down when close to the rate limit
                        remaining = int(resp.headers.get("X-RateLimit-Remaining", 10))
                        if remaining < 5:
                            reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
                            wait = max(0, reset_ts - int(time.time())) + 2
                            log.info("Rate limit nearly exhausted; sleeping %ds …", wait)
                            time.sleep(wait)
                        data = json.loads(resp.read().decode())
                    items = data.get("items", [])
                    for item in items:
                        urls.append(item["clone_url"])
                    if len(items) < 30:
                        last_page = True
                    break  # success – exit retry loop
                except urllib.error.HTTPError as exc:
                    if exc.code == 403:
                        reset_ts = int(exc.headers.get("X-RateLimit-Reset", 0))
                        wait = max(10, reset_ts - int(time.time())) + 2
                        log.warning(
                            "GitHub rate-limited (query='%s'); waiting %ds before retry %d/3 …",
                            query, wait, retry + 1,
                        )
                        time.sleep(wait)
                        retry += 1
                        continue
                    log.warning("GitHub API error %s for query '%s'.", exc.code, query)
                    last_page = True
                    break
                except Exception as exc:
                    log.warning("GitHub request failed: %s", exc)
                    last_page = True
                    break
            else:
                log.warning("Giving up on query='%s' after 3 rate-limit retries.", query)
                return urls
            if last_page:
                break
            time.sleep(0.5)

        return urls[:max_results]

    def _org_repos(self, org: str, max_results: int = 30) -> List[str]:
        """Return clone_urls of Verilog/SystemVerilog repos in a GitHub org."""
        headers = self._github_headers()
        urls: List[str] = []

        for page in range(1, 4):
            if len(urls) >= max_results:
                break
            api_url = (
                f"https://api.github.com/orgs/{org}/repos"
                f"?type=public&per_page=100&page={page}"
            )
            req = urllib.request.Request(api_url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    items = json.loads(resp.read().decode())
                if not isinstance(items, list):
                    break
                for item in items:
                    lang = (item.get("language") or "").lower()
                    if lang in ("verilog", "systemverilog"):
                        urls.append(item["clone_url"])
                if len(items) < 100:
                    break
            except urllib.error.HTTPError as exc:
                if exc.code == 403:
                    reset_ts = int(exc.headers.get("X-RateLimit-Reset", 0))
                    wait = max(10, reset_ts - int(time.time())) + 2
                    log.debug("Org '%s' rate-limited; waiting %ds …", org, wait)
                    time.sleep(wait)
                    continue
                if exc.code == 404:
                    log.debug("Org '%s' not accessible: HTTP %s", org, exc.code)
                    return urls
                break
            except Exception as exc:
                log.debug("Org '%s' sweep failed: %s", org, exc)
                break
            time.sleep(0.4)

        return urls[:max_results]

    def _clone_repo(self, url: str) -> Optional[Path]:
        """Shallow-clone *url* into project_dir if not already present."""
        name = url.rstrip("/").rsplit("/", 1)[-1].replace(".git", "")
        dest = self.cfg.project_dir / name
        if dest.exists():
            log.debug("Skipping %s (directory already exists)", name)
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
                cmd, capture_output=True, text=True, timeout=120, check=True,
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