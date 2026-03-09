# Polymorphic Hardware Accelerator — RTL Similarity Analyzer

Gate-level structural comparison of Verilog / SystemVerilog RTL projects using Icarus Verilog, with cross-domain surprise analysis, module-level interface compatibility scoring, **Polymorphic Heterogeneous Architecture (PHA) synthesis** — automatically proposing how multiple Domain-Specific Accelerators can be merged into a single area-efficient polymorphic design — and **hardware edit distance** with a human-readable reusability scale showing exactly how much effort is needed to transform one project into another.

## Project layout

```
run.py                    # CLI entry point
config.yaml               # All tunable settings in one editable file
Makefile                  # Short aliases for common workflows
Dockerfile                # Production image (iverilog + Python + app code)
Dockerfile.dev            # Dev container image (tools only, source bind-mounted)
docker-compose.yml        # Per-analysis services + interactive dev shell
.devcontainer/
  devcontainer.json       # VS Code dev container settings
src/
  config.py               # Settings dataclass (loads config.yaml + CLI overrides)
  discovery.py            # Local scan + GitHub discovery & clone (additive registry)
  iverilog_analyzer.py    # iverilog → VVP netlist parser → gate counts + graph
  similarity.py           # Gate-type cosine + structural + n-gram similarity matrices
  anomaly.py              # Robust Z-score anomaly detection
  surprise.py             # Cross-domain surprise analysis + pattern semantics
  interface_analyzer.py   # Port-list parser → cross-project compatibility scoring
  pha_analyzer.py         # PHA synthesis: cluster DSAs → shared/merged/unique components
  pha_components.py       # PHA component data model
  edit_distance.py        # Multi-level hardware edit distance (gate Levenshtein + Hungarian module matching + wiring topology)
  diagram_serializer.py   # Architecture → structured text + string-based comparison
  string_algorithms.py    # LCS, alignment, substring algorithms for diagram strings
  llm_analyzer.py         # Optional LLM-assisted PHA analysis
  cache.py                # SHA-256-keyed file cache for gate extraction & similarity matrices
  report.py               # Orchestrates verbal (.txt), visual (PNG), CSV outputs
  report_verbal.py        # Structured text report + executive summary
  report_figures.py       # matplotlib/seaborn figure generation
verilog_proj/             # RTL projects go here (one sub-directory per project)
  .discovery_registry.json  # Additive registry of all discovered GitHub repos
output/                   # Generated reports and figures (git-ignored)
  run_YYYY-MM-DD_HH-MM-SS/  # Timestamped subfolder per run
    executive_summary.txt    # One-page overview of all findings
    report.txt               # Full detailed report
    edit_distance_report.txt # Reusability scale, ranked pairs, transformation plans (if --edit-distance)
    *.csv                    # All data tables
    fig_*.png                # Publication-quality figures
Dockerfile
docker-compose.yml
requirements.txt
```

## Quick start

### With dev container (recommended for development)

Open the repo in VS Code and select **Reopen in Container** (requires the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension). This gives you Python 3.12, `iverilog`, and all pip deps pre-installed — zero local setup.

Inside the dev container, use `make` shortcuts:

```bash
make help              # show all available commands
make local-demo        # full analysis on demo_projects
make local-all         # every analysis on verilog_proj (including edit distance)
make local-ed          # edit distance only (first 20 projects)
make quick             # similarity + anomaly on demo_projects (fast)
make clean             # remove cache and output
```

### With Docker Compose (headless / CI)

```bash
make build             # build the production image (or: docker compose build)
make run               # full pipeline on verilog_proj
make run-demo          # full pipeline on demo_projects
make similarity        # similarity only
make anomaly           # anomaly only
make edit-distance     # edit distance only (first 20 projects)
```

Or use `docker compose run` directly:

```bash
docker compose run --rm rtl-analyzer                                    # full pipeline
docker compose run --rm rtl-analyzer --github                           # discover + clone from GitHub
docker compose run --rm rtl-analyzer --similarity --anomaly             # selected analyses
docker compose run --rm rtl-analyzer --edit-distance --ed-max-projects 10
docker compose run --rm edit-distance                                   # dedicated edit-distance service
docker compose run --rm dev                                             # interactive bash shell with tools
```

### With Make (all common commands)

```bash
make help              # list all targets with descriptions

# Docker targets
make build             # build Docker image
make run               # full analysis (Docker)
make run-demo          # demo_projects analysis (Docker)
make similarity        # similarity only (Docker)
make anomaly           # anomaly detection only (Docker)
make surprise          # surprise analysis only (Docker)
make interface         # interface compatibility only (Docker)
make pha               # PHA synthesis only (Docker)
make edit-distance     # edit distance only (Docker)

# Local / dev-container targets
make local             # full analysis, local Python
make local-demo        # demo_projects, local Python
make local-all         # all analyses including edit distance
make local-ed          # edit distance only, local Python
make quick             # similarity + anomaly on demo_projects (fast)
make full              # every analysis on verilog_proj

# Utility
make clean             # remove cache + output
make clean-cache       # remove .cache/ only
make clean-output      # remove output/run_* only
```

### Configuration

Edit `config.yaml` to adjust analysis parameters without long CLI commands:

```yaml
# Tune similarity weights (auto-normalised to sum to 1.0)
gate_type_weight: 0.50
structural_weight: 0.25
partial_weight: 0.25

# PHA synthesis thresholds
pha_threshold: 0.35
pha_merge_jaccard: 0.60

# Analysis toggles (true = include)
run_similarity: true
run_anomaly: true
run_surprise: true
run_interface: true
run_pha: true
run_edit_distance: false
```

The YAML file is mounted read-only into Docker — just edit and re-run. CLI flags always override YAML values.

Set `GITHUB_TOKEN` in your environment for 30 search requests/min instead of 10 (the default unauthenticated limit). The token is forwarded automatically via `docker-compose.yml`.

Reports appear in `./output/run_<timestamp>/` on the host.

### Without Docker (manual setup)

```bash
pip install -r requirements.txt
# iverilog must be in PATH: apt install iverilog / brew install icarus-verilog

# Or just use make targets (same as inside dev container)
make local             # full analysis on verilog_proj
make local-demo        # full analysis on demo_projects
make local-ed          # edit distance only
make quick             # similarity + anomaly on demo_projects (fast)
make help              # show all targets
```

## CLI options

| Flag                 | Default         | Description                                                               |
| -------------------- | --------------- | ------------------------------------------------------------------------- |
| `--project-dir`      | `verilog_proj`  | Directory of RTL sub-projects                                             |
| `--output-dir`       | `output`        | Base directory; each run creates a timestamped subfolder                  |
| `--github`           | off             | Discover and clone Verilog projects from GitHub before analysis           |
| `--github-query`     | auto            | Custom GitHub search query (overrides the 62 built-in queries)            |
| `--github-token`     | `$GITHUB_TOKEN` | PAT for 30 req/min search limit (vs 10 unauthenticated)                   |
| `--github-max`       | `30`            | Max new repos to clone per run; extras queued for next run                |
| `--all`              | off             | Enable all analyses (similarity, anomaly, surprise, interface, PHA, edit distance) |
| `--similarity`       | off             | Include similarity analysis (gate-type, structural, n-gram)               |
| `--anomaly`          | off             | Include anomaly detection                                                 |
| `--surprise`         | off             | Include cross-domain surprise analysis                                    |
| `--min-surprise`     | `0.15`          | Minimum surprise score (similarity × domain-dissimilarity) to report      |
| `--interface`        | off             | Include interface / port-compatibility analysis                           |
| `--min-compat`       | `0.40`          | Minimum normalised port-match score to report a module pair               |
| `--pha`              | off             | Include PHA synthesis analysis                                            |
| `--edit-distance`    | off             | Include pairwise hardware edit distance analysis                          |
| `--ed-max-projects`  | `0` (all)       | Max projects for edit-distance analysis (0 = all). Limits O(n²) blowup   |
| `--pha-threshold`    | `0.35`          | Min pairwise similarity to cluster DSAs for PHA merging                   |
| `--pha-merge-jaccard`| `0.60`          | Min n-gram Jaccard to merge (fold) components in a PHA                    |
| `--diagram-strings`  | off             | Enable diagram-string serialisation for PHA clusters (LCS/alignment)      |
| `--llm-api-key`      | `$OPENAI_API_KEY`| OpenAI-compatible API key for LLM-assisted PHA analysis                  |
| `--llm-model`        | `gpt-4o-mini`  | LLM model name for PHA analysis                                           |
| `--llm-base-url`     | —               | Custom base URL for an OpenAI-compatible API endpoint                     |
| `--gate-weight`      | `0.50`          | Weight for gate-type cosine similarity                                    |
| `--struct-weight`    | `0.25`          | Weight for structural graph similarity                                    |
| `--partial-weight`   | `0.25`          | Weight for partial gate-pattern n-gram Jaccard                            |
| `--ngram-n`          | `3`             | Gate-sequence length for n-gram partial matching                          |
| `--zscore-threshold` | `2.0`           | Robust Z-score threshold for anomaly flagging                             |
| `--dpi`              | `300`           | Figure resolution in DPI                                                  |
| `--no-cache`         | off             | Disable caching — recompute everything from scratch                       |
| `--clear-cache`      | off             | Delete all cached artefacts before running                                |
| `-v` / `--verbose`   | off             | Enable DEBUG logging (per-project progress, n-gram counts, pair progress) |

## Caching

Gate extraction and similarity matrix computation are **automatically cached** between runs. The cache is keyed by a SHA-256 digest of source file contents, so:

- **Unchanged projects** are loaded instantly from `.cache/gates/` (no iverilog or regex re-parsing).
- **Unchanged project sets** with the same similarity weights reuse the 4 pairwise similarity DataFrames from `.cache/similarity/`.

On a typical 211-project run, the second invocation skips ~15 min of gate extraction and ~10 min of similarity computation.

```bash
# Normal run (cache enabled by default)
docker compose run rtl-analyzer

# Force fresh analysis
docker compose run rtl-analyzer --no-cache

# Wipe the cache before running
docker compose run rtl-analyzer --clear-cache
```

The cache is safe to delete at any time (`rm -rf .cache/`). It is git-ignored and mounted as a persistent Docker volume.

## Outputs

Each run writes to a timestamped subfolder, e.g. `output/run_2026-02-20_17-06-21/`.

### CSV tables

| File                           | Description                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `report.txt`                   | Full verbal report with sections, ranked tables, and explanations                                                                          |
| `gate_counts.csv`              | Per-project gate counts, graph metrics, and line count                                                                                     |
| `top_similar_pairs.csv`        | Top-15 most similar project pairs (combined score)                                                                                         |
| `similarity_combined.csv`      | Weighted combined similarity matrix                                                                                                        |
| `similarity_gate_type.csv`     | Gate-type cosine similarity matrix                                                                                                         |
| `similarity_structural.csv`    | Structural graph similarity matrix                                                                                                         |
| `similarity_ngram_partial.csv` | Partial gate-pattern n-gram Jaccard matrix                                                                                                 |
| `partial_gate_patterns.csv`    | Shared gate sequences for the top similar pairs                                                                                            |
| `anomalies.csv`                | Outlier projects with per-feature Z-scores *(only if `--anomaly`)*                                                                   |
| `surprise_findings.csv`        | Cross-domain surprising pairs with surprise score and shared patterns *(only if `--surprise`)*                                       |
| `domain_classification.csv`    | Domain label assigned to every project *(only if `--surprise`)*                                                                      |
| `interface_compatibility.csv`  | Compatible module pairs with rename/resize recipes and effort rating *(only if `--interface`)*                                       |
| `module_reusability.csv`       | Per-module reusability score based on protocol compliance, port shape prevalence, and interface simplicity *(only if `--interface`)* |
| `pha_clusters.csv`             | PHA cluster summary: members, domains, component counts, area savings *(only if `--pha`)*              |
| `pha_components.csv`           | Per-component breakdown (shared / merged / unique) for each PHA cluster *(only if `--pha`)*            |
| `pha_unified_interfaces.csv`   | Unified interface signals (common vs muxed) per cluster *(only if `--pha`)*                            |
| `edit_distance_report.txt`     | Executive summary, reusability scale, ranked pairs, distance matrix, per-pair transformation plan *(only if `--edit-distance`)* |

### Figures (PNG, configurable DPI)

| File                            | Description                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `fig_heatmap_combined.png`      | Combined similarity heatmap                                                   |
| `fig_heatmap_gate.png`          | Gate-type cosine similarity heatmap                                           |
| `fig_heatmap_structural.png`    | Structural graph similarity heatmap                                           |
| `fig_heatmap_ngram.png`         | Partial n-gram similarity heatmap                                             |
| `fig_gate_distribution.png`     | Stacked gate-composition bar chart per project                                |
| `fig_partial_patterns.png`      | Bubble chart of shared gate sequences                                         |
| `fig_anomaly_scores.png`        | Anomaly Z-score bar chart *(only if anomalies detected)*                      |
| `fig_surprise_map.png`          | Scatter plot of surprise score vs combined similarity *(only if pairs found)* |
| `fig_interface_reusability.png` | Module reusability score chart *(only if pairs found)*                        |
| `fig_pha_synthesis.png`         | PHA component decomposition + area savings + unified interface *(only if clusters found)* |

## Performance tips

The interface compatibility analysis is the most expensive step (O(P² × M²) where P = projects, M = modules/project). With 49 projects and 11 000+ modules the default run can take 10–30 minutes in a single-threaded baseline. Strategies to speed it up:

| Strategy | Command | Effect |
|---|---|---|
| Run only similarity | `--similarity` | Only gate-type / structural / n-gram comparison |
| Run only anomaly detection | `--anomaly` | Skips interface, surprise, PHA, edit distance |
| Raise the compatibility threshold | `--interface --min-compat 0.60` | The pre-filter prunes more pairs before any work; fewer results too |
| Limit projects analysed | Place only a subset in `verilog_proj/` | Quadratic win — halving projects cuts interface work by 4× |
| Run with more CPUs | Docker: increase host CPUs; bare-metal: automatic | Project-pair jobs are dispatched in parallel via `ProcessPoolExecutor` using all logical CPUs (`cfg.max_workers`) |

The code already applies a **port-count upper-bound pre-filter** before calling `_match_ports`: if `min(|ports_a|, |ports_b|) / max(|ports_a|, |ports_b|) < --min-compat`, the pair is dropped without any dictionary lookups. This eliminates the majority of pairs for free at the default threshold of 0.40.

## Docker services

The `docker-compose.yml` defines eight services with shared configuration (YAML anchors):

| Service | Command | Description |
|---|---|---|
| `rtl-analyzer` | `make run` / `docker compose run rtl-analyzer` | Full analysis pipeline (`--all`) |
| `similarity` | `make similarity` / `docker compose run similarity` | Similarity analysis only |
| `anomaly` | `make anomaly` / `docker compose run anomaly` | Anomaly detection only |
| `surprise` | `make surprise` / `docker compose run surprise` | Cross-domain surprise analysis only |
| `interface` | `make interface` / `docker compose run interface` | Interface compatibility only |
| `pha` | `make pha` / `docker compose run pha` | PHA synthesis only |
| `edit-distance` | `make edit-distance` / `docker compose run edit-distance` | Edit distance only (first 20 projects) |
| `dev` | `docker compose run dev` | Interactive bash shell with all tools |

All services mount `verilog_proj/`, `demo_projects/`, `output/`, `config.yaml`, and `.cache/`.

You can combine flags on any service: `docker compose run rtl-analyzer --similarity --anomaly --edit-distance`.

## Hardware edit distance

The `--edit-distance` flag enables a multi-level comparison showing how many operations are needed to transform one project's architecture into another:

| Level | Method | Operations |
|---|---|---|
| **Gate-level** | Levenshtein distance on per-module gate sequences | add gate, delete gate, substitute gate type (cost 1 each) |
| **Module-level** | Hungarian (optimal bipartite) matching of modules | add module, delete module, match/adapt module |
| **Wiring-level** | Symmetric difference of inter-module edges | add connection, remove connection |

### Reusability scale

Each pair receives a **0–100 reusability score** (normalised by the larger project's gate count) and a 5-star tier rating:

| Score | Rating | Meaning |
|---|---|---|
| ≥ 90 | ★★★★★ Drop-in Reusable | Almost identical — plug-and-play with minimal wiring tweaks |
| ≥ 70 | ★★★★☆ Easy Adaptation | Mostly shared modules — a handful of gate swaps |
| ≥ 40 | ★★★☆☆ Moderate Effort | Significant overlap but meaningful changes required |
| ≥ 15 | ★★☆☆☆ Major Rework | Limited overlap — substantial redesign needed |
| ≥ 0  | ★☆☆☆☆ Complete Redesign | Almost nothing shared — better to build from scratch |

### Report contents

The `edit_distance_report.txt` includes:

- **Executive summary** — scale legend, pairwise distance matrix, all pairs ranked by reusability, top-5 easiest / bottom-5 hardest pairs
- **Per-pair detail** — reusability score with visual bar, cost breakdown (gate edits / new modules / removed modules / rewiring), module reuse percentages, and a phased **transformation plan**:
  - Phase 1 — **Reuse** (keep identical modules)
  - Phase 2 — **Adapt** (modify similar modules with specific gate changes)
  - Phase 3 — **Remove** (delete unneeded modules)
  - Phase 4 — **Build** (create new modules)
  - Phase 5 — **Rewire** (update inter-module connections)

### Performance notes

- Projects with > 500 modules (e.g. `64pointFFTProcessor` with 51K modules) automatically use a **fast heuristic** (gate-type histogram + module count + wiring topology diff) instead of the O(n³) Hungarian algorithm.
- When `--edit-distance` is the only analysis flag passed, a **fast-path** skips similarity computation entirely and jumps straight to edit distance — turning a 20+ minute run into seconds.
- Use `--ed-max-projects N` to cap the number of projects (e.g. 20 → 190 pairs instead of 211 → 22,155 pairs).

## How it works

1. **Discovery** — `discovery.py` maintains an additive registry (`verilog_proj/.discovery_registry.json`) across runs. GitHub searches use 62 curated keyword/topic queries plus 21 hardware-org sweeps and a hand-picked seed list. Rate-limit responses (HTTP 403) are handled by reading `X-RateLimit-Reset` and retrying up to 3 times; the code also proactively sleeps when fewer than 5 requests remain.

2. **Compilation** — each project is compiled with `iverilog -g2012` into an intermediate VVP netlist. If the first attempt fails (missing top module), a retry without `-s` is made automatically.

3. **VVP parsing** — `.functor` directives are extracted to build a canonical gate-type count (AND / OR / NOT / XOR / NAND / NOR / XNOR / BUF / DFF / MUX / LATCH / TRISTATE) and a directed gate-level connectivity graph with real wiring edges.

4. **Regex fallback** — projects that fail to compile are analysed by scanning the raw RTL source for gate instantiations and `always @(posedge …)` blocks.

5. **Gate-type cosine similarity** (weight 50%) — each project is a 12-dimensional gate-count vector; cosine distance captures compositional similarity regardless of project size.

6. **Structural graph similarity** (weight 25%) — graph density, average clustering coefficient, degree-sequence entropy, and connected-component count are compared via normalised L1 distance.

7. **Partial gate-pattern matching** (weight 25%) — gate-type n-grams (default tri-grams, e.g. `AND→XOR→DFF`) are extracted via BFS on the connectivity graph. Weighted Jaccard overlap detects shared sub-circuit idioms even when overall sizes differ. Common patterns are mapped to named functional blocks (half-adder cell, LFSR tap, MAC cell, etc.).

8. **Anomaly detection** — robust Z-score (median ± 1.4826 × MAD) across all gate-count and graph features; projects above the configurable threshold are flagged with the driving feature named.

9. **Cross-domain surprise analysis** — every project is classified into a hardware domain (CPU, ML Accelerator, DSP, Arithmetic, Memory, IO, GPU, SoC, FPGA Tool). The surprise score is `combined_similarity × domain_dissimilarity`. High-scoring pairs are explained with their shared gate patterns + functional-block interpretations.

10. **Interface compatibility** — every `.v`/`.sv` file is parsed for module port lists (ANSI and old-style, with comment stripping). Each module pair across different projects is scored by name-normalised port match fraction (stripping `i_`, `o_`, `_n`, etc.). Pairs above `--min-compat` are ranked by effort: *zero* (drop-in swap), *low* (thin assign wrapper), *medium* (adapter module), or *high* (rework required). Standard bus protocols (AXI4-Full/Lite/Stream, AHB, APB, Wishbone, SPI, I2C, UART, PCIe) are auto-detected.

11. **PHA synthesis** — the combined similarity matrix is clustered (agglomerative, single-linkage, threshold `--pha-threshold` default 0.35) to find groups of ≥ 2 DSAs suitable for polymorphic merging.  Within each cluster:
    - **Shared components** — gate n-gram patterns present in *every* member are instantiated once, eliminating (N−1)× redundant copies.
    - **Merged components** — similar patterns (Jaccard ≥ `--pha-merge-jaccard`) from different members are folded into a single configurable block with a mode-select mux (overhead 5–20%).
    - **Unique components** — DSA-specific patterns are retained behind a configuration layer.
    - **Unified interface** — per-DSA interconnect port lists are merged into a single polymorphic port block: common signals pass through directly, subset-only signals are multiplexed.
    - **Area estimation** — shared blocks counted once + merged blocks + mux overhead + unique blocks yield the PHA gate total; the ratio vs. the sum of individual DSAs gives the area savings.

12. **Hardware edit distance** (optional, `--edit-distance`) — Verilog files are re-parsed into per-module gate sequences and instantiation topology. Gate-level Levenshtein distance measures per-module transformation cost; a Hungarian (optimal bipartite) assignment matches modules across projects; wiring edits count the symmetric difference of inter-module edges after mapping. Projects exceeding 500 modules use a fast histogram-based heuristic. Results are normalised into a 0–100 reusability score with a 5-star tier rating.

13. **Reporting** — verbal report with ranked tables and per-pair explanations, publication-quality figures, full CSV exports for all analyses, and (when `--edit-distance` is active) a dedicated `edit_distance_report.txt` with executive summary, ranked pair table, and per-pair transformation plans.
