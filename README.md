# Polymorphic Hardware Accelerator — RTL Similarity Analyzer

Gate-level structural comparison of Verilog / SystemVerilog RTL projects using Icarus Verilog, with cross-domain surprise analysis and module-level interface compatibility scoring.

## Project layout

```
run.py                    # CLI entry point
src/
  config.py               # All tunable settings (AnalysisConfig)
  discovery.py            # Local scan + GitHub discovery & clone (additive registry)
  iverilog_analyzer.py    # iverilog → VVP netlist parser → gate counts + graph
  similarity.py           # Gate-type cosine + structural + n-gram similarity matrices
  anomaly.py              # Robust Z-score anomaly detection
  surprise.py             # Cross-domain surprise analysis + pattern semantics
  interface_analyzer.py   # Port-list parser → cross-project compatibility scoring
  report.py               # Verbal (.txt), visual (PNG @ 300 DPI), CSV outputs
verilog_proj/             # RTL projects go here (one sub-directory per project)
  .discovery_registry.json  # Additive registry of all discovered GitHub repos
output/                   # Generated reports and figures (git-ignored)
  run_YYYY-MM-DD_HH-MM-SS/  # Timestamped subfolder per run
Dockerfile
docker-compose.yml
requirements.txt
```

## Quick start

### With Docker (recommended)

```bash
# 1. Build the image
docker compose build

# 2. Analyse projects already in verilog_proj/
docker compose run rtl-analyzer

# 3. Discover + clone from GitHub, then analyse
docker compose run rtl-analyzer --github

# 4. Custom search query
docker compose run rtl-analyzer --github --github-query "topic:fpga language:verilog stars:>100"

# 5. Verbose output (shows per-project progress, n-gram counts, pair progress)
docker compose run rtl-analyzer --verbose
```

Set `GITHUB_TOKEN` in your environment for 30 search requests/min instead of 10 (the default unauthenticated limit). The token is forwarded automatically via `docker-compose.yml`.

Reports appear in `./output/run_<timestamp>/` on the host.

### Without Docker

```bash
pip install -r requirements.txt
# iverilog must be in PATH: apt install iverilog / brew install icarus-verilog

python run.py                          # local projects only
python run.py --github                 # discover + clone from GitHub, then analyse
python run.py --github --verbose       # with full progress logging
python run.py --no-surprise --no-interface  # skip the heavier analyses
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
| `--no-anomaly`       | off             | Skip anomaly detection                                                    |
| `--no-surprise`      | off             | Skip cross-domain surprise analysis                                       |
| `--min-surprise`     | `0.15`          | Minimum surprise score (similarity × domain-dissimilarity) to report      |
| `--no-interface`     | off             | Skip interface / port-compatibility analysis                              |
| `--min-compat`       | `0.40`          | Minimum normalised port-match score to report a module pair               |
| `--gate-weight`      | `0.50`          | Weight for gate-type cosine similarity                                    |
| `--struct-weight`    | `0.25`          | Weight for structural graph similarity                                    |
| `--partial-weight`   | `0.25`          | Weight for partial gate-pattern n-gram Jaccard                            |
| `--ngram-n`          | `3`             | Gate-sequence length for n-gram partial matching                          |
| `--zscore-threshold` | `2.0`           | Robust Z-score threshold for anomaly flagging                             |
| `--dpi`              | `300`           | Figure resolution in DPI                                                  |
| `-v` / `--verbose`   | off             | Enable DEBUG logging (per-project progress, n-gram counts, pair progress) |

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
| `anomalies.csv`                | Outlier projects with per-feature Z-scores *(skipped if `--no-anomaly`)*                                                                   |
| `surprise_findings.csv`        | Cross-domain surprising pairs with surprise score and shared patterns *(skipped if `--no-surprise`)*                                       |
| `domain_classification.csv`    | Domain label assigned to every project *(skipped if `--no-surprise`)*                                                                      |
| `interface_compatibility.csv`  | Compatible module pairs with rename/resize recipes and effort rating *(skipped if `--no-interface`)*                                       |
| `module_reusability.csv`       | Per-module reusability score based on protocol compliance, port shape prevalence, and interface simplicity *(skipped if `--no-interface`)* |

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

## Performance tips

The interface compatibility analysis is the most expensive step (O(P² × M²) where P = projects, M = modules/project). With 49 projects and 11 000+ modules the default run can take 10–30 minutes in a single-threaded baseline. Strategies to speed it up:

| Strategy | Command | Effect |
|---|---|---|
| Skip interface analysis entirely | `--no-interface` | Removes the slow step completely; all other analyses still run |
| Raise the compatibility threshold | `--min-compat 0.60` | The pre-filter prunes more pairs before any work; fewer results too |
| Skip surprise analysis | `--no-surprise` | Saves n-gram bag recomputation for 49 projects |
| Limit projects analysed | Place only a subset in `verilog_proj/` | Quadratic win — halving projects cuts interface work by 4× |
| Run with more CPUs | Docker: increase host CPUs; bare-metal: automatic | Project-pair jobs are dispatched in parallel via `ProcessPoolExecutor` using all logical CPUs (`cfg.max_workers`) |

The code already applies a **port-count upper-bound pre-filter** before calling `_match_ports`: if `min(|ports_a|, |ports_b|) / max(|ports_a|, |ports_b|) < --min-compat`, the pair is dropped without any dictionary lookups. This eliminates the majority of pairs for free at the default threshold of 0.40.

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

11. **Reporting** — verbal report with ranked tables and per-pair explanations, publication-quality figures, and full CSV exports for all analyses.
