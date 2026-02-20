# Polymorphic Hardware Accelerator — RTL Similarity Analyzer

Gate-level structural comparison of Verilog RTL projects using Icarus Verilog.

## Project layout

```
run.py                  # CLI entry point
src/
  config.py             # All tunable settings (AnalysisConfig)
  discovery.py          # Local scan + GitHub clone
  iverilog_analyzer.py  # VVP netlist parser → gate counts + connectivity graph
  similarity.py         # Gate-type cosine + structural similarity matrices
  anomaly.py            # Robust Z-score anomaly detection
  report.py             # Verbal (.txt), visual (PNG @ 300 DPI), CSV outputs
verilog_proj/           # RTL projects go here (one sub-directory per project)
output/                 # Generated reports and figures (git-ignored)
Dockerfile
docker-compose.yml
requirements.txt
```

## Quick start

### With Docker (recommended)

```bash
# 1. Build the image
docker compose build

# 2. Analyse projects in verilog_proj/
docker compose run rtl-analyzer

# 3. Discover + clone from GitHub, then analyse
docker compose run rtl-analyzer --github

# 4. Custom search query
docker compose run rtl-analyzer --github --github-query "topic:fpga language:verilog stars:>100"
```

Reports appear in `./output/` on the host.

### Without Docker

```bash
pip install -r requirements.txt
# iverilog must be in PATH (apt install iverilog / brew install icarus-verilog)

python run.py                          # local projects only
python run.py --github                 # discover from GitHub
python run.py --github --verbose       # with debug output
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--project-dir` | `verilog_proj` | Directory of RTL sub-projects |
| `--output-dir` | `output` | Where reports land |
| `--github` | off | Clone from GitHub before analysis |
| `--github-query` | auto | GitHub search query |
| `--github-token` | `$GITHUB_TOKEN` | Token for higher rate limit |
| `--github-max` | 30 | Max repos to clone |
| `--no-anomaly` | off | Skip anomaly detection |
| `--gate-weight` | 0.50 | Gate-type similarity weight |
| `--struct-weight` | 0.25 | Structural similarity weight |
| `--partial-weight` | 0.25 | Partial n-gram similarity weight |
| `--ngram-n` | 3 | Gate-sequence length for partial matching |
| `--zscore-threshold` | 2.0 | Anomaly Z-score threshold |
| `--dpi` | 300 | Figure DPI |
| `-v` / `--verbose` | off | Debug logging |

## Outputs

| File | Description |
|------|-------------|
| `output/report.txt` | Structured verbal report (sections, tables, stats) |
| `output/similarity_combined.csv` | Weighted similarity matrix (all 3 metrics) |
| `output/similarity_gate_type.csv` | Pure gate-type cosine similarity |
| `output/similarity_structural.csv` | Structural graph similarity |
| `output/similarity_ngram_partial.csv` | Partial gate-pattern n-gram Jaccard |
| `output/gate_counts.csv` | Per-project gate counts + graph metrics |
| `output/top_similar_pairs.csv` | Top-15 most similar pairs |
| `output/partial_gate_patterns.csv` | Shared gate sequences per project pair |
| `output/anomalies.csv` | Flagged outliers with Z-scores |
| `output/fig_heatmap_combined.png` | Combined similarity heatmap (300 DPI) |
| `output/fig_heatmap_gate.png` | Gate-type similarity heatmap |
| `output/fig_heatmap_structural.png` | Structural similarity heatmap |
| `output/fig_heatmap_ngram.png` | Partial n-gram similarity heatmap |
| `output/fig_gate_distribution.png` | Stacked gate-composition bar chart |
| `output/fig_partial_patterns.png` | Bubble chart of shared gate patterns |
| `output/fig_anomaly_scores.png` | Anomaly Z-score bar chart |

## How it works

1. **Compilation** — each project is compiled with `iverilog -g2012` into an intermediate VVP netlist  
2. **VVP parsing** — `.functor` directives are parsed to extract gate types (AND/OR/NOT/XOR/DFF/…) and their input wiring; this builds a real directed gate-level graph  
3. **Fallback** — projects that fail to compile are analysed by regex-scanning the raw RTL source  
4. **Similarity** — gate-type cosine similarity + structural graph metrics are combined (default 60 % / 40 %)  
5. **Anomaly detection** — robust Z-score (median ± MAD) flags outlier projects  
6. **Reporting** — verbal report, publication-quality figures (300 DPI), and CSV tables
