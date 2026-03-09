# ┌─────────────────────────────────────────────────────────────────────┐
# │  Polymorphic Hardware Accelerator — Makefile                        │
# │                                                                     │
# │  Short aliases for the most common workflows.                       │
# │  Works inside the dev container, Linux, or macOS.                   │
# │  On bare Windows use `docker compose run` directly or WSL.          │
# └─────────────────────────────────────────────────────────────────────┘

PYTHON  ?= python
PROJECT ?= verilog_proj
DEMO    ?= demo_projects
MAX_ED  ?= 20

# ── Docker shortcuts ─────────────────────────────────────────────────

.PHONY: build up down

build:                          ## Build the production Docker image
	docker compose build

up:                             ## Start rtl-analyzer (full pipeline, detached)
	docker compose up rtl-analyzer

down:                           ## Stop all running services
	docker compose down

# ── Full pipeline (Docker) ───────────────────────────────────────────

.PHONY: run run-demo

run:                            ## Full analysis on verilog_proj (Docker)
	docker compose run --rm rtl-analyzer

run-demo:                       ## Full analysis on demo_projects (Docker)
	docker compose run --rm rtl-analyzer --project-dir demo_projects

# ── Individual analyses (Docker) ─────────────────────────────────────

.PHONY: similarity anomaly surprise interface pha edit-distance

similarity:                     ## Similarity only (Docker)
	docker compose run --rm similarity

anomaly:                        ## Anomaly detection only (Docker)
	docker compose run --rm anomaly

surprise:                       ## Cross-domain surprise only (Docker)
	docker compose run --rm surprise

interface:                      ## Interface compatibility only (Docker)
	docker compose run --rm interface

pha:                            ## PHA synthesis only (Docker)
	docker compose run --rm pha

edit-distance:                  ## Edit distance only, first $(MAX_ED) projects (Docker)
	docker compose run --rm edit-distance

# ── Local / dev-container shortcuts ──────────────────────────────────
# These run directly with Python (no Docker layer).

.PHONY: local local-demo local-ed local-all

local:                          ## Full analysis, local Python
	$(PYTHON) run.py --project-dir $(PROJECT)

local-demo:                     ## Full analysis on demo_projects, local Python
	$(PYTHON) run.py --project-dir $(DEMO)

local-all:                      ## All analyses including edit distance, local Python
	$(PYTHON) run.py --project-dir $(PROJECT) --all

local-ed:                       ## Edit distance only, local Python
	$(PYTHON) run.py --project-dir $(PROJECT) --edit-distance --ed-max-projects $(MAX_ED)

local-ed-demo:                  ## Edit distance on demo_projects, local Python
	$(PYTHON) run.py --project-dir $(DEMO) --edit-distance

# ── Combination shortcuts ────────────────────────────────────────────

.PHONY: quick full

quick:                          ## Similarity + anomaly on demo_projects (local, fast)
	$(PYTHON) run.py --project-dir $(DEMO) --similarity --anomaly

full:                           ## Every analysis on verilog_proj (local)
	$(PYTHON) run.py --project-dir $(PROJECT) --all

# ── Utility ──────────────────────────────────────────────────────────

.PHONY: clean clean-cache clean-output help

clean: clean-cache clean-output ## Remove cache + output

clean-cache:                    ## Remove analysis cache
	@if [ -d .cache ]; then find .cache -delete 2>/dev/null || true; echo "Cache cleared."; else echo "Cache not found."; fi

clean-output:                   ## Remove all output runs
	rm -rf output/run_*

help:                           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
