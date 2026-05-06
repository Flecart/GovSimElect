# AgentElect + GT-HarmBench

This repository bundles **two groups of experiments** in a single tree (so the anonymization platform can ingest everything in one shot). Each group has its own setup procedure — pick the one you need.

| Group | What it studies | Where it lives | Setup |
|---|---|---|---|
| **1. AgentElect** | Governance of the commons under elections (extends [GovSim](https://github.com/giorgiopiatti/GovSim) with leader selection, SVO, stochastic regeneration, contracting). | this repo's root | [Group 1 setup](#group-1-setup--agentelect) |
| **2. GT-HarmBench** | Game-theoretic AI safety benchmark (Prisoner's Dilemma / Stag Hunt / Coordination, with contracting and moral-hazard variants). | `external/gt-harmbench/` (vendored subtree) | [Group 2 setup](#group-2-setup--gt-harmbench) |

The two groups are independent — you do not need to set up both to run either one.

![GovSim overview](imgs/govsim_pull_figure.png)

---

## Group 1 setup — AgentElect

Clone and enter the repo:

```bash
git clone <this-repo-url>
cd AgentElect
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration update (important)

- Use `requirements.txt` as the source of truth for dependencies.
- Some older scripts or notebooks that depended on removed files may no longer run without manual patching.
- For API-backed runs:
  - OpenAI models use `OPENAI_API_KEY`
  - OpenRouter models use `OPENROUTER_API_KEY`

Example:

```bash
export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

### Full reproducibility model sweeps (OpenAI + OpenRouter)

Use `scripts/standard/examples_models.sh` to run the 3 contract conditions
(`code_nl`, `code_law`, `no_contract`) across:

- fixed seeds (`0 1 2 3 4`)
- prosocial counts (`0..5`)
- one selected model

Examples:

```bash
# OpenAI example
MODEL="openai/gpt-4o" \
NAME="gpt-4o" \
GROUP_PREFIX="repro_openai_gpt4o" \
bash scripts/standard/examples_models.sh
```

```bash
# OpenRouter example (non-openai/ prefix routes to OpenRouter backend)
MODEL="anthropic/claude-sonnet-4.5" \
NAME="claude-sonnet-4.5" \
GROUP_PREFIX="repro_openrouter_claude45" \
bash scripts/standard/examples_models.sh
```

You can also switch script family:

```bash
# Use stochastic script family instead of standard
SCRIPT_FAMILY="stochastic" \
MODEL="x-ai/grok-4.1-fast" \
NAME="grok-4.1-fast" \
GROUP_PREFIX="repro_stochastic_grok41" \
bash scripts/standard/examples_models.sh
```

### Build `aggregated.csv` from simulation runs

Use this pipeline when your run folders are under `simulation/results`.

1. Aggregate each experiment directory into summary text:

```bash
bash scripts/run_aggregate_special_all.sh "simulation/results/<your-pattern>" tmp.md
```

2. Parse the summary text into a single CSV:

```bash
python3 scripts/logs/parser.py tmp.md aggregated.csv
```

3. Plot and inspect results in `playground.ipynb`.

4. Run statistical significance tests (permutation tests):

```bash
python3 scripts/logs/permutation.py
```

This script uses `aggregated.csv` and `aggregated-sto.csv` and writes `permutation_test_results.csv`.

If you already know the exact result folders you want, replace `<your-pattern>` with a glob (e.g. `simulation/results/gpt-4o-p*-2026-04-*`).

#### Optional: run the full pipeline in one command

```bash
bash scripts/logs/build_aggregated_csv.sh "simulation/results/<your-pattern>" aggregated.csv
```

### Frontend

A React frontend for browsing simulation results lives under [`frontend/`](frontend).

Install dependencies:

```bash
cd frontend
pnpm install
```

Local development (starts Vite on `http://localhost:5173` and the analysis backend on `http://localhost:8050`):

```bash
cd frontend
pnpm dev
```

Production build:

```bash
cd frontend
pnpm build
```

Run the analysis backend on its own (no Vite dev server):

```bash
uv run python3 -m simulation.analysis.app
```

---

## Group 2 setup — GT-HarmBench

The full benchmark, datasets, evaluation pipeline, and trace explorer live under [`external/gt-harmbench/`](external/gt-harmbench/). See its own [README](external/gt-harmbench/README.md) for the complete setup, dataset-generation, and evaluation instructions.

Quick start (from the repo root):

```bash
cd external/gt-harmbench
cp .env.public .env        # then add your API keys
uv sync                    # requires https://docs.astral.sh/uv/
```

Then follow the [GT-HarmBench README](external/gt-harmbench/README.md) for dataset generation, experiment execution, analysis, and the trace viewer.
