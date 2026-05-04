# GovSimElect + GT-HarmBench

This repository bundles **two groups of experiments** in a single tree (so the anonymization platform can ingest everything in one shot). Each group has its own setup procedure — pick the one you need.

| Group | What it studies | Where it lives | Setup |
|---|---|---|---|
| **1. GovSimElect** | Governance of the commons under elections (extends [GovSim](https://github.com/giorgiopiatti/GovSim) with leader selection, SVO, stochastic regeneration, contracting). | this repo's root | [Group 1 setup](#group-1-setup--govsimelect) |
| **2. GT-HarmBench** | Game-theoretic AI safety benchmark (Prisoner's Dilemma / Stag Hunt / Coordination, with contracting and moral-hazard variants). | `external/gt-harmbench/` (vendored subtree) | [Group 2 setup](#group-2-setup--gt-harmbench) |

The two groups are independent — you do not need to set up both to run either one.

![GovSim overview](imgs/govsim_pull_figure.png)

---

## Group 1 setup — GovSimElect

Clone and enter the repo:

```bash
git clone <this-repo-url>
cd GovSimElect
```

Full local-model stack:

```bash
bash ./setup.sh
```

Lighter cloud-only setup (OpenAI / OpenRouter, no local LLM stack):

```bash
bash ./setup_cloud.sh
```

The cloud setup also avoids requiring `sentence_transformers`; if that package is absent, the simulation falls back to a lightweight hash-based embedding for memory retrieval.

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
