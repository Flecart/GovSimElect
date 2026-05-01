# GovSimElect: Governance of the Commons Simulation with Elections

Please see the [original GovSim project](https://github.com/giorgiopiatti/GovSim) for details of the baseline setup for GovSim. Stay tuned for further details regarding GovSimElect or contact me directly @ ryan.art.faulkner@gmail.com.

![GovSim overview](imgs/govsim_pull_figure.png)

TODO: Update workflow above with election phases.

## Code Setup

To use the codes in this repo, first clone this repo:
    
    git clone https://github.com/rfaulkner/GovSimElect.git
    cd GovSimElect

Then, to install the full local-model stack, run:

```setup
bash ./setup.sh
```

For a lighter cloud-only setup that targets OpenAI or OpenRouter and skips the local LLM stack, use:

```setup
bash ./setup_cloud.sh
```

The cloud setup also avoids requiring `sentence_transformers`; if that package is absent, the simulation falls back to a lightweight hash-based embedding for memory retrieval.

## Build `aggregated.csv` from simulation runs

Use this pipeline when your run folders are under `simulation/results`.

1. Aggregate each experiment directory into summary text:

```bash
bash scripts/run_aggregate_special_all.sh "simulation/results/<your-pattern>" tmp.md
```

2. Parse the summary text into a single CSV:

```bash
python3 scripts/logs/parser.py tmp.md aggregated.csv
```

3. Plot and inspect results in:
- `playground.ipynb`

4. Run statistical significance tests (permutation tests) on aggregated outputs:

```bash
python3 scripts/logs/permutation.py
```

This script uses `aggregated.csv` and `aggregated-sto.csv` and writes `permutation_test_results.csv`.

If you already know the exact results folders you want, just replace `<your-pattern>` with a glob (for example: `simulation/results/gpt-4o-p*-2026-04-*`).

### Optional: run the full pipeline in one command

```bash
bash scripts/logs/build_aggregated_csv.sh "simulation/results/<your-pattern>" aggregated.csv
```

## Frontend

The repo also includes a React frontend for browsing simulation results under [`frontend/`](/home/angeloh/Desktop/work/GovSimElect/frontend).

Install the frontend dependencies with:

```bash
cd frontend
pnpm install
```

For local development, run:

```bash
cd frontend
pnpm dev
```

This starts both:
- the Vite frontend on `http://localhost:5173`
- the analysis backend on `http://localhost:8050`

To build the frontend bundle:

```bash
cd frontend
pnpm build
```

If you want to run the analysis backend separately without the Vite dev server:

```bash
uv run python3 -m simulation.analysis.app
```
