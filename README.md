# GovSim + GT-HarmBench

This repository bundles **two groups of experiments** in a single tree (so the anonymization platform can ingest everything in one shot). Each group has its own setup procedure — pick the one you need.

| Group | What it studies | Where it lives | Setup |
|---|---|---|---|
| **1. GovSim** | Governance of the commons with private persona memory, SVO, stochastic regeneration, and contracting. | this repo's root | [Group 1 setup](#group-1-setup--govsim) |
| **2. GT-HarmBench** | Game-theoretic AI safety benchmark (Prisoner's Dilemma / Stag Hunt / Coordination, with contracting and moral-hazard variants). | `external/gt-harmbench/` (vendored subtree) | [Group 2 setup](#group-2-setup--gt-harmbench) |

The two groups are independent — you do not need to set up both to run either one.

![GovSim overview](imgs/govsim_pull_figure.png)

---

## Group 1 setup — GovSim

Clone and enter the repo:

```bash
git clone <this-repo-url>
cd GovSim
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
  - Mem0 Platform memory uses `MEM0_API_KEY`

Example:

```bash
export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

### Agent runtime and memory

The simulation has four environment phases: harvest, post-harvest pool,
restaurant, and home. The election runner has been removed.

Persona memory is backed by Mem0 and isolated with a different run-scoped
`user_id` for every persona. The default uses local Mem0. Select Mem0 Platform
or the OpenAI Agents SDK with Hydra overrides:

```bash
python -m simulation.main \
  llm.runtime=openai_agents \
  llm.backend=OpenAI \
  experiment.agent.memory.mode=platform
```

`llm.runtime=openai` keeps the direct SDK path. `openai_agents` currently
requires the OpenAI backend.

### Paper CPR experiment

The repository includes one configured condition from Gupta et al., *The Role
of Social Learning and Collective Norm Formation in Fostering Cooperation in
LLM Multi-Agent Systems* (arXiv:2510.14401): harsh environment (`r=0.2`),
selfish initialization, and Full alignment. It uses the four Appendix A prompts
for effort, punishment, norm proposal, and voting. Run it with:

```bash
python -m simulation.main experiment=paper_cpr_harsh_selfish_full
```

This condition uses 10 villagers, 50 rounds, initial stock 100, carrying
capacity 300, productivity 0.1, daily consumption 2, initial wealth 0,
punishment penalty 10, punishment cost 0, and collapse threshold 5. Every
villager sees the latest effort and payoff of every villager, while Mem0 history
is isolated per villager. Vote ties are resolved by seeded random choice.

The regeneration implementation also supports capped doubling as its default;
the paper condition explicitly selects logistic regeneration. Override it with:

```bash
python -m simulation.main experiment=paper_cpr_harsh_selfish_full \
  experiment.environment.regeneration.mode=doubling
```

The paper condition routes `anthropic/claude-sonnet-4` to Amazon Bedrock first
when the root LLM backend is OpenRouter. If that provider-specific request
fails, the runner emits a warning once and uses standard OpenRouter routing for
the rest of the run. Supply credentials through the environment; API keys are
never stored in the experiment config:

```bash
OPENROUTER_API_KEY=... OPENAI_API_KEY="$OPENROUTER_API_KEY" \
python -m simulation.main debug=true \
  llm.backend=OpenRouter llm.path=anthropic/claude-sonnet-4 \
  experiment=paper_cpr_harsh_selfish_full
```

`OPENAI_API_KEY` is reused only by Mem0's OpenAI-compatible embedder, whose
base URL is configured for OpenRouter. Each run shares one local Mem0 client,
while every villager remains isolated by a unique Mem0 `user_id`.

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

For CPR norm-formation runs, the explorer automatically switches to the
Commons Field Observatory. It shows stock and wealth trajectories, the latest
effort/harvest/punishment ledger, the evolving group norm, and every full
prompt/response exchange. The transcript can be filtered by villager, round,
phase, or text. Because prompts contain private Mem0 history, the analysis
backend binds to `127.0.0.1` by default.

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
