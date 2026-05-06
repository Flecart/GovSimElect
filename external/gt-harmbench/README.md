# GT-HarmBench: Contracting Evaluation

Code and datasets for the TableGames experiments in *"Mechanism Design Is Not Enough: Prosocial Agents for Cooperative AI"*. Our paper formalizes a *cooperation gap* under incomplete contracts: when the specification language cannot distinguish all welfare-relevant states, no mechanism eliminates the welfare loss from self-interested play. Prosocial agents, who weigh others' welfare alongside their own, provably close this gap. This repo extends [GT-HarmBench](https://arxiv.org/abs/2602.12316) with a contracting evaluation pipeline covering Prisoner's Dilemma and Stag Hunt in 2x2 and 4x4 moral-hazard variants, three contract regimes, and three prompt modes, plus analysis tooling.

The companion GovSimContract code lives in a separate repo.

## Setup

```bash
cp .env.public .env  # add your API keys
uv sync              # https://docs.astral.sh/uv/getting-started/installation/
```

## Datasets

The 4x4 moral-hazard construction (Appendix E). Prepared datasets in `data/`:


| File                                     | Game                                   |
| ---------------------------------------- | -------------------------------------- |
| `gt-harmbench-pd30-2x2-with-targets.csv` | Prisoner's Dilemma, 2x2                |
| `gt-harmbench-pd30-4x4-rewritten.csv`    | Prisoner's Dilemma, 4x4 (moral hazard) |
| `gt-harmbench-sh30-2x2-with-targets.csv` | Stag Hunt, 2x2                         |
| `gt-harmbench-sh30-4x4-rewritten.csv`    | Stag Hunt, 4x4 (moral hazard)          |


Each file contains 30 scenarios. Select via `GAME=pd` or `GAME=sh`.

### Regenerating datasets

```bash
# 4x4 payoff matrices from 2x2 base games
uv run python scripts/generation/variation_game_theoretic_4x4.py \
    data/gt-harmbench.csv data/gt-harmbench-pd30-4x4.csv --sample 30 --pd-only

# rewrite scenarios with moral-hazard framing
uv run python scripts/generation/rewrite_4x4_stories_prisoners_dilemma.py \
    data/gt-harmbench-pd30-4x4.csv \
    --source-2x2 data/gt-harmbench.csv \
    --output data/gt-harmbench-pd30-4x4-rewritten.csv
```

Replace `--pd-only` with `--sh-only` and use `rewrite_4x4_stories_stag_hunt.py` for Stag Hunt.

Verification:

```bash
uv run python scripts/verification/verify_4x4_theory.py    # check matrix properties
uv run python scripts/verification/check_api_errors.py     # did you run out of credits when you left this running?
```

## Experiments

The runner crosses three **contract modes** (Section 4.2) with three **prompt modes** (Section 4.3):

**Contract modes**

- `no_communication` — players act independently
- `code_nl` — players negotiate a natural-language agreement; not enforced
- `code_law` — agreement is compiled to Python and enforced post-decision

**Prompt modes**

- `base` — no framing
- `selfish` — λ = 0; maximize own payoff
- `cooperative` — λ → ∞; maximize total welfare

Single run:

```bash
GAME=pd MODEL=openai/gpt-5.1 LIMIT=30 TIMES=5 PROMPT_MODE=base \
    ./scripts/experiments/run_contracting_eval.sh
```

`PROMPT_MODE=all` runs all three sequentially.


| Variable      | Default          | Description                             |
| ------------- | ---------------- | --------------------------------------- |
| `MODEL`       | `openai/gpt-5.1` | OpenAI or OpenRouter model string       |
| `GAME`        | `pd`             | `pd`, `sh`, or `co`                     |
| `LIMIT`       | full dataset     | Number of scenarios                     |
| `TIMES`       | `5`              | Iterations per scenario                 |
| `PROMPT_MODE` | `base`           | `base`, `selfish`, `cooperative`, `all` |
| `DATASET_4X4` | auto             | Override 4x4 dataset path               |
| `DATASET_2X2` | auto             | Override 2x2 dataset path               |


To replicate the paper's full sweep across models and games:

```bash
./scripts/experiments/run_contracting_model_games.sh
```

## Monitoring effect

Isolates the impact of enforcement *awareness* by holding the contract fixed and varying only whether agents are told it will be enforced (Appendix K.2).

```bash
uv run python scripts/generation/create_welfare_optimal_dataset.py --scenarios-per-game 15
./scripts/experiments/run_monitoring_effect_grok.sh
uv run python scripts/analysis/analyze_monitoring_effect.py logs/monitoring-effect-grok-<TIMESTAMP>
```

## Analysis

Automated extraction of formation/activation rates, welfare metrics, cooperation, effort levels, and token usage:

```bash
./scripts/experiments/analyze_contracting_results.sh                # most recent run
./scripts/experiments/analyze_contracting_results.sh logs/eval-...  # specific run
./scripts/experiments/analyze_all_contracting_results.sh            # batch process all eval directories
```

Outputs CSVs and plots under `logs/<run>/analysis/`.

Notebook used for stats and figures: `eval/analysis/multi_model_analysis.ipynb`.

QRE λ estimation (Appendix H) — pooled MLE with game-clustered bootstrap CIs:

```bash
uv run python scripts/analysis/estimate_lambda_pooled.py
```

## Log layout

```
logs/eval-{TIMESTAMP}-{MODEL}-{GAME}/
├── base/                      # prompt mode
│   ├── 4x4-no-comm/           # contract mode × variant
│   ├── 4x4-code-nl/
│   ├── 4x4-code-law/
│   ├── 2x2-no-comm/
│   ├── 2x2-code-nl/
│   └── 2x2-code-law/
├── selfish/
├── cooperative/
└── analysis/                  # generated by analysis script
```

## Trace viewer

Streamlit UI for browsing negotiation traces and analysis outputs:

```bash
uv run streamlit run traces_viewer/app.py
```

For programmatic trace inspection see `scripts/analysis/trace_extract.py`. We also provide an LLM skill documenting this script at `.cursor/skills/gt-harmbench-trace-analysis/`.