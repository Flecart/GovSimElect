# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GT-HarmBench is a game-theoretic AI safety benchmark that evaluates LLM behavior in strategic scenarios involving value tradeoffs. It uses the `inspect_ai` framework for evaluation.

### Motivation

Current AI safety benchmarks rely heavily on keyword matching rather than genuine behavioral assessment. GT-HarmBench addresses this limitation by:

1. **Testing strategic reasoning**: Evaluating whether models can identify and play Nash equilibria in game-theoretic scenarios
2. **Measuring welfare preferences**: Assessing if models maximize utilitarian welfare (sum of payoffs) or Rawlsian fairness (minimum payoff)
3. **Studying gamification effects**: Comparing model behavior on original scenarios vs. gamified contextualizations

### Dataset

- **Size**: 238 scenarios spanning 8 safety domains (extracted from 505 MIT AI Risk Database scenarios)
- **Domains**: AI systems, Autonomous vehicles, Bias & Fairness, Centralized power, Content moderation, Data privacy, Misinformation, Social media
- **Game types**: 7 core game structures - Prisoner's Dilemma, Stag Hunt, Chicken, Battle of Sexes, Coordination, Zero-Sum, and Individual Decision problems
- **Evaluation targets**: Three strategy profiles per scenario (Nash equilibrium, utilitarian optimum, Rawlsian optimum)

### Moral Hazard Variants

- **Purpose**: Study how unverifiable effort (hidden action) affects strategic decision-making
- **4x4 Moral Hazard**: Extends 2x2 games to 4x4 by splitting each action into high/low effort levels, making effort private and unverifiable
- **Neutral Labels**: Uses "High Effort" / "Low Effort" descriptors instead of morally loaded terms (e.g., "Genuine", "Reckless") to avoid biasing model decisions
- **Key elements**: Each story includes (1) hidden effort framing, (2) disruption state, (3) outcome observability without action observability
- **Scripts**:
  - `scripts/generation/variation_game_theoretic_4x4.py`: Generate 4x4 payoff matrices from 2x2 games
  - `scripts/generation/rewrite_4x4_stories_prisoners_dilemma.py`: Rewrite PD stories with moral hazard context
  - `scripts/generation/rewrite_4x4_stories_stag_hunt.py`: Rewrite Stag Hunt stories with moral hazard context
- **Evaluation**: Use `--prompt-mode` flag (base/selfish/cooperative) to induce different preference modes

### Welfare Metrics

| Metric | Definition | Formula |
|--------|------------|---------|
| **Utilitarian** | Sum of payoffs (total welfare) | `u_row + u_col` |
| **Rawlsian** | Minimum payoff (maximin fairness) | `min(u_row, u_col)` |
| **Nash Social Welfare** | Product of payoffs | `u_row × u_col` |

**Note**: Nash social welfare serves as a theoretical bridge between utilitarian and Rawlsian objectives—it increases with both total welfare and the share of the worst-off player. However, the primary evaluation metrics are `nash_accuracy`, `utilitarian_accuracy`, and `rawlsian_accuracy`.

### CSV Structure

Both `gt-harmbench.csv` and `gt-harmbench-gamify.csv` contain 2x2 normal-form games with columns for:
- Game type (Prisoner's Dilemma, Stag Hunt, Chicken, etc.)
- Narratives for row/column players
- Action lists and payoff matrices
- Target strategy profiles for each welfare criterion

**Key difference**: `gt-harmbench-gamify.csv` embeds payoffs implicitly through qualitative narrative descriptions rather than explicit numerical values, testing whether models can infer strategic incentives from realistic scenarios.

### Research Goals

1. **Benchmark construction**: Create a rigorous game-theoretic benchmark from existing AI safety scenarios
2. **Capability assessment**: Evaluate strategic reasoning capabilities across different model classes
3. **Gamification analysis**: Study how framing scenarios as games affects model behavior
4. **Welfare preferences**: Determine if models exhibit systematic biases toward different welfare criteria

## Commands

All Python commands use `uv run`:
```bash
# Contracting evaluation (primary experiment)
GAME=pd MODEL=openai/gpt-5.1 LIMIT=30 TIMES=5 PROMPT_MODE=base \
    ./scripts/experiments/run_contracting_eval.sh

# Run with all prompt modes (base, selfish, cooperative)
PROMPT_MODE=all LIMIT=30 TIMES=5 ./scripts/experiments/run_contracting_eval.sh

# Batch evaluation across models and games
./scripts/experiments/run_contracting_model_games.sh

# Run main evaluation (legacy)
uv run python3 -m eval.eval \
    --model-name openai/gpt-5.1 \
    --dataset data/gt-harmbench-with-targets.csv \
    --times 1 --temperature 1.0 --experiment-name my-eval

# Run with prompt mode (preference induction)
uv run python3 -m eval.eval \
    --model-name openai/gpt-5.1 \
    --dataset data/gt-harmbench-mh.csv \
    --prompt-mode cooperative \
    --experiment-name mh-coop

# Run due diligence evaluation (game classification + Nash detection)
uv run python3 -m eval.due_diligence \
    --model-name openai/gpt-5.1 \
    --dataset-path data/gt-harmbench.csv \
    --task both --reasoning-effort medium

# View results in browser
inspect view --host 0.0.0.0

# Generate analysis plots
PYTHONPATH=. uv run python -m eval.analysis.cli --plot accuracy --plot welfare --plot heatmap

# Extract metrics to CSV
uv run python3 -m eval.analysis.extract_metrics --directory logs --output metrics_summary.csv

# Domain distribution analysis (join GT-HarmBench with MIT AI Risk Database)
uv run python3 scripts/analysis/domain_distribution.py
```

Dataset generation (three-step workflow):
```bash
# Step 1: Classify scenarios as game-theoretic
uv run python3 scripts/generation/classify_csv_games.py

# Step 2: Generate contextualizations
uv run python3 scripts/generation/generate_contextualizations_from_filter.py

# Step 2b: Heuristic filter
uv run python3 scripts/generation/fix_contextualizations.py

# Step 3: Add targets (Nash, utilitarian, Rawlsian)
uv run python3 -m scripts.generation.add_targets data/contextualization-filtered.csv data/contextualization-with-targets.csv
```

4x4 moral hazard generation:
```bash
# Step 1: Generate 4x4 payoff matrices from 2x2 dataset
uv run python3 scripts/generation/variation_game_theoretic_4x4.py \
  data/gt-harmbench.csv data/gt-harmbench-pd30-4x4.csv --sample 30 --pd-only

# Step 2: Rewrite stories with moral hazard context
uv run python3 scripts/generation/rewrite_4x4_stories_prisoners_dilemma.py \
  data/gt-harmbench-pd30-4x4.csv \
  --source-2x2 data/gt-harmbench.csv \
  --output data/gt-harmbench-pd30-4x4-rewritten.csv

# For Stag Hunt, use rewrite_4x4_stories_stag_hunt.py instead
```

**Note**: Stag Hunt requires additional preprocessing (fix_stag_hunt_payoffs.py, rewrite_sh_base_stories.py) to ensure proper game-theoretic properties. Use compare_sh_datasets.py to verify dataset consistency.

Verification tools:
```bash
# Verify theoretical properties of 4x4 matrices
uv run python3 scripts/verification/verify_4x4_theory.py

# Check logs for API errors
uv run python3 scripts/verification/check_api_errors.py logs/eval-...
```

Contracting evaluation (single mode, direct Python):
```bash
# Run single mode (direct Python, bypassing shell script)
uv run python3 -m eval.contracting_eval \
  --dataset data/test-gt-harmbench-4x4-rewritten.csv \
  --contract-mode code_nl \
  --limit 5 --times 1

# Analyze results
./scripts/experiments/analyze_contracting_results.sh
```

Analysis and visualization:
```bash
# Monitoring effect analysis
uv run python3 scripts/analysis/analyze_monitoring_effect.py logs/monitoring-effect-grok-<TIMESTAMP>

# QRE lambda estimation (pooled, with bootstrap CIs)
uv run python3 scripts/analysis/estimate_lambda_pooled.py logs/eval-... --output lambda_estimates.csv

# Interactive analysis (multi-model comparison, welfare metrics)
# Open eval/analysis/multi_model_analysis.ipynb in Jupyter

# Streamlit trace explorer
uv run streamlit run traces_viewer/app.py
```

Trace analysis:
```bash
# Extract contracting traces as structured JSON
uv run python3 scripts/analysis/trace_extract.py list logs/eval-...
uv run python3 scripts/analysis/trace_extract.py show logs/eval-... --trace-id <trace_id>

# Or use the gt-harmbench-trace-analysis Claude Code skill (.cursor/skills/)
```

**Log directory structure**: `logs/eval-{TIMESTAMP}-{MODEL}/`
- Example: `logs/eval-20260419-143000-openai-gpt-4o/`
- Subdirectories by prompt mode: `base/`, `selfish/`, `cooperative/`
- Each prompt mode contains: `4x4-no-comm/`, `4x4-code-nl/`, `4x4-code-law/`, `2x2-no-comm/`, `2x2-code-nl/`, `2x2-code-law/`

Batch experiments:
```bash
./scripts/experiments/standard.sh    # Main evaluations
./scripts/experiments/gamify.sh      # Gamified dataset
./scripts/experiments/due_diligence.sh
```

## Architecture

### Dataset Generation (Three-Stage Process)

1. **Classification**: LLM classifier filters MIT AI Risk Database scenarios for game-theoretic potential (47% pass rate)
2. **Contextualization**: Generate payoff matrices and narrative contextualizations for each scenario
3. **Target Computation**: Calculate Nash equilibria, utilitarian optima, and Rawlsian optima using game-theoretic solvers

### Evaluation Pipeline (`eval/`)

- `eval.py`: Main evaluation with `TwoWaySolver` that queries the model twice per scenario (row player, column player) and `custom_all_scorer` that computes Nash/utilitarian/Rawlsian accuracy. Supports `--prompt-mode` flag (base/selfish/cooperative) for preference induction.
- `contracting_eval.py`: Contracting evaluation with three modes:
  - **no_communication**: Baseline with no contracting mechanism
  - **code_nl**: Natural language agreements shown during decision-making, with no formal enforcement
  - **code_law**: Natural language agreements translated to Python code and enforced after decisions
- `prompts.py`: Prompt templates for inducing different preference modes (subtle framing, not explicit commands)
- `due_diligence.py`: Capability checks - game type classification and Nash equilibrium detection
- `eval/analysis/`: Analysis CLI and utilities for processing logs and generating visualizations
- `contracting_solver.py`: Custom solver for handling tuple actions in 4x4 moral hazard scenarios
- `contracting_scorer.py`: Multi-strategy scoring for contracting with effort level tracking

### Evaluation Methodology

**Prompt format**: Each scenario includes:
- Game description with rules and actions
- Payoff matrices (numeric and narrative forms)
- Role assignment (row/column player or joint decision maker)
- Strategic context and consequences

**Two-way evaluation**: For two-player games, model is queried from both perspectives:
- Row player perspective → selects action
- Column player perspective → selects action
- Combined actions compared against all three target profiles

**Due diligence tasks**:
- Game classification: Identify which of 7 game types a scenario represents
- Nash detection: Determine which cell in payoff matrix is a Nash equilibrium

### Core Abstractions

**Solvers** (in eval files): Custom `inspect_ai` solvers that handle the two-player game evaluation pattern
- `TwoWaySolver`: Queries model for both row and column player perspectives

**Scorers**: Multi-strategy scoring that tracks:
- `nash_accuracy`: Match against computed Nash equilibria
- `utilitarian_accuracy`/`utilitarian_efficiency`: Sum of payoffs
- `rawlsian_accuracy`/`rawlsian_efficiency`: Min payoff (fairness)
- `nash_social_welfare`: Product of payoffs
- `helpfulness`: Percentage of non-refusal responses
- `refusal_rate`: Percentage of refusal responses

**Metrics** (`src/metrics.py`): Pure functions for social welfare calculations - `utilitarian_payoff`, `rawlsian_payoff`, `nash_social_payoff`

### Data Flow

1. Dataset CSVs contain game scenarios with payoff matrices and stories for row/column players
2. `record_to_sample_*` functions convert CSV rows to `inspect_ai.Sample` objects
3. Solver queries model for both player perspectives
4. Scorer compares responses against precomputed targets and payoff matrices
5. Logs saved to `logs/` directory as `.eval` files

### Key Data Files

**Primary datasets** (used by contracting evaluation):
- `data/gt-harmbench-pd30-4x4-rewritten.csv`: Prisoner's Dilemma 4x4 (moral hazard)
- `data/gt-harmbench-pd30-2x2-with-targets.csv`: Prisoner's Dilemma 2x2
- `data/gt-harmbench-sh30-4x4-rewritten.csv`: Stag Hunt 4x4 (moral hazard)
- `data/gt-harmbench-sh30-2x2-with-targets.csv`: Stag Hunt 2x2

**Other datasets**:
- `data/game_template.csv`: Definitions of 7 core game types
- `data/gt-harmbench-with-targets.csv`: Main evaluation dataset with computed targets
- `data/gt-harmbench-mh.csv`: Moral hazard variants (realistic format)
- `data/gt-harmbench-gamify-mh.csv`: Moral hazard variants (gamify format)
- `data/test-gt-harmbench-4x4-rewritten.csv`: Test dataset (5 scenarios) for quick evaluation
- `data/test-gt-harmbench-2x2.csv`: Corresponding 2x2 test scenarios
- `data/mit.csv`: Source MIT AI Risk Database scenarios

### Contracting Analysis Functions

`eval.analysis.contracting` provides shared analysis functions:

- `load_contracting_logs(log_base_dir, experiments, prompt_modes)` -- load logs into DataFrames
- `compute_contracting_metrics(df)` -- formation/activation rates, payoffs, effort distribution
- `aggregate_experiment_metrics(log_base_dir, prompt_modes)` -- summary metrics, one row per experiment
- `compute_interaction_effects(metrics_df, outcome_col)` -- superadditive interaction effects for complementarity test
- `detect_greenwashing(df, prompt_mode)` -- detect cooperate output + low effort in 4x4 scenarios
- `get_effort_distribution(df)` -- effort level distribution for 4x4 scenarios
- `normalize_welfare(score, bounds, metric)` -- normalize payoff to [0, 1] using per-scenario bounds
- `nash_deviation(score, bounds, welfare)` -- normalized deviation from Nash equilibrium

Plotting: `eval.analysis.contracting_plots` provides `plot_interaction_effects`, `plot_greenwashing_detection`, `plot_formation_rates`, `plot_activation_rates`, `plot_effort_distribution`, `plot_welfare_comparison`, `create_summary_table`.

## Trace Analysis and Log Inspection

`.eval` files are ZIP archives containing JSON. Read summaries from `_journal/summaries/*.json` rather than individual `samples/*.json` for most analysis. Scorer names: `all_strategies_scorer` for main eval, `contracting_scorer` for contracting eval. Use `.get()` with defaults since not all samples have all fields.

Key contracting scorer metadata: `contract_formed`, `contract_complied`, `formation_failure_reason`, `compliance_failure_reason`, `row_effort_level`, `col_effort_level`, `row_action_category`, `contract_text`, `negotiation_history`.

**Trace analysis tools**:
- `scripts/analysis/trace_extract.py`: Extract contracting traces as structured JSON
- `gt-harmbench-trace-analysis` skill: Claude Code skill for trace inspection (`.cursor/skills/`)
- `traces_viewer/app.py`: Streamlit frontend for interactive browsing
- `eval/analysis/multi_model_analysis.ipynb`: Main analysis notebook for multi-model comparisons

## Coding Guidelines

- Use Hydra for configuration (override via CLI: `script.py param=value`)
- Favor pure, side-effect-free helpers for game logic
- Use `TypedDict`/`pydantic` for input/output schemas
- Keep experiments reproducible with deterministic seeds
- Break down large functions into focused single-purpose helpers
- Update README documentation when changing CLI interfaces
