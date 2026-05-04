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
  - `scripts/generation/rewrite_4x4_stories_with_effort.py`: Rewrite stories with moral hazard context using GPT-5.2
  - `scripts/generation/extract_2x2_from_4x4.py`: Extract corresponding 2x2 scenarios from a 4x4 dataset
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
# Run main evaluation
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
uv run python3 scripts/classify_csv_games.py

# Step 2: Generate contextualizations
uv run python3 scripts/generate_contextualizations_from_filter.py

# Step 2b: Heuristic filter
uv run python3 scripts/fix_contextualizations.py

# Step 3: Add targets (Nash, utilitarian, Rawlsian)
uv run python3 -m scripts.add_targets data/contextualization-filtered.csv data/contextualization-with-targets.csv
```

4x4 moral hazard generation:
```bash
# Step 1: Generate 4x4 payoff matrices from 2x2 dataset
uv run python3 scripts/generation/variation_game_theoretic_4x4.py \
  data/gt-harmbench.csv data/gt-harmbench-4x4.csv --pd-only

# Step 2: Rewrite stories with moral hazard context (with automatic 2x2 extraction)
uv run python3 scripts/generation/rewrite_4x4_stories_with_effort.py \
  data/gt-harmbench-4x4.csv \
  --source-2x2 data/gt-harmbench.csv \
  --output data/gt-harmbench-4x4-rewritten.csv

# Test mode (5 rows only)
uv run python3 scripts/generation/rewrite_4x4_stories_with_effort.py \
  data/gt-harmbench-4x4.csv --test
```

Contracting evaluation:
```bash
# Run all contracting modes on 4x4 and 2x2 datasets (parallel per prompt mode)
LIMIT=50 TIMES=10 ./scripts/experiments/run_contracting_eval.sh

# Run with all prompt modes (base, selfish, cooperative in series)
PROMPT_MODE=all LIMIT=50 TIMES=10 ./scripts/experiments/run_contracting_eval.sh

# Run with specific prompt mode
PROMPT_MODE=cooperative LIMIT=50 TIMES=10 ./scripts/experiments/run_contracting_eval.sh

# Specify model and other options
MODEL=openai/gpt-5.1 PROMPT_MODE=all TIMES=10 ./scripts/experiments/run_contracting_eval.sh

# Quick runtime test (5 samples, 1 iteration)
LIMIT=5 TIMES=1 ./scripts/experiments/run_contracting_eval.sh

# Run single mode (direct Python, bypassing shell script)
uv run python3 -m eval.contracting_eval \
  --dataset data/test-gt-harmbench-4x4-rewritten.csv \
  --contract-mode code_nl \
  --limit 5 --times 1

# Analyze results
./scripts/experiments/analyze_contracting_results.sh
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

- `data/game_template.csv`: Definitions of 7 core game types
- `data/gt-harmbench-with-targets.csv`: Main evaluation dataset with computed targets
- `data/gt-harmbench-mh.csv`: Moral hazard variants (realistic format)
- `data/gt-harmbench-gamify-mh.csv`: Moral hazard variants (gamify format)
- `data/gt-harmbench-4x4-rewritten.csv`: 4x4 moral hazard with neutral effort labels (High Effort/Low Effort)
- `data/test-gt-harmbench-4x4-rewritten.csv`: Test dataset (5 scenarios) for quick evaluation
- `data/test-gt-harmbench-2x2.csv`: Corresponding 2x2 test scenarios
- `data/mit.csv`: Source MIT AI Risk Database scenarios

### Contracting Analysis Functions

The `eval.analysis.contracting` module provides shared analysis functions for contracting experiments. These are used by both bash scripts and Jupyter notebooks.

**Core Functions**:

- `load_contracting_logs(log_base_dir, experiments, prompt_modes)` - Load contracting logs from multiple experiments, returns dict of DataFrames with sample-level data

- `compute_contracting_metrics(df)` - Compute all contracting metrics from a DataFrame, returns dict with formation_rate, activation_rate, payoffs, failure reasons, effort distribution

- `aggregate_experiment_metrics(log_base_dir, prompt_modes)` - Load all experiments and compute summary metrics, returns DataFrame with one row per experiment

- `compute_interaction_effects(metrics_df, outcome_col)` - Compute superadditive interaction effects for complementarity test. Formula: (cooperative+contract) - (cooperative alone) - (contract alone) + (baseline). Returns dict with 4x4 and 2x2 interaction values

- `detect_greenwashing(df, prompt_mode)` - Detect Greenwashing behavior (cooperate output + low effort) in 4x4 scenarios, returns DataFrame with instances and rates by prompt mode

- `get_effort_distribution(df, by_mode, by_prompt_mode)` - Extract effort level distribution for 4x4 scenarios, returns DataFrame with counts and percentages

- `get_negotiation_transcripts(df, sample_ids)` - Extract negotiation transcripts from metadata, returns DataFrame with turn-by-turn conversation data

**Normalization Functions**:

- `normalize_score(score, dataset)` - Normalize payoff to 0-100 scale based on dataset bounds
- `compute_normalized_nash_deviation(score, dataset, welfare)` - Compute % improvement from Nash equilibrium

**Example Usage** (in Python or notebook):

```python
from pathlib import Path
from eval.analysis.contracting import (
    load_contracting_logs,
    aggregate_experiment_metrics,
    compute_interaction_effects,
    detect_greenwashing
)

log_base = Path("logs/eval-20260419-143000-openai-gpt-4o")

# Load all experiments with base, selfish, cooperative prompt modes
data = load_contracting_logs(log_base, prompt_modes=("base", "selfish", "cooperative"))
summary = aggregate_experiment_metrics(log_base, prompt_modes=("base", "selfish", "cooperative"))

# Test complementarity prediction
interaction_effects = compute_interaction_effects(summary)
print(f"4x4 interaction effect: {interaction_effects['4x4']:.3f}")
print(f"2x2 interaction effect: {interaction_effects['2x2']:.3f}")

# Detect Greenwashing
greenwashing = detect_greenwashing(data['4x4-code-nl'])
print(f"Greenwashing rate (selfish): {greenwashing['selfish_rate']:.1%}")
```

**Contracting Analysis Notebook**:
- Location: `eval/analysis/contracting_analysis.ipynb`
- Run with: `jupyter notebook eval/analysis/contracting_analysis.ipynb`
- Auto-detects most recent experiment directory and provides interactive analysis

**Plotting Functions** (`eval.analysis.contracting_plots`):

- `plot_interaction_effects(metrics_df, outcome_col, ax)` - 2x2 factorial plot testing complementarity
- `plot_greenwashing_detection(df_4x4, ax)` - Bar chart of Greenwashing rates by prompt mode
- `plot_formation_rates(summary_df, ax)` - Contract formation rate by experiment
- `plot_activation_rates(summary_df, ax)` - Contract activation rate by experiment
- `plot_effort_distribution(metrics_df, ax)` - Effort level distribution (4x4 scenarios)
- `plot_welfare_comparison(summary_df, normalized, ax)` - Welfare comparison across experiments
- `create_summary_table(summary_df, normalized, figsize)` - Create summary table figure

## Trace Analysis and Log Inspection

### Log File Structure

GT-HarmBench uses `inspect_ai` framework which stores evaluation logs in `.eval` files (ZIP archives containing JSON):

```
logs/
├── eval-{TIMESTAMP}-{MODEL}/           # Experiment directory (e.g., eval-20260419-143000-openai-gpt-4o/)
│   ├── base/                            # Prompt mode subdirectory
│   │   ├── 4x4-code-nl/                 # Contract mode subdirectory
│   │   │   ├── 2026-04-14T...eval        # Compressed log file (ZIP)
│   │   │   └── output.log               # Stdout/stderr
│   │   ├── 2x2-code-nl/
│   │   └── ...
│   ├── selfish/
│   │   └── ...
│   ├── cooperative/
│   │   └── ...
│   └── analysis/                        # Generated analysis outputs
```

**Inside .eval file** (ZIP structure):
- `samples/*.json` - Individual sample records with model outputs (PRIMARY DATA SOURCE)
- `_journal/summaries/*.json` - Aggregated score summaries (pre-computed metrics)
- `header.json` - Metadata including model name, eval config
- `reductions.json` - Aggregate metrics

**IMPORTANT**: The `_journal/summaries/*.json` files contain pre-computed scorer outputs in a compact format. For most analysis tasks, read from summaries rather than individual sample files.

### Summary File Structure (Preferred for Analysis)

The `_journal/summaries/` directory contains aggregated results already computed by scorers:

```python
import zipfile
import json

with zipfile.ZipFile("logs/eval-TIMESTAMP/4x4-code-nl/*.eval") as z:
    # Read summaries (pre-computed scorer outputs)
    summary_files = [n for n in z.namelist() if n.startswith('_journal/summaries/')]

    for sf in sorted(summary_files):
        with z.open(sf) as f:
            data = json.load(f)
            # data is a list of sample summaries, or a single summary
```

**Summary entry structure**:
```json
{
  "id": "sample_id",
  "metadata": {
    "formal_game": "Prisoner's Dilemma",
    "actions_row": ["Confess", "Stay silent"],
    "actions_column": ["Confess", "Stay silent"],
    "rewards_matrix": [[[0,0], [-5,5]], [[5,-5], [-1,-1]]],
    "story_row": "...",
    "story_col": "..."
  },
  "target": "[('Confess', 'Confess')]",
  "scores": {
    "all_strategies_scorer": {  // Main eval
      "value": "('Confess', 'Stay silent')",
      "metadata": {
        "answer": "('Confess', 'Stay silent')",
        "correct_nash": false,
        "correct_utilitarian": true,
        "correct_rawlsian": false,
        "row_payoff": -5,
        "col_payoff": 5,
        "utilitarian_payoff": 0,
        "rawlsian_payoff": -5
      }
    },
    "contracting_scorer": {    // Contracting eval
      "value": "NEUTRAL",
      "metadata": {
        "answer_type": "neutral",
        "contract_formed": true,
        "contract_complied": null,
        "enforcement_success": null,
        "formation_failure_reason": null,
        "compliance_failure_reason": "intentional_violation",
        "utilitarian_payoff": -42,
        "rawlsian_payoff": -21,
        "row_action": "Stay silent (Low Effort)",
        "column_action": "Stay silent (Low Effort)",
        "row_base_action": "Stay silent",
        "row_effort_level": "Low Effort",
        "row_action_category": "cooperate",
        "contract_text": "...",
        "negotiation_history": "[{\"turn\": 1, \"player\": \"row\", \"action\": \"propose\", ...}]"
      }
    }
  }
}
```

### Quick Log Reading Pattern

Use this pattern to efficiently read log data:

```python
import zipfile
import json
from pathlib import Path

def read_eval_log(eval_path: str | Path):
    """Read all summaries from an .eval file efficiently.

    Args:
        eval_path: Path to .eval file

    Returns:
        List of summary dicts with id, metadata, scores
    """
    summaries = []
    with zipfile.ZipFile(eval_path, 'r') as z:
        summary_files = sorted([n for n in z.namelist() if n.startswith('_journal/summaries/')])

        for sf in summary_files:
            with z.open(sf) as f:
                data = json.load(f)
                # Handle both list and single item
                items = data if isinstance(data, list) else [data]
                summaries.extend(items)

    return summaries

# Usage
summaries = read_eval_log("logs/eval-TIMESTAMP/4x4-code-nl/*.eval")
print(f"Loaded {len(summaries)} samples")

# Access scorer metadata
for s in summaries:
    scorer_data = s.get('scores', {}).get('all_strategies_scorer', {})
    metadata = scorer_data.get('metadata', {})
    print(f"Sample {s['id']}: utilitarian_payoff={metadata.get('utilitarian_payoff', 'N/A')}")
```

### Key Scorer Metadata Fields

**For main evaluations** (`all_strategies_scorer`):
- `answer` - Actions chosen by model: "('Action1', 'Action2')"
- `correct_nash`, `correct_utilitarian`, `correct_rawlsian` - Boolean correctness flags
- `utilitarian_payoff`, `rawlsian_payoff` - Computed welfare values
- `row_payoff`, `col_payoff` - Individual player payoffs

**For contracting evaluations** (`contracting_scorer`):
- `contract_formed` - Whether players agreed to a contract
- `contract_complied` - Whether a formally enforced contract was honored (`null` for modes without enforcement, including code_nl)
- `formation_failure_reason` - Why contract failed to form (if applicable)
- `compliance_failure_reason` - Why contract was violated (if applicable)
- `row_effort_level`, `col_effort_level` - "High Effort" or "Low Effort" (4x4 scenarios)
- `row_action_category`, `col_action_category` - "cooperate", "defect", etc.
- `contract_text` - The agreed contract terms
- `negotiation_history` - JSON string of negotiation turns

### Finding Logs

```python
from pathlib import Path
import zipfile

# Find most recent experiment directory
logs_base = Path("logs")
eval_dirs = sorted([d for d in logs_base.glob("eval-*") if d.is_dir()],
                   key=lambda x: x.stat().st_mtime, reverse=True)

if eval_dirs:
    log_base = eval_dirs[0]
    print(f"Most recent: {log_base}")

    # List experiment subdirectories
    for exp_dir in sorted(log_base.iterdir()):
        if exp_dir.is_dir():
            eval_files = list(exp_dir.glob("*.eval"))
            if eval_files:
                print(f"  {exp_dir.name}: {len(eval_files)} eval file(s)")
```

### Common Pitfalls

1. **Reading individual sample files instead of summaries**: Individual files in `samples/` have richer data but are slower to read. Use `_journal/summaries/` for most analysis.

2. **Wrong scorer name**: Main eval uses `all_strategies_scorer`, contracting uses `contracting_scorer`. Check which keys exist first.

3. **Missing metadata fields**: Not all samples have all fields. Always use `.get()` with defaults.

4. **Path handling**: `.eval` files are ZIP archives. Use `zipfile.ZipFile()` not `open()`.

5. **Epoch vs regular samples**: Some logs have epoch files like `samples/105_epoch_1.json`. These are still valid samples but follow a naming pattern.

## Coding Guidelines

- Use Hydra for configuration (override via CLI: `script.py param=value`)
- Favor pure, side-effect-free helpers for game logic
- Use `TypedDict`/`pydantic` for input/output schemas
- Keep experiments reproducible with deterministic seeds
- Break down large functions into focused single-purpose helpers
- Update README documentation when changing CLI interfaces
