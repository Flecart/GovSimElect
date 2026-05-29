#!/usr/bin/env bash
# One model, both regimes (no_contract vs code_ddl), full 12-round horizon,
# paper config (regen [2.0,2.0], seed 0). Prints commons metrics for each.
#   MODEL=openai/gpt-4o TAG=gpt4o bash scripts/sweep_ddl.sh
set -euo pipefail
MODEL=${MODEL:?set MODEL, e.g. openai/gpt-5.4}
TAG=${TAG:?set TAG, e.g. gpt54}
SEED=${SEED:-0}
ROUNDS=${ROUNDS:-12}
export PYTHONPATH="./"
export DEONTIC_BIN=${DEONTIC_BIN:-"$HOME/Desktop/work/deontic/.lake/build/bin/deontic"}

# Load API keys from ~/.bashrc (defined without `export`) without printing them.
set -a; eval "$(grep -E '^[[:space:]]*(export[[:space:]]+)?(OPENAI_API_KEY|OPENROUTER_API_KEY)=' ~/.bashrc)"; set +a

run() {  # $1 = experiment, $2 = subgroup
  rm -rf "simulation/results/sweep/${TAG}/$2"
  uv run python -m simulation.main \
    experiment="$1" group_name="sweep/${TAG}/$2" \
    llm.path="$MODEL" seed="$SEED" \
    'experiment.env.regen_factor_range=[2.0,2.0]' \
    "experiment.env.regen_seed=$SEED" \
    "experiment.env.max_num_rounds=$ROUNDS" \
    > "/tmp/sweep_${TAG}_$2.log" 2>&1 || echo "  [$2] run exited non-zero (see /tmp/sweep_${TAG}_$2.log)"
}

echo "### $TAG  model=$MODEL  seed=$SEED  rounds=$ROUNDS"
run fish_iid_stochastic_no_contract noc
run fish_iid_stochastic_code_ddl    ddl
.venv/bin/python scripts/ddl_metrics.py \
  "$(find simulation/results/sweep/${TAG}/noc -name log_env.json | head -1 | xargs dirname)" \
  "$(find simulation/results/sweep/${TAG}/ddl -name log_env.json | head -1 | xargs dirname)"
