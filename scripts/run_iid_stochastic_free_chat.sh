#!/usr/bin/env bash
# Run gpt-5.4 on free-chat contracting + i.i.d. stochastic regeneration.
# Usage: bash scripts/run_iid_stochastic_free_chat.sh
# Override model: MODEL=openai/gpt-4o bash scripts/run_iid_stochastic_free_chat.sh
set -euo pipefail

MODEL=${MODEL:-"openai/gpt-5.4"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
GROUP=${GROUP:-"iid_stochastic_free_chat-8-apr-2026"}

echo "Model : $MODEL"
echo "Group : $GROUP"
echo "Seeds : $SEEDS"
echo "Experiment : fish_iid_stochastic_free_chat"
echo ""

for seed in $SEEDS; do
  echo ">>> Launching seed=$seed"
  uv run python -m simulation.main \
    experiment=fish_iid_stochastic_free_chat \
    group_name="$GROUP" \
    llm.path="$MODEL" \
    llm.backend=OpenAI \
    seed="$seed" \
    "experiment.env.regen_seed=$seed" &
done

wait
echo "All seeds done."
