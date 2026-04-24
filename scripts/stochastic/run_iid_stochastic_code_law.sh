#!/usr/bin/env bash
# Run code_law contracting + i.i.d. stochastic regeneration (seeds 0-4).
# Usage: bash scripts/stochastic/run_iid_stochastic_code_law.sh
# Override model: MODEL=openai/gpt-4o bash scripts/stochastic/run_iid_stochastic_code_law.sh
set -euo pipefail

MODEL=${MODEL:-"openai/gpt-5.4-mini"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
GROUP=${GROUP:-"code_law-5.4-mini"}
NAME=${NAME:-"gpt-5.4-mini"}
TOTAL_AGENTS=${TOTAL_AGENTS:-5}
PROSOCIAL_COUNT=${PROSOCIAL_COUNT:-0}
DATE=$(date +%Y-%m-%d)
HOUR=$(date +%H:%M)
NAME="$NAME-p${PROSOCIAL_COUNT}-$DATE-$HOUR"
PATH_NAME=$NAME/$GROUP

if (( PROSOCIAL_COUNT < 0 || PROSOCIAL_COUNT > TOTAL_AGENTS )); then
  echo "Error: PROSOCIAL_COUNT must be between 0 and TOTAL_AGENTS ($TOTAL_AGENTS)." >&2
  exit 1
fi

echo "Model : $PATH_NAME"
echo "Group : $GROUP"
echo "Seeds : $SEEDS"
echo "Total agents : $TOTAL_AGENTS"
echo "Prosocial agents : $PROSOCIAL_COUNT"
echo ""

for seed in $SEEDS; do
  while true; do
    current_uv_count="$(pgrep -x -c uv || true)"
    current_uv_count=${current_uv_count:-0}
    if (( current_uv_count < 20 )); then
      break
    fi
    echo ">>> Waiting to launch seed=$seed (uv count: $current_uv_count, max: 6)"
    sleep "60"
  done

  echo ">>> Launching seed=$seed"
  persona_overrides=()
  for ((i=0; i<TOTAL_AGENTS; i++)); do
    if (( i < PROSOCIAL_COUNT )); then
      persona="prosocial_fisherman"
    else
      persona="selfish_fisherman"
    fi
    persona_overrides+=("experiment/persona@experiment.personas.persona_${i}=${persona}")
  done

  uv run python -m simulation.main \
    experiment=fish_iid_stochastic_code_law \
    group_name="$PATH_NAME" \
    llm.path="$MODEL" \
    seed="$seed" \
    "experiment.env.regen_seed=$seed" \
    "${persona_overrides[@]}" &
done
wait

echo "All seeds done."
