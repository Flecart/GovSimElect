#!/usr/bin/env bash
# Reproducible sweep helper for GovSimElect.
# Runs NL, code-law, and no-contract for prosocial counts 0..5 and seeds 0..4.
#
# Supported backends:
# - OpenAI:     MODEL="openai/gpt-4o" (requires OPENAI_API_KEY)
# - OpenRouter: MODEL="anthropic/claude-sonnet-4.5" (requires OPENROUTER_API_KEY)
#
# Usage:
#   MODEL="openai/gpt-4o" NAME="gpt-4o" GROUP_PREFIX="repro_openai_gpt4o" \
#   bash scripts/standard/examples_models.sh
set -euo pipefail

MODEL=${MODEL:-"openai/gpt-4o"}
NAME=${NAME:-"gpt-4o"}
GROUP_PREFIX=${GROUP_PREFIX:-"repro"}
SCRIPT_FAMILY=${SCRIPT_FAMILY:-"standard"} # standard | stochastic
SEEDS=${SEEDS:-"0 1 2 3 4"}
PROSOCIAL_COUNTS=${PROSOCIAL_COUNTS:-"0 1 2 3 4 5"}

if [[ "$MODEL" == openai/* ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY is required for MODEL=$MODEL" >&2
    exit 1
  fi
else
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "Error: OPENROUTER_API_KEY is required for MODEL=$MODEL" >&2
    exit 1
  fi
fi

if [[ "$SCRIPT_FAMILY" != "standard" && "$SCRIPT_FAMILY" != "stochastic" ]]; then
  echo "Error: SCRIPT_FAMILY must be 'standard' or 'stochastic'." >&2
  exit 1
fi

BASE_DIR="scripts/${SCRIPT_FAMILY}"
SCRIPT_NL="${BASE_DIR}/run_iid_stochastic_nl.sh"
SCRIPT_CODE_LAW="${BASE_DIR}/run_iid_stochastic_code_law.sh"
SCRIPT_NO_CONTRACT="${BASE_DIR}/run_iid_stochastic_no_contract.sh"

for required_script in "$SCRIPT_NL" "$SCRIPT_CODE_LAW" "$SCRIPT_NO_CONTRACT"; do
  if [[ ! -f "$required_script" ]]; then
    echo "Error: missing required script: $required_script" >&2
    exit 1
  fi
done

echo "Starting reproducibility sweep"
echo "MODEL=$MODEL"
echo "NAME=$NAME"
echo "GROUP_PREFIX=$GROUP_PREFIX"
echo "SCRIPT_FAMILY=$SCRIPT_FAMILY"
echo "SEEDS=$SEEDS"
echo "PROSOCIAL_COUNTS=$PROSOCIAL_COUNTS"
echo ""

for prosocial in $PROSOCIAL_COUNTS; do
  MODEL="$MODEL" NAME="$NAME" GROUP="${GROUP_PREFIX}-1-nl-p${prosocial}" \
    PROSOCIAL_COUNT="$prosocial" SEEDS="$SEEDS" bash "$SCRIPT_NL" &

  MODEL="$MODEL" NAME="$NAME" GROUP="${GROUP_PREFIX}-2-code-law-p${prosocial}" \
    PROSOCIAL_COUNT="$prosocial" SEEDS="$SEEDS" bash "$SCRIPT_CODE_LAW" &

  MODEL="$MODEL" NAME="$NAME" GROUP="${GROUP_PREFIX}-0-no-contract-p${prosocial}" \
    PROSOCIAL_COUNT="$prosocial" SEEDS="$SEEDS" bash "$SCRIPT_NO_CONTRACT" &
done

wait
echo "Reproducibility sweep complete."
