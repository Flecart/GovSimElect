#!/usr/bin/env bash
# Launch all stochastic variants across multiple models.
# Usage: bash scripts/stochastic/stochastic.sh
set -euo pipefail

launch_case() {
  local model="$1"
  local name="$2"
  local group="$3"
  local prosocial_count="$4"
  local script_path="$5"

  MODEL="$model" \
  NAME="$name" \
  GROUP="$group" \
  PROSOCIAL_COUNT="$prosocial_count" \
  bash "$script_path" &
}

launch_model_matrix() {
  local model="$1"
  local name="$2"

  launch_case "$model" "$name" "1-nl" 0 "scripts/stochastic/run_iid_stochastic_nl.sh"
  launch_case "$model" "$name" "1-nl-prosocial" 5 "scripts/stochastic/run_iid_stochastic_nl.sh"
  launch_case "$model" "$name" "2-code-law" 0 "scripts/stochastic/run_iid_stochastic_code_law.sh"
  launch_case "$model" "$name" "2-code-law-prosocial" 5 "scripts/stochastic/run_iid_stochastic_code_law.sh"
  launch_case "$model" "$name" "0-no-contract" 0 "scripts/stochastic/run_iid_stochastic_no_contract.sh"
  launch_case "$model" "$name" "0-no-contract-prosocial" 5 "scripts/stochastic/run_iid_stochastic_no_contract.sh"
}

launch_model_matrix "openai/gpt-5.4-mini" "sto-gpt-5.4-mini"
launch_model_matrix "qwen/qwen3.6-plus" "sto-qwen3.6-plus"
launch_model_matrix "mistralai/mistral-small-2603" "sto-mistral-small-2603"
launch_model_matrix "x-ai/grok-4.1-fast" "sto-grok-4.1-fast"

wait
