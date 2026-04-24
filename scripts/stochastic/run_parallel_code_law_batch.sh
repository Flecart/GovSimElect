#!/usr/bin/env bash
# Run baseline concurrent code_law in parallel.
# Usage: bash scripts/stochastic/run_parallel_code_law_batch.sh
set -euo pipefail

cd "$(dirname "$0")/.."

subset="simulation/results/fishing_v7.0"
RUNS=${RUNS:-5}
BASE_NAME=${BASE_NAME:-"code_law-2"}

echo "Subset : $subset"
echo "Runs : $RUNS"
echo "Base name : $BASE_NAME"
echo ""

for i in $(seq 0 $((RUNS - 1))); do
  echo ">>> Launching run=$i"
  uv run python3 -m simulation.main \
    experiment=fish_baseline_concurrent_code_law \
    debug=true \
    experiment.run_name="${BASE_NAME}" &
done

wait

echo "Runs created under ${subset}/${BASE_NAME}"
