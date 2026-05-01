#!/usr/bin/env bash
set -euo pipefail

pattern="${1:-}"
output_csv="${2:-aggregated.csv}"
tmp_md="${3:-tmp.md}"

if [[ -z "$pattern" ]]; then
  echo "Usage: $0 \"<experiment_glob_pattern>\" [output_csv] [tmp_markdown]" >&2
  echo "Example: $0 \"simulation/results/gpt-4o-p*-2026-04-*\" aggregated.csv tmp.md" >&2
  exit 1
fi

# echo "Step 1/2: aggregating summaries to ${tmp_md}"
# bash scripts/run_aggregate_special_all.sh "$pattern" "$tmp_md"

echo "Step 2/2: parsing summaries to ${output_csv}"
uv run python3 scripts/logs/parser.py "$tmp_md" "$output_csv"

echo "Done: wrote ${output_csv}"
