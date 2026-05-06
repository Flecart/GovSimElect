#!/bin/bash

set -e

# Get the script's directory to establish paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default to most recent log directory if none specified
LOG_BASE="${1:-$(ls -td "$PROJECT_DIR"/logs/eval-*-*/ 2>/dev/null | head -1 | sed 's:/$::')}"

if [ ! -d "$LOG_BASE" ]; then
    echo "Error: Log directory not found: $LOG_BASE"
    echo "Usage: $0 [log_directory]"
    echo "Example: $0 logs/eval-20260419-201903-openai-gpt-4o"
    echo ""
    echo "Or run without arguments to use the most recent log directory."
    exit 1
fi

# Extract model name from log directory name
MODEL_FROM_DIR=$(basename "$LOG_BASE" | sed 's/^eval-[0-9]*-[0-9]*-//' || echo "unknown")

echo "=================================================="
echo "GT-HarmBench Contracting Results Analysis"
echo "=================================================="
echo "Log base: $LOG_BASE"
echo "Model: $MODEL_FROM_DIR"
echo "Project dir: $PROJECT_DIR"
echo ""

# Make sure we're in project dir for Python module resolution
cd "$PROJECT_DIR"

# Get Python binary once to avoid repeated uv overhead
PYTHON_BIN=$(uv run --no-sync which python)

# Detect available prompt modes
PROMPT_MODES=()
for dir in "$LOG_BASE"/*/; do
    if [ -d "$dir" ]; then
        mode_name=$(basename "$dir")
        if [ -d "$dir/4x4-no-comm" ] || [ -d "$dir/4x4-code-nl" ] || [ -d "$dir/2x2-no-comm" ]; then
            PROMPT_MODES+=("$mode_name")
        fi
    fi
done

if [ ${#PROMPT_MODES[@]} -eq 0 ]; then
    echo "Error: No prompt mode subdirectories found under: $LOG_BASE"
    exit 1
else
    echo "Found prompt modes: ${PROMPT_MODES[*]}"
fi
echo ""

EXPERIMENTS=(
    "4x4-no-comm"
    "4x4-code-nl"
    "4x4-code-law"
    "2x2-no-comm"
    "2x2-code-nl"
    "2x2-code-law"
)

for PM in "${PROMPT_MODES[@]}"; do
    echo "=================================================="
    echo "Processing prompt mode: $PM"
    echo "=================================================="
    echo ""

    # Create output directory for this prompt mode
    ANALYSIS_DIR="$LOG_BASE/analysis/$PM"
    mkdir -p "$ANALYSIS_DIR"

    echo "Step 1: Extracting metrics from experiments"
    echo ""

    # Extract metrics from each experiment for this prompt mode
    PIDS=()
    JOBS=()
    for exp in "${EXPERIMENTS[@]}"; do
        LOG_DIR="$LOG_BASE/$PM/$exp"
        OUTPUT="$ANALYSIS_DIR/metrics-$exp.csv"

        if [ -d "$LOG_DIR" ]; then
            echo "  Extracting: $exp → $OUTPUT"
            rm -f "$OUTPUT"
            (
                "$PYTHON_BIN" -m eval.analysis.extract_contracting_metrics \
                    --directory "$LOG_DIR" \
                    --output "$OUTPUT"
            ) >/dev/null 2>&1 &
            PIDS+=("$!")
            JOBS+=("$exp")
        else
            echo "  Skipping: $exp (directory not found)"
        fi
    done

    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  Done: ${JOBS[$i]}"
        else
            echo "    Warning: No metrics found for ${JOBS[$i]}"
        fi
    done
    echo ""

    echo "Step 2: Creating summary table for $PM"
    echo ""

    # Create summary CSV for this prompt mode only
    SUMMARY="$ANALYSIS_DIR/summary.csv"

    "$PYTHON_BIN" - "$ANALYSIS_DIR" "$SUMMARY" "${EXPERIMENTS[@]}" <<'PY'
import csv
import sys
from pathlib import Path

analysis_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
experiments = sys.argv[3:]

columns = [
    "experiment",
    "contract_formation_rate",
    "contract_activation_rate",
    "nash_accuracy",
    "utilitarian_accuracy",
    "rawlsian_accuracy",
    "avg_utilitarian_payoff",
    "avg_rawlsian_payoff",
    "utilitarian_payoff_variance",
    "rawlsian_payoff_variance",
    "cooperation_rate",
    "high_effort_rate",
    "avg_row_payoff",
    "avg_col_payoff",
    "avg_turns_to_agreement",
    "joint_action_dataset",
    "joint_action_count",
    "joint_action_matrix",
    "trace_call_count",
    "trace_elapsed_seconds",
    "trace_input_tokens",
    "trace_output_tokens",
    "trace_total_tokens",
    "trace_reasoning_tokens",
    "trace_calls_with_usage",
    "trace_errors",
    "negotiation_call_count",
    "negotiation_total_tokens",
    "negotiation_elapsed_seconds",
    "decision_call_count",
    "decision_total_tokens",
    "decision_elapsed_seconds",
    "coding_call_count",
    "coding_total_tokens",
    "coding_elapsed_seconds",
]

metric_columns = {
    "nash_accuracy": "nash_accuracy_with_contract",
    "utilitarian_accuracy": "utilitarian_accuracy_with_contract",
    "rawlsian_accuracy": "rawlsian_accuracy_with_contract",
}

with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for exp in experiments:
        metrics_file = analysis_dir / f"metrics-{exp}.csv"
        if not metrics_file.exists():
            continue

        with metrics_file.open(newline="") as mf:
            rows = list(csv.DictReader(mf))
        if not rows:
            continue

        for metrics in rows:
            summary = {"experiment": exp}
            for column in columns[1:]:
                source_column = metric_columns.get(column, column)
                summary[column] = metrics.get(source_column, "")
            writer.writerow(summary)
PY

    echo "Summary table for $PM:"
    cut -d, -f1-15 "$SUMMARY" | column -t -s,
    echo ""

    echo "Step 3: Generating plots for $PM"
    echo ""

    # Check if we have results to plot
    HAS_RESULTS=false
    for exp in "${EXPERIMENTS[@]}"; do
        if [ -f "$ANALYSIS_DIR/metrics-$exp.csv" ]; then
            HAS_RESULTS=true
            break
        fi
    done

    if [ "$HAS_RESULTS" = true ]; then
        echo "Generating analysis plots for $PM..."

        # Extract model name from eval logs for this prompt mode
        MODEL_NAME=$(grep -h "^  Model: " "$LOG_BASE/$PM"/*/output.log 2>/dev/null | head -1 | sed 's/.*Model: \([^ ]*\).*/\1/' || echo "$MODEL_FROM_DIR")

        cd "$PROJECT_DIR"
        PYTHONPATH=. "$PYTHON_BIN" scripts/experiments/plot_contracting_results.py \
            "$SUMMARY" \
            "$ANALYSIS_DIR" \
            "$MODEL_NAME" 2>/dev/null || echo "  Warning: Plotting failed for $PM"
    else
        echo "No results found to plot for $PM."
    fi

    echo ""
    echo "✓ Analysis complete for prompt mode: $PM"
    echo "  Results: $ANALYSIS_DIR"
    echo ""
done

echo "=================================================="
echo "All prompt modes analyzed"
echo "=================================================="
echo ""
echo "Results available in:"
echo "  $LOG_BASE"
echo ""

