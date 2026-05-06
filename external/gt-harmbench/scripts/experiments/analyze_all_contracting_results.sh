#!/bin/bash
# Run analyze_contracting_results.sh for all eval directories in logs/

set -e

# Get the script's directory to establish paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ANALYZE_SCRIPT="$PROJECT_DIR/scripts/experiments/analyze_contracting_results.sh"

if [ ! -f "$ANALYZE_SCRIPT" ]; then
    echo "Error: analyze_contracting_results.sh not found at $ANALYZE_SCRIPT"
    exit 1
fi

# Find all eval directories in logs/
LOGS_BASE="$PROJECT_DIR/logs"
EVAL_DIRS=($(ls -d "$LOGS_BASE"/eval-*-*/ 2>/dev/null || true))

if [ ${#EVAL_DIRS[@]} -eq 0 ]; then
    echo "Error: No eval directories found in $LOGS_BASE"
    exit 1
fi

echo "=================================================="
echo "Analyzing all contracting results"
echo "=================================================="
echo "Found ${#EVAL_DIRS[@]} eval directories"
echo ""

# Process each eval directory
for eval_dir in "${EVAL_DIRS[@]}"; do
    # Remove trailing slash
    eval_dir="${eval_dir%/}"

    echo "=================================================="
    echo "Processing: $eval_dir"
    echo "=================================================="

    if "$ANALYZE_SCRIPT" "$eval_dir"; then
        echo "✓ Success: $eval_dir"
    else
        echo "✗ Failed: $eval_dir"
    fi
    echo ""
done

echo "=================================================="
echo "All analyses complete"
echo "=================================================="
