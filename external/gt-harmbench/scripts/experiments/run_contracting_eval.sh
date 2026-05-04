#!/bin/bash
# Run contracting evaluation on 4x4 and 2x2 datasets
# Supports PROMPT_MODE=all to run base, selfish, cooperative in series
# Runs all 6 experiments in parallel (3 modes × 2 datasets) per prompt mode

set -e

# Get the script's directory to establish paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
LIMIT="${LIMIT:-}"                    # Optional: limit number of samples (e.g., 50, 100)
TIMES="${TIMES:-10}"                  # Number of iterations
MODEL="${MODEL:-openai/gpt-5.1}"      # Model to use
PROMPT_MODE="${PROMPT_MODE:-base}"    # base, selfish, cooperative, or all
GAME="${GAME:-pd}"                    # Game type: pd (Prisoner's Dilemma), sh (Stag Hunt), co (Coordination)
SAMPLE="${SAMPLE:-30}"                # Sample size for coordination games (default: 30)
DATASET_4X4="${DATASET_4X4:-}"       # Optional: override 4x4 dataset path
DATASET_2X2="${DATASET_2X2:-}"       # Optional: override 2x2 dataset path

# Sanitize model name for folder naming (replace slashes with dashes)
MODEL_SANITIZED="${MODEL//\//-}"

# Dynamic naming based on limit
if [ -n "$LIMIT" ]; then
    LIMIT_LABEL="${LIMIT}q"
else
    LIMIT_LABEL="full"
fi
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_BASE="$PROJECT_DIR/logs/eval-${TIMESTAMP}-${MODEL_SANITIZED}-${GAME}"

# Dataset selection (based on GAME type or custom paths)
# If custom datasets are provided, use them; otherwise use defaults based on GAME type
if [ -n "$DATASET_2X2" ]; then
    # Custom 2x2 dataset provided
    GAME_DESC="Custom"
    USE_4X4=false
    if [ -n "$DATASET_4X4" ]; then
        USE_4X4=true
        # Validate 4x4 dataset exists
        if [ ! -f "$PROJECT_DIR/$DATASET_4X4" ]; then
            echo "Error: 4x4 dataset not found: $PROJECT_DIR/$DATASET_4X4"
            exit 1
        fi
    fi
    # Validate 2x2 dataset exists
    if [ ! -f "$PROJECT_DIR/$DATASET_2X2" ]; then
        echo "Error: 2x2 dataset not found: $PROJECT_DIR/$DATASET_2X2"
        exit 1
    fi
elif [ "$GAME" = "sh" ]; then
    DATASET_4X4="data/gt-harmbench-sh30-4x4-rewritten.csv"
    DATASET_2X2="data/gt-harmbench-sh30-2x2-with-targets.csv"
    GAME_DESC="Stag Hunt"
    USE_4X4=true
elif [ "$GAME" = "co" ]; then
    # Coordination: 2x2 only
    DATASET_2X2="data/gt-harmbench-coordination${SAMPLE}-2x2-with-targets.csv"
    GAME_DESC="Coordination"
    USE_4X4=false
else
    DATASET_4X4="data/gt-harmbench-pd30-4x4-rewritten.csv"
    DATASET_2X2="data/gt-harmbench-pd30-2x2-with-targets.csv"
    GAME_DESC="Prisoner's Dilemma"
    USE_4X4=true
fi

if [ -n "$LIMIT" ]; then
    DATASET_DESC="${GAME_DESC} datasets (limit: $LIMIT)"
else
    DATASET_DESC="${GAME_DESC} datasets (full)"
fi

echo "=================================================="
echo "GT-HarmBench Contracting Evaluation"
echo "=================================================="
echo "Project dir: $PROJECT_DIR"
echo "Model: $MODEL"
echo "Game type: $GAME ($GAME_DESC)"
echo "Prompt mode: $PROMPT_MODE"
echo "Iterations: $TIMES"
if [ -n "$LIMIT" ]; then
    echo "Sample limit: $LIMIT"
else
    echo "Sample limit: None (full dataset)"
fi
echo "Log base: $LOG_BASE"
echo ""
echo "Datasets: $DATASET_DESC"
if [ "$USE_4X4" = true ]; then
    echo "  - 4x4: $DATASET_4X4"
fi
echo "  - 2x2: $DATASET_2X2"
echo ""

# Create log directory
mkdir -p "$LOG_BASE"

# Export for subprocesses
export MODEL
export TIMES
export PROMPT_MODE
export PROJECT_DIR
export LIMIT

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local mode=$2
    local name_suffix=$3
    local prompt_mode=$4  # base, selfish, or cooperative

    local log_dir="$LOG_BASE/${prompt_mode}/$name_suffix"
    local exp_name="${LIMIT_LABEL}-${mode}"

    # Create log directory
    mkdir -p "$log_dir"

    cd "$PROJECT_DIR"

    # Build command with optional limit
    # Use -u for unbuffered Python output
    local cmd="uv run --with pandas --with python-dotenv python -u -m eval.contracting_eval"
    cmd+=" --dataset \"$dataset\""
    cmd+=" --contract-mode \"$mode\""
    cmd+=" --times \"$TIMES\""
    cmd+=" --prompt-mode \"$prompt_mode\""
    cmd+=" --experiment-name \"$exp_name\""
    cmd+=" --log-dir \"$log_dir\""
    cmd+=" --model-name \"$MODEL\""
    cmd+=" --game-type \"$GAME\""

    if [ -n "$LIMIT" ]; then
        cmd+=" --limit \"$LIMIT\""
    fi

    # Show output in real-time AND log to file using tee
    eval $cmd 2>&1 | tee "$log_dir/output.log"
}

export -f run_experiment

echo "=================================================="
echo "Experiment Matrix"
echo "=================================================="
echo ""

if [ "$USE_4X4" = true ]; then
    echo "4x4 Dataset:"
    echo "  1. no_communication"
    echo "  2. code_nl"
    echo "  3. code_law"
    echo ""
fi
echo "2x2 Dataset:"
if [ "$USE_4X4" = true ]; then
    echo "  4. no_communication"
    echo "  5. code_nl"
    echo "  6. code_law"
else
    echo "  1. no_communication"
    echo "  2. code_nl"
    echo "  3. code_law"
fi
echo ""
echo "=================================================="
echo ""

# Determine which prompt modes to run
if [ "$PROMPT_MODE" = "all" ]; then
    PROMPT_MODES=("base" "selfish" "cooperative")
else
    PROMPT_MODES=("$PROMPT_MODE")
fi

# Run experiments for each prompt mode
for PM in "${PROMPT_MODES[@]}"; do
    echo "=================================================="
    echo "Running prompt mode: $PM"
    echo "=================================================="
    echo ""

    # Run experiments in parallel using background jobs
    if [ "$USE_4X4" = true ]; then
        # Run both 4x4 and 2x2 experiments (PD/SH)
        # 4x4 experiments
        run_experiment "$DATASET_4X4" "no_communication" "4x4-no-comm" "$PM" &
        PID_1=$!

        run_experiment "$DATASET_4X4" "code_nl" "4x4-code-nl" "$PM" &
        PID_2=$!

        run_experiment "$DATASET_4X4" "code_law" "4x4-code-law" "$PM" &
        PID_3=$!

        # 2x2 experiments
        run_experiment "$DATASET_2X2" "no_communication" "2x2-no-comm" "$PM" &
        PID_4=$!

        run_experiment "$DATASET_2X2" "code_nl" "2x2-code-nl" "$PM" &
        PID_5=$!

        run_experiment "$DATASET_2X2" "code_law" "2x2-code-law" "$PM" &
        PID_6=$!

        echo "=================================================="
        echo "Started all experiments for prompt mode: $PM"
        echo "=================================================="
        echo ""
        echo "Process IDs:"
        echo "  4x4-no-comm:   PID $PID_1"
        echo "  4x4-code-nl:   PID $PID_2"
        echo "  4x4-code-law:  PID $PID_3"
        echo "  2x2-no-comm:   PID $PID_4"
        echo "  2x2-code-nl:   PID $PID_5"
        echo "  2x2-code-law:  PID $PID_6"
    else
        # Run only 2x2 experiments (Coordination)
        run_experiment "$DATASET_2X2" "no_communication" "2x2-no-comm" "$PM" &
        PID_1=$!

        run_experiment "$DATASET_2X2" "code_nl" "2x2-code-nl" "$PM" &
        PID_2=$!

        run_experiment "$DATASET_2X2" "code_law" "2x2-code-law" "$PM" &
        PID_3=$!

        echo "=================================================="
        echo "Started all experiments for prompt mode: $PM"
        echo "=================================================="
        echo ""
        echo "Process IDs:"
        echo "  2x2-no-comm:   PID $PID_1"
        echo "  2x2-code-nl:   PID $PID_2"
        echo "  2x2-code-law:  PID $PID_3"
    fi
    echo ""
    echo "Waiting for experiments to complete..."
    echo ""

    # Wait for all background jobs
    wait

    echo ""
    echo "✓ All experiments completed for prompt mode: $PM"
    echo ""
done

echo ""
echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
echo ""
echo "Results logged to: $LOG_BASE/"
echo ""
echo "=================================================="
echo "Next steps: Extract metrics and generate plots"
echo "=================================================="
echo ""
echo "Run the analysis script:"
echo ""
echo "  ./scripts/experiments/analyze_contracting_results.sh $LOG_BASE"
echo ""
echo "Or simply (uses most recent run):"
echo ""
echo "  ./scripts/experiments/analyze_contracting_results.sh"
echo ""
