#!/bin/bash
# Run contracting evaluations for selected models across Prisoner's Dilemma and Stag Hunt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_contracting_eval.sh"

# Limit outer concurrency because each runner invocation starts several eval jobs itself.
MAX_JOBS="${MAX_JOBS:-2}"
LIMIT="${LIMIT:-30}"
BATCH_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BATCH_LOG_DIR="$PROJECT_DIR/logs/contracting-model-games-$BATCH_TIMESTAMP"
MASTER_LOG="$BATCH_LOG_DIR/master.log"
TIMING_LOG="$BATCH_LOG_DIR/execution_times.tsv"

MODELS=(
    "openai/gpt-5.4-mini"
    "openai/gpt-4o"
    "openrouter/x-ai/grok-4.1-fast"
    "openrouter/google/gemma-4-31b-it"
    "openai/gpt-5.4"
)

GAMES=("pd" "sh")

mkdir -p "$BATCH_LOG_DIR" "$PROJECT_DIR/logs"

printf 'timestamp\tstatus\tmodel\tgame\telapsed_seconds\texit_code\toutput_log\tcommand\tdetails\n' > "$TIMING_LOG"
exec > >(tee -a "$MASTER_LOG") 2>&1

sanitize_model() {
    local model=$1
    printf '%s' "${model//\//-}"
}

existing_run_matches() {
    local sanitized_model=$1
    local game=$2
    local needle="${sanitized_model}-${game}"
    local base
    local path

    shopt -s nullglob
    for path in "$PROJECT_DIR"/logs/eval-*; do
        [ -d "$path" ] || continue
        base="${path##*/}"
        case "$base" in
            *"$needle"*) printf '%s\n' "$base" ;;
        esac
    done
    shopt -u nullglob
}

join_lines_with_commas() {
    local line
    local joined=""

    while IFS= read -r line; do
        if [ -z "$joined" ]; then
            joined="$line"
        else
            joined="$joined,$line"
        fi
    done

    printf '%s' "$joined"
}

append_timing_log() {
    local status=$1
    local model=$2
    local game=$3
    local elapsed=$4
    local exit_code=$5
    local output_log=$6
    local command=$7
    local details=$8
    local timestamp

    timestamp="$(date --iso-8601=seconds)"
    {
        flock 200
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$timestamp" "$status" "$model" "$game" "$elapsed" "$exit_code" "$output_log" "$command" "$details" >&200
    } 200>>"$TIMING_LOG"
}

run_model_game() {
    local model=$1
    local game=$2
    local sanitized_model
    local output_log
    local command_text
    local matches
    local start_epoch
    local elapsed
    local exit_code

    sanitized_model="$(sanitize_model "$model")"
    output_log="$BATCH_LOG_DIR/${sanitized_model}-${game}.log"
    command_text="MODEL=$model GAME=$game LIMIT=$LIMIT $RUNNER"

    matches="$(existing_run_matches "$sanitized_model" "$game" | join_lines_with_commas)"
    if [ -n "$matches" ]; then
        echo "Skipping $model / $game: existing run(s): $matches" | tee "$output_log"
        append_timing_log "skipped" "$model" "$game" "0" "0" "$output_log" "$command_text" "existing_runs=$matches"
        return 0
    fi

    echo "Starting $model / $game"
    echo "Output log: $output_log"
    start_epoch="$(date +%s)"

    set +e
    (
        cd "$PROJECT_DIR"
        env MODEL="$model" GAME="$game" LIMIT="$LIMIT" "$RUNNER"
    ) 2>&1 | tee "$output_log"
    exit_code=${PIPESTATUS[0]}
    set -e

    elapsed="$(( $(date +%s) - start_epoch ))"
    if [ "$exit_code" -eq 0 ]; then
        append_timing_log "completed" "$model" "$game" "$elapsed" "$exit_code" "$output_log" "$command_text" ""
    else
        append_timing_log "failed" "$model" "$game" "$elapsed" "$exit_code" "$output_log" "$command_text" ""
    fi

    return "$exit_code"
}

wait_for_slot() {
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_JOBS must be a positive integer, got: $MAX_JOBS" >&2
    exit 1
fi

echo "Writing batch logs to: $BATCH_LOG_DIR"
echo "Writing master output to: $MASTER_LOG"
echo "Writing execution times to: $TIMING_LOG"
echo "Outer max jobs: $MAX_JOBS"
echo "Sample limit: $LIMIT"
echo ""

pids=()

for model in "${MODELS[@]}"; do
    for game in "${GAMES[@]}"; do
        wait_for_slot
        run_model_game "$model" "$game" &
        pids+=("$!")
    done
done

failures=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        ((failures+=1))
    fi
done

echo ""
echo "Batch complete. Logs: $BATCH_LOG_DIR"
echo "Master log: $MASTER_LOG"
echo "Timing log: $TIMING_LOG"

if [ "$failures" -gt 0 ]; then
    echo "$failures run(s) failed. See per-command logs and execution_times.tsv." >&2
    exit 1
fi
