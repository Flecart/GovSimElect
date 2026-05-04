#!/bin/bash

# Combinations to run:
# 1: Qwen3-8B vs deepseek-v3.2
# 2: claude-4.5-sonnet vs Qwen3-8B
# 3: claude-4.5-sonnet vs llama-3.2-3b-instruct

declare -a pairs=(
    "openrouter/Qwen/Qwen3-8B openrouter/deepseek/deepseek-v3.2"
    "openrouter/anthropic/claude-4.5-sonnet openrouter/Qwen/Qwen3-8B"
    "openrouter/anthropic/claude-4.5-sonnet openrouter/meta-llama/llama-3.2-3b-instruct"
)

# wait for jobs to finish
max_jobs=8
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 0.1
    done
}

run_cross_eval() {
    local m_row=$1
    local m_col=$2
    local m_row_clean=${m_row##*/}
    local m_col_clean=${m_col##*/}

    echo "Running cross play: ${m_row_clean} (Row) vs ${m_col_clean} (Col)"

    uv run python3 -m eval.eval \
        --model-row "$m_row" \
        --model-col "$m_col" \
        --dataset data/gt-harmbench-with-targets.csv \
        --limit 100 \
        --times 1 \
        --experiment-name "cross_play_${m_row_clean}_vs_${m_col_clean}" \
        --log-dir logs/cross_play
}

for pair in "${pairs[@]}"; do
    read -r m1 m2 <<< "$pair"
    
    # m1 as row, m2 as col
    wait_for_slot
    run_cross_eval "$m1" "$m2" &

    # m2 as row, m1 as col
    wait_for_slot
    run_cross_eval "$m2" "$m1" &
done

wait
echo "Cross play experiments completed."
