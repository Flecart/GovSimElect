#!/bin/bash

# Define the list of models using inspect_ai compatible syntax (provider/model)
# This script runs due diligence evaluations (game type recognition and nash equilibria detection)
# for all important models

models=(
    # --- OpenAI ---
    # "openai/gpt-5-mini-2025-08-07"
    # "openai/gpt-5-nano-2025-08-07"
    # "openai/gpt-5.1"
    # "openai/gpt-4o"
    "openrouter/google/gemini-3-flash-preview"

    # "openai/gpt-5.2-2025-12-11"
    
    # --- Anthropic ---
    # "openrouter/anthropic/claude-4.5-sonnet"
    
    # --- Google ---
    # "openrouter/google/gemini-3-flash-preview"
    
    # --- Grok ---
    # "openrouter/x-ai/grok-4.1-fast"
    
    # # --- Meta Llama (via OpenRouter) ---
    # "openrouter/meta-llama/llama-3.3-70b-instruct"
    # "openrouter/meta-llama/llama-3.2-3b-instruct"
    
    # # --- Qwen (via OpenRouter) ---
    # "openrouter/qwen/qwen3-30b-a3b"
    # "openrouter/Qwen/Qwen3-8B"
)

run_eval() {
    local model=$1
    local task=$2
    local reasoning_effort=$3
    echo "Running due diligence eval for model: $model, task: $task, reasoning_effort: $reasoning_effort"
    uv run python3 -m eval.due_diligence \
        --model-name "$model" \
        --dataset-path data/gt-harmbench.csv \
        --limit -1 \
        --task "$task" \
        --log-dir "logs/due_diligence/" \
        --reasoning-effort "$reasoning_effort" 
}

# First, run classification for all models
echo "=== Running Game Classification for all models ==="
for model in "${models[@]}"; do
    run_eval "$model" "classification" "medium" &
done

# Then, run nash equilibrium detection for all models
echo "=== Running Nash Equilibrium Detection for all models ==="
for model in "${models[@]}"; do
    run_eval "$model" "nash" "medium" &
done


wait