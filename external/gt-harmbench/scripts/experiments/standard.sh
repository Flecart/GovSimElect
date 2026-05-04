#!/bin/bash

# Define the list of models using inspect_ai compatible syntax (provider/model)
# Note: For open-source models, 'vllm/' implies a local/remote vLLM server. 
# Change to 'hf/' for local transformers or 'together/' etc. for APIs.

models=(
    # --- Grok ---
    # User Note: "Grok 4.1 there's 3 versions, but only fast available by API"
    # "grok/grok-4.1-fast"
    "openrouter/x-ai/grok-4"

    # # --- Gemini ---
    # # User Note: "Gemini 3... thinking of Default" (Pro is usually the default/standard)
    # "openrouter/google/gemini-3-pro-preview"

    # # --- Anthropic ---
    # # User Note: "Opus 4.5 especially is SOTA... Sonnet 4.5 as fallback"
    # "openrouter/anthropic/claude-4.5-opus"
    # "anthropic/claude-4.5-sonnet"

    # "openrouter/meta-llama/llama-3.3-70b-instruct"
    # "openrouter/meta-llama/llama-3.2-3b-instruct"
    # "openrouter/qwen/qwen3-30b-a3b"
    # "openrouter/Qwen/Qwen3-8B"

    # "openai/gpt-5-nano-2025-08-07"
    # "openai/gpt-4o-mini-2024-07-18"
    # "openai/gpt-4o-2024-08-06"
    # "openai/gpt-5.1"
    # "openai/gpt-5.2"

)

# uv run python3 -m eval.eval \
#   --model-name openrouter/deepseek/deepseek-v3.2 \
#   --dataset data/gt-harmbench-with-targets.csv \
#   --times 1 \
#   --temperature 1

run_eval() {
    local model=$1
    local temperature=$2
    model_clean=${model##*/}  # Extract model name without provider
    echo "Running eval for model: $model with temperature: $temperature"
    uv run python3 -m eval.eval \
        --model-name "$model" \
        --dataset data/gt-harmbench-with-targets.csv \
        --times 1 \
        --temperature "$temperature" \
        --experiment-name gt-experiment-"$model_clean" \
        --log-dir logs/standard
}

max_jobs=2
# Function to wait if we've reached the max concurrent jobs
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 0.1
    done
}

# Loop through each model and run the evaluation
for model in "${models[@]}"; do
    wait_for_slot
    run_eval "$model" 1
done

# they reccommend temp 1 for deepseek