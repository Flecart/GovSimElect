#!/bin/bash

# Define the list of models using inspect_ai compatible syntax (provider/model)
# Note: For open-source models, 'vllm/' implies a local/remote vLLM server. 
# Change to 'hf/' for local transformers or 'together/' etc. for APIs.

models=(
    # --- Grok ---
    # "openrouter/x-ai/grok-4.1-fast"

    # --- Gemini ---
    # "openrouter/google/gemini-3-flash-preview"
    "openrouter/google/gemini-3-flash-preview"

    # --- Anthropic ---
    # "anthropic/claude-4.5-opus"
    # "openrouter/anthropic/claude-4.5-sonnet"

    # --- OpenAI ---
    # "openai/gpt-5-nano-2025-08-07"
    # "openai/gpt-4o-mini-2024-07-18"
    # "openai/gpt-4o-2024-08-06"
    # "openai/gpt-5.1"
    # "openai/gpt-5.2"

    # --- OpenRouter models ---
    # "openrouter/meta-llama/llama-3.3-70b-instruct"
    # "openrouter/meta-llama/llama-3.2-3b-instruct"
    # "openrouter/Qwen/Qwen3-8B"
)

run_eval() {
    local model=$1
    local temperature=$2
    model_clean=${model##*/}  # Extract model name without provider
    echo "Running eval for model: $model with temperature: $temperature"
    uv run python3 -m eval.eval \
        --model-name "$model" \
        --dataset data/gt-harmbench-coordination-gamify.csv \
        --times 1 \
        --temperature "$temperature" \
        --experiment-name coordination-experiment-"$model_clean" \
        --log-dir logs/coordination
}

# Maximum number of concurrent jobs
max_jobs=5
# Function to wait if we've reached the max concurrent jobs
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 0.1
    done
}

# Loop through each model and run the evaluation with temperature 0.7
for model in "${models[@]}"; do
    wait_for_slot
    run_eval "$model" 1 &
done


# run_eval "openrouter/qwen/qwen3-30b-a3b" 0.7 &

# Wait for all background jobs to complete
wait
