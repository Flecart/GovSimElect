# These scripts are just to keep in check what kinds of commands are we running.


#!/bin/bash

# Define the list of models using inspect_ai compatible syntax (provider/model)
# Note: For open-source models, 'vllm/' implies a local/remote vLLM server. 
# Change to 'hf/' for local transformers or 'together/' etc. for APIs.

models=(
    # --- Grok ---
    # User Note: "Grok 4.1 there's 3 versions, but only fast available by API"
    # "openrouter/x-ai/grok-4.1-fast"
    # "openrouter/x-ai/grok-4"

    # # --- Gemini ---
    # # User Note: "Gemini 3... thinking of Default" (Pro is usually the default/standard)
    # "google/gemini-3.0-pro"

    # # --- Anthropic ---
    # # User Note: "Opus 4.5 especially is SOTA... Sonnet 4.5 as fallback"
    # "anthropic/claude-4.5-opus"
    # "openrouter/meta-llama/llama-3.3-70b-instruct"
    # "openrouter/meta-llama/llama-3.2-3b-instruct"
    # "openrouter/qwen/qwen3-30b-a3b"
    # "openrouter/Qwen/Qwen3-8B"

    # "openrouter/anthropic/claude-sonnet-4.5"
    # "openrouter/google/gemini-3-flash-preview"
)

models_tmp_1=(
    # "openrouter/meta-llama/llama-3.3-70b-instruct"
    # "openrouter/meta-llama/llama-3.2-3b-instruct"
    # "openai/gpt-5-nano"
    # "openai/gpt-5-mini"
    # "openai/gpt-4o"
    # "openai/gpt-5.1"
    # "openai/gpt-5.2"
    # "openrouter/google/gemini-3-pro-preview"
    "openrouter/anthropic/claude-4.5-opus"
)

# uv run python3 -m eval.eval \
#   --model-name openrouter/deepseek/deepseek-v3.2 \
#   --dataset data/gt-harmbench-with-targets.csv \
#   --times 1 \
#   --temperature 1

# define function that runs eval for a given model and tepmerature

run_eval() {
    local model=$1
    local temperature=$2
    model_clean=${model##*/}  # Extract model name without provider
    echo "Running eval for model: $model with temperature: $temperature"
    uv run python3 -m eval.eval \
        --model-name "$model" \
        --dataset data/gt-harmbench-gamify.csv \
        --times 1 \
        --temperature "$temperature" \
        --experiment-name gamify_experiment-"$model_clean"
}

# Maximum number of concurrent jobs
max_jobs=4

# Function to wait if we've reached the max concurrent jobs
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 0.1
    done
}

# Loop through models_tmp_1 with temperature 1.0
for model in "${models_tmp_1[@]}"; do
    wait_for_slot
    run_eval "$model" 1.0 &
done

# Loop through each model and run the evaluation with temperature 0.7
for model in "${models[@]}"; do
    wait_for_slot
    run_eval "$model" 0.7 &
done

# Wait for all background jobs to complete
wait
