#!/bin/bash

# Define the list of models using inspect_ai compatible syntax (provider/model)
# Note: For open-source models, 'vllm/' implies a local/remote vLLM server. 
# Change to 'hf/' for local transformers or 'together/' etc. for APIs.

models=(
    # --- Grok ---
    # User Note: "Grok 4.1 there's 3 versions, but only fast available by API"
    # "grok/grok-4.1-fast"

    # # --- Gemini ---
    # # User Note: "Gemini 3... thinking of Default" (Pro is usually the default/standard)
    # "google/gemini-3.0-pro"

    # # --- Anthropic ---
    # # User Note: "Opus 4.5 especially is SOTA... Sonnet 4.5 as fallback"
    # "anthropic/claude-4.5-opus"
    # "anthropic/claude-4.5-sonnet"

    "openrouter/anthropic/claude-sonnet-4.5"
    "openrouter/google/gemini-3-flash-preview"
)

# Loop through each model and run the evaluation
for model in "${models[@]}"; do
    echo "Running eval for model: $model"
    uv run python3 -m eval.eval \
        --model-name "$model" \
        --dataset data/gt-harmbench-with-targets.csv \
        --times 1 \
        --temperature 1 &
done
wait

# they reccommend temp 1 for deepseek