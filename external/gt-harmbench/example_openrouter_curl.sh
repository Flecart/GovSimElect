#!/bin/bash

# Example curl script to call Claude Sonnet 4.5 via OpenRouter
# Replace YOUR_OPENROUTER_API_KEY with your actual API key

curl https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -d '{
    "model": "anthropic/claude-3.5-sonnet",
    "messages": [
      {
        "role": "user",
        "content": "Hello! Can you explain what game theory is in one sentence?"
      }
    ],
    "temperature": 1.0,
    "max_tokens": 1024
  }'
