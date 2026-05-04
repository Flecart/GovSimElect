#!/bin/bash

# Test script for GPT-5-mini API request
# Make sure to set your OPENAI_API_KEY environment variable

API_KEY="${OPENAI_API_KEY}"

if [ -z "$API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "gpt-5-mini",
    "messages": [
      {
        "role": "user",
        "content": "Hello! This is a test message. Please respond with a short greeting. You need to say HELLO"
      }
    ],
    "temperature": 1
  }'
