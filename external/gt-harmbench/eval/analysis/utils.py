"""Utility functions for name mapping and formatting."""

from __future__ import annotations

import re

# Model name mapping: full name -> short display name
MODEL_NAME_MAP: dict[str, str] = {
    # OpenAI models
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "openai-gpt-5.4-mini": "GPT-5.4 Mini",
    "gpt-5.4": "GPT-5.4",
    "openai-gpt-5.4": "GPT-5.4",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4": "GPT-4",
    "gpt-3.5-turbo": "GPT-3.5",
    
    # Anthropic models
    "claude-sonnet-4.5": "Sonnet 4.5",
    "anthropic-claude-sonnet-4.5": "Sonnet 4.5",
    "claude-3.5-sonnet": "Claude 3.5",
    "claude-3-opus": "Claude 3 Opus",
    "claude-3-sonnet": "Claude 3 Sonnet",
    
    # Meta models
    "llama-3.2-3b-instruct": "Llama 3.2 3B",
    "llama-3.1-70b-instruct": "Llama 3.1 70B",
    "llama-70b": "Llama 70B",
    "llama-3": "Llama 3",
    
    # Google models
    "gemma-4-31b-it": "Gemma 4 31B",
    "openrouter-google-gemma-4-31b-it": "Gemma 4 31B",
    "gemini-flash": "Gemini Flash",
    "gemini-3-flash-preview": "Gemi 3 Flash",
    "gemini-pro": "Gemini Pro",
    
    # Other models
    "grok": "Grok 4.1 Fast",
    "grok-4.1-fast": "Grok 4.1 Fast",
    "openrouter-x-ai-grok-4.1-fast": "Grok 4.1 Fast",
    "qwen3-30b-a3b": "Qwen 30B",
    "qwen-qwen3-8b": "Qwen 8B",
    "qwen-qwen3-30b": "Qwen 30B",
    "qwen-8b": "Qwen 8B",
    "qwen-30b": "Qwen 30B",
}

# Game type name mapping: full name -> short display name
GAME_TYPE_NAME_MAP: dict[str, str] = {
    "Prisoner's Dilemma": "PD",
    "Chicken": "Chicken",
    "Matching pennies": "MP",
    "Bach or Stravinski": "BoS",
    "Stag hunt": "Stag",
    "Coordination": "Coord",
    "No conflict": "NC",
}

# Game types to exclude from plots
EXCLUDED_GAME_TYPES: set[str] = {
    "Matching pennies",
}


def shorten_model_name(model_name: str) -> str:
    """Convert a model name to a shorter display name.
    
    Args:
        model_name: Full model name (e.g., "gpt-5.1-2025-08-07")
    
    Returns:
        Short display name (e.g., "GPT-5.1")
    """
    if not model_name or model_name == "?":
        return model_name
    
    # Normalize: lowercase and remove common prefixes/suffixes
    normalized = model_name.lower()
    
    # Remove date suffixes (e.g., "-2025-08-07")
    normalized = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', normalized)
    
    # Remove path prefixes
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    
    # Try exact match first
    if normalized in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[normalized]
    
    # Try substring matching
    for key, value in MODEL_NAME_MAP.items():
        if key.lower() in normalized or normalized in key.lower():
            return value
    
    # Fallback: clean up the name
    # Remove common prefixes
    normalized = normalized.replace("anthropic-", "").replace("openai-", "").replace("meta-", "")
    normalized = normalized.replace("_", "-")
    
    # Capitalize words
    parts = normalized.split("-")
    cleaned = " ".join(word.capitalize() for word in parts if word)
    
    return cleaned if cleaned else model_name


def shorten_game_type_name(game_type: str) -> str:
    """Convert a game type name to a shorter display name.
    
    Args:
        game_type: Full game type name (e.g., "Prisoner's Dilemma")
    
    Returns:
        Short display name (e.g., "PD")
    """
    return GAME_TYPE_NAME_MAP.get(game_type, game_type)


def is_game_type_excluded(game_type: str) -> bool:
    """Check if a game type should be excluded from plots.
    
    Args:
        game_type: Game type name
    
    Returns:
        True if the game type should be excluded
    """
    return game_type in EXCLUDED_GAME_TYPES
