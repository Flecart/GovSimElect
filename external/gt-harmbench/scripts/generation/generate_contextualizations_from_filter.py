#!/usr/bin/env python3
"""
Generate contextualizations only for AI risk scenarios that
have been approved by the game-type classifier.

Pipeline:
1. Read games classification CSV (outputs of classify_csv_games.py).
2. For each scenario marked `should_be_game_theoretic = true`, and for each
   formal game with a corresponding `fits_*` boolean set to true,
   create a (scenario, game) combination.
3. Use the existing contextualization generation + evaluation pipeline
   (from generate_contextualizations_from_taxonomy.py) to:
   - Generate scenarios,
   - Evaluate them with the rubric,
   - Filter by thresholds,
   - Track token usage,
   - Export to CSV.
"""

import asyncio
import os
import sys
import json
from typing import Any, Dict, List, Tuple
import pandas as pd

import hydra
from omegaconf import OmegaConf
from openai import OpenAI, AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import GenerationConfig

from scripts.generate_contextualizations_from_taxonomy import (  # type: ignore[import]
    read_template_csv,
    evaluate_contextualizations_batch,
    filter_contextualizations,
    print_token_report,
    print_evaluation_summary,
    show_failed_samples,
    build_prompt,
)


def normalize_game_column_name(game_name: str) -> str:
    """
    Normalize a game name into the suffix used in the classification CSV.

    Must match the logic in scripts/classify_taxonomy_games.py.
    """
    normalized = game_name.lower()
    for ch in [" ", "-", "'", "\""]:
        normalized = normalized.replace(ch, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    return normalized



async def generate_contextualization(
    client: OpenAI,
    unwrapped_df: pd.DataFrame,
    games: List[Dict[str, str]],
    cfg: GenerationConfig,
) -> Tuple[List[Dict[str, Any]], int, int]:
    total = len(unwrapped_df)
    print(f"\nPlanned contextualizations (filtered, non-batch mode): {total}")
    if total == 0:
        return [], 0, 0
    global GENERATION_PROMPT
    if 'GENERATION_PROMPT' not in globals():
        with open(os.path.join(os.path.dirname(__file__), "./generation_prompt.md"), "r", encoding="utf-8") as f:
            GENERATION_PROMPT = f.read()


    # Prepare batch requests
    all_requests: List[Dict[str, Any]] = []
    for _, row in unwrapped_df.iterrows():
        # Find the game dict where game_name matches row['formal_game']
        game_dict = next((g for g in games if g.get('game_name') == row['formal_game']), None)
        if game_dict is None:
            print(f"Warning: No game found for {row['formal_game']}")
            continue
        prompt = build_prompt(row, game_dict)
        request = {
            "model": cfg.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": GENERATION_PROMPT
                },
                {"role": "user", "content": prompt}
            ],
            "verbosity": "low",
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
        }
        all_requests.append(request)


    semaphore = asyncio.Semaphore(cfg.concurrency)
    tasks = []
    
    async def process_samples(requests: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
        parsed_responses: List[Dict[str, Any]] = []
        total_tokens_in = 0
        total_tokens_out = 0

        for req in requests:
            async with semaphore:
                response = await client.chat.completions.create(**req)
                parsed_responses.append(response.choices[0].message)
                usage = response.usage
                total_tokens_in += usage.prompt_tokens
                total_tokens_out += usage.completion_tokens

        return parsed_responses, total_tokens_in, total_tokens_out

    result = await process_samples(all_requests)
    parsed_responses, total_tokens_in, total_tokens_out = result

    # Prepare a DataFrame to match unwrapped_df, with new columns for LLM results
    result_df = unwrapped_df.copy()
    # Define the new columns to add
    new_cols = [
        "story_row", "story_col", "actions_row", "actions_column",
        "1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff",
        "game_description", "risk_level"
    ]
    for col in new_cols:
        result_df[col] = None

    for i, parsed_response in enumerate(parsed_responses):
        if parsed_response is None:
            continue
        content = parsed_response.get("content", "")
        if not content:
            continue
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"\nJSON decode error for row {i}: {e}")
            continue
        # Fill in the new columns for this row
        result_df.at[i, "story_row"] = data.get("story_row", "").strip()
        result_df.at[i, "story_col"] = data.get("story_col", "").strip()
        result_df.at[i, "actions_row"] = data.get("actions_row", [])
        result_df.at[i, "actions_column"] = data.get("actions_column", [])
        result_df.at[i, "1_1_payoff"] = data.get("1_1_payoff", [])
        result_df.at[i, "1_2_payoff"] = data.get("1_2_payoff", [])
        result_df.at[i, "2_1_payoff"] = data.get("2_1_payoff", [])
        result_df.at[i, "2_2_payoff"] = data.get("2_2_payoff", [])
        # Optionally, fill in game_description and risk_level if present
        result_df.at[i, "game_description"] = data.get("game_description", "")
        result_df.at[i, "risk_level"] = data.get("risk_level", None)

    print(f"\nSuccessfully generated {result_df[new_cols[0]].notnull().sum()} contextualizations (filtered, batch mode)")
    return result_df, total_tokens_in, total_tokens_out


def unwrap_allowed(
    risks_df_game: pd.DataFrame,
    game_templates: pd.DataFrame,
) -> pd.DataFrame:
    """
    From the classified games CSV, extract only the (leaf, game) pairs
    where `should_be_game_theoretic` is true and `fits_*` is true.

    Returns an unwrapped DataFrame with one row per (scenario, game) pair.
    """
    
    fit_to_name = {}
    for _, row in game_templates.iterrows():
        game_name = row["game_name"]
        suffix = normalize_game_column_name(game_name)
        col_name = f"fits_{suffix}"
        fit_to_name[col_name] = game_name

    exploded_templates = risks_df_game.melt(
        id_vars=[col for col in risks_df_game.columns if not col.startswith("fits_")],
        value_vars=[col for col in risks_df_game.columns if col.startswith("fits_")],
        var_name="formal_game",
        value_name="fits_game",
    )
    # filter only the rows where fits_game is True
    exploded_templates.head()
    # drop nas
    exploded_templates = exploded_templates.dropna(subset=["fits_game"]).reset_index(drop=True)
    exploded_templates = exploded_templates[exploded_templates["fits_game"]].reset_index(drop=True)
    # drop fits_game column
    exploded_templates = exploded_templates.drop(columns=["fits_game"])
    
    
    exploded_templates["formal_game"] = exploded_templates["formal_game"].map(fit_to_name)
    return exploded_templates

def generate_batch(
    client: OpenAI,
    unwrapped_df: pd.DataFrame,
    games: List[Dict[str, str]],
    cfg: GenerationConfig,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Generate contextualizations only for the pre-selected games using Batch API.
    """
    total = len(unwrapped_df)
    print(f"\nPlanned contextualizations (filtered, batch mode): {total}")
    if total == 0:
        return [], 0, 0

    # Process pairs directly using batch API
    import json
    import tempfile
    from src.batch_api import (
        create_batch_job,
        poll_batch_job,
        prepare_batch_requests,
        process_batch_results,
        retrieve_batch_results,
    )
    
    global GENERATION_PROMPT
    if 'GENERATION_PROMPT' not in globals():
        with open(os.path.join(os.path.dirname(__file__), "./generation_prompt.md"), "r", encoding="utf-8") as f:
            GENERATION_PROMPT = f.read()


    # Prepare batch requests
    batch_requests: List[Dict[str, Any]] = []
    for _, row in unwrapped_df.iterrows():
        # Find the game dict where game_name matches row['formal_game']
        game_dict = next((g for g in games if g.get('game_name') == row['formal_game']), None)
        if game_dict is None:
            print(f"Warning: No game found for {row['formal_game']}")
            continue
        prompt = build_prompt(row, game_dict)
        request = {
            "model": cfg.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": GENERATION_PROMPT
                },
                {"role": "user", "content": prompt}
            ],
            "verbosity": "low",
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
        }
        batch_requests.append(request)

    # Create temporary batch file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        batch_file_path = f.name

    try:
        # prepare_batch_requests(batch_requests, batch_file_path)
        batch_id = "batch_6956ae6e7c488190be92f1b687270caa" # create_batch_job(client, batch_file_path)
        output_file_id = poll_batch_job(client, batch_id, poll_interval=10)
        batch_results = retrieve_batch_results(client, output_file_id)
        parsed_responses, total_tokens_in, total_tokens_out = process_batch_results(
            batch_results, total
        )
    finally:
        if os.path.exists(batch_file_path):
            os.unlink(batch_file_path)

    # Process parsed responses into contextualization results

    # Prepare a DataFrame to match unwrapped_df, with new columns for LLM results
    result_df = unwrapped_df.copy()
    # Define the new columns to add
    new_cols = [
        "story_row", "story_col", "actions_row", "actions_column",
        "1_1_payoff", "1_2_payoff", "2_1_payoff", "2_2_payoff",
        "game_description", "risk_level"
    ]
    for col in new_cols:
        result_df[col] = None

    for i, parsed_response in enumerate(parsed_responses):
        if parsed_response is None:
            continue
        content = parsed_response.get("content", "")
        if not content:
            continue
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"\nJSON decode error for row {i}: {e}")
            continue
        # Fill in the new columns for this row
        result_df.at[i, "story_row"] = data.get("story_row", "").strip()
        result_df.at[i, "story_col"] = data.get("story_col", "").strip()
        result_df.at[i, "actions_row"] = data.get("actions_row", [])
        result_df.at[i, "actions_column"] = data.get("actions_column", [])
        result_df.at[i, "1_1_payoff"] = data.get("1_1_payoff", [])
        result_df.at[i, "1_2_payoff"] = data.get("1_2_payoff", [])
        result_df.at[i, "2_1_payoff"] = data.get("2_1_payoff", [])
        result_df.at[i, "2_2_payoff"] = data.get("2_2_payoff", [])
        # Optionally, fill in game_description and risk_level if present
        result_df.at[i, "game_description"] = data.get("game_description", "")
        result_df.at[i, "risk_level"] = data.get("risk_level", None)

    print(f"\nSuccessfully generated {result_df[new_cols[0]].notnull().sum()} contextualizations (filtered, batch mode)")
    return result_df, total_tokens_in, total_tokens_out


async def async_main(cfg: GenerationConfig) -> None:
    """Main function: generate contextualizations guided by taxonomy filter outputs."""
    print("Debug config:", cfg)    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set.")

    if not cfg.classification_csv_path:
        raise ValueError(
            "classification_csv_path must be set in the Hydra config for "
            "generate_contextualizations_from_filter.py."
        )

    # Parse CSV
    print("Reading risk scenarios with game classifications...")
    risk_df_games = pd.read_csv(cfg.classification_csv_path)
    # Read game templates
    print("Reading game templates...")
    game_templates = pd.read_csv("data/game_template.csv")
    games = read_template_csv("data/game_template.csv")  # kept for compatibility
    
    print(f"Found {len(game_templates)} game templates")

    # Select allowed (leaf, game) pairs
    risk_df_unwrapped = unwrap_allowed(risk_df_games, game_templates)
    
    # limit to 10 cases
    # sandomly sample 10 cases for testing
    # risk_df_unwrapped = risk_df_unwrapped.sample(n=2, random_state=42).reset_index(drop=True)
    # risk_df_unwrapped = risk_df_unwrapped.drop(risk_df_unwrapped_exclude.index).reset_index(drop=True)

    print(f"Total allowed contextualizations: {len(risk_df_unwrapped)}")

    if cfg.dry_run:
        print(
            f"[DRY_RUN] Would generate contextualizations for "
            f"{len(risk_df_unwrapped)} games using model {cfg.llm_model}."
        )
        return

    print("\n=== Generation Stage (Filtered) ===")
    if cfg.use_batch_api:
        client = OpenAI(api_key=api_key)
        print("Using Batch API mode")
        # Batch API requires synchronous client
        if os.environ.get("OPENROUTER_API_KEY"):
            print("Note: Batch API is OpenAI-only, not available via OpenRouter")
        else:
            contextualizations_df, gen_tokens_in, gen_tokens_out = generate_batch(
                client, risk_df_unwrapped, games, cfg        
            )
    else:
        client = AsyncOpenAI(api_key=api_key)
        print("Using Asynchronous mode")
        contextualizations_df, gen_tokens_in, gen_tokens_out = await generate_contextualization(
            client, risk_df_unwrapped, games, cfg        
        )

    print_token_report("Generation (Filtered)", len(contextualizations_df), gen_tokens_in, gen_tokens_out)

    if contextualizations_df.empty:
        print("No contextualizations were generated; aborting.")
        return

    print("\n=== Evaluation Stage ===")
    if cfg.use_batch_api:
        evaluations, eval_tokens_in, eval_tokens_out = evaluate_contextualizations_batch(
            client, contextualizations_df, cfg
        )
    else:
        print("Asynchronous evaluation mode is not implemented in this script.")
        evaluations, eval_tokens_in, eval_tokens_out = evaluate_contextualizations_batch(
            client, contextualizations_df, cfg
        )
        # raise NotImplementedError("Asynchronous mode is not implemented in this script.")

    print_token_report("Evaluation", len(contextualizations_df), eval_tokens_in, eval_tokens_out)

    filtered_results_df, failed_results_df, quality_scores, equilibria_scores = filter_contextualizations(
        contextualizations_df, evaluations, cfg
    )

    print_evaluation_summary(filtered_results_df, failed_results_df, quality_scores, equilibria_scores, cfg)
    show_failed_samples(failed_results_df, cfg)

    print("\n=== Export Stage ===")
    filtered_results_df.to_csv(cfg.output_csv_path, index=True, index_label="id")
    
    # Save failed contextualizations to a separate CSV
    if not failed_results_df.empty:
        # Derive the bad CSV path from the output path
        output_dir = os.path.dirname(cfg.output_csv_path)
        output_basename = os.path.basename(cfg.output_csv_path)
        # Replace .csv with -bad.csv, or append -bad.csv if no extension
        if output_basename.endswith('.csv'):
            bad_csv_path = os.path.join(output_dir, output_basename[:-4] + '-bad.csv')
        else:
            bad_csv_path = os.path.join(output_dir, output_basename + '-bad.csv')
        print(f"\nSaving {len(failed_results_df)} failed contextualizations to {bad_csv_path}")
        failed_results_df.to_csv(bad_csv_path, index=False)


@hydra.main(version_base=None, config_path="../config", config_name="generate_contextualizations_filtered")
def main(cfg) -> None:
    """Hydra entry point for filtered contextualization generation."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    gen_cfg = GenerationConfig(**cfg_dict)  # type: ignore[arg-type]
    gen_cfg.validate()
    asyncio.run(async_main(gen_cfg))


if __name__ == "__main__":
    main()



