#!/usr/bin/env python3
"""
Classify AI risk scenarios according to whether they should be modeled
as game-theoretic scenarios, and, if so, which standard 2x2 games fit.

For each scenario, the script:
1. Applies the KEEP/FILTER ruleset (game-theoretic vs non-game-theoretic).
2. If kept, assesses fit against each formal game in data/game_template.csv.
3. Returns a boolean structure for all 7 formal games.
4. Tracks token usage across all calls.
"""

import asyncio
import csv
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import hydra
from omegaconf import OmegaConf
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.batch_api import (
    create_batch_job,
    poll_batch_job,
    prepare_batch_requests,
    process_batch_results,
    retrieve_batch_results,
)
from src.config import GenerationConfig
import pandas as pd

def read_template_csv(path: str) -> List[Dict[str, str]]:
    """Read the template CSV and return list of game dictionaries."""
    games: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            games.append(row)
    return games


def normalize_game_column_name(game_name: str) -> str:
    """
    Normalize a game name into a safe CSV column suffix.

    Example:
        "Prisoner's Dilemma" -> "prisoners_dilemma"
    """
    normalized = game_name.lower()
    for ch in [" ", "-", "'", "\""]:
        normalized = normalized.replace(ch, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    return normalized


def build_classification_prompt(row: Dict[str, str], games: List[str]) -> str:
    """Build a prompt for classifying a CSV row against the ruleset and games (mirroring classify_taxonomy_games.py)."""
    category = row.get("Risk category", "")
    subcategory = row.get("Risk subcategory", "")
    desc = row.get("Description", "")

    # If games is a list of dicts, extract names and descriptions
    games_summary_lines: List[str] = []
    if games and isinstance(games[0], dict):
        for game in games:
            name = game.get("game_name", "")
            desc_g = game.get("description", "")
            games_summary_lines.append(f'- "{name}": {desc_g}')
    else:
        for name in games:
            games_summary_lines.append(f'- "{name}": (see documentation)')
    games_summary = "\n".join(games_summary_lines)

    prompt = f"""You are an expert in game theory and risk modeling.

Your task is to decide whether a given risk item could be modeled as a game-theoretic scenario, and, if so, which standard 2x2 games from a fixed library are good fits.

RISK ITEM:
- Full path: {category} > {subcategory if subcategory else category}
- Description: {desc}

You must strictly follow this **decision ruleset**:

KEEP IT → if and only if the risk exhibits rule 1:

Rule 1 (most important) — Multi-Actor Strategic Interaction
- *There are two or more actors*, and each actor’s payoff depends on others’ choices. → keep.
- If actions are interdependent → keep.
- For our purposes, 'No conflict' games still count as game theoretic. Just mark them as No conflict (and other games if applicable).

In general, the conflict games are Prisoner's Dilemma, Chicken, Matching Pennies. Coordination games  are Bach or Stravinski (also known as Battle of the Sexes), Coordination and Stag Hunt. Make sure to consider **each possible game**, regardless of popularity, for each scenario. Often, multiple games will fit, especially similar ones. 

FILTER OUT → if any of the following is true:

Rule A — Pure Single-Actor Quality Failure
- Risk arises from sloppiness, poor documentation, missing testing, etc., with no interacting party.

Rule B — Purely Technical Reliability Issue
- Corner cases, hallucinations, factuality errors, etc. with no other agents involved.
- No agent “plays” against another → filter out.

Rule C — Pure Compliance Restrictions
- Pure data/usage/privacy law constraints with no strategic behavior.
- Unless actors strategically try to bypass them; then go back to Rule 1.

Rule D — Pure Value Judgement Statements
- High-level normative claims with no defined actors or incentives → filter out.

Now, for items that you KEEP as game-theoretic, assess fit to the following
seven 2x2 game templates (do NOT invent new game types):

{games_summary}

TASK:
1. Decide a single boolean: should_be_game_theoretic
   - true  = keep (satisfies at least one KEEP rule).
   - false = filter out (falls under FILTER rules; no genuine strategic interaction).
2. For each of the seven games above, decide a boolean fit:
   - true  = this game is a plausible or natural modeling choice.
   - false = this game is not an appropriate core structure.
   Even if should_be_game_theoretic is false, you must still return booleans
   for all seven games (usually all false in that case).
3. Provide a short textual justification (≤220 characters) summarizing your decision,
   focused on incentives, number of actors, and strategic structure.

In general, be strict about should_be_game_theoretic: only keep if genuine
strategic interaction exists per the ruleset above. However, if kept, be generous about game fit: multiple games may fit simultaneously. We will filter game-type items again later. 

RESPONSE FORMAT (MUST be valid JSON, no extra keys, no comments):
{{
  "should_be_game_theoretic": true or false,
  "fits_games": {{
    "Prisoner's Dilemma": true or false,
    "Chicken": true or false,
    "Bach or Stravinski": true or false,
    "No conflict": true or false,
    "Stag hunt": true or false,
    "Coordination": true or false,
    "Matching pennies": true or false
  }},
  "justification": "<=220 characters explaining keep/filter and game fit"
}}

Do not include any fields other than those in this JSON schema.
Ensure the final output is strictly valid JSON.
"""
    return prompt

def classify_all_rows_batch(
    client: OpenAI,
    risk_df: pd.DataFrame,
    games: List[Dict[str, str]],
    cfg: GenerationConfig,
) -> Optional[Tuple[pd.DataFrame, int, int]]:
    """Run classification over all taxonomy leaves using Batch API."""
    total_items = len(risk_df)
    print(f"\nPlanned classifications (batch mode, one per scenario): {total_items}")

    risk_df_classified = risk_df.copy()


    # Prepare batch requests
    batch_requests: List[Dict[str, Any]] = []
    for _, row in risk_df.iterrows():
        prompt = build_classification_prompt(row, games)
        request = {
            "model": cfg.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in game theory and AI safety risk taxonomies. "
                        "Apply the provided ruleset exactly, and respond ONLY with the "
                        "requested JSON structure."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        batch_requests.append(request)

    # Create temporary batch file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        batch_file_path = f.name
    try:
        prepare_batch_requests(batch_requests, batch_file_path)
        batch_id = create_batch_job(client, batch_file_path)
        print(f"Batch job created: {batch_id}")
        
        # Save batch ID to file for later retrieval
        batch_id_file = cfg.output_csv_path.replace(".csv", "_batch_id.txt")
        with open(batch_id_file, "w", encoding="utf-8") as f:
            f.write(batch_id)
        print(f"Batch ID saved to: {batch_id_file}")
        
        if cfg.skip_batch_polling:
            print("\nSkipping polling (skip_batch_polling=true)")
            print("Use retrieve_classification_batches.py to process results later:")
            print(f"  python scripts/retrieve_classification_batches.py batch_id_file={batch_id_file}")
            return None
        
        print("Polling for batch completion...")
        output_file_id = poll_batch_job(client, batch_id)
        batch_results = retrieve_batch_results(client, output_file_id)
        parsed_responses, total_tokens_in, total_tokens_out = process_batch_results(
            batch_results, total_items
        )
    finally:
        # Clean up batch file
        if os.path.exists(batch_file_path):
            os.unlink(batch_file_path)

    # Process parsed responses into classification results
    for i, parsed_response in enumerate(parsed_responses):
        if parsed_response is None:
            continue

        content = parsed_response.get("content", "")
        if not content:
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            continue

        risk_df_classified.at[i, "should_be_game_theoretic"] = data.get("should_be_game_theoretic", False)
        risk_df_classified.at[i, "justification"] = data.get("justification", "")
        for game in games:
            game_name = game.get("game_name", "")
            col_suffix = normalize_game_column_name(game_name)
            col_name = f"fits_{col_suffix}"
            fits_games = data.get("fits_games", {}) or {}
            risk_df_classified.at[i, col_name] = fits_games.get(game_name, False)


    print(f"\nSuccessfully classified {len(risk_df_classified)} taxonomy leaves (batch mode)")
    return risk_df_classified, total_tokens_in, total_tokens_out

def print_token_report(stage: str, num_items: int, tokens_in: int, tokens_out: int) -> None:
    """Pretty print token usage for a pipeline stage."""
    print("\n" + "=" * 60)
    print(f"{stage} Token Usage Report")
    print("=" * 60)
    print(f"Items processed:       {num_items}")
    print(f"Total Input Tokens:    {tokens_in:,}")
    print(f"Total Output Tokens:   {tokens_out:,}")
    print(f"Total Tokens:          {tokens_in + tokens_out:,}")
    if num_items > 0:
        print(f"Avg Input per item:    {tokens_in / num_items:,.0f}")
        print(f"Avg Output per item:   {tokens_out / num_items:,.0f}")
    print("=" * 60)


async def async_main(cfg: GenerationConfig) -> None:
    """
    Main function to run taxonomy classification.
    
    Supports two modes:
    - Async mode (default): Concurrent API requests using asyncio
    - Batch mode (use_batch_api=true): Batched requests using OpenAI Batch API
    """
    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set.")

    # Read MIT
    print("Reading MIT Risk Items...")
    allowed = ["Risk Category", "Risk Sub-Category"]
    risks_df = pd.read_csv(cfg.taxonomy_path)
    risks_df = risks_df[risks_df["Category level"].isin(allowed)]
    
    print(f"Found {len(risks_df)} risk items")

    # Read game templates
    print("Reading game templates...")
    games = read_template_csv(cfg.template_csv_path)
    print(f"Found {len(games)} game templates")

    total_items = len(risks_df)
    if cfg.dry_run:
        print(
            f"[DRY_RUN] Would classify {total_items} taxonomy leaves "
            f"using model {cfg.llm_model}."
        )
        return

    print("\n=== Classification Stage ===")
    if cfg.use_batch_api:
        print("Using OpenAI Batch API mode (batched requests instead of concurrent)")
        print("  - 50% cost reduction compared to standard API")
        print("  - Jobs process asynchronously (may take minutes to hours)")
        # Batch API requires synchronous OpenAI client (not available via OpenRouter)
        if os.environ.get("OPENROUTER_API_KEY"):
            print("  WARNING: Batch API is OpenAI-only. Falling back to OpenAI API key.")
        client = OpenAI(api_key=api_key)
        result = classify_all_rows_batch(client, risks_df, games, cfg)
        if result is None:
            # Batch was submitted but polling was skipped
            print("\nBatch job submitted. Use retrieve_classification_batches.py to process results later.")
            return
        risks_df_classified, tokens_in, tokens_out = result
    else:
        raise NotImplementedError('Switched to pandas DataFrame for batch mode; no batch not implemented yet.')

    print_token_report("Classification", len(risks_df_classified), tokens_in, tokens_out)
    risks_df_classified = risks_df_classified[['Ev_ID', 'Risk category', 'Risk subcategory', 'Description', 'should_be_game_theoretic', 'justification', 'fits_prisoner_s_dilemma', 'fits_chicken', 'fits_bach_or_stravinski', 'fits_no_conflict', 'fits_stag_hunt', 'fits_coordination', 'fits_matching_pennies']]
    print("\n=== Export Stage ===")
    risks_df_classified.to_csv(
        cfg.output_csv_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )


@hydra.main(version_base=None, config_path="../config", config_name="classify_taxonomy_games")
def main(cfg) -> None:
    """Hydra entry point that wires config into the async classification pipeline."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    gen_cfg = GenerationConfig(**cfg_dict)  # type: ignore[arg-type]
    gen_cfg.validate()
    asyncio.run(async_main(gen_cfg))


if __name__ == "__main__":
    main()



