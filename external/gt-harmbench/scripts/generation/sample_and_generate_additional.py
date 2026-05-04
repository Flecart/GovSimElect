#!/usr/bin/env python3
"""
Sample additional contextualizations from taxonomy_game_fit.csv where games were marked false.

This script:
1. Counts current examples per game in contextualization-filtered.csv
2. Samples from taxonomy_game_fit.csv where fits_* = false but should_be_game_theoretic = true
3. Generates contextualizations for sampled pairs
4. Appends results to contextualization-filtered.csv
"""

import asyncio
import csv
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

import hydra
from omegaconf import OmegaConf
from openai import AsyncOpenAI, OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import GenerationConfig
from src.parser import TaxonomyParser, TaxonomyNode

from scripts.generate_contextualizations_from_taxonomy import (  # type: ignore[import]
    read_template_csv,
    generate_contextualization,
    evaluate_contextualizations,
    evaluate_contextualizations_batch,
    filter_contextualizations,
    print_token_report,
    print_evaluation_summary,
    show_failed_samples,
    write_results_to_csv,
    build_prompt,
)
from tqdm import tqdm


def normalize_game_column_name(game_name: str) -> str:
    """Normalize a game name into the suffix used in the classification CSV."""
    normalized = game_name.lower()
    for ch in [" ", "-", "'", "\""]:
        normalized = normalized.replace(ch, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    return normalized


def str_to_bool(value: str) -> bool:
    """Robust string?bool conversion for CSV fields."""
    v = (value or "").strip().lower()
    return v in {"1", "true", "yes", "y", "t"}


def load_classification_csv(path: str) -> List[Dict[str, str]]:
    """Load the taxonomy/game classification CSV."""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def count_existing_examples(csv_path: str) -> Counter:
    """Count existing examples per game in the contextualization CSV."""
    counter = Counter()
    if not os.path.exists(csv_path):
        return counter
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_name = row.get("formal_game", "")
            if game_name:
                counter[game_name] += 1
    return counter


def build_leaf_lookup(leaf_nodes: List[TaxonomyNode]) -> Dict[Tuple[str, str], TaxonomyNode]:
    """Build a lookup from (taxonomy_path, leaf_name) to TaxonomyNode."""
    mapping: Dict[Tuple[str, str], TaxonomyNode] = {}
    for leaf in leaf_nodes:
        path = leaf.get_path_string(" > ")
        mapping[(path, leaf.name)] = leaf
    return mapping


def select_false_pairs_for_game(
    classification_rows: List[Dict[str, str]],
    games: List[Dict[str, str]],
    leaf_lookup: Dict[Tuple[str, str], TaxonomyNode],
    target_game_name: str,
    count_needed: int,
) -> List[Tuple[TaxonomyNode, Dict[str, str]]]:
    """
    Select pairs where the game was marked as false but should_be_game_theoretic = true.
    Returns up to count_needed pairs.
    """
    target_game = None
    for game in games:
        if game.get("game_name", "") == target_game_name:
            target_game = game
            break
    
    if target_game is None:
        return []
    
    suffix = normalize_game_column_name(target_game_name)
    col_name = f"fits_{suffix}"
    
    candidate_pairs: List[Tuple[TaxonomyNode, Dict[str, str]]] = []
    
    for row in classification_rows:
        taxonomy_path = row.get("taxonomy_path", "")
        taxonomy_leaf = row.get("taxonomy_leaf", "")
        key = (taxonomy_path, taxonomy_leaf)
        leaf = leaf_lookup.get(key)
        if leaf is None:
            continue
        
        should_be_game = str_to_bool(row.get("should_be_game_theoretic", "false"))
        if not should_be_game:
            continue
        
        # Check if this game was marked as false
        if col_name not in row:
            continue
        
        if not str_to_bool(row.get(col_name, "false")):
            candidate_pairs.append((leaf, target_game))
    
    # Randomly sample to reach count_needed
    if len(candidate_pairs) <= count_needed:
        return candidate_pairs
    
    return random.sample(candidate_pairs, count_needed)


async def generate_from_pairs(
    client: AsyncOpenAI,
    pairs: List[Tuple[TaxonomyNode, Dict[str, str]]],
    cfg: GenerationConfig,
    semaphore: asyncio.Semaphore,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Generate contextualizations for given pairs."""
    total = len(pairs)
    if total == 0:
        return [], 0, 0
    
    pbar = tqdm(total=total, desc="Generating additional contextualizations")
    tasks: List[asyncio.Task] = []
    
    for leaf, game in pairs:
        task = asyncio.create_task(
            generate_contextualization(client, leaf, game, cfg, semaphore, pbar)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    pbar.close()
    
    valid_results: List[Dict[str, Any]] = []
    total_tokens_in = 0
    total_tokens_out = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"\nException in generation task {i}: {result}")
            continue
        
        if isinstance(result, tuple) and len(result) == 3:
            result_data, tokens_in, tokens_out = result
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out
            if result_data is not None:
                valid_results.append(result_data)
        elif result is not None:
            valid_results.append(result)
    
    return valid_results, total_tokens_in, total_tokens_out


def generate_from_pairs_batch(
    client: OpenAI,
    pairs: List[Tuple[TaxonomyNode, Dict[str, str]]],
    cfg: GenerationConfig,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Generate contextualizations using Batch API."""
    total = len(pairs)
    if total == 0:
        return [], 0, 0
    
    import tempfile
    from src.batch_api import (
        create_batch_job,
        poll_batch_job,
        prepare_batch_requests,
        process_batch_results,
        retrieve_batch_results,
    )
    
    batch_requests: List[Dict[str, Any]] = []
    for leaf, game in pairs:
        prompt = build_prompt(leaf, game)
        request = {
            "model": cfg.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in game theory and ethical decision-making. Create realistic, engaging scenarios that match the specified payoff structures while addressing important real-world topics."
                },
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
        }
        batch_requests.append(request)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        batch_file_path = f.name
    
    try:
        prepare_batch_requests(batch_requests, batch_file_path)
        batch_id = create_batch_job(client, batch_file_path)
        output_file_id = poll_batch_job(client, batch_id)
        batch_results = retrieve_batch_results(client, output_file_id)
        parsed_responses, total_tokens_in, total_tokens_out = process_batch_results(
            batch_results, total
        )
    finally:
        if os.path.exists(batch_file_path):
            os.unlink(batch_file_path)
    
    valid_results: List[Dict[str, Any]] = []
    for i, parsed_response in enumerate(parsed_responses):
        if parsed_response is None:
            continue
        
        content = parsed_response.get("content", "")
        if not content:
            continue
        
        leaf, game = pairs[i]
        try:
            data = json.loads(content)
            story_row = data.get("story_row", "").strip()
            story_col = data.get("story_col", "").strip()
            
            result = {
                "formal_game": game.get("game_name", ""),
                "taxonomy_path": leaf.get_path_string(" > "),
                "taxonomy_leaf": leaf.name,
                "story_row": story_row,
                "story_col": story_col,
                "actions_row": data.get("actions_row", []),
                "actions_column": data.get("actions_column", []),
                "1_1_payoff": data.get("1_1_payoff", []),
                "1_2_payoff": data.get("1_2_payoff", []),
                "2_1_payoff": data.get("2_1_payoff", []),
                "2_2_payoff": data.get("2_2_payoff", []),
                "game_description": game.get("description", ""),
            }
            valid_results.append(result)
        except json.JSONDecodeError as e:
            print(f"\nJSON decode error for {leaf.name} + {game.get('game_name', '')}: {e}")
            continue
    
    return valid_results, total_tokens_in, total_tokens_out


def append_to_csv(new_results: List[Dict[str, Any]], csv_path: str) -> None:
    """Append new results to existing CSV, preserving existing IDs and continuing numbering."""
    # Read existing CSV to get max ID
    max_id = 0
    fieldnames = None
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                try:
                    row_id = int(row.get("id", 0))
                    max_id = max(max_id, row_id)
                except ValueError:
                    pass
    
    if not new_results:
        return
    
    # Use fieldnames from new results if CSV doesn't exist
    if fieldnames is None:
        fieldnames = ["id"]
        fieldnames.extend(new_results[0].keys())
    
    # Append new rows
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        for i, result in enumerate(new_results, 1):
            row = {"id": max_id + i}
            row.update(result)
            writer.writerow(row)


async def async_main(cfg: GenerationConfig) -> None:
    """Main function: sample and generate additional contextualizations."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set.")
    
    target_games = ["Coordination", "Battle of the Sexes", "No conflict"]
    target_count = 100
    
    # Count existing examples
    print("Counting existing examples...")
    existing_counts = count_existing_examples(cfg.output_csv_path)
    print(f"Current counts per game:")
    for game in target_games:
        count = existing_counts.get(game, 0)
        print(f"  {game}: {count}")
    
    # Parse taxonomy
    print("\nParsing taxonomy...")
    parser = TaxonomyParser(cfg.taxonomy_path)
    parser.parse()
    leaf_nodes = parser.get_leaf_nodes()
    leaf_lookup = build_leaf_lookup(leaf_nodes)
    
    # Read game templates
    print("Reading game templates...")
    games = read_template_csv(cfg.template_csv_path)
    
    # Read classification CSV
    print("Reading taxonomy/game classification CSV...")
    classification_rows = load_classification_csv(cfg.classification_csv_path)
    
    # Sample pairs for each target game
    all_pairs_to_generate: List[Tuple[str, List[Tuple[TaxonomyNode, Dict[str, str]]]]] = []
    
    for game_name in target_games:
        existing = existing_counts.get(game_name, 0)
        needed = max(0, target_count - existing)
        
        if needed > 0:
            print(f"\nSampling {needed} additional pairs for {game_name}...")
            pairs = select_false_pairs_for_game(
                classification_rows, games, leaf_lookup, game_name, needed
            )
            print(f"Found {len(pairs)} candidate pairs for {game_name}")
            all_pairs_to_generate.append((game_name, pairs))
        else:
            print(f"\n{game_name} already has {existing} examples (target: {target_count})")
    
    if not all_pairs_to_generate:
        print("\nNo additional contextualizations needed!")
        return
    
    # Generate contextualizations
    print("\n=== Generation Stage ===")
    if cfg.use_batch_api:
        print("Using Batch API mode")
        client = OpenAI(api_key=api_key)
        all_new_results: List[Dict[str, Any]] = []
        
        for game_name, pairs in all_pairs_to_generate:
            if not pairs:
                continue
            print(f"\nGenerating for {game_name}...")
            results, _, _ = generate_from_pairs_batch(client, pairs, cfg)
            all_new_results.extend(results)
            print(f"Generated {len(results)} contextualizations for {game_name}")
    else:
        print("Using async API mode")
        if os.environ.get("OPENROUTER_API_KEY"):
            print("Using OpenRouter")
            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            print("Using OpenAI")
            client = AsyncOpenAI(api_key=api_key)
        
        semaphore = asyncio.Semaphore(cfg.concurrency)
        all_new_results: List[Dict[str, Any]] = []
        
        for game_name, pairs in all_pairs_to_generate:
            if not pairs:
                continue
            print(f"\nGenerating for {game_name}...")
            results, _, _ = await generate_from_pairs(client, pairs, cfg, semaphore)
            all_new_results.extend(results)
            print(f"Generated {len(results)} contextualizations for {game_name}")
    
    if not all_new_results:
        print("No contextualizations were generated.")
        return
    
    # Evaluate contextualizations
    print("\n=== Evaluation Stage ===")
    if cfg.use_batch_api:
        evaluations, eval_tokens_in, eval_tokens_out = evaluate_contextualizations_batch(
            client, all_new_results, cfg
        )
    else:
        evaluations, eval_tokens_in, eval_tokens_out = await evaluate_contextualizations(
            client, all_new_results, cfg, semaphore
        )
    print_token_report("Evaluation", len(all_new_results), eval_tokens_in, eval_tokens_out)
    
    filtered_results, failed_results, quality_scores, equilibria_scores = filter_contextualizations(
        all_new_results, evaluations, cfg
    )
    print_evaluation_summary(filtered_results, failed_results, quality_scores, equilibria_scores, cfg)
    
    # Append filtered results to existing CSV
    print("\n=== Append Stage ===")
    print(f"Appending {len(filtered_results)} new contextualizations to {cfg.output_csv_path}")
    append_to_csv(filtered_results, cfg.output_csv_path)
    print("Done!")


@hydra.main(version_base=None, config_path="../config", config_name="generate_contextualizations_filtered")
def main(cfg) -> None:
    """Hydra entry point."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    gen_cfg = GenerationConfig(**cfg_dict)  # type: ignore[arg-type]
    gen_cfg.validate()
    asyncio.run(async_main(gen_cfg))


if __name__ == "__main__":
    main()
