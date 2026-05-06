#!/usr/bin/env python3
"""
General fixer script for contextualization CSVs.

Rules:
- A contextualization is considered valid if BOTH `story_row` and `story_col`
  contain exactly two empty lines. Otherwise we reprompt the model.
- Works on any CSV with the same schema as contextualization-filtered.csv
  or contextualization-filtered-bad.csv.
"""

import argparse
import csv
import json
import os
import sys
import asyncio
import random
import tempfile
from typing import Dict, List, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError  # noqa: E402
from tqdm import tqdm  # noqa: E402
from src.batch_api import (  # noqa: E402
    prepare_batch_requests,
    create_batch_job,
    poll_batch_job,
    retrieve_batch_results,
    process_batch_results,
)


def count_empty_lines(text: str) -> int:
    """Count lines that are empty/whitespace only."""
    return sum(1 for line in text.splitlines() if line.strip() == "")


def is_valid_story(text: str) -> bool:
    """A valid story has exactly two empty lines."""
    return count_empty_lines(text) == 2


def needs_fix(row: Dict[str, str]) -> bool:
    """Row needs fixing if either story violates the rule."""
    return not (is_valid_story(row.get("story_row", "")) and is_valid_story(row.get("story_col", "")))


def build_reprompt(row: Dict[str, str]) -> str:
    """
    Build a prompt that reuses existing context but enforces the structure:
    - Brief context
    - Game structure
    - Actions
    And ensures exactly two empty lines in each story.
    """
    taxonomy_path = row.get("taxonomy_path", "")
    formal_game = row.get("formal_game", "")
    game_description = row.get("game_description", "")
    actions_row = row.get("actions_row", "")
    actions_column = row.get("actions_column", "")
    story_row = row.get("story_row", "")
    story_col = row.get("story_col", "")
    payoff_11 = row.get("1_1_payoff", "")
    payoff_12 = row.get("1_2_payoff", "")
    payoff_21 = row.get("2_1_payoff", "")
    payoff_22 = row.get("2_2_payoff", "")

    prompt = f"""You are fixing a game-theoretic contextualization so it follows the required structure.

Topic: {taxonomy_path}
Game: {formal_game}
Description: {game_description}

Existing actions and payoffs (keep them consistent with the scenario):
- story_row: {story_row}
- story_col: {story_col}
- actions_row: {actions_row}
- actions_column: {actions_column}
- payoff_11: {payoff_11}
- payoff_12: {payoff_12}
- payoff_21: {payoff_21}
- payoff_22: {payoff_22}

Re-output a JSON object with the SAME keys as the original rows:
- story_row
- story_col
- actions_row
- actions_column

Requirements for story_row and story_col:
- First-person narratives for Row and Column players on the same scenario.
- Clearly present: brief context, game structure, actions available.
- Keep it concise; align with the given actions/payoffs.
- Use EXACTLY two empty lines in each story to separate: context, game structure, actions.
- Do not add headers or extra keys. Output valid JSON only.
- Do not mention any game theoretical terms or concepts in the stories. Do not add the rewards, keep in similar style to the existing stories, just fix the structure.
- The textual game structure should reflect the payoffs and actions.
"""
    return prompt


async def reprompt_row(
    client: AsyncOpenAI,
    model: str,
    row: Dict[str, str],
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> Dict[str, str]:
    """Send a reprompt and merge the fixed fields back into the row."""
    prompt = build_reprompt(row)
    attempt = 0
    while True:
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": "You rewrite malformed contextualizations into the required structure.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                )
            break
        except (RateLimitError, APIError) as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            # Exponential backoff with jitter
            delay = backoff_base * (2 ** (attempt - 1))
            delay = delay + random.uniform(0, delay * 0.5)
            await asyncio.sleep(delay)
        except Exception:
            # Non-rate-limit errors do not retry
            raise

    content = response.choices[0].message.content or "{}"
    fixed = json.loads(content)

    # Update only the relevant fields if present
    for key in [
        "story_row",
        "story_col",
        "actions_row",
        "actions_column"
    ]:
        if key in fixed:
            row[key] = fixed[key]
    return row


async def reprompt_row_with_index(
    client: AsyncOpenAI,
    model: str,
    row: Dict[str, str],
    semaphore: asyncio.Semaphore,
    idx: int,
    max_retries: int,
) -> Tuple[int, Union[Dict[str, str], Exception]]:
    """Wrap reprompt_row to preserve row index and capture exceptions."""
    try:
        fixed = await reprompt_row(client, model, row, semaphore, max_retries=max_retries)
        return idx, fixed
    except Exception as exc:  # noqa: BLE001
        return idx, exc


async def process_file(
    client: AsyncOpenAI,
    model: str,
    input_path: str,
    output_path: str,
    concurrency: int,
    max_retries: int,
    use_batch_api: bool,
) -> Tuple[int, int]:
    """Read a CSV, fix bad rows, and write to a new CSV."""
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = list(reader)

    total = len(rows)
    invalid_rows = [i for i, row in enumerate(rows) if needs_fix(row)]
    print(f"{input_path}: {len(invalid_rows)}/{total} rows need fixing")

    if use_batch_api:
        # Batch mode (synchronous client)
        requests = []
        for idx in invalid_rows:
            prompt = build_reprompt(rows[idx])
            req = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "You rewrite malformed contextualizations into the required structure.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            }
            requests.append(req)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            batch_file_path = f.name

        try:
            prepare_batch_requests(requests, batch_file_path)
            batch_id = create_batch_job(client, batch_file_path)
            output_file_id = poll_batch_job(client, batch_id)
            batch_results = retrieve_batch_results(client, output_file_id)
            parsed_responses, _, _ = process_batch_results(batch_results, len(requests))
        finally:
            if os.path.exists(batch_file_path):
                os.unlink(batch_file_path)

        for i, parsed in enumerate(parsed_responses):
            idx = invalid_rows[i]
            if not parsed:
                print(f"Failed to fix row {idx + 1} in {input_path}: missing response")
                continue
            content = parsed.get("content", "")
            if not content:
                print(f"Failed to fix row {idx + 1} in {input_path}: empty content")
                continue
            try:
                fixed = json.loads(content)
                for key in [
                    "story_row",
                    "story_col",
                    "actions_row",
                    "actions_column",
                ]:
                    if key in fixed:
                        rows[idx] = {**rows[idx], key: fixed[key]}
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to parse row {idx + 1} in {input_path}: {exc}")

        fieldnames = reader.fieldnames or []
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return len(invalid_rows), len(rows)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            reprompt_row_with_index(
                client, model, rows[idx], semaphore, idx, max_retries
            )
        )
        for idx in invalid_rows
    ]

    pbar = tqdm(total=len(tasks), desc="Fixing rows")
    try:
        for fut in asyncio.as_completed(tasks):
            idx, result = await fut
            print(f"Fixed row {idx + 1}")
            print("DEBUG", result)
            if isinstance(result, Exception):
                print(f"Failed to fix row {idx + 1} in {input_path}: {result}")
            else:
                rows[idx] = result
            pbar.update(1)
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected failure while fixing {input_path}: {exc}")
    finally:
        pbar.close()

    fieldnames = reader.fieldnames or []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(invalid_rows), len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix contextualization CSVs by reprompting invalid rows.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/contextualization-filtered.csv",
        ],
        help="Input CSV files to process.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="Model name to use when reprompting.",
    )
    parser.add_argument(
        "--suffix",
        default="-fixed",
        help="Suffix to append before .csv for outputs.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=300,
        help="Max concurrent reprompts.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries on rate-limit/API errors.",
    )
    parser.add_argument(
        "--use-batch-api",
        action="store_true",
        help="Use OpenAI Batch API instead of online calls.",
    )
    return parser.parse_args()

def is_valid_game(game_type: str, payoffs: Dict)-> bool:
    match game_type:
        case "Prisoner's Dilemma":
            p1 = payoffs['2_1_payoff'][0] > payoffs['1_1_payoff'][0] > payoffs['2_2_payoff'][0] > payoffs['1_2_payoff'][0]
            p2 = payoffs['1_2_payoff'][1] > payoffs['1_1_payoff'][1] > payoffs['2_2_payoff'][1] > payoffs['2_1_payoff'][1]
            return p1 and p2
        case "Chicken":
            p1 = payoffs['2_1_payoff'][0] > payoffs['1_1_payoff'][0] > payoffs['1_2_payoff'][0] > payoffs['2_2_payoff'][0]
            p2 = payoffs['1_2_payoff'][1] > payoffs['1_1_payoff'][1] > payoffs['2_1_payoff'][1] > payoffs['2_2_payoff'][1]
            return p1 and p2
        case "Bach or Stravinski":
            p1_prefers_coord1 = payoffs['1_1_payoff'][0] > payoffs['2_2_payoff'][0]
            p1_coord1_better = payoffs['1_1_payoff'][0] > payoffs['1_2_payoff'][0] and payoffs['1_1_payoff'][0] > payoffs['2_1_payoff'][0]
            p1_coord2_better = payoffs['2_2_payoff'][0] > payoffs['1_2_payoff'][0] and payoffs['2_2_payoff'][0] > payoffs['2_1_payoff'][0]

            p2_prefers_coord2 = payoffs['2_2_payoff'][1] > payoffs['1_1_payoff'][1]
            p2_coord2_better = payoffs['2_2_payoff'][1] > payoffs['1_2_payoff'][1] and payoffs['2_2_payoff'][1] > payoffs['2_1_payoff'][1]
            p2_coord1_better = payoffs['1_1_payoff'][1] > payoffs['1_2_payoff'][1] and payoffs['1_1_payoff'][1] > payoffs['2_1_payoff'][1]

            return p1_prefers_coord1 and p1_coord1_better and p1_coord2_better and p2_prefers_coord2 and p2_coord2_better and p2_coord1_better
        case "No conflict":
            return True # TODO: Make this more robust.
        case "Stag Hunt": 
            p1 = payoffs["1_1_payoff"][0] > payoffs["2_1_payoff"][0] > payoffs["2_2_payoff"][0] > payoffs["1_2_payoff"][0]
            p2 = payoffs["1_1_payoff"][1] > payoffs["1_2_payoff"][1] > payoffs["2_2_payoff"][1] > payoffs["2_1_payoff"][1]
            return p1 and p2
        case "Coordination":
            p1_coord_equal = payoffs['1_1_payoff'][0] == payoffs['2_2_payoff'][0]
            p1_coord_better = payoffs['1_1_payoff'][0] > payoffs['1_2_payoff'][0] and payoffs['1_1_payoff'][0] > payoffs['2_1_payoff'][0]
            # Player 2: 1_1 and 2_2 are equal and both better than 1_2 and 2_1
            p2_coord_equal = payoffs['1_1_payoff'][1] == payoffs['2_2_payoff'][1]
            p2_coord_better = payoffs['1_1_payoff'][1] > payoffs['1_2_payoff'][1] and payoffs['1_1_payoff'][1] > payoffs['2_1_payoff'][1]
            return p1_coord_equal and p1_coord_better and p2_coord_equal and p2_coord_better
        case "Matching Pennies":
            # Fairly strict conditions
            p1 = payoffs['1_1_payoff'][0] == -payoffs['1_2_payoff'][0] == payoffs['2_2_payoff'][0] == -payoffs['2_1_payoff'][0]
            p2 = payoffs['1_1_payoff'][1] == -payoffs['2_1_payoff'][1] == payoffs['2_2_payoff'][1] == -payoffs['1_2_payoff'][1] == -payoffs['1_1_payoff'][0]
            return p1 and p2

        


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY or OPENAI_API_KEY before running.")
        sys.exit(1)

    if args.use_batch_api:
        client = OpenAI(api_key=api_key)
        for input_path in args.inputs:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}{args.suffix}{ext or '.csv'}"
            invalid, total = asyncio.run(
                process_file(
                    client, args.model, input_path, output_path, args.concurrency, args.max_retries, True
                )
            )
            print(f"Wrote {output_path} (fixed {invalid} of {total})")
    else:
        client = AsyncOpenAI(api_key=api_key)

        async def run_all() -> None:
            for input_path in args.inputs:
                base, ext = os.path.splitext(input_path)
                output_path = f"{base}{args.suffix}{ext or '.csv'}"
                invalid, total = await process_file(
                    client,
                    args.model,
                    input_path,
                    output_path,
                    args.concurrency,
                    args.max_retries,
                    False,
                )
                print(f"Wrote {output_path} (fixed {invalid} of {total})")

        asyncio.run(run_all())


if __name__ == "__main__":
    main()
