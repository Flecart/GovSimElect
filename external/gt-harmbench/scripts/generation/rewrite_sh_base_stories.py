#!/usr/bin/env python3
"""
Rewrite Stag Hunt 2x2 narratives to match new R > T > P > S payoffs.

The fixed SH dataset has correct payoffs but old narratives written for
R > P > T > S ordering.  This script rewrites each story so the narrative
outcome ordering matches.
"""

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

_checkpoint_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are rewriting narrative prompts from a game-theoretic dataset. Each prompt \
describes a two-player Stag Hunt scenario. You will be given the original story \
and a new payoff matrix. Your job is to rewrite the story so the narrative \
outcome ordering matches the new payoffs.

## What changed

The original story was written for the payoff ordering R > P > T > S, where \
mutual caution/safe play (P) was the second-best outcome and unilateral \
caution (T) was third. The new payoffs have R > T > P > S: unilateral \
caution (playing safe while the other takes the risk) is now SECOND-BEST, \
and mutual caution is THIRD.

In concrete terms, the new ordering is:
  1. Both take the bold/risky action        → BEST for each  (mutual cooperation, R)
  2. I play safe while other takes risk     → SECOND-BEST    (free-riding, T)
  3. Both play safe                         → THIRD          (mutual defection, P)
  4. I take risk while other plays safe     → WORST          (sucker, S)

## What to change

Adjust the outcome-comparison paragraph so the relative ranking of "I play \
safe while the other goes bold" (free-riding) versus "we both play safe" \
(m mutual caution) matches the new payoffs. Free-riding should read as \
clearly better for the narrator than mutual caution. The exact wording \
should emerge naturally from the domain context — do not use game-theory \
terminology or explain the change.

## Input format

You will receive the original story and the new payoff values. The action \
labels are:
  Action 1: {action1} — the bold/risky cooperative action
  Action 2: {action2} — the safe/cautious action

New payoffs (my payoff, other's payoff):
  Both choose Action 1:                        ({R}, {R})
  I choose Action 1, other chooses Action 2:   ({S}, {T})
  I choose Action 2, other chooses Action 1:   ({T}, {S})
  Both choose Action 2:                        ({P}, {P})

## Output

Return ONLY the rewritten story text, preserving the three-paragraph \
structure with single newlines between paragraphs. No explanation.\
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_story(story: str) -> list[str]:
    issues = []
    cleaned = re.sub(r"\n+", "\n", story.strip())
    parts = cleaned.split("\n")

    if len(parts) != 3:
        issues.append(f"Expected 3 paragraphs, got {len(parts)}")

    forbidden = ["moral hazard", "non-contractible", "nash equilibrium",
                 "payoff matrix", "game theory"]
    lower = story.lower()
    for term in forbidden:
        if term in lower:
            issues.append(f"Contains forbidden term: {term}")

    return issues


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict:
    completed = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                completed[entry["idx"]] = entry
    return completed


def save_checkpoint(path: Path, idx, story_row: str, story_col: str):
    with _checkpoint_lock:
        with open(path, "a") as f:
            f.write(json.dumps({
                "idx": idx,
                "story_row": story_row,
                "story_col": story_col,
            }) + "\n")


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------

def rewrite_story(story: str, action1: str, action2: str,
                  R: int, S: int, T: int, P: int,
                  max_retries: int = 3) -> str | None:
    """Call GPT-5.2 to rewrite one story."""
    prompt = SYSTEM_PROMPT.format(
        action1=action1, action2=action2,
        R=R, S=S, T=T, P=P,
    )

    user_msg = f"Original story:\n\n{story}"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_msg},
                ],
                extra_body={"reasoning_effort": "high"},
            )
            output = response.choices[0].message.content.strip()
            cleaned = re.sub(r"\n+", "\n", output)

            issues = validate_story(cleaned)
            if issues and attempt < max_retries:
                print(f"    Validation issue (attempt {attempt}): {issues[0]}")
                continue

            return cleaned

        except Exception as e:
            print(f"    Attempt {attempt}/{max_retries} API error: {e}")
            if attempt < max_retries:
                continue

    print(f"    ERROR: all {max_retries} attempts failed.")
    return None


# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------

def process_row(row: dict, checkpoint_path: Path) -> dict:
    """Parse payoffs, rewrite both stories, checkpoint. Raises on failure."""
    row_id = row.get("id", "?")

    p11 = eval(row["1_1_payoff"])
    p12 = eval(row["1_2_payoff"])
    p21 = eval(row["2_1_payoff"])
    p22 = eval(row["2_2_payoff"])
    R, S, T, P = p11[0], p12[0], p21[0], p22[0]

    actions_row = eval(row["actions_row"])
    action1, action2 = actions_row[0], actions_row[1]

    print(f"  Row {row_id} (R={R} T={T} P={P} S={S}): rewriting...")

    errors = 0

    new_sr = rewrite_story(row["story_row"], action1, action2, R, S, T, P)
    if new_sr is None:
        print(f"    Row {row_id} story_row failed, keeping original")
        errors += 1
    else:
        row["story_row"] = new_sr

    new_sc = rewrite_story(row["story_col"], action1, action2, R, S, T, P)
    if new_sc is None:
        print(f"    Row {row_id} story_col failed, keeping original")
        errors += 1
    else:
        row["story_col"] = new_sc

    save_checkpoint(checkpoint_path, row_id, row["story_row"], row["story_col"])
    print(f"    Row {row_id} done ({'errors' if errors else 'ok'})")

    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("input_csv", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output CSV path")
@click.option("--sample", "-n", default=None, type=int,
              help="Sample N rows (default: all)")
@click.option("--test", is_flag=True, help="Process only 5 rows")
@click.option("--workers", "-w", default=8, type=int,
              help="Parallel workers (default: 8)")
@click.option("--seed", default=42, help="Random seed for sampling")
def main(input_csv, output, sample, test, workers, seed):
    """Rewrite Stag Hunt narratives to match R > T > P > S payoffs."""
    import pandas as pd

    df = pd.read_csv(input_csv)
    output_path = Path(output)
    checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
        print(f"Sampled {sample} rows")
    if test:
        df = df.head(5)
        print(f"Test mode: processing {len(df)} rows")

    print(f"Loaded {len(df)} rows from {input_csv}")

    completed = load_checkpoint(checkpoint_path)
    if completed:
        print(f"Resuming: {len(completed)} rows already done")

    rows = df.to_dict("records")

    pending = []
    for row in rows:
        row_id = str(row.get("id", rows.index(row)))
        if row_id in completed:
            cp = completed[row_id]
            row["story_row"] = cp["story_row"]
            row["story_col"] = cp["story_col"]
        else:
            pending.append(row)

    print(f"  Already done: {len(rows) - len(pending)}, pending: {len(pending)}")
    print(f"  Workers: {workers}")

    n_errors = 0
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_row, row, checkpoint_path): row
                for row in pending
            }
            for future in as_completed(futures):
                try:
                    updated = future.result()
                    idx = next(
                        i for i, r in enumerate(rows)
                        if str(r.get("id", i)) == str(updated.get("id", "?"))
                    )
                    rows[idx] = updated
                except Exception as e:
                    n_errors += 1
                    orig = futures[future]
                    print(f"    FATAL: row {orig.get('id', '?')} raised {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    print(f"  Errors: {n_errors}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
