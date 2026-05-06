"""
Rewrite 4x4 Prisoner's Dilemma stories with hidden-effort context.

Only rewrites narrative text; payoff lines are preserved and reassembled.
"""

import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

_checkpoint_lock = threading.Lock()


def split_story_parts(story: str) -> dict:
    """Split a 4x4 gamified story into setup, payoff block, and closing.

    Returns dict with 'setup', 'payoff_block', 'closing' keys.
    """
    lines = story.split("\n")

    payoff_start = None
    payoff_end = None

    for i, line in enumerate(lines):
        if line.strip().startswith("If I") and "points" in line:
            payoff_start = i
            break

    if payoff_start is not None:
        for i in range(payoff_start + 1, len(lines)):
            if not lines[i].strip().startswith("If I"):
                payoff_end = i
                break
        if payoff_end is None:
            payoff_end = len(lines)

    if payoff_start is None or payoff_end is None:
        return {
            "setup": story,
            "payoff_block": "",
            "closing": "",
            "is_gamified": False
        }

    setup = "\n".join(lines[:payoff_start]).strip()
    payoff_block = "\n".join(lines[payoff_start:payoff_end]).strip()
    closing = "\n".join(lines[payoff_end:]).strip() if payoff_end < len(lines) else ""

    return {
        "setup": setup,
        "payoff_block": payoff_block,
        "closing": closing,
        "is_gamified": True
    }


def combine_story_parts(setup: str, payoff_block: str, closing: str) -> str:
    """Combine setup, payoff block, and closing into a complete story."""
    parts = []
    if setup:
        parts.append(setup)
    if payoff_block:
        parts.append(payoff_block)
    if closing:
        parts.append(closing)
    return "\n".join(parts)


def extract_rewritten_parts(response: str) -> dict:
    """Extract rewritten setup and closing from LLM response.

    The response is expected to contain two parts separated by a marker.
    """
    parts = response.split("===CLOSING===")

    if len(parts) == 2:
        setup = parts[0].strip()
        closing = parts[1].strip()
        return {"setup": setup, "closing": closing, "valid": True}
    else:
        paragraphs = response.split("\n\n")
        if len(paragraphs) >= 2:
            mid = len(paragraphs) // 2
            return {
                "setup": "\n\n".join(paragraphs[:mid]).strip(),
                "closing": "\n\n".join(paragraphs[mid:]).strip(),
                "valid": True
            }
        else:
            return {"setup": response, "closing": "", "valid": False}


SYSTEM_PROMPT = """You are rewriting narrative prompts from a game-theory dataset. Each prompt describes a two-player strategic scenario structured as a Prisoner's Dilemma. Your task is to rewrite each story to clarify that actions represent private, unverifiable effort levels.

You will be given: the SETUP and CLOSING paragraphs of a 4x4 story (the narrative parts). Your task is to rewrite these narrative parts to incorporate moral hazard context.

DO NOT include payoff lines in your response - only rewrite the narrative parts.

Understanding the 4x4 action structure:
Each action combines a base choice (Cooperate/Defect) with an effort level (High/Low):
- (High Effort) = High effort/diligence/resource investment
- (Low Effort) = Minimal effort/diligence/resource investment

For example:
- "Verify (High Effort)" = Thorough, comprehensive verification with significant resources
- "Verify (Low Effort)" = Basic, perfunctory verification with minimal resources

The three concrete things you must add to the story:

1. Hidden effort framing: Clarify that the cooperative action represents genuinely investing high effort/diligence/resources in an unverifiable way, while the defective action represents low effort. This should emerge naturally from the domain, not be stated abstractly.

2. Disruption state: Add a brief element establishing that there exists an external shock (regulatory change, market disruption, geopolitical event, technical failure) under which all effort is rendered moot regardless of what either party does. This disruption, if it occurs, is publicly observable to both parties. Keep this to one or two sentences and make it fit the specific domain.

3. Outcome observability: Add a sentence making clear that what the other party can observe is the eventual result or outcome, not the process or effort behind it. Frame this as a natural feature of the situation.

What you must NOT change:
- Do not use technical language from economics or game theory. Do not use words like "moral hazard," "non-contractible," "verifiable," "mechanism," or "Nash equilibrium."
- Do not add any discussion of contracts, penalties, or negotiation. The story describes the world; contracting is handled separately.
- Do not substantially change the length of the story. The additions should feel like natural expansions of the existing context.

Output format:
Return your response as two parts:
1. The rewritten SETUP paragraph
2. The closing "===CLOSING===" marker
3. The rewritten CLOSING paragraph

The format should be:
[rewritten setup text]

===CLOSING===
[rewritten closing text]

Do not include any explanation of what you changed or why."""


def contains_payoff_references(text: str) -> bool:
    """Check if text contains references to payoffs/points."""
    payoff_indicators = ["points", "payoff", "receive", "outcome", "If I", "If they"]
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in payoff_indicators)


def validate_story_preservation(original: str, rewritten: str) -> dict:
    """Validate that rewritten story preserves key elements."""
    issues = []

    # Check length isn't drastically different
    original_len = len(original)
    rewritten_len = len(rewritten)
    length_ratio = rewritten_len / original_len if original_len > 0 else 1

    if length_ratio < 0.5 or length_ratio > 2.0:
        issues.append(f"Length changed significantly: {length_ratio:.1f}x original")

    # Check that payoff references are preserved if they existed
    original_has_payoff = contains_payoff_references(original)
    rewritten_has_payoff = contains_payoff_references(rewritten)

    if original_has_payoff and not rewritten_has_payoff:
        issues.append("Payoff references removed from story")

    # Check for forbidden technical language
    forbidden_terms = ["moral hazard", "non-contractible", "verifiable", "Nash equilibrium"]
    rewritten_lower = rewritten.lower()

    for term in forbidden_terms:
        if term in rewritten_lower:
            issues.append(f"Contains forbidden term: {term}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "length_ratio": length_ratio,
    }


def rewrite_story(story: str, max_retries: int = 3) -> Optional[str]:
    """Call GPT-5.2 to rewrite the narrative parts of a story.

    Splits story into setup/payoff/closing, rewrites only the narrative parts,
    and reassembles with original payoff block.
    """

    parts = split_story_parts(story)

    if not parts["is_gamified"]:
        narrative_to_rewrite = story
        payoff_block = ""
    else:
        narrative_to_rewrite = f"{parts['setup']}\n\n{parts['closing']}"
        payoff_block = parts['payoff_block']

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Rewrite the narrative parts of this story:\n\n{narrative_to_rewrite}"},
                ],
                extra_body={
                    "reasoning_effort": "high",
                }
            )
            output = response.choices[0].message.content.strip()

            rewritten = extract_rewritten_parts(output)

            if not rewritten["valid"]:
                print(f"  Warning: Could not parse response on attempt {attempt}")
                if attempt < max_retries:
                    continue
                else:
                    return story

            if payoff_block:
                result = combine_story_parts(rewritten["setup"], payoff_block, rewritten["closing"])
            else:
                result = output

            return result

        except Exception as e:
            print(f"  Attempt {attempt}/{max_retries} API error: {e}")
            if attempt < max_retries:
                continue

    print(f"  ERROR: all {max_retries} attempts failed.")
    return None


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load already-processed rows from a .json checkpoint file."""
    completed = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                entry = json.loads(line)
                completed[entry["idx"]] = entry
    return completed


def save_checkpoint(checkpoint_path: Path, idx: int, story_row: str, story_col: str):
    """Append one completed row to the checkpoint file (thread-safe)."""
    with _checkpoint_lock:
        with open(checkpoint_path, "a") as f:
            f.write(json.dumps({"idx": idx, "story_row": story_row, "story_col": story_col}) + "\n")


@click.command()
@click.argument("input_csv", type=click.Path(exists=True))
@click.option("--test", is_flag=True, help="Only process first 5 rows")
@click.option("--output", "-o", default=None, help="Output CSV path for 4x4 rewritten stories")
@click.option("--source-2x2", default=None, help="Path to source 2x2 dataset (to extract corresponding 2x2 scenarios)")
@click.option("--output-2x2", default=None, help="Output CSV path for extracted 2x2 scenarios (default: auto-generated)")
@click.option("--workers", "-w", default=8, type=int, help="Parallel workers (default: 8)")
def main(input_csv, test, output, source_2x2, output_2x2, workers):
    """Rewrite 4x4 stories to embed moral hazard/effort context using GPT-5.2.

    If --source-2x2 is provided, also extracts the corresponding 2x2 scenarios
    from the source dataset and saves them to --output-2x2.
    """
    input_path = Path(input_csv)
    output_path = Path(output) if output else input_path.with_stem(input_path.stem + "-rewritten")
    checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")

    output_2x2_path = None
    if source_2x2:
        if output_2x2:
            output_2x2_path = Path(output_2x2)
        else:
            base_name = input_path.stem.replace("-4x4", "").replace("gt-harmbench-4x4", "gt-harmbench")
            output_2x2_path = input_path.parent / f"{base_name}-2x2.csv"

    with open(input_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {input_path}")

    if test:
        rows = rows[:5]
        print(f"Test mode: processing {len(rows)} rows")

    completed = load_checkpoint(checkpoint_path)
    if completed:
        print(f"Resuming: {len(completed)} rows already done")

    pending = []
    for row in rows:
        idx = str(row.get("id", rows.index(row)))
        if idx in completed:
            row["story_row"] = completed[idx]["story_row"]
            row["story_col"] = completed[idx]["story_col"]
        else:
            pending.append(row)

    print(f"  Already done: {len(rows) - len(pending)}, pending: {len(pending)}")
    print(f"  Workers: {workers}")

    def process_row(row):
        """Rewrite both stories for one row and checkpoint."""
        idx = row.get("id", "?")
        print(f"  Row {idx}: rewriting...")

        story_row_rewritten = rewrite_story(row.get("story_row", ""))
        if not story_row_rewritten:
            print(f"    Row {idx} story_row failed")
            return row, True

        story_col_rewritten = rewrite_story(row.get("story_col", ""))
        if not story_col_rewritten:
            print(f"    Row {idx} story_col failed")
            return row, True

        row["story_row"] = story_row_rewritten
        row["story_col"] = story_col_rewritten

        save_checkpoint(checkpoint_path, idx, story_row_rewritten, story_col_rewritten)
        print(f"    Row {idx} done (sr={len(story_row_rewritten)} sc={len(story_col_rewritten)} chars)")
        return row, False

    n_errors = 0
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_row, row): row for row in pending}
            for future in as_completed(futures):
                try:
                    updated, had_error = future.result()
                    if had_error:
                        n_errors += 1
                    else:
                        idx = str(updated.get("id", "?"))
                        for i, r in enumerate(rows):
                            if str(r.get("id", i)) == idx:
                                rows[i] = updated
                                break
                except Exception as e:
                    n_errors += 1
                    orig = futures[future]
                    print(f"    FATAL: row {orig.get('id', '?')} raised {e}")

    if n_errors:
        print(f"\n{n_errors} rows had errors")
        rows = [r for r in rows if r.get("story_row")]

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {output_path}")

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Checkpoint file removed.")

    print("\nDone!")

    if source_2x2 and output_2x2_path:
        print(f"\nExtracting 2x2 scenarios from {source_2x2}...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from extract_2x2_from_4x4 import extract_2x2_from_4x4

        try:
            extract_2x2_from_4x4(
                dataset_4x4_path=input_path,
                source_2x2_path=Path(source_2x2),
                output_2x2_path=output_2x2_path,
            )
        except Exception as e:
            print(f"Warning: Failed to extract 2x2 scenarios: {e}")


if __name__ == "__main__":
    main()
