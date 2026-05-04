"""
Extract reasoning traces from GT-HarmBench eval logs.
Outputs a clean JSON/CSV for classification.
Uses inspect_ai.log API to read .eval files.
"""

import json
import argparse
from pathlib import Path
from typing import Optional
import random
from inspect_ai.log import read_eval_log


def extract_reasoning_from_content(content: list) -> str:
    """Extract reasoning text from message content array.
    
    Args:
        content: List of content items (ContentReasoning, ContentText, etc.)
    
    Returns:
        Combined reasoning text
    """
    reasoning_parts = []
    for item in content:
        # Handle both dict format (legacy) and ContentReasoning objects
        if hasattr(item, 'type') and item.type == "reasoning":
            reasoning_parts.append(item.reasoning or "")
        elif isinstance(item, dict) and item.get("type") == "reasoning":
            reasoning_parts.append(item.get("reasoning", ""))
    return "\n\n".join(reasoning_parts)


def extract_action_from_content(content: list) -> str:
    """Extract the final action/answer from message content.
    
    Args:
        content: List of content items (ContentText, etc.)
    
    Returns:
        Action text
    """
    for item in content:
        # Handle both dict format (legacy) and ContentText objects
        if hasattr(item, 'type') and item.type == "text":
            text = item.text or ""
            if text:
                return text
        elif isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text", "")
            if text:
                return text
    return ""


def process_sample(sample) -> Optional[dict]:
    """Process a single sample from inspect_ai EvalLog and extract relevant fields.
    
    Args:
        sample: EvalSample object from inspect_ai
    
    Returns:
        Extracted data dict or None if sample should be skipped
    """
    # Get metadata (now directly from sample object)
    metadata = sample.metadata or {}
    formal_game = metadata.get("formal_game", "")
    
    # Skip matching pennies if desired
    if "matching" in formal_game.lower() or "pennies" in formal_game.lower():
        return None
    
    # Get model info from output
    output = sample.output
    model = output.model if hasattr(output, 'model') else "unknown"
    
    # Get scores from sample.scores
    scores = sample.scores or {}
    all_strategies = scores.get("all_strategies_scorer")
    
    # Extract score values
    score_values = {}
    if all_strategies:
        if hasattr(all_strategies, 'value'):
            score_values = all_strategies.value or {}
        elif isinstance(all_strategies, dict):
            score_values = all_strategies.get("value", {})
    
    # Extract reasoning from both choices (row and column player)
    choices = output.choices if hasattr(output, 'choices') else []
    
    extracted = {
        "id": str(sample.id),
        "model": model,
        "formal_game": formal_game,
        "story_row": metadata.get("story_row", ""),
        "story_col": metadata.get("story_col", ""),
        "actions_row": metadata.get("actions_row", []),
        "actions_col": metadata.get("actions_column", []),
        "rewards_matrix": metadata.get("rewards_matrix", []),
        # Scores
        "nash_score": score_values.get("nash", None),
        "utilitarian_score": score_values.get("utilitarian", None),
        "rawlsian_score": score_values.get("rawlsian", None),
        "nash_social_welfare_score": score_values.get("nash_social_welfare", None),
        # Player traces
        "row_player": {},
        "col_player": {}
    }
    
    # Process row player (first choice)
    if len(choices) > 0:
        row_message = choices[0].message if hasattr(choices[0], 'message') else {}
        row_content = row_message.content if hasattr(row_message, 'content') else []
        extracted["row_player"] = {
            "reasoning": extract_reasoning_from_content(row_content),
            "action": extract_action_from_content(row_content)
        }
    
    # Process column player (second choice)
    if len(choices) > 1:
        col_message = choices[1].message if hasattr(choices[1], 'message') else {}
        col_content = col_message.content if hasattr(col_message, 'content') else []
        extracted["col_player"] = {
            "reasoning": extract_reasoning_from_content(col_content),
            "action": extract_action_from_content(col_content)
        }
    
    return extracted


def load_eval_log(filepath: Path) -> list:
    """Load eval log file using inspect_ai API.
    
    Args:
        filepath: Path to .eval file
    
    Returns:
        List of samples from the log
    """
    try:
        # Use inspect_ai to read the log
        log = read_eval_log(str(filepath))
        return log.samples
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Extract reasoning traces from GT-HarmBench eval logs")
    parser.add_argument("input", type=str, nargs='?', default="logs/reasoning", 
                        help="Input file or directory containing .eval logs (default: logs/reasoning)")
    parser.add_argument("--output", "-o", type=str, default="extracted_traces.json", help="Output file path")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Sample rate (0-1) for each game type")
    parser.add_argument("--games", type=str, nargs="+", default=None, 
                        help="Filter to specific games (e.g., 'Prisoner' 'Chicken')")
    parser.add_argument("--format", type=str, choices=["json", "csv"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Collect all samples
    all_samples = []
    
    if input_path.is_file():
        all_samples = load_eval_log(input_path)
    elif input_path.is_dir():
        # Look for .eval files (inspect_ai format)
        for filepath in input_path.glob("*.eval"):
            all_samples.extend(load_eval_log(filepath))
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    print(f"Loaded {len(all_samples)} samples")
    
    # Process samples
    extracted = []
    game_counts = {}
    
    for sample in all_samples:
        result = process_sample(sample)
        if result is None:
            continue
        
        game = result["formal_game"]
        
        # Filter by game type if specified
        if args.games:
            if not any(g.lower() in game.lower() for g in args.games):
                continue
        
        # Track counts
        game_counts[game] = game_counts.get(game, 0) + 1
        
        extracted.append(result)
    
    print(f"Extracted {len(extracted)} samples after filtering")
    print("Game distribution:")
    for game, count in sorted(game_counts.items()):
        print(f"  {game}: {count}")
    
    # Flatten to separate samples for each player
    flattened = []
    for item in extracted:
        # Create row player sample
        if item["row_player"].get("reasoning") or item["row_player"].get("action"):
            row_sample = {
                "id": f"{item['id']}_row",
                "game_id": item["id"],
                "model": item["model"],
                "formal_game": item["formal_game"],
                "player_role": "row",
                "story": item["story_row"],
                "actions": item["actions_row"],
                "actions_opponent": item["actions_col"],
                "rewards_matrix": item["rewards_matrix"],
                "nash_score": item["nash_score"],
                "utilitarian_score": item["utilitarian_score"],
                "rawlsian_score": item["rawlsian_score"],
                "nash_social_welfare_score": item["nash_social_welfare_score"],
                "reasoning": item["row_player"].get("reasoning", ""),
                "action": item["row_player"].get("action", "")
            }
            flattened.append(row_sample)
        
        # Create column player sample
        if item["col_player"].get("reasoning") or item["col_player"].get("action"):
            col_sample = {
                "id": f"{item['id']}_col",
                "game_id": item["id"],
                "model": item["model"],
                "formal_game": item["formal_game"],
                "player_role": "column",
                "story": item["story_col"],
                "actions": item["actions_col"],
                "actions_opponent": item["actions_row"],
                "rewards_matrix": item["rewards_matrix"],
                "nash_score": item["nash_score"],
                "utilitarian_score": item["utilitarian_score"],
                "rawlsian_score": item["rawlsian_score"],
                "nash_social_welfare_score": item["nash_social_welfare_score"],
                "reasoning": item["col_player"].get("reasoning", ""),
                "action": item["col_player"].get("action", "")
            }
            flattened.append(col_sample)
    
    extracted = flattened
    print(f"Flattened to {len(extracted)} player samples (one per player role)")
    
    # Sample if needed
    if args.sample_rate < 1.0:
        # Stratified sampling by game type
        sampled = []
        by_game = {}
        for item in extracted:
            game = item["formal_game"]
            if game not in by_game:
                by_game[game] = []
            by_game[game].append(item)
        
        for game, items in by_game.items():
            n_sample = max(1, int(len(items) * args.sample_rate))
            sampled.extend(random.sample(items, n_sample))
        
        extracted = sampled
        print(f"Sampled to {len(extracted)} samples")
    
    # Save output
    output_path = Path(args.output)
    
    if args.format == "json":
        with open(output_path, 'w') as f:
            json.dump(extracted, f, indent=2)
    else:
        # CSV format - flatten for analysis
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "game_id", "model", "formal_game", "player_role",
                "nash_score", "utilitarian_score", "rawlsian_score",
                "reasoning", "action"
            ])
            
            for item in extracted:
                writer.writerow([
                    item["id"], item["game_id"], item["model"], item["formal_game"], 
                    item["player_role"],
                    item["nash_score"], item["utilitarian_score"], item["rawlsian_score"],
                    item.get("reasoning", ""), item.get("action", "")
                ])
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()