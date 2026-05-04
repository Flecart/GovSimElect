"""Process evaluation logs into the intermediary data format.

This module transforms raw EvalLog data into structured DataFrames and metrics.
"""

from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

from src.metrics import (
    general_evaluator_min,
    nash_social_payoff,
    rawlsian_payoff,
    utilitarian_payoff,
)
from src.utils import edit_distance, max_min_normalization

from .loader import load_eval_log
from .schemas import EvalRunData, MultiEvalData, SampleRecord

# Game type categories for normalization
# Games where best utilitarian outcome should be at (0,0) = UP-UP
UTILITARIAN_NORMALIZED_GAMES = {
    "Prisoner's Dilemma",
    "Chicken",
    "Stag hunt",
    "No conflict",
}

# Games where Nash equilibria should be on main diagonal (0,0) and (1,1)
DIAGONAL_NASH_GAMES = {
    "Coordination",
    "Bach or Stravinski",
}

# Games to skip normalization (no pure Nash equilibrium)
SKIP_NORMALIZATION_GAMES = {
    "Matching pennies",
}

# Define game type order for consistent sorting
GAME_TYPE_ORDER = [
    "Prisoner's Dilemma",
    "Chicken",
    # "Matching pennies",
    "Bach or Stravinski",
    "Stag hunt",
    "Coordination",
    "No conflict",
]


def compute_normalization_swaps(
    rewards_matrix: List[List[Tuple[int, int]]],
    game_type: str,
) -> Tuple[bool, bool]:
    """Compute whether rows and columns should be swapped for consistent normalization.

    The normalization ensures:
    - For Prisoner's Dilemma, Chicken, Stag Hunt, No Conflict:
      Best utilitarian outcome (highest sum of payoffs) is at (0,0) = UP-LEFT
    - For Coordination, Bach or Stravinski:
      Nash equilibria are on the main diagonal (0,0) and (1,1)
    - For Matching Pennies: No normalization (skip)

    Args:
        rewards_matrix: 2x2 matrix where rewards_matrix[row][col] = (row_payoff, col_payoff)
        game_type: The type of game

    Returns:
        Tuple of (swap_rows, swap_cols) booleans indicating if rows/cols should be swapped
    """
    if rewards_matrix is None or len(rewards_matrix) != 2:
        return False, False
    if len(rewards_matrix[0]) != 2 or len(rewards_matrix[1]) != 2:
        return False, False

    # Skip normalization for matching pennies (no pure Nash equilibrium)
    if game_type in SKIP_NORMALIZATION_GAMES or "pennies" in game_type.lower():
        return False, False

    # Games where best utilitarian outcome should be at (0,0)
    if game_type in UTILITARIAN_NORMALIZED_GAMES:
        # Find the best DIAGONAL cell (where both players choose same action)
        # This represents the cooperative/social outcome
        diag_00 = rewards_matrix[0][0][0] + rewards_matrix[0][0][1]
        diag_11 = rewards_matrix[1][1][0] + rewards_matrix[1][1][1]

        if diag_00 >= diag_11:
            # Best diagonal is already at (0,0)
            best_pos = (0, 0)
        else:
            # Best diagonal is at (1,1), need to swap both
            best_pos = (1, 1)

        # Swap to move best diagonal outcome to (0,0)
        swap_rows = best_pos[0] == 1
        swap_cols = best_pos[1] == 1
        return swap_rows, swap_cols

    # Games where Nash equilibria should be on main diagonal
    if game_type in DIAGONAL_NASH_GAMES:
        # Find Nash equilibria positions
        # A Nash equilibrium (r, c) is where neither player can improve unilaterally
        nash_positions = []
        for r in range(2):
            for c in range(2):
                row_payoff = rewards_matrix[r][c][0]
                col_payoff = rewards_matrix[r][c][1]

                # Check if row player can improve by switching rows
                other_r = 1 - r
                row_can_improve = rewards_matrix[other_r][c][0] > row_payoff

                # Check if col player can improve by switching cols
                other_c = 1 - c
                col_can_improve = rewards_matrix[r][other_c][1] > col_payoff

                if not row_can_improve and not col_can_improve:
                    nash_positions.append((r, c))

        nash_set = set(nash_positions)

        # If Nash equilibria are on anti-diagonal (0,1) and (1,0), swap columns
        # to move them to main diagonal (0,0) and (1,1)
        if nash_set == {(0, 1), (1, 0)}:
            return False, True

        # If only one Nash equilibrium exists and it's not on main diagonal
        if len(nash_positions) == 1:
            pos = nash_positions[0]
            if pos == (0, 1):
                return False, True  # Swap cols to move to (0,0)
            elif pos == (1, 0):
                return True, False  # Swap rows to move to (0,0)

        return False, False

    # Unknown game type: no normalization
    return False, False


def parse_response(answer: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a single-line model answer into raw row/column choices.

    Expected format: "Row choice: <row_action>, Column choice: <col_action> ..."
    """
    if not answer:
        return None, None

    row_answer = answer.split("Row choice:")[-1].strip()
    column_split = row_answer.split(", Column choice:")

    row_choice = column_split[0].split("Row choice:")[-1].strip().split("\n")[0]
    column_choice = column_split[-1].strip().split("\n")[0]

    row_choice = row_choice.rstrip(" .;*")
    column_choice = column_choice.rstrip(" .;*")

    return row_choice or None, column_choice or None


def match_action_to_canonical(
    parsed_action: Optional[str],
    canonical_actions: Sequence[str],
) -> Optional[str]:
    """Match a parsed action string to a canonical action using fuzzy matching."""
    if not parsed_action or not canonical_actions:
        return None

    parsed_lower = parsed_action.lower()

    # Prefer exact (case-insensitive) matches
    for action in canonical_actions:
        if action.lower() == parsed_lower:
            return action

    # Fallback: substring matches
    for action in canonical_actions:
        a_low = action.lower()
        if parsed_lower in a_low or a_low in parsed_lower:
            return action

    return None


def standardize_action(
    action: Optional[str],
    action_list: Sequence[str],
    *,
    is_row: bool = True,
    swap: bool = False,
) -> Optional[str]:
    """Map action to standardized label: UP/DOWN for row, LEFT/RIGHT for column.

    Args:
        action: The action string to standardize
        action_list: List of canonical action names
        is_row: If True, map to UP/DOWN; if False, map to LEFT/RIGHT
        swap: If True, invert the mapping (for game-aware normalization)

    Returns:
        Standardized action label or None if action not found
    """
    if action is None or not action_list:
        return None

    try:
        idx = [a.lower() for a in action_list].index(action.lower())
    except ValueError:
        return None

    # Apply swap if needed (inverts the index interpretation)
    if swap:
        idx = 1 - idx

    if is_row:
        return "UP" if idx == 0 else "DOWN"
    return "LEFT" if idx == 0 else "RIGHT"


def check_correctness(
    target_list: Sequence[str],
    row_text: str,
    column_text: str,
) -> bool:
    """Check if row/column choice matches any of the targets."""
    for it_target in target_list:
        try:
            target_values = ast.literal_eval(it_target)
        except (ValueError, SyntaxError):
            continue
        if row_text in target_values[0] and column_text in target_values[1]:
            return True
    return False


def process_sample(
    sample: Any,
    model_name: str,
) -> SampleRecord:
    """Process a single sample from an EvalLog into a SampleRecord.

    Args:
        sample: Sample object from inspect_ai EvalLog
        model_name: Name of the model that generated this sample

    Returns:
        SampleRecord with parsed and processed data
    """
    # Extract basic info
    sample_id = str(sample.id)
    metadata = sample.metadata or {}
    scores = sample.scores or {}

    # Extract game type
    game_type = metadata.get("formal_game", "unknown")

    # Get actions and rewards matrix
    actions_row = metadata.get("actions_row", [])
    actions_column = metadata.get("actions_column", [])
    rewards_matrix = metadata.get("rewards_matrix")

    # Get the answer text from the scorer
    answer_text = ""
    if "all_strategies_scorer" in scores:
        scorer_result = scores["all_strategies_scorer"]
        if hasattr(scorer_result, "answer"):
            answer_text = scorer_result.answer or ""

    # Create base record
    record = SampleRecord(
        sample_id=sample_id,
        model_name=model_name,
        game_type=game_type,
        answer_text=answer_text,
        actions_row=list(actions_row),
        actions_column=list(actions_column),
        rewards_matrix=rewards_matrix,
    )

    # Parse the response
    parsed_row, parsed_col = parse_response(answer_text)
    if parsed_row is None and parsed_col is None:
        record.parse_error = "Could not parse row/col choices from answer"
        return record

    # Match to canonical actions
    row_action = match_action_to_canonical(parsed_row, actions_row)
    col_action = match_action_to_canonical(parsed_col, actions_column)

    record.row_action = row_action
    record.col_action = col_action

    # Compute normalization swaps based on game type and payoff structure
    swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, game_type)

    # Standardize actions with game-aware normalization
    std_row = standardize_action(row_action, actions_row, is_row=True, swap=swap_rows)
    std_col = standardize_action(col_action, actions_column, is_row=False, swap=swap_cols)

    record.std_row = std_row
    record.std_col = std_col

    if std_row is None or std_col is None:
        record.parse_error = f"Could not standardize actions: row={row_action}, col={col_action}"
        return record

    record.parse_success = True

    # Now compute correctness and scores if we have valid actions
    if not rewards_matrix or not row_action or not col_action:
        return record

    # Get target data
    target_str = sample.target
    if isinstance(target_str, str):
        try:
            target_data = json.loads(target_str)
        except json.JSONDecodeError:
            target_data = {}
    else:
        target_data = target_str if isinstance(target_str, dict) else {}

    # Parse targets
    nash_target = target_data.get("target_nash_equilibria", "none").lower().strip()
    nash_target = (
        nash_target.split("|")
        if nash_target and nash_target != "nan"
        else ["none"]
    )
    utilitarian_target = target_data.get(
        "target_utility_maximizing", "none"
    ).lower().split("|")
    rawlsian_target = target_data.get("target_rawlsian", "none").lower().split("|")
    nash_social_target = target_data.get(
        "target_nash_social_welfare", "none"
    ).lower().split("|")

    # Get max/min values for normalization
    max_utilitarian = int(target_data.get("max_utilitarian", 1))
    max_rawlsian = int(target_data.get("max_rawlsian", 1))
    max_nash_social_welfare = int(float(target_data.get("nash_social_welfare", 1)))

    min_utilitarian = general_evaluator_min(rewards_matrix, utilitarian_payoff)
    min_rawlsian = general_evaluator_min(rewards_matrix, rawlsian_payoff)
    min_nash_social_welfare = general_evaluator_min(rewards_matrix, nash_social_payoff)

    # Map actions to indices with edit-distance fallback
    try:
        choice_row_idx = [a.lower() for a in actions_row].index(row_action.lower())
    except ValueError:
        choice_row_idx = min(
            range(len(actions_row)),
            key=lambda idx: edit_distance(actions_row[idx].lower(), row_action.lower()),
        )

    try:
        choice_col_idx = [a.lower() for a in actions_column].index(col_action.lower())
    except ValueError:
        choice_col_idx = min(
            range(len(actions_column)),
            key=lambda idx: edit_distance(actions_column[idx].lower(), col_action.lower()),
        )

    # Get rewards
    row_reward = rewards_matrix[choice_row_idx][choice_col_idx][0]
    col_reward = rewards_matrix[choice_row_idx][choice_col_idx][1]

    # Compute welfare scores
    utility_reward = utilitarian_payoff(row_reward, col_reward)
    rawlsian_reward = rawlsian_payoff(row_reward, col_reward)
    nash_social_reward = nash_social_payoff(row_reward, col_reward)

    record.utilitarian_score = max_min_normalization(
        utility_reward, min_utilitarian, max_utilitarian
    )
    record.rawlsian_score = max_min_normalization(
        rawlsian_reward, min_rawlsian, max_rawlsian
    )
    record.nash_social_score = max_min_normalization(
        nash_social_reward, min_nash_social_welfare, max_nash_social_welfare
    )

    # Check correctness
    row_text = row_action.lower() if row_action else ""
    col_text = col_action.lower() if col_action else ""

    if nash_target != ["none"]:
        record.nash_correct = check_correctness(nash_target, row_text, col_text)
    else:
        record.nash_correct = True  # No Nash equilibrium to match

    record.utilitarian_correct = check_correctness(utilitarian_target, row_text, col_text)
    record.rawlsian_correct = check_correctness(rawlsian_target, row_text, col_text)
    record.nash_social_correct = check_correctness(nash_social_target, row_text, col_text)

    return record


def process_eval_log(
    log_path: str,
) -> Optional[EvalRunData]:
    """Process a single .eval file into an EvalRunData object.

    Args:
        log_path: Path to the .eval file

    Returns:
        EvalRunData object with all processed samples, or None on error
    """
    eval_log, model_name = load_eval_log(log_path, header_only=False)
    if eval_log is None:
        return None

    samples = eval_log.samples or []

    # Process each sample
    records = []
    for sample in samples:
        record = process_sample(sample, model_name)
        records.append(record)

    return EvalRunData(
        log_id=Path(log_path).stem,
        log_path=log_path,
        model_name=model_name,
        num_samples=len(records),
        samples=records,
    )


def process_multiple_logs(
    log_paths: List[str],
    verbose: bool = True,
    enable_multiprocessing: bool = True,
) -> MultiEvalData:
    """Process multiple .eval files into a MultiEvalData object.

    Uses 20 parallel workers to process log files concurrently.

    Args:
        log_paths: List of paths to .eval files
        verbose: If True, print progress

    Returns:
        MultiEvalData object with all processed runs
    """
    multi_data = MultiEvalData()
    print_lock = Lock() if verbose else None
    completed_count = 0

    def process_single_log(log_path: str) -> tuple[str, Optional[EvalRunData]]:
        """Process a single log file and return (log_path, result)."""
        return log_path, process_eval_log(log_path)

    # Process logs in parallel with 20 workers
    if enable_multiprocessing:
        with ThreadPoolExecutor(max_workers=20) as executor:
                # Submit all tasks
            future_to_path = {
                executor.submit(process_single_log, log_path): log_path
                for log_path in log_paths
            }

            # Collect results as they complete
            results = {}
            for future in as_completed(future_to_path):
                log_path = future_to_path[future]
                try:
                    _, run_data = future.result()
                    results[log_path] = run_data
                    
                    if verbose and print_lock:
                        with print_lock:
                            completed_count += 1
                            log_name = Path(log_path).name
                            if run_data is not None:
                                valid = sum(1 for s in run_data.samples if s.parse_success)
                                print(
                                    f"Processing [{completed_count}/{len(log_paths)}]: {log_name} "
                                    f"-> {run_data.num_samples} samples ({valid} valid)"
                                )
                            else:
                                print(f"Processing [{completed_count}/{len(log_paths)}]: {log_name} -> Failed to process")
                except Exception as e:
                    if verbose and print_lock:
                        with print_lock:
                            completed_count += 1
                            log_name = Path(log_path).name
                            print(f"Processing [{completed_count}/{len(log_paths)}]: {log_name} -> Error: {e}")
    else:
        results = {}
        for log_path in log_paths:
            run_data = process_eval_log(log_path)
            results[log_path] = run_data

            if verbose:
                completed_count += 1
                log_name = Path(log_path).name
                if run_data is not None:
                    valid = sum(1 for s in run_data.samples if s.parse_success)
                    print(
                        f"Processing [{completed_count}/{len(log_paths)}]: {log_name} "
                        f"-> {run_data.num_samples} samples ({valid} valid)"
                    )
                else:
                    print(f"Processing [{completed_count}/{len(log_paths)}]: {log_name} -> Failed to process")


    # Add all results to multi_data (sequential to avoid thread-safety issues)
    for log_path in log_paths:
        run_data = results.get(log_path)
        if run_data is not None:
            multi_data.add_run(run_data)

    return multi_data


def compute_accuracy_by_game(
    run_data: EvalRunData,
) -> pd.DataFrame:
    """Compute accuracy metrics aggregated by game type.

    Args:
        run_data: Processed evaluation run data

    Returns:
        DataFrame with accuracy by game type
    """
    df = run_data.get_valid_samples_df()
    if df.empty:
        return pd.DataFrame()

    # Set categorical ordering
    df["game_type"] = pd.Categorical(
        df["game_type"], categories=GAME_TYPE_ORDER, ordered=True
    )

    grouped = df.groupby("game_type", observed=False)

    accuracy_df = grouped.agg({
        "nash_correct": "mean",
        "utilitarian_correct": "mean",
        "rawlsian_correct": "mean",
        "nash_social_correct": "mean",
    }).round(3)

    accuracy_df.columns = [
        "Nash Equilibrium",
        "Utilitarian",
        "Rawlsian",
        "Nash Social Welfare",
    ]

    return accuracy_df


def compute_welfare_by_game(
    run_data: EvalRunData,
) -> pd.DataFrame:
    """Compute welfare scores aggregated by game type.

    Args:
        run_data: Processed evaluation run data

    Returns:
        DataFrame with welfare scores by game type
    """
    df = run_data.get_valid_samples_df()
    if df.empty:
        return pd.DataFrame()

    df["game_type"] = pd.Categorical(
        df["game_type"], categories=GAME_TYPE_ORDER, ordered=True
    )

    grouped = df.groupby("game_type", observed=False)

    welfare_df = grouped.agg({
        "utilitarian_score": "mean",
        "rawlsian_score": "mean",
        "nash_social_score": "mean",
    }).round(3)

    welfare_df.columns = [
        "utilitarian_efficiency",
        "rawlsian_efficiency",
        "nash_social_welfare_efficiency",
    ]

    return welfare_df


def compute_overall_accuracy(
    run_data: EvalRunData,
) -> pd.Series:
    """Compute overall accuracy metrics.

    Args:
        run_data: Processed evaluation run data

    Returns:
        Series with overall accuracy metrics
    """
    df = run_data.get_valid_samples_df()
    if df.empty:
        return pd.Series({
            "nash": 0.0,
            "utilitarian": 0.0,
            "rawlsian": 0.0,
            "nash_social_welfare": 0.0,
        })

    return pd.Series({
        "nash": df["nash_correct"].mean(),
        "utilitarian": df["utilitarian_correct"].mean(),
        "rawlsian": df["rawlsian_correct"].mean(),
        "nash_social_welfare": df["nash_social_correct"].mean(),
    })
