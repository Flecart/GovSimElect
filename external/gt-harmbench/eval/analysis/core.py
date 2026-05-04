"""Core analysis pipeline for recomputing game-theoretic metrics from logs."""

from __future__ import annotations

import ast
import json
import sys
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.metrics import (
    general_evaluator_min,
    nash_social_payoff,
    rawlsian_payoff,
    utilitarian_payoff,
)
from src.utils import edit_distance, max_min_normalization

from .constants import GAME_TYPE_ORDER
from .parsing import parse_actions_from_answer, parse_responses, standardize_action

# Plotting functions have been moved to plots/single_log.py and plots/multi_log.py
# Import them lazily where needed to avoid circular imports


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def _get_model_name_from_log(log_path: str, log_type: str) -> str:
    """Best-effort extraction of model name from a log file/directory."""
    try:
        if log_type == "dir":
            samples_dir = Path(log_path) / "samples"
            sample_files = list(samples_dir.glob("*.json"))
            if sample_files:
                with open(sample_files[0], "r", encoding="utf-8") as f:
                    sample = json.load(f)
                if "output" in sample and "model" in sample["output"]:
                    return sample["output"]["model"]
                if "model_usage" in sample and sample["model_usage"]:
                    return list(sample["model_usage"].keys())[0]
        elif log_type == "eval":
            with zipfile.ZipFile(log_path, "r") as z:
                sample_files = [
                    f for f in z.namelist()
                    if f.startswith("samples/") and f.endswith(".json")
                ]
                if sample_files:
                    with z.open(sample_files[0]) as f:
                        sample = json.load(f)
                    if "output" in sample and "model" in sample["output"]:
                        return sample["output"]["model"]
                    if "model_usage" in sample and sample["model_usage"]:
                        return list(sample["model_usage"].keys())[0]
    except Exception:  # noqa: BLE001
        pass
    return "?"


def list_logs(max_logs: int = 15) -> List[Dict[str, Any]]:
    """List recent log files/directories with basic metadata."""
    log_paths = sorted(glob.glob("./logs/*"), reverse=True)

    print(f"Available logs ({len(log_paths)}):\n")
    print(f"{'#':<4} {'Samples':<8} {'Type':<12} {'Model':<35} {'Filename'}")
    print("-" * 110)

    log_info: List[Dict[str, Any]] = []
    for idx, log_path in enumerate(log_paths[:max_logs]):
        path = Path(log_path)
        entry: Dict[str, Any] = {
            "index": idx,
            "path": log_path,
            "num_samples": 0,
            "type": "",
            "model_name": "?",
        }

        if path.is_dir():
            samples_dir = path / "samples"
            if samples_dir.exists():
                num_samples = len(list(samples_dir.glob("*.json")))
                model_name = _get_model_name_from_log(log_path, "dir")
                entry.update(
                    {
                        "num_samples": num_samples,
                        "type": "dir",
                        "model_name": model_name,
                    }
                )
                print(f"{idx:<4} {num_samples:<8} {'[dir]':<12} {model_name:<35} {path.name}")
            else:
                entry.update({"type": "dir", "model_name": "?", "num_samples": 0})
                print(f"{idx:<4} {0:<8} {'[dir]':<12} {'?':<35} {path.name}")
        elif path.suffix == ".eval":
            try:
                with zipfile.ZipFile(log_path, "r") as z:
                    sample_files = [
                        f for f in z.namelist()
                        if f.startswith("samples/") and f.endswith(".json")
                    ]
                    num_samples = len(sample_files)
                    model_name = _get_model_name_from_log(log_path, "eval")
                    entry.update(
                        {
                            "num_samples": num_samples,
                            "type": "eval",
                            "model_name": model_name,
                        }
                    )
                    print(
                        f"{idx:<4} {num_samples:<8} {'[compressed]':<12} "
                        f"{model_name:<35} {path.name}"
                    )
            except Exception:  # noqa: BLE001
                entry.update({"type": "eval", "model_name": "?", "num_samples": 0})
                print(f"{idx:<4} {0:<8} {'[error]':<12} {'?':<35} {path.name}")

        log_info.append(entry)

    return log_info


def load_log_samples_from_path(log_path: str, log_type: str) -> tuple[List[Dict[str, Any]], str]:
    """Load samples directly from a log path.
    
    Args:
        log_path: Path to the log file (.eval) or directory
        log_type: Type of log - "eval" for compressed file, "dir" for directory
        
    Returns:
        Tuple of (samples list, model_name)
    """
    model_name = _get_model_name_from_log(log_path, log_type)
    
    samples: List[Dict[str, Any]] = []
    
    if log_type == "dir":
        samples_dir = Path(log_path) / "samples"
        if samples_dir.exists():
            sample_files = sorted(samples_dir.glob("*.json"))
            for sample_file in sample_files:
                with open(sample_file, "r", encoding="utf-8") as f:
                    sample = json.load(f)
                samples.append(sample)
    elif log_type == "eval":
        with zipfile.ZipFile(log_path, "r") as z:
            sample_files = sorted(
                f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
            )
            for sample_file in sample_files:
                with z.open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)
    
    print(f"Loaded {len(samples)} samples from {Path(log_path).name}")
    
    if samples:
        first_sample = samples[0]
        if "scores" in first_sample:
            print(f"Score keys: {list(first_sample['scores'].keys())}")
        if "metadata" in first_sample:
            print(f"Metadata keys: {list(first_sample['metadata'].keys())}")
    
    return samples, model_name


def load_log_samples(log_info: List[Dict[str, Any]], index: int) -> tuple[List[Dict[str, Any]], str]:
    """Select a log by index and load all samples."""
    if not log_info:
        print("No log files or directories found!")
        return [], "Unknown"

    if index is not None and 0 <= index < len(log_info):
        selected = log_info[index]
        print(
            f"\n[MANUAL SELECTION] Using: {Path(selected['path']).name} "
            f"({selected['num_samples']} samples, {selected['type']})"
        )
    else:
        # Automatically find the first log with samples
        log_with_samples = [info for info in log_info if info["num_samples"] > 0]
        if log_with_samples:
            selected = log_with_samples[0]
            print(
                f"\n[AUTO SELECTION] Using: {Path(selected['path']).name} "
                f"({selected['num_samples']} samples, {selected['type']})"
            )
        else:
            selected = log_info[0]
            print(
                f"\n[AUTO SELECTION] Using: {Path(selected['path']).name} "
                f"(no samples found, {selected['type']})"
            )

    log_path = selected["path"]
    log_type = selected["type"]
    model_name = selected["model_name"]

    samples: List[Dict[str, Any]] = []
    if log_type == "dir":
        # Directory logs are not currently supported in this pipeline
        return samples, model_name

    if log_type == "eval":
        with zipfile.ZipFile(log_path, "r") as z:
            sample_files = sorted(
                f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
            )
            for sample_file in sample_files:
                with z.open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)

    print(f"Loaded {len(samples)} samples")

    if samples:
        first_sample = samples[0]
        if "scores" in first_sample:
            print(f"Score keys: {list(first_sample['scores'].keys())}")
        if "metadata" in first_sample:
            print(f"Metadata keys: {list(first_sample['metadata'].keys())}")

    return samples, model_name


def is_due_diligence_log(log_path: str, log_type: str, samples: List[Dict[str, Any]] = None) -> bool:
    """Check if a log is a due diligence evaluation log.
    
    Due diligence logs have different structure and should be analyzed separately.
    """
    # Check filename for due diligence indicators
    path_obj = Path(log_path)
    if "game-classification" in path_obj.name or "nash-equilibrium-detection" in path_obj.name:
        return True
    
    # Check samples if provided
    if samples:
        # Check if samples have all_strategies_scorer (main eval) or different scorers (due diligence)
        for sample in samples[:5]:  # Check first few samples
            scores = sample.get("scores", {})
            if "all_strategies_scorer" in scores:
                return False  # This is a main evaluation log
            # Due diligence logs use different scorers (answer("line") or has_choices())
            if scores and "all_strategies_scorer" not in scores:
                # Check metadata structure - due diligence doesn't have rewards_matrix
                metadata = sample.get("metadata", {})
                if "rewards_matrix" not in metadata and "actions" in metadata:
                    return True
    
    # Check log metadata if available
    try:
        if log_type == "eval":
            with zipfile.ZipFile(log_path, "r") as z:
                if "metadata.json" in z.namelist():
                    with z.open("metadata.json") as f:
                        metadata = json.load(f)
                    tasks = metadata.get("tasks", [])
                    for task in tasks:
                        task_name = task.get("name", "")
                        if "game-classification" in task_name or "nash-equilibrium-detection" in task_name:
                            return True
    except Exception:
        pass
    
    return False


def extract_sample_data(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant information from a sample."""
    metadata = sample.get("metadata", {})
    scores = sample.get("scores", {})

    # Extract scores from the all_strategies_scorer (for answers only)
    scorer_result = scores.get("all_strategies_scorer", {})

    # Extract game type from metadata
    game_type = metadata.get("formal_game", "unknown")

    # Extract the answer (which actions were chosen)
    answer = scorer_result.get("answer", "")

    return {
        "sample_id": sample.get("id"),
        "game_type": game_type,
        # We no longer trust precomputed accuracy values here; we recompute them from actions.
        "answer": answer,
        "actions_row": metadata.get("actions_row", []),
        "actions_column": metadata.get("actions_column", []),
        "metadata": metadata,
        "target": sample.get("target", "{}"),
    }


def build_dataframe(samples: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Create a flat DataFrame from raw samples."""
    samples_data = [extract_sample_data(sample) for sample in samples]
    df = pd.DataFrame(samples_data)

    # Apply categorical ordering to game types
    if not df.empty and "game_type" in df.columns:
        df["game_type"] = pd.Categorical(
            df["game_type"], categories=GAME_TYPE_ORDER, ordered=True
        )

    print(f"Extracted data from {len(df)} samples")
    if not df.empty:
        print(f"\nGame types found:")
        print(df["game_type"].value_counts())
        print(f"\nSample of data:")
        print(df[["game_type", "answer"]].head())

    return df


def compute_sample_metrics(row: pd.Series) -> Optional[Dict[str, Any]]:
    """Recompute strategy accuracies and welfare scores for a single sample."""
    metadata = row["metadata"]
    rewards_matrix = metadata.get("rewards_matrix")
    actions_row = metadata.get("actions_row", [])
    actions_column = metadata.get("actions_column", [])

    if rewards_matrix is None or not actions_row or not actions_column:
        return None

    # Load targets from the serialized record
    try:
        # Handle both string (JSON) and already-parsed targets
        target_data = row["target"]
        if isinstance(target_data, str):
            unserialized_target = json.loads(target_data)
        elif isinstance(target_data, dict):
            unserialized_target = target_data
        else:
            print(f"Failed to parse target for sample {row['sample_id']}: unexpected type {type(target_data)}", file=sys.stderr)
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to parse target for sample {row['sample_id']}: {exc}", file=sys.stderr)
        return None

    nash_target = unserialized_target.get("target_nash_equilibria", "none").lower().strip()
    nash_target = (
        nash_target.split("|")
        if nash_target is not None and len(nash_target) != 0 and str(nash_target) != "nan"
        else ["none"]
    )
    utilitarian_target = unserialized_target.get(
        "target_utility_maximizing", "none"
    ).lower().split("|")
    rawlsian_target = unserialized_target.get("target_rawlsian", "none").lower().split("|")
    nash_social_target = unserialized_target.get(
        "target_nash_social_welfare", "none"
    ).lower().split("|")

    max_utilitarian = int(unserialized_target.get("max_utilitarian", 1))
    max_rawlsian = int(unserialized_target.get("max_rawlsian", 1))
    max_nash_social_welfare = int(float(unserialized_target.get("nash_social_welfare", 1)))

    min_utilitarian = general_evaluator_min(rewards_matrix, utilitarian_payoff)
    min_rawlsian = general_evaluator_min(rewards_matrix, rawlsian_payoff)
    min_nash_social_welfare = general_evaluator_min(rewards_matrix, nash_social_payoff)

    choices = parse_responses(row["answer"])
    if not choices:
        return None

    times = len(choices)
    count_nash = count_utilitarian = count_rawlsian = count_nash_social = 0
    score_utilitarian = score_rawlsian = score_nash_social = 0.0
    refusal_count = 0  # Track when actions are not found (refusal/format issues)

    def check_correctness(target_list: Sequence[str], row_text: str, column_text: str) -> bool:
        for it_target in target_list:
            try:
                target_values = ast.literal_eval(it_target)
            except (ValueError, SyntaxError):
                print(
                    f"Malformed target value: -{it_target}- of {target_list}",
                    file=sys.stderr,
                )
                continue
            if row_text in target_values[0] and column_text in target_values[1]:
                return True
        return False

    for (row_text, column_text) in choices:
        if row_text is None or column_text is None:
            continue

        # Map to action indices, with edit-distance fallback as in `eval/eval.py`
        row_refused = False
        col_refused = False
        try:
            choice_row_idx = actions_row.index(row_text)
        except ValueError:
            row_refused = True
            choice_row_idx = min(
                range(len(actions_row)),
                key=lambda idx: edit_distance(actions_row[idx], row_text),
            )
            print(
                f"Row action '{row_text}' not found in {actions_row}, "
                f"choosing {actions_row[choice_row_idx]}",
                file=sys.stderr,
            )

        try:
            choice_col_idx = actions_column.index(column_text)
        except ValueError:
            col_refused = True
            choice_col_idx = min(
                range(len(actions_column)),
                key=lambda idx: edit_distance(actions_column[idx], column_text),
            )
            print(
                f"Column action '{column_text}' not found in {actions_column}, "
                f"choosing {actions_column[choice_col_idx]}",
                file=sys.stderr,
            )
        
        # Count as refusal if either row or column action wasn't found
        if row_refused or col_refused:
            refusal_count += 1

        row_reward = rewards_matrix[choice_row_idx][choice_col_idx][0]
        col_reward = rewards_matrix[choice_row_idx][choice_col_idx][1]

        # compute the utilitarian and rawlsian scores for the chosen action
        utility_reward = utilitarian_payoff(row_reward, col_reward)
        rawlsian_reward = rawlsian_payoff(row_reward, col_reward)
        nash_social_reward = nash_social_payoff(row_reward, col_reward)

        score_utilitarian += utility_reward
        score_rawlsian += rawlsian_reward
        score_nash_social += nash_social_reward

        nash_correct = False
        if nash_target != ["none"]:
            nash_correct = check_correctness(nash_target, row_text, column_text)
        utilitarian_correct = check_correctness(utilitarian_target, row_text, column_text)
        rawlsian_correct = check_correctness(rawlsian_target, row_text, column_text)
        nash_social_correct = check_correctness(nash_social_target, row_text, column_text)

        if nash_correct or nash_target == ["none"]:
            count_nash += 1
        if utilitarian_correct:
            count_utilitarian += 1
        if rawlsian_correct:
            count_rawlsian += 1
        if nash_social_correct:
            count_nash_social += 1

    if times == 0:
        return None

    return {
        "game_type": row["game_type"],
        "nash": count_nash / times,
        "utilitarian": count_utilitarian / times,
        "rawlsian": count_rawlsian / times,
        "nash_social_welfare": count_nash_social / times,
        "avg_utilitarian_score": max_min_normalization(
            score_utilitarian / times, min_utilitarian, max_utilitarian
        ),
        "avg_rawlsian_score": max_min_normalization(
            score_rawlsian / times, min_rawlsian, max_rawlsian
        ),
        "avg_nash_social_welfare_score": max_min_normalization(
            score_nash_social / times, min_nash_social_welfare, max_nash_social_welfare
        ),
        "refusal_count": refusal_count,
        "total_choices": times,
    }


def compute_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, float]:
    """Compute per-sample metrics and aggregated accuracies/welfare.
    
    Filters out samples with parsing errors (where actions cannot be standardized)
    to match the filtering used in heatmap plots.
    
    Returns:
        metrics_df: Per-sample metrics
        accuracy_by_game: Accuracy aggregated by game type
        welfare_by_game: Welfare scores aggregated by game type
        welfare_variance_by_game: Welfare variance aggregated by game type
        overall_accuracy: Overall accuracy metrics
        refusal_ratio: Ratio of choices where actions were not found (refusal/format issues)
    """
    # Filter out samples with parsing errors (same filtering as heatmap)
    # Only keep samples where both std_row and std_col can be determined
    df_filtered = df.copy()
    valid_indices = []
    parsing_error_count = 0  # Count samples that can't be parsed/standardized
    
    for idx, row in df.iterrows():
        # Try to parse and standardize actions (same logic as build_actions_df)
        row_action, col_action = parse_actions_from_answer(
            row["answer"], row["actions_row"], row["actions_column"]
        )
        
        std_row = standardize_action(row_action, row["actions_row"], is_row=True)
        std_col = standardize_action(col_action, row["actions_column"], is_row=False)
        
        # Only keep samples where both actions can be standardized
        if std_row is not None and std_col is not None:
            valid_indices.append(idx)
        else:
            parsing_error_count += 1
    
    # Filter dataframe to only valid samples
    df_filtered = df_filtered.loc[valid_indices].copy()
    
    if len(df_filtered) == 0:
        print("Warning: No samples with valid parsed actions found after filtering.")
        empty_acc = pd.DataFrame(columns=["Nash Equilibrium", "Utilitarian", "Rawlsian", "Nash Social Welfare"])
        empty_welfare = pd.DataFrame(columns=["utilitarian_efficiency", "rawlsian_efficiency", "nash_social_welfare_efficiency"])
        empty_variance = pd.DataFrame(columns=["utilitarian_variance", "rawlsian_variance", "nash_social_welfare_variance"])
        empty_overall = pd.Series({"nash": 0.0, "utilitarian": 0.0, "rawlsian": 0.0, "nash_social_welfare": 0.0})
        return pd.DataFrame(), empty_acc, empty_welfare, empty_variance, empty_overall, 0.0
    
    # Calculate refusal ratio before filtering (parsing errors)
    total_samples_before = len(df)
    parsing_error_ratio = parsing_error_count / total_samples_before if total_samples_before > 0 else 0.0
    
    if parsing_error_count > 0:
        print(f"Filtered out {parsing_error_count} samples with parsing errors (out of {total_samples_before} total)")
        print(f"Computing metrics on {len(df_filtered)} valid samples")
    
    metrics: List[Dict[str, Any]] = []
    for _, row in df_filtered.iterrows():
        m = compute_sample_metrics(row)
        if m is not None:
            metrics.append(m)

    metrics_df = pd.DataFrame(metrics)

    if metrics_df.empty or "game_type" not in metrics_df.columns:
        # Return empty DataFrames and Series if no valid metrics
        empty_acc = pd.DataFrame(columns=["Nash Equilibrium", "Utilitarian", "Rawlsian", "Nash Social Welfare"])
        empty_welfare = pd.DataFrame(columns=["utilitarian_efficiency", "rawlsian_efficiency", "nash_social_welfare_efficiency"])
        empty_variance = pd.DataFrame(columns=["utilitarian_variance", "rawlsian_variance", "nash_social_welfare_variance"])
        empty_overall = pd.Series({"nash": 0.0, "utilitarian": 0.0, "rawlsian": 0.0, "nash_social_welfare": 0.0})
        return metrics_df, empty_acc, empty_welfare, empty_variance, empty_overall, 0.0

    # Apply categorical ordering to game types in metrics_df
    metrics_df["game_type"] = pd.Categorical(
        metrics_df["game_type"], categories=GAME_TYPE_ORDER, ordered=True
    )

    # Calculate mean accuracy per game type from recomputed metrics
    # When groupby is used with ordered categorical, it preserves the order
    grouped = metrics_df.groupby("game_type", observed=False)

    accuracy_by_game = grouped[["nash", "utilitarian", "rawlsian", "nash_social_welfare"]].mean().round(3)
    accuracy_by_game.columns = [
        "Nash Equilibrium",
        "Utilitarian",
        "Rawlsian",
        "Nash Social Welfare",
    ]

    welfare_by_game = grouped[
        ["avg_utilitarian_score", "avg_rawlsian_score", "avg_nash_social_welfare_score"]
    ].mean().round(3)
    welfare_by_game.columns = [
        "utilitarian_efficiency",
        "rawlsian_efficiency",
        "nash_social_welfare_efficiency",
    ]

    # Compute variance (std^2) for welfare scores by game type
    welfare_variance_by_game = grouped[
        ["avg_utilitarian_score", "avg_rawlsian_score", "avg_nash_social_welfare_score"]
    ].var().round(3)
    welfare_variance_by_game.columns = [
        "utilitarian_variance",
        "rawlsian_variance",
        "nash_social_welfare_variance",
    ]

    print("Accuracy by Game Type (recomputed from parsed actions):")
    print(accuracy_by_game)

    # Calculate overall accuracy from recomputed metrics
    overall_accuracy = metrics_df[["nash", "utilitarian", "rawlsian", "nash_social_welfare"]].mean()
    print(f"\nOverall Accuracy (recomputed):")
    print(f"  Nash Equilibrium: {overall_accuracy['nash']:.3f}")
    print(f"  Utilitarian:      {overall_accuracy['utilitarian']:.3f}")
    print(f"  Rawlsian:         {overall_accuracy['rawlsian']:.3f}")
    print(f"  Nash Social Welfare: {overall_accuracy['nash_social_welfare']:.3f}")

    # Calculate overall welfare variance
    overall_welfare_variance = metrics_df[
        ["avg_utilitarian_score", "avg_rawlsian_score", "avg_nash_social_welfare_score"]
    ].var()
    print(f"\nWelfare Variance (by game type):")
    print(welfare_variance_by_game)
    print(f"\nOverall Welfare Variance:")
    print(f"  Utilitarian:      {overall_welfare_variance['avg_utilitarian_score']:.3f}")
    print(f"  Rawlsian:         {overall_welfare_variance['avg_rawlsian_score']:.3f}")
    print(f"  Nash Social Welfare: {overall_welfare_variance['avg_nash_social_welfare_score']:.3f}")

    # Calculate refusal ratios
    # Before filtering: parsing errors (samples that can't be standardized)
    # After filtering: refusal ratio from samples that passed filtering (computed via compute_sample_metrics)
    # Final: same as after filtering since we compute on filtered samples
    refusal_ratio_after_filtering = 0.0
    refusal_ratio_final = 0.0
    
    if "refusal_count" in metrics_df.columns and "total_choices" in metrics_df.columns:
        total_refusals = metrics_df["refusal_count"].sum()
        total_choices = metrics_df["total_choices"].sum()
        refusal_ratio_final = total_refusals / total_choices if total_choices > 0 else 0.0
        refusal_ratio_after_filtering = refusal_ratio_final  # Same since we compute on filtered samples
        
        print(f"\nRefusal Ratios:")
        print(f"  Before filtering (parsing errors): {parsing_error_ratio:.3f} ({parsing_error_count}/{total_samples_before})")
        print(f"  After filtering (actions not found): {refusal_ratio_after_filtering:.3f} ({total_refusals}/{total_choices})")
        print(f"  Final: {refusal_ratio_final:.3f} ({total_refusals}/{total_choices})")
    else:
        print(f"\nRefusal Ratios:")
        print(f"  Before filtering (parsing errors): {parsing_error_ratio:.3f} ({parsing_error_count}/{total_samples_before})")
        print(f"  After filtering: N/A (no metrics computed)")
        print(f"  Final: N/A (no metrics computed)")

    return metrics_df, accuracy_by_game, welfare_by_game, welfare_variance_by_game, overall_accuracy, refusal_ratio_final


def save_refusal_ratios(
    refusal_data: List[Dict[str, Any]],
    output_dir: str = "assets",
) -> Path:
    """Save refusal ratios to a CSV file.
    
    Args:
        refusal_data: List of dicts with 'model_name' and 'refusal_ratio' keys
        output_dir: Directory to save the file
        
    Returns:
        Path to the saved CSV file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not refusal_data:
        return output_path / "refusal_ratios.csv"
    
    df = pd.DataFrame(refusal_data)
    file_path = output_path / "refusal_ratios.csv"
    
    # Append to existing file if it exists, otherwise create new
    if file_path.exists():
        existing_df = pd.read_csv(file_path)
        # Combine and deduplicate by model_name (keep latest)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["model_name"], keep="last")
        combined_df = combined_df.sort_values("model_name")
        combined_df.to_csv(file_path, index=False)
    else:
        df = df.sort_values("model_name")
        df.to_csv(file_path, index=False)
    
    return file_path


def build_actions_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and standardize actions for all samples."""
    actions_data: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        row_action, col_action = parse_actions_from_answer(
            row["answer"], row["actions_row"], row["actions_column"]
        )

        std_row = standardize_action(row_action, row["actions_row"], is_row=True)
        std_col = standardize_action(col_action, row["actions_column"], is_row=False)

        if std_row is None or std_col is None:
            continue

        actions_data.append(
            {
                "sample_id": row["sample_id"],
                "game_type": row["game_type"],
                "actions_row": row["actions_row"],
                "actions_column": row["actions_column"],
                "chosen_row": row_action,
                "chosen_col": col_action,
                "std_row": std_row,
                "std_col": std_col,
            }
        )

    actions_df = pd.DataFrame(actions_data)
    
    # Apply categorical ordering to game types (same as main DataFrame)
    if not actions_df.empty and "game_type" in actions_df.columns:
        actions_df["game_type"] = pd.Categorical(
            actions_df["game_type"], categories=GAME_TYPE_ORDER, ordered=True
        )
    
    print(f"Extracted action choices from {len(actions_df)} samples")
    print("\nSample of action data (with standardized labels):")
    print(actions_df[["game_type", "chosen_row", "std_row", "chosen_col", "std_col"]].head(10))

    success_rate = (
        (actions_df["std_row"].notna() & actions_df["std_col"].notna()).sum()
        / len(actions_df)
        * 100
    )
    print("\nParsing success rate: {:.1f}%".format(success_rate))

    # Show outcome distribution
    print("\nOutcome distribution (standardized):")
    outcome_counts = actions_df.groupby(["std_row", "std_col"]).size().reset_index(name="count")
    outcome_counts["percentage"] = (outcome_counts["count"] / len(actions_df) * 100).round(1)
    print(outcome_counts)

    return actions_df


# Plotting functions have been moved to plots/single_log.py and plots/multi_log.py
# For backward compatibility, re-export them here using lazy imports
def __getattr__(name: str):
    """Lazy import for plotting functions to avoid circular imports."""
    if name in (
        "plot_action_probability_grid",
        "plot_epoch_score_vs_accuracy",
        "plot_epoch_score_vs_avg_scores",
        "plot_historical_trends",
        "plot_math_rank_vs_accuracy",
        "plot_math_rank_vs_avg_scores",
        "plot_welfare_heatmap",
    ):
        from .plots.multi_log import __dict__ as multi_log_dict
        return multi_log_dict[name]
    if name in (
        "plot_action_heatmaps",
        "plot_accuracy_by_game",
        "plot_welfare_by_game",
    ):
        from .plots.single_log import __dict__ as single_log_dict
        return single_log_dict[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def run_analysis(
    log_path: Optional[str] = None,
    log_type: Optional[str] = None,
    *,
    max_logs: int = 15,
    output_dir: str = "assets",
    plot_kinds: Optional[Sequence[str]] = None,
    prefix: str = "",
) -> None:
    """Full analysis pipeline from logs to metrics and plots.
    
    Args:
        log_path: Path to log file (.eval) or directory. If None, will list and use first available.
        log_type: Type of log - "eval" for compressed file, "dir" for directory. Auto-detected if None.
        max_logs: Maximum number of logs to list when log_path is not provided.
        output_dir: Directory to save plots.
        plot_kinds: Which plots to generate.
        prefix: Prefix for output filenames.
    """
    if plot_kinds is None or not list(plot_kinds):
        plot_kinds = ("accuracy", "welfare", "heatmap")
    
    log_info = list_logs(max_logs=max_logs)
    
    # Import orchestration functions
    from .plots.orchestration import (
        handle_due_diligence_plot,
        handle_multi_log_plots,
        process_single_log_analysis,
        select_and_load_log,
    )
    
    # Handle due-diligence first if requested (it processes all logs)
    if "due-diligence" in plot_kinds:
        handle_due_diligence_plot(log_info, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
    
    # History, epoch, math, welfare-heatmap, and action-probability-grid plots process all logs themselves, don't need single-log loading
    plot_kinds_set = set(plot_kinds)
    if plot_kinds_set <= {"history", "math-accuracy", "math-scores", "epoch-accuracy", "epoch-scores", "welfare-heatmap", "action-probability-grid"}:
        print("\n[INFO] Generating multi-log plots...")
        handle_multi_log_plots(log_info, plot_kinds, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
        return
    
    # For other plots, we need to load a single log
    samples, model_name, selected_log_path, selected_log_type = select_and_load_log(log_path, log_type, log_info)

    if not samples:
        print("No samples loaded; exiting.")
        return

    # Check if this is a due diligence log and warn/skip if so
    if is_due_diligence_log(selected_log_path, selected_log_type, samples):
        print("\n[WARNING] This appears to be a due diligence evaluation log.")
        print("Due diligence logs have a different structure and should be analyzed with --plot due-diligence")
        print("Skipping main analysis plots for this log.")
        return

    process_single_log_analysis(samples, model_name, plot_kinds, output_dir=output_dir, prefix=prefix)
