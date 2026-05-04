"""Load .eval files using inspect_ai.

This module handles loading evaluation logs from disk using the inspect_ai library.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inspect_ai.log import EvalLog, read_eval_log


def list_eval_logs(
    logs_dir: str = "logs",
    max_logs: int = 50,
) -> List[Dict[str, Any]]:
    """List available .eval log files with metadata.

    Args:
        logs_dir: Directory containing .eval files
        max_logs: Maximum number of logs to list

    Returns:
        List of dicts with log metadata (path, model_name, num_samples, etc.)
    """
    log_paths = sorted(glob.glob(f"{logs_dir}/*.eval"), reverse=True)

    log_info: List[Dict[str, Any]] = []
    for log_path in log_paths[:max_logs]:
        path = Path(log_path)
        entry: Dict[str, Any] = {
            "path": log_path,
            "filename": path.name,
            "num_samples": 0,
            "model_name": None,
            "log_type": None,
        }

        try:
            # Read header only for efficiency
            eval_log = read_eval_log(log_path, header_only=True)
            entry["num_samples"] = eval_log.results.total_samples if eval_log.results else 0

            # Try to extract model name from eval metadata
            if hasattr(eval_log, "eval") and eval_log.eval:
                entry["model_name"] = getattr(eval_log.eval, "model", None)

            # Determine log type from filename
            if "game-classification" in path.name:
                entry["log_type"] = "due_diligence_classification"
            elif "nash-equilibrium-detection" in path.name:
                entry["log_type"] = "due_diligence_nash"
            else:
                entry["log_type"] = "evaluation"

        except Exception as e:
            entry["error"] = str(e)

        log_info.append(entry)

    return log_info


def load_eval_log(
    log_path: str,
    header_only: bool = False,
) -> Tuple[Optional[EvalLog], str]:
    """Load a single .eval file.

    Args:
        log_path: Path to the .eval file
        header_only: If True, only load header (no samples)

    Returns:
        Tuple of (EvalLog object or None, model_name)
    """
    try:
        eval_log = read_eval_log(log_path, header_only=header_only)

        # Extract model name
        model_name = "unknown"
        if hasattr(eval_log, "eval") and eval_log.eval:
            model_name = getattr(eval_log.eval, "model", "unknown") or "unknown"

        return eval_log, model_name

    except Exception as e:
        print(f"Error loading {log_path}: {e}")
        return None, "unknown"


def filter_evaluation_logs(
    log_info: List[Dict[str, Any]],
    log_type: Optional[str] = None,
    min_samples: int = 1,
) -> List[Dict[str, Any]]:
    """Filter log info list by type and sample count.

    Args:
        log_info: List of log metadata dicts
        log_type: If provided, only include logs of this type
        min_samples: Minimum number of samples required

    Returns:
        Filtered list of log metadata dicts
    """
    filtered = []
    for entry in log_info:
        if entry.get("num_samples", 0) < min_samples:
            continue
        if log_type and entry.get("log_type") != log_type:
            continue
        if "error" in entry:
            continue
        filtered.append(entry)
    return filtered
