"""Data loading and processing for evaluation plots.

This module provides functions to load .eval files and process them into
an intermediary data format suitable for plotting.
"""

from .loader import filter_evaluation_logs, list_eval_logs, load_eval_log
from .processor import (
    GAME_TYPE_ORDER,
    compute_accuracy_by_game,
    compute_overall_accuracy,
    compute_welfare_by_game,
    process_eval_log,
    process_multiple_logs,
)
from .schemas import EvalRunData, MultiEvalData, SampleRecord

__all__ = [
    # Loader
    "list_eval_logs",
    "load_eval_log",
    "filter_evaluation_logs",
    # Processor
    "process_eval_log",
    "process_multiple_logs",
    "compute_accuracy_by_game",
    "compute_welfare_by_game",
    "compute_overall_accuracy",
    "GAME_TYPE_ORDER",
    # Schemas
    "SampleRecord",
    "EvalRunData",
    "MultiEvalData",
]
