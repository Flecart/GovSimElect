"""Data schemas for the plots module.

Defines the intermediary data structures used between loading and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class SampleRecord:
    """Processed data from a single evaluation sample."""

    sample_id: str
    model_name: str
    game_type: str

    # Raw parsed actions from model response
    row_action: Optional[str] = None
    col_action: Optional[str] = None

    # Standardized actions (UP/DOWN, LEFT/RIGHT)
    std_row: Optional[str] = None
    std_col: Optional[str] = None

    # Correctness flags
    nash_correct: bool = False
    utilitarian_correct: bool = False
    rawlsian_correct: bool = False
    nash_social_correct: bool = False

    # Welfare scores (normalized)
    utilitarian_score: float = 0.0
    rawlsian_score: float = 0.0
    nash_social_score: float = 0.0

    # Metadata for debugging
    answer_text: str = ""
    actions_row: List[str] = field(default_factory=list)
    actions_column: List[str] = field(default_factory=list)
    rewards_matrix: Optional[List] = None

    # Parsing status
    parse_success: bool = False
    parse_error: Optional[str] = None


@dataclass
class EvalRunData:
    """Aggregated data from a single evaluation run (one .eval file)."""

    log_id: str
    log_path: str
    model_name: str
    num_samples: int

    # All sample records
    samples: List[SampleRecord] = field(default_factory=list)

    # Pre-computed DataFrames for convenience
    _samples_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    _accuracy_by_game: Optional[pd.DataFrame] = field(default=None, repr=False)
    _welfare_by_game: Optional[pd.DataFrame] = field(default=None, repr=False)
    _overall_accuracy: Optional[pd.Series] = field(default=None, repr=False)

    @property
    def samples_df(self) -> pd.DataFrame:
        """Get samples as a DataFrame."""
        if self._samples_df is None:
            self._samples_df = pd.DataFrame([
                {
                    "sample_id": s.sample_id,
                    "model_name": s.model_name,
                    "game_type": s.game_type,
                    "row_action": s.row_action,
                    "col_action": s.col_action,
                    "std_row": s.std_row,
                    "std_col": s.std_col,
                    "nash_correct": s.nash_correct,
                    "utilitarian_correct": s.utilitarian_correct,
                    "rawlsian_correct": s.rawlsian_correct,
                    "nash_social_correct": s.nash_social_correct,
                    "utilitarian_score": s.utilitarian_score,
                    "rawlsian_score": s.rawlsian_score,
                    "nash_social_score": s.nash_social_score,
                    "parse_success": s.parse_success,
                }
                for s in self.samples
            ])
        return self._samples_df

    def get_valid_samples_df(self) -> pd.DataFrame:
        """Get only samples that were successfully parsed."""
        df = self.samples_df
        return df[df["parse_success"] == True].copy()


@dataclass
class MultiEvalData:
    """Aggregated data from multiple evaluation runs."""

    runs: List[EvalRunData] = field(default_factory=list)

    def add_run(self, run: EvalRunData) -> None:
        """Add an evaluation run to the collection."""
        self.runs.append(run)

    def get_all_samples_df(self) -> pd.DataFrame:
        """Get all samples from all runs as a single DataFrame."""
        if not self.runs:
            return pd.DataFrame()
        dfs = [run.samples_df for run in self.runs if not run.samples_df.empty]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def get_models(self) -> List[str]:
        """Get list of unique model names."""
        return list(set(run.model_name for run in self.runs if run.model_name))

    def get_run_by_model(self, model_name: str) -> Optional[EvalRunData]:
        """Get the first run matching a model name."""
        for run in self.runs:
            if run.model_name == model_name:
                return run
        return None
