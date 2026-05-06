"""Base classes and utilities for plot renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from ..data.schemas import EvalRunData, MultiEvalData


# Set default plot style
def setup_plot_style() -> None:
    """Configure matplotlib and seaborn for consistent styling."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)


class BasePlotRenderer(ABC):
    """Abstract base class for plot renderers."""

    def __init__(
        self,
        output_dir: str = "assets",
        prefix: str = "",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

    def get_output_path(self, filename: str) -> Path:
        """Get the full output path for a file."""
        if self.prefix:
            filename = f"{self.prefix}-{filename}"
        return self.output_dir / filename

    @abstractmethod
    def render(self, data: Any, **kwargs) -> Path:
        """Render the plot and return the output path."""
        pass


class SingleLogRenderer(BasePlotRenderer):
    """Base class for renderers that work on a single evaluation run."""

    @abstractmethod
    def render(self, run_data: EvalRunData, **kwargs) -> Path:
        """Render the plot for a single evaluation run."""
        pass


class MultiLogRenderer(BasePlotRenderer):
    """Base class for renderers that work on multiple evaluation runs."""

    @abstractmethod
    def render(self, multi_data: MultiEvalData, **kwargs) -> Path:
        """Render the plot for multiple evaluation runs."""
        pass
