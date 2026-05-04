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
    """Configure matplotlib and seaborn for consistent styling.

    Uses tueplot with ICML 2024 format for publication-quality plots.
    """
    try:
        from tueplots import bundles, markers, axes
        plt.rcParams.update(bundles.icml2024(
            family="serif",     # "serif" matches paper body text
            usetex=True,        # uses real LaTeX for text rendering
            column="half",      # "half" (single column)
            nrows=1, ncols=1,   # auto-adjusts aspect ratio
        ))
        plt.rcParams.update(markers.with_edge())
        plt.rcParams.update(axes.lines())

        # Increase font sizes by 50%
        font_scale = 1.5
        font_keys = [
            'font.size',
            'axes.labelsize',
            'axes.titlesize',
            'xtick.labelsize',
            'ytick.labelsize',
            'legend.fontsize',
            'legend.title_fontsize',
            'figure.titlesize',
        ]
        for key in font_keys:
            if key in plt.rcParams and isinstance(plt.rcParams[key], (int, float)):
                plt.rcParams[key] = int(plt.rcParams[key] * font_scale)
    except ImportError:
        # Fallback to seaborn style if tueplot not available
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
