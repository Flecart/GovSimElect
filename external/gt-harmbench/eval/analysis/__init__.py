"""Analysis utilities for Inspect AI logs.

This subpackage provides:
- Parsing helpers for model answers
- Log loading utilities
- Metric recomputation from parsed actions and targets
- Plotting helpers (bar charts and heatmaps)
- Contracting experiment analysis
- A small CLI entrypoint to run the full pipeline
"""

from .contracting import (
    load_contracting_logs,
    compute_contracting_metrics,
    aggregate_experiment_metrics,
    compute_interaction_effects,
    detect_greenwashing,
    get_effort_distribution,
    get_negotiation_transcripts,
    normalize_welfare,
    nash_deviation,
)

__all__ = [
    # Contracting
    "load_contracting_logs",
    "compute_contracting_metrics",
    "aggregate_experiment_metrics",
    "compute_interaction_effects",
    "detect_greenwashing",
    "get_effort_distribution",
    "get_negotiation_transcripts",
    # Normalization (per-scenario)
    "normalize_welfare",
    "nash_deviation",
]
