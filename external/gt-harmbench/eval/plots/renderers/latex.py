"""LaTeX table renderers for evaluation data."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


def _value_to_color(value: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Convert a value to a LaTeX color specification using a gradient.

    Uses a red-yellow-green gradient similar to seaborn's RdYlGn colormap.
    """
    # Normalize value to [0, 1]
    normalized = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    normalized = max(0.0, min(1.0, normalized))

    # RdYlGn-like gradient: red (low) -> yellow (mid) -> green (high)
    if normalized < 0.5:
        # Red to Yellow: R stays high, G increases
        r = int(215 + (255 - 215) * (normalized * 2))  # 215 -> 255
        g = int(48 + (255 - 48) * (normalized * 2))    # 48 -> 255
        b = int(39 + (191 - 39) * (normalized * 2))    # 39 -> 191
    else:
        # Yellow to Green: R decreases, G stays high
        t = (normalized - 0.5) * 2
        r = int(255 - (255 - 26) * t)   # 255 -> 26
        g = int(255 - (255 - 152) * t)  # 255 -> 152
        b = int(191 - (191 - 80) * t)   # 191 -> 80

    return f"{r:02X}{g:02X}{b:02X}"


def _format_model_name(model: str) -> str:
    """Format model name for LaTeX display."""
    # Remove provider prefix
    if "/" in model:
        model = model.split("/")[-1]

    # Common replacements for nicer display - ordered by specificity (longer matches first)
    replacements = [
        ("gpt-5.1", "GPT-5.1"),
        ("gpt-5.2", "GPT-5.2"),
        ("gpt-5-mini", "GPT-5 Mini"),
        ("gpt-5-nano", "GPT-5 Nano"),
        ("gpt-4o", "GPT-4o"),
        ("claude-sonnet-4-5", "Claude 4.5 Sonnet"),
        ("claude-sonnet-4.5", "Claude 4.5 Sonnet"),
        ("claude-4.5-opus", "Claude 4.5 Opus"),
        ("claude-opus-4.5", "Claude 4.5 Opus"),
        ("grok-4.1-fast", "Grok 4.1 Fast"),
        ("grok-4", "Grok 4"),
        ("gemini-3-flash-preview", "Gemini 3 Flash"),
        ("gemini-3-pro-preview", "Gemini 3 Pro"),
        ("gemini-3-flash", "Gemini 3 Flash"),
        ("llama-3.3-70b", "Llama 3.3 70B"),
        ("llama-3.2-3b", "Llama 3.2 3B"),
        ("qwen3-30b", "Qwen3 30B"),
        ("qwen3-8b", "Qwen3 8B"),
    ]

    model_lower = model.lower()
    for key, value in replacements:
        if key in model_lower:
            return value

    # Default: capitalize and replace hyphens
    return model.replace("-", " ").title()


def _format_game_type(game_type: str) -> str:
    """Format game type for LaTeX display."""
    # Capitalize and format nicely
    if "Bach" in game_type:
        return "Battle of the Sexes"
    return game_type.replace("_", " ")


def sort_models_by_name(model: str) -> tuple[int, int]:
    """Helper function to sort models by formatted name."""
    family = ["claude", "gpt", "grok", "gemini", "llama", "qwen"]
    capabilities_order = [
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4",
        "opus",
        "sonnet",
        "deepseek"
        "grok-4.1",
        "pro",
        "flash",
        "3.3",
        "3.2",
        "30b",
        "qwen3-8b",
    ]
    family_index = len(family)
    for index, fam in enumerate(family):
        if fam in model.lower():
            family_index = index
            break
    else:
        family_index = len(family)
        print(f"Unknown model family for sorting: {model}")
    
    for index, cap in enumerate(capabilities_order):
        if cap in model.lower():
            capability_index = index
            break
    else:
        capability_index = len(capabilities_order)
    return (family_index, capability_index)

def render_accuracy_heatmap_latex(
    multi_data: "MultiEvalData",  # noqa: F821
    metric: str = "utilitarian_accuracy",
    output_path: Optional[str] = None,
) -> str:
    """Render accuracy heatmap as a LaTeX table with colored cells.

    Args:
        multi_data: MultiEvalData containing processed evaluation runs
        metric: Which metric to display (default: utilitarian_accuracy)
        output_path: Optional path to save the .tex file

    Returns:
        LaTeX table code as a string
    """
    from ..data import compute_accuracy_by_game, GAME_TYPE_ORDER

    # Collect data from all models
    model_data: Dict[str, Dict[str, float]] = {}

    for run_data in multi_data.runs:
        model_id = run_data.model_name
        accuracy_df = compute_accuracy_by_game(run_data)
        model_data[model_id] = {}

        for _, row in accuracy_df.iterrows():
            game_type = row["game_type"]
            if metric in row:
                model_data[model_id][game_type] = row[metric]

    if not model_data:
        return "% No data available"

    # Get all game types present in data, ordered
    all_game_types = set()
    for data in model_data.values():
        all_game_types.update(data.keys())

    game_types = [gt for gt in GAME_TYPE_ORDER if gt in all_game_types]
    models = list(model_data.keys())
    models.sort(key=sort_models_by_name)
    
    # Sort models by formatted name

    # Find global min/max for color scaling
    all_values = []
    for data in model_data.values():
        all_values.extend(data.values())
    vmin = min(all_values) if all_values else 0
    vmax = max(all_values) if all_values else 1

    # Use 0-1 range for accuracy metrics
    if "accuracy" in metric:
        vmin, vmax = 0.0, 1.0

    # Build LaTeX table
    lines = []

    # Preamble
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")

    # Calculate column spec: first column for game types, then one per model
    n_models = len(models)
    col_spec = "l" + "c" * n_models
    lines.append(r"\begin{tabular}{@{}" + col_spec + r"@{}}")
    lines.append(r"\toprule")

    # Header row with model names
    formatted_models = [_format_model_name(m) for m in models]
    # Rotate headers for better fit
    header_cells = [r"\textbf{Game Type}"]
    for fm in formatted_models:
        header_cells.append(r"\textbf{\scriptsize" + fm + r"}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for game_type in game_types:
        row_cells = [_format_game_type(game_type)]

        for model in models:
            value = model_data[model].get(game_type)
            if value is not None:
                color = _value_to_color(value, vmin, vmax)
                # Format value as percentage or decimal
                formatted_value = f"{value:.2f}"
                row_cells.append(r"\cellcolor[HTML]{" + color + r"}" + formatted_value)
            else:
                row_cells.append("--")

        lines.append(" & ".join(row_cells) + r" \\")

    # Add average row
    lines.append(r"\midrule")
    avg_cells = [r"\textbf{Average}"]
    for model in models:
        values = [v for v in model_data[model].values() if v is not None]
        if values:
            avg = sum(values) / len(values)
            color = _value_to_color(avg, vmin, vmax)
            avg_cells.append(r"\cellcolor[HTML]{" + color + r"}\textbf{" + f"{avg:.2f}" + r"}")
        else:
            avg_cells.append("--")
    lines.append(" & ".join(avg_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Add color bar legend
    metric_display = metric.replace("_", " ").title()
    lines.append(r"\vspace{0.5em}")
    lines.append(r"\newline")
    lines.append(_generate_colorbar_latex(vmin, vmax))

    # Caption
    lines.append(r"\caption{" + metric_display + r" by game type and model. Colors indicate performance from ")
    lines.append(r"\textcolor[HTML]{D73027}{low} to \textcolor[HTML]{1A9850}{high}.}")
    lines.append(r"\label{tab:accuracy_heatmap}")
    lines.append(r"\end{table}")

    latex_code = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_code)

    return latex_code


def _generate_colorbar_latex(vmin: float, vmax: float, n_steps: int = 5) -> str:
    """Generate a horizontal color bar legend in LaTeX."""
    lines = []
    lines.append(r"\begin{center}")
    lines.append(r"\footnotesize")

    # Create a series of colored boxes
    cells = []
    for i in range(n_steps):
        value = vmin + (vmax - vmin) * i / (n_steps - 1)
        color = _value_to_color(value, vmin, vmax)
        label = f"{value:.1f}"
        cells.append(r"\colorbox[HTML]{" + color + r"}{\makebox[1.5em]{" + label + r"}}")

    lines.append(" ".join(cells))
    lines.append(r"\end{center}")

    return "\n".join(lines)


def render_accuracy_heatmap_latex_landscape(
    multi_data: "MultiEvalData",  # noqa: F821
    metric: str = "utilitarian_accuracy",
    output_path: Optional[str] = None,
    deduplicate: bool = True,
) -> str:
    """Render accuracy heatmap as a landscape LaTeX table spanning the full page.

    This version is optimized for many models, using landscape orientation
    and adjustbox for automatic scaling.

    Args:
        multi_data: MultiEvalData containing processed evaluation runs
        metric: Which metric to display (default: utilitarian_accuracy)
        output_path: Optional path to save the .tex file
        deduplicate: If True, keep only one run per model (by formatted name)

    Returns:
        LaTeX table code as a string
    """
    from ..data import GAME_TYPE_ORDER

    # Map metric name to the correct column in samples_df
    metric_col_map = {
        "utilitarian_accuracy": "utilitarian_correct",
        "nash_accuracy": "nash_correct",
        "rawlsian_accuracy": "rawlsian_correct",
        "nash_social_accuracy": "nash_social_correct",
    }
    metric_col = metric_col_map.get(metric, "utilitarian_correct")

    # Collect data from all models
    model_data: Dict[str, Dict[str, float]] = {}
    seen_formatted_names: Dict[str, str] = {}  # formatted_name -> model_id

    for run_data in multi_data.runs:
        model_id = run_data.model_name
        formatted_name = _format_model_name(model_id)

        # If deduplicating and we've seen this formatted name, skip
        if deduplicate and formatted_name in seen_formatted_names:
            continue

        df = run_data.get_valid_samples_df()
        if df.empty:
            continue

        seen_formatted_names[formatted_name] = model_id
        model_data[model_id] = {}

        # Compute accuracy by game type
        for game_type in df["game_type"].unique():
            if "matching" in game_type.lower():
                continue  # Skip matching games
            game_df = df[df["game_type"] == game_type]
            if not game_df.empty and metric_col in game_df.columns:
                model_data[model_id][game_type] = game_df[metric_col].mean()

    if not model_data:
        return "% No data available"

    # Get all game types present in data, ordered
    all_game_types = set()
    for data in model_data.values():
        all_game_types.update(data.keys())

    game_types = [gt for gt in GAME_TYPE_ORDER if gt in all_game_types]
    models = list(model_data.keys())
    models.sort(key=sort_models_by_name)
    models.append("Avg.")
    
    
    # Add average entries    
    model_data["Avg."] = {}
    # compute avg
    for game_type in game_types:
        values = []
        for model in models[:-1]:
            if game_type in model_data[model]:
                values.append(model_data[model][game_type])
        if values:
            model_data["Avg."][game_type] = sum(values) / len(values)
    

    # Use 0-1 range for accuracy metrics
    vmin, vmax = 0.0, 1.0

    # Build LaTeX table
    lines = []

    # Required packages note
    lines.append(r"% Requires: \usepackage{adjustbox}, \usepackage{colortbl}, \usepackage{booktabs}")
    lines.append(r"% Optional for landscape: \usepackage{lscape} or \usepackage{pdflscape}")
    lines.append("")

    # Landscape wrapper (optional)
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{2}")
    lines.append(r"\footnotesize")

    # Use adjustbox to scale to text width
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")

    # Calculate column spec
    n_models = len(models)
    col_spec = "l" + "c" * (n_models - 1) + "|c"
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Header row with model names (rotated)
    formatted_models = [_format_model_name(m) for m in models]
    header_cells = [r"\textbf{Game}"]
    for fm in formatted_models:
        header_cells.append(r"\textbf{\scriptsize " + fm + r"}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for game_type in game_types:
        row_cells = [_format_game_type(game_type)]

        for model in models:
            value = model_data[model].get(game_type)
            if value is not None:
                color = _value_to_color(value, vmin, vmax)
                formatted_value = f"{value:.2f}"
                row_cells.append(r"\cellcolor[HTML]{" + color + r"}" + formatted_value)
            else:
                row_cells.append("--")

        lines.append(" & ".join(row_cells) + r" \\")

    # Average row
    lines.append(r"\midrule")
    avg_cells = [r"\textbf{Avg}"]
    for model in models:
        values = [v for v in model_data[model].values() if v is not None]
        if values:
            avg = sum(values) / len(values)
            color = _value_to_color(avg, vmin, vmax)
            avg_cells.append(r"\cellcolor[HTML]{" + color + r"}\textbf{" + f"{avg:.2f}" + r"}")
        else:
            avg_cells.append("--")
    lines.append(" & ".join(avg_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")

    # Caption
    metric_display = metric.replace("_", " ").title()
    lines.append(r"\caption{" + metric_display + r" across models and game types. ")
    lines.append(r"Cell colors range from \textcolor[HTML]{D73027}{red} (0.0) to \textcolor[HTML]{1A9850}{green} (1.0).}")
    lines.append(r"\label{tab:accuracy_heatmap}")
    lines.append(r"\end{table*}")

    latex_code = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_code)

    return latex_code
