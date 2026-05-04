"""Plotting functions for due diligence evaluation results."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_due_diligence_results(
    log_info: List[Dict[str, Any]],
    *,
    max_logs: int = 100,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Plot due diligence results: accuracy by model and task type.
    
    Creates a grouped bar plot showing:
    - X-axis: Model names
    - Y-axis: Accuracy
    - Grouped by task type (classification vs nash equilibrium detection)
    """
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        print("Warning: inspect_ai.log not available, falling back to manual parsing")
        return _plot_due_diligence_manual(log_info, max_logs=max_logs, output_dir=output_dir, prefix=prefix)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect results from all due diligence logs
    results: Dict[str, Dict[str, float]] = {}  # {model_name: {task_type: accuracy}}
    
    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue
        
        log_path = entry["path"]
        log_filename = Path(log_path).name
        
        print(f"\n[Due Diligence] Processing log: {log_filename}")
        
        # Check if this is a due diligence log by filename
        task_type = None
        model_name = entry.get("model_name", "") or ""
        
        if "game-classification" in log_filename:
            task_type = "Classification"
            # Extract model name from filename (e.g., "game-classification-gpt-5.2-2025-12-11")
            # Filename format: TIMESTAMP_game-classification-MODEL_ID.eval
            parts = log_filename.split("_")
            for part in parts:
                if "game-classification" in part:
                    # Extract model from part like "game-classification-gpt-5.2-2025-12-11"
                    model_part = part.replace("game-classification-", "")
                    if model_part and (not model_name or model_name == "?"):
                        model_name = model_part
                    break
            # Fallback: if model name still not found, try to extract from entry
            if not model_name or model_name == "?":
                model_name = entry.get("model_name", "") or "unknown"
        elif "nash-equilibrium-detection" in log_filename:
            task_type = "Nash Equilibrium"
            # Extract model name from filename
            # Filename format: TIMESTAMP_nash-equilibrium-detection-MODEL_ID.eval
            parts = log_filename.split("_")
            for part in parts:
                if "nash-equilibrium-detection" in part:
                    # Extract model from part like "nash-equilibrium-detection-gpt-5.2-2025-12-11"
                    model_part = part.replace("nash-equilibrium-detection-", "")
                    if model_part and (not model_name or model_name == "?"):
                        model_name = model_part
                    break
            # Fallback: if model name still not found, try to extract from entry
            if not model_name or model_name == "?":
                model_name = entry.get("model_name", "") or "unknown"
        
        if not task_type:
            print(f"  [Skip] Not a due diligence log (no 'game-classification' or 'nash-equilibrium-detection' in filename)")
            continue
        
        print(f"  [Info] Task type: {task_type}, Model: {model_name}")
        
        try:
            # Use inspect_ai to read the log (prioritize this method)
            print(f"  [Reading] Attempting to read with inspect_ai...")
            eval_log = read_eval_log(log_path)
            
            # Check if results exist and have scores
            if not hasattr(eval_log, 'results') or not eval_log.results:
                print(f"  [Warning] No results found in log, skipping...")
                continue
            
            # Results structure: eval_log.results.scores is a list of EvalScore objects
            # Each EvalScore has metrics with accuracy
            if not hasattr(eval_log.results, 'scores') or not eval_log.results.scores:
                print(f"  [Warning] No scores found in results, skipping...")
                continue
            
            # Extract accuracy from the first score's metrics
            # The scorer 'answer' should have accuracy metric
            accuracy = None
            for score in eval_log.results.scores:
                if hasattr(score, 'metrics') and score.metrics:
                    # Look for accuracy metric
                    if 'accuracy' in score.metrics:
                        accuracy_metric = score.metrics['accuracy']
                        if hasattr(accuracy_metric, 'value'):
                            accuracy = float(accuracy_metric.value)
                            print(f"  [Success] Accuracy: {accuracy:.3f} (from {score.name} scorer)")
                            break
            
            if accuracy is not None:
                if model_name not in results:
                    results[model_name] = {}
                results[model_name][task_type] = accuracy
            else:
                print(f"  [Warning] No accuracy metric found in scores")
        except Exception as e:
            # If inspect_ai fails, print error and fall back to manual parsing
            print(f"  [Error] Failed to read with inspect_ai: {e}")
            print(f"  [Fallback] Attempting manual parsing...")
            try:
                # Fall back to manual parsing
                with zipfile.ZipFile(log_path, "r") as z:
                    # Try to read samples directly
                    sample_files = sorted(
                        f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")
                    )
                    
                    correct_count = 0
                    total_count = 0
                    
                    for sample_file in sample_files:
                        with z.open(sample_file) as f:
                            sample = json.load(f)
                        
                        # Check if sample has scores
                        if "scores" in sample:
                            for scorer_name, score_data in sample["scores"].items():
                                if isinstance(score_data, dict) and "value" in score_data:
                                    total_count += 1
                                    if score_data["value"] == 1.0 or score_data["value"] == 1:
                                        correct_count += 1
                    
                    if total_count > 0:
                        accuracy = correct_count / total_count
                        if model_name not in results:
                            results[model_name] = {}
                        results[model_name][task_type] = float(accuracy)
                        print(f"  [Success] Accuracy (manual): {accuracy:.3f} ({correct_count}/{total_count})")
                    else:
                        print(f"  [Warning] No valid scores found in manual parsing")
            except Exception as manual_error:
                print(f"  [Error] Manual parsing also failed: {manual_error}")
                continue
    
    return _create_due_diligence_plot(results, output_path, prefix)


def _plot_due_diligence_manual(
    log_info: List[Dict[str, Any]],
    *,
    max_logs: int = 100,
    output_dir: str = "assets",
    prefix: str = "",
) -> Path:
    """Fallback manual parsing for due diligence logs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results: Dict[str, Dict[str, float]] = {}
    
    for entry in log_info[:max_logs]:
        if entry.get("type") != "eval" or entry.get("num_samples", 0) == 0:
            continue
        
        log_path = entry["path"]
        model_name = entry.get("model_name", "") or ""
        
        try:
            with zipfile.ZipFile(log_path, "r") as z:
                # Try to read metadata.json
                if "metadata.json" not in z.namelist():
                    continue
                
                with z.open("metadata.json") as f:
                    metadata = json.load(f)
                
                tasks = metadata.get("tasks", [])
                for task in tasks:
                    task_name = task.get("name", "")
                    if not task_name:
                        continue
                    
                    if "game-classification" in task_name:
                        task_type = "Classification"
                        if "-" in task_name:
                            parts = task_name.split("-")
                            if len(parts) >= 3:
                                extracted_model = "-".join(parts[2:])
                                if not model_name or model_name == "?":
                                    model_name = extracted_model
                    elif "nash-equilibrium-detection" in task_name:
                        task_type = "Nash Equilibrium"
                        if "-" in task_name:
                            parts = task_name.split("-")
                            if len(parts) >= 4:
                                extracted_model = "-".join(parts[3:])
                                if not model_name or model_name == "?":
                                    model_name = extracted_model
                    else:
                        continue
                    
                    metrics = task.get("metrics", {})
                    accuracy = metrics.get("accuracy", None)
                    
                    if accuracy is not None:
                        if model_name not in results:
                            results[model_name] = {}
                        results[model_name][task_type] = float(accuracy)
        except Exception:
            continue
    
    return _create_due_diligence_plot(results, output_path, prefix)


def _create_due_diligence_plot(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
    prefix: str,
) -> Path:
    """Create the actual plot from results dictionary."""
    if not results:
        print("No due diligence results found in logs.")
        return output_path / f"{prefix}due_diligence_results.png"
    
    # Prepare data for plotting
    models = sorted(results.keys())
    
    # Build data arrays
    classification_accs = [results.get(m, {}).get("Classification", 0.0) for m in models]
    nash_accs = [results.get(m, {}).get("Nash Equilibrium", 0.0) for m in models]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.8), 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, classification_accs, width, label="Classification", alpha=0.8)
    bars2 = ax.bar(x + width/2, nash_accs, width, label="Nash Equilibrium", alpha=0.8)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Due Diligence Results: Accuracy by Model and Task", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    
    plt.tight_layout()
    
    filename = f"{prefix}due_diligence_results.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nDue Diligence Results Summary:")
    print(f"{'Model':<30} {'Classification':<15} {'Nash Equilibrium':<15}")
    print("-" * 60)
    for model in models:
        cls_acc = results.get(model, {}).get("Classification", 0.0)
        nash_acc = results.get(model, {}).get("Nash Equilibrium", 0.0)
        print(f"{model:<30} {cls_acc:<15.3f} {nash_acc:<15.3f}")
    
    return filepath
