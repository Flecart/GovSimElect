"""Parse Epoch AI benchmark data to get best scores for models."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def load_epoch_data(zip_path: Path) -> pd.DataFrame:
    """Load and aggregate benchmark data from Epoch AI zip file.
    
    Args:
        zip_path: Path to benchmark_data.zip
        
    Returns:
        DataFrame with columns: model_version, best_score (aggregated across benchmarks)
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Epoch benchmark data not found at {zip_path}")
    
    all_scores: Dict[str, list[float]] = {}
    
    with zipfile.ZipFile(zip_path, "r") as z:
        # Get all CSV files (excluding README)
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        
        for csv_file in csv_files:
            try:
                # Read CSV from zip
                with z.open(csv_file) as f:
                    df = pd.read_csv(f)
                
                # Check if required columns exist
                if "Model version" not in df.columns or "Best score (across scorers)" not in df.columns:
                    continue
                
                # Extract model scores
                for _, row in df.iterrows():
                    model_version = str(row["Model version"]).strip()
                    best_score = row["Best score (across scorers)"]
                    
                    # Skip NaN values
                    if pd.isna(best_score):
                        continue
                    
                    try:
                        score = float(best_score)
                        if model_version not in all_scores:
                            all_scores[model_version] = []
                        all_scores[model_version].append(score)
                    except (ValueError, TypeError):
                        continue
            except Exception:
                # Skip files that can't be parsed
                continue
    
    # Aggregate scores: take the mean across all benchmarks for each model
    aggregated = {}
    for model_version, scores in all_scores.items():
        if scores:
            aggregated[model_version] = {
                "best_score": sum(scores) / len(scores),  # Mean across benchmarks
                "num_benchmarks": len(scores),
            }
    
    result_df = pd.DataFrame([
        {"model_version": model, "best_score": data["best_score"], "num_benchmarks": data["num_benchmarks"]}
        for model, data in aggregated.items()
    ])
    
    return result_df


def normalize_model_name(model_name: str) -> str:
    """Normalize model name for matching (lowercase, remove special chars)."""
    # Remove common prefixes/suffixes
    normalized = model_name.lower()
    normalized = re.sub(r"^openrouter/", "", normalized)
    normalized = re.sub(r"^openai/", "", normalized)
    normalized = re.sub(r"^anthropic/", "", normalized)
    normalized = re.sub(r"^google/", "", normalized)
    normalized = re.sub(r"^meta-llama/", "", normalized)
    normalized = re.sub(r"^meta/", "", normalized)
    normalized = re.sub(r"^qwen/", "", normalized)
    normalized = re.sub(r"^fireworks/", "", normalized)
    normalized = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", normalized)  # Remove dates
    normalized = re.sub(r"-\d{4}-\d{2}-\d{2}_xhigh$", "", normalized)  # Remove dates with suffix
    normalized = re.sub(r"-instruct$", "", normalized)  # Remove -instruct suffix
    normalized = re.sub(r"-preview$", "", normalized)  # Remove -preview suffix
    normalized = re.sub(r"-a3b$", "", normalized)  # Remove -a3b suffix
    normalized = re.sub(r"-reasoner$", "", normalized)  # Remove -reasoner suffix
    normalized = normalized.strip()
    return normalized


def map_model_to_epoch(
    our_model_name: str,
    epoch_df: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    """Map our model name to Epoch benchmark entry.
    
    Returns:
        Dict with 'best_score', or None if not found
    """
    if epoch_df.empty:
        return None
    
    our_normalized = normalize_model_name(our_model_name)
    
    # Try exact match first
    for _, row in epoch_df.iterrows():
        epoch_normalized = normalize_model_name(row["model_version"])
        if our_normalized == epoch_normalized:
            return {
                "best_score": float(row["best_score"]),
            }
    
    # Try substring matching
    for _, row in epoch_df.iterrows():
        epoch_normalized = normalize_model_name(row["model_version"])
        # Check if key parts match
        our_parts = set(our_normalized.split("-"))
        epoch_parts = set(epoch_normalized.split("-"))
        
        # If significant overlap, consider it a match
        if len(our_parts & epoch_parts) >= 2:
            return {
                "best_score": float(row["best_score"]),
            }
    
    # Try fuzzy matching - check if our model name contains key parts of epoch name
    # or vice versa
    for _, row in epoch_df.iterrows():
        epoch_normalized = normalize_model_name(row["model_version"])
        # Extract key identifiers (numbers, version strings)
        our_keywords = set(re.findall(r"\d+\.?\d*|[a-z]+", our_normalized))
        epoch_keywords = set(re.findall(r"\d+\.?\d*|[a-z]+", epoch_normalized))
        
        # If we share significant keywords (especially version numbers), match
        common_keywords = our_keywords & epoch_keywords
        if len(common_keywords) >= 2 or (len(common_keywords) >= 1 and len(our_keywords) <= 3):
            return {
                "best_score": float(row["best_score"]),
            }
    
    return None


def get_epoch_scores(
    model_names: list[str],
    zip_path: Optional[Path] = None,
    cache_file: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Get Epoch benchmark scores for a list of model names.
    
    Args:
        model_names: List of model names from our logs
        zip_path: Path to benchmark_data.zip (default: data/benchmark_data.zip)
        cache_file: Optional path to cache parsed data
        
    Returns:
        Dict mapping model_name -> {'best_score': float}
    """
    if zip_path is None:
        # Default to data/benchmark_data.zip relative to project root
        project_root = Path(__file__).parent.parent.parent
        zip_path = project_root / "data" / "benchmark_data.zip"
    
    # Try to load from cache first
    if cache_file and cache_file.exists():
        try:
            cached_df = pd.read_csv(cache_file)
            epoch_df = cached_df
        except Exception:
            epoch_df = load_epoch_data(zip_path)
    else:
        epoch_df = load_epoch_data(zip_path)
    
    # Save to cache
    if cache_file and not epoch_df.empty:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        epoch_df.to_csv(cache_file, index=False)
    
    # Map each model
    scores = {}
    for model_name in model_names:
        mapping = map_model_to_epoch(model_name, epoch_df)
        if mapping:
            scores[model_name] = mapping
    
    return scores

