#!/usr/bin/env python3
"""Pooled QRE lambda recovery: estimate (lambda, beta) per model with bootstrap CIs."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import tqdm as tqdm_module

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.analysis.lambda_recovery import (
    bootstrap_ci_pooled,
    collect_model_scenarios,
    load_per_scenario_payoffs,
)
from eval.analysis.utils import shorten_model_name

# Configuration
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS = {
    "pd": {
        "2x2": PROJECT_ROOT / "data" / "gt-harmbench-pd30-2x2-with-targets.csv",
        "4x4": PROJECT_ROOT / "data" / "gt-harmbench-pd30-4x4-rewritten.csv",
    },
    "sh": {
        "2x2": PROJECT_ROOT / "data" / "gt-harmbench-sh30-2x2-with-targets.csv",
        "4x4": PROJECT_ROOT / "data" / "gt-harmbench-sh30-4x4-rewritten.csv",
    },
}

# Grid search configuration
LAMBDA_GRID = np.linspace(0, 10, 40)
BETA_GRID = np.linspace(0.1, 20, 40)
N_BOOTSTRAP = 300
N_WORKERS_PER_MODEL = 12


def discover_batch_runs(logs_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover eval run directories for all models.

    Args:
        logs_dir: Base logs directory containing eval-* subdirs.

    Returns:
        Dict mapping model_part to {game: eval_dir}.
    """
    runs: dict[str, dict[str, Path]] = {}

    for d in sorted(logs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("eval-"):
            continue

        # Parse game suffix (e.g., "eval-TIMESTAMP-MODEL-pd" -> game="pd")
        remainder = re.sub(r"^eval-\d{8}-\d{6}-", "", name)
        game = None
        model_part = None
        if remainder.endswith("-pd"):
            game = "pd"
            model_part = remainder[:-3]
        elif remainder.endswith("-sh"):
            game = "sh"
            model_part = remainder[:-3]
        else:
            continue

        if model_part not in runs:
            runs[model_part] = {}
        runs[model_part][game] = d

    return runs


def _worker_wrapper(work_item: dict, progress_queue: Queue, results_queue: Queue) -> None:
    """Wrapper to run worker and put result in queue.

    Args:
        work_item: Dict containing model_part, games, payoff_by_game, payoffs_per_scenario, worker_id.
        progress_queue: Queue for reporting progress updates.
        results_queue: Queue for sending final results.
    """
    try:
        result = _process_one_model(work_item, progress_queue)
        results_queue.put(result)
    except Exception as e:
        import traceback
        print(f"Error processing {work_item.get('model_part', 'unknown')}: {e}")
        traceback.print_exc()
        results_queue.put(None)


def _process_one_model(work_item: dict, progress_queue: Queue | None = None) -> dict | None:
    """Worker function to process one model's QRE estimation.

    Args:
        work_item: Dict containing model_part, games, payoff_by_game, payoffs_per_scenario, worker_id.
        progress_queue: Queue for reporting progress updates.

    Returns:
        Result dict with model info and estimates, or None if skipped.
    """
    model_part = work_item["model_part"]
    games = work_item["games"]
    payoff_by_game = work_item["payoff_by_game"]
    worker_id = work_item["worker_id"]

    model_display = shorten_model_name(model_part)

    # Collect scenarios for this model
    scenarios = collect_model_scenarios(
        eval_dirs_by_game=games,
        payoff_dict_by_game=payoff_by_game,
        prompt_mode="base",
        contract_mode="no-comm",
        dataset_size="2x2",
    )

    if not scenarios:
        return None

    # Run estimation with progress callback
    result = bootstrap_ci_pooled(
        scenarios,
        n_bootstrap=N_BOOTSTRAP,
        lambda_grid=LAMBDA_GRID,
        beta_grid=BETA_GRID,
        n_workers=N_WORKERS_PER_MODEL,
        desc=model_display,
        worker_id=worker_id,
        show_nested_progress=False,  # Hide grid search progress bars
        progress_queue=progress_queue,  # Report progress
    )

    # Store results
    result["model"] = model_part
    result["model_display"] = model_display
    result["n_scenarios"] = len(scenarios)

    return result


def main():
    """Main entry point for pooled QRE estimation."""
    print("=" * 60)
    print("Pooled QRE Lambda Recovery")
    print("=" * 60)
    print()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover batch runs
    print("Discovering eval runs...")
    batch_runs = discover_batch_runs(LOGS_DIR)
    print(f"Found {len(batch_runs)} models:")
    for model_part, games in batch_runs.items():
        display_name = shorten_model_name(model_part)
        games_str = ", ".join(sorted(games.keys()))
        print(f"  {display_name} ({model_part}): {games_str}")
    print()

    # Load per-scenario payoff matrices (2x2 only)
    print("Loading per-scenario payoff matrices (2x2 only)...")
    PAYOFFS_PER_SCENARIO = {}
    for game, datasets in DATASETS.items():
        payoffs = load_per_scenario_payoffs(datasets["2x2"], None)  # Only load 2x2
        PAYOFFS_PER_SCENARIO[game] = payoffs
        print(f"  {game}: {len(payoffs['2x2'])} scenarios")
    print()

    # Configuration summary
    print("Configuration:")
    print(f"  Grid: 40×40 coarse + 20×20 refine")
    print(f"  Lambda range: [{LAMBDA_GRID[0]:.2f}, {LAMBDA_GRID[-1]:.2f}]")
    print(f"  Beta range: [{BETA_GRID[0]:.2f}, {BETA_GRID[-1]:.2f}]")
    print(f"  Bootstrap: {N_BOOTSTRAP} resamples")
    print(f"  Parallelization: {len(batch_runs)} models × {N_WORKERS_PER_MODEL} workers = {len(batch_runs) * N_WORKERS_PER_MODEL} cores total")
    print()

    # Run pooled QRE estimation for each model in parallel
    print("Running pooled QRE estimation per model (parallel)...")
    print()

    # Prepare work items for parallel processing
    work_items = []
    for model_part, games in batch_runs.items():
        # Build payoff dict keyed by game: {game -> {scenario_id -> payoff_matrix}}
        payoff_by_game = {
            game: PAYOFFS_PER_SCENARIO[game]["2x2"]
            for game in games
            if game in PAYOFFS_PER_SCENARIO
        }

        if not any(payoff_by_game.values()):
            print(f"  Skipping {shorten_model_name(model_part)}: no 2x2 payoff matrices")
            continue

        work_items.append({
            "model_part": model_part,
            "games": games,
            "payoff_by_game": payoff_by_game,
            "payoffs_per_scenario": PAYOFFS_PER_SCENARIO,
        })

    if not work_items:
        print("No models to process.")
        return

    # Add worker_id to each work item for tqdm position tracking
    for i, item in enumerate(work_items):
        item["worker_id"] = i

    # Process models in parallel with progress tracking
    pooled_results = []
    completed = 0
    progress_queue = Queue()

    print(f"Processing {len(work_items)} models using {len(work_items) * N_WORKERS_PER_MODEL} cores...")
    print()

    # Start all workers as non-daemon processes
    processes = []
    results_queue = Queue()

    for work_item in work_items:
        p = Process(
            target=_worker_wrapper,
            args=(work_item, progress_queue, results_queue),
            daemon=False,  # Allow creating nested Pools
        )
        p.start()
        processes.append(p)

    # Track progress per worker (bootstrap samples completed)
    worker_progress = {i: 0 for i in range(len(work_items))}
    total_per_worker = N_BOOTSTRAP

    # Progress bar showing minimum progress across all workers
    with tqdm_module.tqdm(total=len(work_items) * total_per_worker, desc="Bootstrap progress", ncols=80) as pbar:
        while any(p.is_alive() for p in processes):
            # Check for progress updates
            while not progress_queue.empty():
                worker_id, current = progress_queue.get()
                old_total = sum(worker_progress.values())
                worker_progress[worker_id] = current
                new_total = sum(worker_progress.values())
                pbar.update(new_total - old_total)

            # Check for completed results
            while not results_queue.empty():
                result = results_queue.get()
                if result is not None:
                    completed += 1
                    model_display = result["model_display"]
                    lam_lo, lam_hi = result["lambda_ci"]
                    beta_lo, beta_hi = result["beta_ci"]
                    print(f"\n[{completed}/{len(work_items)}] {model_display}: "
                          f"λ = {result['lambda_hat']:.4f} [{lam_lo:.4f}, {lam_hi:.4f}], "
                          f"β = {result['beta_hat']:.4f} [{beta_lo:.4f}, {beta_hi:.4f}]")
                    pooled_results.append(result)

        # Final check for any remaining updates
        while not progress_queue.empty():
            worker_id, current = progress_queue.get()
            pbar.update(current - worker_progress[worker_id])
            worker_progress[worker_id] = current

        while not results_queue.empty():
            result = results_queue.get()
            if result is not None:
                completed += 1
                model_display = result["model_display"]
                lam_lo, lam_hi = result["lambda_ci"]
                beta_lo, beta_hi = result["beta_ci"]
                print(f"\n[{completed}/{len(work_items)}] {model_display}: "
                      f"λ = {result['lambda_hat']:.4f} [{lam_lo:.4f}, {lam_hi:.4f}], "
                      f"β = {result['beta_hat']:.4f} [{beta_lo:.4f}, {beta_hi:.4f}]")
                pooled_results.append(result)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print()

    # Export results
    print("Exporting results...")

    # Summary table with point estimates and CIs
    table_rows = []
    for r in pooled_results:
        lam_lo, lam_hi = r["lambda_ci"]
        beta_lo, beta_hi = r["beta_ci"]
        table_rows.append({
            "model": r["model_display"],
            "lambda_hat": r["lambda_hat"],
            "lambda_ci_lo": lam_lo,
            "lambda_ci_hi": lam_hi,
            "beta_hat": r["beta_hat"],
            "beta_ci_lo": beta_lo,
            "beta_ci_hi": beta_hi,
            "log_likelihood": r["log_likelihood"],
            "n_scenarios": r["n_scenarios"],
        })

    results_df = pd.DataFrame(table_rows)
    results_df.to_csv(RESULTS_DIR / "qre_pooled_estimates.csv", index=False)
    print(f"  Exported: {RESULTS_DIR / 'qre_pooled_estimates.csv'}")

    # Bootstrap distribution for downstream analysis
    boot_records = []
    for r in pooled_results:
        for lam, beta in zip(r["bootstrap_lambdas"], r["bootstrap_betas"]):
            boot_records.append({
                "model": r["model_display"],
                "lambda": lam,
                "beta": beta,
            })

    boot_df = pd.DataFrame(boot_records)
    boot_df.to_csv(RESULTS_DIR / "qre_bootstrap_distribution.csv", index=False)
    print(f"  Exported: {RESULTS_DIR / 'qre_bootstrap_distribution.csv'}")
    print()

    # Final summary table
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print()

    def fmt_ci(mean, lo, hi):
        return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"

    for r in pooled_results:
        lam_str = fmt_ci(r["lambda_hat"], *r["lambda_ci"])
        beta_str = fmt_ci(r["beta_hat"], *r["beta_ci"])
        print(f"{r['model_display']}:")
        print(f"  λ: {lam_str}")
        print(f"  β: {beta_str}")
        print(f"  n_scenarios: {r['n_scenarios']}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
