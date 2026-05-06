#!/usr/bin/env python3
"""Profile Log-Likelihood Diagnostic for Lambda Recovery.

Investigates why GPT-5.4 Mini's MLE for lambda is pinned at the lower boundary
(lambda = 0) with a degenerate bootstrap CI of [0, 0].

Computes L_profile(lambda) = max_beta L(lambda, beta) over a dense lambda grid
to determine whether:
(A) MLE is on boundary because data wants lambda < 0 (gradient at 0 points negative)
(B) MLE is genuinely at lambda = 0 with flat profile (flat near 0)
(C) Small interior optimum near zero missed by grid resolution (peak > L_profile(0))

Usage:
    uv run python3 scripts/analysis/profile_likelihood_diagnostic.py

Output:
    - Console tables with lambda, beta_hat, L_profile, delta_L
    - profile_likelihood_gpt54mini.png plot
    - Case classification for GPT-5.4 Mini
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.analysis.lambda_recovery import (
    _pooled_ll_cold,
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

# Dense lambda grid for profile likelihood (boundary-focused)
LAMBDA_PROFILE_GRID = np.array([0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])

# Fine beta grid for inner optimization (same as refinement step in main pipeline)
BETA_FINE_GRID = np.linspace(0.1, 20, 50)


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


def compute_profile_likelihood(
    scenarios: List[Tuple[List[List[List[float]]], List[List[int]]]],
    lambda_grid: np.ndarray = LAMBDA_PROFILE_GRID,
    beta_grid: np.ndarray = BETA_FINE_GRID,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute profile log-likelihood L_profile(lambda) = max_beta L(lambda, beta).

    For each lambda value, finds the beta that maximizes the pooled log-likelihood.
    Uses the same cold-start QRE solver as the main pipeline.

    Args:
        scenarios: List of (payoff_matrix, counts) tuples.
        lambda_grid: Lambda values at which to evaluate profile likelihood.
        beta_grid: Beta search grid for inner optimization.

    Returns:
        Tuple of (beta_hat, L_profile, convergence_flags):
        - beta_hat[i] is the beta that maximizes L(lambda_grid[i], beta)
        - L_profile[i] is the maximized log-likelihood at lambda_grid[i]
        - convergence_flags[i] is True if QRE converged for all scenarios
    """
    n_lambda = len(lambda_grid)
    beta_hat = np.zeros(n_lambda)
    L_profile = np.full(n_lambda, -np.inf)
    converged = np.zeros(n_lambda, dtype=bool)

    print(f"Computing profile likelihood over {n_lambda} lambda values...")
    print(f"  Beta grid: {len(beta_grid)} points from [{beta_grid[0]:.2f}, {beta_grid[-1]:.2f}]")

    for i, lam in enumerate(lambda_grid):
        # Grid search over beta to find beta_hat(lambda)
        best_ll = -np.inf
        best_beta = beta_grid[0]
        all_converged = True

        for beta in beta_grid:
            ll, _, conv = _pooled_ll_cold(lam, beta, scenarios)

            if not conv:
                all_converged = False
                continue

            if ll > best_ll:
                best_ll = ll
                best_beta = beta

        beta_hat[i] = best_beta
        L_profile[i] = best_ll
        converged[i] = all_converged

        print(f"  lambda={lam:.4f}: beta_hat={best_beta:.4f}, L={best_ll:.6f}, converged={all_converged}")

    return beta_hat, L_profile, converged


def print_profile_table(
    model_name: str,
    lambda_grid: np.ndarray,
    beta_hat: np.ndarray,
    L_profile: np.ndarray,
) -> None:
    """Print formatted table of profile likelihood results.

    Args:
        model_name: Model display name.
        lambda_grid: Lambda values.
        beta_hat: Optimal beta at each lambda.
        L_profile: Profile log-likelihood at each lambda.
    """
    print()
    print("=" * 80)
    print(f"Profile Log-Likelihood Table: {model_name}")
    print("=" * 80)
    print(f"{'lambda':>10} {'beta_hat':>10} {'L_profile':>14} {'delta_L':>12}")
    print("-" * 80)

    L0 = L_profile[0]
    for lam, beta, ll in zip(lambda_grid, beta_hat, L_profile):
        delta = ll - L0
        print(f"{lam:10.4f} {beta:10.4f} {ll:14.6f} {delta:12.6f}")

    print("-" * 80)
    print()


def compute_gradient_at_zero(
    lambda_grid: np.ndarray,
    L_profile: np.ndarray,
) -> Tuple[float, bool]:
    """Compute numerical gradient of L_profile at lambda = 0 using forward difference.

    Uses the smallest positive lambda value for the forward difference.

    Args:
        lambda_grid: Lambda values (must include 0 and at least one positive value).
        L_profile: Profile log-likelihood values.

    Returns:
        Tuple of (gradient, is_valid):
        - gradient: (L(lambda_epsilon) - L(0)) / lambda_epsilon
        - is_valid: True if computation was successful
    """
    # Find smallest positive lambda
    positive_indices = np.where(lambda_grid > 0)[0]
    if len(positive_indices) == 0:
        return 0.0, False

    eps_idx = positive_indices[0]
    epsilon = lambda_grid[eps_idx]

    if epsilon == 0:
        return 0.0, False

    gradient = (L_profile[eps_idx] - L_profile[0]) / epsilon
    return gradient, True


def classify_profile_shape(
    lambda_grid: np.ndarray,
    L_profile: np.ndarray,
    gradient_at_zero: float,
    flat_threshold: float = 1e-3,
) -> Tuple[str, str]:
    """Classify the profile likelihood shape to diagnose boundary MLE.

    Args:
        lambda_grid: Lambda values.
        L_profile: Profile log-likelihood values.
        gradient_at_zero: Numerical gradient at lambda = 0.
        flat_threshold: Delta_L threshold for considering profile as flat.

    Returns:
        Tuple of (case, explanation):
        - case: "A", "B", "C", or "AMBIGUOUS"
        - explanation: Human-readable explanation
    """
    L0 = L_profile[0]

    # Check for monotonic decrease (Case A)
    if gradient_at_zero < 0:
        is_monotonic = True
        for i in range(1, len(L_profile)):
            if L_profile[i] > L0 + flat_threshold:
                is_monotonic = False
                break

        if is_monotonic:
            return ("A",
                    "Gradient at lambda=0 is negative (L decreases as lambda increases). "
                    "Profile is monotonically decreasing across the grid. "
                    "Conclusion: Boundary MLE is REAL — the data wants lambda < 0, "
                    "which we forbid by construction. The [0, 0] CI is correct.")

    # Check for flat profile (Case B)
    is_flat = True
    for i in range(1, min(4, len(L_profile))):  # Check first few points
        if abs(L_profile[i] - L0) > flat_threshold:
            is_flat = False
            break

    if is_flat:
        return ("B",
                "Profile likelihood is flat near lambda=0 (delta_L within ±1e-3). "
                "Conclusion: Likely SOLVER or BOOTSTRAP BUG — bootstrap should not "
                "produce degenerate CI if the true likelihood surface is flat.")

    # Check for interior peak (Case C)
    max_idx = np.argmax(L_profile)
    if max_idx > 0 and L_profile[max_idx] > L0 + flat_threshold:
        peak_lambda = lambda_grid[max_idx]
        return ("C",
                f"Profile likelihood peaks at lambda={peak_lambda:.4f} > 0, "
                f"with L_profile({peak_lambda:.4f}) = {L_profile[max_idx]:.6f} > "
                f"L_profile(0) = {L0:.6f}. "
                "Conclusion: GRID RESOLUTION ISSUE — the true MLE is interior but "
                "our grid spacing missed it. Need finer grid near lambda=0.")

    # Ambiguous
    return ("AMBIGUOUS",
            "Profile shape does not match any expected pattern. "
            "Gradient at zero is positive but no clear interior peak found. "
            "Recommend manual investigation of the likelihood surface.")


def create_profile_plot(
    lambda_grid_mini: np.ndarray,
    L_profile_mini: np.ndarray,
    lambda_grid_gpt54: np.ndarray,
    L_profile_gpt54: np.ndarray,
    output_path: Path,
) -> None:
    """Create profile likelihood plot with symlog x-axis.

    Args:
        lambda_grid_mini: Lambda grid for GPT-5.4 Mini.
        L_profile_mini: Profile likelihood for GPT-5.4 Mini.
        lambda_grid_gpt54: Lambda grid for GPT-5.4.
        L_profile_gpt54: Profile likelihood for GPT-5.4.
        output_path: Path to save plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot GPT-5.4 Mini (boundary case)
    ax.plot(lambda_grid_mini, L_profile_mini, 'o-', label='GPT-5.4 Mini (boundary MLE)',
            color='red', linewidth=2, markersize=8)

    # Plot GPT-5.4 (interior case - sanity check)
    ax.plot(lambda_grid_gpt54, L_profile_gpt54, 's-', label='GPT-5.4 (interior MLE)',
            color='blue', linewidth=2, markersize=8)

    # Reference line at lambda=0
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=L_profile_mini[0], color='red', linestyle=':', alpha=0.5,
               label=f'L_profile(0) = {L_profile_mini[0]:.2f}')

    # Symlog x-axis with linear region near zero
    ax.set_xscale('symlog', linthresh=0.001, linscale=0.5)

    # Formatting
    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Profile Log-Likelihood L_profile(λ)', fontsize=12)
    ax.set_title('Profile Log-Likelihood Diagnostic: Boundary vs Interior MLE', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Custom x-axis ticks
    ax.xaxis.set_major_locator(ticker.LogLocator(subs='all'))
    ax.xaxis.set_minor_locator(ticker.LogLocator(subs='all'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    """Main entry point for profile likelihood diagnostic."""
    print("=" * 80)
    print("Profile Log-Likelihood Diagnostic for Lambda Recovery")
    print("=" * 80)
    print()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover batch runs
    print("Discovering eval runs...")
    batch_runs = discover_batch_runs(LOGS_DIR)
    print(f"Found {len(batch_runs)} models")

    # Load per-scenario payoff matrices (2x2 only)
    print()
    print("Loading per-scenario payoff matrices (2x2 only)...")
    PAYOFFS_PER_SCENARIO = {}
    for game, datasets in DATASETS.items():
        payoffs = load_per_scenario_payoffs(datasets["2x2"], None)  # Only load 2x2
        PAYOFFS_PER_SCENARIO[game] = payoffs
        print(f"  {game}: {len(payoffs['2x2'])} scenarios")
    print()

    # Models to analyze
    target_models = {
        "openai-gpt-5.4-mini": "GPT-5.4 Mini (boundary case)",
        "openai-gpt-5.4": "GPT-5.4 (interior case - sanity check)",
    }

    results = {}

    for model_part, description in target_models.items():
        if model_part not in batch_runs:
            print(f"WARNING: {model_part} not found in batch runs, skipping")
            continue

        games = batch_runs[model_part]
        model_display = shorten_model_name(model_part)

        print(f"Processing {description} ({model_part})...")

        # Build payoff dict keyed by game
        payoff_by_game = {
            game: PAYOFFS_PER_SCENARIO[game]["2x2"]
            for game in games
            if game in PAYOFFS_PER_SCENARIO
        }

        # Collect scenarios
        scenarios = collect_model_scenarios(
            eval_dirs_by_game=games,
            payoff_dict_by_game=payoff_by_game,
            prompt_mode="base",
            contract_mode="no-comm",
            dataset_size="2x2",
        )

        if not scenarios:
            print(f"  WARNING: No scenarios found for {model_display}")
            continue

        print(f"  Loaded {len(scenarios)} scenarios")

        # Compute profile likelihood
        beta_hat, L_profile, converged = compute_profile_likelihood(
            scenarios,
            lambda_grid=LAMBDA_PROFILE_GRID,
            beta_grid=BETA_FINE_GRID,
        )

        # Store results
        results[model_part] = {
            "description": description,
            "lambda_grid": LAMBDA_PROFILE_GRID.copy(),
            "beta_hat": beta_hat,
            "L_profile": L_profile,
            "converged": converged,
        }

        # Print table
        print_profile_table(model_display, LAMBDA_PROFILE_GRID, beta_hat, L_profile)

        # Compute gradient at zero
        gradient, valid = compute_gradient_at_zero(LAMBDA_PROFILE_GRID, L_profile)
        if valid:
            print(f"Numerical gradient at lambda=0: {gradient:.6f}")
            print(f"  Sign: {'NEGATIVE' if gradient < 0 else 'POSITIVE' if gradient > 0 else 'ZERO'}")
            print(f"  Magnitude: {abs(gradient):.6f}")
        else:
            print("WARNING: Could not compute gradient at zero")
        print()

    # Classify GPT-5.4 Mini
    if "openai-gpt-5.4-mini" in results:
        mini_results = results["openai-gpt-5.4-mini"]
        gradient, _ = compute_gradient_at_zero(
            mini_results["lambda_grid"],
            mini_results["L_profile"],
        )

        case, explanation = classify_profile_shape(
            mini_results["lambda_grid"],
            mini_results["L_profile"],
            gradient,
        )

        print("=" * 80)
        print(f"CASE CLASSIFICATION: GPT-5.4 Mini")
        print("=" * 80)
        print(f"Case: {case}")
        print(f"Explanation: {explanation}")
        print("=" * 80)
        print()

    # Sanity check: GPT-5.4 should have interior peak
    if "openai-gpt-5.4" in results:
        gpt54_results = results["openai-gpt-5.4"]
        max_idx = np.argmax(gpt54_results["L_profile"])
        peak_lambda = gpt54_results["lambda_grid"][max_idx]
        peak_L = gpt54_results["L_profile"][max_idx]
        L0 = gpt54_results["L_profile"][0]

        print("=" * 80)
        print("SANITY CHECK: GPT-5.4 (expected interior fit)")
        print("=" * 80)
        print(f"Peak at lambda = {peak_lambda:.4f}")
        print(f"L_profile({peak_lambda:.4f}) = {peak_L:.6f}")
        print(f"L_profile(0) = {L0:.6f}")
        print(f"Delta = {peak_L - L0:.6f}")

        if peak_lambda > 0 and peak_L > L0 + 1e-3:
            print("✓ PASS: Interior peak found as expected")
        else:
            print("✗ FAIL: No interior peak found — diagnostic may be buggy!")
        print("=" * 80)
        print()

    # Create plot if both models available
    if "openai-gpt-5.4-mini" in results and "openai-gpt-5.4" in results:
        output_path = RESULTS_DIR / "profile_likelihood_gpt54mini.png"
        create_profile_plot(
            results["openai-gpt-5.4-mini"]["lambda_grid"],
            results["openai-gpt-5.4-mini"]["L_profile"],
            results["openai-gpt-5.4"]["lambda_grid"],
            results["openai-gpt-5.4"]["L_profile"],
            output_path,
        )

    print("Done!")


if __name__ == "__main__":
    main()
