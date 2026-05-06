"""Estimate the lambda-cooperative parameter from observed decision matrices via QRE."""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import tqdm
import zipfile

logger = logging.getLogger(__name__)


def normalize_payoff_matrix(
    payoff_matrix: list[list[list[float]]],
) -> list[list[list[float]]]:
    """Normalize a payoff matrix so that max|entry| = 1.

    Computes M_s = max over all (i, j, k) of |Π_s[i, j, k]| and divides
    every entry by M_s. The welfare term W[i,j] = u_r[i,j] + u_c[i,j] must
    be computed from the *normalized* payoffs (W lies in roughly [-2, 2]).
    Count matrices are unchanged.

    Args:
        payoff_matrix: 3D list [row][col][player] of payoffs.

    Returns:
        Normalized payoff matrix (same shape), or the original if M_s == 0.
    """
    arr = np.array(payoff_matrix, dtype=float)
    m = np.max(np.abs(arr))
    if m == 0:
        return payoff_matrix
    arr /= m
    return arr.tolist()


def load_per_scenario_payoffs(
    dataset_2x2: Path | None,
    dataset_4x4: Path | None,
) -> dict[str, dict[str, list[list[list[float]]]]]:
    """Load per-scenario payoff matrices from dataset CSVs.

    Args:
        dataset_2x2: Path to 2x2 dataset CSV (with targets).
        dataset_4x4: Path to 4x4 rewritten dataset CSV.

    Returns:
        Dict mapping scenario_id to payoff matrix.
        Has keys "2x2" and "4x4", each mapping scenario_id to payoff matrix.
    """
    payoff_matrices = {"2x2": {}, "4x4": {}}

    if dataset_2x2 and dataset_2x2.exists():
        with open(dataset_2x2) as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario_id = row.get("id")
                if not scenario_id:
                    continue
                try:
                    p11 = json.loads(row["1_1_payoff"])
                    p12 = json.loads(row["1_2_payoff"])
                    p21 = json.loads(row["2_1_payoff"])
                    p22 = json.loads(row["2_2_payoff"])
                    payoff_matrices["2x2"][scenario_id] = [[p11, p12], [p21, p22]]
                except (KeyError, json.JSONDecodeError):
                    continue

    if dataset_4x4 and dataset_4x4.exists():
        with open(dataset_4x4) as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario_id = row.get("id")
                if not scenario_id:
                    continue
                pm_str = row.get("payoff_matrix_4x4")
                if pm_str:
                    try:
                        payoff_matrices["4x4"][scenario_id] = json.loads(pm_str)
                    except json.JSONDecodeError:
                        continue

    return payoff_matrices


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _compute_lambda_cooperative_payoffs(
    payoff_matrix: list[list[list[float]]],
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lambda-cooperative modified payoffs for both players.

    Args:
        payoff_matrix: 3D list [row][col][player] of original payoffs.
        lam: Lambda parameter (prosociality).

    Returns:
        Tuple of (V_row, V_col) where each is a 2D array of modified payoffs.
    """
    n_rows = len(payoff_matrix)
    n_cols = len(payoff_matrix[0]) if payoff_matrix else 0

    V_row = np.zeros((n_rows, n_cols))
    V_col = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            r_pay = payoff_matrix[i][j][0]
            c_pay = payoff_matrix[i][j][1]
            social_welfare = r_pay + c_pay
            V_row[i, j] = r_pay + lam * social_welfare
            V_col[i, j] = c_pay + lam * social_welfare

    return V_row, V_col


def _solve_asymmetric_qre(
    lam: float,
    beta: float,
    payoff_matrix: list[list[list[float]]],
    sigma_r_init: np.ndarray | None = None,
    sigma_c_init: np.ndarray | None = None,
    max_iter: int = 5000,
    tol: float = 1e-6,
    damping: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Solve the two-sided logit QRE via alternating best response with damping.

    Implements the fixed-point iteration from equations (B.1)-(B.2) with damping:
        sigma_r(i) ∝ exp(beta * sum_j sigma_c(j) * V_r^lambda[i,j])
        sigma_c(j) ∝ exp(beta * sum_i sigma_r(i) * V_c^lambda[i,j])

    The damped update is: sigma^{t+1} = damping * sigma^t + (1 - damping) * BR(sigma^t)

    Args:
        lam: Lambda parameter.
        beta: Inverse temperature.
        payoff_matrix: 3D payoff matrix.
        sigma_r_init: Initial row strategy (uniform if None).
        sigma_c_init: Initial column strategy (uniform if None).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        damping: Damping coefficient (0.5 = average of old and new strategy).

    Returns:
        Tuple of (sigma_r, sigma_c, n_iterations, converged).
    """
    V_row, V_col = _compute_lambda_cooperative_payoffs(payoff_matrix, lam)
    n_rows, n_cols = V_row.shape

    if sigma_r_init is not None:
        sigma_r = sigma_r_init.copy()
    else:
        sigma_r = np.ones(n_rows) / n_rows
    if sigma_c_init is not None:
        sigma_c = sigma_c_init.copy()
    else:
        sigma_c = np.ones(n_cols) / n_cols

    for it in range(max_iter):
        EU_r = V_row @ sigma_c
        EU_c = sigma_r @ V_col

        br_sigma_r = _softmax(beta * EU_r)
        br_sigma_c = _softmax(beta * EU_c)

        new_sigma_r = damping * sigma_r + (1 - damping) * br_sigma_r
        new_sigma_c = damping * sigma_c + (1 - damping) * br_sigma_c

        if (np.max(np.abs(new_sigma_r - sigma_r)) < tol and
                np.max(np.abs(new_sigma_c - sigma_c)) < tol):
            return new_sigma_r, new_sigma_c, it + 1, True

        sigma_r = new_sigma_r
        sigma_c = new_sigma_c

    max_change_r = np.max(np.abs(new_sigma_r - sigma_r)) if it > 0 else 0.0
    max_change_c = np.max(np.abs(new_sigma_c - sigma_c)) if it > 0 else 0.0
    logger.warning(
        "QRE did not converge: lam=%.4f, beta=%.4f after %d iterations. "
        "Final residuals: max(|sigma_r - sigma_r^new|)=%.2e, max(|sigma_c - sigma_c^new|)=%.2e",
        lam, beta, max_iter, max_change_r, max_change_c,
    )
    return new_sigma_r, new_sigma_c, max_iter, False


def solve_qre(
    lam: float,
    beta: float,
    payoff_matrix: list[list[list[float]]],
    sigma_r_init: np.ndarray | None = None,
    sigma_c_init: np.ndarray | None = None,
    max_iter: int = 5000,
    tol: float = 1e-6,
    damping: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Solve QRE via alternating best response with damping (asymmetric solver).

    Uses the asymmetric solver for all games with damped updates to suppress
    oscillation around unstable symmetric saddles.

    Args:
        lam: Lambda parameter.
        beta: Inverse temperature.
        payoff_matrix: 3D payoff matrix.
        sigma_r_init: Initial row strategy (uniform if None).
        sigma_c_init: Initial column strategy (uniform if None).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        damping: Damping coefficient (0.5 = average of old and new strategy).

    Returns:
        Tuple of (sigma_r, sigma_c, n_iterations, converged).
    """
    sigma_r, sigma_c, n_iter, converged = _solve_asymmetric_qre(
        lam, beta, payoff_matrix, sigma_r_init, sigma_c_init, max_iter, tol, damping,
    )
    return sigma_r, sigma_c, n_iter, converged


def _scenario_id_for_qre(item: dict, scorer_meta: dict) -> str | None:
    """Resolve dataset scenario id for joining with payoff CSV rows.

    Prefer contracting scorer's base_scenario_id; fall back to sample metadata id
    (inspect_ai summaries often omit base_scenario_id on the scorer dict).
    """
    sid = scorer_meta.get("base_scenario_id")
    if sid is not None and str(sid).strip() != "":
        return str(sid)
    sample_meta = item.get("metadata") or {}
    sid = sample_meta.get("base_scenario_id") or sample_meta.get("id")
    if sid is not None and str(sid).strip() != "":
        return str(sid)
    return None


def _row_col_from_all_strategies_scorer(scorer: dict) -> tuple[str, str] | None:
    """Parse row/column actions from main eval (all_strategies_scorer) summary entry."""
    answer = scorer.get("answer")
    if not answer or not isinstance(answer, str):
        return None
    first_line = answer.strip().split("\n")[0]
    m = re.search(
        r"Row choice:\s*([^,]+)\s*,\s*Column choice:\s*(.+?)\s*$",
        first_line,
        re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip().rstrip(" .;*")


def _index_in_action_list(chosen: str, actions: list[str]) -> int | None:
    """Index of `chosen` in the scenario's ordered action list (case-sensitive, then case-fold)."""
    if not chosen or not actions:
        return None
    for i, a in enumerate(actions):
        if a == chosen:
            return i
    c = chosen.strip().lower()
    for i, a in enumerate(actions):
        if a.strip().lower() == c:
            return i
    return None


def extract_joint_actions_from_eval(eval_path: Path) -> dict[str, list[list[int]]]:
    """Extract joint action counts per scenario from a .eval file.

    Reads raw .eval ZIP files and extracts per-sample action data,
    grouping by base_scenario_id to compute joint action frequencies.

    Args:
        eval_path: Path to .eval file.

    Returns:
        Dict mapping base_scenario_id to joint action matrix (2D list of counts).
    """
    scenario_matrices: dict[str, list[list[int]]] = {}
    scenario_dims: dict[str, tuple[int, int]] = {}

    with zipfile.ZipFile(eval_path, "r") as z:
        summary_files = sorted(n for n in z.namelist() if n.startswith("_journal/summaries/"))

        for sf in summary_files:
            with z.open(sf) as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]

                for item in items:
                    scores = item.get("scores", {})
                    row_action: str | None = None
                    col_action: str | None = None
                    scorer_meta: dict = {}

                    for sname in scores:
                        if sname.startswith("contracting_scorer"):
                            scorer_data = scores[sname]
                            scorer_meta = scorer_data.get("metadata", {}) or {}
                            row_action = scorer_meta.get("row_action")
                            col_action = scorer_meta.get("column_action") or scorer_meta.get(
                                "col_action"
                            )
                            break

                    if row_action is None or col_action is None:
                        ass = scores.get("all_strategies_scorer")
                        if isinstance(ass, dict):
                            parsed = _row_col_from_all_strategies_scorer(ass)
                            if parsed:
                                row_action, col_action = parsed

                    scenario_id = _scenario_id_for_qre(item, scorer_meta)
                    sample_meta = item.get("metadata") or {}
                    ar = sample_meta.get("actions_row")
                    ac = sample_meta.get("actions_column")
                    if not scenario_id or not row_action or not col_action:
                        continue
                    if not isinstance(ar, list) or not isinstance(ac, list):
                        continue

                    ri = _index_in_action_list(row_action, ar)
                    ci = _index_in_action_list(col_action, ac)
                    if ri is None or ci is None:
                        continue

                    if scenario_id not in scenario_matrices:
                        scenario_matrices[scenario_id] = [
                            [0] * len(ac) for _ in range(len(ar))
                        ]
                        scenario_dims[scenario_id] = (len(ar), len(ac))
                    else:
                        exp_r, exp_c = scenario_dims[scenario_id]
                        if exp_r != len(ar) or exp_c != len(ac):
                            continue

                    scenario_matrices[scenario_id][ri][ci] += 1

    return scenario_matrices


def _scenario_ll(
    lam: float,
    beta: float,
    payoff_matrix: list[list[list[float]]],
    counts: list[list[int]],
    max_iter: int = 5000,
    tol: float = 1e-6,
    damping: float = 0.5,
) -> tuple[float, int, bool]:
    """Log-likelihood of a single scenario's count matrix under QRE.

    Args:
        lam: Lambda parameter.
        beta: Inverse temperature.
        payoff_matrix: 3D payoff matrix for this scenario (normalized).
        counts: 2D count matrix of observed joint actions.
        max_iter: Maximum iterations for QRE solver.
        tol: Convergence tolerance.
        damping: Damping coefficient for QRE solver.

    Returns:
        Tuple of (log_likelihood, n_iterations, converged).
        Returns (-inf, n_iter, False) if QRE fails to converge.
    """
    sigma_r, sigma_c, n_iter, converged = solve_qre(
        lam, beta, payoff_matrix,
        sigma_r_init=None,
        sigma_c_init=None,
        max_iter=max_iter,
        tol=tol,
        damping=damping,
    )

    if not converged:
        return -np.inf, n_iter, False

    n_rows, n_cols = len(counts), len(counts[0])
    ll = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            c = counts[i][j]
            if c > 0:
                p = sigma_r[i] * sigma_c[j]
                if p > 1e-300:
                    ll += c * np.log(p)
                else:
                    return -np.inf, n_iter, False
    return ll, n_iter, True


def _pooled_ll_cold(
    lam: float,
    beta: float,
    scenarios: list[tuple[list[list[list[float]]], list[list[int]]]],
    max_iter: int = 5000,
    tol: float = 1e-6,
    damping: float = 0.5,
) -> tuple[float, int, bool]:
    """Pooled log-likelihood across all scenarios with cold-start QRE.

    Each scenario is solved independently from uniform initialization.
    Returns -inf if any scenario fails to converge.

    Args:
        lam: Shared lambda parameter.
        beta: Shared inverse temperature.
        scenarios: List of (payoff_matrix, counts) tuples.
        max_iter: Maximum iterations for QRE solver.
        tol: Convergence tolerance.
        damping: Damping coefficient for QRE solver.

    Returns:
        Tuple of (total_ll, total_iterations, all_converged).
    """
    total = 0.0
    total_iter = 0
    all_converged = True

    for payoff_matrix, counts in scenarios:
        ll, n_iter, converged = _scenario_ll(
            lam, beta, payoff_matrix, counts,
            max_iter=max_iter,
            tol=tol,
            damping=damping,
        )

        total_iter += n_iter

        if not converged or ll == -np.inf:
            all_converged = False
            return -np.inf, total_iter, False

        total += ll

    return total, total_iter, True


def _grid_search_cold(
    scenarios: list[tuple[list[list[list[float]]], list[list[int]]]],
    lambda_grid: np.ndarray,
    beta_grid: np.ndarray,
    n_workers: int = 8,
    desc: str = "Grid search",
    show_progress: bool = True,
) -> dict:
    """Grid search with cold-start QRE at each (lambda, beta) point.

    Each grid point is solved independently from uniform initialization.
    Lambda and beta dimensions are both parallelizable.

    Args:
        scenarios: List of (payoff_matrix, counts) tuples.
        lambda_grid: Lambda search grid.
        beta_grid: Beta search grid.
        n_workers: Number of parallel workers.
        desc: Description for progress bar.
        show_progress: Whether to show progress bar.

    Returns:
        Dict with keys: lambda_hat, beta_hat, log_likelihood, iteration_stats.
    """
    work_items = [
        (float(lam), float(beta), scenarios)
        for lam in lambda_grid
        for beta in beta_grid
    ]

    all_results = []
    all_iters = []

    if show_progress and desc:
        if n_workers > 1 and len(work_items) > 1:
            with Pool(min(n_workers, len(work_items))) as pool:
                results = list(tqdm.tqdm(
                    pool.imap_unordered(_grid_point_worker, work_items),
                    total=len(work_items),
                    desc=desc,
                    ncols=80,
                ))
                all_results = results
        else:
            all_results = [
                _grid_point_worker((lam, beta, scenarios))
                for lam, beta, scenarios in tqdm.tqdm(
                    work_items, desc=desc, ncols=80,
                )
            ]
    else:
        if n_workers > 1 and len(work_items) > 1:
            with Pool(min(n_workers, len(work_items))) as pool:
                all_results = list(pool.imap_unordered(_grid_point_worker, work_items))
        else:
            all_results = [
                _grid_point_worker((lam, beta, scenarios))
                for lam, beta, scenarios in work_items
            ]

    best_ll = -np.inf
    best_lam = 0.0
    best_beta = 1.0

    for lam, beta, ll, total_iter in all_results:
        all_iters.append(total_iter)
        if ll > best_ll:
            best_ll = ll
            best_lam = lam
            best_beta = beta

    iter_arr = np.array(all_iters) if all_iters else np.array([0])
    iter_stats = {
        "median_iterations": float(np.median(iter_arr)),
        "max_iterations": float(np.max(iter_arr)),
        "mean_iterations": float(np.mean(iter_arr)),
    }

    return {
        "lambda_hat": best_lam,
        "beta_hat": best_beta,
        "log_likelihood": float(best_ll),
        "iteration_stats": iter_stats,
    }


def _grid_point_worker(args: tuple) -> tuple[float, float, float, int]:
    """Worker for parallel grid search: evaluate one (lambda, beta) point."""
    lam, beta, scenarios = args
    ll, total_iter, _ = _pooled_ll_cold(lam, beta, scenarios)
    return lam, beta, ll, total_iter


def _check_iteration_stats(iter_stats: dict, threshold: int = 1000) -> None:
    """Log iteration stats at info level."""
    median = iter_stats["median_iterations"]
    logger.info(
        "QRE iteration stats: median=%.0f, max=%.0f, mean=%.0f",
        median, iter_stats["max_iterations"], iter_stats["mean_iterations"],
    )


def _is_payoff_symmetric(
    payoff_matrix: list[list[list[float]]],
    tol: float = 1e-8,
) -> bool:
    """Check whether a 2-player payoff matrix is symmetric.

    A payoff matrix is symmetric when u_c[i, j] = u_r[j, i] for all i, j.
    """
    n_rows = len(payoff_matrix)
    n_cols = len(payoff_matrix[0]) if payoff_matrix else 0
    if n_rows != n_cols:
        return False
    for i in range(n_rows):
        for j in range(n_cols):
            r_pay = payoff_matrix[i][j][0]
            c_pay = payoff_matrix[i][j][1]
            r_transposed = payoff_matrix[j][i][0]
            if abs(c_pay - r_transposed) > tol:
                return False
    return True


def _check_symmetric_solutions_at_low_beta(
    scenarios: list[tuple[list[list[list[float]]], list[list[int]]]],
    lam: float,
    beta_hat: float,
    sigma_r_dict: dict,
    sigma_c_dict: dict,
    tol: float = 1e-3,
    beta_threshold: float = 1.0,
) -> None:
    """Check that symmetric games have symmetric solutions at low beta.

    For each scenario with a symmetric payoff matrix, compute |sigma_r* - sigma_c*|.
    If > tol at beta < beta_threshold, log a warning.
    """
    if beta_hat >= beta_threshold:
        return

    n_asymmetric = 0
    for idx, (payoff_matrix, _counts) in enumerate(scenarios):
        if not _is_payoff_symmetric(payoff_matrix):
            continue

        sigma_r = sigma_r_dict.get(idx)
        sigma_c = sigma_c_dict.get(idx)
        if sigma_r is None or sigma_c is None:
            continue

        diff = np.max(np.abs(sigma_r - sigma_c))
        if diff > tol:
            n_asymmetric += 1
            logger.warning(
                "Symmetric game has asymmetric solution at low beta: "
                "lambda=%.4f, beta_hat=%.4f, scenario=%d, |sigma_r - sigma_c|=%.2e.",
                lam, beta_hat, idx, diff,
            )

    if n_asymmetric > 0:
        logger.warning(
            "Found %d/%d symmetric scenarios with asymmetric solutions at beta_hat=%.4f.",
            n_asymmetric, len(scenarios), beta_hat,
        )


def _check_point_estimate_convergence(
    scenarios: list[tuple[list[list[list[float]]], list[list[int]]]],
    lambda_hat: float,
    beta_hat: float,
) -> None:
    """Post-fit check: verify QRE converged for all scenarios at point estimate.

    Raises:
        RuntimeError: If any scenario fails to converge at the point estimate.
    """
    failed_scenarios = []
    residuals = []

    for idx, (payoff_matrix, counts) in enumerate(scenarios):
        sigma_r, sigma_c, n_iter, converged = solve_qre(
            lambda_hat, beta_hat, payoff_matrix,
        )

        if not converged:
            failed_scenarios.append(idx)
            sigma_r_new, sigma_c_new, _, _ = solve_qre(
                lambda_hat, beta_hat, payoff_matrix,
                sigma_r_init=sigma_r,
                sigma_c_init=sigma_c,
            )
            residual_r = np.max(np.abs(sigma_r_new - sigma_r))
            residual_c = np.max(np.abs(sigma_c_new - sigma_c))
            residuals.append((idx, residual_r, residual_c, n_iter))

    if failed_scenarios:
        logger.error(
            "QRE failed to converge at point estimate (lambda=%.4f, beta=%.4f) "
            "for %d/%d scenarios: %s",
            lambda_hat, beta_hat, len(failed_scenarios), len(scenarios),
            failed_scenarios[:10],
        )
        for idx, res_r, res_c, n_iter in residuals[:5]:
            logger.error(
                "  Scenario %d: n_iter=%d, residual_r=%.2e, residual_c=%.2e",
                idx, n_iter, res_r, res_c,
            )
        raise RuntimeError(
            f"QRE failed to converge at point estimate (lambda={lambda_hat:.4f}, "
            f"beta={beta_hat:.4f}) for {len(failed_scenarios)}/{len(scenarios)} scenarios."
        )

    logger.info(
        "Post-fit check passed: all %d scenarios converged at (lambda=%.4f, beta=%.4f)",
        len(scenarios), lambda_hat, beta_hat,
    )


def estimate_pooled_qre(
    scenarios: list[tuple[list[list[list[float]]], list[list[int]]]],
    lambda_grid: np.ndarray | None = None,
    beta_grid: np.ndarray | None = None,
    n_workers: int = 8,
    desc: str = "QRE estimation",
    show_progress: bool = True,
) -> dict:
    """Estimate shared (lambda, beta) by maximising pooled log-likelihood.

    Uses coarse-to-fine 2-D grid search with cold-start QRE at each point:
    1. Coarse search over full grid (default 40x40), fully parallelized
    2. Fine search around best point (20x20 refined), same cold-start approach

    Args:
        scenarios: List of (payoff_matrix, counts) tuples.
        lambda_grid: Lambda search grid (full range). Default: np.linspace(0, 3, 40).
        beta_grid: Beta search grid (full range). Default: np.linspace(0.1, 10, 40).
        n_workers: Number of parallel workers (all grid points parallelizable).
        desc: Description for progress bars.
        show_progress: Whether to show grid search progress bars.

    Returns:
        Dict with keys: lambda_hat, beta_hat, log_likelihood, iteration_stats.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0, 3, 40)
    if beta_grid is None:
        beta_grid = np.linspace(0.1, 10, 40)

    # Coarse search
    result = _grid_search_cold(
        scenarios, lambda_grid, beta_grid, n_workers,
        desc=f"{desc} (coarse)" if show_progress else None,
        show_progress=show_progress,
    )
    best_lam = result["lambda_hat"]
    best_beta = result["beta_hat"]
    best_ll = result["log_likelihood"]
    iter_stats = result.get("iteration_stats", {})

    if iter_stats:
        _check_iteration_stats(iter_stats)

    # Verify symmetric games have symmetric solutions at low beta
    if best_beta < 1.0:
        sigma_r_dict = {}
        sigma_c_dict = {}
        for idx, (payoff_matrix, _counts) in enumerate(scenarios):
            try:
                sigma_r, sigma_c, _, converged = solve_qre(best_lam, best_beta, payoff_matrix)
                if converged:
                    sigma_r_dict[idx] = sigma_r
                    sigma_c_dict[idx] = sigma_c
            except Exception:
                pass
        _check_symmetric_solutions_at_low_beta(
            scenarios, best_lam, best_beta, sigma_r_dict, sigma_c_dict,
        )

    # Fine search around best point
    lam_idx = np.searchsorted(lambda_grid, best_lam)
    beta_idx = np.searchsorted(beta_grid, best_beta)

    lam_lo = max(0, lam_idx - 2)
    lam_hi = min(len(lambda_grid) - 1, lam_idx + 2)
    beta_lo_idx = max(0, beta_idx - 2)
    beta_hi_idx = min(len(beta_grid) - 1, beta_idx + 2)

    if lam_hi > lam_lo:
        lambda_refine = np.linspace(lambda_grid[lam_lo], lambda_grid[lam_hi], 20)
    else:
        lambda_refine = np.linspace(best_lam * 0.8, best_lam * 1.2 + 1e-6, 20)

    if beta_hi_idx > beta_lo_idx:
        beta_refine = np.linspace(beta_grid[beta_lo_idx], beta_grid[beta_hi_idx], 20)
    else:
        beta_refine = np.linspace(best_beta * 0.8, best_beta * 1.2 + 1e-6, 20)

    result_refine = _grid_search_cold(
        scenarios, lambda_refine, beta_refine, n_workers,
        desc=f"{desc} (refine)" if show_progress else None,
        show_progress=show_progress,
    )

    if result_refine["log_likelihood"] > best_ll:
        refine_stats = result_refine.get("iteration_stats", {})
        if refine_stats:
            _check_iteration_stats(refine_stats)
        _check_point_estimate_convergence(
            scenarios, result_refine["lambda_hat"], result_refine["beta_hat"],
        )
        return result_refine

    _check_point_estimate_convergence(scenarios, best_lam, best_beta)

    return {
        "lambda_hat": float(best_lam),
        "beta_hat": float(best_beta),
        "log_likelihood": float(best_ll),
        "iteration_stats": iter_stats,
    }


def bootstrap_ci_pooled(
    scenarios: list[tuple[list[list[list[float]]], list[list[int]]]],
    n_bootstrap: int = 1000,
    lambda_grid: np.ndarray | None = None,
    beta_grid: np.ndarray | None = None,
    ci: float = 0.95,
    seed: int = 42,
    n_workers: int = 8,
    desc: str = "Bootstrap",
    worker_id: int = 0,
    show_nested_progress: bool = True,
    progress_queue=None,
) -> dict:
    """Scenario-clustered bootstrap CIs for pooled (lambda, beta).

    Resamples scenarios with replacement (preserving all observations within
    each scenario), refits (lambda, beta) for each resample using cold-start
    grid search, and returns percentile CIs.

    Args:
        scenarios: List of (payoff_matrix, counts) tuples.
        n_bootstrap: Number of bootstrap resamples (default 1000).
        lambda_grid: Lambda search grid.
        beta_grid: Beta search grid.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        n_workers: Number of parallel workers for grid evaluation within each
            bootstrap sample.
        desc: Description for progress bars.
        worker_id: Worker ID for progress tracking.
        show_nested_progress: Whether to show grid search progress bars.
        progress_queue: Queue for reporting progress updates (worker_id, current_sample).

    Returns:
        Dict with keys: lambda_hat, beta_hat, log_likelihood, lambda_ci,
        beta_ci, bootstrap_lambdas, bootstrap_betas.
    """
    rng = np.random.default_rng(seed)
    n_scenarios = len(scenarios)

    point = estimate_pooled_qre(
        scenarios, lambda_grid, beta_grid, n_workers,
        desc=None,
        show_progress=show_nested_progress,
    )

    all_indices = [
        rng.choice(n_scenarios, size=n_scenarios, replace=True)
        for _ in range(n_bootstrap)
    ]

    boot_lambdas = []
    boot_betas = []

    for i, indices in enumerate(all_indices):
        resampled = [scenarios[j] for j in indices]
        est = estimate_pooled_qre(
            resampled, lambda_grid, beta_grid, n_workers,
            desc=None,
            show_progress=False,
        )
        boot_lambdas.append(est["lambda_hat"])
        boot_betas.append(est["beta_hat"])

        if progress_queue is not None:
            progress_queue.put((worker_id, i + 1))

    alpha = (1 - ci) / 2
    point["lambda_ci"] = (
        float(np.percentile(boot_lambdas, alpha * 100)),
        float(np.percentile(boot_lambdas, (1 - alpha) * 100)),
    )
    point["beta_ci"] = (
        float(np.percentile(boot_betas, alpha * 100)),
        float(np.percentile(boot_betas, (1 - alpha) * 100)),
    )
    point["bootstrap_lambdas"] = np.array(boot_lambdas)
    point["bootstrap_betas"] = np.array(boot_betas)

    return point


def collect_model_scenarios(
    eval_dirs_by_game: dict[str, Path],
    payoff_dict_by_game: dict[str, dict[str, list[list[list[float]]]]],
    prompt_mode: str = "base",
    contract_mode: str = "no-comm",
    dataset_size: str = "2x2",
) -> list[tuple[list[list[list[float]]], list[list[int]]]]:
    """Collect (payoff_matrix, joint_action_counts) tuples for one model.

    Loads raw .eval files from each game's eval directory, extracts per-scenario
    joint action counts, joins with per-scenario payoff matrices, and normalizes
    payoffs.

    Args:
        eval_dirs_by_game: Dict mapping game name (e.g. "pd", "sh") to eval root dir.
        payoff_dict_by_game: Dict mapping game name to dict mapping scenario_id to
            payoff matrix (from load_per_scenario_payoffs).
        prompt_mode: Prompt mode subdirectory (e.g. "base").
        contract_mode: Contract mode subdirectory (e.g. "no-comm").
        dataset_size: "2x2" or "4x4" to select payoff dict key.

    Returns:
        List of (payoff_matrix, joint_action_counts) tuples,
        one per scenario that has both payoff data and observed actions.
        Payoff matrices are normalized (max|entry| = 1).
    """
    scenarios: list[tuple[list[list[list[float]]], list[list[int]]]] = []

    exp_subdir = f"{dataset_size}-{contract_mode}"

    for game, eval_dir in eval_dirs_by_game.items():
        payoff_dict = payoff_dict_by_game.get(game, {})
        eval_path = eval_dir / prompt_mode / exp_subdir

        if not eval_path.exists():
            continue

        eval_files = list(eval_path.glob("*.eval"))
        if not eval_files:
            continue

        all_actions: dict[str, list[list[int]]] = {}
        for ef in eval_files:
            actions = extract_joint_actions_from_eval(ef)
            for sid, matrix in actions.items():
                if sid not in all_actions:
                    all_actions[sid] = matrix
                else:
                    n_rows = len(matrix)
                    n_cols = len(matrix[0]) if matrix else 0
                    for i in range(n_rows):
                        for j in range(n_cols):
                            all_actions[sid][i][j] += matrix[i][j]

        for sid, counts in all_actions.items():
            n_obs = sum(sum(r) for r in counts)
            if n_obs == 0:
                continue
            pm = payoff_dict.get(sid)
            if pm is not None:
                pm_normalized = normalize_payoff_matrix(pm)
                scenarios.append((pm_normalized, counts))

    return scenarios
