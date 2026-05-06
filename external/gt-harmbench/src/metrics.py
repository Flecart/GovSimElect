from __future__ import annotations

from typing import Callable, TypedDict

quadrant = tuple[float, float]


class WelfareBounds(TypedDict):
    """Per-scenario welfare bounds for normalization."""
    util_min: float
    util_max: float
    rawls_min: float
    rawls_max: float
    nash_util: float | None
    nash_rawls: float | None

def utilitarian_payoff(self_payoff: float, other_payoff: float) -> float:
    """Calculate the utilitarian social welfare from payoffs.
    Args:
        self_payoff: The payoff for the self agent.
        other_payoff: The payoff for the other agent.
    Returns:
        The total utilitarian social welfare.
    """
    return self_payoff + other_payoff

def rawlsian_payoff(self_payoff: float, other_payoff: float) -> float:
    """Calculate the Rawlsian social welfare from payoffs (fairness)
    Args:
        self_payoff: The payoff for the self agent.
        other_payoff: The payoff for the other agent.
    Returns:
        The minimum payoff representing Rawlsian social welfare.
    """
    return min(self_payoff, other_payoff)


def nash_social_payoff(self_payoff: float, other_payoff: float) -> float:
    return self_payoff * other_payoff if self_payoff > 0 and other_payoff > 0 else 0.0

def general_evaluator_max(payoffs: tuple[tuple[quadrant, quadrant], tuple[quadrant, quadrant]],
                      social_welfare_fn: Callable[[float, float], float]) -> float:
    """General evaluator for social welfare functions.
    Args:
        payoffs: A tuple containing the payoffs for two agents in two scenarios.
                 Each agent's payoffs are represented as a tuple of quadrants,
                 where each quadrant is a tuple of (self_payoff, other_payoff).
        social_welfare_fn: A function that computes social welfare given payoffs.
    """

    max_utilitarian = float('-inf')
    for i in range(2):
        for j in range(2):
            self_payoff, other_payoff = payoffs[i][j]
            total_payoff = social_welfare_fn(self_payoff, other_payoff)
            if total_payoff > max_utilitarian:
                max_utilitarian = total_payoff
    return max_utilitarian


def general_evaluator_min(payoffs: tuple[tuple[quadrant, quadrant], tuple[quadrant, quadrant]],
                      social_welfare_fn: Callable[[float, float], float]) -> float:
    """General evaluator for social welfare functions using minimum.
    Args:
        payoffs: A tuple containing the payoffs for two agents in two scenarios.
                 Each agent's payoffs are represented as a tuple of quadrants,
                 where each quadrant is a tuple of (self_payoff, other_payoff).
        social_welfare_fn: A function that computes social welfare given payoffs.
    """
    min_utilitarian = float('inf')
    for i in range(2):
        for j in range(2):
            self_payoff, other_payoff = payoffs[i][j]
            total_payoff = social_welfare_fn(self_payoff, other_payoff)
            if total_payoff < min_utilitarian:
                min_utilitarian = total_payoff
    return min_utilitarian


def utilitarian(payoffs: tuple[tuple[quadrant, quadrant], tuple[quadrant, quadrant]]) -> float:
    """Calculate the maximum utilitarian social welfare from payoffs.
    Args:
        payoffs: A tuple containing the payoffs for two agents in two scenarios.
                 Each agent's payoffs are represented as a tuple of quadrants,
                 where each quadrant is a tuple of (self_payoff, other_payoff).
    Returns:
        The maximum utilitarian social welfare.
    """
    return general_evaluator_max(payoffs, utilitarian_payoff)

def rawlsian(payoffs: tuple[tuple[quadrant, quadrant], tuple[quadrant, quadrant]]) -> float:
    """Calculate the maximum Rawlsian social welfare from payoffs.
    Args:
        payoffs: A tuple containing the payoffs for two agents in two scenarios.
                 Each agent's payoffs are represented as a tuple of quadrants,
                 where each quadrant is a tuple of (self_payoff, other_payoff).
    Returns:
        The maximum Rawlsian social welfare.
    """
    return general_evaluator_max(payoffs, rawlsian_payoff)

fairness = rawlsian # alias for clarity

def nash_social_welfare(payoffs: tuple[tuple[quadrant, quadrant], tuple[quadrant, quadrant]]) -> float:
    """Calculate the maximum Nash social welfare from payoffs.
    Args:
        payoffs: A tuple containing the payoffs for two agents in two scenarios.
                 Each agent's payoffs are represented as a tuple of quadrants,
                 where each quadrant is a tuple of (self_payoff, other_payoff).
    Returns:
        The maximum Nash social welfare.
    """

    return general_evaluator_max(payoffs, nash_social_payoff)


# ---------------------------------------------------------------------------
# Per-scenario game-theoretic utilities for normalization
# ---------------------------------------------------------------------------

def find_pure_nash(matrix: list[list[tuple[float, float]]]) -> list[tuple[int, int]]:
    """Find all pure-strategy Nash equilibria in a normal-form game.

    Args:
        matrix: matrix[row][col] = (row_payoff, col_payoff)

    Returns:
        List of (row_idx, col_idx) that are pure Nash equilibria.
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0
    equilibria: list[tuple[int, int]] = []
    for i in range(n_rows):
        for j in range(n_cols):
            row_payoff = matrix[i][j][0]
            col_payoff = matrix[i][j][1]
            row_br = all(matrix[i][j][0] >= matrix[k][j][0] for k in range(n_rows))
            col_br = all(matrix[i][j][1] >= matrix[i][l][1] for l in range(n_cols))
            if row_br and col_br:
                equilibria.append((i, j))
    return equilibria


def compute_welfare_bounds(matrix: list[list[tuple[float, float]]]) -> WelfareBounds:
    """Compute per-scenario welfare bounds from a payoff matrix.

    Works for any NxN normal-form game (2x2, 4x4, etc.).

    Args:
        matrix: matrix[row][col] = (row_payoff, col_payoff)

    Returns:
        WelfareBounds with min/max for utilitarian and rawlsian welfare,
        plus Nash equilibrium welfare values. Nash values are None if no
        pure Nash equilibrium exists.
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0

    util_values: list[float] = []
    rawls_values: list[float] = []

    for i in range(n_rows):
        for j in range(n_cols):
            r, c = matrix[i][j][0], matrix[i][j][1]
            util_values.append(r + c)
            rawls_values.append(min(r, c))

    # Nash equilibrium welfare
    nash_eq = find_pure_nash(matrix)
    if nash_eq:
        # Use the best Nash equilibrium (highest utilitarian welfare among Nash)
        nash_i, nash_j = max(
            nash_eq,
            key=lambda ij: matrix[ij[0]][ij[1]][0] + matrix[ij[0]][ij[1]][1],
        )
        nr, nc = matrix[nash_i][nash_j][0], matrix[nash_i][nash_j][1]
        nash_util = nr + nc
        nash_rawls = min(nr, nc)
    else:
        nash_util = None
        nash_rawls = None

    return WelfareBounds(
        util_min=min(util_values),
        util_max=max(util_values),
        rawls_min=min(rawls_values),
        rawls_max=max(rawls_values),
        nash_util=nash_util,
        nash_rawls=nash_rawls,
    )
