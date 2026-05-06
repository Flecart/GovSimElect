"""
Verify that generated 4x4 matrices satisfy the axioms and structural conditions
from the theory appendix (Section: Table Game Structure).

This script is a standalone check: it reads the CSV, reconstructs the 4x4 matrix
from the base 2x2 payoffs, and verifies every claim the theory makes.
"""

import json
import sys
from pathlib import Path

import click
import pandas as pd

# STRATEGY INDICES
CH = 0  # Coop-High
CL = 1  # Coop-Low
DH = 2  # Defect-High
DL = 3  # Defect-Low

STRATEGY_NAMES = ["Coop-High", "Coop-Low", "Defect-High", "Defect-Low"]

COOP_ACTIONS = [CH, CL]
DEFECT_ACTIONS = [DH, DL]


# PARAMETRIC CONSTRUCTION

def compute_epsilon(R: float, P: float) -> float:
    """Effort cost: epsilon = (R - P) / 6."""
    return (R - P) / 6


def compute_eta(R: float, P: float) -> float:
    """Symmetry-breaking perturbation: eta = (R - P) / 12 = epsilon / 2."""
    return (R - P) / 12


def build_theory_matrix(R: float, S: float, T: float, P: float):
    """
    Build the 4x4 matrix exactly as specified in the theory appendix.
    Returns matrix[row][col] = (row_payoff, col_payoff) in natural (unscaled) units.
    """
    e = compute_epsilon(R, P)
    eta = compute_eta(R, P)
    mid = (R + P) / 2  # = R - 3*epsilon

    return [
        #                         vs CH                vs CL                 vs DH              vs DL
        [(R - e, R - e), (R - 2 * e - eta, R), (S - e, T - e), (S - e, T - 2 * e)],  # CH
        [(R, R - 2 * e - eta), (mid, mid), (S - 3 * e, T - e), (S - 3 * e, T - 2 * e)],  # CL
        [(T - e, S - e), (T - e, S - 3 * e), (P - e, P - e), (P, P - 2 * e)],  # DH
        [(T - 2 * e, S - e), (T - 2 * e, S - 3 * e), (P - 2 * e, P), (P - 3 * e, P - 3 * e)],  # DL
    ]

# STRUCTURAL CONDITIONS

def check_pd_conditions(R, S, T, P):
    """Check Theorem 1 (PD parametric family) conditions P1-P5."""
    results = {}
    results["P1"] = {
        "cond": "T > R > P > S",
        "pass": T > R > P > S,
        "vals": f"T={T} R={R} P={P} S={S}",
    }
    results["P2"] = {
        "cond": "3(T-R) > R-P",
        "pass": 3 * (T - R) > (R - P),
        "vals": f"3*({T}-{R})={3*(T-R):.3f} vs {R}-{P}={R-P:.3f}",
    }
    results["P3"] = {
        "cond": "3(P-S) > R-P",
        "pass": 3 * (P - S) > (R - P),
        "vals": f"3*({P}-{S})={3*(P-S):.3f} vs {R}-{P}={R-P:.3f}",
    }
    results["P4"] = {
        "cond": "2(S+T) > R+3P",
        "pass": 2 * (S + T) > R + 3 * P,
        "vals": f"2*({S}+{T})={2*(S+T):.3f} vs {R}+3*{P}={R+3*P:.3f}",
    }
    results["P5"] = {
        "cond": "3(S+T) < 5R+P",
        "pass": 3 * (S + T) < 5 * R + P,
        "vals": f"3*({S}+{T})={3*(S+T):.3f} vs 5*{R}+{P}={5*R+P:.3f}",
    }
    return results


def check_sh_conditions(R, S, T, P):
    """Check Theorem 3 (SH parametric family) conditions S1-S4."""
    results = {}
    results["S1"] = {
        "cond": "R > T > P > S",
        "pass": R > T > P > S,
        "vals": f"R={R} T={T} P={P} S={S}",
    }
    results["S2"] = {
        "cond": "2R > S+T",
        "pass": 2 * R > S + T,
        "vals": f"2*{R}={2*R:.3f} vs {S}+{T}={S+T:.3f}",
    }
    results["S3"] = {
        "cond": "2(S+T) > R+3P",
        "pass": 2 * (S + T) > R + 3 * P,
        "vals": f"2*({S}+{T})={2*(S+T):.3f} vs {R}+3*{P}={R+3*P:.3f}",
    }
    results["S4"] = {
        "cond": "3(S+T) < 5R+P",
        "pass": 3 * (S + T) < 5 * R + P,
        "vals": f"3*({S}+{T})={3*(S+T):.3f} vs 5*{R}+{P}={5*R+P:.3f}",
    }
    return results


# AXIOM CHECKS

def check_axiom_i(matrix):
    """
    Axiom (i): Social improvement via cooperation.
    For every player, replacing a defection action with a cooperation action
    strictly improves total welfare (for any opponent action).
    """
    violations = []
    for coop_idx in COOP_ACTIONS:
        for defect_idx in DEFECT_ACTIONS:
            for opp_idx in range(4):
                # Row player switching from defect to coop
                w_coop = matrix[coop_idx][opp_idx][0] + matrix[coop_idx][opp_idx][1]
                w_defect = matrix[defect_idx][opp_idx][0] + matrix[defect_idx][opp_idx][1]
                if w_coop <= w_defect:
                    violations.append(
                        f"[i-row] W({STRATEGY_NAMES[coop_idx]}, {STRATEGY_NAMES[opp_idx]})="
                        f"{w_coop:.4f} <= W({STRATEGY_NAMES[defect_idx]}, {STRATEGY_NAMES[opp_idx]})="
                        f"{w_defect:.4f}"
                    )
                # Column player switching from defect to coop
                w_coop_col = matrix[opp_idx][coop_idx][0] + matrix[opp_idx][coop_idx][1]
                w_defect_col = matrix[opp_idx][defect_idx][0] + matrix[opp_idx][defect_idx][1]
                if w_coop_col <= w_defect_col:
                    violations.append(
                        f"[i-col] W({STRATEGY_NAMES[opp_idx]}, {STRATEGY_NAMES[coop_idx]})="
                        f"{w_coop_col:.4f} <= W({STRATEGY_NAMES[opp_idx]}, {STRATEGY_NAMES[defect_idx]})="
                        f"{w_defect_col:.4f}"
                    )
    return violations


def check_axiom_ii(matrix):
    """
    Axiom (ii): Unique welfare maximum at (CH, CH).
    W(CH, CH) > W(i, j) for all (i,j) != (CH, CH).
    """
    violations = []
    ch_ch_w = matrix[CH][CH][0] + matrix[CH][CH][1]
    for i in range(4):
        for j in range(4):
            if (i, j) == (CH, CH):
                continue
            w = matrix[i][j][0] + matrix[i][j][1]
            if w >= ch_ch_w:
                violations.append(
                    f"W({STRATEGY_NAMES[i]}, {STRATEGY_NAMES[j]})={w:.4f} "
                    f">= W(CH,CH)={ch_ch_w:.4f}"
                )
    return violations


def check_axiom_iii_pareto(matrix):
    """
    Mutual cooperation preference: every cooperative profile Pareto-dominates
    every defective profile.
    """
    violations = []
    for ci in COOP_ACTIONS:
        for cj in COOP_ACTIONS:
            for di in DEFECT_ACTIONS:
                for dj in DEFECT_ACTIONS:
                    cr, cc = matrix[ci][cj][0], matrix[ci][cj][1]
                    dr, dc = matrix[di][dj][0], matrix[di][dj][1]
                    if not (cr > dr and cc > dc):
                        violations.append(
                            f"({STRATEGY_NAMES[ci]},{STRATEGY_NAMES[cj]})=({cr:.2f},{cc:.2f}) "
                            f"does not Pareto-dominate "
                            f"({STRATEGY_NAMES[di]},{STRATEGY_NAMES[dj]})=({dr:.2f},{dc:.2f})"
                        )
    return violations


def check_axiom_iv_s(matrix):
    """
    Strict dominant defection (PD): every defection action strictly dominates
    every cooperation action for each player at every opponent action.
    """
    violations = []
    for coop_idx in COOP_ACTIONS:
        for defect_idx in DEFECT_ACTIONS:
            for opp_idx in range(4):
                # Row player
                if matrix[defect_idx][opp_idx][0] <= matrix[coop_idx][opp_idx][0]:
                    violations.append(
                        f"[iv_s-row] {STRATEGY_NAMES[defect_idx]} vs {STRATEGY_NAMES[coop_idx]} "
                        f"against {STRATEGY_NAMES[opp_idx]}: "
                        f"{matrix[defect_idx][opp_idx][0]:.4f} <= {matrix[coop_idx][opp_idx][0]:.4f}"
                    )
                # Column player
                if matrix[opp_idx][defect_idx][1] <= matrix[opp_idx][coop_idx][1]:
                    violations.append(
                        f"[iv_s-col] {STRATEGY_NAMES[defect_idx]} vs {STRATEGY_NAMES[coop_idx]} "
                        f"against {STRATEGY_NAMES[opp_idx]}: "
                        f"{matrix[opp_idx][defect_idx][1]:.4f} <= {matrix[opp_idx][coop_idx][1]:.4f}"
                    )
    return violations


def check_axiom_iv_p(matrix):
    """
    Partial defection incentive (SH): for every player, there exists at least
    one (opponent, coop, defect) where defection beats cooperation.
    """
    violations = []
    for player in ["row", "col"]:
        found = False
        for coop_idx in COOP_ACTIONS:
            for defect_idx in DEFECT_ACTIONS:
                for opp_idx in range(4):
                    if player == "row":
                        if matrix[defect_idx][opp_idx][0] > matrix[coop_idx][opp_idx][0]:
                            found = True
                            break
                    else:
                        if matrix[opp_idx][defect_idx][1] > matrix[opp_idx][coop_idx][1]:
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            violations.append(f"[iv_p] No defection incentive for {player} player")
    return violations


# EQUILIBRIA/OPTIMA

def find_pure_nash(matrix):
    """Find all pure strategy Nash equilibria."""
    n = 4
    nash = []
    for i in range(n):
        for j in range(n):
            row_br = all(matrix[i][j][0] >= matrix[k][j][0] for k in range(n))
            col_br = all(matrix[i][j][1] >= matrix[i][l][1] for l in range(n))
            if row_br and col_br:
                nash.append((i, j))
    return nash


def find_utilitarian_optimum(matrix):
    """Find all cells maximising total welfare."""
    best_w = max(matrix[i][j][0] + matrix[i][j][1] for i in range(4) for j in range(4))
    return [
        (i, j) for i in range(4) for j in range(4)
        if matrix[i][j][0] + matrix[i][j][1] == best_w
    ], best_w


def find_rawlsian_optimum(matrix):
    """Find all cells maximising the minimum individual payoff."""
    best_min = max(min(matrix[i][j][0], matrix[i][j][1]) for i in range(4) for j in range(4))
    return [
        (i, j) for i in range(4) for j in range(4)
        if min(matrix[i][j][0], matrix[i][j][1]) == best_min
    ], best_min


def check_pd_equilibrium_theory(matrix, R, S, T, P):
    """
    Verify Theorem 2 (PD equilibrium structure):
    - (DH, DH) is unique pure NE
    - (CH, CH) is unique utilitarian and Rawlsian max
    - Efficiency gap = 2(R - P)
    """
    issues = []
    labels = STRATEGY_NAMES

    nash = find_pure_nash(matrix)
    util_opt, util_w = find_utilitarian_optimum(matrix)
    rawl_opt, rawl_w = find_rawlsian_optimum(matrix)

    # Nash
    if nash != [(DH, DH)]:
        issues.append(
            f"Nash: expected [(DH,DH)], got {[(labels[i],labels[j]) for i,j in nash]}"
        )

    # Utilitarian
    if util_opt != [(CH, CH)]:
        issues.append(
            f"Utilitarian: expected [(CH,CH)], got {[(labels[i],labels[j]) for i,j in util_opt]}"
        )

    # Rawlsian
    if rawl_opt != [(CH, CH)]:
        issues.append(
            f"Rawlsian: expected [(CH,CH)], got {[(labels[i],labels[j]) for i,j in rawl_opt]}"
        )

    # Efficiency gap
    w_star = matrix[CH][CH][0] + matrix[CH][CH][1]
    w_si = matrix[DH][DH][0] + matrix[DH][DH][1]
    expected_gap = 2 * (R - P)
    actual_gap = w_star - w_si
    if abs(actual_gap - expected_gap) > 1e-9:
        issues.append(
            f"Efficiency gap: expected 2(R-P)={expected_gap:.4f}, got {actual_gap:.4f}"
        )

    return issues


def check_sh_equilibrium_theory(matrix, R, S, T, P):
    """
    Verify Theorem 4 (SH equilibrium structure):
    - (CH, CH) is unique utilitarian and Rawlsian max
    - (DH, DH) is pure NE
    - (CH, CL) and (CL, CH) are NE when 4T < 3R + P
    - Efficiency gap = 2(R - P)
    """
    issues = []
    labels = STRATEGY_NAMES

    nash = find_pure_nash(matrix)
    util_opt, util_w = find_utilitarian_optimum(matrix)
    rawl_opt, rawl_w = find_rawlsian_optimum(matrix)

    # Utilitarian
    if util_opt != [(CH, CH)]:
        issues.append(
            f"Utilitarian: expected [(CH,CH)], got {[(labels[i],labels[j]) for i,j in util_opt]}"
        )

    # Rawlsian
    if rawl_opt != [(CH, CH)]:
        issues.append(
            f"Rawlsian: expected [(CH,CH)], got {[(labels[i],labels[j]) for i,j in rawl_opt]}"
        )

    # (DH, DH) is NE
    if (DH, DH) not in nash:
        issues.append(f"(DH,DH) is not Nash; Nash = {[(labels[i],labels[j]) for i,j in nash]}")

    # S5 check: must be in multi-NE regime
    if not (4 * T < 3 * R + P):
        issues.append(
            f"S5 violated: 4T={4*T} >= 3R+P={3*R+P} (not in multi-NE regime)"
        )

    # Multi-NE regime: must have exactly {(DH,DH), (CH,CL), (CL,CH)}
    expected_ne = {(DH, DH), (CH, CL), (CL, CH)}
    actual_ne = set(nash)
    if actual_ne != expected_ne:
        missing = expected_ne - actual_ne
        extra = actual_ne - expected_ne
        parts = []
        if missing:
            parts.append(f"missing {[(labels[i],labels[j]) for i,j in missing]}")
        if extra:
            parts.append(f"unexpected {[(labels[i],labels[j]) for i,j in extra]}")
        issues.append(
            f"NE set mismatch: expected {[(labels[i],labels[j]) for i,j in expected_ne]}, "
            f"got {[(labels[i],labels[j]) for i,j in nash]} ({'; '.join(parts)})"
        )

    # Efficiency gap
    w_star = matrix[CH][CH][0] + matrix[CH][CH][1]
    w_si = matrix[DH][DH][0] + matrix[DH][DH][1]
    expected_gap = 2 * (R - P)
    actual_gap = w_star - w_si
    if abs(actual_gap - expected_gap) > 1e-9:
        issues.append(
            f"Efficiency gap: expected 2(R-P)={expected_gap:.4f}, got {actual_gap:.4f}"
        )

    return issues


# MATRIX RECONSTRUCTION VERIFICATION

def verify_matrix_reconstruction(R, S, T, P, stored_matrix):
    """Verify that the stored matrix matches the parametric construction.

    The stored matrix is GCD-normalized (clearing denominator 12 then dividing
    by GCD of all entries). We check whether stored = theory * scale for some
    positive constant scale, which is the correct relation after normalization.
    """
    theory_matrix = build_theory_matrix(R, S, T, P)

    # Find a non-zero cell to determine the scale factor
    scale = None
    for i in range(4):
        for j in range(4):
            tr, tc = theory_matrix[i][j]
            sr, sc = stored_matrix[i][j]
            if abs(tr) > 1e-9 and abs(tc) > 1e-9:
                scale_r = sr / tr
                scale_c = sc / tc
                if abs(scale_r - scale_c) < 1e-9:
                    scale = scale_r
                    break
        if scale is not None:
            break

    if scale is None:
        return ["Cannot determine scale factor (all theory cells zero?)"]

    # Verify every cell: stored = theory * scale (within rounding)
    issues = []
    for i in range(4):
        for j in range(4):
            tr, tc = theory_matrix[i][j]
            sr, sc = stored_matrix[i][j]
            expected_r = round(tr * scale)
            expected_c = round(tc * scale)
            if abs(sr - expected_r) > 0 or abs(sc - expected_c) > 0:
                issues.append(
                    f"[{i},{j}]: theory*{scale:.1f}=({expected_r},{expected_c}), "
                    f"stored=({sr},{sc})"
                )

    return issues


# MAIN VERIFICATION

def verify_csv(csv_path: str) -> bool:
    """Verify all rows in a CSV file. Returns True if all pass."""
    df = pd.read_csv(csv_path)
    filename = Path(csv_path).name
    game_type = df["formal_game"].iloc[0]
    n_rows = len(df)

    print(f"\n{'='*72}")
    print(f"Verifying: {filename}")
    print(f"Game type: {game_type}")
    print(f"Rows: {n_rows}")
    print(f"{'='*72}")

    is_pd = game_type == "Prisoner's Dilemma"
    is_sh = game_type in ("Stag Hunt", "Stag hunt")

    all_pass = True
    summary = {
        "construction": 0,
        "structural": 0,
        "axiom_i": 0,
        "axiom_ii": 0,
        "axiom_iii": 0,
        "axiom_iv": 0,
        "equilibrium": 0,
    }

    for idx, (_, row) in enumerate(df.iterrows()):
        # Parse base 2x2
        payoff_11 = eval(row["1_1_payoff"])
        payoff_12 = eval(row["1_2_payoff"])
        payoff_21 = eval(row["2_1_payoff"])
        payoff_22 = eval(row["2_2_payoff"])
        R, S, T, P = payoff_11[0], payoff_12[0], payoff_21[0], payoff_22[0]

        # Parse stored 4x4
        stored_matrix = json.loads(row["payoff_matrix_4x4"])
        nash_target = row["target_nash_4x4"]
        util_target = row["target_utilitarian_4x4"]

        # Build theory matrix
        theory_matrix = build_theory_matrix(R, S, T, P)

        row_issues = []

        # Matrix reconstruction
        recon_issues = verify_matrix_reconstruction(R, S, T, P, stored_matrix)
        if not recon_issues:
            summary["construction"] += 1
        else:
            row_issues.append(f"Construction mismatch: {recon_issues[:3]}")
            all_pass = False

        # Structural conditions
        if is_pd:
            cond_results = check_pd_conditions(R, S, T, P)
            all_conds_pass = all(c["pass"] for c in cond_results.values())
        else:
            cond_results = check_sh_conditions(R, S, T, P)
            # For SH, note whether the data uses the script's ordering vs theory's
            all_conds_pass = all(c["pass"] for c in cond_results.values())
        if all_conds_pass:
            summary["structural"] += 1
        else:
            failed = {k: v for k, v in cond_results.items() if not v["pass"]}
            row_issues.append(f"Structural conditions failed: {failed}")
            all_pass = False

        # Axiom (i): Welfare improvement
        ax_i = check_axiom_i(theory_matrix)
        if not ax_i:
            summary["axiom_i"] += 1
        else:
            if is_pd:
                row_issues.append(f"Axiom (i) failed: {ax_i[0]} ({len(ax_i)} violations)")
                all_pass = False

        # Axiom (ii): Unique welfare max at (CH,CH)
        ax_ii = check_axiom_ii(theory_matrix)
        if not ax_ii:
            summary["axiom_ii"] += 1
        else:
            row_issues.append(f"Axiom (ii) failed: {ax_ii}")
            all_pass = False

        # Axiom (iii): Pareto dominance
        ax_iii = check_axiom_iii_pareto(theory_matrix)
        if not ax_iii:
            summary["axiom_iii"] += 1
        else:
            row_issues.append(f"Axiom (iii) failed: {ax_iii[0]} ({len(ax_iii)} violations)")
            all_pass = False

        # Axiom (iv)
        if is_pd:
            ax_iv = check_axiom_iv_s(theory_matrix)
        else:
            ax_iv = check_axiom_iv_p(theory_matrix)
        if not ax_iv:
            summary["axiom_iv"] += 1
        else:
            row_issues.append(f"Axiom (iv) failed: {ax_iv}")
            all_pass = False

        # Equilibrium structure
        if is_pd:
            eq_issues = check_pd_equilibrium_theory(theory_matrix, R, S, T, P)
        else:
            eq_issues = check_sh_equilibrium_theory(theory_matrix, R, S, T, P)
        if not eq_issues:
            summary["equilibrium"] += 1
        else:
            row_issues.append(f"Equilibrium: {eq_issues}")
            all_pass = False

        # Report per-row
        status = "PASS" if not row_issues else "ISSUE"
        if row_issues:
            print(f"  Row {idx}: {status}")
            for issue in row_issues:
                print(f"    - {issue}")

    # Summary
    print(f"\n{'─'*72}")
    print(f"Summary for {filename}:")
    print(f"  Matrix construction matches theory: {summary['construction']}/{n_rows}")
    print(f"  Structural conditions (Thm 1/3):    {summary['structural']}/{n_rows}")
    print(f"  Axiom (i)  welfare improvement:     {summary['axiom_i']}/{n_rows}")
    print(f"  Axiom (ii) unique welfare max:      {summary['axiom_ii']}/{n_rows}")
    print(f"  Axiom (iii) Pareto dominance:       {summary['axiom_iii']}/{n_rows}")
    print(f"  Axiom (iv)  defection incentive:    {summary['axiom_iv']}/{n_rows}")
    print(f"  Equilibrium structure (Thm 2/4):    {summary['equilibrium']}/{n_rows}")
    print(f"{'─'*72}")

    return all_pass


@click.command()
@click.argument("csv_files", nargs=-1, required=True)
def main(csv_files):
    """Verify 4x4 matrices against theory axioms and conditions."""
    all_pass = True
    for csv_path in csv_files:
        if not verify_csv(csv_path):
            all_pass = False

    if all_pass:
        print("\nAll checks passed.")
    else:
        print("\nSome checks FAILED. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
