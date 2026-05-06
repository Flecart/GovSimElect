#!/usr/bin/env python3
"""
Compare old and new Stag Hunt 4x4 datasets.

OLD: data/gt-harmbench-sh30-4x4.csv  (R > P > T > S ordering)
NEW: data/gt-harmbench-sh-fixed-30-4x4.csv  (R > T > P > S ordering)

Reports differences in:
  - Base 2x2 payoff distributions
  - Structural condition satisfaction
  - Axiom compliance (i, ii, iii, iv)
  - Nash equilibrium structure
  - Welfare optima
  - Efficiency gap

Usage:
    uv run python3 scripts/verification/compare_sh_datasets.py
    uv run python3 scripts/verification/compare_sh_datasets.py \
        --old data/gt-harmbench-sh30-4x4.csv \
        --new data/gt-harmbench-sh-fixed-30-4x4.csv
"""

import json
from collections import Counter
from pathlib import Path

import click
import pandas as pd

CH, CL, DH, DL = 0, 1, 2, 3
STRAT = ["CH", "CL", "DH", "DL"]


def parse_payoffs(df):
    """Extract (R, S, T, P) lists from a DataFrame."""
    R, S, T, P = [], [], [], []
    for _, r in df.iterrows():
        p11 = eval(r["1_1_payoff"])
        p12 = eval(r["1_2_payoff"])
        p21 = eval(r["2_1_payoff"])
        p22 = eval(r["2_2_payoff"])
        R.append(p11[0]); S.append(p12[0]); T.append(p21[0]); P.append(p22[0])
    return R, S, T, P


def compute_nash(matrix):
    n = 4
    nash = []
    for i in range(n):
        for j in range(n):
            row_br = all(matrix[i][j][0] >= matrix[k][j][0] for k in range(n))
            col_br = all(matrix[i][j][1] >= matrix[i][l][1] for l in range(n))
            if row_br and col_br:
                nash.append((i, j))
    return nash


def check_axioms(matrix):
    """Check all 4 axioms. Returns dict of pass counts and details."""
    n = 4
    coop = [0, 1]
    defect = [2, 3]

    def w(i, j):
        return matrix[i][j][0] + matrix[i][j][1]

    # (i) Welfare improvement
    ax_i_violations = 0
    for ci in coop:
        for di in defect:
            for oi in range(n):
                if w(ci, oi) <= w(di, oi):
                    ax_i_violations += 1
                    break
                if w(oi, ci) <= w(oi, di):
                    ax_i_violations += 1
                    break

    # (ii) Unique welfare max at (CH, CH)
    ch_ch_w = w(CH, CH)
    ax_ii_pass = all(w(i, j) < ch_ch_w for i in range(n) for j in range(n) if (i, j) != (CH, CH))

    # (iii) Pareto dominance
    ax_iii_pass = True
    for ci in coop:
        for cj in coop:
            for di in defect:
                for dj in defect:
                    if not (matrix[ci][cj][0] > matrix[di][dj][0] and matrix[ci][cj][1] > matrix[di][dj][1]):
                        ax_iii_pass = False

    # (iv_p) Partial defection incentive
    ax_iv_pass = True
    for player in ["row", "col"]:
        found = False
        for ci in coop:
            for di in defect:
                for oi in range(n):
                    if player == "row":
                        if matrix[di][oi][0] > matrix[ci][oi][0]:
                            found = True
                    else:
                        if matrix[oi][di][1] > matrix[oi][ci][1]:
                            found = True
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            ax_iv_pass = False

    return {
        "axiom_i_pass": ax_i_violations == 0,
        "axiom_i_violations": ax_i_violations,
        "axiom_ii_pass": ax_ii_pass,
        "axiom_iii_pass": ax_iii_pass,
        "axiom_iv_pass": ax_iv_pass,
    }


def analyze_dataset(df, label):
    """Full analysis of one dataset."""
    n = len(df)
    R, S, T, P = parse_payoffs(df)

    # Structural conditions
    s1 = sum(1 for r, s, t, p in zip(R, S, T, P) if r > t > p > s)
    s2 = sum(1 for r, s, t, p in zip(R, S, T, P) if 2 * r > s + t)
    s3 = sum(1 for r, s, t, p in zip(R, S, T, P) if 2 * (s + t) > r + 3 * p)
    s4 = sum(1 for r, s, t, p in zip(R, S, T, P) if 3 * (s + t) < 5 * r + p)

    # Axioms
    axiom_results = [check_axioms(json.loads(r["payoff_matrix_4x4"])) for _, r in df.iterrows()]
    ax_i_pass = sum(1 for a in axiom_results if a["axiom_i_pass"])
    ax_ii_pass = sum(1 for a in axiom_results if a["axiom_ii_pass"])
    ax_iii_pass = sum(1 for a in axiom_results if a["axiom_iii_pass"])
    ax_iv_pass = sum(1 for a in axiom_results if a["axiom_iv_pass"])

    # Nash equilibria
    nash_sets = []
    for _, r in df.iterrows():
        matrix = json.loads(r["payoff_matrix_4x4"])
        ne = compute_nash(matrix)
        nash_sets.append(ne)

    nash_count_dist = Counter(len(ne) for ne in nash_sets)
    has_dh_dh = sum(1 for ne in nash_sets if (DH, DH) in ne)
    has_asym = sum(1 for ne in nash_sets if (CH, CL) in ne or (CL, CH) in ne)

    # Welfare
    util_w = []
    rawl_w = []
    eff_gap = []
    for _, r in df.iterrows():
        matrix = json.loads(r["payoff_matrix_4x4"])
        w_star = matrix[CH][CH][0] + matrix[CH][CH][1]
        w_si = matrix[DH][DH][0] + matrix[DH][DH][1]
        util_w.append(w_star)
        rawl_w.append(min(matrix[CH][CH][0], matrix[CH][CH][1]))
        eff_gap.append(w_star - w_si)

    return {
        "label": label,
        "n": n,
        "payoffs": {
            "R": f"{min(R)}-{max(R)} (mean={sum(R)/n:.1f})",
            "S": f"{min(S)}-{max(S)} (mean={sum(S)/n:.1f})",
            "T": f"{min(T)}-{max(T)} (mean={sum(T)/n:.1f})",
            "P": f"{min(P)}-{max(P)} (mean={sum(P)/n:.1f})",
            "T-P gap": f"{min(t-p for t,p in zip(T,P))}-{max(t-p for t,p in zip(T,P))} (mean={sum(t-p for t,p in zip(T,P))/n:.1f})",
        },
        "conditions": {
            "S1 (R>T>P>S)": f"{s1}/{n}",
            "S2 (2R>S+T)": f"{s2}/{n}",
            "S3 (2(S+T)>R+3P)": f"{s3}/{n}",
            "S4 (3(S+T)<5R+P)": f"{s4}/{n}",
        },
        "axioms": {
            "(i)  welfare improvement": f"{ax_i_pass}/{n}",
            "(ii) unique welfare max": f"{ax_ii_pass}/{n}",
            "(iii) Pareto dominance": f"{ax_iii_pass}/{n}",
            "(iv)  defection incentive": f"{ax_iv_pass}/{n}",
        },
        "nash": {
            "count distribution": dict(nash_count_dist),
            "(DH,DH) present": f"{has_dh_dh}/{n}",
            "asymmetric NE present": f"{has_asym}/{n}",
        },
        "welfare": {
            "utilitarian W*": f"{min(util_w)}-{max(util_w)} (mean={sum(util_w)/n:.1f})",
            "Rawlsian min": f"{min(rawl_w)}-{max(rawl_w)} (mean={sum(rawl_w)/n:.1f})",
            "efficiency gap": f"{min(eff_gap)}-{max(eff_gap)} (mean={sum(eff_gap)/n:.1f})",
        },
    }


def print_comparison(old_result, new_result):
    """Print side-by-side comparison."""
    w = 35
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"{'STAG HUNT DATASET COMPARISON':^80}")
    print(f"{sep}\n")

    # Header
    print(f"{'':>{w}}  {'OLD (R>P>T>S)':^20}  {'NEW (R>T>P>S)':^20}")
    print(f"{'':>{w}}  {'-'*20}  {'-'*20}")

    # Payoffs
    print(f"\n{'--- Base 2x2 Payoffs ---':^80}")
    for key in ["R", "S", "T", "P", "T-P gap"]:
        old_v = old_result["payoffs"][key]
        new_v = new_result["payoffs"][key]
        print(f"  {key:>{w-2}}  {old_v:^20}  {new_v:^20}")

    # Structural conditions
    print(f"\n{'--- Structural Conditions (Thm 3) ---':^80}")
    for key in old_result["conditions"]:
        old_v = old_result["conditions"][key]
        new_v = new_result["conditions"][key]
        marker = ""
        if old_v != new_v:
            marker = "  <--"
        print(f"  {key:>{w-2}}  {old_v:^20}  {new_v:^20}{marker}")

    # Axioms
    print(f"\n{'--- Axiom Compliance ---':^80}")
    for key in old_result["axioms"]:
        old_v = old_result["axioms"][key]
        new_v = new_result["axioms"][key]
        marker = ""
        if old_v != new_v:
            marker = "  <--"
        print(f"  {key:>{w-2}}  {old_v:^20}  {new_v:^20}{marker}")

    # Nash
    print(f"\n{'--- Nash Equilibria ---':^80}")
    for key in old_result["nash"]:
        old_v = str(old_result["nash"][key])
        new_v = str(new_result["nash"][key])
        marker = ""
        if old_v != new_v:
            marker = "  <--"
        print(f"  {key:>{w-2}}  {old_v:^20}  {new_v:^20}{marker}")

    # Welfare
    print(f"\n{'--- Welfare (4x4) ---':^80}")
    for key in old_result["welfare"]:
        old_v = old_result["welfare"][key]
        new_v = new_result["welfare"][key]
        print(f"  {key:>{w-2}}  {old_v:^20}  {new_v:^20}")

    print(f"\n{sep}")

    # Summary of key differences
    print("\nKey differences:")
    print(f"  1. Ordering: OLD has P>T (T-P gap negative), NEW has T>P (gap positive)")
    if old_result["conditions"]["S3 (2(S+T)>R+3P)"] != new_result["conditions"]["S3 (2(S+T)>R+3P)"]:
        print(f"  2. S3 condition: OLD {old_result['conditions']['S3 (2(S+T)>R+3P)']}, "
              f"NEW {new_result['conditions']['S3 (2(S+T)>R+3P)']}")
    if old_result["axioms"]["(i)  welfare improvement"] != new_result["axioms"]["(i)  welfare improvement"]:
        print(f"  3. Axiom (i): OLD {old_result['axioms']['(i)  welfare improvement']}, "
              f"NEW {new_result['axioms']['(i)  welfare improvement']}")
    if old_result["nash"]["asymmetric NE present"] != new_result["nash"]["asymmetric NE present"]:
        print(f"  4. Asymmetric NE: OLD {old_result['nash']['asymmetric NE present']}, "
              f"NEW {new_result['nash']['asymmetric NE present']}")


@click.command()
@click.option("--old", "old_path", default="data/gt-harmbench-sh30-4x4.csv")
@click.option("--new", "new_path", default="data/gt-harmbench-sh-fixed-30-4x4.csv")
def main(old_path, new_path):
    """Compare old and new Stag Hunt 4x4 datasets."""
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)

    old_result = analyze_dataset(old_df, "OLD")
    new_result = analyze_dataset(new_df, "NEW")

    print_comparison(old_result, new_result)


if __name__ == "__main__":
    main()
