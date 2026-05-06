#!/usr/bin/env python3
"""
Generate Stag Hunt games with integer payoffs satisfying the theory's conditions.

The theory (Theorem 3) requires four conditions for valid 4x4 SH lifts:
  (S1) R > T > P > S
  (S2) 2R > S + T
  (S3) 2(S + T) > R + 3P     — welfare lower bound
  (S4) 3(S + T) < 5R + P     — welfare upper bound
  (S5) 4T < 3R + P           — multi-NE regime: ensures 3 pure NE

A simple T/P swap of the original data does NOT satisfy S3 (the original
payoff ranges make T too small relative to R and P). Instead, this script
enumerates all valid integer (R, S, T, P) tuples within a bounded range,
randomly samples N, and borrows narratives from the existing gt-harmbench
SH rows.

The output CSV preserves the gt-harmbench column schema so it can feed
directly into the 4x4 generation pipeline. Narratives are borrowed as-is;
a separate rewrite step handles narrative-payoff alignment.

Usage:
    uv run python3 scripts/generation/fix_stag_hunt_payoffs.py \
        --output data/gt-harmbench-sh-fixed.csv --sample 100

    # Test mode:
    uv run python3 scripts/generation/fix_stag_hunt_payoffs.py \
        --output data/gt-harmbench-sh-fixed.csv --sample 10

    # Custom payoff range:
    uv run python3 scripts/generation/fix_stag_hunt_payoffs.py \
        --output data/gt-harmbench-sh-fixed.csv --sample 100 --max-payoff 15
"""

import copy
import random
from pathlib import Path

import click
import pandas as pd


# ---------------------------------------------------------------------------
# Structural conditions (from theory Theorem 3)
# ---------------------------------------------------------------------------

def satisfies_sh_conditions(R: int, S: int, T: int, P: int) -> bool:
    """Check all five SH conditions (Theorem 3 + multi-NE regime)."""
    if not (R > T > P > S):                    # S1
        return False
    if not (2 * R > S + T):                    # S2
        return False
    if not (2 * (S + T) > R + 3 * P):          # S3
        return False
    if not (3 * (S + T) < 5 * R + P):          # S4
        return False
    if not (4 * T < 3 * R + P):                # S5 (multi-NE regime)
        return False
    return True


def enumerate_valid_games(max_payoff: int = 15) -> list[tuple[int, int, int, int]]:
    """Enumerate all valid integer (R, S, T, P) tuples satisfying S1-S4."""
    valid = []
    for R in range(3, max_payoff + 1):
        for S in range(0, R):
            for T in range(S + 1, R):      # S1: R > T, T > P > S → T > S+1 min
                for P in range(S, T):      # S1: T > P > S
                    if satisfies_sh_conditions(R, S, T, P):
                        valid.append((R, S, T, P))
    return valid


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def build_row(template_row: dict, R: int, S: int, T: int, P: int) -> dict:
    """Build a CSV row from a template narrative row and new payoffs."""
    new_row = copy.deepcopy(template_row)
    new_row["1_1_payoff"] = str((R, R))
    new_row["1_2_payoff"] = str((S, T))
    new_row["2_1_payoff"] = str((T, S))
    new_row["2_2_payoff"] = str((P, P))
    new_row["formal_game"] = "Stag hunt"
    return new_row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--output", "-o", required=True, help="Output CSV path")
@click.option("--sample", "-n", default=100, help="Number of games to sample (default: 100)")
@click.option("--base-dataset", default="data/gt-harmbench.csv",
              help="Source dataset for narrative templates")
@click.option("--max-payoff", default=15, help="Maximum payoff value to enumerate (default: 15)")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def main(output, sample, base_dataset, max_payoff, seed):
    """Generate N valid Stag Hunt games with integer payoffs (S1-S4)."""
    random.seed(seed)

    # Enumerate valid games
    print(f"Enumerating valid integer SH games (max_payoff={max_payoff})...")
    valid_games = enumerate_valid_games(max_payoff)
    print(f"Found {len(valid_games)} valid (R,S,T,P) tuples")

    if len(valid_games) < sample:
        raise click.ClickException(
            f"Only {len(valid_games)} valid games found, need {sample}. "
            f"Try increasing --max-payoff."
        )

    # Sample
    sampled = random.sample(valid_games, sample)
    print(f"Sampled {sample} games")

    # Print distribution summary
    r_values = [g[0] for g in sampled]
    s_values = [g[1] for g in sampled]
    t_values = [g[2] for g in sampled]
    p_values = [g[3] for g in sampled]
    print(f"  R range: {min(r_values)}-{max(r_values)}")
    print(f"  S range: {min(s_values)}-{max(s_values)}")
    print(f"  T range: {min(t_values)}-{max(t_values)}")
    print(f"  P range: {min(p_values)}-{max(p_values)}")
    print(f"  T-P gap range: {min(t-p for t,p in zip(t_values, p_values))}-"
          f"{max(t-p for t,p in zip(t_values, p_values))}")

    # Load narratives from existing SH rows
    print(f"Loading narrative templates from {base_dataset}...")
    df = pd.read_csv(base_dataset)
    sh_rows = df[df["formal_game"].isin(["Stag Hunt", "Stag hunt"])].to_dict("records")
    print(f"Found {len(sh_rows)} SH narrative templates")

    if len(sh_rows) == 0:
        raise click.ClickException("No Stag Hunt rows found in base dataset for narratives.")

    # Build output rows: cycle through narratives
    results = []
    for i, (R, S, T, P) in enumerate(sampled):
        template = sh_rows[i % len(sh_rows)]
        row = build_row(template, R, S, T, P)

        # Assign a unique numeric id (avoids int() failures in eval infrastructure)
        row["id"] = str(3000 + i)

        results.append(row)

    # Save
    out_df = pd.DataFrame(results)
    out_path = Path(output)
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved {len(results)} rows to {out_path}")

    # Quick validation
    print("\nValidation:")
    n_pass = 0
    n_fail = 0
    for _, r in out_df.iterrows():
        p11 = eval(r["1_1_payoff"])
        p12 = eval(r["1_2_payoff"])
        p21 = eval(r["2_1_payoff"])
        p22 = eval(r["2_2_payoff"])
        R, S, T, P = p11[0], p12[0], p21[0], p22[0]
        if satisfies_sh_conditions(R, S, T, P):
            n_pass += 1
        else:
            n_fail += 1
            if n_fail <= 3:
                print(f"  FAIL: R={R} S={S} T={T} P={P}")
    print(f"  Pass: {n_pass}/{len(results)}, Fail: {n_fail}/{len(results)}")


if __name__ == "__main__":
    main()
