#!/usr/bin/env python3
"""Sample N random rows from a CSV file, preserving all fields correctly."""

import argparse
import csv
import random
from pathlib import Path


def sample_csv(input_path: Path, output_path: Path, n: int, seed: int = None):
    """Sample N random rows from a CSV file.

    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        n: Number of rows to sample
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Read all rows
    with open(input_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Sample n rows (or all if n > len(rows))
    sample_size = min(n, len(rows))
    sampled = random.sample(rows, sample_size)

    # Write to output
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    print(f"Sampled {sample_size} rows from {len(rows)} total")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample N random rows from a CSV")
    parser.add_argument("input_csv", type=Path, help="Input CSV path")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    parser.add_argument("n", type=int, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if not args.input_csv.exists():
        print(f"Error: {args.input_csv} does not exist")
        exit(1)

    sample_csv(args.input_csv, args.output_csv, args.n, args.seed)
