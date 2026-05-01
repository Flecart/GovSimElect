#!/usr/bin/env python3
"""Move matching result directories from simulation/results to .venv/results."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_matches(source_root: Path, scopes: list[str], match_mode: str) -> list[Path]:
    """Return directories matching `code-law` and/or `nl` in their name."""
    matches: list[Path] = []
    for scope in scopes:
        scope_dir = source_root / scope
        if not scope_dir.is_dir():
            continue
        for path in scope_dir.rglob("*"):
            if not path.is_dir():
                continue
            name = path.name.lower()
            has_code_law = "code-law" in name
            has_nl = "nl" in name
            is_match = (
                has_code_law and has_nl
                if match_mode == "both"
                else has_code_law or has_nl
            )
            if is_match:
                matches.append(path)
    # Move deepest directories first to avoid moving parent before child.
    return sorted(matches, key=lambda p: len(p.parts), reverse=True)


def move_dirs(
    source_root: Path,
    dest_root: Path,
    directories: list[Path],
    dry_run: bool,
    replace_existing: bool,
) -> tuple[int, int]:
    moved = 0
    skipped = 0
    for src in directories:
        rel = src.relative_to(source_root)
        dest = dest_root / rel
        if dest.exists():
            if not replace_existing:
                skipped += 1
                print(f"SKIP (already exists): {dest}")
                continue
            print(f"REPLACE existing destination: {dest}")
            if not dry_run:
                shutil.rmtree(dest)
        print(f"MOVE: {src} -> {dest}")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            if src.exists():
                shutil.rmtree(src)

            # Clean up any now-empty parent folders up to source_root.
            parent = src.parent
            while parent != source_root and parent.exists():
                try:
                    parent.rmdir()
                    parent = parent.parent
                except OSError:
                    break
        moved += 1
    return moved, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move directories under simulation/results/std and simulation/results/sto "
            "whose names contain 'code-law' and/or 'nl' into .venv/results "
            "while preserving their original relative paths."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("./simulation/results"),
        help="Root directory containing std/ and sto/ (default: simulation/results).",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path(".venv/results"),
        help="Destination root directory (default: .venv/results).",
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["std", "sto"],
        help="Subdirectories under source-root to scan (default: std sto).",
    )
    parser.add_argument(
        "--match-mode",
        choices=["either", "both"],
        default="either",
        help=(
            "How to match directory names: 'either' matches names containing "
            "'code-law' OR 'nl' (default), 'both' requires both in the same name."
        ),
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "If destination already exists, delete destination and replace it with source "
            "instead of skipping."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform moves. Without this flag, only print planned moves.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()
    dry_run = not args.apply

    if not source_root.is_dir():
        raise SystemExit(f"Source root does not exist or is not a directory: {source_root}")

    matches = find_matches(
        source_root=source_root, scopes=args.scopes, match_mode=args.match_mode
    )
    if not matches:
        print("No matching directories found.")
        return 0

    moved, skipped = move_dirs(
        source_root=source_root,
        dest_root=dest_root,
        directories=matches,
        dry_run=dry_run,
        replace_existing=args.replace_existing,
    )
    mode = "DRY RUN" if dry_run else "APPLY"
    print(f"{mode} complete. matched={len(matches)} moved={moved} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
