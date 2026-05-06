#!/usr/bin/env python3
"""Check for API errors in .eval files.

Did you run out of credits when you left your experiment running overnight? This script is for you.

Searches ALL content for OpenAI/OpenRouter error messages (rate limits, quota, etc.)
May produce false positives.
"""

import argparse
import subprocess
import re
import random
from pathlib import Path
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)


# API error patterns
ERROR_PATTERNS = [
    r'insufficient_quota',
    r'insufficient quota',
    r'rate.*limit.*exceed',
    r'quota.*exceed',
    r'"code": 402',
    r"'code': 402",
    r'error code: 402',
    r'status code 402',
    r'http 402',
    r'payment required',
    r'insufficient credits',
    r'insufficient credit',
]

ERROR_REGEX = re.compile('|'.join(ERROR_PATTERNS), re.IGNORECASE)


def check_eval_file_for_errors(eval_path: Path) -> dict:
    """Check .eval file for API errors using unzip+grep."""
    result = {
        'has_errors': False,
        'total_samples': 0,
        'affected_samples': 0,
        'error_types': defaultdict(int),
        'examples': [],
    }

    try:
        # Use unzip to list all files in the archive
        listing = subprocess.run(
            ['unzip', '-Z1', str(eval_path)],
            capture_output=True,
            text=True,
            timeout=10
        )

        if listing.returncode != 0:
            return result

        # Get ALL files, not just samples
        all_files = [f for f in listing.stdout.split('\n') if f.strip()]

        # Count sample files for total
        sample_files = [f for f in all_files if f.startswith('samples/')]
        result['total_samples'] = len(sample_files)

        # Search ALL files for errors using unzip | grep
        grep_result = subprocess.run(
            ['unzip', '-p', str(eval_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if grep_result.returncode == 0:
            content = grep_result.stdout
            content_lower = content.lower()

            # Check for each error pattern
            for pattern in ERROR_PATTERNS:
                if re.search(pattern, content_lower):
                    result['has_errors'] = True
                    result['error_types'][pattern] += content_lower.lower().count(pattern.lower())

                    # Extract examples
                    if len(result['examples']) < 25:
                        for match in re.finditer(pattern, content, re.IGNORECASE):
                            start = max(0, match.start() - 40)
                            end = min(len(content), match.end() + 80)
                            context = content[start:end].replace('\n', ' ')[:120]
                            result['examples'].append({
                                'pattern': pattern,
                                'context': context
                            })
                            if len(result['examples']) >= 25:
                                break

            # If errors found, estimate affected samples by counting error occurrences
            # This is a heuristic - each error occurrence likely means one affected sample
            if result['has_errors']:
                total_errors = sum(result['error_types'].values())
                result['affected_samples'] = min(total_errors, result['total_samples'])

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception):
        pass

    return result


def extract_metadata(eval_path: Path) -> dict:
    """Extract metadata from path."""
    metadata = {
        'prompt_mode': 'unknown',
        'contract_mode': 'unknown',
        'dataset_size': 'unknown',
        'model': 'unknown',
        'game': 'unknown',
    }
    
    for part in eval_path.parts:
        if part in ['base', 'selfish', 'cooperative']:
            metadata['prompt_mode'] = part
        elif part.startswith('2x2-'):
            metadata['contract_mode'] = part[4:]
            metadata['dataset_size'] = '2x2'
        elif part.startswith('4x4-'):
            metadata['contract_mode'] = part[4:]
            metadata['dataset_size'] = '4x4'
        elif part in ['no-comm', 'code-nl', 'code-law']:
            metadata['contract_mode'] = part
        elif 'eval-' in part:
            m = re.search(r'eval-[\d-]+-(.+?)(?:-pd|-sh)?$', part)
            if m:
                metadata['model'] = m.group(1)
            if '-pd' in part:
                metadata['game'] = 'pd'
            elif '-sh' in part:
                metadata['game'] = 'sh'
    
    if metadata['game'] == 'unknown':
        if '-pd' in str(eval_path):
            metadata['game'] = 'pd'
        elif '-sh' in str(eval_path):
            metadata['game'] = 'sh'
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Check for API errors in .eval files")
    parser.add_argument('--log-dir', type=str, default='logs/')
    parser.add_argument('--examples', type=int, default=10, help="Number of random examples to print when errors are found")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for example selection (default: nondeterministic)")
    args = parser.parse_args()

    log_base = Path(args.log_dir)
    if not log_base.exists():
        print(f"Error: {log_base} not found")
        return 1

    print("Finding .eval files...")
    eval_files = list(log_base.glob('eval-*/*.eval'))
    eval_files.extend(log_base.glob('eval-*/*/*.eval'))
    eval_files.extend(log_base.glob('eval-*/*/*/*.eval'))
    print(f"Found {len(eval_files)} .eval files\n")

    print("Scanning for API errors...")
    cells = defaultdict(lambda: {'files': [], 'total': 0, 'affected': 0, 'error_types': defaultdict(int), 'examples': []})
    all_examples: list[dict] = []

    for eval_path in tqdm(eval_files, desc="Scanning", unit="file"):
        meta = extract_metadata(eval_path)
        result = check_eval_file_for_errors(eval_path)

        key = (meta['model'], meta['game'], meta['prompt_mode'], meta['contract_mode'], meta['dataset_size'])
        cells[key]['files'].append(str(eval_path))
        cells[key]['total'] += result['total_samples']
        cells[key]['affected'] += result['affected_samples']
        if result['has_errors']:
            cells[key]['has_errors'] = True
        for err_type, count in result['error_types'].items():
            cells[key]['error_types'][err_type] += count
        if result['examples']:
            # Keep a few per cell for debugging; also collect a global pool for random sampling.
            if len(cells[key]['examples']) < 25:
                remaining = 25 - len(cells[key]['examples'])
                cells[key]['examples'].extend(result['examples'][:remaining])
            for ex in result['examples']:
                all_examples.append({
                    'model': meta['model'],
                    'game': meta['game'],
                    'prompt_mode': meta['prompt_mode'],
                    'contract_mode': meta['contract_mode'],
                    'dataset_size': meta['dataset_size'],
                    'eval_file': str(eval_path),
                    'pattern': ex.get('pattern', ''),
                    'context': ex.get('context', ''),
                })

    cells_with_errors = {k: v for k, v in cells.items() if v.get('has_errors')}

    # Aggregate all error types
    all_error_types = defaultdict(int)
    for data in cells_with_errors.values():
        for err_type, count in data['error_types'].items():
            all_error_types[err_type] += count

    print(f"\nScan complete. Found errors in {len(cells_with_errors)} cells\n")

    if not cells_with_errors:
        print("=" * 80)
        print("NO API ERRORS FOUND")
        print("=" * 80)
        return 0

    print("=" * 100)
    print("API ERROR SUMMARY")
    print("=" * 100)
    print(f"\nTotal cells: {len(cells)}")
    print(f"Cells with errors: {len(cells_with_errors)}")
    print(f"\nError patterns found:")
    for pattern, count in sorted(all_error_types.items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count} matches")

    print("\n" + "=" * 100)
    print("CELLS WITH ERRORS")
    print("=" * 100)
    print(f"{'Model':<25} {'Game':<5} {'Prompt':<12} {'Contract':<10} {'Affected':<10} {'Total':<10} {'%':<8}")
    print("-" * 100)

    for (model, game, prompt, contract, size), data in sorted(cells_with_errors.items()):
        affected = data['affected']
        total = data['total']
        pct = (affected / total * 100) if total > 0 else 0
        print(f"{model:<25} {game:<5} {prompt:<12} {contract:<10} {affected:<10} {total:<10} {pct:<8.1f}")

    # Random examples (helpful for quickly seeing what matched)
    if args.seed is not None:
        random.seed(args.seed)
    n = max(0, int(args.examples))
    if n > 0 and all_examples:
        print("\n" + "=" * 100)
        print(f"RANDOM EXAMPLES (n={min(n, len(all_examples))})")
        print("=" * 100)
        for ex in random.sample(all_examples, k=min(n, len(all_examples))):
            print(f"- {ex['model']} {ex['game']} {ex['prompt_mode']} {ex['contract_mode']} {ex['dataset_size']}")
            print(f"  file: {ex['eval_file']}")
            print(f"  pattern: {ex['pattern']}")
            print(f"  context: {ex['context']}")

    print("\n" + "=" * 100)
    print(f"\nFound API errors in {len(cells_with_errors)} cells")
    return 1


if __name__ == '__main__':
    exit(main())
