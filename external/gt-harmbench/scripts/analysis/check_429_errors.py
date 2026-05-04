#!/usr/bin/env python3
"""Check for API errors in .eval files.

Searches ALL content for OpenAI/OpenRouter error messages.

Usage:
    python scripts/analysis/check_429_errors.py
"""

import argparse
import subprocess
import re
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


# API error patterns (focused on rate limiting)
ERROR_PATTERNS = [
    # Rate limit / quota errors
    r'insufficient_quota',
    r'insufficient quota',
    r'rate.*limit.*exceed',
    r'quota.*exceed',
]

ERROR_REGEX = re.compile('|'.join(ERROR_PATTERNS), re.IGNORECASE)


def check_eval_file_for_errors(eval_path: Path) -> dict:
    """Check .eval file for API errors using unzip+grep (fast, comprehensive)."""
    result = {
        'has_errors': False,
        'total_samples': 0,
        'affected_samples': 0,
        'error_types': defaultdict(int),
        'examples': []
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
        # This is fast and catches errors anywhere in the archive
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
                    if len(result['examples']) < 5:
                        for match in re.finditer(pattern, content, re.IGNORECASE):
                            start = max(0, match.start() - 40)
                            end = min(len(content), match.end() + 80)
                            context = content[start:end].replace('\n', ' ')[:120]
                            result['examples'].append({
                                'pattern': pattern,
                                'context': context
                            })
                            if len(result['examples']) >= 5:
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


def scan_eval_file(eval_path: Path) -> dict:
    """Scan a single .eval file for 429 errors."""
    result = {
        'has_429': False,
        'total_samples': 0,
        'affected_samples': 0,
        'examples': []
    }
    
    try:
        with zipfile.ZipFile(eval_path, 'r') as z:
            sample_files = [n for n in z.namelist() if n.startswith('samples/')]
            result['total_samples'] = len(sample_files)
            
            for sample_file in sample_files:
                sample_id = sample_file.split('/')[-1].replace('.json', '')
                
                try:
                    with z.open(sample_file) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        
                        if has_429_error(content):
                            result['has_429'] = True
                            result['affected_samples'] += 1
                            
                            if len(result['examples']) < 2:
                                # Extract context around the error
                                idx = content_lower = content.lower()
                                for pattern in ['error code: 429', "'code': 429", '"code": 429', 'insufficient_quota']:
                                    idx = content_lower.find(pattern)
                                    if idx >= 0:
                                        start = max(0, idx - 50)
                                        end = min(len(content), idx + 150)
                                        ctx = content[start:end].replace('\n', ' ')[:200]
                                        result['examples'].append({'sample': sample_id, 'context': ctx})
                                        break
                except:
                    pass
    except:
        pass
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Check for API errors in .eval files")
    parser.add_argument('--log-dir', type=str, default='logs/')
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
        if result['examples'] and not cells[key]['examples']:
            cells[key]['examples'] = result['examples']

    cells_with_errors = {k: v for k, v in cells.items() if v.get('has_errors')}

    # Aggregate all error types
    all_error_types = defaultdict(int)
    for data in cells_with_errors.values():
        for err_type, count in data['error_types'].items():
            all_error_types[err_type] += count

    print(f"\nScan complete. Found errors in {len(cells_with_errors)} cells\n")

    if not cells_with_errors:
        print("=" * 80)
        print("✅ NO API ERRORS FOUND")
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

    # Show examples
    print("\n" + "=" * 100)
    print("EXAMPLE ERRORS")
    print("=" * 100)
    shown = 0
    for data in cells_with_errors.values():
        if data.get('examples') and shown < 5:
            for ex in data['examples']:
                print(f"\nPattern: {ex['pattern']}")
                print(f"Context: {ex['context']}")
                shown += 1
        if shown >= 5:
            break

    print("\n" + "=" * 100)
    print(f"\n❌ Found API errors in {len(cells_with_errors)} cells")
    return 1


if __name__ == '__main__':
    exit(main())
