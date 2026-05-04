"""Backward-compatible wrapper for the new analysis CLI.

This file is kept so existing references to `eval/analysis_script.py`
continue to work. The real implementation now lives under
`eval/analysis/cli.py` and `eval/analysis/core.py`.
"""

from __future__ import annotations

from pathlib import Path

from eval.analysis.cli import run_cli


if __name__ == "__main__":
    # Process all .eval files in logs/ directory with "gamify" prefix
    logs_dir = Path("logs")
    gamify_eval_files = sorted(logs_dir.glob("*.eval"))
    
    print(f"Processing {len(gamify_eval_files)} eval files from logs/ with prefix 'gamify'...")
    for eval_file in gamify_eval_files:
        print(f"\nProcessing: {eval_file.name}")
        run_cli(
            log_path=eval_file,
            log_type="eval",
            max_logs=100,  # High number since we're specifying exact files
            output_dir=Path("assets"),
            plots=("accuracy", "welfare", "heatmap"),
            prefix="gamify",
        )
    
    # Process all .eval files in logs/standard/ directory without prefix
    standard_dir = Path("logs/standard")
    if standard_dir.exists():
        standard_eval_files = sorted(standard_dir.glob("*.eval"))
        
        print(f"\n\nProcessing {len(standard_eval_files)} eval files from logs/standard/ without prefix...")
        for eval_file in standard_eval_files:
            print(f"\nProcessing: {eval_file.name}")
            run_cli(
                log_path=eval_file,
                log_type="eval",
                max_logs=100,  # High number since we're specifying exact files
                output_dir=Path("assets"),
                plots=("accuracy", "welfare", "heatmap"),
                prefix=None,  # No prefix
            )
    else:
        print(f"\nWarning: {standard_dir} does not exist, skipping standard logs")


