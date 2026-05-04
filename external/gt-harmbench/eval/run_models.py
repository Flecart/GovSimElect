"""Run evaluations on multiple models in parallel with progress tracking."""

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

MODELS = [
    "x-ai/grok-4.1-fast:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "kwaipilot/kat-coder-pro:free",
    "z-ai/glm-4.5-air:free",
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

DATASET = "data/contextualization-with-targets.csv"
TIMES = 1
MAX_PARALLEL = 7

def run_model(model_name: str) -> tuple[str, int, float]:
    """Run evaluation for a single model."""
    start_time = time.time()
    print(f"🚀 Starting {model_name} at {datetime.now().strftime('%H:%M:%S')}")
    
    cmd = [
        "uv", "run", "--env-file", ".env",
        "python3", "eval/eval.py",
        "--model-name", model_name,
        "--times", str(TIMES),
        DATASET
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        duration = time.time() - start_time
        print(f"✅ Completed {model_name} in {duration:.1f}s")
        return model_name, 0, duration
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ Failed {model_name} after {duration:.1f}s")
        print(f"   Error: {e.stderr[:200]}")
        return model_name, 1, duration

def main():
    print(f"{'='*60}")
    print(f"Running {len(MODELS)} models with max {MAX_PARALLEL} parallel")
    print(f"Dataset: {DATASET}, Times: {TIMES}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run models in parallel
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(run_model, model): model for model in MODELS}
        
        results = []
        completed = 0
        total = len(MODELS)
        
        for future in as_completed(futures):
            model_name, exit_code, duration = future.result()
            results.append((model_name, exit_code, duration))
            completed += 1
            
            print(f"\n{'─'*60}")
            print(f"Progress: {completed}/{total} models completed")
            print(f"Remaining: {total - completed}")
            print(f"{'─'*60}\n")
    
    total_duration = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    success = [(m, d) for m, code, d in results if code == 0]
    failed = [(m, d) for m, code, d in results if code != 0]
    
    print(f"\n✅ Successful: {len(success)}/{len(MODELS)}")
    for model, duration in success:
        print(f"   - {model:<40} ({duration:.1f}s)")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(MODELS)}")
        for model, duration in failed:
            print(f"   - {model:<40} ({duration:.1f}s)")
    
    print(f"\n⏱️  Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"📊 View results: inspect view")
    print(f"{'='*60}\n")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
