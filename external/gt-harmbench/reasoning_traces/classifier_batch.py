"""
Classify reasoning traces using OpenAI Batch API.
Much faster and 50% cheaper than real-time API for large jobs.
"""

import json
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batch_api import (
    prepare_batch_requests,
    create_batch_job,
    poll_batch_job,
    retrieve_batch_results,
    process_batch_results
)

from openai import OpenAI

# Import taxonomy and prompt from classifier.py
from reasoning_traces.classifier import TAXONOMY, CLASSIFICATION_PROMPT


def create_classification_requests(samples, model="gpt-4o-mini"):
    """Create batch API requests for classification."""
    requests = []
    
    for sample in samples:
        # Extract data
        reasoning = sample.get('reasoning', '')
        game_type = sample.get('scores', {}).get('game_type', 'Unknown')
        scenario = sample.get('story', '')
        action = sample.get('action', '')
        
        # Create prompt
        prompt = CLASSIFICATION_PROMPT.format(
            taxonomy=TAXONOMY,
            game_type=game_type,
            scenario_summary=scenario[:500] + "..." if len(scenario) > 500 else scenario,
            reasoning=reasoning[:3000] + "..." if len(reasoning) > 3000 else reasoning,
            action=action
        )
        
        # Create request
        request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        # GPT-5 models use max_completion_tokens, others use max_tokens
        if "gpt-5" in model or "o3" in model or "o4" in model:
            request["max_completion_tokens"] = 4096
        else:
            request["max_tokens"] = 1024
            request["response_format"] = {"type": "json_object"}
        
        requests.append(request)
    
    return requests


def parse_batch_classifications(batch_results, samples):
    """Parse batch API results into classification format."""
    classified_samples = []
    
    for i, (result, sample) in enumerate(zip(batch_results, samples)):
        if result is None:
            print(f"Warning: No result for sample {i} (game_id: {sample.get('game_id')})")
            classification = {"categories": [], "error": "No batch result"}
        else:
            content = result.get("content", "")
            try:
                classification = json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON for sample {i}")
                classification = {"categories": [], "error": "JSON parse failed", "raw": content}
        
        # Add classification to sample
        classified_sample = {**sample, "classification": classification}
        classified_samples.append(classified_sample)
    
    return classified_samples


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify reasoning traces using Batch API")
    parser.add_argument("input_file", help="Input JSON file with extracted reasoning")
    parser.add_argument("--output", required=True, help="Output JSON file for classifications")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use (default: gpt-4o-mini)")
    parser.add_argument("--batch-file", default="batch_requests.jsonl", help="Intermediate batch file")
    parser.add_argument("--skip-poll", action="store_true", help="Skip polling, just submit batch")
    parser.add_argument("--batch-id", help="Resume from existing batch ID")
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = OpenAI()
    
    if args.batch_id:
        # Resume existing batch
        print(f"Resuming batch job: {args.batch_id}")
        batch_id = args.batch_id
        
        # Load samples for parsing results
        with open(args.input_file, 'r') as f:
            samples = json.load(f)
    else:
        # Load input samples
        print(f"Loading samples from {args.input_file}")
        with open(args.input_file, 'r') as f:
            samples = json.load(f)
        
        print(f"Loaded {len(samples)} samples")
        
        # Create batch requests
        print("Creating batch requests...")
        requests = create_classification_requests(samples, model=args.model)
        
        # Write batch file
        prepare_batch_requests(requests, args.batch_file)
        print(f"Wrote {len(requests)} requests to {args.batch_file}")
        
        # Upload and create batch job
        batch_id = create_batch_job(client, args.batch_file)
        
        # Save batch_id to file for easy cancellation/resume
        batch_id_file = args.output.replace('.json', '_batch_id.txt')
        with open(batch_id_file, 'w') as f:
            f.write(batch_id)
        
        print(f"\n{'='*70}")
        print(f"Batch ID: {batch_id}")
        print(f"Saved to: {batch_id_file}")
        print(f"{'='*70}")
        print(f"\nTo CANCEL this batch, run:")
        print(f"  python -c 'from openai import OpenAI; OpenAI().batches.cancel(\"{batch_id}\")'")
        print(f"\nTo RESUME later, run:")
        print(f"  --batch-id {batch_id}")
        print(f"{'='*70}\n")
        
        if args.skip_poll:
            print("\nSkipping polling. Run again with --batch-id to retrieve results.")
            return
    
    # Poll for completion
    output_file_id = poll_batch_job(client, batch_id, poll_interval=60)
    
    # Retrieve results
    results = retrieve_batch_results(client, output_file_id)
    
    # Process results
    batch_results, tokens_in, tokens_out = process_batch_results(results, len(samples))
    
    print(f"\nToken usage:")
    print(f"  Prompt tokens: {tokens_in:,}")
    print(f"  Completion tokens: {tokens_out:,}")
    print(f"  Total: {tokens_in + tokens_out:,}")
    
    # Calculate cost (Batch API is 50% off standard pricing)
    # gpt-4o-mini: $0.075/$0.30 per 1M tokens (batch pricing)
    cost = (tokens_in * 0.075 + tokens_out * 0.30) / 1_000_000
    print(f"  Estimated cost: ${cost:.3f}")
    
    # Parse classifications
    print("\nParsing classifications...")
    classified_samples = parse_batch_classifications(batch_results, samples)
    
    # Count categories
    category_counts = {}
    for sample in classified_samples:
        for cat in sample.get('classification', {}).get('categories', []):
            category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory frequencies:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(samples)
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(classified_samples, f, indent=2)
    
    print(f"\nSaved {len(classified_samples)} classified samples to {args.output}")


if __name__ == "__main__":
    main()
