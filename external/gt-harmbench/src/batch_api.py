"""
Batch API utilities for OpenAI Batch API processing.

This module provides functions to:
1. Prepare batch request files (JSONL format)
2. Upload batches and create batch jobs
3. Poll for completion
4. Retrieve and parse results
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def prepare_batch_requests(
    requests: List[Dict[str, Any]], output_path: str
) -> None:
    """
    Write batch requests to a JSONL file.

    Args:
        requests: List of request payloads (each should be a dict with model, messages, etc.)
        output_path: Path to write the JSONL file
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, req in enumerate(requests):
            batch_request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": req,
            }
            f.write(json.dumps(batch_request) + "\n")


def create_batch_job(
    client: OpenAI, batch_file_path: str, completion_window: str = "24h"
) -> str:
    """
    Upload batch file and create a batch job.

    Args:
        client: OpenAI client (synchronous)
        batch_file_path: Path to the JSONL batch file
        completion_window: Completion window (default: "24h")

    Returns:
        Batch job ID
    """
    print(f"Uploading batch file: {batch_file_path}")
    with open(batch_file_path, "rb") as f:
        file_response = client.files.create(file=f, purpose="batch")
    file_id = file_response.id

    print(f"Creating batch job with file_id: {file_id}")
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    batch_id = batch_response.id
    print(f"Batch job created: {batch_id}")
    return batch_id


def poll_batch_job(
    client: OpenAI, batch_id: str, poll_interval: int = 60
) -> str:
    """
    Poll batch job until completion.

    Args:
        client: OpenAI client (synchronous)
        batch_id: Batch job ID
        poll_interval: Seconds between polls (default: 60)

    Returns:
        Output file ID

    Raises:
        Exception: If batch job fails
    """
    import datetime
    
    print(f"Polling batch job {batch_id} (checking every {poll_interval}s)...")
    start_time = time.time()
    last_completed = 0
    
    while True:
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
        
        # Show progress if available
        request_counts = batch_status.request_counts
        if request_counts:
            total = request_counts.total
            completed = request_counts.completed
            failed = request_counts.failed
            pct = (completed / total * 100) if total > 0 else 0
            
            # Calculate ETA
            if completed > last_completed and completed > 0:
                rate = completed / elapsed  # items per second
                remaining = total - completed
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "calculating..."
            
            print(f"Status: {status} | Progress: {completed}/{total} ({pct:.1f}%) | "
                  f"Failed: {failed} | Elapsed: {elapsed_str} | ETA: {eta_str}")
            last_completed = completed
        else:
            print(f"Status: {status} | Elapsed: {elapsed_str}")

        if status == "completed":
            output_file_id = batch_status.output_file_id
            if output_file_id:
                total_time = str(datetime.timedelta(seconds=int(elapsed)))
                print(f"\n✓ Batch completed in {total_time}! Output file ID: {output_file_id}")
                return output_file_id
            else:
                raise Exception(f"Batch {batch_id} completed but no output_file_id found")
        elif status == "failed":
            error = getattr(batch_status, "error", None)
            raise Exception(f"Batch job {batch_id} failed: {error}")
        elif status in ("cancelling", "cancelled"):
            raise Exception(f"Batch job {batch_id} was cancelled")

        time.sleep(poll_interval)


def retrieve_batch_results(client: OpenAI, output_file_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve and parse batch results.

    Args:
        client: OpenAI client (synchronous)
        output_file_id: Output file ID from completed batch

    Returns:
        List of result dictionaries
    """
    print(f"Retrieving batch results from file: {output_file_id}")
    result_file = client.files.retrieve(output_file_id)
    file_content = client.files.content(output_file_id)
    
    # Handle both bytes and file-like objects
    if isinstance(file_content, bytes):
        content_str = file_content.decode("utf-8")
    else:
        content_str = file_content.read().decode("utf-8")

    results: List[Dict[str, Any]] = []
    for line in content_str.splitlines():
        if line.strip():
            results.append(json.loads(line))

    print(f"Retrieved {len(results)} results from batch")
    return results


def process_batch_results(
    results: List[Dict[str, Any]], expected_count: int
) -> Tuple[List[Optional[Dict[str, Any]]], int, int]:
    """
    Process batch results and extract responses with token usage.

    Args:
        results: List of batch result dictionaries
        expected_count: Expected number of results

    Returns:
        Tuple of (list of parsed responses, total_tokens_in, total_tokens_out)
    """
    parsed_responses: List[Optional[Dict[str, Any]]] = []
    total_tokens_in = 0
    total_tokens_out = 0

    # Create a mapping from custom_id to result
    result_map: Dict[str, Dict[str, Any]] = {}
    for result in results:
        custom_id = result.get("custom_id", "")
        if custom_id:
            result_map[custom_id] = result

    # Process results in order
    for i in range(expected_count):
        custom_id = f"request-{i}"
        if custom_id not in result_map:
            parsed_responses.append(None)
            continue

        result = result_map[custom_id]
        response = result.get("response", {})

        # Check for errors
        if result.get("error"):
            print(f"Error in batch result {custom_id}: {result['error']}")
            parsed_responses.append(None)
            continue

        # Extract token usage
        body = response.get("body", {})
        usage = body.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        total_tokens_in += tokens_in
        total_tokens_out += tokens_out

        # Extract response content
        choices = body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            parsed_responses.append({"content": content, "usage": usage})
        else:
            parsed_responses.append(None)

    return parsed_responses, total_tokens_in, total_tokens_out

