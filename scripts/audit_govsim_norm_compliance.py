#!/usr/bin/env python3
"""Judge norm violations and cross-reference monitoring and punishment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simulation.scenarios.govsim_monitoring.audit import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    build_compliance_cases,
    cross_tabulate,
    judge_prompt,
    load_trace,
    validate_verdicts,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Use an LLM judge to classify natural-language norm violations, then "
          "cross-reference successful monitoring and punishment."
      )
  )
  parser.add_argument("trace", type=Path, help="Path to consolidated_results.json")
  parser.add_argument("--model", default="anthropic/claude-sonnet-5")
  parser.add_argument("--max-tokens", type=int, default=10000)
  parser.add_argument("--output", type=Path, help="Optional JSON output path")
  parser.add_argument(
      "--no-routing-fallback",
      action="store_true",
      help="Fail instead of retrying without strict Amazon Bedrock routing.",
  )
  return parser.parse_args()


def request_judgment(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    allow_routing_fallback: bool,
):
  request = {
      "model": model,
      "messages": [
          {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
          {"role": "user", "content": prompt},
      ],
      "temperature": 0,
      "max_tokens": max_tokens,
      "response_format": {"type": "json_object"},
  }
  try:
    return client.chat.completions.create(
        **request,
        extra_body={
            "provider": {"order": ["amazon-bedrock"], "allow_fallbacks": False}
        },
    )
  except Exception as error:
    if not allow_routing_fallback:
      raise
    print(
        "WARNING: strict Amazon Bedrock routing failed; retrying with standard "
        f"OpenRouter routing ({error}).",
        file=sys.stderr,
    )
    return client.chat.completions.create(**request)


def main() -> None:
  args = parse_args()
  load_dotenv(REPO_ROOT / ".env", override=True)
  api_key = os.getenv("OPENROUTER_API_KEY")
  if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

  initialization, rounds = load_trace(args.trace)
  cases = build_compliance_cases(initialization, rounds)
  client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
  response = request_judgment(
      client,
      model=args.model,
      prompt=judge_prompt(cases),
      max_tokens=args.max_tokens,
      allow_routing_fallback=not args.no_routing_fallback,
  )
  content = response.choices[0].message.content
  if not content:
    raise RuntimeError("Judge returned an empty response")
  judgment = json.loads(content)
  verdicts = validate_verdicts(cases, judgment.get("results", []))
  result = {
      "trace": str(args.trace),
      "judge_model": response.model,
      "total_agent_rounds": len(cases),
      **cross_tabulate(rounds, verdicts),
  }
  rendered = json.dumps(result, indent=2, ensure_ascii=False)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
  print(rendered)


if __name__ == "__main__":
  main()

