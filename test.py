#!/usr/bin/env python3
"""Minimal OpenAI GPT-5.4-nano prompt test."""

import os

from openai import OpenAI


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set OPENAI_API_KEY before running test.py")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": "Write one short sentence about climate policy."}],
        temperature=None,
        top_p=None,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
