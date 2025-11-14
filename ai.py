"""Thin AI helper that wraps DeepSeek for text summaries."""

from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Build the OpenAI client once so every request reuses the same session."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing `DEEPSEEK_API_KEY` environment variable.")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=api_key, base_url=base_url)


def summarize_text(source_text: str) -> str:
    """Return a concise summary for the provided transcript text."""
    cleaned = (source_text or "").strip()
    if not cleaned:
        raise ValueError("Empty transcript cannot be summarized.")

    client = _get_client()
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_SUMMARY_MODEL", "deepseek-chat"),
        messages=[
            {
                "role": "system",
                "content": "You write short, clear summaries for meeting transcripts.",
            },
            {
                "role": "user",
                "content": f"Summarize this transcript in 3 sentences max:\n{cleaned}",
            },
        ],
        stream=False,
    )
    return response.choices[0].message.content.strip()