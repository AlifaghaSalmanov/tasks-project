from __future__ import annotations

import os
from functools import lru_cache

from faster_whisper.transcribe import WhisperModel
from openai import OpenAI


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing `DEEPSEEK_API_KEY` environment variable.")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_whisper_model():
    model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def summarize_text(source_text: str) -> str:
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
                "content": f"Summarize this conversation briefly, without adding phrases like 'Based on the transcript.' Also, include the names of people in the summary: {cleaned}",
            },
        ],
        stream=False,
    )
    return response.choices[0].message.content.strip()