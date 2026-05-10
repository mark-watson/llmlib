"""Shared helpers for text processing, safe parsing, and HTTP."""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    """POST a JSON payload and return the parsed JSON response.

    Args:
        url: Target URL.
        payload: JSON-serializable dict.
        headers: Extra HTTP headers.

    Returns:
        Parsed JSON response dict.

    Raises:
        urllib.error.HTTPError: On non-2xx responses.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters.

    Args:
        text: The full text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start <= 0 or start >= text_len:
            break
    return chunks


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Attempt to parse *text* as JSON, returning *default* on failure.

    Args:
        text: JSON string.
        default: Value to return if parsing fails.

    Returns:
        Parsed object or *default*.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.debug("JSON parse failed for: %r", text)
        return default


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in the range [-1, 1].
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
