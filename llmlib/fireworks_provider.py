"""Fireworks.ai provider wrapper (OpenAI-compatible API)."""

from __future__ import annotations

import logging
import os
from typing import Any

from llmlib.base_model_wrapper import BaseModelWrapper
from llmlib.embedding_store import EmbeddingStore
from llmlib.utils import http_post_json

logger = logging.getLogger(__name__)

_DEFAULT_BASE = "https://api.fireworks.ai/inference/v1"


class FireworksWrapper(BaseModelWrapper):
    """Wrap Fireworks.ai for chat completions and embeddings."""

    def __init__(
        self,
        key: str | None = None,
        embeddings_dir: str = "./db_embeddings",
        model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        base_url: str = _DEFAULT_BASE,
    ) -> None:
        super().__init__(embeddings_dir=embeddings_dir)
        if key is None:
            key = os.getenv("FIREWORKS_API_KEY")
        if not key:
            raise ValueError(
                "Fireworks API key is required. Pass 'key' or set FIREWORKS_API_KEY."
            )
        self._key = key
        self._model = model
        self._embedding_model = embedding_model
        self._base = base_url.rstrip("/")
        self._store: EmbeddingStore | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"}

    def _get_store(self) -> EmbeddingStore:
        if self._store is None:
            self._store = EmbeddingStore(
                persist_dir=self.embeddings_dir,
                embed_fn=self._embed,
            )
        return self._store

    def _embed(self, text: str) -> list[float]:
        url = f"{self._base}/embeddings"
        payload = {
            "model": self._embedding_model,
            "input": text,
        }
        data = http_post_json(url, payload, headers=self._headers())
        try:
            return data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected Fireworks embedding response: %s", exc)
            return []

    def _chat_complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> dict[str, Any]:
        url = f"{self._base}/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        return http_post_json(url, payload, headers=self._headers())

    # ------------------------------------------------------------------
    # Text completion
    # ------------------------------------------------------------------

    def get_completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.5,
        top_p: float = 0.75,
    ) -> str:
        """Generate a text completion by treating *prompt* as a user message."""
        messages = [{"role": "user", "content": prompt}]
        response = self._chat_complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return self._extract_text(response)

    def get_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> str:
        """Run a multi-turn chat completion.

        Args:
            messages: OpenAI-style chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.

        Returns:
            Assistant reply text.
        """
        response = self._chat_complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return self._extract_text(response)

    def _extract_text(self, response: dict[str, Any]) -> str:
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected Fireworks response structure: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Local embeddings
    # ------------------------------------------------------------------

    def create_local_embeddings_files_in_dir(self, path: str) -> None:
        """Index all files under *path* into a ChromaDB-backed embedding store."""
        store = self._get_store()
        store.index_directory(path, glob="**/*")

    def query_local_embeddings(self, query: str, n: int = 10) -> list[dict[str, Any]]:
        """Query the local embedding store and return ranked results."""
        store = self._get_store()
        return store.query(query, n=n)
