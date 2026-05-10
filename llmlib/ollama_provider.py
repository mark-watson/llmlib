"""Ollama provider wrapper (native Ollama REST API)."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

from llmlib.base_model_wrapper import BaseModelWrapper
from llmlib.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)

_DEFAULT_BASE = "https://ollama.com"


class OllamaWrapper(BaseModelWrapper):
    """Wrap Ollama (cloud or local) for generation, chat, and embeddings.

    Uses the native Ollama REST API (``/api/generate``, ``/api/chat``,
    ``/api/embed``).  Set ``OLLAMA_API_KEY`` for cloud access, or leave it
    empty for an unauthenticated local instance.
    """

    def __init__(
        self,
        key: str | None = None,
        embeddings_dir: str = "./db_embeddings",
        model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        base_url: str = _DEFAULT_BASE,
    ) -> None:
        super().__init__(embeddings_dir=embeddings_dir)
        if key is None:
            key = os.getenv("OLLAMA_API_KEY", "")
        self._key = key
        self._model = model
        self._embedding_model = embedding_model
        self._base = base_url.rstrip("/")
        self._store: EmbeddingStore | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._key:
            h["Authorization"] = f"Bearer {self._key}"
        return h

    def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base}/api/{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_store(self) -> EmbeddingStore:
        if self._store is None:
            self._store = EmbeddingStore(
                persist_dir=self.embeddings_dir,
                embed_fn=self._embed,
            )
        return self._store

    def _embed(self, text: str) -> list[float]:
        payload = {
            "model": self._embedding_model,
            "input": text,
        }
        data = self._request("embed", payload)
        try:
            # Ollama /api/embed returns {"embeddings": [[...]]}
            embeddings = data["embeddings"]
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            return embeddings
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Ollama embed response: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Text completion (generate)
    # ------------------------------------------------------------------

    def get_completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.5,
        top_p: float = 0.75,
    ) -> str:
        """Generate text via Ollama ``/api/generate``.

        Only newly generated tokens are returned (the prompt is not echoed).
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        data = self._request("generate", payload)
        try:
            return data["response"]
        except KeyError as exc:
            logger.error("Unexpected Ollama generate response: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    def get_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> str:
        """Run a multi-turn chat via Ollama ``/api/chat``.

        Args:
            messages: OpenAI-style role/content list.
            max_tokens: Maximum tokens to generate (maps to ``num_predict``).
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.

        Returns:
            Assistant reply text.
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        data = self._request("chat", payload)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Ollama chat response: %s", exc)
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
