"""Google Gemini provider wrapper using the Gemini REST API."""

from __future__ import annotations

import logging
import os
import urllib.request
from typing import Any

from llmlib.base_model_wrapper import BaseModelWrapper
from llmlib.embedding_store import EmbeddingStore
from llmlib.utils import http_post_json

logger = logging.getLogger(__name__)

_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiWrapper(BaseModelWrapper):
    """Wrap Google Gemini for text completion (with/without search) and embeddings."""

    def __init__(
        self,
        key: str | None = None,
        embeddings_dir: str = "./db_embeddings",
        model: str = "gemini-1.5-flash",
        embedding_model: str = "models/text-embedding-004",
        base_url: str = _DEFAULT_BASE,
    ) -> None:
        super().__init__(embeddings_dir=embeddings_dir)
        if key is None:
            key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Google API key is required. Pass 'key' or set GOOGLE_API_KEY."
            )
        self._key = key
        self._model = model
        self._embedding_model = embedding_model
        self._base = base_url.rstrip("/")
        self._store: EmbeddingStore | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_store(self) -> EmbeddingStore:
        if self._store is None:
            self._store = EmbeddingStore(
                persist_dir=self.embeddings_dir,
                embed_fn=self._embed,
            )
        return self._store

    def _embed(self, text: str) -> list[float]:
        url = (
            f"{self._base}/models/{self._embedding_model}:embedContent"
            f"?key={self._key}"
        )
        payload = {"content": {"parts": [{"text": text}]}}
        data = http_post_json(url, payload, headers={})
        try:
            return data["embedding"]["values"]
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Gemini embedding response: %s", exc)
            return []

    def _generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.5,
        top_p: float = 0.75,
        search: bool = False,
    ) -> dict[str, Any]:
        """Call Gemini generateContent and return the raw candidate dict."""
        url = f"{self._base}/models/{self._model}:generateContent?key={self._key}"
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        }
        if search:
            payload["tools"] = [{"googleSearch": {}}]
        return http_post_json(url, payload, headers={})

    def _extract_text(self, response: dict[str, Any]) -> str:
        try:
            parts = response["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected Gemini response structure: %s", exc)
            return ""

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
        """Generate a text completion without grounding search."""
        response = self._generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            search=False,
        )
        return self._extract_text(response)

    def get_completion_with_search(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.5,
        top_p: float = 0.75,
    ) -> dict[str, Any]:
        """Generate a text completion with Google Search grounding enabled.

        Returns:
            A dict with keys ``text`` (str) and ``grounding`` (list of
            grounding chunk dicts from the API).
        """
        response = self._generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            search=True,
        )
        text = self._extract_text(response)
        grounding: list[dict[str, Any]] = []
        try:
            chunks = (
                response["candidates"][0]
                .get("groundingMetadata", {})
                .get("groundingChunks", [])
            )
            grounding = chunks
        except (KeyError, IndexError, TypeError):
            pass
        return {"text": text, "grounding": grounding}

    # ------------------------------------------------------------------
    # Chat completion (multi-turn)
    # ------------------------------------------------------------------

    def get_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        search: bool = False,
    ) -> str:
        """Run a multi-turn chat completion.

        Args:
            messages: List of role/content dicts. Roles should be ``user`` or
                ``model`` (Gemini uses ``model`` instead of ``assistant``).
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            search: Enable Google Search grounding.

        Returns:
            Assistant reply text.
        """
        contents = []
        for m in messages:
            role = m.get("role", "user")
            # Normalize common "assistant" role to Gemini "model"
            if role == "assistant":
                role = "model"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        url = f"{self._base}/models/{self._model}:generateContent?key={self._key}"
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        }
        if search:
            payload["tools"] = [{"googleSearch": {}}]

        response = http_post_json(url, payload, headers={})
        return self._extract_text(response)

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
