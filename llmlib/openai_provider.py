"""OpenAI provider wrapper."""

from __future__ import annotations

import logging
import os
from typing import Any

from llmlib.base_model_wrapper import BaseModelWrapper
from llmlib.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


def _require_openai() -> Any:
    try:
        import openai as _openai
    except ImportError as exc:
        raise ImportError(
            "openai is required for OpenAiWrapper. Install it with: "
            "pip install llmlib[openai]"
        ) from exc
    return _openai


class OpenAiWrapper(BaseModelWrapper):
    """Wrap OpenAI APIs for text completion and local vector search."""

    def __init__(
        self,
        key: str | None = None,
        embeddings_dir: str = "./db_embeddings",
        completion_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        super().__init__(embeddings_dir=embeddings_dir)
        openai = _require_openai()
        if key is None:
            key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key is required. Pass 'key' or set OPENAI_API_KEY."
            )
        self._client = openai.OpenAI(api_key=key)
        self._completion_model = completion_model
        self._embedding_model = embedding_model
        self._store: EmbeddingStore | None = None

    def _get_store(self) -> EmbeddingStore:
        if self._store is None:
            self._store = EmbeddingStore(
                persist_dir=self.embeddings_dir,
                embed_fn=self._embed,
            )
        return self._store

    def _embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text,
            model=self._embedding_model,
        )
        return response.data[0].embedding

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
        """Generate a completion for *prompt* using the chat completions API.

        The prompt is sent as a user message with a lightweight system prompt
        encouraging a concise continuation.
        """
        response = self._client.chat.completions.create(
            model=self._completion_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide concise completions."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        try:
            return response.choices[0].message.content or ""
        except (IndexError, AttributeError) as exc:
            logger.error("Unexpected OpenAI response structure: %s", exc)
            return ""

    def get_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> str:
        """Run a multi-turn chat completion.

        Args:
            messages: OpenAI chat message list, e.g.
                ``[{"role": "user", "content": "Hello"}]``.
            model: Override the default completion model.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.

        Returns:
            Assistant reply text.
        """
        response = self._client.chat.completions.create(
            model=model or self._completion_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        try:
            return response.choices[0].message.content or ""
        except (IndexError, AttributeError) as exc:
            logger.error("Unexpected OpenAI response structure: %s", exc)
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
