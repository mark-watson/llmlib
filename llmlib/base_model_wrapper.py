"""Abstract base class for LLM provider wrappers."""

from __future__ import annotations

import abc
from typing import Any


class BaseModelWrapper(abc.ABC):
    """Abstract base for model wrappers providing completions and local embeddings."""

    def __init__(self, embeddings_dir: str = "./db_embeddings") -> None:
        self.embeddings_dir = embeddings_dir

    @abc.abstractmethod
    def get_completion(self, prompt: str, *, max_tokens: int = 64, temperature: float = 0.5, top_p: float = 0.75) -> str:
        """Generate a text completion for *prompt*.

        Args:
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.

        Returns:
            Generated text string.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_local_embeddings_files_in_dir(self, path: str) -> None:
        """Index all files under *path* into a local embedding store.

        Args:
            path: Directory containing documents to index.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def query_local_embeddings(self, query: str, n: int = 10) -> list[dict[str, Any]]:
        """Query the local embedding store.

        Args:
            query: Query string.
            n: Maximum number of results to return.

        Returns:
            A list of result dicts, each containing at least ``text`` and ``score``.
        """
        raise NotImplementedError
