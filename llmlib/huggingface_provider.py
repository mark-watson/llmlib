"""Hugging Face provider wrapper using transformers and sentence-transformers."""

from __future__ import annotations

import logging
import os
from typing import Any

from llmlib.base_model_wrapper import BaseModelWrapper
from llmlib.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


def _require_transformers() -> Any:
    try:
        import transformers as _transformers
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for HuggingFaceAiWrapper. "
            "Install them with: pip install llmlib[huggingface]"
        ) from exc
    return _transformers


def _require_sentence_transformers() -> Any:
    try:
        import sentence_transformers as _st
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for HuggingFaceAiWrapper. "
            "Install it with: pip install llmlib[huggingface]"
        ) from exc
    return _st


class HuggingFaceAiWrapper(BaseModelWrapper):
    """Wrap local Hugging Face models for text completion and local vector search."""

    def __init__(
        self,
        embeddings_dir: str = "./db_embeddings",
        generation_model: str = "facebook/opt-iml-1.3b",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: int = -1,
        torch_dtype: str | None = "bfloat16",
    ) -> None:
        super().__init__(embeddings_dir=embeddings_dir)
        self._generation_model = generation_model
        self._embedding_model = embedding_model
        self._device = device  # -1 = CPU, 0 = cuda:0, etc.
        self._torch_dtype = torch_dtype
        self._gen_pipeline: Any | None = None
        self._embed_model: Any | None = None
        self._store: EmbeddingStore | None = None

    def _get_gen_pipeline(self) -> Any:
        """Lazy-load the text-generation pipeline."""
        if self._gen_pipeline is None:
            transformers = _require_transformers()
            import torch

            dtype = None
            if self._torch_dtype:
                dtype = getattr(torch, self._torch_dtype, None)
                if dtype is None:
                    raise ValueError(f"Unsupported torch_dtype: {self._torch_dtype}")

            self._gen_pipeline = transformers.pipeline(
                "text-generation",
                model=self._generation_model,
                device=self._device,
                torch_dtype=dtype,
            )
        return self._gen_pipeline

    def _get_embed_model(self) -> Any:
        """Lazy-load the sentence-transformers embedding model."""
        if self._embed_model is None:
            st = _require_sentence_transformers()
            self._embed_model = st.SentenceTransformer(self._embedding_model)
        return self._embed_model

    def _embed(self, text: str) -> list[float]:
        model = self._get_embed_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

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
        """Generate a completion for *prompt* using a local HF model.

        Only newly generated tokens (after the prompt) are returned.
        """
        pipeline = self._get_gen_pipeline()
        result = pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        try:
            generated_text = result[0]["generated_text"]
        except (IndexError, KeyError, TypeError) as exc:
            logger.error("Unexpected pipeline output: %s", exc)
            return ""

        # Return only the newly generated part.
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):]
        return generated_text

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

    def _get_store(self) -> EmbeddingStore:
        if self._store is None:
            self._store = EmbeddingStore(
                persist_dir=self.embeddings_dir,
                embed_fn=self._embed,
            )
        return self._store
