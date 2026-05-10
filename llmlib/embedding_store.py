"""Lightweight embedding store backed by ChromaDB."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from llmlib.utils import chunk_text

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Index text documents and query them via vector similarity."""

    def __init__(
        self,
        persist_dir: str,
        embed_fn: Callable[[str], list[float]],
        collection_name: str = "llmlib_documents",
    ) -> None:
        self._persist_dir = persist_dir
        self._embed_fn = embed_fn
        self._collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None

    def _ensure_client(self) -> None:
        """Lazy-init ChromaDB persistent client."""
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for EmbeddingStore. Install it with: "
                "pip install llmlib[chromadb]"
            ) from exc

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_directory(self, path: str, glob: str = "**/*") -> None:
        """Read all files under *path* matching *glob*, chunk them, and index.

        Args:
            path: Root directory to walk.
            glob: Pattern passed to :class:`pathlib.Path.rglob`.
        """
        self._ensure_client()
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        file_count = 0
        chunk_count = 0

        for file_path in root.rglob(glob):
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", file_path, exc)
                continue

            file_count += 1
            chunks = chunk_text(text, chunk_size=512, overlap=50)
            for idx, chunk in enumerate(chunks):
                docs.append(chunk)
                ids.append(f"{file_path.resolve()}_{idx}")
                metadatas.append({"source": str(file_path.resolve()), "chunk": idx})
                chunk_count += 1

        if not docs:
            logger.warning("No documents found in %s (glob=%s)", path, glob)
            return

        logger.info("Embedding %d chunks from %d files...", chunk_count, file_count)
        embeddings = [self._embed_fn(doc) for doc in docs]
        self._collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Indexed %d chunks.", chunk_count)

    def query(self, query: str, n: int = 10) -> list[dict[str, Any]]:
        """Return the top-*n* most similar chunks to *query*.

        Args:
            query: Query string.
            n: Number of results.

        Returns:
            List of dicts with keys ``text``, ``score``, ``metadata``.
        """
        self._ensure_client()
        query_embedding = self._embed_fn(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "distances", "metadatas"],
        )

        out: list[dict[str, Any]] = []
        # ChromaDB returns results grouped by query; we only have one query.
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, dist, meta in zip(documents, distances, metadatas):
            # Chroma cosine distance is 1 - cosine_similarity; convert to similarity.
            score = 1.0 - float(dist)
            out.append({"text": doc, "score": score, "metadata": meta})
        return out

    def clear(self) -> None:
        """Drop the current collection and recreate it."""
        self._ensure_client()
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
