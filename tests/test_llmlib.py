"""Unit tests for llmlib core components."""

from __future__ import annotations

import pytest

from llmlib.base_model_wrapper import BaseModelWrapper
from llmlib.utils import chunk_text, cosine_similarity, safe_json_loads


class DummyWrapper(BaseModelWrapper):
    """Minimal concrete implementation for ABC testing."""

    def get_completion(self, prompt, *, max_tokens=64, temperature=0.5, top_p=0.75):
        return f"echo: {prompt}"

    def create_local_embeddings_files_in_dir(self, path):
        pass

    def query_local_embeddings(self, query, n=10):
        return [{"text": query, "score": 1.0, "metadata": {}}]


def test_base_model_wrapper_init():
    wrapper = DummyWrapper(embeddings_dir="./test_db")
    assert wrapper.embeddings_dir == "./test_db"


def test_chunk_text_basic():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=512, overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 512 for c in chunks)


def test_chunk_text_bad_args():
    with pytest.raises(ValueError, match="positive"):
        chunk_text("foo", chunk_size=0)
    with pytest.raises(ValueError, match="non-negative"):
        chunk_text("foo", chunk_size=10, overlap=-1)
    with pytest.raises(ValueError, match="smaller than"):
        chunk_text("foo", chunk_size=10, overlap=10)


def test_cosine_similarity_identical():
    vec = [1.0, 2.0, 3.0]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_safe_json_loads_valid():
    assert safe_json_loads('{"a": 1}') == {"a": 1}


def test_safe_json_loads_invalid():
    assert safe_json_loads("not json", default="fallback") == "fallback"


def test_dummy_wrapper_completion():
    w = DummyWrapper()
    assert w.get_completion("hi") == "echo: hi"


def test_dummy_wrapper_query():
    w = DummyWrapper()
    results = w.query_local_embeddings("sports")
    assert len(results) == 1
    assert results[0]["text"] == "sports"


class TestOpenAiWrapperInit:
    """Mock-based tests for OpenAiWrapper (no real API calls)."""

    def test_raises_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from llmlib.openai_provider import OpenAiWrapper

        with pytest.raises(ValueError, match="OpenAI API key"):
            OpenAiWrapper()

    def test_accepts_explicit_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        from llmlib.openai_provider import OpenAiWrapper

        # openai.OpenAI is mocked to avoid real network calls.
        import openai
        original = openai.OpenAI
        calls = []

        class FakeClient:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr(openai, "OpenAI", FakeClient)
        try:
            wrapper = OpenAiWrapper(key="sk-fake")
            assert wrapper._completion_model == "gpt-3.5-turbo"
            assert calls == [{"api_key": "sk-fake"}]
        finally:
            monkeypatch.setattr(openai, "OpenAI", original)


class TestHuggingFaceAiWrapperInit:
    """Smoke tests for HuggingFaceAiWrapper instantiation."""

    def test_init_defaults(self):
        from llmlib.huggingface_provider import HuggingFaceAiWrapper

        wrapper = HuggingFaceAiWrapper()
        assert wrapper._generation_model == "facebook/opt-iml-1.3b"
        assert wrapper._embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert wrapper._device == -1

    def test_import_guard_openai(self):
        """Ensure openai provider import is lazy (we already imported it above)."""
        import llmlib.openai_provider as oap
        assert oap is not None
