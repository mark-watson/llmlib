"""Mock-based tests for the new HTTP-backed providers."""

from __future__ import annotations

import json
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from llmlib.gemini_provider import GeminiWrapper
from llmlib.fireworks_provider import FireworksWrapper
from llmlib.ollama_provider import OllamaWrapper


class MockResponse:
    """Minimal stand-in for urllib.request.urlopen response."""

    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ------------------------------------------------------------------
# Gemini
# ------------------------------------------------------------------

@patch("llmlib.utils.urllib.request.urlopen")
def test_gemini_completion(mock_urlopen):
    mock_urlopen.return_value = MockResponse({
        "candidates": [{"content": {"parts": [{"text": "Hello world"}]}}]
    })
    wrapper = GeminiWrapper(key="FAKE_KEY")
    text = wrapper.get_completion("Say hello")
    assert text == "Hello world"


@patch("llmlib.utils.urllib.request.urlopen")
def test_gemini_completion_with_search(mock_urlopen):
    mock_urlopen.return_value = MockResponse({
        "candidates": [{
            "content": {"parts": [{"text": "It is blue"}]},
            "groundingMetadata": {
                "groundingChunks": [{"web": {"uri": "https://example.com"}}]
            },
        }]
    })
    wrapper = GeminiWrapper(key="FAKE_KEY")
    result = wrapper.get_completion_with_search("Why is the sky blue?")
    assert result["text"] == "It is blue"
    assert result["grounding"][0]["web"]["uri"] == "https://example.com"


@patch("llmlib.utils.urllib.request.urlopen")
def test_gemini_chat(mock_urlopen):
    mock_urlopen.return_value = MockResponse({
        "candidates": [{"content": {"parts": [{"text": "42"}]}}]
    })
    wrapper = GeminiWrapper(key="FAKE_KEY")
    text = wrapper.get_chat_completion([
        {"role": "user", "content": "What is 6*7?"}
    ])
    assert text == "42"


@patch("llmlib.utils.urllib.request.urlopen")
def test_gemini_embed(mock_urlopen):
    mock_urlopen.return_value = MockResponse({
        "embedding": {"values": [0.1, 0.2, 0.3]}
    })
    wrapper = GeminiWrapper(key="FAKE_KEY")
    emb = wrapper._embed("hello")
    assert emb == [0.1, 0.2, 0.3]


# ------------------------------------------------------------------
# Fireworks
# ------------------------------------------------------------------

@patch("llmlib.utils.urllib.request.urlopen")
def test_fireworks_completion(mock_urlopen):
    mock_urlopen.return_value = MockResponse({
        "choices": [{"message": {"content": "I am fine"}}]
    })
    wrapper = FireworksWrapper(key="FAKE_KEY")
    text = wrapper.get_completion("How are you?")
    assert text == "I am fine"


@patch("llmlib.utils.urllib.request.urlopen")
def test_fireworks_chat(mock_urlopen):
    mock_urlopen.return_value = MockResponse({
        "choices": [{"message": {"content": "Sure!"}}]
    })
    wrapper = FireworksWrapper(key="FAKE_KEY")
    text = wrapper.get_chat_completion([
        {"role": "user", "content": "Help me"}
    ])
    assert text == "Sure!"


@patch("llmlib.utils.urllib.request.urlopen")
def test_fireworks_embed(mock_urlopen):
    mock_urlopen.return_value = MockResponse({
        "data": [{"embedding": [0.5, 0.6]}]
    })
    wrapper = FireworksWrapper(key="FAKE_KEY")
    emb = wrapper._embed("hello")
    assert emb == [0.5, 0.6]


# ------------------------------------------------------------------
# Ollama
# ------------------------------------------------------------------

@patch("urllib.request.urlopen")
def test_ollama_completion(mock_urlopen):
    mock_urlopen.return_value = MockResponse({
        "response": "The capital of France is Paris."
    })
    wrapper = OllamaWrapper()
    text = wrapper.get_completion("What is the capital of France?")
    assert text == "The capital of France is Paris."


@patch("urllib.request.urlopen")
def test_ollama_chat(mock_urlopen):
    mock_urlopen.return_value = MockResponse({
        "message": {"role": "assistant", "content": "Paris"}
    })
    wrapper = OllamaWrapper()
    text = wrapper.get_chat_completion([
        {"role": "user", "content": "Capital of France?"}
    ])
    assert text == "Paris"


@patch("urllib.request.urlopen")
def test_ollama_embed(mock_urlopen):
    mock_urlopen.return_value = MockResponse({
        "embeddings": [[0.1, 0.2, 0.3]]
    })
    wrapper = OllamaWrapper()
    emb = wrapper._embed("hello")
    assert emb == [0.1, 0.2, 0.3]


@patch("urllib.request.urlopen")
def test_ollama_local_no_key(mock_urlopen):
    """Local Ollama should work with no API key."""
    mock_urlopen.return_value = MockResponse({
        "response": "42"
    })
    wrapper = OllamaWrapper(base_url="http://localhost:11434")
    assert wrapper._key == ""
    text = wrapper.get_completion("Life, universe, everything")
    assert text == "42"


# ------------------------------------------------------------------
# Init validation
# ------------------------------------------------------------------

def test_gemini_init_raises_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Google API key"):
        GeminiWrapper()


def test_fireworks_init_raises_without_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Fireworks API key"):
        FireworksWrapper()


def test_ollama_init_allows_no_key(monkeypatch):
    """Ollama does not require an API key (local inference)."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    wrapper = OllamaWrapper()
    assert wrapper._key == ""
