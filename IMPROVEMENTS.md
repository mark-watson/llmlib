# llmlib Improvement Plan

## 1. Architecture Improvements

### 1.1 Package Structure
- [x] Add `llmlib/__init__.py` to expose public API cleanly via `__all__`.
- [x] Rename internal modules to avoid name shadowing:
  - `llmlib/openai.py` -> `llmlib/openai_provider.py` (avoids shadowing the `openai` package).
  - `llmlib/huggingface.py` -> `llmlib/huggingface_provider.py`.
- [x] Create `llmlib/utils.py` for shared helpers (text chunking, safe JSON parsing, cosine similarity, HTTP JSON POST).

### 1.2 Base Class
- [x] Convert `BaseModelWrapper` to a proper abstract base class (`abc.ABC`).
- [x] Fix the `__init__` bug where assignments were missing `self.`.
- [x] Add type hints to all method signatures.
- [x] Add `NotImplementedError` to abstract methods.

### 1.3 Embedding Store Abstraction
- [x] Decouple embedding storage from provider wrappers.
- [x] Introduce a lightweight `EmbeddingStore` class backed by **ChromaDB** (used directly, not through LangChain).
- [x] Remove heavy LangChain and LlamaIndex dependencies; replace with:
  - `sentence-transformers` for local embedding models.
  - `transformers` + `torch` for HF text generation.
  - `openai>=1.0` for OpenAI text generation and embeddings.

### 1.4 Provider Modernization
- [x] **OpenAI**: Migrate from deprecated `openai.Completion.create` to the `openai.OpenAI` client (`chat.completions.create`).
- [x] **HuggingFace**: Use a dedicated `pipeline("text-generation", ...)` instance and `sentence-transformers` for embeddings instead of llama_index/LangChain wrappers.

## 2. Coding Improvements

### 2.1 Type Safety & Style
- [x] Add Python 3.9+ type hints (`list[str]`, `dict[str, Any]`, etc.).
- [x] Use `logging` instead of `pprint`/`print` for debug output.
- [x] Standardize on PEP8 naming and formatting.

### 2.2 Error Handling
- [x] Replace bare `except Exception` with specific exceptions.
- [x] Validate API keys at initialization with a clear `ValueError`.
- [x] Handle missing embedding directories gracefully.

### 2.3 API Consistency
- [x] Make `create_local_embeddings_files_in_dir` idempotent and able to incrementally add files.
- [x] Ensure `query_local_embeddings` returns a structured result (list of dicts with `text` and `score`).
- [x] Add `get_chat_completion(messages, ...)` to all chat-capable providers.

### 2.4 Metadata Fixes
- [x] Synchronize `setup.py` version with `pyproject.toml`.
- [x] Update `requires-python` to `>=3.9`.
- [x] Fix license mismatch (standardize on Apache 2.0).
- [x] Add missing dependencies to `install_requires` / `extras_require`.

## 3. Documentation Improvements

### 3.1 README.md
- [x] Rewrite with clear installation steps (`pip install llmlib[openai]`, `llmlib[huggingface]`, `llmlib[all]`).
- [x] Add a "Quickstart" section.
- [x] Document environment variables (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `FIREWORKS_API_KEY`, `OLLAMA_API_KEY`).
- [x] Add API reference tables for all providers.

### 3.2 Docstrings
- [x] Add Google-style docstrings to every public class and method.
- [x] Document return types and raised exceptions.

## 4. Examples & Testing Improvements

### 4.1 Examples Directory
- [x] Create `examples/` and move all current test scripts there.
- [x] Add `examples/basic_completion.py` demonstrating all providers.
- [x] Add `examples/embeddings_search.py` demonstrating indexing and similarity search.
- [x] Add `examples/gemini_search.py` for Google Search grounding.
- [x] Add `examples/fireworks_chat.py` for Fireworks chat completion.
- [x] Add `examples/ollama_chat.py` for Ollama chat completion.

### 4.2 Test Suite
- [x] Create `tests/` directory with `pytest` tests.
- [x] Add unit tests for `BaseModelWrapper` instantiation and `EmbeddingStore` chunking.
- [x] Add mock-based tests for OpenAI provider initialization (no API key needed).
- [x] Add mock-based tests for Gemini, Fireworks, and Ollama providers via `urllib` monkeypatching.

### 4.3 Cleanup
- [x] Remove `llmlib/README.md` (duplicate of root).
- [x] Update `.gitignore` to ignore `db_embeddings/`, `__pycache__/`, `*.egg-info/`.
- [x] Delete obsolete `test_*.py` files from the project root.

## 5. New Providers

### 5.1 Google Gemini
- [x] `llmlib/gemini_provider.py` — `GeminiWrapper` class.
- [x] Uses Gemini REST API (`generateContent`, `embedContent`).
- [x] Supports `GOOGLE_API_KEY` env var.
- [x] `get_completion` without search grounding.
- [x] `get_completion_with_search` with Google Search grounding (returns `{"text": ..., "grounding": [...]}`).
- [x] `get_chat_completion` for multi-turn chat.
- [x] Reuses `EmbeddingStore` for local document indexing.

### 5.2 Fireworks.ai
- [x] `llmlib/fireworks_provider.py` — `FireworksWrapper` class.
- [x] Uses Fireworks OpenAI-compatible REST API (`/chat/completions`, `/embeddings`).
- [x] Supports `FIREWORKS_API_KEY` env var.
- [x] `get_completion` (single-turn) and `get_chat_completion` (multi-turn).
- [x] Reuses `EmbeddingStore` for local document indexing.

### 5.3 Ollama (Cloud or Local)
- [x] `llmlib/ollama_provider.py` — `OllamaWrapper` class.
- [x] Uses native Ollama REST API (`/api/generate`, `/api/chat`, `/api/embed`).
- [x] Supports `OLLAMA_API_KEY` env var (optional for local use).
- [x] `get_completion` via `/api/generate`.
- [x] `get_chat_completion` via `/api/chat`.
- [x] Configurable `base_url` (default `https://ollama.com`, local can be `http://localhost:11434`).
- [x] Reuses `EmbeddingStore` for local document indexing.
