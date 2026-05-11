# llmlib

Portability library for OpenAI, Hugging Face, Google Gemini, Fireworks.ai, Ollama, and local model APIs.

[![PyPI](https://img.shields.io/pypi/v/llmlib.svg)](https://pypi.org/project/llmlib/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mark-watson/llmlib/blob/master/LICENSE)

## Installation

Using `uv` (recommended):

```bash
# OpenAI provider only
uv add llmlib --optional openai

# Hugging Face provider only (local models)
uv add llmlib --optional huggingface

# All providers + ChromaDB embedding store
uv add llmlib --optional all

# Development dependencies (tests)
uv sync --extra all --group dev
```

Using `pip`:

```bash
pip install llmlib[all]
```

## Quickstart

Set your API key for the provider you want to use:

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export FIREWORKS_API_KEY="..."
export OLLAMA_API_KEY="..."   # optional for local Ollama
```

Run a completion:

```python
from llmlib import OpenAiWrapper

llm = OpenAiWrapper()
text = llm.get_completion("Once upon a time", max_tokens=64)
print(text)
```

Index local documents and search:

```python
llm.create_local_embeddings_files_in_dir("./data/")
results = llm.query_local_embeddings("definition of sports", n=3)
for r in results:
    print(r["score"], r["text"][:100])
```

## Supported Providers

| Provider | Class | Env Var | Default Model |
|---|---|---|---|
| OpenAI | `OpenAiWrapper` | `OPENAI_API_KEY` | `gpt-3.5-turbo` |
| Hugging Face | `HuggingFaceAiWrapper` | — | `facebook/opt-iml-1.3b` |
| Google Gemini | `GeminiWrapper` | `GOOGLE_API_KEY` | `gemini-3-flash-preview` |
| Fireworks.ai | `FireworksWrapper` | `FIREWORKS_API_KEY` | `llama-v3p1-8b-instruct` |
| Ollama | `OllamaWrapper` | `OLLAMA_API_KEY` (optional) | `llama3.2` |

## API Reference

### BaseModelWrapper (abstract)

- `get_completion(prompt, *, max_tokens=64, temperature=0.5, top_p=0.75) -> str`
- `create_local_embeddings_files_in_dir(path) -> None`
- `query_local_embeddings(query, n=10) -> list[dict]`

### OpenAiWrapper

- Constructor: `OpenAiWrapper(key=None, embeddings_dir="./db_embeddings", completion_model="gpt-3.5-turbo", embedding_model="text-embedding-3-small")`
- `get_chat_completion(messages, *, model=None, max_tokens=256, temperature=0.7, top_p=1.0) -> str`

### HuggingFaceAiWrapper

- Constructor: `HuggingFaceAiWrapper(embeddings_dir="./db_embeddings", generation_model="facebook/opt-iml-1.3b", embedding_model="sentence-transformers/all-MiniLM-L6-v2", device=-1, torch_dtype="bfloat16")`

### GeminiWrapper

- Constructor: `GeminiWrapper(key=None, embeddings_dir="./db_embeddings", model="gemini-3-flash-preview", embedding_model="text-embedding-004")`
- `get_completion_with_search(prompt, *, max_tokens=256, temperature=0.5, top_p=0.75) -> dict` — returns `{"text": str, "grounding": list}`
- `get_chat_completion(messages, *, max_tokens=256, temperature=0.7, top_p=1.0, search=False) -> str`

### FireworksWrapper

- Constructor: `FireworksWrapper(key=None, embeddings_dir="./db_embeddings", model="accounts/fireworks/models/llama-v3p1-8b-instruct", embedding_model="nomic-ai/nomic-embed-text-v1.5")`
- `get_chat_completion(messages, *, max_tokens=256, temperature=0.7, top_p=1.0) -> str`

### OllamaWrapper

- Constructor: `OllamaWrapper(key=None, embeddings_dir="./db_embeddings", model="llama3.2", embedding_model="nomic-embed-text", base_url="https://ollama.com")`
- `get_chat_completion(messages, *, max_tokens=256, temperature=0.7, top_p=1.0) -> str`
- Pass `base_url="http://localhost:11434"` for a local Ollama instance.

## Examples

See the `examples/` directory:

- `examples/basic_completion.py` — text completion with all providers.
- `examples/embeddings_search.py` — indexing and similarity search.
- `examples/gemini_search.py` — Gemini with Google Search grounding.
- `examples/fireworks_chat.py` — Fireworks.ai chat completion.
- `examples/ollama_chat.py` — Ollama chat completion.

Run them with `uv run`:

```bash
uv run examples/basic_completion.py
uv run examples/embeddings_search.py
uv run examples/gemini_search.py
uv run examples/fireworks_chat.py
uv run examples/ollama_chat.py
```

## Running Tests

```bash
uv run pytest
```

## License

Apache 2.0
