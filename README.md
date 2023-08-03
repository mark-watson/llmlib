# Portability lib for OpenAI, Hugging Face, and FastChat local model APIs

[![PyPI](https://img.shields.io/pypi/v/llmlib.svg)](https://pypi.org/project/llmlib/)
[![Changelog](https://img.shields.io/github/v/release/mark-watson/llmlib?include_prereleases&label=changelog)](https://github.com/mark-watson/llmlib/releases)
[![Tests](https://github.com/mark-watson/llmlib/workflows/Test/badge.svg)](https://github.com/mark-watson/llmlib/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mark-watson/llmlib/blob/master/LICENSE)

## This library was written for examples in a book I wrote

You can read my book free online or can be purchased:

[Safe For Humans AI - A "humans-first" approach to designing and building AI systems](https://leanpub.com/safe-for-humans-AI/read)

## Supported APIs

- get_completion(prompt, max_tokens=64)
- create_local_embeddings_files_in_dir(path_to_document_files_dir)
- query_local_embeddings(query)

These APIs are implemented separately in two Python classes:

- OpenAiWrapper
- HuggingFaceAiWrapper

## Supported APIs and Models

Currently this compatibiolity library only works with selected OpenAI APIs and the small Hugging Face model **facebook/opt-iml-1.3b** that can be run locally on a laptop with no GPU and only 8G of RAM.


Run the example programs:

    python test_completion_openai.py
    python test_local_index_openai.py
    python test_completion_hf.py
    python test_local_index_hf.py

