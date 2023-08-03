# Portability lib for OpenAI, Hugging Face, and FastChat local model APIs

    pip install -U langchain llama_index torch transformers openai

## Supported APIs

- get_completion(prompt, max_tokens=64)
- create_local_embeddings_files_in_dir(path_to_document_files_dir)
- query_local_embeddings(query)

These APIs are implemented separately in two Python classes:

- OpenAiWrapper
- HuggingFaceAiWrapper

## Supported APIs and Models

Currently this compatibiolity library only works with selected OpenAI APIs and the small Hugging Face model **facebook/opt-iml-1.3b** that can be run locally on a laptop with no GPU and only 8G of RAM.
