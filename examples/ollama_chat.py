"""Example: Ollama chat and generation (local or cloud)."""

from __future__ import annotations


def demo_ollama_completion():
    print("--- Ollama Generate ---")
    from llmlib import OllamaWrapper

    llm = OllamaWrapper()
    text = llm.get_completion("Once upon a time", max_tokens=64)
    print(text)
    print()


def demo_ollama_chat():
    print("--- Ollama Chat ---")
    from llmlib import OllamaWrapper

    llm = OllamaWrapper()
    messages = [
        {"role": "user", "content": "What is the capital of Spain?"},
        {"role": "assistant", "content": "Madrid."},
        {"role": "user", "content": "What is the population of Madrid?"},
    ]
    text = llm.get_chat_completion(messages, max_tokens=128)
    print(text)
    print()


def demo_ollama_local():
    print("--- Ollama Local Instance ---")
    from llmlib import OllamaWrapper

    llm = OllamaWrapper(base_url="http://localhost:11434")
    text = llm.get_completion("Tell me a joke.", max_tokens=64)
    print(text)
    print()


if __name__ == "__main__":
    demo_ollama_completion()
    # demo_ollama_chat()
    # demo_ollama_local()
