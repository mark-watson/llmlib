"""Example: basic text completion with all providers."""

from __future__ import annotations


def demo_openai():
    print("--- OpenAI ---")
    from llmlib import OpenAiWrapper

    llm = OpenAiWrapper()
    text = llm.get_completion("Once upon a time", max_tokens=64)
    print(text)
    print()


def demo_huggingface():
    print("--- Hugging Face (local) ---")
    from llmlib import HuggingFaceAiWrapper

    llm = HuggingFaceAiWrapper()
    text = llm.get_completion("Once upon a time", max_tokens=64)
    print(text)
    print()


def demo_gemini():
    print("--- Google Gemini ---")
    from llmlib import GeminiWrapper

    llm = GeminiWrapper()
    text = llm.get_completion("Once upon a time", max_tokens=64)
    print(text)
    print()


def demo_fireworks():
    print("--- Fireworks.ai ---")
    from llmlib import FireworksWrapper

    llm = FireworksWrapper()
    text = llm.get_completion("Once upon a time", max_tokens=64)
    print(text)
    print()


def demo_ollama():
    print("--- Ollama ---")
    from llmlib import OllamaWrapper

    llm = OllamaWrapper()
    text = llm.get_completion("Once upon a time", max_tokens=64)
    print(text)
    print()


if __name__ == "__main__":
    demo_openai()
    # demo_huggingface()  # Uncomment if you have the model downloaded locally.
    # demo_gemini()       # Uncomment if GOOGLE_API_KEY is set.
    # demo_fireworks()    # Uncomment if FIREWORKS_API_KEY is set.
    # demo_ollama()       # Uncomment if Ollama is available.
