"""Example: Gemini completion with and without Google Search grounding."""

from __future__ import annotations


def demo_gemini_basic():
    print("--- Gemini Basic Completion ---")
    from llmlib import GeminiWrapper

    llm = GeminiWrapper()
    text = llm.get_completion("Explain quantum computing in one sentence.", max_tokens=128)
    print(text)
    print()


def demo_gemini_search():
    print("--- Gemini with Google Search Grounding ---")
    from llmlib import GeminiWrapper

    llm = GeminiWrapper()
    result = llm.get_completion_with_search(
        "What is the current weather in San Francisco?",
        max_tokens=256,
    )
    print("Answer:", result["text"])
    if result["grounding"]:
        print("Sources:")
        for chunk in result["grounding"]:
            print("  -", chunk)
    print()


def demo_gemini_chat():
    print("--- Gemini Chat ---")
    from llmlib import GeminiWrapper

    llm = GeminiWrapper()
    messages = [
        {"role": "user", "content": "What is the capital of Japan?"},
        {"role": "assistant", "content": "Tokyo."},
        {"role": "user", "content": "What is its population?"},
    ]
    text = llm.get_chat_completion(messages, max_tokens=128)
    print(text)
    print()


if __name__ == "__main__":
    demo_gemini_basic()
    # demo_gemini_search()
    # demo_gemini_chat()
