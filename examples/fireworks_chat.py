"""Example: Fireworks.ai chat completion."""

from __future__ import annotations


def demo_fireworks_completion():
    print("--- Fireworks Completion ---")
    from llmlib import FireworksWrapper

    llm = FireworksWrapper()
    text = llm.get_completion("Write a haiku about AI.", max_tokens=64)
    print(text)
    print()


def demo_fireworks_chat():
    print("--- Fireworks Chat ---")
    from llmlib import FireworksWrapper

    llm = FireworksWrapper()
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to reverse a list."},
    ]
    text = llm.get_chat_completion(messages, max_tokens=256)
    print(text)
    print()


if __name__ == "__main__":
    demo_fireworks_completion()
    # demo_fireworks_chat()
