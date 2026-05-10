"""Example: index local documents and perform similarity search."""

from __future__ import annotations


def demo_openai_embeddings():
    print("--- OpenAI Embeddings ---")
    from llmlib import OpenAiWrapper

    llm = OpenAiWrapper()
    llm.create_local_embeddings_files_in_dir("./data/")

    for query in ["definition of sports", "activities in sports"]:
        print(f"Query: {query!r}")
        results = llm.query_local_embeddings(query, n=3)
        for r in results:
            print(f"  score={r['score']:.3f} | text={r['text'][:120]!r}...")
        print()


def demo_hf_embeddings():
    print("--- Hugging Face Embeddings ---")
    from llmlib import HuggingFaceAiWrapper

    llm = HuggingFaceAiWrapper()
    llm.create_local_embeddings_files_in_dir("./data/")

    for query in ["definition of sports", "activities in sports"]:
        print(f"Query: {query!r}")
        results = llm.query_local_embeddings(query, n=3)
        for r in results:
            print(f"  score={r['score']:.3f} | text={r['text'][:120]!r}...")
        print()


if __name__ == "__main__":
    demo_openai_embeddings()
    # demo_hf_embeddings()  # Uncomment if you have the model downloaded locally.
