from llmlib.huggingface import HuggingFaceAiWrapper

llm = HuggingFaceAiWrapper()

gen_text = llm.get_completion("Once upon a time", max_tokens=64)
print(gen_text)