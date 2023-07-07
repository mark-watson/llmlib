from llmlib.openai import OpenAiWrapper 

openai = OpenAiWrapper()

gen_text = openai.get_completion("Once upon a time", max_tokens=64)
print(gen_text)