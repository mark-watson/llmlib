from llmlib.openai import OpenAiWrapper 

openai = OpenAiWrapper()

openai.create_local_embeddings(["Once upon a time", "The dog and cat ran away"])

a = openai.query_local_embeddings("dogs and cats", n=10)  # returns a list of tuples
print(a)