from llmlib.openai import OpenAiWrapper 

openai = OpenAiWrapper()

openai.create_local_embeddings_files_in_dir("./data/")

a = openai.query_local_embeddings("definition of sports")
print(a)

a = openai.query_local_embeddings("activities in sports")
print(a)