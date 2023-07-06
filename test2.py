from llmlib.openai import OpenAiWrapper 

openai = OpenAiWrapper()

openai.create_local_embeddings_files_in_dir("./data/")
a = openai.query_local_embeddings("definition of sports", n=10)  # returns a list of tuples
print(a)