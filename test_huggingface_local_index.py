from llmlib.huggingface import HuggingFaceAiWrapper 

hf = HuggingFaceAiWrapper()

hf.create_local_embeddings_files_in_dir("./data/")

a = hf.query_local_embeddings("definition of sports")
print(a)

a = hf.query_local_embeddings("activities in sports")
print(a)