
class BaseModelWrapper():
    def __init__(self, embeddings_dir="./db_embeddings"):
        llm = None
        embeddings = None
        embeddings_dir = embeddings_dir
    # complete text:
    def get_completion(self, prompt, max_tokens=64):
        pass
    # create local embeddings:
    def create_local_embeddings_files_in_dir(self, path):
        " path is a directory "
        pass
    # query local embeddings:
    def query_local_embeddings(self, query, n=10):
        pass