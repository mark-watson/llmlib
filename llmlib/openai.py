# Wrapper for OpenAI APIs for llmlib

import openai
import os
from llmlib.base_model_wrapper import BaseModelWrapper

class OpenAiWrapper(BaseModelWrapper):
    def __init__(self, key=None, embeddings_dir="./db_embeddings"):
        super().__init__(embeddings_dir=embeddings_dir)
        if key is None:
            key = os.getenv("OPENAI_API_KEY")
            if key is None:
                raise Exception("OPENAI_API_KEY environment variable not set")
        openai.api_key = key

    # complete text:
    def get_completion(self, prompt, max_tokens=64):
        c = openai.Completion.create(prompt=prompt, max_tokens=max_tokens,
                                     temperature=0.5, top_p=0.75,
                                     model="text-davinci-003")
        try:
            return c["choices"][0]["text"]
        except Exception as e:
            print(e)
            return ""
    # create local embeddings:
    def create_local_embeddings(self, texts):
        " texts is a list of strings "
        pass
    def create_local_embeddings_files_in_dir(self, path):
        " path is a directory "
        pass
    # query local embeddings:
    def query_local_embeddings(self, query, n=10):
        pass
