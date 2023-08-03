# Wrapper for Hugging Face APIs for llmlib

from llmlib.base_model_wrapper import BaseModelWrapper

from llama_index import ListIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index import ListIndex, PromptHelper, SimpleDirectoryReader
from transformers import pipeline

import torch
from langchain.llms.base import LLM
from llama_index import LLMPredictor

from pprint import pprint

class CustomLLM(LLM):
    model_name = "facebook/opt-iml-1.3b"
    # I am not using a GPU, but you can add device="cuda:0"
    # to the pipeline call if you have a local GPU or
    # are running this on Google Colab:
    pipeline = pipeline("text-generation",
                        model=model_name,
                        model_kwargs={"torch_dtype":torch.bfloat16})

    def _call(self, prompt, stop = None):
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_new_tokens=200)
        pprint(response)
        first_response = response[0]["generated_text"]
        # only return newly generated tokens
        returned_text = first_response[prompt_length:]
        return returned_text

    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self):
        return "custom"

class HuggingFaceAiWrapper(BaseModelWrapper):
    def __init__(self, key=None, embeddings_dir="./db_embeddings"):
        super().__init__(embeddings_dir=embeddings_dir)
        self.llm_predictor = LLMPredictor(llm=CustomLLM())
        self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
        self.service_context = \
          ServiceContext.from_defaults(llm_predictor=self.llm_predictor,
                                       embed_model=self.embed_model)
        max_input_size = 512
        num_output = 64
        max_chunk_overlap = 0 # 10
        self.prompt_helper = PromptHelper(max_input_size, num_output,
                                          max_chunk_overlap)
        self.pipeline = None

    # complete text:
    def get_completion(self, prompt, max_tokens=64):
        if self.pipeline is None:
            self.pipeline = pipeline("text-generation",
                                     model="facebook/opt-iml-1.3b",
                                     model_kwargs={"torch_dtype":torch.bfloat16})   
        c = self.pipeline(prompt, max_new_tokens=max_tokens)
        pprint(c)
        try:
            return c[0]["generated_text"]
        except Exception as e:
            print(e)
            return ""
        

    def create_local_embeddings_files_in_dir(self, path):
        " path is a directory "
        self.documents = SimpleDirectoryReader(path).load_data()
        self.index = ListIndex.from_documents(documents=self.documents, 
                                              llm_predictor=self.llm_predictor,
                                              prompt_helper=self.prompt_helper)
        self.index = self.index.as_query_engine(llm_predictor=self.llm_predictor)

    # query local embeddings:
    def query_local_embeddings(self, query, n=10):
        answer = self.index.query(query)
        return answer