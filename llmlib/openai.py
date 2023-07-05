# Wrapper for OpenAI APIs for llmlib

import openai
import os
from llmlib.base_model_wrapper import BaseModelWrapper
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

class OpenAiWrapper(BaseModelWrapper):
    def __init__(self, key=None, embeddings_dir="./db_embeddings"):
        super().__init__(embeddings_dir=embeddings_dir)
        if key is None:
            key = os.getenv("OPENAI_API_KEY")
            if key is None:
                raise Exception("OPENAI_API_KEY environment variable not set")
        #self.openai.api_key = key
        self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.embeddings_dir = embeddings_dir
        self.db = None
        self.index = None

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
    def _init_index(self):
        if self.embedding is None:
            self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        if self.index is None:
            self.index = VectorstoreIndexCreator(vectorstore_kwargs={
                "persist_directory": self.embeddings_dir,
                "vectorstore_type": "chromadb"})

    def create_local_embeddings(self, texts):
        " texts is a list of strings "
        self._init_index()
        self.index.add_texts(texts)
        self.retriever = None
        self.qa = None

    def create_local_embeddings_files_in_dir(self, path):
        " path is a directory "
        self._init_index()
        loader = DirectoryLoader(path, glob="**/*.md")
        self.from_loaders([loader])
    # query local embeddings:
    def query_local_embeddings(self, query, n=10):
        if self.index is None:
            self.index = VectorstoreIndexCreator(vectorstore_kwargs={
                "persist_directory": self.embeddings_dir,
                "vectorstore_type": "chromadb"})
        if self.db is None:
            self.db = Chroma(persist_directory=self.embeddings_dir,
                             embedding_function=self.embedding)
        if self.retriever is None:
            self.retriever = self.db.as_retriever()
        if self.qa is None:
            self.qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                                  chain_type="stuff",
                                                  retriever=self.retriever)
        answer = self.qa.run(query)