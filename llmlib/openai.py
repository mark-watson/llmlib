# Wrapper for OpenAI APIs for llmlib

import openai
import os
from llmlib.base_model_wrapper import BaseModelWrapper

class OpenAiWrapper(BaseModelWrapper):
    def __init__(self, key=None):
        super().__init__()
        if key is None:
            key = os.getenv("OPENAI_API_KEY")
            if key is None:
                raise Exception("OPENAI_API_KEY environment variable not set")
        openai.api_key = key

  #  generated boilerplate, probably discard:
    def get_completion(self, prompt, max_tokens=64, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, stop=None, n=1, stream=False, logprobs=None, echo=False, engine="davinci", **kwargs):  
        return openai.Completion.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, stop=stop, n=n, stream=stream, logprobs=logprobs, echo=echo, engine=engine, **kwargs)