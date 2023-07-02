# Main entry for llmlib

import os

# Default to OpenAI API

API_TYPES = ["openai", "fastchat", "huggingface"]
API_TYPE = os.getenv("LLMLIB_API_TYPE", "openai")

global llm_wrapper

if API_TYPE == "openai":
    llm_wrapper = openai.OpenAiWrapper()
elif API_TYPE == "fastchat":
    from llmlib import fastchat
    pass
elif API_TYPE == "huggingface":
    from llmlib import huggingface
    pass
else:
    raise Exception(f"Unknown API type: {API_TYPE}")

