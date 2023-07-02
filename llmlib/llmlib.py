# Main entry for llmlib

import os

# Default to OpenAI API

API_TYPES = ["openai", "fastchat", "huggingface"]
API_TYPE = os.getenv("LLMLIB_API_TYPE", "openai")

if API_TYPE == "openai":
    key = os.getenv("OPENAI_API_KEY")
    if key is None:
        raise Exception("OPENAI_API_KEY environment variable not set")
    from llmlib import openai
    openai.api_key = key
elif API_TYPE == "fastchat":
    from llmlib import fastchat
    pass
elif API_TYPE == "huggingface":
    from llmlib import huggingface
    pass
else:
    raise Exception(f"Unknown API type: {API_TYPE}")

