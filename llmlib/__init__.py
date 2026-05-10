"""llmlib — portability library for OpenAI, Hugging Face, and local model APIs."""
from __future__ import annotations

from llmlib.base_model_wrapper import BaseModelWrapper
from llmlib.fireworks_provider import FireworksWrapper
from llmlib.gemini_provider import GeminiWrapper
from llmlib.huggingface_provider import HuggingFaceAiWrapper
from llmlib.ollama_provider import OllamaWrapper
from llmlib.openai_provider import OpenAiWrapper

__version__ = "0.3.0"
__all__ = [
    "BaseModelWrapper",
    "OpenAiWrapper",
    "HuggingFaceAiWrapper",
    "GeminiWrapper",
    "FireworksWrapper",
    "OllamaWrapper",
]
