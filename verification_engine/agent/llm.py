"""LLM client initialization - Groq (Llama 3.3 70B)"""

import os
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """Singleton wrapper for Groq LLM client"""

    _instance: Optional['LLMClient'] = None
    _client: Optional[Groq] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._ensure_client()

    def _ensure_client(self):
        """Lazy initialize the Groq client"""
        if self._client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            self._client = Groq(api_key=api_key)

    @property
    def client(self) -> Groq:
        """Get the Groq client instance"""
        self._ensure_client()
        return self._client


def get_llm_client() -> Groq:
    """Convenience function to get LLM client"""
    return LLMClient().client


def get_llm_model() -> str:
    """Get the configured LLM model name"""
    from verification_engine.agent.config import GROQ_MODEL
    return GROQ_MODEL