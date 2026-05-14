# verification_engine/agent/llm.py
import os
from dotenv import load_dotenv
from groq import Groq
from typing import Optional

# Load environment variables from .env file
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
                # For demo mode, don't fail - just use fallback
                print("⚠️ GROQ_API_KEY not set. Using fallback verification (demo mode).")
                self._client = None
                return
            self._client = Groq(api_key=api_key)

    @property
    def client(self) -> Optional[Groq]:
        """Get the Groq client instance (may be None in demo mode)"""
        self._ensure_client()
        return self._client


def get_llm_client() -> Optional[Groq]:
    """Get LLM client or None if not configured"""
    return LLMClient().client


def get_llm_model() -> str:
    """Get the configured LLM model name"""
    from verification_engine.agent.config import GROQ_MODEL
    return GROQ_MODEL
