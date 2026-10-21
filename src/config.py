"""Configuration module for Gemini API development.

Automatically loads environment variables from .env and exposes standard model constants
and client factory methods using the modern google-genai SDK.
"""

from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from google import genai

# Automatically load environment variables at import time
_dotenv_path = find_dotenv(usecwd=True)
if not _dotenv_path:
    _workspace_env = Path(__file__).resolve().parent.parent / ".env"
    if _workspace_env.exists():
        _dotenv_path = str(_workspace_env)
    else:
        _parent_env = Path(__file__).resolve().parent.parent.parent / ".env"
        if _parent_env.exists():
            _dotenv_path = str(_parent_env)

if _dotenv_path:
    load_dotenv(_dotenv_path)
else:
    load_dotenv()

# Standard 2026 model constants
FLASH_MODEL = "gemini-3.8-flash"
REASONING_MODEL = "gemini-4-argon"
LITE_MODEL = "gemini-3.5-flash-lite"
ANTIGRAVITY_AGENT = "antigravity-preview-09-2026"


def get_client() -> genai.Client:
    """Return an initialized Google GenAI Client instance.

    Uses environment variables GEMINI_API_KEY or GOOGLE_API_KEY automatically.
    """
    return genai.Client()


__all__ = [
    "ANTIGRAVITY_AGENT",
    "FLASH_MODEL",
    "LITE_MODEL",
    "REASONING_MODEL",
    "get_client",
]
