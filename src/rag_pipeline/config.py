"""Configuration and API key setup for the financial assistant."""

import os
import pathlib
from typing import Dict
from dotenv import load_dotenv


# Constants
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_dbs")
MEMORY_DIR = os.path.join(PROJECT_ROOT, "conversation_history")
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
PAPER_TRADING = True

# Memory configuration
MAX_CONVERSATION_HISTORY = 10
DEFAULT_MEMORY_FILE = os.path.join(MEMORY_DIR, "conversation_history.json")

# Create vector DB directory if it doesn't exist
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)


def get_api_keys() -> Dict[str, str]:
    """
    Get API keys from environment.
    
    Returns:
        Dictionary with required API keys
    
    Raises:
        ValueError: If any required keys are missing
    """
    load_dotenv()  
    
    keys = {
        "LLM_API_KEY": os.environ.get('GOOGLE_API_KEY'),
        "SEC_API_KEY": os.environ.get('SEC_API_KEY'),
        "ALPACA_API_KEY": os.environ.get('APCA_API_KEY'),
        "ALPACA_SECRET_KEY": os.environ.get('APCA_API_SECRET')
    }
    
    missing_keys = [k for k, v in keys.items() if not v]
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
    
    return keys