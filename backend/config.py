import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
CHAT_MODEL = os.getenv('CHAT_MODEL', '').strip() or 'gpt-4o-mini'
MOCK_RETRIEVAL = os.getenv('MOCK_RETRIEVAL', 'false').strip().lower() == 'true'


class BackendConfigurationError(RuntimeError):
    """Missing local configuration; messages must never include credentials."""


def get_api_key():
    # Accept the original team variable as well as the SDK's standard name.
    key = (os.getenv('OPENAI_API_KEY', '').strip()
           or os.getenv('LLM_API_KEY', '').strip())
    if not key:
        raise BackendConfigurationError(
            'Set OPENAI_API_KEY (or LLM_API_KEY) in the repository-root .env and restart the backend.'
        )
    return key
