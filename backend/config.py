import os
from dotenv import load_dotenv
load_dotenv()
CHAT_MODEL = os.getenv('CHAT_MODEL', 'gpt-4o-mini')
MOCK_RETRIEVAL = os.getenv('MOCK_RETRIEVAL', 'false').lower() == 'true'
