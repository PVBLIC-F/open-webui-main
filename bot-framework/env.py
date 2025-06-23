import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

WEBUI_URL = os.getenv("WEBUI_URL", "http://localhost:8080")
TOKEN = os.getenv("TOKEN") 