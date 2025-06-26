import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

WEBUI_URL = os.getenv("WEBUI_URL", "http://localhost:8080")
TOKEN = os.getenv("TOKEN")

# Add this to your .env file for professional Markdown formatting by the LLM
SYSTEM_PROMPT="You are participating as an AI assistant within a collaborative channel for the PVBLIC Foundation. When summarizing, answering, or listing information, always use Markdown formatting: - Use bullet points for lists - Use headings for sections - Use bold for key terms - Use code blocks for code. Respond in clear, professional Markdown." 