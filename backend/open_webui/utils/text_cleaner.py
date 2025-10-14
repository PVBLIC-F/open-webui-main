"""
Text Cleaning Utilities

Centralized text cleaning for email processing, embeddings, and vector storage.
Extracted from chat summary filter for reuse across the application.
"""

import re
import json
import unicodedata
from html import unescape
from urllib.parse import unquote


class TextCleaner:
    """Centralized text cleaning utilities - static methods for performance"""

    @staticmethod
    def clean_text(text) -> str:
        """Clean text by properly handling escape sequences and special characters"""
        if not text:
            return ""

        # Convert to string if needed
        if isinstance(text, (list, tuple)):
            text = "\n".join(str(item) for item in text)

        # Handle quoted JSON strings
        if isinstance(text, str) and text.startswith('"') and text.endswith('"'):
            try:
                parsed_text = json.loads(f"{text}")
                if isinstance(parsed_text, str):
                    text = parsed_text
            except:
                text = text[1:-1]

        if isinstance(text, str):
            try:
                # Handle HTML entities and URL encoding
                text = unescape(text)
                text = unquote(text)
                text = unicodedata.normalize("NFKC", text)

                # Remove control characters but keep newlines and tabs
                control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
                text = control_chars.sub("", text)
            except:
                pass

            # Handle escape sequences
            text = text.replace("\\n\\n", "\n\n")
            text = text.replace("\\n", "\n")
            text = text.replace("\\t", "\t")
            text = text.replace("\\r", "")
            text = text.replace('\\"', '"')
            text = text.replace("\\'", "'")
            text = text.replace("\\\\", "\\")

            # Normalize whitespace
            try:
                text = re.sub(r" {2,}", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                lines = text.split("\n")
                lines = [line.strip() for line in lines]
                text = "\n".join(lines)
            except:
                pass

        return text.strip() if isinstance(text, str) else ""

    @staticmethod
    def clean_for_pinecone(text) -> str:
        """Extra cleaning for Pinecone storage - removes problematic characters"""
        if not text:
            return ""

        text = TextCleaner.clean_text(text)
        text = re.sub(r"[\x00-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.encode("utf-8", "ignore").decode("utf-8")

        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    @staticmethod
    def ensure_string(text) -> str:
        """Ensure the input is a string"""
        if isinstance(text, list):
            return "\n".join(str(item) for item in text)
        elif text is None:
            return ""
        return str(text)

