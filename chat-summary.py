"""
Advanced Chat Summarization and Knowledge Retrieval Filter for OpenWebUI

This filter provides dual RAG capabilities:
1. Chat History Management: Uses Pinecone for intelligent chat summarization and retrieval
2. Company Knowledge Retrieval: Uses Ragie API for semantic search across company documents

Key Features:
- Dual vector database support (Pinecone + Ragie)
- Content-aware text splitting optimized for conversations
- Intelligent quality scoring and filtering
- Comprehensive caching and rate limiting
- Background processing with Redis/RQ
- Professional error handling and logging

Architecture:
- Pinecone: Chat summaries and conversation history (type="chat")
- Ragie: Company knowledge base from Google Drive sync (type="knowledge")  
- Both systems run in parallel for optimal performance
"""

import os
import logging
import asyncio
import time
import re
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import httpx
import urllib.parse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from redis import Redis
from rq import Queue, Retry
import hashlib
from functools import lru_cache
import uuid

# Add imports for content-aware chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from tiktoken import get_encoding

# Enhanced text processing imports for content-aware splitting
import unicodedata
from html import unescape
from urllib.parse import unquote

# Chat processing uses content-aware text splitting (no file processing needed)
# All content processing is optimized for conversation and summary text
CONTENT_AWARE_SPLITTING_ENABLED = True

dotenv_path = os.getenv("DOTENV_PATH")
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chat_history_filter_optimized")


# AsyncLimiter for API backpressure management
class AsyncLimiter:
    """
    Rate limiter for async operations that implements both concurrency limiting and
    time-based rate limiting to prevent overwhelming external services.
    """

    def __init__(self, max_concurrent=10, max_per_second=5):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.max_per_second = max_per_second
        self.window_size = 1.0  # 1 second window for rate calculation
        self.task_times = []
        self.lock = asyncio.Lock()  # Lock for thread-safety when updating task_times

    async def acquire(self):
        """Acquire permission to proceed, implementing both concurrency and rate limiting"""
        # Apply rate limiting by time window
        async with self.lock:
            now = time.time()
            # Clean up old task timestamps
            self.task_times = [t for t in self.task_times if now - t < self.window_size]

            # If we've hit the rate limit, wait until we're under the limit
            if len(self.task_times) >= self.max_per_second:
                wait_time = self.window_size - (now - self.task_times[0])
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.3f}s")
                    await asyncio.sleep(wait_time)

            # Record this task
            self.task_times.append(time.time())

        # Apply concurrency limiting
        await self.sem.acquire()

    def release(self):
        """Release a permit, allowing another operation to proceed"""
        self.sem.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Unstructured.io imports
try:
    from unstructured.partition.auto import partition
    from unstructured.chunking.title import chunk_by_title

    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logging.warning(
        "Unstructured.io not available. Document processing will be limited."
    )
dotenv_path = os.getenv("DOTENV_PATH")
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chat_history_filter_optimized")


def _debug_message_log(stage, messages):
    logger.info(
        f"[DEBUG] {stage}: num={len(messages)} roles={[m.get('role') for m in messages]}"
    )
    logger.info(
        f"[DEBUG] {stage} SYSTEM MSGS: {[(m.get('content','')[:60]) for m in messages if m.get('role')=='system']}"
    )
    logger.info(
        f"[DEBUG] {stage} FIRST MSGS: {[m.get('content','')[:40] for m in messages[:5]]}"
    )


# Configure token-aware and semantic splitters
encoding = get_encoding("cl100k_base")
token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base", chunk_size=1500, chunk_overlap=150
)
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    chunk_size=1500,
    chunk_overlap=150,
)


# Content-aware chunking and document processing class
class DocumentProcessor:
    """Document processor with content-aware chunking capabilities for chat content."""

    def __init__(self):
        # Default chunk parameters
        self.chunk_size = 1500
        self.chunk_overlap = 150

        # Initialize the content-aware text splitter
        self.content_aware_splitter = ContentAwareTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def clean_text(self, text):
        """
        Clean text for chat processing using the same enhanced approach as Filter._clean_text.
        """
        if not text:
            return ""

        # First, convert to string if needed
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
            # Enhanced cleaning for chat content processing
            try:
                # Handle HTML entities and URL encoding
                text = unescape(text)
                text = unquote(text)

                # Normalize Unicode characters
                text = unicodedata.normalize("NFKC", text)

                # Remove control characters but keep newlines and tabs
                control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
                text = control_chars.sub("", text)
            except:
                pass

            # This two-stage replacement handles the common case in your examples
            text = text.replace("\\n\\n", "\n\n")  # First replace double newlines
            text = text.replace("\\n", "\n")  # Then handle single newlines

            # Handle other escape sequences
            text = text.replace("\\t", "\t")
            text = text.replace("\\r", "")
            text = text.replace('\\"', '"')
            text = text.replace("\\'", "'")
            text = text.replace("\\\\", "\\")

            # Enhanced whitespace normalization
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
    def quick_quality_score(text: str) -> int:
        """Simple 0-5 quality score for text chunks."""
        score = 0
        # Has reasonable length
        if len(text) > 200:
            score += 1
        # Contains complete sentences
        if text.count(".") >= 2:
            score += 1
        # Has paragraph structure
        if "\n\n" in text:
            score += 1
        # Not just bullet points/enumeration
        if not (text.count("\n- ") > 5 or text.count("\n• ") > 5):
            score += 1
        # Contains context words
        if re.search(
            r"\b(because|therefore|however|additionally)\b", text, re.IGNORECASE
        ):
            score += 1
        return score

    def detect_document_type(self, chunks: List[str]) -> str:
        """Detect document type based on content patterns."""
        if not chunks or len(chunks) == 0:
            return "unknown"
        # Sample from first chunks for pattern matching
        sample_text = chunks[0][:4000] + (chunks[1][:2000] if len(chunks) > 1 else "")

        # Chat and conversation patterns
        chat_patterns = r"\b(user|assistant|human|ai|system|message|conversation|chat|dialogue|response)\b"
        academic_patterns = r"\b(abstract|introduction|methodology|conclusion|references|literature review|hypothesis|findings|research|study|analyzed|journal|proceedings)\b"
        meeting_patterns = r"\b(memorandum|meeting minutes|agenda|attendees|participants|convened|discussion|roundtable|conference|session|workshop|panel)\b"
        report_patterns = r"\b(quarterly|annual|financial|report|overview|executive summary|assessment|findings|recommendations|monitoring|evaluation|data analysis)\b"

        # Try to match against pattern sets
        if re.search(chat_patterns, sample_text, re.IGNORECASE):
            return "chat"
        elif re.search(academic_patterns, sample_text, re.IGNORECASE):
            return "academic"
        elif re.search(meeting_patterns, sample_text, re.IGNORECASE):
            return "meeting"
        elif re.search(report_patterns, sample_text, re.IGNORECASE):
            return "report"
        return "general"

    async def content_aware_chunk(self, text: str, elements=None) -> List[str]:
        """
        Apply content-aware chunking to text optimized for chat content.
        Args:
            text: The text to chunk
            elements: Pre-extracted structured elements (optional, not used for chat)
        Returns:
            list of text chunks
        """
        logger.info("Starting content-aware chunking process for chat content")

        try:
            # Use our specialized chat content-aware splitter
            logger.info(
                f"Processing text with content-aware splitter ({len(text)} chars)"
            )
            chunks = self.content_aware_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} content-aware chunks")
            return chunks

        except Exception as e:
            logger.warning(
                f"Content-aware chunking failed: {e}. Using fallback recursive splitting."
            )
            # Fallback to recursive splitting
            chunks = recursive_splitter.split_text(text)
            logger.info(
                f"Created {len(chunks)} chunks using fallback recursive splitting"
            )
            return chunks

    async def process_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Process text into content-aware chunks with metadata.
        Args:
            text: Text content to process
            metadata: Optional metadata to include
        Returns:
            Tuple of (chunks, enhanced_metadata)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to processor")
            return [], metadata or {}

        # Clean the text first using consistent approach
        cleaned_text = self.clean_text(text)

        # Apply content-aware chunking
        chunks = await self.content_aware_chunk(cleaned_text)

        # Detect document type
        doc_type = self.detect_document_type(chunks)

        # Enhance metadata with document type
        enhanced_metadata = metadata.copy() if metadata else {}
        enhanced_metadata["doc_type"] = doc_type
        enhanced_metadata["chunk_count"] = len(chunks)
        enhanced_metadata["processed_at"] = datetime.now(timezone.utc).isoformat()
        enhanced_metadata["processing_method"] = "content_aware_chat_splitter"

        return chunks, enhanced_metadata


class Filter:
    """
    Semantic search: matches 'chat' for same user (all threads) or 'knowledge'.
    Logs role/content at every step so you can debug summary/filter state.
    Warns if a summary is missing after it's supposed to be present.
    """

    # Shared HTTP client with connection pooling for better performance
    http_client = httpx.AsyncClient(
        timeout=10.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        http2=True,  # Enable HTTP/2 if supported
    )

    # In-memory LRU cache for the most frequent embedding requests
    # This reduces Redis lookups for common patterns
    @lru_cache(maxsize=1000)
    def _get_embedding_key_hash(self, text):
        """Generate a consistent hash for the embedding text"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # Helper function to ensure content is always a string
    def _ensure_string(self, text):
        """Ensure the input is a string"""
        if isinstance(text, list):
            # If it's a list, join with spaces or newlines
            return "\n".join(str(item) for item in text)
        elif text is None:
            return ""
        else:
            return str(text)

    # Helper function to safely check if content starts with a specific string
    def _content_startswith(self, content, prefix):
        """Safely check if content starts with a specific string, handling both strings and lists"""
        content_str = self._ensure_string(content)
        return content_str.startswith(prefix)

    def _clean_document_name(self, document_name: str) -> str:
        """Clean document name for better display."""
        if not document_name:
            return "Unknown Document"
        
        # Remove file extensions for cleaner display
        name = document_name
        if '.' in name:
            name_parts = name.rsplit('.', 1)
            if len(name_parts) == 2 and len(name_parts[1]) <= 5:  # Likely a file extension
                name = name_parts[0]
        
        # Replace underscores and hyphens with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Capitalize words
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name
    
    def _format_citation_content(self, text: str, document_name: str, start_time: float = None, end_time: float = None, media_type: str = "document") -> str:
        """Format citation content for better display in the UI."""
        if not text:
            return ""
        
        # Clean the text first
        clean_text = self._clean_text(text)
        
        # Don't truncate - send full content to frontend
        # The frontend will handle display appropriately
        
        # Create a structured format
        formatted_parts = []
        
        # Add media type context
        if media_type == "audio" and start_time is not None and end_time is not None:
            duration = end_time - start_time
            formatted_parts.append(f"🎵 Audio Segment ({start_time:.1f}s - {end_time:.1f}s, {duration:.1f}s duration)")
        elif media_type == "video" and start_time is not None and end_time is not None:
            duration = end_time - start_time
            formatted_parts.append(f"🎬 Video Segment ({start_time:.1f}s - {end_time:.1f}s, {duration:.1f}s duration)")
        elif media_type == "document":
            formatted_parts.append(f"📄 Document Excerpt")
        
        # Add the content
        formatted_parts.append("")  # Empty line for spacing
        formatted_parts.append(clean_text)
        
        return "\n".join(formatted_parts)

    # Improved text cleaning function
    def _clean_text(self, text):
        """
        Clean text by properly handling escape sequences and special characters.
        Enhanced for chat and conversation content processing.

        Args:
            text: The text to clean

        Returns:
            Cleaned text with proper newlines and without raw escape sequences
        """
        if not text:
            return ""

        # First, convert to string if needed
        if isinstance(text, (list, tuple)):
            text = "\n".join(str(item) for item in text)

        # Handle quoted JSON strings - this is common in API responses
        if isinstance(text, str) and text.startswith('"') and text.endswith('"'):
            try:
                # Try to parse as JSON string
                parsed_text = json.loads(f"{text}")
                if isinstance(parsed_text, str):
                    text = parsed_text
            except:
                # If not valid JSON, just strip the quotes
                text = text[1:-1]

        # Special case: handle literal escaped newlines in string
        if isinstance(text, str):
            # Enhanced cleaning for chat content processing
            try:
                # Handle HTML entities and URL encoding
                text = unescape(text)
                text = unquote(text)

                # Normalize Unicode characters
                text = unicodedata.normalize("NFKC", text)

                # Remove control characters but keep newlines and tabs
                control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
                text = control_chars.sub("", text)
            except:
                # Fallback to basic cleaning if enhanced cleaning fails
                pass

            # This two-stage replacement handles the common case in your examples
            text = text.replace("\\n\\n", "\n\n")  # First replace double newlines
            text = text.replace("\\n", "\n")  # Then handle single newlines

            # Handle other escape sequences
            text = text.replace("\\t", "\t")
            text = text.replace("\\r", "")
            text = text.replace('\\"', '"')
            text = text.replace("\\'", "'")
            text = text.replace("\\\\", "\\")

            # Enhanced whitespace normalization for chat content
            try:
                # Normalize whitespace
                text = re.sub(r" {2,}", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)

                # Remove excessive whitespace from lines
                lines = text.split("\n")
                lines = [line.strip() for line in lines]
                text = "\n".join(lines)
            except:
                # Fallback to basic cleanup
                pass

        # Final cleanup
        return text.strip() if isinstance(text, str) else ""

    def _clean_for_pinecone(self, text):
        """
        Extra text cleaning specifically for text being sent to Pinecone.
        Removes problematic characters that cause JSON serialization issues.

        Args:
            text: Text to clean

        Returns:
            Cleaned text safe for Pinecone storage
        """
        if not text:
            return ""

        # First apply regular cleaning
        text = self._clean_text(text)

        # Remove control characters and zero-width spaces
        import re

        text = re.sub(r"[\x00-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]", "", text)

        # Normalize whitespace (replace multiple spaces, tabs, etc with single space)
        text = re.sub(r"\s+", " ", text).strip()

        # Handle potential JSON escaping issues by ensuring proper UTF-8 encoding
        text = text.encode("utf-8", "ignore").decode("utf-8")

        # Truncate extremely long texts to prevent size issues
        max_length = 10000  # Reasonable max length for metadata fields
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text

    class Valves(BaseModel):
        pinecone_api_key: str = Field(
            default_factory=lambda: os.getenv("PINECONE_API_KEY", "")
        )
        pinecone_environment: str = Field(
            default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "")
        )
        pinecone_index_name: str = Field(
            default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "")
        )
        namespace: str = Field(
            default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "")
        )
        vector_dim: int = Field(
            default_factory=lambda: int(os.getenv("VECTOR_DIM", ""))
        )
        top_k: int = Field(default_factory=lambda: int(os.getenv("TOP_K", "")))
        display_k: int = Field(default_factory=lambda: int(os.getenv("DISPLAY_K", "")))
        min_score_chat: float = Field(
            default_factory=lambda: float(os.getenv("MIN_SCORE_CHAT", ""))
        )
        min_score_knowledge: float = Field(
            default_factory=lambda: float(os.getenv("MIN_SCORE_KNOWLEDGE", ""))
        )
        min_quality_score: int = Field(
            default_factory=lambda: int(os.getenv("MIN_QUALITY_SCORE", ""))
        )
        keep_recent: int = Field(
            default_factory=lambda: int(os.getenv("KEEP_RECENT_TURNS", ""))
        )
        keep_without_summary: int = Field(
            default_factory=lambda: int(os.getenv("KEEP_WITHOUT_SUMMARY", ""))
        )
        turns_before: int = Field(
            default_factory=lambda: int(os.getenv("TURNS_BEFORE_SUMMARY", ""))
        )
        openai_embedding_model: str = Field(
            default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "")
        )
        openai_embedding_base_url: str = Field(
            default_factory=lambda: os.getenv("OPENAI_EMBEDDING_BASE_URL", "")
        )
        openai_api_key: str = Field(
            default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
        )
        openrouter_api_key: str = Field(
            default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
        )
        openrouter_api_base_url: str = Field(
            default_factory=lambda: os.getenv("OPENROUTER_API_BASE_URL", "")
        )
        openrouter_model: str = Field(
            default_factory=lambda: os.getenv("OPENROUTER_MODEL", "")
        )
        
        # Ragie configuration
        ragie_api_key: str = Field(
            default_factory=lambda: os.getenv("RAGIE_API_KEY", "")
        )
        ragie_partition: str = Field(
            default_factory=lambda: os.getenv("RAGIE_PARTITION", "default")
        )
        ragie_top_k: int = Field(
            default_factory=lambda: max(1, min(50, int(os.getenv("RAGIE_TOP_K", "5"))))
        )
        ragie_score_threshold: float = Field(
            default_factory=lambda: max(0.0, min(1.0, float(os.getenv("RAGIE_SCORE_THRESHOLD", "0.25"))))
        )
        ragie_recency_bias: bool = Field(
            default_factory=lambda: os.getenv("RAGIE_RECENCY_BIAS", "true").lower() == "true"
        )
        
        # Base URL for proxy endpoints (for absolute URLs)
        base_url: str = Field(
            default_factory=lambda: os.getenv("WEBUI_URL", "")
        )

    def __init__(self):
        self.valves = self.Valves()
        logger.info("Loaded configuration: %s", self.valves.json())
        if not self.valves.pinecone_api_key or not self.valves.pinecone_index_name:
            raise ValueError("Missing Pinecone key or index name")

        # Pinecone setup with newest client (instance-based API)
        self.pc = Pinecone(api_key=self.valves.pinecone_api_key)

        # Check if index exists, and create if needed
        try:
            if self.valves.pinecone_index_name not in self.pc.list_indexes().names():
                logger.warning(
                    f"Index {self.valves.pinecone_index_name} not found. Check your configuration."
                )

            # Get the index
            self.idx = self.pc.Index(self.valves.pinecone_index_name)
            logger.info(
                f"Pinecone initialized: index={self.valves.pinecone_index_name}, namespace={self.valves.namespace}"
            )
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

        # OpenAI embedding client
        if self.valves.openai_embedding_base_url:
            self.embedding_client = AsyncOpenAI(
                api_key=self.valves.openai_api_key,
                base_url=self.valves.openai_embedding_base_url,
                http_client=self.http_client,  # Use shared HTTP client
            )
            logger.info(
                f"Initialized OpenAI embedding client with base_url={self.valves.openai_embedding_base_url}"
            )
        else:
            self.embedding_client = AsyncOpenAI(
                api_key=self.valves.openai_api_key,
                http_client=self.http_client,  # Use shared HTTP client
            )
            logger.info("Initialized OpenAI embedding client with default base_url.")
        # OpenRouter chat client
        self.chat_client = AsyncOpenAI(
            api_key=self.valves.openrouter_api_key,
            base_url=self.valves.openrouter_api_base_url,
            http_client=self.http_client,  # Use shared HTTP client
        )
        logger.info("OpenRouter client initialized for chat summarization.")
        
        # Initialize Ragie configuration
        if self.valves.ragie_api_key:
            logger.info("Ragie API key configured - knowledge retrieval enabled")
            logger.info(f"Ragie configuration: partition={self.valves.ragie_partition}, top_k={self.valves.ragie_top_k}, recency_bias={self.valves.ragie_recency_bias}")
        else:
            logger.warning("Ragie API key not provided - knowledge retrieval will be disabled")

        # Initialize improved async limiters for better backpressure management
        self.embedding_limiter = AsyncLimiter(max_concurrent=20, max_per_second=40)
        self.openrouter_limiter = AsyncLimiter(max_concurrent=5, max_per_second=3)
        self.pinecone_limiter = AsyncLimiter(max_concurrent=10, max_per_second=10)

        # Setup semaphore for rate-limiting embedding API calls (legacy)
        self.embedding_semaphore = asyncio.Semaphore(
            20
        )  # Allow up to 20 concurrent embedding calls

        # Query deduplication tracking
        self._active_queries = {}  # Tracks in-flight queries by hash
        self._query_results_cache = (
            {}
        )  # Short-lived in-memory cache for duplicate queries

        # Redis connection pool and RQ queue setup
        redis_url = os.getenv("REDIS_URL")
        pool_size = int(os.getenv("REDIS_POOL_SIZE", ""))
        timeout = int(os.getenv("REDIS_TIMEOUT", ""))
        pool = Redis.from_url(
            url=redis_url,
            max_connections=pool_size,
            socket_timeout=timeout,
        )
        self.redis_conn = pool
        self.rq_queue = Queue(name="pinecone-upserts", connection=self.redis_conn)
        logger.info(
            f"Redis connected at {redis_url} (pool_size={pool_size}, timeout={timeout}s)"
        )
        # TTLs for caches
        self.embed_cache_ttl = int(os.getenv("REDIS_EMBEDDING_CACHE_TTL", ""))
        self.search_cache_ttl = int(os.getenv("REDIS_SEARCH_CACHE_TTL", ""))
        # Knowledge cache configuration
        self.knowledge_cache_ttl = int(
            os.getenv("REDIS_KNOWLEDGE_CACHE_TTL", str(self.search_cache_ttl))
        )
        self.enable_knowledge_cache = (
            os.getenv("REDIS_ENABLE_KNOWLEDGE_CACHE", "true").lower() == "true"
        )
        logger.info(
            f"Knowledge caching {'enabled' if self.enable_knowledge_cache else 'disabled'} with TTL: {self.knowledge_cache_ttl}s"
        )
        # Initialize counters for embedding call metrics
        self.embed_calls = 0
        self.embed_cache_hits = 0
        self.embed_errors = 0
        self.embed_last_metrics_time = time.time()
        # Initialize retrieval metrics
        self.knowledge_cache_hits = 0
        self.knowledge_cache_misses = 0
        self.retrieval_metrics_time = time.time()
        # Initialize the document processor for content-aware chunking
        self.document_processor = DocumentProcessor()
        
        # Default chunk parameters for content-aware splitting
        self.chunk_size = 1500
        self.chunk_overlap = 150

        # Log quality score threshold
        logger.info(
            f"Minimum quality score threshold set to: {self.valves.min_quality_score}"
        )

    async def _run_io(self, func, *args, **kwargs):
        """Legacy method - use asyncio.to_thread instead"""
        logger.info(
            f"Running IO task (legacy method): func={getattr(func, '__name__', repr(func))}, args={args}, kwargs={kwargs}"
        )
        return await asyncio.to_thread(func, *args, **kwargs)

    def _get_embed_cache_key(self, text):
        """Generate a standardized cache key for embeddings"""
        # Use the hash helper function
        hash_key = self._get_embedding_key_hash(text)
        return f"emb:{hash_key}"

    def _query_hash(self, vector, filter_expr, namespace=None):
        """Generate a hash for a query to detect duplicates"""
        vector_hash = self._hash_vector(vector)
        filter_hash = hashlib.md5(
            json.dumps(filter_expr, sort_keys=True).encode()
        ).hexdigest()
        namespace_part = f":{namespace}" if namespace else ""
        return f"query:{vector_hash}:{filter_hash}{namespace_part}"

    async def _deduplicated_query(self, vector, filter_expr, namespace=None):
        """Execute Pinecone query with deduplication for identical in-flight queries"""
        if not vector:
            return {"matches": []}

        query_hash = self._query_hash(vector, filter_expr, namespace)

        # Check if this query is already in our short-term memory cache
        if query_hash in self._query_results_cache:
            cached_result = self._query_results_cache[query_hash]
            # Check if cache entry is still fresh (5 seconds)
            if time.time() - cached_result["timestamp"] < 5.0:
                logger.info(f"Query cache hit for hash {query_hash}")
                return cached_result["result"]

        # Check if identical query is already in progress
        if query_hash in self._active_queries:
            logger.info(
                f"Duplicate query detected: {query_hash}, reusing in-flight request"
            )
            return await self._active_queries[query_hash]

        # Create a new unique task for this query
        async def execute_query():
            try:
                # Apply rate limiting
                async with self.pinecone_limiter:
                    logger.debug(f"Executing Pinecone query with hash {query_hash}")
                    # Use a helper function to run the synchronous Pinecone query in a thread pool
                    result = await asyncio.to_thread(
                        lambda: self.idx.query(
                            vector=vector,
                            top_k=self.valves.top_k,
                            namespace=(
                                self.valves.namespace
                                if namespace is None
                                else namespace
                            ),
                            filter=filter_expr,
                            include_metadata=True,
                            include_values=False,
                        )
                    )

                    # Cache the result briefly (5 seconds)
                    self._query_results_cache[query_hash] = {
                        "result": result,
                        "timestamp": time.time(),
                    }

                    # Clean up older cache entries if cache grows too large
                    if len(self._query_results_cache) > 50:  # Keep last 50 queries
                        now = time.time()
                        expired_keys = [
                            k
                            for k, v in self._query_results_cache.items()
                            if now - v["timestamp"] > 5.0
                        ]
                        for k in expired_keys:
                            self._query_results_cache.pop(k, None)

                    return result
            finally:
                # Remove from active queries when done
                self._active_queries.pop(query_hash, None)

        # Register and execute the query
        query_task = asyncio.create_task(execute_query())
        self._active_queries[query_hash] = query_task
        return await query_task

    def _log_embedding_metrics(self, force=False):
        """Log embedding API call metrics periodically"""
        now = time.time()
        # Log metrics every 5 minutes or when forced
        if force or (now - self.embed_last_metrics_time > 300):
            total = self.embed_calls or 1  # Avoid division by zero
            hit_rate = (self.embed_cache_hits / total) * 100 if total > 0 else 0
            logger.info(
                f"Embedding metrics: calls={self.embed_calls}, "
                f"cache_hits={self.embed_cache_hits} ({hit_rate:.1f}%), "
                f"errors={self.embed_errors}"
            )
            self.embed_last_metrics_time = now

    def _log_retrieval_metrics(self, force=False):
        """Log differential caching metrics periodically"""
        now = time.time()
        # Log metrics every 5 minutes or when forced
        if force or (now - self.retrieval_metrics_time > 300):
            total = self.knowledge_cache_hits + self.knowledge_cache_misses
            if total > 0:
                hit_rate = (self.knowledge_cache_hits / total) * 100
                logger.info(
                    f"Knowledge retrieval metrics: "
                    f"cache_hits={self.knowledge_cache_hits}, "
                    f"cache_misses={self.knowledge_cache_misses}, "
                    f"hit_rate={hit_rate:.1f}%"
                )
            self.retrieval_metrics_time = now

    async def _batch_embed(self, texts_list):
        """
        Efficiently batch embedding API calls to reduce API overhead.
        This will handle a list of text items and return embeddings for all.
        """
        if not texts_list:
            return []
        # First, check what we can get from cache
        results = []
        texts_to_embed = []
        indices_to_embed = []
        # Identify which texts need to be embedded vs. which are in cache
        for i, text in enumerate(texts_list):
            # Convert to string if needed - FIX FOR LIST ERROR
            text = self._ensure_string(text)
            # Clean text - added text cleaning here
            text = self._clean_text(text)
            if not text or not text.strip():
                # Handle empty text
                results.append([0.0] * self.valves.vector_dim)
                continue
            key = self._get_embed_cache_key(text)
            cached = self.redis_conn.get(key)
            if cached:
                self.embed_cache_hits += 1
                try:
                    results.append(json.loads(cached))
                except Exception as e:
                    logger.warning(f"Cache deserialization error for key={key}: {e}")
                    # If we can't deserialize, we need to re-embed this text
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
                    # Add a placeholder in results that will be replaced later
                    results.append(None)
            else:
                # Text not in cache, need to generate embedding
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                # Add a placeholder in results that will be replaced later
                results.append(None)
        # If nothing needs embedding, return the cached results
        if not texts_to_embed:
            self._log_embedding_metrics()
            return results

        # We need to call the embedding API for texts_to_embed
        self.embed_calls += 1

        # Process texts in optimal sized batches if there are many
        batch_size = 10  # OpenAI embedding API works well with batches of ~10

        # If we have a large number of texts, process in batches for better throughput
        if len(texts_to_embed) > batch_size:
            logger.info(
                f"Processing {len(texts_to_embed)} embeddings in batches of {batch_size}"
            )
            all_vectors = []

            # Process batches concurrently with controlled parallelism
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i : i + batch_size]
                batch_indices = indices_to_embed[i : i + batch_size]

                # Use the embedding limiter for backpressure control
                async with self.embedding_limiter:
                    try:
                        # Make the API call with timeout protection
                        logger.info(
                            f"Embedding API call for batch of {len(batch_texts)} texts"
                        )
                        start_time = time.time()
                        resp = await asyncio.wait_for(
                            self.embedding_client.embeddings.create(
                                model=self.valves.openai_embedding_model,
                                input=batch_texts,
                            ),
                            timeout=10.0,  # Slightly longer timeout for batch
                        )
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Embedding API batch call completed in {elapsed:.2f}s"
                        )

                        # Process the results and update cache
                        if resp.data and len(resp.data) == len(batch_texts):
                            for idx, embed_resp, text in zip(
                                batch_indices, resp.data, batch_texts
                            ):
                                vector = embed_resp.embedding
                                # Cache the result for future use
                                key = self._get_embed_cache_key(text)
                                try:
                                    cache_data = json.dumps(vector)
                                    self.redis_conn.set(
                                        key, cache_data, ex=self.embed_cache_ttl
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Cache storage error for key={key}: {e}"
                                    )
                                # Update the results list
                                results[idx] = vector
                        else:
                            logger.error(f"Unexpected embedding API response format")
                            self.embed_errors += 1
                            # Fill in any missing results with zero vectors
                            for idx in batch_indices:
                                if results[idx] is None:
                                    results[idx] = [0.0] * self.valves.vector_dim
                    except asyncio.TimeoutError:
                        logger.error(f"Embedding API timeout after 10 seconds")
                        self.embed_errors += 1
                        # Fill in any missing results with zero vectors
                        for idx in batch_indices:
                            if results[idx] is None:
                                results[idx] = [0.0] * self.valves.vector_dim
                    except Exception as e:
                        logger.error(f"Embedding API error: {e}")
                        self.embed_errors += 1
                        # Fill in any missing results with zero vectors
                        for idx in batch_indices:
                            if results[idx] is None:
                                results[idx] = [0.0] * self.valves.vector_dim
        else:
            # For smaller batches, just use the original approach with our new limiter
            async with self.embedding_limiter:
                try:
                    # Make the API call with timeout protection
                    logger.info(f"Embedding API call for {len(texts_to_embed)} texts")
                    start_time = time.time()
                    resp = await asyncio.wait_for(
                        self.embedding_client.embeddings.create(
                            model=self.valves.openai_embedding_model,
                            input=texts_to_embed,
                        ),
                        timeout=8.0,  # 8 second timeout
                    )
                    elapsed = time.time() - start_time
                    logger.info(f"Embedding API call completed in {elapsed:.2f}s")
                    # Process the results and update cache
                    if resp.data and len(resp.data) == len(texts_to_embed):
                        for idx, embed_resp, text in zip(
                            indices_to_embed, resp.data, texts_to_embed
                        ):
                            vector = embed_resp.embedding
                            # Cache the result for future use
                            key = self._get_embed_cache_key(text)
                            try:
                                cache_data = json.dumps(vector)
                                self.redis_conn.set(
                                    key, cache_data, ex=self.embed_cache_ttl
                                )
                            except Exception as e:
                                logger.error(f"Cache storage error for key={key}: {e}")
                            # Update the results list
                            results[idx] = vector
                    else:
                        logger.error(f"Unexpected embedding API response format")
                        self.embed_errors += 1
                        # Fill in any missing results with zero vectors
                        for idx in indices_to_embed:
                            if results[idx] is None:
                                results[idx] = [0.0] * self.valves.vector_dim
                except asyncio.TimeoutError:
                    logger.error(f"Embedding API timeout after 8 seconds")
                    self.embed_errors += 1
                    # Fill in any missing results with zero vectors
                    for idx in indices_to_embed:
                        if results[idx] is None:
                            results[idx] = [0.0] * self.valves.vector_dim
                except Exception as e:
                    logger.error(f"Embedding API error: {e}")
                    self.embed_errors += 1
                    # Fill in any missing results with zero vectors
                    for idx in indices_to_embed:
                        if results[idx] is None:
                            results[idx] = [0.0] * self.valves.vector_dim
        # Log metrics periodically
        self._log_embedding_metrics()
        return results

    async def _embed(self, texts: list) -> list:
        """
        Optimized embedding function with improved caching, batching, and error handling
        This is a convenience wrapper around _batch_embed for backward compatibility
        """
        if not texts:
            logger.warning("Empty texts provided to _embed")
            return [0.0] * self.valves.vector_dim
        # Call the batched embedding function and return the first result
        results = await self._batch_embed(texts)
        return results[0] if results else [0.0] * self.valves.vector_dim

    def _hash_vector(self, vec: list) -> str:
        """
        Create a stable hash of vector for caching.
        Rounds vector values to reduce minor variations and improve cache hits.
        """
        if not vec:
            return "null_vector"
        # Round vector values to 5 decimal places to improve cache hit rate
        rounded_vec = [round(float(x), 5) for x in vec]
        vec_str = json.dumps(rounded_vec)
        return hashlib.md5(vec_str.encode()).hexdigest()[:12]

    async def _retrieve_knowledge(self, vec: list, __event_emitter__=None) -> list:
        """Retrieve knowledge items from Ragie using direct API call."""
        logger.info("[DEBUG] === STARTING RAGIE KNOWLEDGE RETRIEVAL ===")
        if not self.valves.ragie_api_key:
            logger.warning("Ragie API key not provided, skipping knowledge retrieval")
            return []
        
        # Check if we have a current query stored
        if not hasattr(self, 'current_query'):
            logger.warning("No current query available for Ragie")
            return []
        
        try:
            logger.info(f"Querying Ragie for knowledge documents with query: {self.current_query[:50]}...")
            
            # Execute Ragie search using direct API call
            start_time = time.time()
            
            # Prepare the request
            url = "https://api.ragie.ai/retrievals"
            payload = {
                "query": self.current_query,
                "top_k": self.valves.ragie_top_k,
                "partition": self.valves.ragie_partition,
            }
            
            # Add optional parameters
            if self.valves.ragie_recency_bias:
                payload["recency_bias"] = True
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {self.valves.ragie_api_key}"
            }
            
            logger.info(f"[DEBUG] Executing Ragie API call with payload: {payload}")
            
            # Make the API call using the shared HTTP client
            response = await self.http_client.post(url, json=payload, headers=headers)
            
            # Check response status
            if response.status_code != 200:
                logger.error(f"[DEBUG] Ragie API returned status {response.status_code}: {response.text}")
                return []
            
            response_data = response.json()
            
            # For compatibility with asyncio.to_thread timing
            response = response_data
            query_time = time.time() - start_time
            
            logger.info(f"[DEBUG] Ragie query completed in {query_time:.3f}s")
            
            # Process Ragie results based on actual API response structure
            items = []
            
            if response and 'scored_chunks' in response:
                chunks = response['scored_chunks']
                logger.info(f"[DEBUG] Ragie returned {len(chunks)} chunks")
                
                # Debug: Show all chunks and their scores first
                logger.info(f"[DEBUG] All Ragie chunks with scores (threshold={self.valves.ragie_score_threshold}):")
                for i, chunk in enumerate(chunks):
                    score = chunk.get('score', 1.0)
                    doc_name = chunk.get('document_name', 'Unknown')
                    logger.info(f"[DEBUG]   {i+1}. {doc_name} - Score: {score:.4f}")
                
                filtered_by_score = 0
                for chunk in chunks:
                    # Get score
                    score = chunk.get('score', 1.0)
                    
                    # Check score threshold
                    if score < self.valves.ragie_score_threshold:
                        filtered_by_score += 1
                        continue
                    
                    # Extract data from chunk
                    text = chunk.get('text', '')
                    document_name = chunk.get('document_name', 'Unknown Document')
                    document_id = chunk.get('document_id', '')
                    chunk_id = chunk.get('id', 'unknown')
                    
                    # Get additional metadata
                    document_metadata = chunk.get('document_metadata', {})
                    file_path = document_metadata.get('file_path', '')
                    source_url = document_metadata.get('source_url', '')
                    
                    # Get audio/video links and timestamps from Ragie
                    chunk_metadata = chunk.get('metadata', {})
                    start_time = chunk_metadata.get('start_time')
                    end_time = chunk_metadata.get('end_time')
                    links = chunk.get('links', {})
                    
                    # Clean the text
                    text = self._clean_text(text)
                    
                    # Emit citation events for audio/video streaming and document links
                    # The citations will display in the UI, so we don't need text attribution
                    if source_url and document_name:
                        # Check if this chunk has any streaming links before processing
                        has_audio = links.get('self_audio_stream', {}).get('href') is not None
                        has_video = links.get('self_video_stream', {}).get('href') is not None
                        
                        logger.info(f"[DEBUG] Chunk {chunk_id} - has_audio: {has_audio}, has_video: {has_video}")
                        logger.info(f"[DEBUG] Available links: {list(links.keys())}")
                        if links.get('document_audio_stream'):
                            logger.info(f"[DEBUG] document_audio_stream: {links['document_audio_stream']}")
                        if links.get('document_video_stream'):
                            logger.info(f"[DEBUG] document_video_stream: {links['document_video_stream']}")
                        
                        # Add streaming links using proxy for authentication and SSL compatibility
                        # Route all Ragie URLs through our proxy to handle SSL/TLS properly
                        
                        # Collect both audio and video URLs if available
                        streaming_urls = {}
                        
                        if has_audio and links.get('self_audio_stream', {}).get('href'):
                            # Use the chunk-specific audio URL provided by Ragie API
                            ragie_audio_url = links['self_audio_stream']['href']
                            logger.info(f"[DEBUG] Using chunk-specific audio URL from Ragie API: {ragie_audio_url}")
                            
                            base_url = self.valves.base_url.rstrip('/') if self.valves.base_url else ""
                            proxy_audio_url = f"{base_url}/proxy/ragie/stream?url={urllib.parse.quote(ragie_audio_url, safe='')}"
                            streaming_urls['audio'] = proxy_audio_url
                            logger.info(f"[DEBUG] Generated proxy audio URL: {proxy_audio_url}")
                        
                        if has_video and links.get('self_video_stream', {}).get('href'):
                            # Use the chunk-specific video URL provided by Ragie API
                            ragie_video_url = links['self_video_stream']['href']
                            logger.info(f"[DEBUG] Using chunk-specific video URL from Ragie API: {ragie_video_url}")
                            
                            base_url = self.valves.base_url.rstrip('/') if self.valves.base_url else ""
                            proxy_video_url = f"{base_url}/proxy/ragie/stream?url={urllib.parse.quote(ragie_video_url, safe='')}"
                            streaming_urls['video'] = proxy_video_url
                            logger.info(f"[DEBUG] Generated proxy video URL: {proxy_video_url}")
                        
                        # Create a single citation with both audio and video options
                        if __event_emitter__:
                            # Create citation name based on available media
                            media_icons = []
                            if 'audio' in streaming_urls:
                                media_icons.append("🎵")
                            if 'video' in streaming_urls:
                                media_icons.append("🎬")
                            
                            citation_name = f"{''.join(media_icons)} {document_name}"
                            if start_time is not None and end_time is not None:
                                duration = end_time - start_time
                                citation_name += f" ({start_time:.1f}s-{end_time:.1f}s, {duration:.1f}s)"
                            
                            # Use video URL as primary, audio as fallback, then Google Drive
                            primary_url = streaming_urls.get('video') or streaming_urls.get('audio') or source_url
                            
                            # Determine media type based on what's available
                            if streaming_urls.get('video'):
                                media_type = "video"
                            elif streaming_urls.get('audio'):
                                media_type = "audio"
                            else:
                                media_type = "document"
                            
                            # Send raw text so frontend can parse JSON and format appropriately
                            await __event_emitter__({
                                "type": "citation",
                                "data": {
                                    "document": [text],  # Send raw text, not formatted
                                    "metadata": [
                                        {
                                            "date_accessed": datetime.now().isoformat(),
                                            "source": self._clean_document_name(document_name),
                                            "start_time": start_time,
                                            "end_time": end_time,
                                            "media_type": media_type,
                                            "duration": f"{end_time - start_time:.1f}s" if start_time and end_time else None,
                                            "embed_player": bool(streaming_urls),
                                            "stream_url": primary_url,
                                            "audio_url": streaming_urls.get('audio'),
                                            "video_url": streaming_urls.get('video'),
                                            "fallback_url": source_url,  # Google Drive link as fallback
                                            "streaming_note": "Streaming chunk-specific media from Ragie API via authenticated proxy." if streaming_urls else "Streaming not available, using Google Drive fallback."
                                        }
                                    ],
                                    "source": {
                                        "name": citation_name,
                                        "url": primary_url,
                                        "display_type": "embedded_player" if streaming_urls else "external_link",
                                        "audio_url": streaming_urls.get('audio'),
                                        "video_url": streaming_urls.get('video'),
                                        "fallback_url": source_url  # Google Drive link as fallback
                                    },
                                },
                            })

                        
                        # Use clean text without attribution since citations handle the links
                        attributed_text = text
                        
                        # Enhanced logging for audio/video content
                        media_info = []
                        if has_audio:
                            if start_time is not None and end_time is not None:
                                duration = end_time - start_time
                                media_info.append(f"🎵 Audio stream ({start_time:.1f}s-{end_time:.1f}s, {duration:.1f}s)")
                            else:
                                media_info.append("🎵 Audio stream")
                        if has_video:
                            if start_time is not None and end_time is not None:
                                duration = end_time - start_time
                                media_info.append(f"🎬 Video stream ({start_time:.1f}s-{end_time:.1f}s, {duration:.1f}s)")
                            else:
                                media_info.append("🎬 Video stream")
                        
                        # Show what type of content this is
                        content_type = "📄 Document"
                        if has_audio or has_video:
                            content_type = "🎬 Media"
                        elif 'self_image' in links:
                            content_type = "🖼️ Image"
                        
                        media_str = f" + {', '.join(media_info)}" if media_info else ""
                        logger.info(f"[DEBUG] ✅ {content_type}: {document_name} -> {source_url}{media_str}")
                    else:
                        attributed_text = text
                        logger.info(f"[DEBUG] No citation events - missing document name or URL")
                    
                    logger.info(
                        f"[RAGIE KNOWLEDGE] document={document_name}, "
                        f"score={score:.4f}, chunk_id={chunk_id}, source_url={source_url[:50]}..." if source_url else f"score={score:.4f}, chunk_id={chunk_id}"
                    )
                    
                    # Debug: Show a preview of the clean text (citations are handled separately)
                    # IMPORTANT: Only truncate for logging, NOT for the actual response
                    if len(attributed_text) > 300:
                        preview_text = attributed_text[:300] + "..."
                        logger.info(f"[DEBUG] Clean text for LLM (citations handled separately): {preview_text}")
                        logger.info(f"[DEBUG] Full text length: {len(attributed_text)} characters")
                    else:
                        preview_text = attributed_text
                        logger.info(f"[DEBUG] Clean text for LLM (citations handled separately): {preview_text}")
                    
                    # Store the FULL text, not the truncated preview
                    items.append({
                        "type": "knowledge",
                        "text": attributed_text,  # Use full text, not preview_text
                        "document_name": document_name,
                        "score": score,
                        "quality_score": 5,  # Ragie results are pre-qualified
                        "metadata": {
                            "source": "ragie",
                            "document_id": document_id,
                            "chunk_id": chunk_id,
                            "file_path": file_path,
                            "source_url": source_url,
                            "start_time": start_time,
                            "end_time": end_time,
                            "links": links,
                            **document_metadata
                        }
                    })
            
                if filtered_by_score > 0:
                    logger.info(f"[DEBUG] Filtered out {filtered_by_score} Ragie chunks below score threshold {self.valves.ragie_score_threshold}")
            else:
                logger.info("[DEBUG] Ragie returned no scored chunks in response")
            
            logger.info(f"[DEBUG] Retrieved {len(items)} knowledge items from Ragie")
            if items:
                logger.info(f"[DEBUG] Ragie results preview: {items[0]['document_name']} (score: {items[0]['score']:.3f})")
            
            # Update metrics (for compatibility)
            self.knowledge_cache_hits += 1  # Since Ragie handles its own caching
            self._log_retrieval_metrics()
            
            logger.info("[DEBUG] === COMPLETED RAGIE KNOWLEDGE RETRIEVAL ===")
            return items
            
        except Exception as e:
            logger.error(f"Ragie knowledge query failed: {e}")
            logger.info("[DEBUG] === FAILED RAGIE KNOWLEDGE RETRIEVAL ===")
            return []

    async def _query_knowledge_from_pinecone(self, vec: list) -> list:
        """Query Pinecone directly for knowledge items with quality filter."""
        # Knowledge filter with quality score threshold
        filter_expr = {
            "type": {"$eq": "knowledge"},
            "$or": [
                {"quality_score": {"$gte": self.valves.min_quality_score}},
                {
                    "quality_score": {"$exists": False}
                },  # Include vectors without quality_score field
            ],
        }
        logger.info(
            f"Querying Pinecone for KNOWLEDGE with filter: {filter_expr} (min_quality_score={self.valves.min_quality_score})"
        )
        try:
            # Use the deduplication method to avoid redundant identical queries
            start_time = time.time()
            res = await self._deduplicated_query(vec, filter_expr)
            query_time = time.time() - start_time
            matches = getattr(res, "matches", []) or []
            logger.info(
                f"Pinecone knowledge query returned {len(matches)} matches in {query_time:.3f}s"
            )

            # Progressive refinement if we don't have enough quality matches
            if (
                len(
                    [
                        m
                        for m in matches
                        if getattr(m, "score", 0) >= self.valves.min_score_knowledge
                    ]
                )
                < 3
            ):
                logger.info(
                    "Initial knowledge query returned too few quality matches, trying progressive refinement"
                )

                # Relaxed filter that still ensures quality standards
                relaxed_filter = {
                    "type": {"$eq": "knowledge"},
                    # Allow lower quality scores but not too low
                    "$or": [
                        {
                            "quality_score": {
                                "$gte": max(1, self.valves.min_quality_score - 1)
                            }
                        },
                        {"quality_score": {"$exists": False}},
                    ],
                }

                # Execute relaxed query with same vector
                logger.info(f"Progressive query with relaxed filter: {relaxed_filter}")
                start_time = time.time()
                relaxed_res = await self._deduplicated_query(vec, relaxed_filter)
                query_time = time.time() - start_time
                relaxed_matches = getattr(relaxed_res, "matches", []) or []
                logger.info(
                    f"Progressive knowledge query returned {len(relaxed_matches)} matches in {query_time:.3f}s"
                )

                # Only use these relaxed matches if they're better than what we had
                if len(relaxed_matches) > len(matches):
                    logger.info(
                        f"Using progressive query results: {len(relaxed_matches)} matches vs {len(matches)} from original"
                    )
                    matches = relaxed_matches

            # Process the knowledge results
            items = []
            filtered_count = 0
            for m in matches:
                score = getattr(m, "score", 0.0)
                if score < self.valves.min_score_knowledge:
                    filtered_count += 1
                    continue
                # Get quality score from metadata (already filtered at query time, but log for visibility)
                quality_score = m.metadata.get("quality_score", 0)
                if isinstance(quality_score, str):
                    try:
                        quality_score = int(quality_score)
                    except:
                        quality_score = 0
                # UPDATED: Get document name from filename or document_id fields
                label = m.metadata.get("filename") or m.metadata.get(
                    "document_id", "N/A"
                )
                # UPDATED: Use chunk_text instead of text
                text = self._clean_text(m.metadata.get("chunk_text", ""))
                logger.info(
                    f"[KNOWLEDGE MATCH] label={label}, score={score:.4f}, quality={quality_score}"
                )
                items.append(
                    {
                        "type": "knowledge",
                        "text": text,
                        "document_name": label,
                        "score": score,
                        "quality_score": quality_score,
                    }
                )
            if filtered_count > 0:
                logger.info(
                    f"Filtered out {filtered_count} knowledge matches with score < {self.valves.min_score_knowledge}"
                )
            return items
        except Exception as e:
            logger.error(f"Pinecone knowledge query failed: {e}")
            return []

    async def _retrieve_chat_history(
        self, vec: list, chat_id: str, user_id: str
    ) -> list:
        """Retrieve chat history items with quality filter."""
        logger.info("[DEBUG] === STARTING PINECONE CHAT HISTORY RETRIEVAL ===")
        if not user_id or not chat_id:
            logger.info("Missing user_id or chat_id, skipping chat history retrieval")
            return []
        # Chat-specific filter for this user and chat with quality threshold
        filter_expr = {
            "type": {"$eq": "chat"},
            "user_id": {"$eq": user_id},
            "chat_id": {"$eq": chat_id},
            "$or": [
                {"quality_score": {"$gte": self.valves.min_quality_score}},
                {
                    "quality_score": {"$exists": False}
                },  # Include vectors without quality_score field
            ],
        }
        logger.info(
            f"Querying Pinecone for CHAT with filter: {filter_expr} (min_quality_score={self.valves.min_quality_score})"
        )
        try:
            # Execute the query with deduplication
            start_time = time.time()
            res = await self._deduplicated_query(vec, filter_expr)
            query_time = time.time() - start_time
            matches = getattr(res, "matches", []) or []
            logger.info(
                f"Pinecone chat query returned {len(matches)} matches in {query_time:.3f}s"
            )
            
            # Debug: Show all chat matches and their scores first
            if matches:
                logger.info(f"[DEBUG] All Pinecone chat matches with scores (min_score_chat={self.valves.min_score_chat}):")
                for i, match in enumerate(matches):
                    score = getattr(match, "score", 0)
                    metadata = getattr(match, "metadata", {})
                    summary_type = metadata.get("summary_type", "unknown")
                    timestamp = metadata.get("timestamp", "no-timestamp")
                    logger.info(f"[DEBUG]   {i+1}. {summary_type} ({timestamp}) - Score: {score:.4f}")
            else:
                logger.info("[DEBUG] No chat matches returned from Pinecone")

            # Progressive refinement for chat history if needed
            if (
                len(
                    [
                        m
                        for m in matches
                        if getattr(m, "score", 0) >= self.valves.min_score_chat
                    ]
                )
                < 1
            ):
                logger.info(
                    "Initial chat history query returned too few quality matches, trying progressive refinement"
                )

                # For chat history, we can relax the quality score requirement
                relaxed_filter = {
                    "type": {"$eq": "chat"},
                    "user_id": {"$eq": user_id},
                    "chat_id": {"$eq": chat_id},
                    # No quality score filter for chat history fallback
                }

                # Execute relaxed query
                logger.info(f"Progressive chat query with relaxed filter")
                start_time = time.time()
                relaxed_res = await self._deduplicated_query(vec, relaxed_filter)
                query_time = time.time() - start_time
                relaxed_matches = getattr(relaxed_res, "matches", []) or []
                logger.info(
                    f"Progressive chat query returned {len(relaxed_matches)} matches in {query_time:.3f}s"
                )

                # Use these matches if better
                if len(relaxed_matches) > len(matches):
                    logger.info(
                        f"Using progressive chat query results: found {len(relaxed_matches)} matches"
                    )
                    matches = relaxed_matches

            # Process the chat results (never cached)
            items = []
            filtered_count = 0
            for m in matches:
                score = getattr(m, "score", 0.0)
                if score < self.valves.min_score_chat:
                    filtered_count += 1
                    continue
                # Get quality score from metadata (already filtered at query time, but log for visibility)
                quality_score = m.metadata.get("quality_score", 0)
                if isinstance(quality_score, str):
                    try:
                        quality_score = int(quality_score)
                    except:
                        quality_score = 0
                topic = m.metadata.get("topic", "General")
                # UPDATED: Use chunk_text instead of text
                text = self._clean_text(m.metadata.get("chunk_text", ""))
                # Fallback to text field for backwards compatibility with existing records
                if not text:
                    text = self._clean_text(m.metadata.get("text", ""))
                logger.info(
                    f"[CHAT MATCH] topic={topic}, score={score:.4f}, quality={quality_score}"
                )
                items.append(
                    {
                        "type": "chat",
                        "text": text,
                        "document_name": topic,  # Use topic as the display label
                        "score": score,
                        "quality_score": quality_score,
                    }
                )
            if filtered_count > 0:
                logger.info(
                    f"Filtered out {filtered_count} chat matches with score < {self.valves.min_score_chat}"
                )
            
            logger.info(f"[DEBUG] Retrieved {len(items)} chat history items from Pinecone")
            if items:
                logger.info(f"[DEBUG] Pinecone results preview: topic={items[0]['document_name']} (score: {items[0]['score']:.3f})")
            logger.info("[DEBUG] === COMPLETED PINECONE CHAT HISTORY RETRIEVAL ===")
            return items
        except Exception as e:
            logger.error(f"Pinecone chat query failed: {e}")
            logger.info("[DEBUG] === FAILED PINECONE CHAT HISTORY RETRIEVAL ===")
            return []

    async def _retrieve(self, vec: list, chat_id: str, user_id: str, __event_emitter__=None) -> list:
        """Retrieve function with separate strategies for knowledge vs chat content."""
        search_start = time.time()
        logger.info(
            f"Retrieval started for chat_id={repr(chat_id)}, user_id={repr(user_id)}, quality_threshold={self.valves.min_quality_score}"
        )
        if not vec or len(vec) != self.valves.vector_dim:
            logger.warning(
                f"Invalid vector (len={len(vec) if vec else 0}), skipping retrieval"
            )
            return []

        # Create a group of async tasks for concurrent execution
        async def process_batch(queries):
            """Process a batch of retrieval tasks concurrently with controlled concurrency"""
            # Use asyncio.gather to run multiple tasks concurrently but with controlled concurrency
            tasks = []

            for query_info in queries:
                query_type = query_info["type"]
                if query_type == "knowledge":
                    tasks.append(self._retrieve_knowledge(vec, __event_emitter__))
                elif query_type == "chat":
                    tasks.append(self._retrieve_chat_history(vec, chat_id, user_id))

            # Execute all tasks concurrently and gather results
            if tasks:
                results = await asyncio.gather(*tasks)
                return [item for sublist in results for item in sublist]
            return []

        # Define query types
        queries = [
            {"type": "knowledge"},
            {"type": "chat"},
        ]

        # Execute both types of queries using our batch processor
        items = await process_batch(queries)

        # Log performance metrics
        search_time = time.time() - search_start
        knowledge_items = [item for item in items if item["type"] == "knowledge"]
        chat_items = [item for item in items if item["type"] == "chat"]

        logger.info(
            f"Combined retrieval results: {len(knowledge_items)} knowledge items, "
            f"{len(chat_items)} chat items, total={len(items)}, time={search_time:.3f}s "
            f"(using quality filter: min_quality_score={self.valves.min_quality_score})"
        )
        return items

    def _get_chat_id(self, body: dict) -> str:
        meta = body.get("metadata", {})
        return (
            body.get("chat_id")
            or body.get("id")
            or body.get("session_id")
            or meta.get("chat_id")
            or meta.get("session_id")
            or "default"
        )

    async def inlet(
        self, body: Dict[str, Any], *, __user__: Dict[str, Any] = None, __event_emitter__=None, **_
    ) -> Dict[str, Any]:
        __user__ = __user__ or {}
        messages = body.get("messages", []) or []
        _debug_message_log("INLET-BEFORE", messages)
        chat_id = self._get_chat_id(body)
        user_id = __user__.get("id")
        logger.info(f"Processing inlet for chat_id={chat_id}, user_id={user_id}")
        # FIXED: Ensure we properly handle content that might be lists
        user_texts = [
            self._ensure_string(m["content"])
            for m in messages
            if m.get("role") == "user"
        ]
        if not user_texts:
            logger.info(
                "No user message present; skipping embedding and context injection."
            )
            _debug_message_log("INLET-AFTER (no change)", messages)
            return body
        # Clean the input text
        cleaned_text = self._clean_text(user_texts[-1])
        # Store the current query for Ragie
        self.current_query = cleaned_text
        logger.info(f"[DEBUG] User query for retrieval: {cleaned_text[:100]}...")
        vec = await self._embed([cleaned_text])
        items = await self._retrieve(vec, chat_id, user_id, __event_emitter__)
        chat_summaries = [i for i in items if i["type"] == "chat"]
        knowledge_items = [i for i in items if i["type"] == "knowledge"]
        logger.info(
            f"Found {len(chat_summaries)} chat summaries and {len(knowledge_items)} knowledge items "
            f"(with quality filtering: min_quality_score={self.valves.min_quality_score})"
        )
        
        # Emit status event to show count in UI
        if __event_emitter__:
            total_items = len(chat_summaries) + len(knowledge_items)
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"✅ Found {total_items} relevant items ({len(knowledge_items)} knowledge, {len(chat_summaries)} chat history)", "done": True}
            })
        # FIXED: Update startswith to use the safe helper method
        summary_msg = next(
            (
                m
                for m in messages
                if m.get("role") == "system"
                and self._content_startswith(m.get("content", ""), "Chat summary:")
            ),
            None,
        )
        if chat_summaries and not summary_msg:
            best_summary = max(chat_summaries, key=lambda x: x["score"])
            summary_msg = {
                "role": "system",
                "content": f"Chat summary: {best_summary['text']}",
            }
            logger.info(
                f"Using retrieved chat summary with score={best_summary['score']:.4f}, quality={best_summary.get('quality_score', 'N/A')}"
            )
        user_assistant_msgs = [
            m for m in messages if m.get("role") in ("user", "assistant")
        ]
        rag_blocks = []
        if knowledge_items:
            rag_blocks = [
                f"{i['document_name']}: {i['text']}"
                for i in knowledge_items[: self.valves.display_k]
            ]
            # Debug: Log what's being sent to the LLM
            for i, block in enumerate(rag_blocks):
                if "/api/proxy/ragie/stream" in block:
                    logger.info(f"[DEBUG] RAG block {i+1} contains proxy URLs: ...{block[-200:]}")
                else:
                    logger.info(f"[DEBUG] RAG block {i+1} has NO proxy URLs: {block[:100]}...")
        # FIXED: Update startswith to use the safe helper method
        filtered = [
            m
            for m in messages
            if m.get("role") != "system"
            or (
                not self._content_startswith(m.get("content", ""), "Chat summary:")
                and not self._content_startswith(
                    m.get("content", ""), "Relevant knowledge:"
                )
            )
        ]
        if summary_msg:
            new_msg_list = [summary_msg]
            if rag_blocks:
                new_msg_list.append(
                    {
                        "role": "system",
                        "content": "Relevant knowledge:\n" + "\n".join(rag_blocks),
                    }
                )
            new_msg_list.extend(user_assistant_msgs[-self.valves.keep_recent * 2 :])
            logger.info(
                f"[INLET] Using summary + {len(new_msg_list)-1} recent messages"
            )
        else:
            new_msg_list = []
            if rag_blocks:
                logger.info(
                    f"Injecting {len(rag_blocks)} RAG context items into prompt."
                )
                new_msg_list.append(
                    {
                        "role": "system",
                        "content": "Relevant knowledge:\n" + "\n".join(rag_blocks),
                    }
                )
            if len(user_assistant_msgs) > self.valves.keep_without_summary * 2:
                logger.info(
                    f"[INLET] No summary available, limiting to {self.valves.keep_without_summary} recent turns"
                )
                new_msg_list.extend(
                    user_assistant_msgs[-self.valves.keep_without_summary * 2 :]
                )
            else:
                logger.info(
                    f"[INLET] No summary available, using all {len(user_assistant_msgs)//2} turns"
                )
                new_msg_list.extend(user_assistant_msgs)
        # FIXED: Update startswith to use the safe helper method
        if summary_msg and not any(
            self._content_startswith(m.get("content", ""), "Chat summary:")
            for m in messages
        ):
            logger.info(
                "Adding a new chat summary from Pinecone (none found in incoming messages)"
            )
        _debug_message_log("INLET-AFTER", new_msg_list)
        body["messages"] = new_msg_list
        logger.info(f"Inlet processing completed for chat_id={chat_id}")
        return body

    async def outlet(
        self, body: Dict[str, Any], *, __user__: Dict[str, Any] = None, **_
    ) -> Dict[str, Any]:
        __user__ = __user__ or {}
        messages = body.get("messages", []) or []
        _debug_message_log("OUTLET-BEFORE", messages)
        chat_id = self._get_chat_id(body)
        user_id = __user__.get("id")
        logger.info(f"Processing outlet for chat_id={chat_id}, user_id={user_id}")
        if not user_id:
            logger.warning("Missing user_id in request, summary will not be stored")
            return body
        chat_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        if len(chat_msgs) < self.valves.turns_before * 2:
            logger.info("Not enough chat rounds for summarization yet.")
            _debug_message_log("OUTLET-AFTER (no change)", messages)
            return body
        # We've determined we need a summary, but we'll process it asynchronously
        logger.info(
            f"OUTLET: Scheduling background summary generation for chat_id={chat_id}, user_id={user_id}"
        )
        # Get existing messages without summaries
        # FIXED: Update startswith to use the safe helper method
        filtered = [
            m
            for m in messages
            if not (
                m.get("role") == "system"
                and self._content_startswith(m.get("content", ""), "Chat summary:")
            )
        ]
        user_assistant_msgs = [
            m for m in filtered if m.get("role") in ("user", "assistant")
        ]
        last_exchange = chat_msgs[-2:]  # Last user-assistant exchange for summarization
        # Schedule the background task for summarization
        # This is the key change - summarization happens in the background
        asyncio.create_task(
            self._generate_and_store_summary(chat_id, user_id, last_exchange)
        )
        # Just return the messages without waiting for summary
        logger.info(
            f"Outlet processing completed for chat_id={chat_id} (summary generation scheduled in background)"
        )
        _debug_message_log(
            "OUTLET-AFTER (no change, summary will be generated in background)",
            messages,
        )
        return body

    # Optimized OpenRouter API call with proper timeout and retry handling
    async def _call_openrouter(
        self, messages, model=None, timeout_seconds=15.0, retries=2
    ):
        """Make an OpenRouter API call with timeout protection and retries"""
        model = model or self.valves.openrouter_model
        logger.info(
            f"Calling OpenRouter API with model {model}, timeout={timeout_seconds}s"
        )
        retry_count = 0
        last_error = None
        while retry_count <= retries:
            try:
                # Use our OpenRouter limiter for backpressure control
                async with self.openrouter_limiter:
                    logger.debug(
                        f"Acquired OpenRouter limiter, making API call (attempt {retry_count+1}/{retries+1})"
                    )
                    # Add timeout protection
                    return await asyncio.wait_for(
                        self.chat_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.3,  # Lower temperature for more consistent results
                        ),
                        timeout=timeout_seconds,
                    )
            except asyncio.TimeoutError:
                last_error = (
                    f"OpenRouter API call timed out after {timeout_seconds} seconds"
                )
                logger.warning(f"Timeout {retry_count+1}/{retries+1}: {last_error}")
            except Exception as e:
                last_error = f"OpenRouter API error: {str(e)}"
                logger.warning(f"Error {retry_count+1}/{retries+1}: {last_error}")
            # If we get here, there was an error - increment counter and retry if remaining
            retry_count += 1
            if retry_count <= retries:
                # Wait with exponential backoff before retrying
                wait_time = 1.0 * (2 ** (retry_count - 1))  # 1, 2, 4, 8 seconds...
                logger.info(
                    f"Waiting {wait_time}s before retry {retry_count}/{retries}"
                )
                await asyncio.sleep(wait_time)
        # If we get here, all retries have failed
        raise Exception(last_error)

    async def _generate_fallback_summary(self, last_exchange_processed):
        """Generate a basic summary when the API call fails"""
        try:
            # Extract user question
            user_content = self._ensure_string(
                next(
                    (
                        m.get("content", "")
                        for m in last_exchange_processed
                        if m.get("role") == "user"
                    ),
                    "",
                )
            )
            # Extract first line/sentence of assistant response
            assistant_content = self._ensure_string(
                next(
                    (
                        m.get("content", "")
                        for m in last_exchange_processed
                        if m.get("role") == "assistant"
                    ),
                    "",
                )
            )
            # Clean both texts before processing
            user_content = self._clean_text(user_content)
            assistant_content = self._clean_text(assistant_content)
            # Create a simplified summary using just the user question and first part of the answer
            user_summary = user_content[:150] + (
                "..." if len(user_content) > 150 else ""
            )
            # Get first sentence or paragraph of assistant response
            first_para = (
                assistant_content.split("\n\n")[0]
                if "\n\n" in assistant_content
                else assistant_content
            )
            first_sentence = (
                first_para.split(". ")[0] if ". " in first_para else first_para
            )
            assistant_summary = first_sentence[:200] + (
                "..." if len(first_sentence) > 200 else ""
            )
            # Combine into a simple summary
            summary = f"The user asked about {user_summary}. The assistant provided information about {assistant_summary}."
            return summary
        except Exception as e:
            logger.error(f"Error generating fallback summary: {e}")
            # Even simpler fallback if we can't extract anything
            return "The user and assistant had a conversation. The details could not be summarized automatically."

    async def _generate_and_store_summary(
        self, chat_id: str, user_id: str, last_exchange: List[Dict[str, Any]]
    ):
        """Generate and store summary in the background without blocking the user interface"""
        try:
            # FIXED: Ensure content is properly handled when it might be a list
            last_exchange_processed = []
            for msg in last_exchange:
                content = self._ensure_string(msg.get("content", ""))
                # Clean text before processing
                content = self._clean_text(content)
                last_exchange_processed.append(
                    {"role": msg.get("role", ""), "content": content}
                )
            uc = "\n".join(
                f"{m['role']}: {m['content']}" for m in last_exchange_processed
            )
            logger.info(
                f"Background: Requesting summary from OpenRouter for chat {chat_id}"
            )
            summary = ""
            using_fallback = False
            try:
                # Generate summary using the optimized API call with retries
                resp = await self._call_openrouter(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert summarization assistant who creates comprehensive, informative summaries. "
                                "When summarizing the conversation exchange between the user and assistant:"
                                "\n\n1. Begin with a clear statement of the user's primary question or request"
                                "\n2. Provide a thorough account of the assistant's response, including key facts, explanations, and comparisons"
                                "\n3. Include specific numerical data, measurements, and scientific details when present"
                                "\n4. Capture any interesting nuances or additional context the assistant provided"
                                "\n5. Structure the summary in 2-3 well-developed paragraphs that flow logically"
                                "\n6. Write in a neutral, informative tone while maintaining clarity"
                                "\n7. Use complete sentences with proper transitions between ideas"
                                "\n\nYour summaries should be detailed enough to serve as a useful reference for the conversation, "
                                "approximately 100-150 words, neither overly concise nor unnecessarily verbose."
                            ),
                        },
                        {"role": "user", "content": uc},
                    ]
                )
                summary = resp.choices[0].message.content.strip()
            except Exception as api_error:
                # API call failed, use fallback summary
                logger.warning(
                    f"Summary API failed: {api_error}. Using fallback summary generation."
                )
                summary = await self._generate_fallback_summary(last_exchange_processed)
                using_fallback = True
            # Clean the summary text to ensure proper formatting
            summary = self._clean_text(summary)
            preview = summary[:100].replace("\n", " ")
            logger.info(
                f"Background: Summary generated for chat_id={chat_id}: {preview}... {'(FALLBACK)' if using_fallback else ''}"
            )
            # Apply content-aware chunking to the summary if it's long
            summary_chunks = []
            if len(summary) > 2000:
                logger.info(
                    f"Summary is long ({len(summary)} chars), applying content-aware chunking"
                )
                try:
                    # Use the content-aware splitter directly for chat summaries
                    content_splitter = ContentAwareTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    summary_chunks = content_splitter.split_text(summary)
                    logger.info(
                        f"Created {len(summary_chunks)} content-aware chunks from the summary"
                    )
                except Exception as chunk_err:
                    logger.error(
                        f"Content-aware chunking error: {chunk_err}, using full summary"
                    )
                    summary_chunks = [summary]
            else:
                summary_chunks = [summary]

            # Process chunks in parallel using our improved batching approach
            async def process_chunks():
                # First, determine batch size based on number of chunks
                batch_size = min(
                    5, len(summary_chunks)
                )  # Process up to 5 chunks at a time

                processed_chunks = []
                # Process chunks in batches
                for i in range(0, len(summary_chunks), batch_size):
                    batch = summary_chunks[i : i + batch_size]

                    # Create tasks for this batch
                    batch_tasks = []
                    for j, chunk in enumerate(batch):
                        # Clean text for chat processing
                        cleaned_chunk = self._clean_text(chunk)
                        # Quality scoring (CPU-bound, good for task batching)
                        quality_score = self.document_processor.quick_quality_score(
                            cleaned_chunk
                        )
                        # Track processed info
                        processed_chunks.append(
                            {
                                "text": cleaned_chunk,
                                "quality_score": quality_score,
                                "index": i + j,
                            }
                        )

                        # Create embedding task for this chunk
                        batch_tasks.append(self._embed([cleaned_chunk]))

                    # Run embedding tasks concurrently for this batch
                    logger.info(
                        f"Processing embeddings for batch of {len(batch)} chunks"
                    )
                    batch_vectors = await asyncio.gather(*batch_tasks)

                    # Assign vectors to chunks
                    for j, vector in enumerate(batch_vectors):
                        chunk_idx = i + j
                        if chunk_idx < len(processed_chunks):
                            processed_chunks[chunk_idx]["vector"] = vector

                return processed_chunks

            # Run the chunk processing
            processed_chunks = await process_chunks()
            logger.info(f"Processed {len(processed_chunks)} chunks with embeddings")

            # Generate topic classification - handle async with our backpressure control
            topic = "General"
            try:
                if using_fallback:
                    # Simple topic extraction when already in fallback mode
                    # Just use the first word of the user message if available
                    user_content = next(
                        (
                            m.get("content", "")
                            for m in last_exchange_processed
                            if m.get("role") == "user"
                        ),
                        "",
                    )
                    first_word = (
                        user_content.strip().split()[0]
                        if user_content.strip()
                        else "General"
                    )
                    topic = (
                        first_word if first_word and len(first_word) > 3 else "General"
                    )
                else:
                    # Try API call for topic with shorter timeout
                    async with self.openrouter_limiter:
                        topic_resp = await self._call_openrouter(
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an expert topic classifier. Your task is to identify the main topic of a conversation "
                                        "summary and provide a SINGLE WORD that best represents the topic. Just return the single word, "
                                        "with no additional explanation, punctuation, or formatting."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": f"Identify the main topic of this conversation summary in a single word:\n\n{summary}",
                                },
                            ],
                            timeout_seconds=8.0,  # Shorter timeout for topic
                        )
                        topic = topic_resp.choices[0].message.content.strip()
                # Ensure it's just a single word (take first word if multiple were provided)
                topic = topic.split()[0] if topic else "General"
                # Remove any punctuation
                topic = "".join(c for c in topic if c.isalnum())
            except Exception as topic_error:
                logger.warning(
                    f"Topic classification failed: {topic_error}. Using 'General' topic."
                )
                topic = "General"

            # Build metadata and vectors for each chunk
            upsert_data = []
            for i, chunk_info in enumerate(processed_chunks):
                chunk_text = chunk_info["text"]
                chunk_vec = chunk_info.get("vector", [0.0] * self.valves.vector_dim)
                chunk_quality = chunk_info["quality_score"]

                # Create unique ID for this chunk
                rec_id = f"{chat_id}-{int(time.time())}-{i}"

                # Build metadata
                metadata = {
                    "type": "chat",
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "chunk_text": self._clean_for_pinecone(
                        chunk_text
                    ),  # Use enhanced cleaning for Pinecone
                    "summary_id": rec_id,
                    "summary_length": len(chunk_text),
                    "vector_dim": len(chunk_vec),
                    "model": self.valves.openrouter_model,
                    "topic": topic,  # Single word topic
                    "fallback": using_fallback,  # Flag if we used fallback
                    "chunk_index": i,  # Position in the original summary
                    "total_chunks": len(processed_chunks),  # Total number of chunks
                    "quality_score": chunk_quality,  # Quality score
                    "doc_type": "chat_summary",  # Document type
                }

                # Add to upsert data
                upsert_data.append(
                    {
                        "id": rec_id,
                        "values": [
                            float(v) if v is not None else 0.0 for v in chunk_vec
                        ],  # Better handling of vector values
                        "metadata": metadata,
                    }
                )

            logger.info(
                f"Background: Prepared {len(upsert_data)} items for upsert - chat_id={chat_id}, topic={topic}"
            )

            # Ensure vector is JSON serializable
            try:
                json.dumps(upsert_data)
                logger.info("Background: Upsert data is JSON serializable")
            except Exception as json_err:
                logger.error(
                    f"Background: Upsert data is not JSON serializable: {json_err}"
                )
                # Fix any issues with vector values
                for item in upsert_data:
                    item["values"] = [
                        float(v) if v is not None else 0.0 for v in item["values"]
                    ]
                logger.info(
                    "Background: Converted all vectors to lists of floats for serialization"
                )

            # Use Retry object for RQ
            job = self.rq_queue.enqueue(
                "tasks.background_upsert",
                upsert_data,
                self.valves.namespace,
                retry=Retry(max=3),
                ttl=3600,
            )
            logger.info(
                f"Background: Scheduled upsert job {job.id} in Redis queue for {len(upsert_data)} vectors"
            )
        except Exception as e:
            logger.error(
                f"Background summary generation and storage failed: {e}", exc_info=True
            )

    # Content-aware document processing method for external use
    async def process_document(self, text, document_id=None, metadata=None):
        """
        Process a document with content-aware chunking and prepare for vectorization.
        Args:
            text: Document text to process
            document_id: Unique identifier for the document (generated if not provided)
            metadata: Additional metadata to include
        Returns:
            Dict with processed document info and chunks
        """
        if not document_id:
            document_id = f"doc_{uuid.uuid4()}"
        if not metadata:
            metadata = {}
        # Clean the input text
        cleaned_text = self._clean_text(text)
        start_time = time.time()
        logger.info(f"Processing document {document_id} ({len(cleaned_text)} chars)")
        # Process text into content-aware chunks
        chunks, enhanced_metadata = await self.document_processor.process_text(
            cleaned_text, metadata
        )
        # Clean each chunk
        cleaned_chunks = [self._clean_text(chunk) for chunk in chunks]
        # Add document_id to metadata
        enhanced_metadata["document_id"] = document_id
        # Generate embeddings for all chunks in a batch to optimize API calls
        vectors = await self._batch_embed(cleaned_chunks)
        # Create records for each chunk
        records = []
        for i, (chunk, vector) in enumerate(zip(cleaned_chunks, vectors)):
            # Calculate quality score
            quality_score = self.document_processor.quick_quality_score(chunk)
            # Position metadata
            relative_position = i / len(chunks)
            is_intro = relative_position < 0.2  # First 20% of document
            is_conclusion = relative_position > 0.8  # Last 20% of document
            # Create chunk metadata
            chunk_metadata = {
                **enhanced_metadata,  # Include all enhanced metadata
                "chunk_id": i,
                "chunk_text": self._clean_for_pinecone(
                    chunk
                ),  # Extra cleaning for Pinecone
                "total_chunks": len(chunks),
                "position_in_document": i,
                "relative_position": relative_position,
                "is_intro": is_intro,
                "is_conclusion": is_conclusion,
                "quality_score": quality_score,
                "vector_id": f"{document_id}_{i}",
            }
            records.append(
                {
                    "id": f"{document_id}_{i}",
                    "values": [
                        float(v) if v is not None else 0.0 for v in vector
                    ],  # Better handling of vector values
                    "metadata": chunk_metadata,
                }
            )
        processing_time = time.time() - start_time
        logger.info(
            f"Document processing complete: {document_id}, "
            f"chunks={len(chunks)}, time={processing_time:.2f}s"
        )
        return {
            "document_id": document_id,
            "chunks": len(chunks),
            "metadata": enhanced_metadata,
            "records": records,
            "processing_time": processing_time,
        }

    # Debug method for queue status
    def debug_queue_status(self):
        """Debug the status of the Redis queue"""
        try:
            queue_length = self.rq_queue.count
            job_ids = self.rq_queue.job_ids
            failed_job_registry = self.rq_queue.failed_job_registry
            failed_job_ids = (
                failed_job_registry.get_job_ids() if failed_job_registry else []
            )
            logger.info(f"Queue status - Length: {queue_length}")
            logger.info(f"Queue job IDs: {job_ids}")
            logger.info(f"Failed job IDs: {failed_job_ids}")
            # Check individual jobs
            for job_id in job_ids:
                job = self.rq_queue.fetch_job(job_id)
                if job:
                    logger.info(
                        f"Job {job_id} - Status: {job.get_status()}, Function: {job.func_name}"
                    )
            for job_id in failed_job_ids:
                job = self.rq_queue.fetch_job(job_id)
                if job:
                    logger.info(f"Failed job {job_id} - Exception: {job.exc_info}")
        except Exception as e:
            logger.error(f"Error checking queue status: {e}")

    async def __aenter__(self):
        logger.info("Entered Filter async context manager.")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        logger.info("Exiting Filter context manager, closing http client.")
        await self.http_client.aclose()


# Enhanced text processing and content-aware chunking setup
class ContentAwareTextSplitter:
    """
    Content-aware text splitter optimized for chat and conversation content.
    Recognizes conversation structure, maintains context, and creates semantically meaningful chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        min_chunk_size: int = 100,
        max_chunk_size: int = 3000,
        preserve_conversation_flow: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_conversation_flow = preserve_conversation_flow

        # Patterns for detecting conversation structure
        self.conversation_patterns = [
            # Speaker indicators
            (r"^(User|Assistant|Human|AI|System):\s*", "speaker_change"),
            # Conversation turns
            (r"\n\n(?=\w)", "conversation_break"),
            # Questions and responses
            (r"\?\s*\n", "question_end"),
            # Topic transitions
            (
                r"\b(Now|Next|Moving on|In conclusion|To summarize)\b",
                "topic_transition",
            ),
            # Paragraph boundaries
            (r"\n\s*\n", "paragraph_break"),
        ]

        logger.info(
            f"ContentAwareTextSplitter initialized: chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def clean_text(self, text):
        """
        Clean text for chat processing using consistent enhanced approach.
        """
        if not text:
            return ""

        # First, convert to string if needed
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
            # Enhanced cleaning for chat content processing
            try:
                # Handle HTML entities and URL encoding
                text = unescape(text)
                text = unquote(text)

                # Normalize Unicode characters
                text = unicodedata.normalize("NFKC", text)

                # Remove control characters but keep newlines and tabs
                control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
                text = control_chars.sub("", text)
            except:
                pass

            # This two-stage replacement handles the common case in your examples
            text = text.replace("\\n\\n", "\n\n")  # First replace double newlines
            text = text.replace("\\n", "\n")  # Then handle single newlines

            # Handle other escape sequences
            text = text.replace("\\t", "\t")
            text = text.replace("\\r", "")
            text = text.replace('\\"', '"')
            text = text.replace("\\'", "'")
            text = text.replace("\\\\", "\\")

            # Enhanced whitespace normalization
            try:
                text = re.sub(r" {2,}", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                lines = text.split("\n")
                lines = [line.strip() for line in lines]
                text = "\n".join(lines)
            except:
                pass

        return text.strip() if isinstance(text, str) else ""

    def _detect_conversation_structure(self, text: str) -> list:
        """
        Analyze text to identify conversation structure elements.
        Returns list of (start_pos, end_pos, element_type, content)
        """
        elements = []
        lines = text.split("\n")
        current_pos = 0

        logger.debug(f"Analyzing conversation structure for {len(lines)} lines...")

        for i, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line)

            # Check for conversation patterns
            for pattern, element_type in self.conversation_patterns:
                if re.search(pattern, line.strip(), re.IGNORECASE):
                    elements.append((line_start, line_end, element_type, line.strip()))
                    logger.debug(f"Found {element_type}: '{line.strip()[:30]}...'")
                    break

            current_pos = line_end + 1  # +1 for newline

        logger.debug(f"Detected {len(elements)} conversation structure elements")
        return elements

    def _find_optimal_split_points(self, text: str, elements: list) -> list:
        """
        Find optimal split points based on conversation structure.
        """
        split_points = [0]
        elements.sort(key=lambda x: x[0])

        current_chunk_start = 0
        logger.debug(f"Finding optimal split points (target: {self.chunk_size} chars)")

        for start_pos, end_pos, element_type, content in elements:
            distance_from_start = start_pos - current_chunk_start

            # Prioritize conversation breaks for splitting
            if (
                distance_from_start >= self.chunk_size - self.chunk_overlap
                and element_type
                in ["conversation_break", "speaker_change", "paragraph_break"]
            ):

                if distance_from_start >= self.min_chunk_size:
                    split_points.append(start_pos)
                    logger.debug(f"Split at {start_pos} ({element_type})")
                    current_chunk_start = start_pos

            # Force split if exceeding max size
            elif distance_from_start >= self.max_chunk_size:
                split_points.append(start_pos)
                logger.debug(f"Forced split at {start_pos} (max size exceeded)")
                current_chunk_start = start_pos

        if split_points[-1] != len(text):
            split_points.append(len(text))

        logger.debug(f"Generated {len(split_points)} split points")
        return split_points

    def _handle_overlap(self, text: str, split_points: list) -> list:
        """
        Adjust split points to include meaningful overlap between chunks.
        """
        if len(split_points) <= 2:
            return split_points

        adjusted_points = [split_points[0]]

        for i in range(1, len(split_points) - 1):
            current_start = split_points[i]

            # For conversation content, look for sentence boundaries for overlap
            overlap_start = max(adjusted_points[-1], current_start - self.chunk_overlap)

            # Try to find a sentence boundary within the overlap region
            overlap_text = text[overlap_start:current_start]
            sentence_end = overlap_text.rfind(". ")
            if sentence_end > 0:
                overlap_start = overlap_start + sentence_end + 2

            adjusted_points.append(overlap_start)

        adjusted_points.append(split_points[-1])
        return adjusted_points

    def split_text(self, text: str) -> list:
        """
        Split text into content-aware chunks optimized for conversation content.
        """
        if not text or len(text) <= self.min_chunk_size:
            return [text] if text else []

        logger.debug(
            f"Starting content-aware text splitting for {len(text)} characters"
        )

        try:
            # Clean the text first using consistent approach
            cleaned_text = self.clean_text(text)

            # Detect conversation structure
            elements = self._detect_conversation_structure(cleaned_text)

            # Find optimal split points
            split_points = self._find_optimal_split_points(cleaned_text, elements)

            # Handle overlap
            split_points = self._handle_overlap(cleaned_text, split_points)

            # Create chunks
            chunks = []
            for i in range(len(split_points) - 1):
                start = split_points[i]
                end = split_points[i + 1]

                chunk_text = cleaned_text[start:end].strip()

                if chunk_text and len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                elif chunks and chunk_text:
                    # Merge small chunks with previous chunk
                    chunks[-1] += "\n\n" + chunk_text

            logger.debug(f"Content-aware splitting completed: {len(chunks)} chunks")

            if not chunks:
                # Fallback to simple splitting
                return self._fallback_split(cleaned_text)

            return chunks

        except Exception as e:
            logger.warning(f"Content-aware splitting failed: {e}. Using fallback.")
            return self._fallback_split(text)

    def _fallback_split(self, text: str) -> list:
        """
        Fallback to recursive character splitting.
        """
        logger.debug(f"Using fallback recursive splitting for {len(text)} characters")
        chunks = recursive_splitter.split_text(text)
        logger.debug(f"Fallback splitting produced {len(chunks)} chunks")
        return chunks


dotenv_path = os.getenv("DOTENV_PATH")