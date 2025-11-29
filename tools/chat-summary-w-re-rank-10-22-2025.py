"""
Advanced Chat Summarization Filter for OpenWebUI

This filter provides intelligent chat summarization and retrieval using Pinecone:
1. Chat History Management: Uses Pinecone for intelligent chat summarization and retrieval
2. Pinecone Reranking: Uses Pinecone's hosted reranking models for improved relevance

Key Features:
- Pinecone vector database for chat history storage and retrieval
- Content-aware text splitting optimized for conversations
- Intelligent quality scoring and filtering
- Comprehensive caching and rate limiting
- Background processing with Redis/RQ
- Professional error handling and logging
- Pinecone reranking for improved search relevance

Architecture:
- Pinecone: Chat summaries and conversation history (type="chat")
- Pinecone Reranking: Uses Pinecone's inference API for document reranking

Refactored Structure:
- Organized into focused service classes within single file
- Clean separation of concerns
- Orchestrator pattern for coordination
- Performance optimized with no code duplication
"""

import os
import logging
import asyncio
import time
import re
import json
import hashlib
import uuid
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set
from functools import lru_cache

import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from redis import Redis
from rq import Queue, Retry
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from tiktoken import get_encoding

# Try to import unstructured.io for advanced document processing
try:
    from unstructured.partition.auto import partition
    from unstructured.chunking.title import chunk_by_title

    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logging.warning(
        "Unstructured.io not available. Document processing will be limited."
    )

# Load environment variables
dotenv_path = os.getenv("DOTENV_PATH")
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chat_history_filter_optimized")

# Configure token-aware and semantic splitters (module level)
encoding = get_encoding("cl100k_base")
token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base", chunk_size=1500, chunk_overlap=150
)
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    chunk_size=1500,
    chunk_overlap=150,
)


def _debug_message_log(stage: str, messages: list) -> None:
    """Debug logging helper for message state"""
    logger.info(
        f"[DEBUG] {stage}: num={len(messages)} roles={[m.get('role') for m in messages]}"
    )
    logger.info(
        f"[DEBUG] {stage} SYSTEM MSGS: {[(m.get('content','')[:60]) for m in messages if m.get('role')=='system']}"
    )
    logger.info(
        f"[DEBUG] {stage} FIRST MSGS: {[m.get('content','')[:40] for m in messages[:5]]}"
    )


# ============================================================================
# UTILITY CLASSES
# ============================================================================


class AsyncLimiter:
    """
    Rate limiter for async operations that implements both concurrency limiting and
    time-based rate limiting to prevent overwhelming external services.
    """

    def __init__(self, max_concurrent=10, max_per_second=5):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.max_per_second = max_per_second
        self.window_size = 1.0
        self.task_times = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to proceed, implementing both concurrency and rate limiting"""
        async with self.lock:
            now = time.time()
            self.task_times = [t for t in self.task_times if now - t < self.window_size]

            if len(self.task_times) >= self.max_per_second:
                wait_time = self.window_size - (now - self.task_times[0])
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.3f}s")
                    await asyncio.sleep(wait_time)

            self.task_times.append(time.time())

        await self.sem.acquire()

    def release(self):
        """Release a permit, allowing another operation to proceed"""
        self.sem.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


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
                # Note: unescape and unquote removed as they're no longer needed
                text = unicodedata.normalize("NFKC", text)

                # Remove control characters but keep newlines and tabs
                control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
                text = control_chars.sub("", text)
            except:
                pass

            # Handle escape sequences - order matters!
            # First handle double backslashes to avoid conflicts
            text = text.replace("\\\\", "\\")
            # Then handle specific escape sequences
            text = text.replace("\\n\\n", "\n\n")
            text = text.replace("\\n", "\n")
            text = text.replace("\\t", "\t")
            text = text.replace("\\r", "")
            text = text.replace('\\"', '"')
            text = text.replace("\\'", "'")
            # Handle any remaining escaped characters (but preserve intentional backslashes)
            text = re.sub(r"\\([^\\])", r"\1", text)

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
    def clean_document_name(document_name: str) -> str:
        """Clean document name for better display"""
        if not document_name:
            return "Unknown Document"

        name = document_name
        if "." in name:
            name_parts = name.rsplit(".", 1)
            if len(name_parts) == 2 and len(name_parts[1]) <= 5:
                name = name_parts[0]

        name = name.replace("_", " ").replace("-", " ")
        name = " ".join(word.capitalize() for word in name.split())

        return name

    @staticmethod
    def ensure_string(text) -> str:
        """Ensure the input is a string"""
        if isinstance(text, list):
            return "\n".join(str(item) for item in text)
        elif text is None:
            return ""
        return str(text)

    @staticmethod
    def _clean_summary_text(text: str) -> str:
        """Additional cleaning specifically for summary text to remove escape sequences"""
        if not text:
            return ""

        # Remove any remaining escape sequences that might have been missed
        # This is more aggressive cleaning for summary text
        text = re.sub(r"\\(.)", r"\1", text)  # Remove any remaining backslash escapes

        # Clean up any double quotes that might have been escaped
        text = text.replace('""', '"')  # Remove double quotes
        text = text.replace('" "', '"')  # Remove quotes around spaces

        # Normalize spacing around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)  # Remove spaces before punctuation
        text = re.sub(
            r"([.,!?;:])\s*([.,!?;:])", r"\1\2", text
        )  # Remove spaces between punctuation

        # Clean up any remaining formatting issues
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = text.strip()

        return text


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================


class CacheManager:
    """Manages Redis caching for embeddings and query results"""

    def __init__(
        self, redis_conn: Redis, embed_ttl: int, search_ttl: int, knowledge_ttl: int
    ):
        self.redis = redis_conn
        self.embed_ttl = embed_ttl
        self.search_ttl = search_ttl
        self.knowledge_ttl = knowledge_ttl

    @lru_cache(maxsize=1000)
    def get_embed_cache_key(self, text: str) -> str:
        """Generate a standardized cache key for embeddings (LRU cached for performance)"""
        hash_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"emb:{hash_key}"

    async def get_cached_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding vector"""
        key = self.get_embed_cache_key(text)
        try:
            cached = await asyncio.to_thread(self.redis.get, key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Cache read error for key={key}: {e}")
        return None

    async def cache_embedding(self, text: str, vector: list) -> None:
        """Cache an embedding vector"""
        key = self.get_embed_cache_key(text)
        try:
            cache_data = json.dumps(vector)
            await asyncio.to_thread(self.redis.set, key, cache_data, ex=self.embed_ttl)
        except Exception as e:
            logger.error(f"Cache storage error for key={key}: {e}")


# ============================================================================
# EMBEDDING SERVICE
# ============================================================================


class EmbeddingService:
    """Handles embedding generation with caching and rate limiting"""

    def __init__(
        self,
        client: AsyncOpenAI,
        cache_manager: CacheManager,
        limiter: AsyncLimiter,
        model: str,
        vector_dim: int,
    ):
        self.client = client
        self.cache = cache_manager
        self.limiter = limiter
        self.model = model
        self.vector_dim = vector_dim

        # Metrics
        self.embed_calls = 0
        self.embed_cache_hits = 0
        self.embed_errors = 0
        self.last_metrics_time = time.time()

    async def embed_single(self, text: str) -> list:
        """Generate embedding for a single text (convenience method)"""
        results = await self.embed_batch([text])
        return results[0] if results else [0.0] * self.vector_dim

    async def embed_batch(self, texts_list: list) -> list:
        """Efficiently batch embedding API calls with caching"""
        if not texts_list:
            return []

        # Check cache and identify texts needing embedding
        results = []
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts_list):
            text = TextCleaner.ensure_string(text)
            text = TextCleaner.clean_text(text)

            if not text or not text.strip():
                results.append([0.0] * self.vector_dim)
                continue

            # Check cache
            cached = await self.cache.get_cached_embedding(text)
            if cached:
                self.embed_cache_hits += 1
                results.append(cached)
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                results.append(None)

        # If everything was cached, return early
        if not texts_to_embed:
            self._log_metrics()
            return results

        self.embed_calls += 1

        # Process in batches for optimal throughput
        batch_size = 10

        if len(texts_to_embed) > batch_size:
            await self._process_large_batch(
                texts_to_embed, indices_to_embed, results, batch_size
            )
        else:
            await self._process_small_batch(texts_to_embed, indices_to_embed, results)

        self._log_metrics()
        return results

    async def _process_large_batch(
        self,
        texts_to_embed: list,
        indices_to_embed: list,
        results: list,
        batch_size: int,
    ) -> None:
        """Process large batch of embeddings"""
        logger.info(
            f"Processing {len(texts_to_embed)} embeddings in batches of {batch_size}"
        )

        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i : i + batch_size]
            batch_indices = indices_to_embed[i : i + batch_size]

            async with self.limiter:
                await self._embed_and_cache_batch(
                    batch_texts, batch_indices, results, timeout=10.0
                )

    async def _process_small_batch(
        self, texts_to_embed: list, indices_to_embed: list, results: list
    ) -> None:
        """Process small batch of embeddings"""
        async with self.limiter:
            await self._embed_and_cache_batch(
                texts_to_embed, indices_to_embed, results, timeout=8.0
            )

    async def _embed_and_cache_batch(
        self, texts: list, indices: list, results: list, timeout: float
    ) -> None:
        """Make API call and cache results"""
        try:
            logger.info(f"Embedding API call for {len(texts)} texts")
            start_time = time.time()

            resp = await asyncio.wait_for(
                self.client.embeddings.create(model=self.model, input=texts),
                timeout=timeout,
            )

            elapsed = time.time() - start_time
            logger.info(f"Embedding API call completed in {elapsed:.2f}s")

            # Process and cache results
            if resp.data and len(resp.data) == len(texts):
                for idx, embed_resp, text in zip(indices, resp.data, texts):
                    vector = embed_resp.embedding
                    await self.cache.cache_embedding(text, vector)
                    results[idx] = vector
            else:
                logger.error("Unexpected embedding API response format")
                self.embed_errors += 1
                self._fill_with_zeros(indices, results)

        except asyncio.TimeoutError:
            logger.error(f"Embedding API timeout after {timeout} seconds")
            self.embed_errors += 1
            self._fill_with_zeros(indices, results)

        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            self.embed_errors += 1
            self._fill_with_zeros(indices, results)

    def _fill_with_zeros(self, indices: list, results: list) -> None:
        """Fill failed embeddings with zero vectors"""
        for idx in indices:
            if results[idx] is None:
                results[idx] = [0.0] * self.vector_dim

    def _log_metrics(self, force=False) -> None:
        """Log embedding metrics periodically"""
        now = time.time()
        if force or (now - self.last_metrics_time > 300):
            total = self.embed_calls or 1
            hit_rate = (self.embed_cache_hits / total) * 100 if total > 0 else 0
            logger.info(
                f"Embedding metrics: calls={self.embed_calls}, "
                f"cache_hits={self.embed_cache_hits} ({hit_rate:.1f}%), "
                f"errors={self.embed_errors}"
            )
            self.last_metrics_time = now


# ============================================================================
# VECTOR UTILITIES
# ============================================================================


class VectorUtils:
    """Utility functions for vector operations"""

    @staticmethod
    def hash_vector(vec: list) -> str:
        """Create a stable hash of vector for caching"""
        if not vec:
            return "null_vector"
        rounded_vec = [round(float(x), 5) for x in vec]
        vec_str = json.dumps(rounded_vec)
        return hashlib.md5(vec_str.encode()).hexdigest()[:12]

    @staticmethod
    def query_hash(vector: list, filter_expr: dict, namespace: str = None) -> str:
        """Generate a hash for a query to detect duplicates"""
        vector_hash = VectorUtils.hash_vector(vector)
        filter_hash = hashlib.md5(
            json.dumps(filter_expr, sort_keys=True).encode()
        ).hexdigest()
        namespace_part = f":{namespace}" if namespace else ""
        return f"query:{vector_hash}:{filter_hash}{namespace_part}"


# ============================================================================
# PINECONE MANAGEMENT
# ============================================================================


class RerankedResult:
    """Lightweight result object for re-ranked query results"""

    def __init__(self, matches):
        self.matches = matches


class QueryDeduplicator:
    """Handles query deduplication to avoid redundant identical queries"""

    def __init__(self):
        self._active_queries = {}
        self._query_results_cache = {}

    async def deduplicated_query(self, query_func, query_hash: str):
        """Execute query with deduplication for identical in-flight queries"""

        # Check short-term memory cache
        if query_hash in self._query_results_cache:
            cached_result = self._query_results_cache[query_hash]
            if time.time() - cached_result["timestamp"] < 5.0:
                logger.info(f"Query cache hit for hash {query_hash}")
                return cached_result["result"]

        # Check if identical query is already in progress
        if query_hash in self._active_queries:
            logger.info(
                f"Duplicate query detected: {query_hash}, reusing in-flight request"
            )
            return await self._active_queries[query_hash]

        # Execute new query
        async def execute():
            try:
                result = await query_func()

                # Cache the result briefly
                self._query_results_cache[query_hash] = {
                    "result": result,
                    "timestamp": time.time(),
                }

                # Clean up old cache entries
                if len(self._query_results_cache) > 50:
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
                self._active_queries.pop(query_hash, None)

        query_task = asyncio.create_task(execute())
        self._active_queries[query_hash] = query_task
        return await query_task


class PineconeManager:
    """Manages all Pinecone operations with query deduplication"""

    def __init__(
        self,
        pc_client: Pinecone,
        index_name: str,
        namespace: str,
        limiter: AsyncLimiter,
        rq_queue: Queue,
    ):
        self.pc = pc_client
        self.idx = pc_client.Index(index_name)
        self.namespace = namespace
        self.limiter = limiter
        self.rq_queue = rq_queue
        self.deduplicator = QueryDeduplicator()

    async def query(
        self,
        vector: list,
        filter_expr: dict,
        top_k: int,
        namespace: str = None,
        enable_rerank: bool = False,
        reranking_model: str = "pinecone-rerank-v0",
        query_text: str = None,
        reranking_multiplier: int = 3,
    ) -> dict:
        """Query Pinecone with deduplication, rate limiting, and optional re-ranking"""
        if not vector:
            return {"matches": []}

        namespace = namespace if namespace is not None else self.namespace
        query_hash = VectorUtils.query_hash(vector, filter_expr, namespace)

        async def execute_query():
            async with self.limiter:
                logger.debug(f"Executing Pinecone query with hash {query_hash}")

                if enable_rerank and query_text:
                    # Two-stage retrieval with re-ranking
                    return await self._query_with_reranking(
                        vector,
                        filter_expr,
                        top_k,
                        namespace,
                        reranking_model,
                        query_text,
                        reranking_multiplier,
                    )
                else:
                    # Standard query without re-ranking
                    return await asyncio.to_thread(
                        lambda: self.idx.query(
                            vector=vector,
                            top_k=top_k,
                            namespace=namespace,
                            filter=filter_expr,
                            include_metadata=True,
                            include_values=False,
                        )
                    )

        return await self.deduplicator.deduplicated_query(execute_query, query_hash)

    async def _query_with_reranking(
        self,
        vector: list,
        filter_expr: dict,
        top_k: int,
        namespace: str,
        reranking_model: str,
        query_text: str,
        reranking_multiplier: int,
    ) -> dict:
        """Execute two-stage retrieval: initial query + Pinecone re-ranking"""
        # Input validation
        if not query_text or not query_text.strip():
            logger.warning(
                "Empty query text provided for re-ranking, falling back to standard query"
            )
            return await asyncio.to_thread(
                lambda: self.idx.query(
                    vector=vector,
                    top_k=top_k,
                    namespace=namespace,
                    filter=filter_expr,
                    include_metadata=True,
                    include_values=False,
                )
            )

        try:
            # Stage 1: Initial retrieval with more results for re-ranking
            initial_top_k = (
                top_k * reranking_multiplier
            )  # Retrieve multiplier x more for re-ranking
            logger.info(
                f"🔄 Pinecone re-ranking: retrieving {initial_top_k} initial results (multiplier: {reranking_multiplier})"
            )

            initial_result = await asyncio.to_thread(
                lambda: self.idx.query(
                    vector=vector,
                    top_k=initial_top_k,
                    namespace=namespace,
                    filter=filter_expr,
                    include_metadata=True,
                    include_values=False,
                )
            )

            matches = getattr(initial_result, "matches", []) or []
            if not matches:
                logger.info("No initial matches found for re-ranking")
                return initial_result

            logger.info(f"🔄 Re-ranking {len(matches)} results using {reranking_model}")

            # Stage 2: Re-ranking using Pinecone's hosted models
            reranked_matches = await self._rerank_with_pinecone(
                matches, query_text, reranking_model, top_k
            )

            # Create result object with re-ranked matches
            if reranked_matches:
                logger.info(f"✅ Re-ranking completed: {len(reranked_matches)} results")
                return RerankedResult(reranked_matches)
            else:
                logger.warning("Re-ranking returned no results, using original matches")
                return initial_result

        except Exception as e:
            logger.error(f"❌ Error in Pinecone re-ranking: {e}")
            logger.info("🔄 Falling back to standard query")
            # Fallback to standard query
            return await asyncio.to_thread(
                lambda: self.idx.query(
                    vector=vector,
                    top_k=top_k,
                    namespace=namespace,
                    filter=filter_expr,
                    include_metadata=True,
                    include_values=False,
                )
            )

    async def _rerank_with_pinecone(
        self, matches: list, query_text: str, model: str, top_k: int
    ) -> list:
        """Use Pinecone's hosted re-ranking models"""
        try:
            # Prepare documents for re-ranking (following Pinecone's format)
            documents = []
            match_map = {}  # Pre-build mapping for efficiency

            for match in matches:
                # Extract text content from metadata
                text_content = match.metadata.get("chunk_text") or match.metadata.get(
                    "text", ""
                )
                if text_content:
                    documents.append({"id": match.id, "text": text_content})
                    match_map[match.id] = match

            if not documents:
                logger.warning("No documents with text content found for re-ranking")
                return []

            logger.info(f"🔄 Re-ranking {len(documents)} documents using {model}")

            # Use Pinecone's inference.rerank method
            # Note: cohere-rerank-3.5 doesn't support 'truncate' parameter
            rerank_result = await asyncio.to_thread(
                lambda: self.pc.inference.rerank(
                    model=model,
                    query=query_text,
                    documents=documents,
                    top_n=top_k,
                    rank_fields=["text"],
                    return_documents=True,
                )
            )

            # Map re-ranked results back to original matches efficiently
            if hasattr(rerank_result, "data") and rerank_result.data:
                reranked_matches = []

                # Reorder matches based on re-ranking results
                for reranked_doc in rerank_result.data:
                    doc_id = reranked_doc["document"]["id"]
                    if doc_id in match_map:
                        original_match = match_map[doc_id]
                        # Add re-ranking score to match metadata
                        original_match.metadata["rerank_score"] = reranked_doc["score"]
                        reranked_matches.append(original_match)

                logger.info(
                    f"✅ Pinecone re-ranking completed: {len(reranked_matches)} results"
                )
                return reranked_matches
            else:
                logger.warning("Pinecone re-ranking returned no results")
                return []

        except Exception as e:
            logger.error(f"❌ Error in Pinecone re-ranking: {e}")
            return []

    async def schedule_upsert(self, upsert_data: list, namespace: str = None) -> str:
        """Schedule background upsert job via RQ"""
        namespace = namespace if namespace is not None else self.namespace

        # Ensure data is JSON serializable
        for item in upsert_data:
            item["values"] = [
                float(v) if v is not None else 0.0 for v in item["values"]
            ]

        job = await asyncio.to_thread(
            self.rq_queue.enqueue,
            "tasks.background_upsert",
            upsert_data,
            namespace,
            retry=Retry(max=3),
            ttl=3600,
        )

        logger.info(f"Scheduled upsert job {job.id} for {len(upsert_data)} vectors")
        return job.id

    def debug_queue_status(self) -> None:
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


class ChatHistoryRetriever:
    """Retrieves chat history from Pinecone with quality filtering"""

    def __init__(
        self,
        pinecone_manager: PineconeManager,
        min_quality_score: int,
        min_score_chat: float,
        top_k: int,
        enable_rerank: bool = False,
        reranking_model: str = "pinecone-rerank-v0",
        reranking_multiplier: int = 3,
    ):
        self.pinecone = pinecone_manager
        self.min_quality_score = min_quality_score
        self.min_score_chat = min_score_chat
        self.top_k = top_k
        self.enable_rerank = enable_rerank
        self.reranking_model = reranking_model
        self.reranking_multiplier = reranking_multiplier

    async def retrieve(
        self, vector: list, user_id: str, chat_id: str, query_text: str = None
    ) -> list:
        """Retrieve relevant chat history for the user and chat with optional re-ranking"""
        logger.info("=== STARTING PINECONE CHAT HISTORY RETRIEVAL ===")

        if not user_id or not chat_id:
            logger.info("Missing user_id or chat_id")
            return []

        # Build filter
        filter_expr = {
            "type": {"$eq": "chat"},
            "user_id": {"$eq": user_id},
            "chat_id": {"$eq": chat_id},
            "$or": [
                {"quality_score": {"$gte": self.min_quality_score}},
                {"quality_score": {"$exists": False}},
            ],
        }

        try:
            # Query Pinecone with optional re-ranking
            start_time = time.time()
            res = await self.pinecone.query(
                vector=vector,
                filter_expr=filter_expr,
                top_k=self.top_k,
                enable_rerank=self.enable_rerank,
                reranking_model=self.reranking_model,
                query_text=query_text,
                reranking_multiplier=self.reranking_multiplier,
            )
            query_time = time.time() - start_time
            matches = getattr(res, "matches", []) or []

            logger.info(
                f"Pinecone chat query returned {len(matches)} matches in {query_time:.3f}s"
            )

            # Debug: Show all chat matches and their scores first
            if matches:
                logger.info(
                    f"[DEBUG] All Pinecone chat matches with scores (min_score_chat={self.min_score_chat}):"
                )
                for i, match in enumerate(matches):
                    score = getattr(match, "score", 0)
                    metadata = getattr(match, "metadata", {})
                    summary_type = metadata.get("summary_type", "unknown")
                    timestamp = metadata.get("timestamp", "no-timestamp")
                    logger.info(
                        f"[DEBUG]   {i+1}. {summary_type} ({timestamp}) - Score: {score:.4f}"
                    )
            else:
                logger.info("[DEBUG] No chat matches returned from Pinecone")

            # Progressive refinement if needed
            if self._needs_refinement(matches):
                matches = await self._progressive_refinement(
                    vector, user_id, chat_id, matches, query_text
                )

            # Process results
            items = self._process_matches(matches)

            logger.info(
                f"[DEBUG] Retrieved {len(items)} chat history items from Pinecone"
            )
            if items:
                logger.info(
                    f"[DEBUG] Pinecone results preview: topic={items[0]['document_name']} (score: {items[0]['score']:.3f})"
                )
            logger.info("=== COMPLETED PINECONE CHAT HISTORY RETRIEVAL ===")
            return items

        except Exception as e:
            logger.error(f"Pinecone chat query failed: {e}")
            logger.info("=== FAILED PINECONE CHAT HISTORY RETRIEVAL ===")
            return []

    def _needs_refinement(self, matches: list) -> bool:
        """Check if we need progressive refinement"""
        quality_matches = [
            m for m in matches if getattr(m, "score", 0) >= self.min_score_chat
        ]
        return len(quality_matches) < 1

    async def _progressive_refinement(
        self,
        vector: list,
        user_id: str,
        chat_id: str,
        original_matches: list,
        query_text: str = None,
    ) -> list:
        """Try progressive refinement with relaxed filters"""
        logger.info("Trying progressive refinement for chat history")

        relaxed_filter = {
            "type": {"$eq": "chat"},
            "user_id": {"$eq": user_id},
            "chat_id": {"$eq": chat_id},
        }

        start_time = time.time()
        relaxed_res = await self.pinecone.query(
            vector=vector,
            filter_expr=relaxed_filter,
            top_k=self.top_k,
            enable_rerank=self.enable_rerank,
            reranking_model=self.reranking_model,
            query_text=query_text,
            reranking_multiplier=self.reranking_multiplier,
        )
        query_time = time.time() - start_time
        relaxed_matches = getattr(relaxed_res, "matches", []) or []

        logger.info(
            f"Progressive chat query returned {len(relaxed_matches)} matches in {query_time:.3f}s"
        )

        if len(relaxed_matches) > len(original_matches):
            logger.info(
                f"Using progressive query results: {len(relaxed_matches)} matches"
            )
            return relaxed_matches

        return original_matches

    def _process_matches(self, matches: list) -> list:
        """Process Pinecone matches into standard format"""
        items = []
        filtered_count = 0

        for m in matches:
            score = getattr(m, "score", 0.0)
            if score < self.min_score_chat:
                filtered_count += 1
                continue

            quality_score = m.metadata.get("quality_score", 0)
            if isinstance(quality_score, str):
                try:
                    quality_score = int(quality_score)
                except:
                    quality_score = 0

            topic = m.metadata.get("topic", "General")
            text = TextCleaner.clean_text(
                m.metadata.get("chunk_text") or m.metadata.get("text", "")
            )

            logger.info(
                f"[CHAT MATCH] topic={topic}, score={score:.4f}, quality={quality_score}"
            )

            items.append(
                {
                    "type": "chat",
                    "text": text,
                    "document_name": topic,
                    "score": score,
                    "quality_score": quality_score,
                }
            )

        if filtered_count > 0:
            logger.info(
                f"Filtered {filtered_count} chat matches with score < {self.min_score_chat}"
            )

        return items


# ============================================================================
# SUMMARIZATION SERVICE
# ============================================================================


class SummarizationService:
    """Handles chat summarization using OpenRouter"""

    def __init__(self, client: AsyncOpenAI, model: str, limiter: AsyncLimiter):
        self.client = client
        self.model = model
        self.limiter = limiter

    async def generate_summary(self, messages: list) -> str:
        """Generate summary from conversation messages"""
        try:
            resp = await self._call_api(messages, timeout=15.0)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Summary API failed: {e}. Using fallback.")
            return self._generate_fallback_summary(messages)

    async def generate_topic(self, summary: str) -> str:
        """Generate short topic phrase from summary (1-3 words for readability)"""
        try:
            topic_resp = await self._call_api(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert topic classifier. Identify the main topic "
                            "in 1-3 words maximum. Return just the topic phrase, no punctuation or explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Identify the main topic (1-3 words):\n\n{summary}",
                    },
                ],
                timeout=8.0,
            )
            topic = topic_resp.choices[0].message.content.strip()

            # Take first 3 words max if LLM returned more
            words = topic.split()[:3]
            topic = " ".join(words) if words else "General"

            # Remove problematic punctuation but KEEP spaces
            topic = re.sub(r"[^\w\s-]", "", topic)

            # Capitalize each word for better display
            topic = " ".join(word.capitalize() for word in topic.split())

            return topic if topic else "General"
        except Exception as e:
            logger.warning(f"Topic generation failed: {e}")
            return "General"

    async def _call_api(self, messages: list, timeout: float, retries: int = 2):
        """Make OpenRouter API call with retries"""
        retry_count = 0
        last_error = None

        while retry_count <= retries:
            try:
                async with self.limiter:
                    return await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model, messages=messages, temperature=0.3
                        ),
                        timeout=timeout,
                    )
            except asyncio.TimeoutError:
                last_error = f"API call timed out after {timeout}s"
                logger.warning(f"Timeout {retry_count+1}/{retries+1}: {last_error}")
            except Exception as e:
                last_error = f"API error: {str(e)}"
                logger.warning(f"Error {retry_count+1}/{retries+1}: {last_error}")

            retry_count += 1
            if retry_count <= retries:
                wait_time = 1.0 * (2 ** (retry_count - 1))
                await asyncio.sleep(wait_time)

        raise Exception(last_error)

    def _generate_fallback_summary(self, messages: list) -> str:
        """Generate basic summary when API fails"""
        try:
            user_content = TextCleaner.ensure_string(
                next(
                    (m.get("content", "") for m in messages if m.get("role") == "user"),
                    "",
                )
            )
            assistant_content = TextCleaner.ensure_string(
                next(
                    (
                        m.get("content", "")
                        for m in messages
                        if m.get("role") == "assistant"
                    ),
                    "",
                )
            )

            user_content = TextCleaner.clean_text(user_content)
            assistant_content = TextCleaner.clean_text(assistant_content)

            user_summary = user_content[:150] + (
                "..." if len(user_content) > 150 else ""
            )

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

            return f"The user asked about {user_summary}. The assistant provided information about {assistant_summary}."
        except Exception as e:
            logger.error(f"Error generating fallback summary: {e}")
            return "The user and assistant had a conversation."

    async def generate_and_store(
        self,
        chat_id: str,
        user_id: str,
        last_exchange: list,
        embedding_service,
        pinecone_manager: PineconeManager,
        document_processor,
    ) -> None:
        """Generate summary and store in Pinecone (background task)"""
        try:
            # Clean messages
            cleaned_messages = []
            for msg in last_exchange:
                content = TextCleaner.ensure_string(msg.get("content", ""))
                content = TextCleaner.clean_text(content)
                cleaned_messages.append(
                    {"role": msg.get("role", ""), "content": content}
                )

            # Build prompt
            conversation_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in cleaned_messages
            )

            messages_for_api = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert summarization assistant who creates comprehensive, informative summaries. "
                        "When summarizing the conversation exchange between the user and assistant:\n\n"
                        "1. Begin with a clear statement of the user's primary question or request\n"
                        "2. Provide a thorough account of the assistant's response\n"
                        "3. Include specific numerical data and details when present\n"
                        "4. Structure the summary in 2-3 well-developed paragraphs\n"
                        "5. Write in a neutral, informative tone\n\n"
                        "Your summaries should be approximately 100-150 words."
                    ),
                },
                {"role": "user", "content": conversation_text},
            ]

            # Generate summary
            logger.info(f"Background: Requesting summary for chat {chat_id}")
            summary = await self.generate_summary(messages_for_api)
            using_fallback = "The user asked about" in summary  # Simple heuristic

            summary = TextCleaner.clean_text(summary)
            # Additional cleaning for summary text to remove any remaining escape sequences
            summary = TextCleaner._clean_summary_text(summary)
            logger.info(
                f"Background: Summary generated: {summary[:100]}... {'(FALLBACK)' if using_fallback else ''}"
            )

            # Generate topic
            if using_fallback:
                # Simple topic extraction when already in fallback mode
                user_content = next(
                    (
                        m.get("content", "")
                        for m in cleaned_messages
                        if m.get("role") == "user"
                    ),
                    "",
                )
                words = user_content.strip().split()[:3] if user_content.strip() else []
                topic = " ".join(words).capitalize() if words else "General"
                topic = re.sub(r"[^\w\s-]", "", topic)  # Clean but keep spaces
                logger.info(f"Using fallback topic extraction: {topic}")
            else:
                topic = await self.generate_topic(summary)

            # Chunk if needed
            summary_chunks = await self._chunk_summary(summary, document_processor)

            # Generate embeddings for chunks
            chunk_embeddings = await embedding_service.embed_batch(summary_chunks)

            # Build upsert data
            upsert_data = self._build_upsert_data(
                summary_chunks,
                chunk_embeddings,
                chat_id,
                user_id,
                topic,
                using_fallback,
            )

            # Schedule background upsert
            await pinecone_manager.schedule_upsert(upsert_data)

            logger.info(f"Background: Summary generation complete for chat {chat_id}")

        except Exception as e:
            logger.error(f"Background summary generation failed: {e}", exc_info=True)

    async def _chunk_summary(self, summary: str, document_processor) -> list:
        """Chunk summary if it's long"""
        if len(summary) <= 2000:
            return [summary]

        logger.info(f"Summary is long ({len(summary)} chars), chunking")
        try:
            chunks = document_processor.content_aware_splitter.split_text(summary)
            logger.info(f"Created {len(chunks)} chunks from summary")
            return chunks
        except Exception as e:
            logger.error(f"Chunking error: {e}, using full summary")
            return [summary]

    def _build_upsert_data(
        self,
        chunks: list,
        embeddings: list,
        chat_id: str,
        user_id: str,
        topic: str,
        using_fallback: bool,
    ) -> list:
        """Build upsert data for Pinecone"""
        upsert_data = []

        for i, (chunk_text, chunk_vec) in enumerate(zip(chunks, embeddings)):
            # Calculate quality score
            quality_score = self._quick_quality_score(chunk_text)

            rec_id = f"{chat_id}-{int(time.time())}-{i}"

            metadata = {
                "type": "chat",
                "chat_id": chat_id,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "chunk_text": TextCleaner.clean_for_pinecone(chunk_text),
                "summary_id": rec_id,
                "summary_length": len(chunk_text),
                "vector_dim": len(chunk_vec),
                "model": self.model,
                "topic": topic,
                "fallback": using_fallback,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "quality_score": quality_score,
                "doc_type": "chat_summary",
            }

            upsert_data.append(
                {
                    "id": rec_id,
                    "values": [float(v) if v is not None else 0.0 for v in chunk_vec],
                    "metadata": metadata,
                }
            )

        return upsert_data

    @staticmethod
    def _quick_quality_score(text: str) -> int:
        """Simple 0-5 quality score for text chunks"""
        score = 0
        if len(text) > 200:
            score += 1
        if text.count(".") >= 2:
            score += 1
        if "\n\n" in text:
            score += 1
        if not (text.count("\n- ") > 5 or text.count("\n• ") > 5):
            score += 1
        if re.search(
            r"\b(because|therefore|however|additionally)\b", text, re.IGNORECASE
        ):
            score += 1
        return score


# ============================================================================
# MESSAGE BUILDER
# ============================================================================


class MessageBuilder:
    """Builds message context from retrieved items"""

    @staticmethod
    def build_context(
        messages: list,
        chat_items: list,
        knowledge_items: list,
        keep_recent: int,
        keep_without_summary: int,
        display_k: int,
    ) -> list:
        """Build context messages from retrieved items"""

        # Find existing summary
        summary_msg = next(
            (
                m
                for m in messages
                if m.get("role") == "system"
                and MessageBuilder._content_startswith(
                    m.get("content", ""), "Chat summary:"
                )
            ),
            None,
        )

        # Use best chat summary if available
        if chat_items and not summary_msg:
            best_summary = max(chat_items, key=lambda x: x["score"])
            summary_msg = {
                "role": "system",
                "content": f"Chat summary: {best_summary['text']}",
            }
            logger.info(
                f"Using retrieved chat summary with score={best_summary['score']:.4f}"
            )

        # Build RAG blocks
        rag_blocks = []
        if knowledge_items:
            rag_blocks = [
                f"{i['document_name']}: {i['text']}"
                for i in knowledge_items[:display_k]
            ]

        # Get user/assistant messages
        user_assistant_msgs = [
            m for m in messages if m.get("role") in ("user", "assistant")
        ]

        # Build new message list
        if summary_msg:
            new_msg_list = [summary_msg]
            if rag_blocks:
                new_msg_list.append(
                    {
                        "role": "system",
                        "content": "Relevant knowledge:\n" + "\n".join(rag_blocks),
                    }
                )
            new_msg_list.extend(user_assistant_msgs[-keep_recent * 2 :])
            logger.info(f"Using summary + {len(new_msg_list)-1} recent messages")
        else:
            new_msg_list = []
            if rag_blocks:
                new_msg_list.append(
                    {
                        "role": "system",
                        "content": "Relevant knowledge:\n" + "\n".join(rag_blocks),
                    }
                )

            if len(user_assistant_msgs) > keep_without_summary * 2:
                logger.info(
                    f"No summary, limiting to {keep_without_summary} recent turns"
                )
                new_msg_list.extend(user_assistant_msgs[-keep_without_summary * 2 :])
            else:
                logger.info(
                    f"No summary, using all {len(user_assistant_msgs)//2} turns"
                )
                new_msg_list.extend(user_assistant_msgs)

        return new_msg_list

    @staticmethod
    def _content_startswith(content, prefix: str) -> bool:
        """Safely check if content starts with prefix"""
        content_str = TextCleaner.ensure_string(content)
        return content_str.startswith(prefix)


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================


class ContentAwareTextSplitter:
    """Content-aware text splitter optimized for chat and conversation content"""

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        min_chunk_size: int = 100,
        max_chunk_size: int = 3000,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        self.conversation_patterns = [
            (r"^(User|Assistant|Human|AI|System):\s*", "speaker_change"),
            (r"\n\n(?=\w)", "conversation_break"),
            (r"\?\s*\n", "question_end"),
            (
                r"\b(Now|Next|Moving on|In conclusion|To summarize)\b",
                "topic_transition",
            ),
            (r"\n\s*\n", "paragraph_break"),
        ]

    def split_text(self, text: str) -> list:
        """Split text into content-aware chunks"""
        if not text or len(text) <= self.min_chunk_size:
            return [text] if text else []

        try:
            cleaned_text = TextCleaner.clean_text(text)
            elements = self._detect_structure(cleaned_text)
            split_points = self._find_split_points(cleaned_text, elements)
            split_points = self._handle_overlap(cleaned_text, split_points)

            chunks = []
            for i in range(len(split_points) - 1):
                chunk_text = cleaned_text[split_points[i] : split_points[i + 1]].strip()

                if chunk_text and len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                elif chunks and chunk_text:
                    chunks[-1] += "\n\n" + chunk_text

            return chunks if chunks else self._fallback_split(cleaned_text)

        except Exception as e:
            logger.warning(f"Content-aware splitting failed: {e}. Using fallback.")
            return self._fallback_split(text)

    def _detect_structure(self, text: str) -> list:
        """Detect conversation structure elements"""
        elements = []
        lines = text.split("\n")
        current_pos = 0

        for line in lines:
            line_start = current_pos
            line_end = current_pos + len(line)

            for pattern, element_type in self.conversation_patterns:
                if re.search(pattern, line.strip(), re.IGNORECASE):
                    elements.append((line_start, line_end, element_type, line.strip()))
                    break

            current_pos = line_end + 1

        return elements

    def _find_split_points(self, text: str, elements: list) -> list:
        """Find optimal split points based on structure"""
        split_points = [0]
        elements.sort(key=lambda x: x[0])
        current_chunk_start = 0

        for start_pos, end_pos, element_type, content in elements:
            distance = start_pos - current_chunk_start

            if distance >= self.chunk_size - self.chunk_overlap and element_type in [
                "conversation_break",
                "speaker_change",
                "paragraph_break",
            ]:
                if distance >= self.min_chunk_size:
                    split_points.append(start_pos)
                    current_chunk_start = start_pos
            elif distance >= self.max_chunk_size:
                split_points.append(start_pos)
                current_chunk_start = start_pos

        if split_points[-1] != len(text):
            split_points.append(len(text))

        return split_points

    def _handle_overlap(self, text: str, split_points: list) -> list:
        """Adjust split points for meaningful overlap"""
        if len(split_points) <= 2:
            return split_points

        adjusted = [split_points[0]]

        for i in range(1, len(split_points) - 1):
            current_start = split_points[i]
            overlap_start = max(adjusted[-1], current_start - self.chunk_overlap)

            overlap_text = text[overlap_start:current_start]
            sentence_end = overlap_text.rfind(". ")
            if sentence_end > 0:
                overlap_start = overlap_start + sentence_end + 2

            adjusted.append(overlap_start)

        adjusted.append(split_points[-1])
        return adjusted

    def _fallback_split(self, text: str) -> list:
        """Fallback to recursive character splitting"""
        return recursive_splitter.split_text(text)


class DocumentProcessor:
    """Document processor with content-aware chunking for chat content"""

    def __init__(self):
        self.chunk_size = 1500
        self.chunk_overlap = 150
        self.content_aware_splitter = ContentAwareTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    async def process_text(self, text: str, metadata: dict = None) -> Tuple[list, dict]:
        """Process text into content-aware chunks with metadata"""
        if not text or not text.strip():
            logger.warning("Empty text provided to processor")
            return [], metadata or {}

        cleaned_text = TextCleaner.clean_text(text)
        chunks = self.content_aware_splitter.split_text(cleaned_text)
        doc_type = self._detect_document_type(chunks)

        enhanced_metadata = metadata.copy() if metadata else {}
        enhanced_metadata.update(
            {
                "doc_type": doc_type,
                "chunk_count": len(chunks),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "processing_method": "content_aware_chat_splitter",
            }
        )

        return chunks, enhanced_metadata

    @staticmethod
    def quick_quality_score(text: str) -> int:
        """Simple 0-5 quality score for text chunks"""
        score = 0
        if len(text) > 200:
            score += 1
        if text.count(".") >= 2:
            score += 1
        if "\n\n" in text:
            score += 1
        if not (text.count("\n- ") > 5 or text.count("\n• ") > 5):
            score += 1
        if re.search(
            r"\b(because|therefore|however|additionally)\b", text, re.IGNORECASE
        ):
            score += 1
        return score

    def _detect_document_type(self, chunks: list) -> str:
        """Detect document type based on content patterns"""
        if not chunks:
            return "unknown"

        sample_text = chunks[0][:4000] + (chunks[1][:2000] if len(chunks) > 1 else "")

        patterns = {
            "chat": r"\b(user|assistant|human|ai|system|message|conversation|chat|dialogue|response)\b",
            "academic": r"\b(abstract|introduction|methodology|conclusion|references|literature review)\b",
            "meeting": r"\b(memorandum|meeting minutes|agenda|attendees|participants|convened)\b",
            "report": r"\b(quarterly|annual|financial|report|overview|executive summary)\b",
        }

        for doc_type, pattern in patterns.items():
            if re.search(pattern, sample_text, re.IGNORECASE):
                return doc_type

        return "general"


# ============================================================================
# MAIN FILTER (ORCHESTRATOR)
# ============================================================================


class Filter:
    """
    Main orchestrator for chat summarization and knowledge retrieval.
    Coordinates specialized services but delegates actual work to them.
    """

    # Shared HTTP client for performance
    http_client = httpx.AsyncClient(
        timeout=10.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        http2=True,
    )

    class Valves(BaseModel):
        """Configuration for the filter"""

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
        rag_reranking_engine: str = Field(
            default_factory=lambda: os.getenv("RAG_RERANKING_ENGINE", "")
        )
        rag_reranking_model: str = Field(
            default_factory=lambda: os.getenv(
                "RAG_RERANKING_MODEL", "pinecone-rerank-v0"
            )
        )
        rag_enable_rerank: bool = Field(
            default_factory=lambda: os.getenv("RAG_ENABLE_RERANK", "true").lower()
            == "true"
        )
        rag_reranking_multiplier: int = Field(
            default_factory=lambda: max(
                2, min(5, int(os.getenv("RAG_RERANKING_MULTIPLIER", "3")))
            )
        )

    def __init__(self):
        """Initialize orchestrator and all services"""
        self.valves = self.Valves()
        logger.info("Loaded configuration: %s", self.valves.json())

        if not self.valves.pinecone_api_key or not self.valves.pinecone_index_name:
            raise ValueError("Missing Pinecone key or index name")

        # Initialize Pinecone
        pc = Pinecone(api_key=self.valves.pinecone_api_key)

        try:
            if self.valves.pinecone_index_name not in pc.list_indexes().names():
                logger.warning(
                    f"Index {self.valves.pinecone_index_name} not found. Check your configuration."
                )

            # Get the index
            idx = pc.Index(self.valves.pinecone_index_name)
            logger.info(
                f"Pinecone initialized: index={self.valves.pinecone_index_name}, namespace={self.valves.namespace}"
            )
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

        # Setup Redis
        redis_url = os.getenv("REDIS_URL")
        pool_size = int(os.getenv("REDIS_POOL_SIZE", ""))
        timeout = int(os.getenv("REDIS_TIMEOUT", ""))

        redis_conn = Redis.from_url(
            url=redis_url, max_connections=pool_size, socket_timeout=timeout
        )
        rq_queue = Queue(name="pinecone-upserts", connection=redis_conn)

        logger.info(
            f"Redis connected at {redis_url} (pool_size={pool_size}, timeout={timeout}s)"
        )

        # Initialize cache manager
        embed_ttl = int(os.getenv("REDIS_EMBEDDING_CACHE_TTL", ""))
        search_ttl = int(os.getenv("REDIS_SEARCH_CACHE_TTL", ""))
        knowledge_ttl = int(os.getenv("REDIS_KNOWLEDGE_CACHE_TTL", str(search_ttl)))

        enable_knowledge_cache = (
            os.getenv("REDIS_ENABLE_KNOWLEDGE_CACHE", "true").lower() == "true"
        )
        logger.info(
            f"Knowledge caching {'enabled' if enable_knowledge_cache else 'disabled'} with TTL: {knowledge_ttl}s"
        )

        cache_manager = CacheManager(redis_conn, embed_ttl, search_ttl, knowledge_ttl)

        # Initialize OpenAI client for embeddings
        if self.valves.openai_embedding_base_url:
            embedding_client = AsyncOpenAI(
                api_key=self.valves.openai_api_key,
                base_url=self.valves.openai_embedding_base_url,
                http_client=self.http_client,
            )
            logger.info(
                f"Initialized OpenAI embedding client with base_url={self.valves.openai_embedding_base_url}"
            )
        else:
            embedding_client = AsyncOpenAI(
                api_key=self.valves.openai_api_key, http_client=self.http_client
            )
            logger.info("Initialized OpenAI embedding client with default base_url.")

        # Initialize OpenRouter client for chat
        chat_client = AsyncOpenAI(
            api_key=self.valves.openrouter_api_key,
            base_url=self.valves.openrouter_api_base_url,
            http_client=self.http_client,
        )
        logger.info("OpenRouter client initialized for chat summarization.")

        # Validate and log Pinecone re-ranking configuration
        self._validate_reranking_config()

        # Create limiters
        embedding_limiter = AsyncLimiter(max_concurrent=20, max_per_second=40)
        openrouter_limiter = AsyncLimiter(max_concurrent=5, max_per_second=3)
        pinecone_limiter = AsyncLimiter(max_concurrent=10, max_per_second=10)

        # Initialize services
        self.embeddings = EmbeddingService(
            client=embedding_client,
            cache_manager=cache_manager,
            limiter=embedding_limiter,
            model=self.valves.openai_embedding_model,
            vector_dim=self.valves.vector_dim,
        )

        self.pinecone = PineconeManager(
            pc_client=pc,
            index_name=self.valves.pinecone_index_name,
            namespace=self.valves.namespace,
            limiter=pinecone_limiter,
            rq_queue=rq_queue,
        )

        self.chat_retriever = ChatHistoryRetriever(
            pinecone_manager=self.pinecone,
            min_quality_score=self.valves.min_quality_score,
            min_score_chat=self.valves.min_score_chat,
            top_k=self.valves.top_k,
            enable_rerank=self.valves.rag_enable_rerank
            and self.valves.rag_reranking_engine == "pinecone",
            reranking_model=self.valves.rag_reranking_model,
            reranking_multiplier=self.valves.rag_reranking_multiplier,
        )

        self.summarizer = SummarizationService(
            client=chat_client,
            model=self.valves.openrouter_model,
            limiter=openrouter_limiter,
        )

        self.message_builder = MessageBuilder()
        self.document_processor = DocumentProcessor()

        # Log quality score threshold
        logger.info(
            f"Minimum quality score threshold set to: {self.valves.min_quality_score}"
        )

        logger.info("Filter initialized with all services")

    def _validate_reranking_config(self) -> None:
        """Validate Pinecone re-ranking configuration and log status"""
        # Valid Pinecone re-ranking models
        valid_models = {"pinecone-rerank-v0", "cohere-rerank-3.5", "bge-reranker-v2-m3"}

        reranking_enabled = (
            self.valves.rag_enable_rerank
            and self.valves.rag_reranking_engine == "pinecone"
        )

        if reranking_enabled:
            if self.valves.rag_reranking_model not in valid_models:
                logger.warning(
                    f"Invalid re-ranking model '{self.valves.rag_reranking_model}'. "
                    f"Valid models: {', '.join(valid_models)}. Disabling re-ranking."
                )
                reranking_enabled = False
            else:
                logger.info("✅ Pinecone re-ranking enabled for chat history retrieval")
                logger.info(
                    f"Pinecone re-ranking configuration: engine={self.valves.rag_reranking_engine}, "
                    f"model={self.valves.rag_reranking_model}, enabled={self.valves.rag_enable_rerank}"
                )

        if not reranking_enabled:
            logger.info(
                "Pinecone re-ranking disabled - using standard vector similarity search"
            )

    async def inlet(
        self, body: dict, *, __user__: dict = None, __event_emitter__=None, **_
    ) -> dict:
        """
        Orchestrate retrieval: coordinate services to inject relevant context.
        This method delegates work to specialized services.
        """
        __user__ = __user__ or {}
        messages = body.get("messages", []) or []
        chat_id = self._get_chat_id(body)
        user_id = __user__.get("id")

        _debug_message_log("INLET-BEFORE", messages)
        logger.info(f"Processing inlet for chat_id={chat_id}, user_id={user_id}")

        # Extract user query
        user_texts = [
            TextCleaner.ensure_string(m["content"])
            for m in messages
            if m.get("role") == "user"
        ]

        if not user_texts:
            logger.info("No user message present")
            _debug_message_log("INLET-AFTER (no change)", messages)
            return body

        # Clean and prepare query
        query = TextCleaner.clean_text(user_texts[-1])

        logger.info(f"[DEBUG] User query for retrieval: {query[:100]}...")

        # DELEGATE: Generate embedding
        search_start = time.time()
        vector = await self.embeddings.embed_single(query)

        # Validate vector
        if not vector or len(vector) != self.valves.vector_dim:
            logger.warning(
                f"Invalid vector (len={len(vector) if vector else 0}), skipping retrieval"
            )
            _debug_message_log("INLET-AFTER (invalid vector)", messages)
            return body

        # DELEGATE: Retrieve from Pinecone chat history
        logger.info(
            f"Retrieval started for chat_id={chat_id}, user_id={user_id}, quality_threshold={self.valves.min_quality_score}"
        )

        chat_items = await self.chat_retriever.retrieve(vector, user_id, chat_id, query)
        knowledge_items = []  # No knowledge retrieval in Pinecone-only version

        # Log performance metrics
        search_time = time.time() - search_start

        logger.info(
            f"Pinecone retrieval results: {len(chat_items)} chat items, "
            f"time={search_time:.3f}s (quality filter: min_quality_score={self.valves.min_quality_score})"
        )

        # Emit status
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"✔ Found {len(chat_items)} relevant chat history items",
                        "done": True,
                    },
                }
            )

        # DELEGATE: Build context messages
        new_messages = self.message_builder.build_context(
            messages,
            chat_items,
            knowledge_items,
            self.valves.keep_recent,
            self.valves.keep_without_summary,
            self.valves.display_k,
        )

        # Check if we're adding a new summary
        summary_msg = next(
            (
                m
                for m in messages
                if m.get("role") == "system"
                and MessageBuilder._content_startswith(
                    m.get("content", ""), "Chat summary:"
                )
            ),
            None,
        )
        if (
            chat_items
            and summary_msg is None
            and not any(
                MessageBuilder._content_startswith(
                    m.get("content", ""), "Chat summary:"
                )
                for m in messages
            )
        ):
            logger.info(
                "Adding a new chat summary from Pinecone (none found in incoming messages)"
            )

        _debug_message_log("INLET-AFTER", new_messages)
        body["messages"] = new_messages
        logger.info(f"Inlet processing completed for chat_id={chat_id}")
        return body

    async def outlet(self, body: dict, *, __user__: dict = None, **_) -> dict:
        """
        Orchestrate summarization: coordinate services to generate and store summaries.
        This method delegates work to specialized services.
        """
        __user__ = __user__ or {}
        messages = body.get("messages", []) or []
        chat_id = self._get_chat_id(body)
        user_id = __user__.get("id")

        _debug_message_log("OUTLET-BEFORE", messages)
        logger.info(f"Processing outlet for chat_id={chat_id}, user_id={user_id}")

        if not user_id:
            logger.warning("Missing user_id, summary will not be stored")
            return body

        chat_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]

        if len(chat_msgs) < self.valves.turns_before * 2:
            logger.info("Not enough chat rounds for summarization yet")
            _debug_message_log("OUTLET-AFTER (no change)", messages)
            return body

        # DELEGATE: Schedule background summarization
        last_exchange = chat_msgs[-2:]
        logger.info(f"Scheduling background summary generation for chat_id={chat_id}")

        asyncio.create_task(
            self.summarizer.generate_and_store(
                chat_id,
                user_id,
                last_exchange,
                self.embeddings,
                self.pinecone,
                self.document_processor,
            )
        )

        _debug_message_log("OUTLET-AFTER (summary scheduled in background)", messages)
        logger.info(f"Outlet processing completed (summary scheduled in background)")
        return body

    async def process_document(
        self, text: str, document_id: str = None, metadata: dict = None
    ) -> dict:
        """
        Process a document with content-aware chunking and prepare for vectorization.
        Public API method for external document processing.

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

        # Clean text
        cleaned_text = TextCleaner.clean_text(text)
        start_time = time.time()
        logger.info(f"Processing document {document_id} ({len(cleaned_text)} chars)")

        # Process into chunks
        chunks, enhanced_metadata = await self.document_processor.process_text(
            cleaned_text, metadata
        )

        # Clean each chunk
        cleaned_chunks = [TextCleaner.clean_text(chunk) for chunk in chunks]

        # Add document_id to metadata
        enhanced_metadata["document_id"] = document_id

        # Generate embeddings
        vectors = await self.embeddings.embed_batch(cleaned_chunks)

        # Create records
        records = []
        for i, (chunk, vector) in enumerate(zip(cleaned_chunks, vectors)):
            quality_score = self.document_processor.quick_quality_score(chunk)
            relative_position = i / len(chunks)
            is_intro = relative_position < 0.2
            is_conclusion = relative_position > 0.8

            chunk_metadata = {
                **enhanced_metadata,
                "chunk_id": i,
                "chunk_text": TextCleaner.clean_for_pinecone(chunk),
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
                    "values": [float(v) if v is not None else 0.0 for v in vector],
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

    def debug_queue_status(self) -> None:
        """Debug the status of the Redis queue"""
        return self.pinecone.debug_queue_status()

    def _get_chat_id(self, body: dict) -> str:
        """Extract chat ID from request body"""
        meta = body.get("metadata", {})
        return (
            body.get("chat_id")
            or body.get("id")
            or body.get("session_id")
            or meta.get("chat_id")
            or meta.get("session_id")
            or "default"
        )

    async def __aenter__(self):
        """Async context manager entry"""
        logger.info("Entered Filter async context manager")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit - cleanup resources"""
        logger.info("Exiting Filter context manager")
        await self.http_client.aclose()
