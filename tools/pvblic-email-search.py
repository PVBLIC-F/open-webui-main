"""
title: Gmail Semantic Search
author: Open WebUI
author_url: https://github.com/open-webui
version: 3.1.0
license: MIT
description: High-accuracy Gmail semantic search with per-user data isolation. Features: Cohere reranking (default), query-aware boosting (sender/recipient/attachment detection), optimized scoring weights (80% semantic), stop word filtering, and V4 metadata schema support. Uses Pinecone namespaces for secure per-user isolation.
required_open_webui_version: 0.3.0
requirements: openai
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Tuple, Callable
import logging
import json
import time
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Constants
MAX_QUERY_LENGTH = 500
MIN_QUERY_LENGTH = 3
SEARCH_TIMEOUT_SECONDS = 30
EMBEDDING_DIMENSION = 1536
CACHE_SIZE_LIMIT = 50
CACHE_CLEANUP_SIZE = 12
CACHE_EXPIRY_HOURS = 1
MIN_SCORE_THRESHOLD = 0.05
RERANKING_MULTIPLIER = 5  # Retrieve 5x more results for reranking (better recall)
FALLBACK_MULTIPLIER = 3   # Retrieve 3x more results for fallback
DEFAULT_YEAR_DAYS = 365   # Default fallback for year calculations

# Stop words to remove from queries for better embeddings
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'it', 'its', 'they', 'their',
    'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
    'all', 'any', 'each', 'some', 'no', 'not', 'only', 'same', 'so', 'than',
    'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
    'email', 'emails', 'mail', 'mails', 'message', 'messages',  # Email-specific
}

# Attachment-related keywords for boosting
ATTACHMENT_KEYWORDS = {
    'attachment', 'attachments', 'attached', 'document', 'documents',
    'file', 'files', 'pdf', 'spreadsheet', 'excel', 'word', 'doc',
    'image', 'photo', 'picture', 'screenshot', 'presentation', 'ppt',
    'download', 'downloaded', 'zip', 'csv', 'report'
}


class Tools:
    class Valves(BaseModel):
        top_k: int = Field(
            default=25, description="Maximum number of email results to return (1-50)"
        )
        min_score: float = Field(
            default=MIN_SCORE_THRESHOLD,
            description="Minimum relevance score threshold (0.0-1.0). Higher = more strict matching. Lowered to 0.05 to catch more email matches.",
        )
        show_summary_only: bool = Field(
            default=False,
            description="(Deprecated in V4) V4 uses single vector per email. This setting has no effect.",
        )
        use_pinecone_reranking: bool = Field(
            default=True,
            description="Use Pinecone's hosted reranking models for better relevance (recommended)",
        )
        reranking_model: str = Field(
            default="cohere-rerank-3.5",
            description="Pinecone reranking model: 'cohere-rerank-3.5' (recommended), 'pinecone-rerank-v0', or 'bge-reranker-v2-m3'",
        )
        include_sent: bool = Field(
            default=True, description="Include emails you sent (not just received)"
        )
        format_results: str = Field(
            default="detailed", description="Result format: 'detailed' or 'compact'"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True  # Enable citation support
        self._query_cache = {}  # Simple in-memory cache {query_hash: (embedding, timestamp)}
    
    def _get_validated_user_id(self, __user__: dict) -> str:
        """Extract and validate user ID from request."""
        user_id = __user__.get("id")
        if not user_id:
            logger.error("❌ User not authenticated")
            return "anonymous"
        
        # Validate UUID format
        if user_id != "anonymous" and not re.match(r'^[a-f0-9-]{36}$', user_id):
            logger.error(f"❌ Invalid user_id format: {user_id}")
            return "anonymous"
        
        return user_id
    
    def _prepare_query(self, query: str) -> tuple:
        """
        Validate query and extract time filter, sender/recipient hints, and attachment hints.
        Returns (cleaned_query, time_cutoff, query_hints).
        """
        # Validate query
        validation_error = self._validate_query(query)
        if validation_error:
            logger.error(f"❌ Query validation failed: {validation_error}")
            query = "search emails"  # Fallback
        
        # Extract time filter
        time_cutoff, time_filter_info = self._extract_time_filter(query)
        
        # Convert year to date range if needed
        if time_cutoff and isinstance(time_cutoff, int) and 1900 <= time_cutoff <= 2100:
            year = time_cutoff
            current_year = datetime.now().year
            
            if 1900 <= year <= current_year + 1:
                start_year = int(datetime(year, 1, 1).timestamp())
                end_year = int(datetime(year, 12, 31, 23, 59, 59).timestamp())
                time_cutoff = (start_year, end_year)
                logger.info(f"📅 Year filter converted to range: {year}")
        
        # Extract query hints (sender, recipient, attachment expectations)
        query_hints = self._extract_query_hints(query)
        
        # Validate query length
        if not query or len(query.strip()) < MIN_QUERY_LENGTH:
            logger.warning(f"Query too short: '{query}'")
            query = self._get_fallback_query(query)
        
        return query, time_cutoff, query_hints
    
    def _extract_query_hints(self, query: str) -> dict:
        """
        Extract semantic hints from query for boosting:
        - Sender hints: "from john", "sent by sarah"
        - Recipient hints: "to mike", "sent to team"
        - Attachment hints: "pdf", "document", "attachment"
        """
        query_lower = query.lower()
        hints = {
            'sender_hints': [],
            'recipient_hints': [],
            'expects_attachment': False,
            'clean_query': query,
        }
        
        # Extract sender hints: "from X", "by X", "sent by X"
        sender_patterns = [
            r'(?:from|by|sent by)\s+([a-zA-Z][a-zA-Z0-9._-]*(?:\s+[a-zA-Z][a-zA-Z0-9._-]*)?)',
        ]
        for pattern in sender_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                name = match.strip()
                if name and len(name) > 1 and name not in STOP_WORDS:
                    hints['sender_hints'].append(name)
                    logger.info(f"🔍 Sender hint extracted: '{name}'")
        
        # Extract recipient hints: "to X", "sent to X"
        recipient_patterns = [
            r'(?:to|sent to)\s+([a-zA-Z][a-zA-Z0-9._-]*(?:\s+[a-zA-Z][a-zA-Z0-9._-]*)?)',
        ]
        for pattern in recipient_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                name = match.strip()
                if name and len(name) > 1 and name not in STOP_WORDS:
                    hints['recipient_hints'].append(name)
                    logger.info(f"🔍 Recipient hint extracted: '{name}'")
        
        # Check for attachment-related keywords
        query_words = set(query_lower.split())
        attachment_matches = query_words & ATTACHMENT_KEYWORDS
        if attachment_matches:
            hints['expects_attachment'] = True
            logger.info(f"📎 Attachment hint detected: {attachment_matches}")
        
        # Create cleaned query (remove stop words for better embeddings)
        words = query.split()
        cleaned_words = [w for w in words if w.lower() not in STOP_WORDS or len(w) > 3]
        hints['clean_query'] = ' '.join(cleaned_words) if cleaned_words else query
        
        if hints['clean_query'] != query:
            logger.debug(f"🧹 Query cleaned: '{query}' -> '{hints['clean_query']}'")
        
        return hints
    
    def _get_fallback_query(self, original_query: str) -> str:
        """Generate smart fallback query based on intent."""
        original_lower = original_query.lower()
        
        if any(word in original_lower for word in ['summarize', 'summary', 'overview']):
            return "summarize emails"
        elif any(word in original_lower for word in ['find', 'search', 'look']):
            return "search emails"
        elif any(word in original_lower for word in ['show', 'list', 'display']):
            return "show emails"
        else:
            return "search emails"
    
    async def _get_embedding_function(self):
        """Get the embedding function from Open WebUI app state."""
        from open_webui.main import app as webui_app
        
        embedding_function = webui_app.state.EMBEDDING_FUNCTION
        if isinstance(embedding_function, dict):
            embedding_function = embedding_function.get('function') or embedding_function.get('embedding_function')
        
        return embedding_function
    
    async def _perform_search(self, query_embedding: list, user_id: str, user_namespace: str):
        """Execute the vector search and return results."""
        from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
        
        collection_name = "gmail"
        
        logger.info(f"🔍 Searching Gmail for user {user_id[:8]}...")
        logger.info(f"   Collection: '{collection_name}'")
        logger.info(f"   Namespace: '{user_namespace}' (per-user isolation)")
        logger.info(f"   Top-K: {self.valves.top_k}")
        
        # Get vector client
        actual_client = VECTOR_DB_CLIENT
        if isinstance(VECTOR_DB_CLIENT, dict):
            actual_client = VECTOR_DB_CLIENT.get('client') or VECTOR_DB_CLIENT.get('vector_db_client')
        
        if not actual_client:
            logger.error("❌ No vector DB client available")
            return None, "no_client"
        
        # Execute search
        results, search_method = await self._execute_gmail_search(
            collection_name, query_embedding, user_id, user_namespace, None
        )
        
        return results, search_method
    
    async def _process_search_results(self, results, search_method, time_cutoff, cleaned_query, query_hints=None):
        """Process and rank search results with query-aware boosting."""
        query_hints = query_hints or {}
        
        # Extract matches
        if not results or not hasattr(results, "metadatas") or not results.metadatas or not results.metadatas[0]:
            logger.warning(f"⚠️ No results from vector search using {search_method}")
            matches = []
        else:
            matches = self._extract_matches_from_results(results)
            logger.info(f"✅ Found {len(matches)} initial results from vector search using {search_method}")
        
        # Handle no results
        if not matches:
            # Try without time filter if we had one
            if time_cutoff:
                logger.info("🔄 No emails found with time filter, trying without...")
                matches = self._extract_matches_from_results(results)
                filtered_matches = self._filter_and_deduplicate(matches, time_cutoff=None, query_hints=query_hints)
                
                if filtered_matches:
                    logger.info(f"✅ Found {len(filtered_matches)} emails without time filter")
                    return filtered_matches
            
            return []
        
        logger.info(f"Retrieved {len(matches)} initial results")
        
        # Filter and deduplicate (with query-aware boosting)
        filtered_matches = self._filter_and_deduplicate(matches, time_cutoff, query_hints)
        
        # Re-rank (with query hints for additional boosting)
        if self.valves.use_pinecone_reranking:
            filtered_matches = self._rerank_with_pinecone(filtered_matches, cleaned_query)
            # Apply query hint boosts after Pinecone reranking
            filtered_matches = self._apply_query_hint_boosts(filtered_matches, query_hints)
        else:
            filtered_matches = self._rerank_results(filtered_matches, cleaned_query, query_hints)
        
        # Limit to top_k
        return filtered_matches[: self.valves.top_k]

    async def search_gmail(
        self,
        query: str,
        __user__: dict = {},
        __event_emitter__: Optional[Callable] = None,
        __event_call__: Optional[Callable] = None,
    ) -> str:
        """
        Search your Gmail inbox using semantic AI search.

        This searches ALL your synced emails (inbox, sent, labels) and finds
        the most relevant ones based on the meaning of your query.

        Examples:
        - "emails from John about the Q4 budget"
        - "meeting notes from last week"
        - "project proposals I received"
        - "emails I sent to Sarah about contracts"

        :param query: Natural language search query
        :return: Formatted list of relevant emails
        """
        
        # VERSION CHECK
        logger.info("=" * 80)
        logger.info("🚀 GMAIL SEARCH V3.1.0 - HIGH ACCURACY MODE")
        logger.info("=" * 80)
        
        # Step 1: Validate user and prepare query
        user_id = self._get_validated_user_id(__user__)
        cleaned_query, time_cutoff, query_hints = self._prepare_query(query)
        user_namespace = f"email-{user_id}"

        try:
            # Emit initial status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Searching your Gmail inbox...",
                            "done": False,
                        },
                    }
                )
            
            logger.info(f"🔍 Gmail search initiated by user {user_id[:8]}***: '{query}'")
            
            # Step 2: Generate query embedding (use cleaned query for better embeddings)
            embedding_function = await self._get_embedding_function()
            embedding_query = query_hints.get('clean_query', cleaned_query)
            if embedding_function:
                query_embedding = await self._get_query_embedding(embedding_query, embedding_function)
            else:
                logger.error("❌ No embedding function available")
                query_embedding = [0.0] * EMBEDDING_DIMENSION
            
            logger.info(f"✅ Query embedding generated: dimension={len(query_embedding)}")
            
            # Step 3: Perform vector search
            results, search_method_used = await self._perform_search(
                query_embedding, user_id, user_namespace
            )
            
            # Step 4: Process and rank results (with query hints for boosting)
            filtered_matches = await self._process_search_results(
                results, search_method_used, time_cutoff, cleaned_query, query_hints
            )
            
            # Handle no results
            if not filtered_matches:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "No emails found matching your search",
                                "done": True,
                            },
                        }
                    )
                return self._format_no_results(query)
            
            logger.info(f"Returning {len(filtered_matches)} unique emails after filtering and re-ranking")
            
            # Step 5: Format and return results
            formatted_output = await self._format_results(filtered_matches, query, __event_emitter__)
            
            # Final status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Search completed - found {len(filtered_matches)} results",
                            "done": True,
                        },
                    }
                )
            
            logger.info(f"✅ Gmail search complete: {len(filtered_matches)} results for user {user_id}")
            
            return formatted_output

        except ImportError as e:
            logger.error(f"❌ Gmail search import error: {e}")
            return (
                "❌ **System Configuration Error**\n\n"
                "Gmail search is not properly configured. Please contact your administrator.\n\n"
                f"Technical details: {str(e)}"
            )
        except Exception as e:
            logger.error(f"❌ Gmail search error for user {user_id}: {e}", exc_info=True)
            
            if "Collection" in str(e) and "does not exist" in str(e):
                return self._format_collection_not_found_error()
            else:
                return (
                    f"❌ **Gmail Search Error**\n\n"
                    f"An unexpected error occurred: {str(e)}\n\n"
                    f"Please try again or contact support if the issue persists."
                )

    def _validate_query(self, query: str) -> Optional[str]:
        """Validate search query input"""
        if not query or not query.strip():
            return "❌ Error: Search query cannot be empty"

        if len(query) < MIN_QUERY_LENGTH:
            return f"❌ Error: Query too short (minimum {MIN_QUERY_LENGTH} characters)"

        if len(query) > MAX_QUERY_LENGTH:
            return f"❌ Error: Query too long (maximum {MAX_QUERY_LENGTH} characters)"

        return None

    def _extract_time_filter(self, query: str) -> Tuple[Optional[int], str]:
        """
        Extract time period from query and return cutoff timestamp.
        
        Returns:
            Tuple of (cutoff_timestamp, time_filter_info)
        """
        query_lower = query.lower()
        current_time = int(time.time())
        
        # Enhanced time period patterns - handle both recent and historical periods
        time_patterns = [
            # Recent periods (from now)
            (r'(?:last|past|previous|over|in)\s+(?:the\s+)?(\d+)\s+days?', lambda m: int(m.group(1)), 'recent'),
            (r'(?:last|past|previous|over|in)\s+(?:the\s+)?(\d+)\s+weeks?', lambda m: int(m.group(1)) * 7, 'recent'),
            (r'(?:last|past|previous|over|in)\s+(?:the\s+)?(\d+)\s+months?', lambda m: int(m.group(1)) * 30, 'recent'),
            (r'(?:last|past|previous|over|in)\s+(?:the\s+)?(\d+)\s+years?', lambda m: int(m.group(1)) * 365, 'recent'),
            
            # Historical periods (from specific years/months)
            (r'(\d{4})\s+(?:emails|emails\s+from)', lambda m: int(m.group(1)), 'year'),
            (r'(?:emails|emails\s+from)\s+(\d{4})', lambda m: int(m.group(1)), 'year'),
            (r'(?:email|emails)\s+(?:my\s+)?from\s+(\d{4})', lambda m: int(m.group(1)), 'year'),
            (r'from\s+(\d{4})', lambda m: int(m.group(1)), 'year'),
            (r'(?:my\s+)?emails?\s+(?:in|from)\s+(\d{4})', lambda m: int(m.group(1)), 'year'),
            (r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', 
             lambda m: self._get_days_since_month_year(m.group(0)), 'historical'),
            (r'(\d{4})\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)', 
             lambda m: self._get_days_since_month_year(m.group(0)), 'historical'),
            
            # Generic period patterns
            (r'(?:last|past|previous|over|in)\s+(?:the\s+)?week', lambda m: 7, 'recent'),
            (r'(?:last|past|previous|over|in)\s+(?:the\s+)?month', lambda m: 30, 'recent'),
            (r'(?:last|past|previous|over|in)\s+(?:the\s+)?year', lambda m: 365, 'recent'),
            
            # Specific time references
            (r'today', lambda m: 1, 'recent'),
            (r'yesterday', lambda m: 2, 'recent'),
            (r'this\s+week', lambda m: 7, 'recent'),
            (r'this\s+month', lambda m: 30, 'recent'),
            (r'this\s+year', lambda m: 365, 'recent'),
            
            # Historical references
            (r'(?:last\s+)?year', lambda m: 365, 'recent'),  # Last year (recent context)
            (r'(?:in\s+)?(\d{4})', lambda m: self._get_days_since_year(int(m.group(1))), 'historical'),  # Specific year
        ]
        
        cutoff_timestamp = None
        cleaned_query = query
        time_context = None
        
        logger.info(f"🔍 Analyzing query for time patterns: '{query_lower}'")
        
        for pattern, days_func, context in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    days = days_func(match)
                    time_context = context
                    
                    if context == 'recent':
                        # Recent periods: count backwards from now
                        cutoff_timestamp = current_time - (days * 86400)
                        logger.info(f"📅 Recent time filter: last {days} days (cutoff: {datetime.fromtimestamp(cutoff_timestamp).strftime('%Y-%m-%d')})")
                    elif context == 'historical':
                        # Historical periods: use the calculated days
                        cutoff_timestamp = current_time - (days * 86400)
                        logger.info(f"📅 Historical time filter: {days} days ago (cutoff: {datetime.fromtimestamp(cutoff_timestamp).strftime('%Y-%m-%d')})")
                    elif context == 'year':
                        # Year-based periods: return the year directly for later processing
                        cutoff_timestamp = days  # Store the year as the "timestamp"
                        logger.info(f"📅 Year filter detected: {days}")
                    
                    # Just log the time filter detection - we keep the full original query
                    time_filter_info = f"Time filter: {context} ({days if context != 'year' else f'year {days}'})"
                    logger.info(f"📅 Time filter detected: {time_filter_info}")
                    break
                except Exception as e:
                    logger.warning(f"Error processing time pattern '{pattern}': {e}")
                    continue
        
        if cutoff_timestamp is None:
            logger.info("📅 No time filter detected in query")
            time_filter_info = "No time filter"
        else:
            # Check if cutoff_timestamp is a year (for year-based queries)
            if isinstance(cutoff_timestamp, int) and 1900 <= cutoff_timestamp <= 2100:
                logger.info(f"📅 Year filter detected: {cutoff_timestamp} (will be converted to date range)")
                time_filter_info = f"Year filter: {cutoff_timestamp}"
            else:
                logger.info(f"⏰ Will filter emails to after {datetime.fromtimestamp(cutoff_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                time_filter_info = f"Date filter: after {datetime.fromtimestamp(cutoff_timestamp).strftime('%Y-%m-%d')}"
        
        return cutoff_timestamp, time_filter_info
    
    async def _execute_gmail_search(self, collection_name: str, query_embedding: list, user_id: str, user_namespace: str, user_obj) -> tuple:
        """
        Execute Gmail search in user's namespace.
        
        Uses per-user namespace (email-{user_id}) for data isolation.
        Each user's emails are stored in their own namespace.
        """
        from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
        
        # Optimize for two-stage retrieval
        initial_top_k = self.valves.top_k * RERANKING_MULTIPLIER if self.valves.use_pinecone_reranking else self.valves.top_k * FALLBACK_MULTIPLIER
        
        actual_client = VECTOR_DB_CLIENT
        if isinstance(VECTOR_DB_CLIENT, dict):
            actual_client = VECTOR_DB_CLIENT.get('client') or VECTOR_DB_CLIENT.get('vector_db_client')
        
        if not actual_client:
            logger.error("❌ No vector DB client available")
            return None, "no_client"
        
        try:
            # Direct search with per-user namespace
            logger.info(f"   Executing Pinecone query with namespace='{user_namespace}'")
            
            results = actual_client.search(
                collection_name=collection_name,
                vectors=[query_embedding],
                limit=initial_top_k,
                namespace=user_namespace  # Per-user namespace for isolation
            )
            
            logger.info(f"✅ Search completed in namespace '{user_namespace}'")
            return results, "namespace_search"
            
        except Exception as search_error:
            logger.error(f"❌ Gmail search failed: {search_error}", exc_info=True)
            return None, "failed"
    
    def _get_days_since_year(self, year: int) -> int:
        """Calculate days since a specific year"""
        try:
            target_date = datetime(year, 1, 1)
            current_date = datetime.now()
            days_diff = (current_date - target_date).days
            return max(0, days_diff)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Error calculating days since year {year}: {e}")
            return DEFAULT_YEAR_DAYS  # Default to 1 year ago
    
    def _get_days_since_month_year(self, month_year_str: str) -> int:
        """Calculate days since a specific month and year"""
        try:
            # Parse month year string like "September 2024" or "2024 September"
            month_names = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            
            parts = month_year_str.lower().split()
            year = None
            month = None
            
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    year = int(part)
                elif part in month_names:
                    month = month_names[part]
            
            if year and month:
                target_date = datetime(year, month, 1)
                current_date = datetime.now()
                days_diff = (current_date - target_date).days
                return max(0, days_diff)  # Ensure non-negative
            else:
                return DEFAULT_YEAR_DAYS  # Default fallback
        except Exception as e:
            logger.warning(f"Error calculating days since month/year '{month_year_str}': {e}")
            return DEFAULT_YEAR_DAYS  # Default to 1 year ago

    def _format_collection_not_found_error(self) -> str:
        """Format error message when Gmail collection doesn't exist"""
        return (
            "❌ **Gmail Not Synced**\n\n"
            "Your Gmail inbox hasn't been synced yet.\n\n"
            "**To enable Gmail search:**\n"
            "1. Contact your admin to enable Gmail sync for your account\n"
            "2. Wait for initial sync to complete (~5-10 minutes)\n"
            "3. Then try searching again\n\n"
            "**Note:** Gmail emails are now stored in a secure namespace with user isolation.\n\n"
            "**Debug Info:** Check the server logs for detailed error information."
        )

    async def _get_query_embedding(self, query: str, embedding_func) -> list:
        """
        Get query embedding with caching.

        Cache key is based on query hash to avoid recomputing same embeddings.
        Cache expires after 1 hour.
        """
        import hashlib

        # Create cache key
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()

        # Check cache
        if query_hash in self._query_cache:
            cached_embedding, cached_time = self._query_cache[query_hash]
            if time.time() - cached_time < CACHE_EXPIRY_HOURS * 3600:  # Cache expiry
                logger.debug(f"Using cached embedding for query")
                return cached_embedding

        # Generate new embedding
        try:
            # Check if embedding_func is callable
            if not callable(embedding_func):
                logger.error(f"❌ embedding_func is not callable, type: {type(embedding_func)}")
                raise ValueError(f"embedding_func is not callable, got {type(embedding_func)}")
            
            embedding = embedding_func(query, prefix="", user=None)

            # Cache it
            self._query_cache[query_hash] = (embedding, time.time())

            # Limit cache size to prevent memory leaks
            if len(self._query_cache) > CACHE_SIZE_LIMIT:
                # Remove oldest entries (remove 25% when limit exceeded)
                sorted_keys = sorted(self._query_cache.keys(), key=lambda k: self._query_cache[k][1])
                for key in sorted_keys[:CACHE_CLEANUP_SIZE]:  # Remove oldest entries
                    del self._query_cache[key]

            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def _extract_matches_from_results(self, results) -> list:
        """Extract match objects from SearchResult"""
        matches = []
        num_results = len(results.metadatas[0])

        for i in range(num_results):
            # Calculate score from distance (lower distance = higher score)
            # Assumes cosine distance or euclidean distance normalized to [0, 1]
            distance = (
                results.distances[0][i]
                if results.distances and results.distances[0]
                else 1.0
            )

            # Convert distance to similarity score
            # For cosine distance: similarity = 1 - distance
            score = max(0.0, min(1.0, 1.0 - distance))

            match_obj = type(
                "Match", (), {"score": score, "metadata": results.metadatas[0][i]}
            )()
            matches.append(match_obj)

        return matches

    def _filter_and_deduplicate(self, matches: list, time_cutoff: Optional[int] = None, query_hints: dict = None) -> list:
        """
        Filter matches by type, score, time period and de-duplicate by email_id.
        V4 schema: Single vector per email (no chunking).
        """
        query_hints = query_hints or {}
        seen_emails = {}  # email_id -> best match

        # Debug: Log score distribution
        scores = [m.score for m in matches]
        if scores:
            logger.info(
                f"📊 Score distribution: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}"
            )
            logger.info(
                f"📊 Scores below threshold ({self.valves.min_score}): {sum(1 for s in scores if s < self.valves.min_score)}/{len(scores)}"
            )

        time_filtered_count = 0

        for idx, match in enumerate(matches):
            metadata = match.metadata
            score = match.score
            
            # Filter by time period if specified
            if time_cutoff:
                email_timestamp = metadata.get("date_timestamp", 0)
                
                # Handle both single timestamp and date range tuple
                if isinstance(time_cutoff, tuple):
                    # Date range: (start_timestamp, end_timestamp)
                    start_time, end_time = time_cutoff
                    if email_timestamp < start_time or email_timestamp > end_time:
                        time_filtered_count += 1
                        logger.debug(f"Filtered out email outside date range: {metadata.get('subject', 'N/A')[:50]} (timestamp: {email_timestamp})")
                        continue
                else:
                    # Single timestamp: filter emails before this date
                    if email_timestamp < time_cutoff:
                        time_filtered_count += 1
                        logger.debug(f"Filtered out email outside time range: {metadata.get('subject', 'N/A')[:50]}")
                        continue

            # Note: show_summary_only is deprecated in V4 (single vector per email)

            # Filter by type (ensure it's an email)
            if metadata.get("type") != "email":
                logger.debug(f"Filtered out non-email type (score: {score:.3f})")
                continue
            
            # Filter out obvious automated emails (optional - can disable)
            from_email = metadata.get("from_email", "").lower()
            subject_lower = metadata.get("subject", "").lower()
            
            # Completely skip automated/CI/newsletter emails
            auto_skip_patterns = [
                '@github.com',  # GitHub notifications
                'via saas login',  # Forwarded automated emails
                'run failed', 'build failed',  # CI/CD alerts
            ]
            
            should_skip = any(pattern in from_email or pattern in subject_lower for pattern in auto_skip_patterns)
            if should_skip:
                logger.debug(f"🔕 Skipping automated email: {metadata.get('subject', '')[:50]}")
                continue

            # Filter by quality_score (skip low-quality emails)
            quality_score = metadata.get("quality_score", 0)
            if isinstance(quality_score, str):
                try:
                    quality_score = int(quality_score)
                except:
                    quality_score = 0
            
            if quality_score < 2:  # Skip very low quality emails
                logger.debug(f"Filtered out low quality email (quality={quality_score})")
                continue
            
            # Filter by min_score
            if score < self.valves.min_score:
                logger.debug(
                    f"Filtered out low score: {score:.3f} < {self.valves.min_score} - Email: {metadata.get('subject', 'N/A')[:50]}"
                )
                continue

            # De-duplicate by email_id (keep highest scoring chunk per email)
            email_id = metadata.get("email_id")
            if email_id:
                if email_id not in seen_emails or score > seen_emails[email_id].score:
                    seen_emails[email_id] = match
            else:
                # No email_id (shouldn't happen), add anyway
                seen_emails[f"unknown_{idx}"] = match

        if time_filtered_count > 0:
            logger.info(f"⏰ Filtered out {time_filtered_count} emails outside specified time range")
            # If we filtered out ALL emails due to time, provide helpful feedback
            if time_filtered_count == len(matches) and time_cutoff:
                if isinstance(time_cutoff, tuple):
                    start_date = datetime.fromtimestamp(time_cutoff[0]).strftime('%Y-%m-%d')
                    end_date = datetime.fromtimestamp(time_cutoff[1]).strftime('%Y-%m-%d')
                    logger.warning(f"⚠️ Time filter removed ALL emails. Date range: {start_date} to {end_date}")
                    logger.warning(f"⚠️ This suggests no emails exist in the specified date range")
                else:
                    cutoff_date = datetime.fromtimestamp(time_cutoff).strftime('%Y-%m-%d')
                    current_date = datetime.now().strftime('%Y-%m-%d')
                    logger.warning(f"⚠️ Time filter removed ALL emails. Cutoff: {cutoff_date}, Current: {current_date}")
                    logger.warning(f"⚠️ This suggests your emails are older than the specified time range")
            
        logger.info(f"📊 Final filtering results: {len(seen_emails)} emails after deduplication and filtering")

        return list(seen_emails.values())
    
    def _apply_query_hint_boosts(self, matches: list, query_hints: dict) -> list:
        """
        Apply query-specific boosts based on extracted hints.
        This runs AFTER Pinecone reranking to fine-tune results.
        
        Boosts:
        - Sender match: +25% if sender hint matches from_name/from_email
        - Recipient match: +20% if recipient hint matches to_name/to_email
        - Attachment match: +15% if expects_attachment and has_attachments
        """
        if not query_hints:
            return matches
        
        sender_hints = [h.lower() for h in query_hints.get('sender_hints', [])]
        recipient_hints = [h.lower() for h in query_hints.get('recipient_hints', [])]
        expects_attachment = query_hints.get('expects_attachment', False)
        
        if not sender_hints and not recipient_hints and not expects_attachment:
            return matches  # No hints to apply
        
        boosted_count = 0
        
        for match in matches:
            metadata = match.metadata
            boost = 1.0
            
            # Sender boost
            if sender_hints:
                from_name = metadata.get('from_name', '').lower()
                from_email = metadata.get('from_email', '').lower()
                for hint in sender_hints:
                    if hint in from_name or hint in from_email:
                        boost *= 1.25  # +25% boost
                        logger.debug(f"  📧 Sender boost applied for '{hint}' in {from_name or from_email}")
                        break
            
            # Recipient boost
            if recipient_hints:
                to_name = metadata.get('to_name', '').lower()
                to_email = metadata.get('to_email', '').lower()
                for hint in recipient_hints:
                    if hint in to_name or hint in to_email:
                        boost *= 1.20  # +20% boost
                        logger.debug(f"  📧 Recipient boost applied for '{hint}' in {to_name or to_email}")
                        break
            
            # Attachment boost
            if expects_attachment and metadata.get('has_attachments', False):
                boost *= 1.15  # +15% boost
                logger.debug(f"  📎 Attachment boost applied")
            
            if boost > 1.0:
                match.score = min(1.0, match.score * boost)  # Cap at 1.0
                boosted_count += 1
        
        if boosted_count > 0:
            logger.info(f"🎯 Applied query hint boosts to {boosted_count} emails")
            # Re-sort by new scores
            matches.sort(key=lambda m: m.score, reverse=True)
        
        return matches

    def _rerank_with_pinecone(self, matches: list, query: str) -> list:
        """
        Use Pinecone's hosted reranking models following their best practices.
        
        This follows Pinecone's recommended two-stage retrieval process:
        1. Initial vector search (already done)
        2. Rerank using hosted models for better relevance
        
        Reference: https://docs.pinecone.io/guides/optimize/increase-relevance
        """
        # IMPORTANT: Pinecone has a 512 token limit per COMBINED query+document pair
        # Tokens ≈ 3-3.5 chars (conservative estimate)
        # Query can be up to 500 chars (~143 tokens), leaving ~369 tokens for document
        # 369 tokens * 3 chars/token = ~1107 chars, so use 1000 chars to be safe
        MAX_DOC_CHARS = 1000  # Conservative limit to stay under 512 token limit for query+doc
        
        if not matches:
            return matches
            
        try:
            from pinecone import Pinecone
            from open_webui.config import PINECONE_API_KEY
            
            # Check if we have Pinecone API key
            if not PINECONE_API_KEY:
                logger.warning("⚠️ Pinecone API key not available for reranking")
                return self._rerank_results(matches, query)
            
            logger.info(f"🔄 Using Pinecone hosted reranking for {len(matches)} results")
            logger.info(f"   Model: {self.valves.reranking_model}")
            
            # Initialize Pinecone client
            pc = Pinecone(PINECONE_API_KEY)
            
            # Adjust MAX_DOC_CHARS based on actual query length
            # Total limit: 512 tokens for query+doc
            # Estimate: 1 token ≈ 3 chars (conservative)
            query_tokens_estimate = len(query) // 3
            remaining_tokens = 512 - query_tokens_estimate - 10  # 10 token buffer
            adjusted_max_doc_chars = min(MAX_DOC_CHARS, remaining_tokens * 3)
            
            logger.info(f"   Query tokens estimate: ~{query_tokens_estimate}, adjusted doc limit: {adjusted_max_doc_chars} chars")
            
            # Prepare documents for reranking (following Pinecone's format)
            # V4 schema: Use rerank_text (3000 chars) for optimal re-ranking
            documents = []
            for match in matches:
                subject = match.metadata.get('subject', '')[:150]  # Limit subject to 150 chars
                from_name = match.metadata.get('from_name', '')
                from_email = match.metadata.get('from_email', '')
                from_info = f"{from_name} <{from_email}>" if from_name else from_email
                from_info = from_info[:80]  # Limit from to 80 chars
                
                # V4: Use rerank_text (3000 chars), fall back to chunk_text (1500 chars)
                content = match.metadata.get('rerank_text', '') or match.metadata.get('chunk_text', '')
                
                # Calculate remaining space for content
                header_length = len(f"Subject: {subject}\nFrom: {from_info}\nContent: ")
                max_content_length = adjusted_max_doc_chars - header_length
                
                # Ensure we have at least some content space
                if max_content_length < 100:
                    max_content_length = 100
                
                # Truncate content to fit within limit
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                
                doc_text = f"Subject: {subject}\n"
                doc_text += f"From: {from_info}\n"
                doc_text += f"Content: {content}"
                documents.append(doc_text)
            
            # Log document size stats for debugging
            doc_lengths = [len(doc) for doc in documents]
            max_doc_len = max(doc_lengths) if doc_lengths else 0
            avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
            logger.info(f"   Document sizes: max={max_doc_len} chars, avg={avg_doc_len:.0f} chars")
            logger.info(f"   Query length: {len(query)} chars")
            
            # Use Pinecone's inference.rerank method (following best practices)
            results = pc.inference.rerank(
                model=self.valves.reranking_model,
                query=query,
                documents=documents,
                top_n=len(matches),
                return_documents=False  # We only need scores, not full documents
            )
            
            # Reorder matches based on reranking results
            if results.data:
                reranked_matches = []
                pinecone_scores = []
                
                # First pass: collect all Pinecone scores and matches
                for result in results.data:
                    if result.index < len(matches):
                        match = matches[result.index]
                        original_score = getattr(match, 'score', 0.0)
                        pinecone_score = result.score
                        
                        # Store original data
                        match.pinecone_rerank_score = pinecone_score
                        match.original_score = original_score
                        pinecone_scores.append(pinecone_score)
                        reranked_matches.append(match)
                
                # Second pass: percentile-based normalization
                if pinecone_scores:
                    min_pinecone = min(pinecone_scores)
                    max_pinecone = max(pinecone_scores)
                    
                    # Use percentile-based normalization (0.1 to 0.9 range)
                    # This preserves relative differences better than fixed scaling
                    if max_pinecone > min_pinecone:
                        # Normalize to 0.1-0.9 range based on relative position
                        for i, match in enumerate(reranked_matches):
                            pinecone_score = match.pinecone_rerank_score
                            # Calculate percentile position (0 to 1)
                            percentile = (pinecone_score - min_pinecone) / (max_pinecone - min_pinecone)
                            # Map to 0.1-0.9 range
                            normalized_score = 0.1 + (percentile * 0.8)
                            match.score = normalized_score
                    else:
                        # All scores are the same, give them a middle score
                        for match in reranked_matches:
                            match.score = 0.5
                
                logger.info(f"✅ Pinecone reranking completed: {len(reranked_matches)} results reranked using {self.valves.reranking_model}")
                if pinecone_scores:
                    logger.info(f"📊 Pinecone score range: {min(pinecone_scores):.2e} - {max(pinecone_scores):.2e}")
                    logger.info(f"📊 Normalized score range: {min(m.score for m in reranked_matches):.3f} - {max(m.score for m in reranked_matches):.3f}")
                return reranked_matches
            else:
                logger.warning("⚠️ Pinecone reranking returned no results, using original ranking")
                return matches
                
        except ImportError as e:
            logger.error(f"❌ Pinecone SDK not available: {e}")
            logger.info("🔄 Falling back to custom reranking")
            return self._rerank_results(matches, query)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error in Pinecone reranking: {e}")
            
            # Provide specific guidance for token limit errors
            if "token limit" in error_msg.lower() or "exceeds the maximum" in error_msg.lower():
                logger.error(f"⚠️ Token limit exceeded in reranking request")
                logger.error(f"   This should not happen with current truncation (max {MAX_DOC_CHARS} chars)")
                logger.error(f"   Consider reducing MAX_DOC_CHARS further or using summary instead of full content")
            
            logger.info("🔄 Falling back to custom reranking")
            return self._rerank_results(matches, query)

    def _rerank_results(self, matches: list, query: str = "", query_hints: dict = None) -> list:
        """
        Re-rank results using composite score optimized for accuracy.
        
        V3 Formula (accuracy-focused):
        - Semantic relevance: 80% (primary factor for accuracy)
        - Recency: 12% (slight boost for recent emails)
        - Importance: 8% (labels like IMPORTANT, STARRED)
        - Query hint boosts: Applied as multipliers
        """
        query_hints = query_hints or {}
        
        # Extract query keywords for exact match boosting (filter out stop words)
        query_keywords = set()
        if query:
            for word in query.lower().split():
                if word not in STOP_WORDS and len(word) > 2:
                    query_keywords.add(word)
        
        # Extract hints
        sender_hints = [h.lower() for h in query_hints.get('sender_hints', [])]
        recipient_hints = [h.lower() for h in query_hints.get('recipient_hints', [])]
        expects_attachment = query_hints.get('expects_attachment', False)

        for match in matches:
            metadata = match.metadata

            # Base semantic score (most important for accuracy)
            semantic_score = match.score

            # V4: Use pre-computed recency_score if available, otherwise calculate
            recency_score = metadata.get("recency_score")
            if recency_score is None:
                # Fallback calculation
                date_timestamp = metadata.get("date_timestamp", 0)
                if date_timestamp:
                    age_days = (int(time.time()) - date_timestamp) / 86400
                    # Smooth decay: 1.0 at day 0, ~0.5 at 30 days, ~0.1 at 180 days
                    recency_score = max(0.1, 1.0 / (1.0 + age_days / 30))
                else:
                    recency_score = 0.5

            # Detect automated/notification emails
            from_email = metadata.get("from_email", "").lower()
            from_name = metadata.get("from_name", "").lower()
            subject = metadata.get("subject", "").lower()
            
            is_notification = False
            notification_patterns = [
                'noreply@', 'no-reply@', 'notifications@', 'alerts@',
                'donotreply@', 'do-not-reply@',
                'community@', 'newsletter@', 'updates@',
                'via saas login', 'hubhelp@'
            ]
            
            for pattern in notification_patterns:
                if pattern in from_email or pattern in subject:
                    is_notification = True
                    break
            
            # Importance multiplier from labels
            importance_multiplier = 1.0
            labels = metadata.get("labels", [])
            if isinstance(labels, list):
                labels_list = labels
            elif isinstance(labels, str):
                labels_list = [l.strip() for l in labels.split(",") if l.strip()]
            else:
                labels_list = []
            
            if is_notification:
                importance_multiplier = 0.5  # Penalize but not as heavily
            elif "IMPORTANT" in labels_list:
                importance_multiplier = 1.4
            elif "STARRED" in labels_list:
                importance_multiplier = 1.3
            
            # === QUERY HINT BOOSTS ===
            hint_boost = 1.0
            
            # Sender hint match
            if sender_hints:
                for hint in sender_hints:
                    if hint in from_name or hint in from_email:
                        hint_boost *= 1.30  # +30% for sender match
                        logger.debug(f"  📧 Sender match: '{hint}'")
                        break
            
            # Recipient hint match
            if recipient_hints:
                to_name = metadata.get('to_name', '').lower()
                to_email = metadata.get('to_email', '').lower()
                for hint in recipient_hints:
                    if hint in to_name or hint in to_email:
                        hint_boost *= 1.25  # +25% for recipient match
                        logger.debug(f"  📧 Recipient match: '{hint}'")
                        break
            
            # Attachment hint match
            if expects_attachment and metadata.get('has_attachments', False):
                hint_boost *= 1.20  # +20% for attachment match
                logger.debug(f"  📎 Attachment match")
            
            # === KEYWORD BOOST ===
            # Match keywords in subject, from_name, and content
            keyword_boost = 1.0
            if query_keywords:
                subject_words = set(subject.split())
                from_words = set(from_name.split())
                content_text = metadata.get('chunk_text', '').lower()[:500]
                content_words = set(content_text.split())
                
                # Subject matches are most valuable
                subject_overlap = len(query_keywords & subject_words)
                from_overlap = len(query_keywords & from_words)
                content_overlap = len(query_keywords & content_words)
                
                if subject_overlap > 0:
                    keyword_boost += min(0.4, subject_overlap * 0.15)  # Max +40% from subject
                if from_overlap > 0:
                    keyword_boost += min(0.2, from_overlap * 0.10)  # Max +20% from sender
                if content_overlap > 0:
                    keyword_boost += min(0.15, content_overlap * 0.03)  # Max +15% from content

            # === CALCULATE FINAL SCORE ===
            # Accuracy-focused weights: 80% semantic, 12% recency, 8% importance
            base_score = (
                semantic_score * 0.80  # Semantic relevance (primary)
                + recency_score * 0.12  # Recency (secondary)
                + (semantic_score * importance_multiplier * 0.08)  # Importance-adjusted
            )
            
            # Apply boosts (multiplicative)
            final_score = base_score * keyword_boost * hint_boost
            
            # Cap at 1.0
            final_score = min(1.0, final_score)

            # Store scores for debugging
            match.semantic_score = semantic_score
            match.final_score = final_score
            match.score = final_score  # Update the main score
            match.is_notification = is_notification
            match.keyword_boost = keyword_boost
            match.hint_boost = hint_boost

        # Sort by final score (descending)
        matches.sort(key=lambda m: m.final_score, reverse=True)

        return matches

    async def _format_results(
        self, matches: list, query: str, __event_emitter__: Optional[Callable] = None
    ) -> str:
        """Format search results for display"""

        results = []
        citations = []

        for i, match in enumerate(matches):
            score = getattr(match, "score", 0.0)
            
            # Filter by minimum score
            if score < self.valves.min_score:
                continue

            metadata = getattr(match, "metadata", {})

            # Extract email details (V4 schema)
            subject = metadata.get("subject", "No Subject")
            from_name = metadata.get("from_name", "")
            from_email = metadata.get("from_email", "")
            to_name = metadata.get("to_name", "")
            to_email = metadata.get("to_email", "")
            date = metadata.get("date", "")
            labels_raw = metadata.get("labels", [])
            # V4: Labels are stored as arrays in Pinecone
            if isinstance(labels_raw, list):
                labels = labels_raw
            elif isinstance(labels_raw, str):
                # Backwards compatibility: handle comma-separated string
                labels = [l.strip() for l in labels_raw.split(",") if l.strip()]
            else:
                labels = []
            is_reply = metadata.get("is_reply", False)
            word_count = metadata.get("word_count", 0)
            quality_score = metadata.get("quality_score", 0)

            # Build sender display
            if from_name:
                sender = f"{from_name} <{from_email}>"
            else:
                sender = (
                    from_email if from_email else metadata.get("from_raw", "Unknown")
                )

            # Build recipient display
            if to_name:
                recipient = f"{to_name} <{to_email}>"
            else:
                recipient = to_email if to_email else metadata.get("to_raw", "Unknown")

            # Determine if this was sent or received
            direction = "📤 Sent" if is_reply or "SENT" in labels else "📥 Received"

            # V4: Get email content (chunk_text for full, snippet for preview)
            chunk_text = metadata.get("chunk_text", "") or metadata.get("rerank_text", "")
            snippet = metadata.get("snippet", "")
            
            # Format based on settings
            if self.valves.format_results == "compact":
                result_text = self._format_compact(i + 1, subject, sender, date, score)
            else:
                result_text = self._format_detailed(
                    i + 1,
                    subject,
                    sender,
                    recipient,
                    date,
                    chunk_text,  # Pass actual email content
                    labels,
                    direction,
                    word_count,
                    score,
                    quality_score,
                )

            results.append(result_text)

            # Prepare citation data (V4: use chunk_text or snippet)
            citations.append(
                {
                    "subject": subject,
                    "from": sender,
                    "date": date,
                    "text": chunk_text[:500] if chunk_text else snippet,
                    "score": score,
                }
            )

        if not results:
            logger.warning(f"⚠️ No results after formatting. Total matches processed: {len(matches)}")
            logger.warning(f"⚠️ Min score threshold: {self.valves.min_score}")
            return self._format_no_results(query)

        # Emit citation events
        if __event_emitter__:
            await self._emit_citations(citations, __event_emitter__)

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Found {len(results)} relevant email(s)",
                        "done": True,
                    },
                }
            )

        # Count notifications vs conversational
        notification_count = sum(1 for m in matches if getattr(m, 'is_notification', False))
        conversational_count = len(matches) - notification_count
        
        # Build final output
        header = f"# 📧 Gmail Search Results\n\n"
        header += f"**Query:** _{query}_\n\n"
        header += f"**Found {len(results)} relevant email(s)**\n\n"
        
        # Add quality breakdown
        if notification_count > 0:
            header += f"_({conversational_count} conversational, {notification_count} notifications filtered down)_\n\n"
        
        header += "---\n\n"

        return header + "\n\n".join(results)

    def _format_compact(
        self, num: int, subject: str, sender: str, date: str, score: float
    ) -> str:
        """Format result in compact mode"""
        return (
            f"**{num}. {subject}**  \n" f"From: {sender} | {date} | Score: {score:.2f}"
        )

    def _format_detailed(
        self,
        num: int,
        subject: str,
        sender: str,
        recipient: str,
        date: str,
        content: str,  # chunk_text from V4 schema
        labels: list,
        direction: str,
        word_count: int,
        score: float,
        quality_score: int,
    ) -> str:
        """Format result in detailed mode with email content (V4 schema)"""

        result = f"### {num}. {subject}\n\n"
        result += f"**{direction}**  \n"
        result += f"**From:** {sender}  \n"
        result += f"**To:** {recipient}  \n"
        result += f"**Date:** {date}  \n"

        if labels and len(labels) > 0:
            # Show relevant labels (exclude common ones)
            interesting_labels = [
                l for l in labels if l not in ["UNREAD", "CATEGORY_PERSONAL"]
            ]
            if interesting_labels:
                labels_str = ", ".join(interesting_labels[:3])
                result += f"**Labels:** {labels_str}  \n"

        # Show actual email content (chunk_text from V4)
        if content and len(content) > 10:
            # Limit content display to 500 chars for readability
            display_content = content[:500]
            if len(content) > 500:
                display_content += "..."
            result += f"\n**Content:**\n>{display_content}\n"

        result += (
            f"\n_Words: {word_count} | Relevance: {score:.2f} | Quality: {quality_score}_"
        )

        return result

    def _format_no_results(self, query: str) -> str:
        """Format no results message"""
        return (
            f"# 📭 No Emails Found\n\n"
            f"No emails matched your search: _{query}_\n\n"
            f"**Possible reasons:**\n"
            f"- Gmail sync may not be enabled for your account\n"
            f"- No emails exist in the specified time range\n"
            f"- Search terms may be too specific\n\n"
            f"**Try these solutions:**\n"
            f"1. **Contact your admin** to enable Gmail sync\n"
            f"2. **Use broader search terms** (e.g., 'budget' instead of 'Q4 budget planning')\n"
            f"3. **Try different time ranges** (e.g., 'last 30 days' instead of 'last 120 days')\n"
            f"4. **Search by sender** (e.g., 'emails from john')\n"
            f"5. **Lower the min_score threshold** in tool settings\n\n"
            f"**Debug:** Check server logs for detailed error information."
        )

    async def _emit_citations(self, citations: list, __event_emitter__) -> None:
        """Emit citation events for each email result"""

        for citation in citations:
            await __event_emitter__(
                {
                    "type": "citation",
                    "data": {
                        "document": [citation["text"]],
                        "metadata": [
                            {
                                "source": f"Gmail: {citation['subject']}",
                                "from": citation["from"],
                                "date": citation["date"],
                                "score": citation["score"],
                            }
                        ],
                        "source": {
                            "name": citation["subject"],
                        },
                    },
                }
            )