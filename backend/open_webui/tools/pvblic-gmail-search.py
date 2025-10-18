"""
title: Gmail Semantic Search
author: Open WebUI
author_url: https://github.com/open-webui
version: 1.1.0
license: MIT
description: Search your entire Gmail inbox using AI-powered semantic search. Searches across all your emails (INBOX, SENT, and all labels) to find relevant messages based on meaning, not just keywords.
required_open_webui_version: 0.3.0
requirements: openai
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import json
import time

logger = logging.getLogger(__name__)

# Constants
MAX_QUERY_LENGTH = 500
MIN_QUERY_LENGTH = 3
SEARCH_TIMEOUT_SECONDS = 30


class Tools:
    class Valves(BaseModel):
        top_k: int = Field(
            default=25, description="Maximum number of email results to return (1-50)"
        )
        min_score: float = Field(
            default=0.05,
            description="Minimum relevance score threshold (0.0-1.0). Higher = more strict matching. Lowered to 0.05 to catch more email matches.",
        )
        show_summary_only: bool = Field(
            default=False,
            description="Show only email summaries (faster). Disable to also search email content chunks",
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
        self._query_cache = (
            {}
        )  # Simple in-memory cache {query_hash: (embedding, timestamp)}

    async def search_gmail(
        self,
        query: str,
        __user__: dict = {},
        __event_emitter__=None,
        __event_call__=None,
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

        user_id = __user__.get("id")
        if not user_id:
            return "❌ Error: User not authenticated"

        # Validate query input
        validation_error = self._validate_query(query)
        if validation_error:
            return validation_error

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

            # Import dependencies
            from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
            from open_webui.main import app as webui_app

            logger.info(f"🔍 Gmail search initiated by user {user_id}: '{query}'")

            # Generate query embedding (with caching)
            query_embedding = await self._get_query_embedding(
                query, webui_app.state.EMBEDDING_FUNCTION
            )

            if not query_embedding or len(query_embedding) == 0:
                return "❌ Error: Failed to generate search embedding"

            # Try to search Gmail emails directly (skip collection check)
            collection_name = "gmail"  # Shared collection for all Gmail emails
            logger.info(
                f"Searching Gmail collection '{collection_name}' for user {user_id} with top_k: {self.valves.top_k}"
            )

            try:
                # Search using the new namespace-based approach with user isolation
                results = await self._search_gmail_with_user_isolation(
                    VECTOR_DB_CLIENT, collection_name, query_embedding, user_id, self.valves.top_k * 2
                )
            except Exception as search_error:
                logger.warning(f"Gmail search failed: {search_error}")
                # If search fails, it likely means no Gmail emails are synced
                return self._format_collection_not_found_error()

            # Process results
            # SearchResult has: ids, documents, metadatas, distances (all are lists of lists)
            if (
                not results
                or not hasattr(results, "metadatas")
                or not results.metadatas
                or not results.metadatas[0]
            ):
                logger.warning(f"⚠️  No results from vector search")
                return self._format_no_results(query)

            # Extract matches from SearchResult
            matches = self._extract_matches_from_results(results)

            if not matches:
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

            logger.info(f"Retrieved {len(matches)} initial results")

            # Post-filter and de-duplicate results
            filtered_matches = self._filter_and_deduplicate(matches)

            # Re-rank with recency and importance boost
            filtered_matches = self._rerank_results(filtered_matches)

            # Limit to top_k
            filtered_matches = filtered_matches[: self.valves.top_k]

            logger.info(
                f"Returning {len(filtered_matches)} unique emails after filtering and re-ranking"
            )

            # Format results
            formatted_output = await self._format_results(
                filtered_matches, query, __event_emitter__
            )

            logger.info(
                f"✅ Gmail search complete: {len(filtered_matches)} results for user {user_id}"
            )

            return formatted_output

        except ImportError as e:
            logger.error(f"❌ Gmail search import error: {e}")
            return (
                "❌ **System Configuration Error**\n\n"
                "Gmail search is not properly configured. Please contact your administrator.\n\n"
                f"Technical details: {str(e)}"
            )
        except Exception as e:
            logger.error(
                f"❌ Gmail search error for user {user_id}: {e}", exc_info=True
            )

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

    def _check_collection_exists(self, vector_db_client, collection_name: str) -> bool:
        """Check if a Pinecone collection exists"""
        try:
            # Try to get collection info
            vector_db_client.has_collection(collection_name)
            return True
        except Exception as e:
            logger.warning(f"Collection '{collection_name}' not found: {e}")
            return False

    def _check_gmail_collection_exists(self, vector_db_client, collection_name: str, user_id: str) -> bool:
        """Check if Gmail collection exists with new namespace-based approach (deprecated - now using direct search)"""
        # This method is kept for backward compatibility but is no longer used
        # The tool now tries to search directly and handles failures gracefully
        return True  # Always return True to avoid blocking the search attempt

    async def _search_gmail_with_user_isolation(self, vector_db_client, collection_name: str, query_embedding: list, user_id: str, limit: int):
        """Search Gmail emails with proper user isolation using namespace or collection-based approach"""
        try:
            # Check if the vector DB client supports namespace and user filtering
            if hasattr(vector_db_client, 'search_with_user_filter'):
                # Pinecone with namespace support
                from open_webui.config import PINECONE_NAMESPACE_GMAIL
                gmail_namespace = PINECONE_NAMESPACE_GMAIL.value if hasattr(PINECONE_NAMESPACE_GMAIL, 'value') else PINECONE_NAMESPACE_GMAIL
                
                logger.info(f"Using namespace-based search: collection='{collection_name}', namespace='{gmail_namespace}', user_id='{user_id}'")
                return vector_db_client.search_with_user_filter(
                    collection_name=collection_name,
                    vectors=[query_embedding],
                    limit=limit,
                    user_id=user_id,
                    namespace=gmail_namespace
                )
            else:
                # Fallback to collection-based approach (e.g., Chroma)
                user_collection = f"gmail_{user_id}"
                logger.info(f"Using collection-based search: collection='{user_collection}'")
                return vector_db_client.search(
                    collection_name=user_collection,
                    vectors=[query_embedding],
                    limit=limit
                )
        except Exception as e:
            logger.error(f"Gmail search with user isolation failed: {e}")
            raise

    def _format_collection_not_found_error(self) -> str:
        """Format error message when Gmail collection doesn't exist"""
        return (
            "❌ **Gmail Not Synced**\n\n"
            "Your Gmail inbox hasn't been synced yet.\n\n"
            "**To enable Gmail search:**\n"
            "1. Contact your admin to enable Gmail sync for your account\n"
            "2. Wait for initial sync to complete (~5-10 minutes)\n"
            "3. Then try searching again\n\n"
            "**Note:** Gmail emails are now stored in a secure namespace with user isolation."
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
            if time.time() - cached_time < 3600:  # 1 hour cache
                logger.debug(f"Using cached embedding for query")
                return cached_embedding

        # Generate new embedding
        try:
            embedding = embedding_func(query, prefix="", user=None)

            # Cache it
            self._query_cache[query_hash] = (embedding, time.time())

            # Limit cache size to 100 entries
            if len(self._query_cache) > 100:
                # Remove oldest entry
                oldest_key = min(
                    self._query_cache.keys(), key=lambda k: self._query_cache[k][1]
                )
                del self._query_cache[oldest_key]

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

    def _filter_and_deduplicate(self, matches: list) -> list:
        """
        Filter matches by type, chunk_type, score and de-duplicate by email_id.
        Keeps the highest scoring chunk per email.
        """
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

        for idx, match in enumerate(matches):
            metadata = match.metadata
            score = match.score

            # Filter by chunk_type if show_summary_only
            if self.valves.show_summary_only:
                if metadata.get("chunk_type") != "summary":
                    logger.debug(f"Filtered out non-summary chunk (score: {score:.3f})")
                    continue

            # Filter by type (ensure it's an email)
            if metadata.get("type") != "email":
                logger.debug(f"Filtered out non-email type (score: {score:.3f})")
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

        return list(seen_emails.values())

    def _rerank_results(self, matches: list) -> list:
        """
        Re-rank results using composite score (semantic + recency + importance).

        Formula: final_score = semantic * 0.6 + recency * 0.3 + importance * 0.1
        """
        current_time = int(time.time())

        for match in matches:
            metadata = match.metadata

            # Base semantic score
            semantic_score = match.score

            # Recency score (newer emails ranked higher)
            date_timestamp = metadata.get("date_timestamp", 0)
            if date_timestamp:
                # Calculate age in days
                age_seconds = current_time - date_timestamp
                age_days = age_seconds / 86400

                # Recency score: 1.0 for recent, decreasing with age
                if age_days < 7:
                    recency_score = 1.0
                elif age_days < 30:
                    recency_score = 0.7
                elif age_days < 90:
                    recency_score = 0.3
                else:
                    recency_score = 0.1
            else:
                recency_score = 0.5  # Unknown date

            # Importance multiplier (boost for IMPORTANT, STARRED labels)
            importance_multiplier = 1.0
            labels = metadata.get("labels", [])
            if isinstance(labels, list):
                if "IMPORTANT" in labels:
                    importance_multiplier = 1.3
                elif "STARRED" in labels:
                    importance_multiplier = 1.2

            # Calculate final score with weighted components
            final_score = (
                semantic_score * 0.6  # Semantic relevance (60%)
                + recency_score * 0.3  # How recent (30%)
                + (
                    semantic_score * 0.1 * importance_multiplier
                )  # Importance boost (10%)
            )

            # Store both scores
            match.semantic_score = semantic_score
            match.final_score = final_score

        # Sort by final score (descending)
        matches.sort(key=lambda m: m.final_score, reverse=True)

        return matches

    async def _format_results(
        self, matches: list, query: str, __event_emitter__=None
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

            # Extract email details
            subject = metadata.get("subject", "No Subject")
            from_name = metadata.get("from_name", "")
            from_email = metadata.get("from_email", "")
            to_name = metadata.get("to_name", "")
            to_email = metadata.get("to_email", "")
            date = metadata.get("date", "")
            email_summary = metadata.get("email_summary", "")
            chunk_type = metadata.get("chunk_type", "content")
            labels = metadata.get("labels", [])
            is_reply = metadata.get("is_reply", False)
            word_count = metadata.get("word_count", 0)

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
                    email_summary,
                    labels,
                    direction,
                    word_count,
                    score,
                    chunk_type,
                )

            results.append(result_text)

            # Prepare citation data
            chunk_text = metadata.get("chunk_text", email_summary)
            citations.append(
                {
                    "subject": subject,
                    "from": sender,
                    "date": date,
                    "text": chunk_text[:500] if chunk_text else email_summary,
                    "score": score,
                }
            )

        if not results:
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

        # Build final output
        header = f"# 📧 Gmail Search Results\n\n"
        header += f"**Query:** _{query}_\n\n"
        header += f"**Found {len(results)} relevant email(s)**\n\n"
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
        summary: str,
        labels: list,
        direction: str,
        word_count: int,
        score: float,
        chunk_type: str,
    ) -> str:
        """Format result in detailed mode"""

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

        if summary:
            result += f"\n**Summary:** {summary}\n"

        result += (
            f"\n_Words: {word_count} | Relevance: {score:.2f} | Type: {chunk_type}_"
        )

        return result

    def _format_no_results(self, query: str) -> str:
        """Format no results message"""
        return (
            f"# 📭 No Emails Found\n\n"
            f"No emails matched your search: _{query}_\n\n"
            f"**Tips:**\n"
            f"- Try broader search terms\n"
            f"- Check if Gmail sync is enabled for your account\n"
            f"- Try searching for sender names or subjects\n"
            f"- Lower the min_score threshold in tool settings"
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