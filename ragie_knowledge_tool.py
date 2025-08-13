"""
title: Ragie Knowledge Retrieval with Citations
author: PVBLIC Foundation
author_url: https://pvblic.org
description: Retrieves relevant knowledge from Ragie API with streaming audio/video citations and chat history from Pinecone
required_open_webui_version: 0.4.0
requirements: httpx, pinecone-client, openai, pydantic, python-dotenv
version: 1.0.0
licence: MIT
"""

import os
import logging
import asyncio
import time
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
import urllib.parse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from pinecone import Pinecone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tools:
    class Valves(BaseModel):
        # Pinecone configuration
        pinecone_api_key: str = Field(
            default="", description="Pinecone API key for chat history retrieval"
        )
        pinecone_index_name: str = Field(
            default="", description="Pinecone index name for chat summaries"
        )
        
        # OpenAI configuration for embeddings
        openai_api_key: str = Field(
            default="", description="OpenAI API key for embeddings"
        )
        
        # Ragie configuration
        ragie_api_key: str = Field(
            default="", description="Ragie API key for knowledge retrieval"
        )
        ragie_partition: str = Field(
            default="default", description="Ragie partition to search in"
        )
        ragie_top_k: int = Field(
            default=5, description="Number of knowledge items to retrieve from Ragie"
        )
        ragie_score_threshold: float = Field(
            default=0.25, description="Minimum relevance score for Ragie results"
        )
        
        # Proxy configuration
        webui_base_url: str = Field(
            default="", description="Base URL for Open WebUI (for proxy URLs)"
        )

    def __init__(self):
        """Initialize the Tool with citation support disabled to use custom citations."""
        self.citation = False  # Disable built-in citations to use custom ones
        self.valves = self.Valves()
        
        # Initialize clients
        self.pinecone_client = None
        self.openai_client = None
        self.httpx_client = httpx.AsyncClient()
        
        # Load environment variables if valves are empty
        if not self.valves.pinecone_api_key:
            self.valves.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        if not self.valves.pinecone_index_name:
            self.valves.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "")
        if not self.valves.openai_api_key:
            self.valves.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.valves.ragie_api_key:
            self.valves.ragie_api_key = os.getenv("RAGIE_API_KEY", "")
        if not self.valves.webui_base_url:
            self.valves.webui_base_url = os.getenv("WEBUI_URL", "")

    async def _initialize_clients(self):
        """Initialize API clients if not already done."""
        if not self.pinecone_client and self.valves.pinecone_api_key:
            self.pinecone_client = Pinecone(api_key=self.valves.pinecone_api_key)
            
        if not self.openai_client and self.valves.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.valves.openai_api_key)

    async def _embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using OpenAI."""
        await self._initialize_clients()
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        response = await self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536
        )
        return response.data[0].embedding

    async def _retrieve_from_ragie(self, query: str, __event_emitter__=None) -> List[Dict]:
        """Retrieve knowledge from Ragie API and emit citations."""
        if not self.valves.ragie_api_key:
            logger.warning("Ragie API key not provided")
            return []

        try:
            # Prepare Ragie API request
            headers = {
                "Authorization": f"Bearer {self.valves.ragie_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "top_k": self.valves.ragie_top_k,
                "partition": self.valves.ragie_partition,
                "score_threshold": self.valves.ragie_score_threshold,
                "include_summary": True
            }

            # Make API request
            response = await self.httpx_client.post(
                "https://api.ragie.ai/retrievals",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            ragie_data = response.json()
            chunks = ragie_data.get("scored_chunks", [])
            
            knowledge_items = []
            
            for chunk in chunks:
                score = chunk.get("score", 0.0)
                if score < self.valves.ragie_score_threshold:
                    continue
                    
                text = chunk.get("text", "")
                chunk_metadata = chunk.get("metadata", {})
                document_name = chunk_metadata.get("document_name", "Unknown Document")
                source_url = chunk_metadata.get("source_url", "")
                start_time = chunk_metadata.get("start_time")
                end_time = chunk_metadata.get("end_time")
                links = chunk.get("links", {})
                
                # Emit citation for the document
                if __event_emitter__ and source_url:
                    await __event_emitter__({
                        "type": "citation",
                        "data": {
                            "document": [text],
                            "metadata": [
                                {
                                    "date_accessed": datetime.now().isoformat(),
                                    "source": document_name,
                                    "score": score
                                }
                            ],
                            "source": {
                                "name": f"📄 {document_name}",
                                "url": source_url
                            },
                        },
                    })

                # Emit citations for streaming audio/video links
                base_url = self.valves.webui_base_url.rstrip('/') if self.valves.webui_base_url else ""
                
                if links.get('self_audio_stream') and __event_emitter__:
                    ragie_audio_url = links['self_audio_stream'].get('href')
                    if ragie_audio_url:
                        if base_url:
                            proxy_audio_url = f"{base_url}/api/proxy/ragie/stream?url={urllib.parse.quote(ragie_audio_url)}"
                        else:
                            proxy_audio_url = f"/api/proxy/ragie/stream?url={urllib.parse.quote(ragie_audio_url)}"
                        
                        citation_name = f"🎵 {document_name}"
                        if start_time is not None and end_time is not None:
                            duration = end_time - start_time
                            citation_name += f" ({start_time:.1f}s-{end_time:.1f}s, {duration:.1f}s)"
                        
                        await __event_emitter__({
                            "type": "citation",
                            "data": {
                                "document": [text],
                                "metadata": [
                                    {
                                        "date_accessed": datetime.now().isoformat(),
                                        "source": document_name,
                                        "start_time": start_time,
                                        "end_time": end_time,
                                        "media_type": "audio",
                                        "score": score
                                    }
                                ],
                                "source": {
                                    "name": citation_name,
                                    "url": proxy_audio_url
                                },
                            },
                        })

                if links.get('self_video_stream') and __event_emitter__:
                    ragie_video_url = links['self_video_stream'].get('href')
                    if ragie_video_url:
                        if base_url:
                            proxy_video_url = f"{base_url}/api/proxy/ragie/stream?url={urllib.parse.quote(ragie_video_url)}"
                        else:
                            proxy_video_url = f"/api/proxy/ragie/stream?url={urllib.parse.quote(ragie_video_url)}"
                        
                        citation_name = f"🎬 {document_name}"
                        if start_time is not None and end_time is not None:
                            duration = end_time - start_time
                            citation_name += f" ({start_time:.1f}s-{end_time:.1f}s, {duration:.1f}s)"
                        
                        await __event_emitter__({
                            "type": "citation",
                            "data": {
                                "document": [text],
                                "metadata": [
                                    {
                                        "date_accessed": datetime.now().isoformat(),
                                        "source": document_name,
                                        "start_time": start_time,
                                        "end_time": end_time,
                                        "media_type": "video",
                                        "score": score
                                    }
                                ],
                                "source": {
                                    "name": citation_name,
                                    "url": proxy_video_url
                                },
                            },
                        })

                knowledge_items.append({
                    "text": text,
                    "document_name": document_name,
                    "score": score,
                    "source_url": source_url,
                    "start_time": start_time,
                    "end_time": end_time
                })

            logger.info(f"Retrieved {len(knowledge_items)} knowledge items from Ragie")
            return knowledge_items

        except Exception as e:
            logger.error(f"Ragie knowledge retrieval failed: {e}")
            return []

    async def _retrieve_chat_history(self, query_embedding: List[float], chat_id: str) -> List[Dict]:
        """Retrieve relevant chat history from Pinecone."""
        await self._initialize_clients()
        if not self.pinecone_client or not self.valves.pinecone_index_name:
            logger.warning("Pinecone not configured, skipping chat history retrieval")
            return []

        try:
            index = self.pinecone_client.Index(self.valves.pinecone_index_name)
            
            # Query for chat summaries
            filter_expr = {
                "type": "chat",
                "chat_id": chat_id
            }
            
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
                filter=filter_expr
            )
            
            chat_items = []
            for match in results.matches:
                if match.score > 0.7:  # High threshold for chat history
                    metadata = match.metadata or {}
                    chat_items.append({
                        "text": metadata.get("text", ""),
                        "score": match.score,
                        "timestamp": metadata.get("timestamp", "")
                    })
            
            logger.info(f"Retrieved {len(chat_items)} chat history items from Pinecone")
            return chat_items

        except Exception as e:
            logger.error(f"Pinecone chat history retrieval failed: {e}")
            return []

    async def retrieve_knowledge(
        self,
        query: str,
        chat_id: str = "default",
        __user__: dict = None,
        __event_emitter__=None
    ) -> str:
        """
        Retrieve relevant knowledge from Ragie and chat history from Pinecone.
        
        :param query: The search query
        :param chat_id: Chat ID for retrieving relevant chat history
        """
        try:
            # Emit status update
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "🔍 Searching knowledge base and chat history...", "done": False}
                })

            # Generate embedding for the query
            query_embedding = await self._embed_text(query)
            
            # Retrieve from both sources concurrently
            knowledge_task = self._retrieve_from_ragie(query, __event_emitter__)
            chat_task = self._retrieve_chat_history(query_embedding, chat_id)
            
            knowledge_items, chat_items = await asyncio.gather(knowledge_task, chat_task)
            
            # Emit completion status
            if __event_emitter__:
                total_items = len(knowledge_items) + len(chat_items)
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"✅ Found {total_items} relevant items ({len(knowledge_items)} knowledge, {len(chat_items)} chat history)", "done": True}
                })

            # Format results
            result_parts = []
            
            if knowledge_items:
                result_parts.append("## 📚 Relevant Knowledge")
                for item in knowledge_items[:3]:  # Limit to top 3
                    result_parts.append(f"**{item['document_name']}** (Score: {item['score']:.3f})")
                    result_parts.append(item['text'][:300] + "..." if len(item['text']) > 300 else item['text'])
                    result_parts.append("")
            
            if chat_items:
                result_parts.append("## 💬 Related Chat History")
                for item in chat_items[:2]:  # Limit to top 2
                    result_parts.append(f"**Previous Discussion** (Score: {item['score']:.3f})")
                    result_parts.append(item['text'][:200] + "..." if len(item['text']) > 200 else item['text'])
                    result_parts.append("")

            if not knowledge_items and not chat_items:
                return "No relevant knowledge or chat history found for your query."

            result_parts.append("---")
            result_parts.append("💡 **Note**: Click on the citations above to access streaming audio/video segments or full documents.")
            
            return "\n".join(result_parts)

        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {e}")
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"❌ Error: {str(e)}", "done": True}
                })
            return f"Error retrieving knowledge: {str(e)}"

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup."""
        if self.httpx_client:
            await self.httpx_client.aclose()
