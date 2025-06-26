"""
Utility functions for the bot framework
Consolidates HTTP client logic and streaming utilities
"""
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, AsyncGenerator
from env import WEBUI_URL, TOKEN


class WebUIClient:
    """Consolidated HTTP client for Open WebUI API interactions"""
    
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def _session(self):
        """Get aiohttp session for direct use"""
        return aiohttp.ClientSession()
    
    async def get_channels(self) -> List[Dict]:
        """Get all accessible channels"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/channels/", 
                headers=self.headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                return []
    
    async def get_messages(self, channel_id: str, limit: int = 50) -> List[Dict]:
        """Get recent messages from a channel"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/channels/{channel_id}/messages?limit={limit}"
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                return []
    
    async def post_message(self, channel_id: str, content: str, data: Dict = None) -> Optional[Dict]:
        """Post a message to a channel"""
        payload = {"content": content}
        if data:
            payload["data"] = data
            
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/channels/{channel_id}/messages/post"
            async with session.post(url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                return None
    
    async def update_message(self, channel_id: str, message_id: str, content: str, data: Dict = None) -> bool:
        """Update an existing message"""
        payload = {"content": content}
        if data:
            payload["data"] = data
            
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/channels/{channel_id}/messages/{message_id}/update"
            async with session.post(url, headers=self.headers, json=payload) as response:
                return response.status == 200
    
    async def chat_completions(self, payload: Dict) -> Dict:
        """Make a chat completion request (for RAG and non-streaming)"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/chat/completions"
            async with session.post(url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=error_text,
                        headers=response.headers,
                    )
    
    async def stream_chat_completion(
        self, 
        model_id: str, 
        messages: List[Dict], 
        files: List[Dict] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion using existing Open WebUI infrastructure"""
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": True
        }
        
        if files:
            payload["files"] = files
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/chat/completions"
            async with session.post(url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue


class StreamingMessageUpdater:
    """Handles real-time message updates during streaming"""
    
    def __init__(self, client: WebUIClient, channel_id: str, message_id: str):
        self.client = client
        self.channel_id = channel_id
        self.message_id = message_id
        self.content = ""
        self.last_update = 0
        self.update_interval = 0.1  # Update every 100ms
    
    async def append_content(self, chunk: str):
        """Append content and update message if enough time has passed"""
        self.content += chunk
        current_time = asyncio.get_event_loop().time()
        
        if current_time - self.last_update >= self.update_interval:
            await self.client.update_message(
                self.channel_id, 
                self.message_id, 
                self.content
            )
            self.last_update = current_time
    
    async def finalize(self):
        """Send final update"""
        await self.client.update_message(
            self.channel_id, 
            self.message_id, 
            self.content
        )


def should_respond_to_message(message_data: Dict, triggers: List[str] = None) -> bool:
    """Check if bot should respond to a message"""
    content = message_data.get("content", "").strip()
    
    # Primary: Check for @ model selection
    if message_data.get("data", {}).get("atSelectedModel"):
        return True
    
    # Fallback: Check trigger words (if any are configured)
    if triggers:
        content_lower = content.lower()
        return any(trigger in content_lower for trigger in triggers)
    
    # If no triggers configured, don't respond to avoid spam
    return False


def extract_bot_metadata(message_data: Dict) -> Dict:
    """Extract bot-specific metadata from message"""
    data = message_data.get("data", {})
    return {
        "is_bot": data.get("bot", False),
        "selected_model": data.get("atSelectedModel"),
        "files": data.get("files", [])
    } 