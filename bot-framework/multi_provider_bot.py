#!/usr/bin/env python3
"""
Multi-Provider AI Bot for Open WebUI Channels
Supports OpenWebUI, OpenAI, Anthropic, and Ollama providers.
"""

import asyncio
import aiohttp
import os
import time
from typing import List, Set, Optional
from dotenv import load_dotenv

class MultiProviderAIBot:
    def __init__(self, env_file=".env"):
        # Load environment variables
        load_dotenv(env_file)
        
        # Open WebUI Configuration
        self.webui_url = os.getenv("WEBUI_URL", "http://localhost:8080")
        self.webui_token = os.getenv("TOKEN")
        if not self.webui_token:
            raise ValueError("TOKEN environment variable is required for Open WebUI access")
        
        # AI Provider Configuration
        self.ai_provider = os.getenv("AI_PROVIDER", "openwebui").lower()
        self.ai_model = os.getenv("AI_MODEL", "llama3.2:latest")
        self.ai_api_key = os.getenv("AI_API_KEY", "")
        self.ai_base_url = os.getenv("AI_BASE_URL", "")
        
        # Bot Configuration
        self.bot_name = os.getenv("BOT_NAME", "AI Assistant")
        self.system_prompt = os.getenv("BOT_SYSTEM_PROMPT", "You are a helpful AI assistant in a chat channel. Keep responses concise and friendly.")
        self.poll_interval = int(os.getenv("BOT_POLL_INTERVAL", "2"))
        self.message_limit = int(os.getenv("BOT_MESSAGE_LIMIT", "5"))
        self.max_message_age = int(os.getenv("BOT_MAX_MESSAGE_AGE", "120"))
        
        # Trigger Words
        trigger_string = os.getenv("BOT_TRIGGERS", "bot,ai,assistant,?,hello,hi,hey,help,what,how,why")
        self.triggers = [trigger.strip().lower() for trigger in trigger_string.split(",")]
        
        # WebUI headers
        self.webui_headers = {"Authorization": f"Bearer {self.webui_token}", "Content-Type": "application/json"}
        self.seen_messages: Set[str] = set()
        self.bot_user_id = None  # Will be set during initialization
        
        # Validate provider configuration
        self._validate_provider_config()
        
        print(f"🤖 {self.bot_name} configured:")
        print(f"   Provider: {self.ai_provider}")
        print(f"   Model: {self.ai_model}")
        print(f"   Poll interval: {self.poll_interval}s")
        print(f"   Triggers: {', '.join(self.triggers)}")
        
    def _validate_provider_config(self):
        """Validate provider-specific configuration"""
        if self.ai_provider == "openai" and not self.ai_api_key:
            raise ValueError("AI_API_KEY is required for OpenAI provider")
        elif self.ai_provider == "anthropic" and not self.ai_api_key:
            raise ValueError("AI_API_KEY is required for Anthropic provider")
        elif self.ai_provider == "ollama" and not self.ai_base_url:
            self.ai_base_url = "http://localhost:11434"  # Default Ollama URL
    
    def _get_ai_headers(self) -> dict:
        """Get headers for AI provider"""
        if self.ai_provider == "openai":
            return {
                "Authorization": f"Bearer {self.ai_api_key}",
                "Content-Type": "application/json"
            }
        elif self.ai_provider == "anthropic":
            return {
                "x-api-key": self.ai_api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
        elif self.ai_provider == "ollama":
            return {"Content-Type": "application/json"}
        else:  # openwebui
            return self.webui_headers
    
    def _get_ai_url(self) -> str:
        """Get API URL for AI provider"""
        if self.ai_provider == "openai":
            return self.ai_base_url or "https://api.openai.com/v1/chat/completions"
        elif self.ai_provider == "anthropic":
            return self.ai_base_url or "https://api.anthropic.com/v1/messages"
        elif self.ai_provider == "ollama":
            return f"{self.ai_base_url}/api/chat"
        else:  # openwebui
            return f"{self.webui_url}/api/chat/completions"
    
    def _format_ai_payload(self, message: str) -> dict:
        """Format payload for AI provider"""
        if self.ai_provider == "anthropic":
            return {
                "model": self.ai_model,
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": f"{self.system_prompt}\n\nUser: {message}"}
                ]
            }
        else:  # openai, ollama, openwebui
            return {
                "model": self.ai_model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message}
                ],
                "stream": False
            }
    
    def _extract_ai_response(self, response_data: dict) -> Optional[str]:
        """Extract response text from AI provider response"""
        if self.ai_provider == "anthropic":
            if "content" in response_data and response_data["content"]:
                return response_data["content"][0].get("text", "")
        else:  # openai, ollama, openwebui
            if "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"]
            elif "message" in response_data:
                return response_data["message"]["content"]
        return None
    
    async def get_bot_user_id(self) -> Optional[str]:
        """Get the bot's user ID from Open WebUI"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.webui_url}/api/v1/auths/", headers=self.webui_headers) as response:
                if response.status == 200:
                    user_data = await response.json()
                    return user_data.get("id")
                return None

    async def get_channels(self) -> List[dict]:
        """Get all accessible channels from Open WebUI"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.webui_url}/api/v1/channels/", headers=self.webui_headers) as response:
                if response.status == 200:
                    return await response.json()
                print(f"❌ Failed to get channels: {response.status}")
                return []
    
    async def get_messages(self, channel_id: str) -> List[dict]:
        """Get recent messages from a channel"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.webui_url}/api/v1/channels/{channel_id}/messages?limit={self.message_limit}"
            async with session.get(url, headers=self.webui_headers) as response:
                if response.status == 200:
                    return await response.json()
                return []
    
    async def send_message(self, channel_id: str, content: str) -> bool:
        """Send a message to a channel"""
        async with aiohttp.ClientSession() as session:
            # Add metadata to identify this as a bot message
            payload = {
                "content": content,
                "data": {
                    "bot": True,
                    "ai_provider": self.ai_provider,
                    "ai_model": self.ai_model,
                    "bot_name": self.bot_name
                }
            }
            print(f"🔧 Sending payload: {payload}")
            url = f"{self.webui_url}/api/v1/channels/{channel_id}/messages/post"
            async with session.post(url, headers=self.webui_headers, json=payload) as response:
                response_text = await response.text()
                print(f"🔧 API Response ({response.status}): {response_text}")
                return response.status == 200
    
    async def get_ai_response(self, message: str) -> str:
        """Get AI response from configured provider"""
        try:
            headers = self._get_ai_headers()
            url = self._get_ai_url()
            payload = self._format_ai_payload(message)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_text = self._extract_ai_response(result)
                        if ai_text:
                            return ai_text
                        else:
                            print(f"⚠️  Unexpected response format from {self.ai_provider}")
                    else:
                        error_text = await response.text()
                        print(f"❌ {self.ai_provider} API error {response.status}: {error_text}")
            
            # Fallback response
            return f"Hi! I'm {self.bot_name}. I'm having trouble connecting to {self.ai_provider} right now, but I'm here to help!"
            
        except Exception as e:
            print(f"❌ Error getting AI response: {e}")
            return f"Hello! I'm {self.bot_name}. I encountered an error, but I'm still here to chat!"
    
    def _is_bot_message(self, message_data: dict) -> bool:
        """Check if a message was sent by this bot using metadata"""
        return message_data.get("data", {}).get("bot") is True
    
    def should_respond(self, content: str) -> bool:
        """Check if bot should respond to a message"""
        content_lower = content.lower()
        return any(trigger in content_lower for trigger in self.triggers)
    
    def has_bot_access(self, channel: dict) -> bool:
        """Check if bot access is enabled for this channel"""
        data = channel.get("data")
        if data is None:
            return False
        return data.get("bot_access", False)
    
    async def process_channel(self, channel: dict):
        """Process messages in a single channel"""
        channel_id = channel["id"]
        channel_name = channel["name"]
        
        # Check if bot access is enabled for this channel
        if not self.has_bot_access(channel):
            return  # Skip this channel if bot access is disabled
        
        messages = await self.get_messages(channel_id)
        
        for message in reversed(messages):  # Process oldest first
            msg_id = message["id"]
            user_name = message["user"]["name"]
            user_id = message["user"]["id"]
            content = message["content"]
            
            # Skip if already processed
            if msg_id in self.seen_messages:
                continue
                
            # Mark as seen
            self.seen_messages.add(msg_id)
            
            # Skip bot's own messages (identify by metadata)
            if self._is_bot_message(message):
                print(f"🤖 Skipping bot message: {content[:50]}...")
                continue
            
            # Skip old messages
            msg_time = message["created_at"] / 1_000_000_000
            if time.time() - msg_time > self.max_message_age:
                continue
            
            print(f"📝 [{channel_name}] {user_name}: {content}")
            
            # Check if we should respond
            if self.should_respond(content):
                print(f"🤖 {self.bot_name} ({self.ai_provider}) is thinking...")
                
                # Generate AI response
                ai_response = await self.get_ai_response(content)
                
                # Send response
                if await self.send_message(channel_id, ai_response):
                    print(f"✅ {self.bot_name}: {ai_response[:100]}{'...' if len(ai_response) > 100 else ''}")
                else:
                    print(f"❌ Failed to send response to {channel_name}")
    
    async def run(self):
        """Main bot loop"""
        print(f"🚀 Starting {self.bot_name} with {self.ai_provider} provider...")
        
        # Bot will identify its own messages by content patterns
        
        # Get channels
        channels = await self.get_channels()
        if not channels:
            print("❌ No accessible channels found")
            return
            
        # Filter channels with bot access
        bot_enabled_channels = [ch for ch in channels if self.has_bot_access(ch)]
        
        print(f"✅ Found {len(channels)} channels, {len(bot_enabled_channels)} with bot access enabled:")
        for channel in channels:
            bot_status = "🤖" if self.has_bot_access(channel) else "🚫"
            print(f"   {bot_status} {channel['name']}")
        
        if not bot_enabled_channels:
            print("\n⚠️  No channels have bot access enabled!")
            print("   Enable bot access in channel settings to allow the bot to respond.")
            return
        
        print(f"\n🎯 {self.bot_name} is ready! Send a message containing any of these words:")
        print(f"   {', '.join(self.triggers)}")
        print()
        
        # Main loop
        while True:
            try:
                # Process each channel
                for channel in channels:
                    await self.process_channel(channel)
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                print(".", end="", flush=True)
                
            except KeyboardInterrupt:
                print(f"\n👋 {self.bot_name} is shutting down...")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("🔄 Continuing in 5 seconds...")
                await asyncio.sleep(5)

def main():
    """Main entry point"""
    try:
        bot = MultiProviderAIBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 