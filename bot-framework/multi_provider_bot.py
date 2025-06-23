#!/usr/bin/env python3
"""
AI Bot for Open WebUI Channels
Uses Open WebUI's model selection via @ symbol.
"""

import asyncio
import aiohttp
import os
import time
import json
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
        
        # Bot will use Open WebUI's models via @ selection
        # No external provider configuration needed
        
        # Bot Configuration
        self.bot_name = os.getenv("BOT_NAME", "AI Assistant")
        self.system_prompt = os.getenv("BOT_SYSTEM_PROMPT", "You are a knowledgeable AI assistant in a chat channel. Provide comprehensive, detailed, and thorough responses. When asked for overviews or explanations, give in-depth information with multiple perspectives, examples, and context. Be informative and educational while remaining conversational and helpful.")
        self.poll_interval = int(os.getenv("BOT_POLL_INTERVAL", "2"))
        self.message_limit = int(os.getenv("BOT_MESSAGE_LIMIT", "5"))
        self.max_message_age = int(os.getenv("BOT_MAX_MESSAGE_AGE", "120"))
        self.debug = os.getenv("BOT_DEBUG", "false").lower() == "true"
        
        # Trigger Words - Optional fallback, main interaction is via @ model selection
        trigger_string = os.getenv("BOT_TRIGGERS", "")
        self.triggers = [trigger.strip().lower() for trigger in trigger_string.split(",") if trigger.strip()]
        
        # WebUI headers
        self.webui_headers = {"Authorization": f"Bearer {self.webui_token}", "Content-Type": "application/json"}
        self.seen_messages: Set[str] = set()
        
        print(f"🤖 {self.bot_name} configured:")
        print(f"   Uses Open WebUI models via @ selection")
        print(f"   Poll interval: {self.poll_interval}s")
        print(f"   System prompt loaded: {len(self.system_prompt)} characters")
        print(f"   System prompt preview: {self.system_prompt[:100]}...")
        if self.triggers:
            print(f"   Trigger words: {', '.join(self.triggers)}")
        else:
            print(f"   Interaction: @ model selection only")
        

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

    async def send_message(self, channel_id: str, content: str, selected_model: dict = None) -> bool:
        """Send a message to a channel with bot metadata"""
        async with aiohttp.ClientSession() as session:
            # Determine display name based on selected model
            if selected_model:
                model_name = selected_model.get("name", selected_model.get("id", "Unknown"))
                display_name = f"{self.bot_name} ({model_name})"
            else:
                display_name = f"{self.bot_name}"
            
            # Add metadata to identify this as a bot message
            payload = {
                "content": content,
                "data": {
                    "bot": True,
                    "bot_name": display_name,  # Show selected model in display name
                    "selected_model": selected_model.get("id") if selected_model else None
                }
            }
            url = f"{self.webui_url}/api/v1/channels/{channel_id}/messages/post"
            async with session.post(url, headers=self.webui_headers, json=payload) as response:
                return response.status == 200

    async def send_streaming_message(self, channel_id: str, content: str, selected_model: dict = None) -> bool:
        """Send a message with streaming effect by posting and then updating progressively"""
        async with aiohttp.ClientSession() as session:
            # Determine display name based on selected model
            if selected_model:
                model_name = selected_model.get("name", selected_model.get("id", "Unknown"))
                display_name = f"{self.bot_name} ({model_name})"
            else:
                display_name = f"{self.bot_name}"
            
            # First, post a "thinking" message
            thinking_payload = {
                "content": "🤔 Thinking...",
                "data": {
                    "bot": True,
                    "bot_name": display_name,
                    "selected_model": selected_model.get("id") if selected_model else None
                }
            }
            
            post_url = f"{self.webui_url}/api/v1/channels/{channel_id}/messages/post"
            async with session.post(post_url, headers=self.webui_headers, json=thinking_payload) as response:
                if response.status != 200:
                    return False
                
                message_data = await response.json()
                message_id = message_data.get("id")
                
                if not message_id:
                    return False
            
            # Wait a moment to show "thinking"
            await asyncio.sleep(1)
            
            # Now stream the content by updating the message progressively
            words = content.split()
            current_text = ""
            
            for i, word in enumerate(words):
                current_text += word + " "
                
                # Update message every 3 words or at the end
                if (i + 1) % 3 == 0 or i == len(words) - 1:
                    update_payload = {
                        "content": current_text.strip(),
                        "data": {
                            "bot": True,
                            "bot_name": display_name,
                            "selected_model": selected_model.get("id") if selected_model else None
                        }
                    }
                    
                    update_url = f"{self.webui_url}/api/v1/channels/{channel_id}/messages/{message_id}/update"
                    async with session.post(update_url, headers=self.webui_headers, json=update_payload) as update_response:
                        if update_response.status != 200 and self.debug:
                            print(f"⚠️  Failed to update message: {update_response.status}")
                    
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.15)
            
            return True

    async def send_thinking_indicator(self, channel_id: str, selected_model: dict = None) -> str:
        """Send immediate thinking indicator and return message ID for later update"""
        async with aiohttp.ClientSession() as session:
            # Determine display name based on selected model
            if selected_model:
                model_name = selected_model.get("name", selected_model.get("id", "Unknown"))
                display_name = f"{self.bot_name} ({model_name})"
            else:
                display_name = f"{self.bot_name}"
            
            # Post thinking message immediately
            thinking_payload = {
                "content": "🤔 Thinking...",
                "data": {
                    "bot": True,
                    "bot_name": display_name,
                    "selected_model": selected_model.get("id") if selected_model else None
                }
            }
            
            post_url = f"{self.webui_url}/api/v1/channels/{channel_id}/messages/post"
            async with session.post(post_url, headers=self.webui_headers, json=thinking_payload) as response:
                if response.status == 200:
                    message_data = await response.json()
                    return message_data.get("id")
                return None

    async def update_message_content(self, channel_id: str, message_id: str, content: str, selected_model: dict = None) -> bool:
        """Update a specific message with new content"""
        async with aiohttp.ClientSession() as session:
            # Determine display name based on selected model
            if selected_model:
                model_name = selected_model.get("name", selected_model.get("id", "Unknown"))
                display_name = f"{self.bot_name} ({model_name})"
            else:
                display_name = f"{self.bot_name}"
            
            update_payload = {
                "content": content,
                "data": {
                    "bot": True,
                    "bot_name": display_name,
                    "selected_model": selected_model.get("id") if selected_model else None
                }
            }
            
            update_url = f"{self.webui_url}/api/v1/channels/{channel_id}/messages/{message_id}/update"
            async with session.post(update_url, headers=self.webui_headers, json=update_payload) as update_response:
                return update_response.status == 200

    async def update_thinking_with_response(self, channel_id: str, content: str, selected_model: dict = None) -> bool:
        """Update the thinking message with streaming response"""
        # Get the message ID from the last thinking message
        message_id = getattr(self, '_last_thinking_message_id', None)
        if not message_id:
            # Fallback to regular message
            return await self.send_message(channel_id, content, selected_model)
        
        async with aiohttp.ClientSession() as session:
            # Determine display name based on selected model
            if selected_model:
                model_name = selected_model.get("name", selected_model.get("id", "Unknown"))
                display_name = f"{self.bot_name} ({model_name})"
            else:
                display_name = f"{self.bot_name}"
            
            # Stream the content by updating the message progressively
            words = content.split()
            current_text = ""
            
            for i, word in enumerate(words):
                current_text += word + " "
                
                # Update message every 3 words or at the end
                if (i + 1) % 3 == 0 or i == len(words) - 1:
                    update_payload = {
                        "content": current_text.strip(),
                        "data": {
                            "bot": True,
                            "bot_name": display_name,
                            "selected_model": selected_model.get("id") if selected_model else None
                        }
                    }
                    
                    update_url = f"{self.webui_url}/api/v1/channels/{channel_id}/messages/{message_id}/update"
                    async with session.post(update_url, headers=self.webui_headers, json=update_payload) as update_response:
                        if update_response.status != 200 and self.debug:
                            print(f"⚠️  Failed to update message: {update_response.status}")
                    
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.1)
            
            return True

    async def get_rag_response(self, message: str, files: List[dict], model_id: str = None, channel_id: str = None, selected_model: dict = None) -> str:
        """Get RAG-enhanced AI response using OpenWebUI's chat completions endpoint with files"""
        try:
            if self.debug:
                print(f"🔍 DEBUG: get_rag_response called with {len(files)} files")
                print(f"🔍 DEBUG: Files structure: {json.dumps(files, indent=2)}")
            
            headers = self.webui_headers
            url = f"{self.webui_url}/api/chat/completions"
            
            # Use EXACT same payload structure as main chat interface for RAG
            payload = {
                "stream": True,  # Enable streaming for better UX
                "model": model_id,  # Use the selected model
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message}
                ],
                "files": files,  # Use files exactly as received from channel
                "session_id": f"bot-session-{int(time.time())}",
                "chat_id": f"bot-chat-{int(time.time())}",
                "id": f"bot-message-{int(time.time())}"
            }
            
            if self.debug:
                print(f"🔍 DEBUG: Making RAG request to {url}")
                print(f"🔍 DEBUG: Payload: {json.dumps(payload, indent=2)}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if self.debug:
                        print(f"🔍 DEBUG: RAG response status: {response.status}")
                    
                    if response.status == 200:
                        # Handle streaming response with real-time updates
                        content = ""
                        message_id = getattr(self, '_last_thinking_message_id', None)
                        update_counter = 0  # Add counter to throttle updates
                        
                        async for line in response.content:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # Remove 'data: ' prefix
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            content += delta["content"]
                                            update_counter += 1
                                            
                                            # Only update every 10 tokens AND add delay to prevent spam
                                            if channel_id and message_id and update_counter % 10 == 0:
                                                await self.update_message_content(channel_id, message_id, content, selected_model)
                                                await asyncio.sleep(0.1)  # Small delay to prevent API spam
                                                
                                except json.JSONDecodeError:
                                    continue
                        
                        # Final update with complete content
                        if channel_id and message_id and content:
                            await self.update_message_content(channel_id, message_id, content, selected_model)
                        
                        if self.debug:
                            print(f"🔍 DEBUG: RAG response length: {len(content)}")
                        return content if content else "I apologize, but I couldn't process the knowledge base files properly."
                    else:
                        error_text = await response.text()
                        print(f"❌ RAG API Error {response.status}: {error_text}")
                        return f"I encountered an error while processing the knowledge base files. Please try again."
        except Exception as e:
            print(f"❌ Error in get_rag_response: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return "I apologize, but I encountered an error while processing the knowledge base files. Please try again."

    async def get_ai_response(self, message: str, model_id: str, channel_id: str = None, selected_model: dict = None) -> str:
        """Get AI response from Open WebUI using the selected model"""
        try:
            headers = self.webui_headers
            url = f"{self.webui_url}/api/chat/completions"
            
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message}
                ],
                "stream": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        # Handle streaming response with real-time updates
                        content = ""
                        message_id = getattr(self, '_last_thinking_message_id', None)
                        update_counter = 0  # Add counter to throttle updates
                        
                        async for line in response.content:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # Remove 'data: ' prefix
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            content += delta["content"]
                                            update_counter += 1
                                            
                                            # Only update every 10 tokens AND add delay to prevent spam
                                            if channel_id and message_id and update_counter % 10 == 0:
                                                await self.update_message_content(channel_id, message_id, content, selected_model)
                                                await asyncio.sleep(0.1)  # Small delay to prevent API spam
                                                
                                except json.JSONDecodeError:
                                    continue
                        
                        # Final update with complete content
                        if channel_id and message_id and content:
                            await self.update_message_content(channel_id, message_id, content, selected_model)
                        
                        return content if content else "I apologize, but I couldn't generate a proper response."
                    else:
                        error_text = await response.text()
                        print(f"❌ API Error {response.status}: {error_text}")
                        return f"I encountered an error while processing your request. Please try again."
        except Exception as e:
            print(f"❌ Error in get_ai_response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def _is_bot_message(self, message_data: dict) -> bool:
        """Check if a message was sent by this bot using metadata"""
        return message_data.get("data", {}).get("bot") is True
    
    def should_respond(self, content: str, message_data: dict = None) -> bool:
        """Check if bot should respond to a message"""
        # Primary method: Check if user selected a model via @ symbol
        if message_data and message_data.get("data", {}).get("atSelectedModel"):
            return True
        
        # Fallback to trigger words (if any are configured)
        if self.triggers:
            content_lower = content.lower()
            return any(trigger in content_lower for trigger in self.triggers)
        
        # If no triggers configured, don't respond to avoid spam
        return False
    
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
            content = message["content"]
            # Extract knowledge files from message data (same as chat)
            files = message.get("data", {}).get("files", [])
            
            if self.debug:
                print(f"🔍 DEBUG: Message files detected: {len(files)} files")
            
            # Skip if already processed
            if msg_id in self.seen_messages:
                continue
                
            # Mark as seen
            self.seen_messages.add(msg_id)
            
            # Skip bot's own messages (identify by metadata)
            if self._is_bot_message(message):
                if self.debug:
                    print(f"🤖 Skipping bot message: {content[:50]}...")
                continue
            
            # Skip old messages
            msg_time = message["created_at"] / 1_000_000_000
            if time.time() - msg_time > self.max_message_age:
                continue
            
            kb_indicator = f" 📚({len(files)} KB files)" if files else ""
            print(f"📝 [{channel_name}] {user_name}: {content}{kb_indicator}")
            
            # Check if we should respond
            if self.should_respond(content, message):
                # Check if user selected a specific model via @ symbol
                selected_model = message.get("data", {}).get("atSelectedModel")
                if selected_model:
                    model_name = selected_model.get("name", selected_model.get("id", "Unknown"))
                    model_id = selected_model.get("id")
                    print(f"🎯 User selected model: {model_name}")
                    model_info = f" ({model_name})"
                else:
                    # This shouldn't happen with @ selection only, but just in case
                    print("⚠️  No model selected - this shouldn't happen with @ selection")
                    continue
                
                print(f"🤖 {self.bot_name}{model_info} is thinking...")
                
                # Show thinking indicator immediately
                self._last_thinking_message_id = await self.send_thinking_indicator(channel_id, selected_model)
                
                # If files are detected, use RAG processing
                if files and len(files) > 0:
                    print(f"📚 Processing {len(files)} knowledge base files with RAG")
                    ai_response = await self.get_rag_response(content, files, model_id, channel_id, selected_model)
                else:
                    # No files, use regular AI response with selected model
                    ai_response = await self.get_ai_response(content, model_id, channel_id, selected_model)
                
                # Response is already streamed in real-time, just log completion
                display_name = f"{model_name}" if selected_model else "Open WebUI"
                print(f"✅ {self.bot_name} ({display_name}): {ai_response[:100]}{'...' if len(ai_response) > 100 else ''}")
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
    
    async def run(self):
        """Main bot loop"""
        print(f"🚀 Starting {self.bot_name} with Open WebUI models...")
        
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
        
        if self.triggers:
            print(f"\n🎯 {self.bot_name} is ready! Send a message containing trigger words:")
            print(f"   {', '.join(self.triggers)}")
        else:
            print(f"\n🎯 {self.bot_name} is ready! Type @ to select a model and send your message.")
        print()
        
        # Main loop
        while True:
            try:
                # Process each channel
                for channel in channels:
                    await self.process_channel(channel)
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                if not self.debug:
                    print(".", end="", flush=True)
                
            except KeyboardInterrupt:
                print(f"\n👋 {self.bot_name} is shutting down...")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("🔄 Continuing in 5 seconds...")
                await asyncio.sleep(5)

def main():
    """Main entry point for standalone execution"""
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