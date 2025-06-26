#!/usr/bin/env python3
"""
AI Bot for Open WebUI Channels
Uses Open WebUI's model selection via @ symbol.
"""

import asyncio
import os
import time
import json
import re
import logging
from pathlib import Path
from typing import List, Set, Optional
from dotenv import load_dotenv
import html

from utils import WebUIClient, should_respond_to_message, extract_bot_metadata

# Configure logging to write to backend/data/logs/
def setup_logging():
    """Configure logging to write to backend/data/logs/ directory"""
    # Get the project root (parent of bot-framework)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    logs_dir = project_root / "backend" / "data" / "logs"
    
    # Ensure logs directory exists
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path
    log_file = logs_dir / "channel_bot.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger('MultiProviderAIBot')

# Initialize logger
logger = setup_logging()


class MultiProviderAIBot:
    def __init__(self, env_file=".env"):
        # Load environment variables
        load_dotenv(env_file)
        
        # Open WebUI Configuration
        self.webui_url = os.getenv("WEBUI_URL", "http://localhost:8080")
        self.webui_token = os.getenv("TOKEN")
        if not self.webui_token:
            raise ValueError("TOKEN environment variable is required for Open WebUI access")
        
        # Initialize HTTP client
        self.client = WebUIClient(self.webui_url, self.webui_token)
        
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
        
        # Message tracking
        self.seen_messages: Set[str] = set()
        
        logger.info(f"🤖 {self.bot_name} configured:")
        logger.info(f"   Uses Open WebUI models via @ selection")
        logger.info(f"   Poll interval: {self.poll_interval}s")
        logger.info(f"   System prompt loaded: {len(self.system_prompt)} characters")
        logger.info(f"   System prompt preview: {self.system_prompt[:100]}...")
        if self.triggers:
            logger.info(f"   Trigger words: {', '.join(self.triggers)}")
        else:
            logger.info(f"   Interaction: @ model selection only")

    def _create_bot_payload(self, content: str, selected_model: dict = None) -> dict:
        """Create standardized bot message payload"""
        if selected_model:
            model_name = selected_model.get("name", selected_model.get("id", "Unknown"))
            display_name = f"{self.bot_name} ({model_name})"
        else:
            display_name = self.bot_name
        
        return {
            "content": content,
            "data": {
                "bot": True,
                "bot_name": display_name,
                "selected_model": selected_model.get("id") if selected_model else None
            }
        }

    async def send_message(self, channel_id: str, content: str, selected_model: dict = None) -> bool:
        """Send a message to a channel with bot metadata"""
        payload = self._create_bot_payload(content, selected_model)
        result = await self.client.post_message(channel_id, content, payload["data"])
        return result is not None

    async def send_thinking_indicator(self, channel_id: str, selected_model: dict = None) -> Optional[str]:
        """Send immediate thinking indicator and return message ID for later update"""
        payload = self._create_bot_payload("🤔 Thinking...", selected_model)
        result = await self.client.post_message(channel_id, "🤔 Thinking...", payload["data"])
        return result.get("id") if result else None

    async def update_message_content(self, channel_id: str, message_id: str, content: str, selected_model: dict = None) -> bool:
        """Update a specific message with new content"""
        payload = self._create_bot_payload(content, selected_model)
        return await self.client.update_message(channel_id, message_id, content, payload["data"])

    async def get_channels(self) -> List[dict]:
        """Get all accessible channels from Open WebUI"""
        return await self.client.get_channels()
    
    async def get_messages(self, channel_id: str) -> List[dict]:
        """Get recent messages from a channel"""
        return await self.client.get_messages(channel_id, self.message_limit)

    def format_llm_response(self, content: str) -> str:
        """Unified formatting for all LLM responses (user or RAG)."""
        if not content:
            return content
        # Unescape HTML entities (if any)
        content = html.unescape(content)
        # Ensure blank line before headings
        content = re.sub(r'(?<!\n)\n?(#+ )', r'\n\n\1', content)
        # Ensure blank line before lists
        content = re.sub(r'(?<!\n)\n?([-*] )', r'\n\n\1', content)
        content = re.sub(r'(?<!\n)\n?(\d+\. )', r'\n\n\1', content)
        # Replace 3+ newlines with just 2
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Remove accidental double spaces
        content = re.sub(r' {2,}', ' ', content)
        return content.strip()

    def _build_chat_payload(self, message: str, model_id: str, files: List[dict] = None) -> dict:
        """Build standardized chat completion payload for both RAG and AI responses"""
        # Build messages like chat interface
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # Filter and process messages like chat interface
        messages = [msg for msg in messages if msg]
        processed_messages = []
        for msg in messages:
            processed_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Final filter like chat interface
        messages = [msg for msg in processed_messages if msg.get("role") == "user" or msg.get("content", "").strip()]
        
        # Build payload structure matching chat interface
        stream = True  # Chat interface default
        payload = {
            "stream": stream,
            "model": model_id,
            "messages": messages,
            "params": {},  # Empty like chat when no settings
            "files": files if files and len(files) > 0 else None,
            "filter_ids": None,  # Like chat when selectedFilterIds empty
            "tool_ids": None,    # Like chat when selectedToolIds empty  
            "tool_servers": [],  # Like chat $toolServers when empty
            "features": {
                "image_generation": False,  # Like chat when disabled
                "code_interpreter": False,
                "web_search": False, 
                "memory": False  # Like chat when $settings?.memory is false
            },
            "variables": {},  # Like chat when getPromptVariables returns empty
            "model_item": None,  # Like chat when $models.find returns undefined
            "session_id": f"bot-session-{int(time.time())}",
            "chat_id": f"bot-chat-{int(time.time())}",
            "id": f"bot-message-{int(time.time())}",
            "background_tasks": {
                "follow_up_generation": False  # Like chat when $settings?.autoFollowUps is false
            }
        }
        
        # Add stream options if streaming
        if stream:
            payload["stream_options"] = {"include_usage": True}
        
        # Remove None values like chat interface
        final_payload = {}
        for k, v in payload.items():
            if v is not None:
                final_payload[k] = v
        
        return final_payload

    async def get_rag_response(self, message: str, files: List[dict], model_id: str = None, channel_id: str = None, selected_model: dict = None) -> str:
        """Get RAG-enhanced AI response using OpenWebUI's chat completions endpoint with files"""
        try:
            if self.debug:
                logger.info(f"🔍 DEBUG: get_rag_response called with {len(files)} files")
            
            # Format files for RAG processing
            formatted_files = []
            for file in files:
                if self.debug:
                    logger.info(f"🔍 DEBUG: Processing file: {json.dumps(file, indent=2)}")
                
                # Handle different file types
                if file.get("type") == "collection":
                    # Knowledge base collection
                    formatted_file = {
                        "id": file.get("id"),
                        "name": file.get("name", "Knowledge Base"),
                        "type": "collection"
                    }
                    if file.get("legacy"):
                        formatted_file["legacy"] = True
                        formatted_file["collection_names"] = file.get("collection_names", [])
                    formatted_files.append(formatted_file)
                elif file.get("collection_name"):
                    # Legacy collection
                    formatted_file = {
                        "id": file.get("collection_name"),
                        "name": file.get("name", "Legacy Collection"),
                        "legacy": True
                    }
                    formatted_files.append(formatted_file)
                elif file.get("id"):
                    # Individual file
                    formatted_file = {
                        "id": file.get("id"),
                        "name": file.get("name", "File"),
                        "type": "file"
                    }
                    if file.get("legacy"):
                        formatted_file["legacy"] = True
                    formatted_files.append(formatted_file)
                else:
                    # Unknown format, try to use as-is
                    if self.debug:
                        logger.warning(f"⚠️ Unknown file format, using as-is: {file}")
                    formatted_files.append(file)
            
            if self.debug:
                logger.info(f"🔍 DEBUG: Formatted files: {json.dumps(formatted_files, indent=2)}")
            
            # Build payload using shared method
            payload = self._build_chat_payload(message, model_id, formatted_files)
            
            if self.debug:
                logger.info("DEBUG: Sending RAG payload to backend:", repr(payload))
            
            # Handle streaming response
            if payload.get("stream"):
                # Use streaming - accumulate content
                content = ""
                async for chunk in self.client.stream_chat_completion(
                    model_id, 
                    payload["messages"], 
                    payload.get("files")
                ):
                    content += chunk
                    # Update message progressively
                    if channel_id and self._last_thinking_message_id:
                        await self.update_message_content(channel_id, self._last_thinking_message_id, content, selected_model)
                
                if self.debug:
                    logger.info("DEBUG: Final streaming content:", repr(content))
                
                return content
            else:
                # Non-streaming fallback
                response_data = await self.client.chat_completions(payload)
                
                if self.debug:
                    logger.info(f"🔍 DEBUG: Response data: {json.dumps(response_data, indent=2)}")
                
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0]["message"]["content"]
                    if channel_id and self._last_thinking_message_id:
                        await self.update_message_content(channel_id, self._last_thinking_message_id, content, selected_model)
                    return content
                elif response_data.get("status") and response_data.get("task_id"):
                    # Task-based response handling
                    task_id = response_data["task_id"]
                    if self.debug:
                        logger.info(f"🔍 DEBUG: Task-based response, task_id: {task_id}")
                    
                    content = "I'm processing your request. Please check back in a moment for the complete response."
                    
                    if channel_id and self._last_thinking_message_id:
                        await self.update_message_content(channel_id, self._last_thinking_message_id, content, selected_model)
                    
                    return content
                else:
                    logger.error(f"❌ Unexpected response format: {response_data}")
                    return "I encountered an unexpected response format. Please try again."
        except Exception as e:
            logger.error(f"❌ Error in get_rag_response: {str(e)}")
            error_msg = "I apologize, but I encountered an error while processing your files. Please try again."
            
            # Update the thinking indicator with the error message
            if channel_id and self._last_thinking_message_id:
                await self.update_message_content(channel_id, self._last_thinking_message_id, error_msg, selected_model)
            
            return error_msg

    async def get_ai_response(self, message: str, model_id: str, channel_id: str = None, selected_model: dict = None) -> str:
        """Get AI response using OpenWebUI's chat completions endpoint"""
        try:
            # Build payload using shared method (no files for regular AI chat)
            payload = self._build_chat_payload(message, model_id, None)
            
            if self.debug:
                logger.info(f"🔍 DEBUG: Making AI request to {self.webui_url}/api/chat/completions")
                logger.info(f"🔍 DEBUG: Payload: {json.dumps(payload, indent=2)}")
            
            # Handle streaming response
            if payload.get("stream"):
                # Use streaming - accumulate content
                content = ""
                async for chunk in self.client.stream_chat_completion(
                    model_id, 
                    payload["messages"], 
                    payload.get("files")
                ):
                    content += chunk
                    # Update message progressively
                    if channel_id and self._last_thinking_message_id:
                        await self.update_message_content(channel_id, self._last_thinking_message_id, content, selected_model)
                
                if self.debug:
                    logger.info("DEBUG: Final streaming content:", repr(content))
                
                return content
            else:
                # Non-streaming fallback
                response_data = await self.client.chat_completions(payload)
                
                if self.debug:
                    logger.info(f"🔍 DEBUG: Response data: {json.dumps(response_data, indent=2)}")
                
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0]["message"]["content"]
                    if channel_id and self._last_thinking_message_id:
                        await self.update_message_content(channel_id, self._last_thinking_message_id, content, selected_model)
                    return content
                elif response_data.get("status") and response_data.get("task_id"):
                    # Task-based response handling
                    task_id = response_data["task_id"]
                    if self.debug:
                        logger.info(f"🔍 DEBUG: Task-based response, task_id: {task_id}")
                    
                    content = "I'm processing your request. Please check back in a moment for the complete response."
                    
                    if channel_id and self._last_thinking_message_id:
                        await self.update_message_content(channel_id, self._last_thinking_message_id, content, selected_model)
                    
                    return content
                else:
                    logger.error(f"❌ Unexpected response format: {response_data}")
                    return "I encountered an unexpected response format. Please try again."
                
        except Exception as e:
            logger.error(f"❌ Error in get_ai_response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def _is_bot_message(self, message_data: dict) -> bool:
        """Check if a message was sent by this bot using metadata"""
        return message_data.get("data", {}).get("bot") is True
    
    def should_respond(self, content: str, message_data: dict = None) -> bool:
        """Check if bot should respond to a message"""
        return should_respond_to_message(message_data or {}, self.triggers)
    
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
                logger.info(f"🔍 DEBUG: Message files detected: {len(files)} files")
            
            # Skip if already processed
            if msg_id in self.seen_messages:
                continue
                
            # Mark as seen
            self.seen_messages.add(msg_id)
            
            # Skip bot's own messages (identify by metadata)
            if self._is_bot_message(message):
                if self.debug:
                    logger.info(f"🤖 Skipping bot message: {content[:50]}...")
                continue
            
            # Skip old messages
            msg_time = message["created_at"] / 1_000_000_000
            if time.time() - msg_time > self.max_message_age:
                continue
            
            kb_indicator = f" 📚({len(files)} KB files)" if files else ""
            logger.info(f"📝 [{channel_name}] {user_name}: {content}{kb_indicator}")
            
            # Check if we should respond
            if self.should_respond(content, message):
                processed_content = content.strip()
                
                # Check if user selected a specific model via @ symbol
                selected_model = message.get("data", {}).get("atSelectedModel")
                if selected_model:
                    model_name = selected_model.get("name", selected_model.get("id", "Unknown"))
                    model_id = selected_model.get("id")
                    logger.info(f"🎯 User selected model: {model_name}")
                    model_info = f" ({model_name})"
                else:
                    # This shouldn't happen with @ selection only, but just in case
                    logger.warning("⚠️  No model selected - this shouldn't happen with @ selection")
                    continue
                
                logger.info(f"🤖 {self.bot_name}{model_info} is thinking...")
                
                # Show thinking indicator immediately
                self._last_thinking_message_id = await self.send_thinking_indicator(channel_id, selected_model)
                if self.debug:
                    logger.info(f"🔍 DEBUG: Thinking message ID: {self._last_thinking_message_id}")
                
                # If files are detected, use RAG processing
                if files and len(files) > 0:
                    logger.info(f"📚 Processing {len(files)} knowledge base files with RAG")
                    if self.debug:
                        logger.info(f"🔍 DEBUG: Files structure: {json.dumps(files, indent=2)}")
                    
                    # Try RAG first
                    ai_response = await self.get_rag_response(processed_content, files, model_id, channel_id, selected_model)
                    
                    # If RAG fails or returns an error message, fall back to regular AI
                    if ai_response and any(error_indicator in ai_response.lower() for error_indicator in 
                                        ["error", "unexpected", "needs to be handled", "encountered"]):
                        logger.warning(f"⚠️ RAG failed, falling back to regular AI response")
                        # Update thinking indicator to show we're switching to regular AI
                        if channel_id and self._last_thinking_message_id:
                            await self.update_message_content(channel_id, self._last_thinking_message_id, 
                                                            "Processing with regular AI (RAG unavailable)...", selected_model)
                        
                        # Try regular AI response
                        ai_response = await self.get_ai_response(processed_content, model_id, channel_id, selected_model)
                else:
                    # No files, use regular AI response with selected model
                    ai_response = await self.get_ai_response(processed_content, model_id, channel_id, selected_model)
                
                # Response is already streamed in real-time, just log completion
                display_name = f"{model_name}" if selected_model else "Open WebUI"
                logger.info(f"✅ {self.bot_name} ({display_name}): {ai_response[:100]}{'...' if len(ai_response) > 100 else ''}")
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
    
    async def run(self):
        """Main bot loop"""
        logger.info(f"🚀 Starting {self.bot_name} with Open WebUI models...")
        
        # Get channels
        channels = await self.get_channels()
        if not channels:
            logger.error("❌ No accessible channels found")
            return
            
        # Filter channels with bot access
        bot_enabled_channels = [ch for ch in channels if self.has_bot_access(ch)]
        
        logger.info(f"✅ Found {len(channels)} channels, {len(bot_enabled_channels)} with bot access enabled:")
        for channel in channels:
            bot_status = "🤖" if self.has_bot_access(channel) else "🚫"
            logger.info(f"   {bot_status} {channel['name']}")
        
        if not bot_enabled_channels:
            logger.warning("\n⚠️  No channels have bot access enabled!")
            logger.warning("   Enable bot access in channel settings to allow the bot to respond.")
            return
        
        logger.info(f"\n🎯 {self.bot_name} is ready! To interact with the AI:")
        logger.info(f"   1. Type @ to select a model")
        logger.info(f"   2. Check the AI checkbox to enable AI responses")
        if self.triggers:
            logger.info(f"   Alternative: Messages containing: {', '.join(self.triggers)}")
        logger.info("")
        
        # Main loop
        while True:
            try:
                # Process each channel
                for channel in channels:
                    await self.process_channel(channel)
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                if not self.debug:
                    logger.info(".", end="", flush=True)
                
            except KeyboardInterrupt:
                logger.info(f"\n👋 {self.bot_name} is shutting down...")
                break
            except Exception as e:
                logger.error(f"\n❌ Unexpected error: {e}")
                logger.info("🔄 Continuing in 5 seconds...")
                await asyncio.sleep(5)


def main():
    """Main entry point for standalone execution"""
    try:
        bot = MultiProviderAIBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("\n👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 