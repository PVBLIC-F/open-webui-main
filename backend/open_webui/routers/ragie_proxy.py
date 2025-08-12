"""
Ragie Proxy Router

This module provides authenticated proxy endpoints for streaming Ragie audio and video content.
It handles authentication with the Ragie API and streams content back to clients.
"""

import os
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import logging

from open_webui.env import SRC_LOG_LEVELS

router = APIRouter()

# Set up logging
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAGIE_PROXY", logging.INFO))

# Get Ragie API key from environment
RAGIE_API_KEY = os.getenv("RAGIE_API_KEY", "")

@router.get("/api/proxy/ragie/stream")
async def proxy_ragie_stream(url: str):
    """
    Proxy any Ragie streaming URL with authentication.
    
    This endpoint accepts actual Ragie streaming URLs from the links object
    and proxies them with proper authentication headers.
    
    Args:
        url: The Ragie streaming URL to proxy (from chunk.links.*.href)
    
    Returns:
        StreamingResponse: The media stream from Ragie
    """
    if not RAGIE_API_KEY:
        raise HTTPException(status_code=500, detail="Ragie API key not configured")
    
    # Validate that it's a Ragie URL for security
    if not url.startswith("https://api.ragie.ai/"):
        raise HTTPException(status_code=400, detail="Invalid URL: Only Ragie URLs are allowed")
    
    # Determine content type from URL parameters
    if "media_type=audio" in url or "audio" in url:
        accept_type = "audio/mpeg"
        media_type = "audio/mpeg"
        file_extension = "mp3"
    elif "media_type=video" in url or "video" in url:
        accept_type = "video/mp4"
        media_type = "video/mp4"
        file_extension = "mp4"
    else:
        # Default to audio if content type is unclear
        accept_type = "audio/mpeg"
        media_type = "audio/mpeg"
        file_extension = "mp3"
    
    headers = {
        "Authorization": f"Bearer {RAGIE_API_KEY}",
        "Accept": accept_type
    }
    
    try:
        log.info(f"Proxying Ragie stream: {url[:100]}...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                log.error(f"Ragie stream failed: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Failed to stream from Ragie")
            
            return StreamingResponse(
                iter([response.content]),
                media_type=media_type,
                headers={
                    "Content-Disposition": f"inline; filename=ragie_stream.{file_extension}",
                    "Cache-Control": "public, max-age=3600"
                }
            )
            
    except httpx.RequestError as e:
        log.error(f"Network error proxying Ragie stream: {e}")
        raise HTTPException(status_code=503, detail="Network error accessing Ragie")
    except Exception as e:
        log.error(f"Unexpected error proxying Ragie stream: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")