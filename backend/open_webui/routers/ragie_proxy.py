"""
Ragie Proxy Router - Handles authenticated streaming of Ragie audio/video content

This router provides proxy endpoints that add proper authentication headers
to Ragie streaming URLs, allowing users to access audio/video segments
without exposing API keys in the frontend.
"""

import os
import httpx
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
import logging

from open_webui.env import SRC_LOG_LEVELS

router = APIRouter()

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAGIE_PROXY", logging.INFO))

# Get Ragie API key from environment
RAGIE_API_KEY = os.getenv("RAGIE_API_KEY", "")

@router.get("/api/proxy/ragie/audio/{document_id}/{chunk_id}")
async def proxy_ragie_audio(document_id: str, chunk_id: str):
    """
    Proxy Ragie audio streaming with authentication.
    
    Args:
        document_id: The Ragie document ID
        chunk_id: The Ragie chunk ID
    
    Returns:
        StreamingResponse: The audio stream from Ragie
    """
    if not RAGIE_API_KEY:
        raise HTTPException(status_code=500, detail="Ragie API key not configured")
    
    # Construct the Ragie streaming URL
    ragie_url = f"https://api.ragie.ai/documents/{document_id}/chunks/{chunk_id}/content?media_type=audio/mpeg"
    
    headers = {
        "Authorization": f"Bearer {RAGIE_API_KEY}",
        "Accept": "audio/mpeg"
    }
    
    try:
        log.info(f"Proxying Ragie audio stream: {document_id}/{chunk_id}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(ragie_url, headers=headers)
            
            if response.status_code != 200:
                log.error(f"Ragie audio stream failed: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Failed to stream audio from Ragie")
            
            # Stream the audio content back to the client
            return StreamingResponse(
                iter([response.content]),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"inline; filename=audio_segment_{chunk_id}.mp3",
                    "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
                }
            )
            
    except httpx.RequestError as e:
        log.error(f"Network error proxying Ragie audio: {e}")
        raise HTTPException(status_code=503, detail="Network error accessing Ragie")
    except Exception as e:
        log.error(f"Unexpected error proxying Ragie audio: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/api/proxy/ragie/video/{document_id}/{chunk_id}")
async def proxy_ragie_video(document_id: str, chunk_id: str):
    """
    Proxy Ragie video streaming with authentication.
    
    Args:
        document_id: The Ragie document ID
        chunk_id: The Ragie chunk ID
    
    Returns:
        StreamingResponse: The video stream from Ragie
    """
    if not RAGIE_API_KEY:
        raise HTTPException(status_code=500, detail="Ragie API key not configured")
    
    # Construct the Ragie streaming URL
    ragie_url = f"https://api.ragie.ai/documents/{document_id}/chunks/{chunk_id}/content?media_type=video/mp4"
    
    headers = {
        "Authorization": f"Bearer {RAGIE_API_KEY}",
        "Accept": "video/mp4"
    }
    
    try:
        log.info(f"Proxying Ragie video stream: {document_id}/{chunk_id}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(ragie_url, headers=headers)
            
            if response.status_code != 200:
                log.error(f"Ragie video stream failed: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Failed to stream video from Ragie")
            
            # Stream the video content back to the client
            return StreamingResponse(
                iter([response.content]),
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f"inline; filename=video_segment_{chunk_id}.mp4",
                    "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
                }
            )
            
    except httpx.RequestError as e:
        log.error(f"Network error proxying Ragie video: {e}")
        raise HTTPException(status_code=503, detail="Network error accessing Ragie")
    except Exception as e:
        log.error(f"Unexpected error proxying Ragie video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/api/proxy/ragie/full-audio/{document_id}")
async def proxy_ragie_full_audio(document_id: str):
    """
    Proxy full Ragie audio document streaming with authentication.
    
    Args:
        document_id: The Ragie document ID
    
    Returns:
        StreamingResponse: The full audio document stream from Ragie
    """
    if not RAGIE_API_KEY:
        raise HTTPException(status_code=500, detail="Ragie API key not configured")
    
    # Construct the Ragie streaming URL for full document
    ragie_url = f"https://api.ragie.ai/documents/{document_id}/content?media_type=audio/mpeg"
    
    headers = {
        "Authorization": f"Bearer {RAGIE_API_KEY}",
        "Accept": "audio/mpeg"
    }
    
    try:
        log.info(f"Proxying Ragie full audio document: {document_id}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(ragie_url, headers=headers)
            
            if response.status_code != 200:
                log.error(f"Ragie full audio stream failed: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Failed to stream full audio from Ragie")
            
            # Stream the audio content back to the client
            return StreamingResponse(
                iter([response.content]),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"inline; filename=full_audio_{document_id}.mp3",
                    "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
                }
            )
            
    except httpx.RequestError as e:
        log.error(f"Network error proxying Ragie full audio: {e}")
        raise HTTPException(status_code=503, detail="Network error accessing Ragie")
    except Exception as e:
        log.error(f"Unexpected error proxying Ragie full audio: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/api/proxy/ragie/full-video/{document_id}")
async def proxy_ragie_full_video(document_id: str):
    """
    Proxy full Ragie video document streaming with authentication.
    
    Args:
        document_id: The Ragie document ID
    
    Returns:
        StreamingResponse: The full video document stream from Ragie
    """
    if not RAGIE_API_KEY:
        raise HTTPException(status_code=500, detail="Ragie API key not configured")
    
    # Construct the Ragie streaming URL for full document
    ragie_url = f"https://api.ragie.ai/documents/{document_id}/content?media_type=video/mp4"
    
    headers = {
        "Authorization": f"Bearer {RAGIE_API_KEY}",
        "Accept": "video/mp4"
    }
    
    try:
        log.info(f"Proxying Ragie full video document: {document_id}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(ragie_url, headers=headers)
            
            if response.status_code != 200:
                log.error(f"Ragie full video stream failed: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Failed to stream full video from Ragie")
            
            # Stream the video content back to the client
            return StreamingResponse(
                iter([response.content]),
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f"inline; filename=full_video_{document_id}.mp4",
                    "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
                }
            )
            
    except httpx.RequestError as e:
        log.error(f"Network error proxying Ragie full video: {e}")
        raise HTTPException(status_code=503, detail="Network error accessing Ragie")
    except Exception as e:
        log.error(f"Unexpected error proxying Ragie full video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/api/proxy/ragie/download/audio/{document_id}/{chunk_id}")
async def download_ragie_audio_segment(document_id: str, chunk_id: str):
    """
    Download Ragie audio segment with authentication.
    
    Args:
        document_id: The Ragie document ID
        chunk_id: The Ragie chunk ID
    
    Returns:
        StreamingResponse: The audio file download from Ragie
    """
    if not RAGIE_API_KEY:
        raise HTTPException(status_code=500, detail="Ragie API key not configured")
    
    # Construct the Ragie download URL
    ragie_url = f"https://api.ragie.ai/documents/{document_id}/chunks/{chunk_id}/content?media_type=audio/mpeg&download=true"
    
    headers = {
        "Authorization": f"Bearer {RAGIE_API_KEY}",
        "Accept": "audio/mpeg"
    }
    
    try:
        log.info(f"Downloading Ragie audio segment: {document_id}/{chunk_id}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(ragie_url, headers=headers)
            
            if response.status_code != 200:
                log.error(f"Ragie audio download failed: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Failed to download audio from Ragie")
            
            # Return the audio file for download
            return StreamingResponse(
                iter([response.content]),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"attachment; filename=audio_segment_{chunk_id}.mp3",
                    "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
                }
            )
            
    except httpx.RequestError as e:
        log.error(f"Network error downloading Ragie audio: {e}")
        raise HTTPException(status_code=503, detail="Network error accessing Ragie")
    except Exception as e:
        log.error(f"Unexpected error downloading Ragie audio: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/api/proxy/ragie/download/video/{document_id}/{chunk_id}")
async def download_ragie_video_segment(document_id: str, chunk_id: str):
    """
    Download Ragie video segment with authentication.
    
    Args:
        document_id: The Ragie document ID
        chunk_id: The Ragie chunk ID
    
    Returns:
        StreamingResponse: The video file download from Ragie
    """
    if not RAGIE_API_KEY:
        raise HTTPException(status_code=500, detail="Ragie API key not configured")
    
    # Construct the Ragie download URL
    ragie_url = f"https://api.ragie.ai/documents/{document_id}/chunks/{chunk_id}/content?media_type=video/mp4&download=true"
    
    headers = {
        "Authorization": f"Bearer {RAGIE_API_KEY}",
        "Accept": "video/mp4"
    }
    
    try:
        log.info(f"Downloading Ragie video segment: {document_id}/{chunk_id}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(ragie_url, headers=headers)
            
            if response.status_code != 200:
                log.error(f"Ragie video download failed: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Failed to download video from Ragie")
            
            # Return the video file for download
            return StreamingResponse(
                iter([response.content]),
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f"attachment; filename=video_segment_{chunk_id}.mp4",
                    "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
                }
            )
            
    except httpx.RequestError as e:
        log.error(f"Network error downloading Ragie video: {e}")
        raise HTTPException(status_code=503, detail="Network error accessing Ragie")
    except Exception as e:
        log.error(f"Unexpected error downloading Ragie video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
