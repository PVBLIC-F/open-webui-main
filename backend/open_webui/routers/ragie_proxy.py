"""
Ragie Proxy Router

This module provides authenticated proxy endpoints for streaming Ragie audio and video content.
It handles authentication with the Ragie API, SSL/TLS compatibility, and streams content back 
to clients.

Features:
- Proxy endpoint for explicit Ragie URL proxying (/api/proxy/ragie/stream)
- Automatic interception of direct Ragie URLs (/documents/{id}/chunks/{id}/content)
- User authentication and authorization
- SSL/TLS compatibility handling
- Range request support for media seeking
- Comprehensive error handling and logging
"""

import os
import httpx
from urllib.parse import urlparse, parse_qs

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
import logging

from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.auth import get_verified_user

router = APIRouter()

# Set up logging
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAGIE_PROXY", logging.INFO))

@router.get("/api/proxy/ragie/stream")
async def proxy_ragie_stream(url: str, request: Request, user=Depends(get_verified_user)):
    """
    Proxy Ragie streaming URLs with authentication and SSL compatibility.
    
    This endpoint accepts Ragie streaming URLs and proxies them with proper
    authentication headers, handling SSL/TLS compatibility issues that may
    occur when accessing Ragie directly from the browser.
    
    Args:
        url: The Ragie streaming URL to proxy (must be from api.ragie.ai)
        request: FastAPI request object for header forwarding
        user: Authenticated user (injected by dependency)
    
    Returns:
        StreamingResponse: The authenticated media stream from Ragie
        
    Raises:
        HTTPException: 400 for invalid URLs, 500 for configuration errors,
                      503 for network errors
    """
    # Log the authenticated user
    log.info(f"Ragie proxy request from user: {user.email if user else 'unknown'}")
    
    # Get Ragie API key from environment at runtime
    RAGIE_API_KEY = os.getenv("RAGIE_API_KEY", "")
    
    if not RAGIE_API_KEY:
        log.error("RAGIE_API_KEY not found in environment")
        raise HTTPException(status_code=500, detail="Ragie API key not configured")
    
    # Validate that it's a Ragie URL for security
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "api.ragie.ai":
        raise HTTPException(status_code=400, detail="Invalid URL: Only Ragie API URLs are allowed")

    # Determine desired media type from query parameters, fall back conservatively
    qs = parse_qs(parsed.query)
    requested_media_type = (qs.get("media_type", [""])[0] or "").strip()
    if requested_media_type.startswith("audio/"):
        accept_type = requested_media_type
    elif requested_media_type.startswith("video/"):
        accept_type = requested_media_type
    else:
        # Let the upstream decide if unspecified or JSON/text; default to octet-stream
        accept_type = "*/*"
    
    # Build upstream headers
    headers = {"Authorization": f"Bearer {RAGIE_API_KEY}", "Accept": accept_type}
    # Forward range requests for proper media seeking
    range_header = request.headers.get("range")
    if range_header:
        headers["Range"] = range_header
    
    try:
        log.info(f"Proxying Ragie stream: {url}")

        # Configure SSL/TLS settings for better compatibility with Ragie API
        # Use more lenient SSL settings while maintaining security for api.ragie.ai
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0), 
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            verify=True  # Keep SSL verification but let httpx handle compatibility
        ) as client:
            async with client.stream("GET", url, headers=headers, follow_redirects=True) as resp:
                if resp.status_code not in (200, 206):
                    # Try to surface upstream error body
                    text = await resp.aread()
                    log.error(f"Ragie stream failed: {resp.status_code} - {text[:200]!r}")
                    raise HTTPException(status_code=resp.status_code, detail="Failed to stream from Ragie")

                upstream_content_type = resp.headers.get("content-type", accept_type)

                # Select headers to forward to the client
                response_headers = {}
                for header_name in ("content-length", "content-range", "accept-ranges", "cache-control", "content-disposition"):
                    if header_name in resp.headers:
                        response_headers[header_name.title()] = resp.headers[header_name]
                # Ensure inline disposition to allow in-browser playback when Ragie doesn't set it
                response_headers.setdefault("Content-Disposition", "inline")

                async def byte_iterator():
                    async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                        if chunk:
                            yield chunk

                return StreamingResponse(
                    byte_iterator(),
                    status_code=resp.status_code,
                    media_type=upstream_content_type,
                    headers=response_headers,
                )
            
    except httpx.RequestError as e:
        log.error(f"Network error proxying Ragie stream: {e}")
        raise HTTPException(status_code=503, detail="Network error accessing Ragie")
    except Exception as e:
        log.error(f"Unexpected error proxying Ragie stream: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/documents/{document_id}/chunks/{chunk_id}/content")
async def intercept_ragie_direct(
    document_id: str, 
    chunk_id: str, 
    request: Request,
    user=Depends(get_verified_user)
):
    """
    Intercept direct Ragie URLs and proxy them with authentication.
    
    This endpoint catches direct Ragie URLs that bypass the proxy and automatically
    routes them through the authenticated proxy. This ensures that even if the LLM
    generates direct Ragie URLs, they will still work with proper authentication.
    
    Args:
        document_id: The Ragie document ID
        chunk_id: The Ragie chunk ID
        request: FastAPI request object (contains query parameters)
        user: Authenticated user from dependency injection
        
    Returns:
        StreamingResponse: The authenticated media stream from Ragie
    """
    log.info(f"Intercepting direct Ragie URL: documents/{document_id}/chunks/{chunk_id}/content")
    
    # Reconstruct the original Ragie URL with all query parameters
    ragie_url = f"https://api.ragie.ai/documents/{document_id}/chunks/{chunk_id}/content"
    
    # Preserve all query parameters from the original request
    if request.query_params:
        query_string = str(request.query_params)
        ragie_url += f"?{query_string}"
    
    log.info(f"Proxying intercepted URL: {ragie_url}")
    
    # Delegate to the main proxy function
    return await proxy_ragie_stream(ragie_url, request, user)