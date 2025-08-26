"""
Simple Ragie Proxy - Just proxy the damn URLs with auth headers
"""

import os
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import logging

router = APIRouter(tags=["ragie"], prefix="/api")
log = logging.getLogger(__name__)

@router.options("/proxy/ragie/stream")
async def proxy_ragie_stream_options():
    """Handle CORS preflight requests for media players."""
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Content-Type",
            "Access-Control-Max-Age": "3600"
        }
    )

@router.get("/proxy/ragie/stream")
async def proxy_ragie_stream(url: str, request: Request) -> StreamingResponse:
    """
    Simple proxy for Ragie streaming URLs.
    Takes a Ragie URL, adds auth headers, streams the response.
    """
    
    # Validate it's a Ragie URL
    if not url.startswith("https://api.ragie.ai/"):
        raise HTTPException(status_code=400, detail="Only Ragie URLs allowed")
    
    # Get auth from environment
    api_key = os.getenv("RAGIE_API_KEY")
    partition = os.getenv("RAGIE_PARTITION")
    
    if not api_key:
        raise HTTPException(status_code=500, detail="RAGIE_API_KEY not configured")
    
    # Build headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "*/*"
    }
    
    if partition:
        headers["partition"] = partition
    
    # Add Range header if browser sent one (for video seeking)
    if "range" in request.headers:
        headers["Range"] = request.headers["range"]
    
    log.info(f"Proxying: {url[:100]}...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("GET", url, headers=headers) as response:
                
                if response.status_code not in (200, 206):
                    log.error(f"Ragie error {response.status_code}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Ragie API error: {response.status_code}"
                    )
                
                # Forward the content type and other important headers
                response_headers = {}
                for header in ["content-type", "content-length", "content-range", "accept-ranges"]:
                    if header in response.headers:
                        response_headers[header] = response.headers[header]
                
                # Add CORS headers for browser media players
                response_headers.update({
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                    "Access-Control-Allow-Headers": "Range, Content-Type",
                    "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges"
                })
                
                log.info(f"Streaming {response.headers.get('content-type', 'unknown')} - Status: {response.status_code}")
                
                # Create a safe streaming generator that handles disconnections
                async def safe_stream():
                    try:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            if chunk:
                                yield chunk
                    except (httpx.StreamClosed, ConnectionResetError, BrokenPipeError) as e:
                        log.debug(f"Stream closed by client: {type(e).__name__}")
                        return
                    except Exception as e:
                        log.warning(f"Stream error: {e}")
                        return
                
                return StreamingResponse(
                    safe_stream(),
                    status_code=response.status_code,
                    headers=response_headers
                )
                
    except httpx.TimeoutException:
        log.error("Timeout accessing Ragie")
        raise HTTPException(status_code=504, detail="Timeout accessing Ragie API")
    except Exception as e:
        log.error(f"Error proxying Ragie stream: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
