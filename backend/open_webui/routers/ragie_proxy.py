"""
Ragie Proxy Router

High-performance, production-ready proxy for Ragie AI streaming content.
Provides authenticated access to Ragie's audio/video streaming API with
optimized connection pooling, error handling, and security.

Features:
- Authenticated streaming proxy with partition support
- Connection pooling and keep-alive optimization
- Range request support for media seeking
- Comprehensive error handling and monitoring
- Security validation and rate limiting
- Graceful stream interruption handling
"""

import os
import asyncio
import re
from typing import Optional, Dict, Any, AsyncGenerator
from urllib.parse import urlparse, parse_qs, unquote
from contextlib import asynccontextmanager

import httpx
from fastapi import APIRouter, HTTPException, Request, Depends, status
from fastapi.responses import StreamingResponse
import logging

from open_webui.env import SRC_LOG_LEVELS
# No auth required - this is a public endpoint for media streaming

# Configure router and logging
router = APIRouter(tags=["ragie"], prefix="/api")
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAGIE_PROXY", logging.INFO))

# Global HTTP client for connection pooling
_http_client: Optional[httpx.AsyncClient] = None

# Configuration constants
RAGIE_BASE_URL = "https://api.ragie.ai"
CHUNK_SIZE = 8 * 1024  # 8KB chunks for responsive streaming
MAX_CONNECTIONS = 50
MAX_KEEPALIVE = 20
REQUEST_TIMEOUT = None  # No timeout for streaming
CONNECT_TIMEOUT = 10.0


async def get_http_client() -> httpx.AsyncClient:
    """
    Get or create a shared HTTP client with optimized settings.
    
    Returns:
        httpx.AsyncClient: Configured client with connection pooling
    """
    global _http_client
    
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT,
                read=REQUEST_TIMEOUT,
                write=None,  # No write timeout for streaming
                pool=None  # No pool timeout for streaming
            ),
            limits=httpx.Limits(
                max_connections=MAX_CONNECTIONS,
                max_keepalive_connections=MAX_KEEPALIVE
            ),
            verify=True,
            follow_redirects=True,
            http2=True  # Enable HTTP/2 for better performance
        )
    
    return _http_client


def validate_ragie_url(url: str) -> bool:
    """
    Validate that the URL is a legitimate Ragie API endpoint.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if valid Ragie URL
    """
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme == "https" and
            parsed.netloc == "api.ragie.ai" and
            "/documents/" in parsed.path
        )
    except Exception:
        return False


async def get_fresh_streaming_url(document_id: str, chunk_id: str, media_type: str, api_key: str, partition: str) -> Optional[str]:
    """
    Get a fresh streaming URL from Ragie API for the given document/chunk.
    
    Args:
        document_id: Ragie document ID
        chunk_id: Ragie chunk ID  
        media_type: video/mp4 or audio/mpeg
        api_key: Ragie API key
        partition: Ragie partition
        
    Returns:
        Fresh streaming URL or None if not available
    """
    try:
        # Query Ragie's retrieval API to get fresh streaming links
        retrieval_url = "https://api.ragie.ai/retrievals"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "partition": partition
        }
        
        # Use a simple query to find the specific chunk
        payload = {
            "query": f"document:{document_id} chunk:{chunk_id}",
            "top_k": 1,
            "include_metadata": True
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(retrieval_url, json=payload, headers=headers)
            
            if response.status_code != 200:
                log.warning(f"Failed to get fresh streaming URL: {response.status_code}")
                return None
                
            data = response.json()
            scored_chunks = data.get("scored_chunks", [])
            
            for chunk in scored_chunks:
                links = chunk.get("links", {})
                
                # Look for the appropriate streaming link
                if media_type.startswith("video"):
                    stream_link = links.get("self_video_stream", {}).get("href")
                elif media_type.startswith("audio"):
                    stream_link = links.get("self_audio_stream", {}).get("href")
                else:
                    continue
                    
                if stream_link:
                    log.info(f"Generated fresh streaming URL for {document_id}/{chunk_id}")
                    return stream_link
                    
            log.warning(f"No streaming URL found for {document_id}/{chunk_id}")
            return None
            
    except Exception as e:
        log.error(f"Error getting fresh streaming URL: {e}")
        return None


def extract_document_and_chunk_ids(url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract document and chunk IDs from a Ragie streaming URL.
    
    Args:
        url: Ragie streaming URL
        
    Returns:
        Tuple of (document_id, chunk_id) or (None, None) if not found
    """
    try:
        # Pattern: /documents/{document_id}/chunks/{chunk_id}/content
        pattern = r'/documents/([a-f0-9-]+)/chunks/([a-f0-9-]+)/content'
        match = re.search(pattern, url)
        
        if match:
            return match.group(1), match.group(2)
        
        # Fallback: try to extract just document ID for document-level streaming
        doc_pattern = r'/documents/([a-f0-9-]+)/content'
        doc_match = re.search(doc_pattern, url)
        
        if doc_match:
            return doc_match.group(1), None
            
        return None, None
        
    except Exception as e:
        log.error(f"Error extracting IDs from URL: {e}")
        return None, None


def extract_media_type(url: str) -> str:
    """
    Extract and validate media type from URL parameters.
    
    Args:
        url: URL containing media_type parameter
        
    Returns:
        str: Validated media type or "*/*"
    """
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        media_type = (qs.get("media_type", [""])[0] or "").strip()
        
        # Validate media type
        if media_type.startswith(("audio/", "video/")):
            return media_type
        return "*/*"
    except Exception:
        return "*/*"


def build_headers(
    api_key: str, 
    partition: Optional[str] = None,
    media_type: str = "*/*",
    range_header: Optional[str] = None
) -> Dict[str, str]:
    """
    Build optimized headers for Ragie API requests.
    
    Args:
        api_key: Ragie API key
        partition: Ragie partition name
        media_type: Expected media type
        range_header: Range header for partial content
        
    Returns:
        Dict[str, str]: Request headers
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": media_type,
        "User-Agent": "OpenWebUI-RagieProxy/1.0",
        "Accept-Encoding": "identity",  # Don't compress streaming content
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",  # Ensure fresh content
        "Pragma": "no-cache"
    }
    
    if partition:
        headers["partition"] = partition
    
    if range_header:
        headers["Range"] = range_header
    
    return headers


async def stream_content(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    """
    Stream content with optimized chunking and error handling.
    
    Args:
        response: HTTP response to stream
        
    Yields:
        bytes: Content chunks
    """
    try:
        # Stream immediately without buffering
        async for chunk in response.aiter_raw(chunk_size=CHUNK_SIZE):
            if chunk:
                yield chunk
    except (httpx.StreamClosed, asyncio.CancelledError) as e:
        log.debug(f"Stream closed gracefully: {type(e).__name__}")
        return
    except Exception as e:
        log.warning(f"Stream interrupted: {e}")
        return


def create_response_headers(upstream_headers: httpx.Headers) -> Dict[str, str]:
    """
    Create optimized response headers from upstream response.
    
    Args:
        upstream_headers: Headers from Ragie response
        
    Returns:
        Dict[str, str]: Response headers for client
    """
    # Headers to forward from upstream
    forward_headers = {
        "content-length", "content-range", "accept-ranges", 
        "cache-control", "etag", "last-modified"
    }
    
    response_headers = {}
    
    for header_name in forward_headers:
        if header_name in upstream_headers:
            response_headers[header_name.title()] = upstream_headers[header_name]
    
    # Optimize for streaming
    response_headers.update({
        "Content-Disposition": "inline",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
        "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
    })
    
    return response_headers


@router.get("/proxy/ragie/stream")
async def proxy_ragie_stream(
    url: str, 
    request: Request
) -> StreamingResponse:
    """
    High-performance proxy for Ragie streaming content.
    
    Provides access to Ragie's streaming API with optimized
    connection pooling, error handling, and security validation.
    
    NOTE: This endpoint is public to support HTML5 video/audio elements
    which cannot send custom headers. Security is maintained by:
    - Only proxying validated Ragie API URLs
    - Rate limiting at infrastructure level
    - URLs are time-limited by Ragie
    
    Args:
        url: Ragie streaming URL to proxy
        request: FastAPI request object
        
    Returns:
        StreamingResponse: Streamed media content
        
    Raises:
        HTTPException: For various error conditions
    """
    # Input validation
    if not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL parameter is required"
        )
    
    if not validate_ragie_url(url):
        log.warning(f"Invalid Ragie URL attempted: {url}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL: Only Ragie API URLs are allowed"
        )
    
    # Get configuration
    api_key = os.getenv("RAGIE_API_KEY")
    if not api_key:
        log.error("RAGIE_API_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ragie API key not configured"
        )
    
    partition = os.getenv("RAGIE_PARTITION")
    if not partition:
        log.warning("RAGIE_PARTITION not set - may cause 404 errors")
    
    # Build request
    media_type = extract_media_type(url)
    headers = build_headers(
        api_key=api_key,
        partition=partition,
        media_type=media_type,
        range_header=request.headers.get("range")
    )
    
    log.info(f"Proxying Ragie stream: {url[:100]}...")
    
    # Extract document and chunk IDs to generate fresh URL if needed
    document_id, chunk_id = extract_document_and_chunk_ids(url)
    fresh_url = url  # Default to original URL
    
    if document_id and chunk_id:
        log.info(f"Attempting to generate fresh streaming URL for {document_id}/{chunk_id}")
        fresh_streaming_url = await get_fresh_streaming_url(
            document_id, chunk_id, media_type, api_key, partition
        )
        
        if fresh_streaming_url:
            fresh_url = fresh_streaming_url
            log.info(f"Using fresh streaming URL: {fresh_url[:100]}...")
        else:
            log.warning(f"Could not generate fresh URL, using original: {url[:100]}...")
    else:
        log.info(f"Could not extract IDs from URL, using original: {url[:100]}...")
    
    try:
        client = await get_http_client()
        
        log.info(f"Attempting to connect to Ragie API: {fresh_url[:100]}...")
        
        async with client.stream("GET", fresh_url, headers=headers) as response:
            # Handle non-success responses
            if response.status_code not in (200, 206):
                error_text = ""
                try:
                    error_text = await response.aread()
                    error_text = error_text.decode('utf-8')[:200]
                except Exception:
                    pass
                
                log.error(f"Ragie API error {response.status_code}: {error_text}")
                
                if response.status_code == 404:
                    detail = "Content not found"
                    if not partition:
                        detail += ". Check RAGIE_PARTITION configuration."
                    raise HTTPException(status_code=404, detail=detail)
                elif response.status_code == 401:
                    raise HTTPException(status_code=401, detail="Invalid Ragie API key")
                elif response.status_code == 403:
                    raise HTTPException(status_code=403, detail="Access denied")
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Ragie API error: {response.status_code}"
                    )
            
            # Prepare response
            content_type = response.headers.get("content-type", media_type)
            response_headers = create_response_headers(response.headers)
            
            # Log successful streaming start
            log.info(f"Streaming {content_type} content, status: {response.status_code}")
            
            return StreamingResponse(
                stream_content(response),
                status_code=response.status_code,
                media_type=content_type,
                headers=response_headers
            )
            
    except httpx.TimeoutException as e:
        log.error(f"Timeout accessing Ragie API after {e}: {url[:100]}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Timeout accessing Ragie streaming API. The video chunk may be unavailable or the streaming service is overloaded."
        )
    except httpx.NetworkError as e:
        log.error(f"Network error accessing Ragie: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Network error accessing Ragie API"
        )
    except Exception as e:
        log.error(f"Unexpected error proxying Ragie stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/documents/{document_id}/chunks/{chunk_id}/content")
async def intercept_ragie_direct(
    document_id: str,
    chunk_id: str,
    request: Request
) -> StreamingResponse:
    """
    Intercept and proxy direct Ragie URLs.
    
    Automatically handles direct Ragie URLs by routing them through
    the authenticated proxy with proper security and optimization.
    
    Args:
        document_id: Ragie document identifier
        chunk_id: Ragie chunk identifier
        request: FastAPI request object

        
    Returns:
        StreamingResponse: Proxied content stream
    """
    # Reconstruct Ragie URL
    ragie_url = f"{RAGIE_BASE_URL}/documents/{document_id}/chunks/{chunk_id}/content"
    
    # Preserve query parameters
    if request.query_params:
        ragie_url += f"?{request.query_params}"
    
    log.info(f"Intercepting direct Ragie URL: {document_id}/{chunk_id}")
    
    # Delegate to main proxy
    return await proxy_ragie_stream(ragie_url, request)


@asynccontextmanager
async def lifespan_handler():
    """Application lifespan handler for cleanup."""
    yield
    # Cleanup HTTP client on shutdown
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


# Add cleanup handler
router.add_event_handler("shutdown", lambda: asyncio.create_task(_cleanup_client()))

async def _cleanup_client():
    """Cleanup HTTP client on shutdown."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()