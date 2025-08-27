"""
Video Player Router - Serves HTML video player pages for authenticated streaming
"""

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
import urllib.parse
import logging
import html
import re

log = logging.getLogger(__name__)

router = APIRouter()


def _is_safe_url(url: str) -> bool:
    """
    Validate that the URL is safe for use in HTML context.
    Prevents XSS attacks and ensures only HTTP/HTTPS URLs are allowed.
    """
    if not url or not isinstance(url, str):
        return False
    
    # Check for basic URL format
    if not re.match(r'^https?://', url, re.IGNORECASE):
        return False
    
    # Prevent javascript: and data: URLs
    if re.match(r'^(javascript|data|vbscript):', url, re.IGNORECASE):
        return False
    
    # Additional validation: ensure it's a reasonable URL
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc:
            return False
        # Only allow specific domains (optional - can be configured)
        # For now, allow any HTTPS domain
        return True
    except Exception:
        return False


@router.get("/video", response_class=HTMLResponse)
async def video_player(
    request: Request,
    stream: str = Query(..., description="The video stream URL to play")
):
    """
    Simple, clean video player that loads the stream directly.
    """
    try:
        # Decode and validate the stream URL
        stream_url = urllib.parse.unquote(stream)
        
        # Handle relative URLs by converting them to absolute URLs
        if stream_url.startswith('/'):
            # Convert relative URL to absolute URL using request info
            base_url = f"{request.url.scheme}://{request.url.netloc}"
            stream_url = f"{base_url}{stream_url}"
            log.info(f"Converted relative URL to absolute: {stream_url}")
        
        # Security: Validate URL format and prevent XSS
        if not _is_safe_url(stream_url):
            log.warning(f"Rejected unsafe URL: {stream_url}")
            raise HTTPException(status_code=400, detail="Invalid or unsafe URL")
            
        # Escape URL for safe HTML injection
        safe_stream_url = html.escape(stream_url, quote=True)
        log.info(f"Video player loading stream: {stream_url}")
        
        # Simple, clean HTML5 video player with blob streaming
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #000; display: flex; align-items: center; justify-content: center; min-height: 100vh; font-family: system-ui; }}
        .container {{ text-align: center; color: white; }}
        video {{ width: 100%; height: auto; max-height: 100vh; margin: 20px 0; }}
        .loading {{ font-size: 18px; margin: 20px; }}
        .error {{ color: #ff6b6b; margin: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div id="loading" class="loading">Loading video...</div>
        <video id="videoPlayer" controls preload="none" playsinline style="display: none;">
            <p style="color: white; text-align: center;">
                Your browser doesn't support video playback.
            </p>
        </video>
        <div id="error" class="error" style="display: none;">
            Failed to load video. <a href="{safe_stream_url}" style="color: #4A9EFF;">Try direct link</a>
        </div>
    </div>
    
    <script>
        async function loadVideo() {{
            const video = document.getElementById('videoPlayer');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            
            try {{
                console.log('Fetching video from:', '{safe_stream_url}');
                
                // Fetch the video stream with authentication - disable range requests
                const response = await fetch('{safe_stream_url}', {{
                    method: 'GET',
                    headers: {{
                        'Range': '', // Explicitly disable range requests
                        'Cache-Control': 'no-cache'
                    }}
                }});
                
                console.log('Response status:', response.status);
                console.log('Response headers:', Object.fromEntries(response.headers.entries()));
                
                if (!response.ok) {{
                    throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
                }}
                
                // Get content type to verify it's video
                const contentType = response.headers.get('content-type');
                console.log('Content type:', contentType);
                
                // Create blob URL from the response
                const blob = await response.blob();
                console.log('Blob created, size:', blob.size, 'type:', blob.type);
                
                const videoUrl = URL.createObjectURL(blob);
                console.log('Blob URL created:', videoUrl);
                
                // Set video source and show player
                video.src = videoUrl;
                video.style.display = 'block';
                loading.style.display = 'none';
                
                // Clean up blob URL when video is done
                video.addEventListener('loadeddata', () => {{
                    console.log('Video loaded successfully');
                }});
                
                video.addEventListener('error', (e) => {{
                    console.error('Video playback error:', e);
                    console.error('Video error details:', video.error);
                }});
                
            }} catch (err) {{
                console.error('Video loading failed:', err);
                console.error('Error details:', err.message);
                loading.style.display = 'none';
                error.style.display = 'block';
                error.innerHTML = `Failed to load video: ${{err.message}}<br><a href="{safe_stream_url}" style="color: #4A9EFF;">Try direct link</a>`;
            }}
        }}
        
        // Start loading when page loads
        document.addEventListener('DOMContentLoaded', loadVideo);
    </script>
</body>
</html>
        """
        
        # Add security headers
        headers = {
            "Content-Security-Policy": "default-src 'self'; media-src 'self' https: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "SAMEORIGIN"
        }
        return HTMLResponse(content=html_content, headers=headers)
        
    except Exception as e:
        log.error(f"Error serving video player: {e}")
        raise HTTPException(status_code=500, detail="Failed to create video player")


@router.get("/audio", response_class=HTMLResponse)
async def audio_player(
    request: Request,
    stream: str = Query(..., description="The audio stream URL to play")
):
    """
    Simple, clean audio player that loads the stream directly.
    """
    try:
        # Decode and validate the stream URL
        stream_url = urllib.parse.unquote(stream)
        
        # Handle relative URLs by converting them to absolute URLs
        if stream_url.startswith('/'):
            # Convert relative URL to absolute URL using request info
            base_url = f"{request.url.scheme}://{request.url.netloc}"
            stream_url = f"{base_url}{stream_url}"
            log.info(f"Converted relative URL to absolute: {stream_url}")
        
        # Security: Validate URL format and prevent XSS
        if not _is_safe_url(stream_url):
            log.warning(f"Rejected unsafe URL: {stream_url}")
            raise HTTPException(status_code=400, detail="Invalid or unsafe URL")
            
        # Escape URL for safe HTML injection
        safe_stream_url = html.escape(stream_url, quote=True)
        log.info(f"Audio player loading stream: {stream_url}")
        
        # Simple, clean HTML5 audio player
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #f5f5f5; display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
        .audio-container {{ background: white; border-radius: 12px; padding: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
        audio {{ width: 100%; }}
    </style>
</head>
<body>
    <div class="audio-container">
        <audio controls preload="metadata">
            <source src="{safe_stream_url}" type="audio/mpeg">
            <source src="{safe_stream_url}" type="audio/mp4">
            <p>Your browser doesn't support audio playback.
            <br><a href="{safe_stream_url}" style="color: #4A9EFF;">Download audio</a></p>
        </audio>
    </div>
</body>
</html>
        """
        
        # Add security headers
        headers = {
            "Content-Security-Policy": "default-src 'self'; media-src 'self' https: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "SAMEORIGIN"
        }
        return HTMLResponse(content=html_content, headers=headers)
        
    except Exception as e:
        log.error(f"Error serving audio player: {e}")
        raise HTTPException(status_code=500, detail="Failed to create audio player")
