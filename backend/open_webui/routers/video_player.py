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
        
        # Enhanced HTML5 video player for chunked streaming
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            background: #000; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            min-height: 100vh; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .video-container {{
            width: 100%;
            max-width: 1200px;
            text-align: center;
        }}
        video {{ 
            width: 100%; 
            height: auto; 
            max-height: 80vh;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .status {{
            color: white;
            margin: 20px 0;
            font-size: 16px;
        }}
        .error {{
            color: #ff6b6b;
            background: rgba(255,107,107,0.1);
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid rgba(255,107,107,0.3);
        }}
        .download-link {{
            color: #4A9EFF;
            text-decoration: none;
            padding: 10px 20px;
            border: 2px solid #4A9EFF;
            border-radius: 6px;
            display: inline-block;
            margin: 20px 10px;
            transition: all 0.3s ease;
        }}
        .download-link:hover {{
            background: #4A9EFF;
            color: white;
        }}
        .loading {{
            color: #4A9EFF;
            font-size: 18px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="video-container">
        <div id="loading" class="loading">Click play to start streaming</div>
        <div id="error" class="error" style="display: none;"></div>
        
        <video id="videoPlayer" controls preload="none" playsinline crossorigin="anonymous">
            <p style="color: white; text-align: center;">
                Your browser doesn't support video playback.
            </p>
        </video>
        
        <div class="status" id="status">Click play button to start streaming</div>
        
        <div>
            <a href="{safe_stream_url}" class="download-link" target="_blank">Download Video</a>
            <a href="{safe_stream_url}" class="download-link" onclick="window.open('{safe_stream_url}', '_blank')">Open in New Tab</a>
        </div>
    </div>

    <script>
        const video = document.getElementById('videoPlayer');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const status = document.getElementById('status');
        
        let initialClipLoaded = false;
        let fullStreamStarted = false;
        
        function showError(message) {{
            error.style.display = 'block';
            error.innerHTML = message;
            loading.style.display = 'none';
            status.textContent = 'Error loading video';
        }}
        
        function showStatus(message) {{
            status.textContent = message;
        }}
        
        function hideLoading() {{
            loading.style.display = 'none';
        }}
        
        // Load initial clip (15-second preview)
        async function loadInitialClip() {{
            if (initialClipLoaded) return;
            
            try {{
                showStatus('Loading initial clip...');
                
                // Create a MediaSource for the initial clip
                const mediaSource = new MediaSource();
                const videoUrl = URL.createObjectURL(mediaSource);
                video.src = videoUrl;
                
                mediaSource.addEventListener('sourceopen', async () => {{
                    try {{
                        // Fetch initial clip with range header for first 15 seconds
                        const response = await fetch('{safe_stream_url}', {{
                            headers: {{
                                'Range': 'bytes=0-'
                            }}
                        }});
                        
                        if (!response.ok) {{
                            throw new Error(`HTTP ${{response.status}}`);
                        }}
                        
                        const sourceBuffer = mediaSource.addSourceBuffer('video/mp4; codecs="avc1.42E01E, mp4a.40.2"');
                        
                        sourceBuffer.addEventListener('updateend', () => {{
                            if (!sourceBuffer.updating) {{
                                mediaSource.endOfStream();
                                initialClipLoaded = true;
                                hideLoading();
                                showStatus('Initial clip loaded. Click play to start full stream.');
                            }}
                        }});
                        
                        const data = await response.arrayBuffer();
                        sourceBuffer.appendBuffer(data);
                        
                    }} catch (err) {{
                        console.error('Failed to load initial clip:', err);
                        showError('Failed to load initial clip. Try downloading instead.');
                    }}
                }});
                
            }} catch (err) {{
                console.error('Initial clip loading failed:', err);
                showError('Failed to load initial clip. Try downloading instead.');
            }}
        }}
        
        // Start full stream when user clicks play
        async function startFullStream() {{
            if (fullStreamStarted) return;
            
            try {{
                fullStreamStarted = true;
                showStatus('Starting full video stream...');
                
                // Set the full stream URL directly
                video.src = '{safe_stream_url}';
                video.load();
                
            }} catch (err) {{
                console.error('Failed to start full stream:', err);
                showError('Failed to start full stream. Try refreshing the page.');
            }}
        }}
        
        // Video event handlers
        video.addEventListener('loadstart', () => {{
            if (!fullStreamStarted) {{
                showStatus('Loading initial clip...');
            }} else {{
                showStatus('Loading full video stream...');
            }}
        }});
        
        video.addEventListener('loadedmetadata', () => {{
            if (!fullStreamStarted) {{
                showStatus('Initial clip loaded. Click play to start full stream.');
            }} else {{
                showStatus(`Full video loaded: ${{video.videoWidth}}x${{video.videoHeight}}, Duration: ${{Math.round(video.duration)}}s`);
            }}
        }});
        
        video.addEventListener('canplay', () => {{
            if (!fullStreamStarted) {{
                showStatus('Initial clip ready. Click play to start full stream.');
            }} else {{
                showStatus('Full video ready to play');
            }}
        }});
        
        video.addEventListener('playing', () => {{
            if (!fullStreamStarted) {{
                // User clicked play on initial clip, start full stream
                startFullStream();
            }} else {{
                showStatus('Full video is playing');
            }}
        }});
        
        video.addEventListener('pause', () => {{
            showStatus('Video paused');
        }});
        
        video.addEventListener('ended', () => {{
            if (!fullStreamStarted) {{
                showStatus('Initial clip ended. Click play to start full stream.');
            }} else {{
                showStatus('Full video ended');
            }}
        }});
        
        video.addEventListener('error', (e) => {{
            console.error('Video error:', e);
            console.error('Video error details:', video.error);
            
            let errorMessage = 'Failed to load video stream.';
            
            if (video.error) {{
                switch(video.error.code) {{
                    case 1:
                        errorMessage = 'Video loading aborted. The stream may be unavailable.';
                        break;
                    case 2:
                        errorMessage = 'Network error. Check your connection and try again.';
                        break;
                    case 3:
                        errorMessage = 'Video decoding failed. The stream format may be unsupported.';
                        break;
                    case 4:
                        errorMessage = 'Audio not supported. Try downloading instead.';
                        break;
                }}
            }}
            
            showError(errorMessage + '<br><br>Try refreshing the page or use the download links below.');
        }});
        
        // Load initial clip when page loads
        document.addEventListener('DOMContentLoaded', loadInitialClip);
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case ' ':
                    e.preventDefault();
                    if (video.paused) video.play();
                    else video.pause();
                    break;
                case 'ArrowRight':
                    video.currentTime += 10;
                    break;
                case 'ArrowLeft':
                    video.currentTime -= 10;
                    break;
                case 'ArrowUp':
                    video.volume = Math.min(1, video.volume + 0.1);
                    break;
                case 'ArrowDown':
                    video.volume = Math.max(0, video.volume - 0.1);
                    break;
            }}
        }});
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
        
        # Enhanced HTML5 audio player for streaming
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            background: #f5f5f5; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            min-height: 100vh; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .audio-container {{ 
            background: white; 
            border-radius: 16px; 
            padding: 40px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 500px;
            width: 90%;
        }}
        .audio-title {{
            font-size: 24px;
            font-weight: 600;
            color: #333;
            margin-bottom: 30px;
        }}
        audio {{ 
            width: 100%; 
            margin: 20px 0;
        }}
        .status {{
            color: #666;
            margin: 20px 0;
            font-size: 14px;
            min-height: 20px;
        }}
        .error {{
            color: #ff6b6b;
            background: rgba(255,107,107,0.1);
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid rgba(255,107,107,0.3);
        }}
        .download-link {{
            color: #4A9EFF;
            text-decoration: none;
            padding: 12px 24px;
            border: 2px solid #4A9EFF;
            border-radius: 8px;
            display: inline-block;
            margin: 15px 10px;
            transition: all 0.3s ease;
            font-weight: 500;
        }}
        .download-link:hover {{
            background: #4A9EFF;
            color: white;
            transform: translateY(-2px);
        }}
        .loading {{
            color: #4A9EFF;
            font-size: 16px;
            margin: 20px 0;
        }}
        .controls {{
            margin-top: 30px;
        }}
        .control-btn {{
            background: #f0f0f0;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .control-btn:hover {{
            background: #e0e0e0;
        }}
        .control-btn:active {{
            transform: scale(0.95);
        }}
    </style>
</head>
<body>
    <div class="audio-container">
        <div class="audio-title">🎵 Audio Player</div>
        
        <div id="loading" class="loading">Click play to start streaming</div>
        <div id="error" class="error" style="display: none;"></div>
        
        <audio id="audioPlayer" controls preload="none" crossorigin="anonymous">
            <p>Your browser doesn't support audio playback.</p>
        </audio>
        
        <div class="status" id="status">Click play button to start streaming</div>
        
        <div class="controls">
            <button class="control-btn" onclick="document.getElementById('audioPlayer').play()">▶️ Play</button>
            <button class="control-btn" onclick="document.getElementById('audioPlayer').pause()">⏸️ Pause</button>
            <button class="control-btn" onclick="document.getElementById('audioPlayer').currentTime = 0">⏮️ Restart</button>
            <button class="control-btn" onclick="document.getElementById('audioPlayer').volume = Math.max(0, document.getElementById('audioPlayer').volume - 0.1)">🔉 Vol-</button>
            <button class="control-btn" onclick="document.getElementById('audioPlayer').volume = Math.min(1, document.getElementById('audioPlayer').volume + 0.1)">🔊 Vol+</button>
        </div>
        
        <div style="margin-top: 30px;">
            <a href="{safe_stream_url}" class="download-link" target="_blank">📥 Download Audio</a>
            <a href="{safe_stream_url}" class="download-link" onclick="window.open('{safe_stream_url}', '_blank')">🔗 Open in New Tab</a>
        </div>
    </div>

    <script>
        const audio = document.getElementById('audioPlayer');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const status = document.getElementById('status');
        
        let initialClipLoaded = false;
        let fullStreamStarted = false;
        
        function showError(message) {{
            error.style.display = 'block';
            error.innerHTML = message;
            loading.style.display = 'none';
            status.textContent = 'Error loading audio';
        }}
        
        function showStatus(message) {{
            status.textContent = message;
        }}
        
        function hideLoading() {{
            loading.style.display = 'none';
        }}
        
        // Load initial clip (15-second preview)
        async function loadInitialClip() {{
            if (initialClipLoaded) return;
            
            try {{
                showStatus('Loading initial clip...');
                
                // Create a MediaSource for the initial clip
                const mediaSource = new MediaSource();
                const audioUrl = URL.createObjectURL(mediaSource);
                audio.src = audioUrl;
                
                mediaSource.addEventListener('sourceopen', async () => {{
                    try {{
                        // Fetch initial clip with range header for first 15 seconds
                        const response = await fetch('{safe_stream_url}', {{
                            headers: {{
                                'Range': 'bytes=0-'
                            }}
                        }});
                        
                        if (!response.ok) {{
                            throw new Error(`HTTP ${{response.status}}`);
                        }}
                        
                        const sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');
                        
                        sourceBuffer.addEventListener('updateend', () => {{
                            if (!sourceBuffer.updating) {{
                                mediaSource.endOfStream();
                                initialClipLoaded = true;
                                hideLoading();
                                showStatus('Initial clip loaded. Click play to start full stream.');
                            }}
                        }});
                        
                        const data = await response.arrayBuffer();
                        sourceBuffer.appendBuffer(data);
                        
                    }} catch (err) {{
                        console.error('Failed to load initial clip:', err);
                        showError('Failed to load initial clip. Try downloading instead.');
                    }}
                }});
                
            }} catch (err) {{
                console.error('Initial clip loading failed:', err);
                showError('Failed to load initial clip. Try downloading instead.');
            }}
        }}
        
        // Start full stream when user clicks play
        async function startFullStream() {{
            if (fullStreamStarted) return;
            
            try {{
                fullStreamStarted = true;
                showStatus('Starting full audio stream...');
                
                // Set the full stream URL directly
                audio.src = '{safe_stream_url}';
                audio.load();
                
            }} catch (err) {{
                console.error('Failed to start full stream:', err);
                showError('Failed to start full stream. Try refreshing the page.');
            }}
        }}
        
        // Audio event handlers
        audio.addEventListener('loadstart', () => {{
            if (!fullStreamStarted) {{
                showStatus('Loading initial clip...');
            }} else {{
                showStatus('Loading full audio stream...');
            }}
        }});
        
        audio.addEventListener('loadedmetadata', () => {{
            if (!fullStreamStarted) {{
                showStatus('Initial clip loaded. Click play to start full stream.');
            }} else {{
                showStatus(`Full audio loaded: Duration: ${{Math.round(audio.duration)}}s`);
            }}
        }});
        
        audio.addEventListener('canplay', () => {{
            if (!fullStreamStarted) {{
                showStatus('Initial clip ready. Click play to start full stream.');
            }} else {{
                showStatus('Full audio ready to play');
            }}
        }});
        
        audio.addEventListener('playing', () => {{
            if (!fullStreamStarted) {{
                // User clicked play on initial clip, start full stream
                startFullStream();
            }} else {{
                showStatus('Full audio is playing');
            }}
        }});
        
        audio.addEventListener('pause', () => {{
            showStatus('Audio paused');
        }});
        
        audio.addEventListener('ended', () => {{
            if (!fullStreamStarted) {{
                showStatus('Initial clip ended. Click play to start full stream.');
            }} else {{
                showStatus('Full audio ended');
            }}
        }});
        
        audio.addEventListener('error', (e) => {{
            console.error('Audio error:', e);
            console.error('Audio error details:', audio.error);
            
            let errorMessage = 'Failed to load audio stream.';
            
            if (audio.error) {{
                switch(audio.error.code) {{
                    case 1:
                        errorMessage = 'Audio loading aborted. The stream may be unavailable.';
                        break;
                    case 2:
                        errorMessage = 'Network error. Check your connection and try again.';
                        break;
                    case 3:
                        errorMessage = 'Audio decoding failed. The stream format may be unsupported.';
                        break;
                    case 4:
                        errorMessage = 'Video not supported. Try downloading instead.';
                        break;
                }}
            }}
            
            showError(errorMessage + '<br><br>Try refreshing the page or use the download links below.');
        }});
        
        // Load initial clip when page loads
        document.addEventListener('DOMContentLoaded', loadInitialClip);
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case ' ':
                    e.preventDefault();
                    if (audio.paused) audio.play();
                    else audio.pause();
                    break;
                case 'ArrowRight':
                    audio.currentTime += 10;
                    break;
                case 'ArrowLeft':
                    audio.currentTime -= 10;
                    break;
                case 'ArrowUp':
                    audio.volume = Math.min(1, audio.volume + 0.1);
                    break;
                case 'ArrowDown':
                    audio.volume = Math.max(0, audio.volume - 0.1);
                    break;
            }}
        }});
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
        log.error(f"Error serving audio player: {e}")
        raise HTTPException(status_code=500, detail="Failed to create audio player")
