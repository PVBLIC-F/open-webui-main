"""
Video Player Router - Serves HTML video player pages for authenticated streaming
"""

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
import urllib.parse
import logging
import httpx

log = logging.getLogger(__name__)

router = APIRouter()

@router.get("/video/stream")
async def video_stream_proxy(
    request: Request,
    url: str = Query(..., description="The video stream URL to proxy")
):
    """
    Directly proxy the video stream with proper headers for browser playback.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Prepare headers for the upstream request
            upstream_headers = {
                "User-Agent": "Mozilla/5.0 (compatible; OpenWebUI/1.0)",
                "Accept": "video/mp4,video/*,*/*",
            }
            
            # Forward Range header if present (for video seeking)
            if "Range" in request.headers:
                upstream_headers["Range"] = request.headers["Range"]
            
            # Make request to the actual stream URL
            response = await client.get(url, headers=upstream_headers)
            
            # Prepare response headers
            response_headers = {
                "Content-Type": response.headers.get("Content-Type", "video/mp4"),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "Range, Content-Type",
            }
            
            # Forward important headers from upstream
            for header in ["Content-Length", "Content-Range", "Last-Modified", "ETag"]:
                if header in response.headers:
                    response_headers[header] = response.headers[header]
            
            # Return streaming response
            return StreamingResponse(
                response.aiter_bytes(chunk_size=8192),
                status_code=response.status_code,
                headers=response_headers
            )
    except Exception as e:
        log.error(f"Error proxying video stream: {e}")
        raise HTTPException(status_code=500, detail="Failed to proxy video stream")


@router.get("/video", response_class=HTMLResponse)
async def video_player(
    request: Request,
    url: str = Query(..., description="The video stream URL to play"),
    title: str = Query("Video Player", description="Title for the video")
):
    """
    Serves an HTML page with a video player that uses our stream proxy.
    """
    try:
        # Create a proxy URL for the stream
        proxy_url = f"/api/player/video/stream?url={urllib.parse.quote(url, safe='')}"
        
        # Create HTML page with Video.js player for better streaming support
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://vjs.zencdn.net/8.6.1/video-js.css" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #000; }}
        .video-js {{ 
            width: 100%; 
            height: 100vh;
        }}
        .vjs-big-play-button {{
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }}
    </style>
</head>
<body>
    <video
        id="video-player"
        class="video-js vjs-default-skin"
        controls
        preload="auto"
        width="100%"
        height="100%"
        playsinline
        data-setup="{{}}">
        <source src="{proxy_url}" type="video/mp4">
        <p class="vjs-no-js">
            To view this video please enable JavaScript, and consider upgrading to a web browser that
            <a href="https://videojs.com/html5-video-support/" target="_blank">supports HTML5 video</a>.
        </p>
    </video>

    <script src="https://vjs.zencdn.net/8.6.1/video.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            var player = videojs('video-player', {{
                // Responsive design
                fluid: true,
                responsive: true,
                
                // Playback controls
                controls: true,
                playbackRates: [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
                
                // Loading behavior - 'metadata' is recommended for better performance
                preload: 'metadata',
                
                // Autoplay settings - 'muted' is most likely to work
                autoplay: false,
                muted: false,
                
                // Mobile optimization
                playsinline: true,
                
                // User interaction settings
                userActions: {{
                    click: true,
                    doubleClick: true,
                    hotkeys: {{
                        fullscreenKey: function(event) {{
                            return (event.which === 70); // 'f' key
                        }},
                        muteKey: function(event) {{
                            return (event.which === 77); // 'm' key
                        }},
                        playPauseKey: function(event) {{
                            return (event.which === 32 || event.which === 75); // space or 'k' key
                        }}
                    }}
                }},
                
                // Error handling
                suppressNotSupportedError: false,
                
                // Sources configuration
                sources: [{{
                    src: '{proxy_url}',
                    type: 'video/mp4'
                }}],
                
                // HTML5 tech specific options
                html5: {{
                    nativeControlsForTouch: false,
                    preloadTextTracks: false
                }},
                
                // Breakpoints for responsive design
                breakpoints: {{
                    tiny: 300,
                    xsmall: 400,
                    small: 500,
                    medium: 600,
                    large: 700,
                    xlarge: 800,
                    huge: 900
                }},
                
                // Inactivity timeout (in milliseconds)
                inactivityTimeout: 2000
            }});
            
            // Event handlers following best practices
            player.ready(function() {{
                console.log('Video.js player ready');
                console.log('Stream URL:', '{proxy_url}');
                
                // Set up additional error recovery
                player.on('error', function() {{
                    var error = player.error();
                    console.error('Video.js error:', error);
                    
                    // Attempt to recover from certain errors
                    if (error && error.code === 4) {{
                        console.log('Attempting to reload video source...');
                        setTimeout(function() {{
                            player.src({{
                                src: '{proxy_url}',
                                type: 'video/mp4'
                            }});
                            player.load();
                        }}, 1000);
                    }}
                }});
            }});
            
            // Loading and playback event tracking
            player.on('loadstart', function() {{
                console.log('Video load started');
            }});
            
            player.on('loadedmetadata', function() {{
                console.log('Video metadata loaded');
            }});
            
            player.on('canplay', function() {{
                console.log('Video can play');
            }});
            
            player.on('canplaythrough', function() {{
                console.log('Video can play through');
            }});
            
            player.on('waiting', function() {{
                console.log('Video waiting for data');
            }});
            
            player.on('playing', function() {{
                console.log('Video playing');
            }});
            
            // Handle fullscreen changes
            player.on('fullscreenchange', function() {{
                console.log('Fullscreen changed:', player.isFullscreen());
            }});
        }});
    </script>
</body>
</html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        log.error(f"Error serving video player: {e}")
        raise HTTPException(status_code=500, detail="Failed to create video player")


@router.get("/audio", response_class=HTMLResponse)
async def audio_player(
    request: Request,
    url: str = Query(..., description="The audio stream URL to play"),
    title: str = Query("Audio Player", description="Title for the audio")
):
    """
    Serves an HTML page with an audio player for the given stream URL.
    """
    try:
        # Use the URL as-is since it should already be properly formatted
        decoded_url = url
        
        # Create HTML page with audio player
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        
        .audio-container {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 500px;
            width: 100%;
        }}
        
        .audio-title {{
            margin-bottom: 20px;
            color: #333;
            font-size: 18px;
            font-weight: 600;
        }}
        
        audio {{
            width: 100%;
            margin: 20px 0;
        }}
        
        .error-message {{
            color: #e74c3c;
            margin-top: 20px;
        }}
        
        .loading {{
            color: #666;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="audio-container">
        <div class="audio-title">{title}</div>
        <div id="loading" class="loading">Loading audio...</div>
        <audio 
            id="audioPlayer"
            controls 
            preload="metadata"
            style="display: none;"
            onloadstart="hideLoading()"
            oncanplay="showAudio()"
            onerror="showError()"
        >
            <source src="{decoded_url}" type="audio/mpeg">
            <source src="{decoded_url}" type="audio/mp4">
            <p class="error-message">Your browser doesn't support HTML5 audio.</p>
        </audio>
        <div id="error" class="error-message" style="display: none;">
            Failed to load audio. The stream may have expired or be unavailable.
        </div>
    </div>

    <script>
        function hideLoading() {{
            document.getElementById('loading').style.display = 'none';
        }}
        
        function showAudio() {{
            document.getElementById('audioPlayer').style.display = 'block';
        }}
        
        function showError() {{
            document.getElementById('loading').style.display = 'none';
            document.getElementById('audioPlayer').style.display = 'none';
            document.getElementById('error').style.display = 'block';
        }}
        
        // Auto-hide loading after 10 seconds if audio hasn't loaded
        setTimeout(function() {{
            if (document.getElementById('loading').style.display !== 'none') {{
                showError();
            }}
        }}, 10000);
    </script>
</body>
</html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        log.error(f"Error serving audio player: {e}")
        raise HTTPException(status_code=500, detail="Failed to create audio player")
