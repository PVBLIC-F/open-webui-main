"""
Video Player Router - Serves HTML video player pages for authenticated streaming
"""

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
import urllib.parse
import logging

log = logging.getLogger(__name__)

router = APIRouter()

@router.get("/video", response_class=HTMLResponse)
async def video_player(
    request: Request,
    url: str = Query(..., description="The video stream URL to play"),
    title: str = Query("Video Player", description="Title for the video")
):
    """
    Serves an HTML page with a video player for the given stream URL.
    This allows iframes to properly display video content.
    """
    try:
        # Use the URL as-is since it should already be properly formatted
        decoded_url = url
        
        # Create a simple redirect page that opens the stream directly
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
            padding: 0;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: white;
        }}
        
        .container {{
            text-align: center;
            max-width: 500px;
            padding: 40px;
        }}
        
        .video-icon {{
            font-size: 80px;
            margin-bottom: 20px;
            opacity: 0.8;
        }}
        
        .title {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #fff;
        }}
        
        .subtitle {{
            font-size: 16px;
            color: #ccc;
            margin-bottom: 30px;
            line-height: 1.5;
        }}
        
        .play-button {{
            display: inline-block;
            padding: 15px 30px;
            background: #4A9EFF;
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-size: 18px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 10px;
        }}
        
        .play-button:hover {{
            background: #357ABD;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(74, 158, 255, 0.4);
        }}
        
        .secondary-button {{
            display: inline-block;
            padding: 12px 25px;
            background: transparent;
            color: #ccc;
            text-decoration: none;
            border: 2px solid #555;
            border-radius: 20px;
            font-size: 16px;
            transition: all 0.3s ease;
            margin: 10px;
        }}
        
        .secondary-button:hover {{
            border-color: #777;
            color: #fff;
        }}
        
        .info {{
            margin-top: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            font-size: 14px;
            color: #bbb;
        }}
        
        .auto-redirect {{
            margin-top: 20px;
            font-size: 14px;
            color: #888;
        }}
        
        .countdown {{
            font-weight: bold;
            color: #4A9EFF;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="video-icon">🎬</div>
        <div class="title">{title}</div>
        <div class="subtitle">
            This video requires authenticated streaming.<br>
            Click below to open the video stream directly.
        </div>
        
        <a href="{decoded_url}" class="play-button" target="_blank">
            ▶ Play Video Stream
        </a>
        
        <br>
        
        <a href="{decoded_url}" class="secondary-button" target="_self">
            🔗 Open in Same Tab
        </a>
        
        <div class="info">
            <strong>Why this approach?</strong><br>
            Authenticated video streams work best when opened directly in the browser,
            bypassing iframe security restrictions that prevent embedded playback.
        </div>
        
        <div class="auto-redirect">
            Auto-redirecting in <span class="countdown" id="countdown">5</span> seconds...
        </div>
    </div>

    <script>
        // Auto-redirect after 5 seconds
        let countdown = 5;
        const countdownElement = document.getElementById('countdown');
        
        const timer = setInterval(() => {{
            countdown--;
            countdownElement.textContent = countdown;
            
            if (countdown <= 0) {{
                clearInterval(timer);
                window.location.href = '{decoded_url}';
            }}
        }}, 1000);
        
        // Allow user to cancel auto-redirect by interacting with the page
        document.addEventListener('click', () => {{
            clearInterval(timer);
            document.querySelector('.auto-redirect').style.display = 'none';
        }});
        
        document.addEventListener('keydown', () => {{
            clearInterval(timer);
            document.querySelector('.auto-redirect').style.display = 'none';
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
