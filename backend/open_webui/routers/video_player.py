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
        
        # Create HTML page with video player
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
        }}
        
        .video-container {{
            width: 100%;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}
        
        video {{
            width: 100%;
            height: auto;
            max-height: 100vh;
            background: #000;
        }}
        
        .error-message {{
            color: #fff;
            text-align: center;
            padding: 20px;
        }}
        
        .loading {{
            color: #fff;
            text-align: center;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="video-container">
        <div id="loading" class="loading">Loading video...</div>
        <video 
            id="videoPlayer"
            controls 
            preload="metadata"
            style="display: none;"
            onloadstart="hideLoading()"
            oncanplay="showVideo()"
            onerror="showError()"
        >
            <source src="{decoded_url}" type="video/mp4">
            <p class="error-message">Your browser doesn't support HTML5 video.</p>
        </video>
        <div id="error" class="error-message" style="display: none;">
            Failed to load video. The stream may have expired or be unavailable.
        </div>
    </div>

    <script>
        function hideLoading() {{
            document.getElementById('loading').style.display = 'none';
        }}
        
        function showVideo() {{
            document.getElementById('videoPlayer').style.display = 'block';
        }}
        
        function showError() {{
            document.getElementById('loading').style.display = 'none';
            document.getElementById('videoPlayer').style.display = 'none';
            document.getElementById('error').style.display = 'block';
        }}
        
        // Auto-hide loading after 10 seconds if video hasn't loaded
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
