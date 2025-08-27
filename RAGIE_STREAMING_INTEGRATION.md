# Ragie API Streaming Integration Guide

## Overview

This document outlines the complete integration path for streaming audio and video content from Ragie's API, following the official API structure documented at [docs.ragie.ai](https://docs.ragie.ai/docs/changes-to-the-api).

## 🎯 **Key Implementation: Chunk-Level Streaming**

Based on Ragie's official documentation, our implementation prioritizes **chunk-level streaming** for exact matches, providing users with precise video/audio segments that correspond to their search queries.

## 📋 **API Response Structure**

### Ragie Retrieval Response Format

```json
{
  "scored_chunks": [
    {
      "text": "This is the text of a chunk",
      "score": 0.19090909090909092,
      "id": "chunkId",
      "index": 2,
      "metadata": {
        "end_time": 167.5,
        "start_time": 115.37
      },
      "document_id": "docId",
      "document_name": "presentation.mp4",
      "document_metadata": {},
      "links": {
        "self": {
          "href": "https://api.ragie.ai/documents/docId/chunks/chunkId",
          "type": "application/json"
        },
        "self_text": {
          "href": "https://api.ragie.ai/documents/docId/chunks/chunkId/content?media_type=text/plain-text",
          "type": "text/plain-text"
        },
        "self_video_stream": {
          "href": "https://api.ragie.ai/documents/docId/chunks/chunkId/content?media_type=video/mp4",
          "type": "video/mp4"
        },
        "self_video_download": {
          "href": "https://api.ragie.ai/documents/docId/chunks/chunkId/content?media_type=video/mp4&download=true",
          "type": "video/mp4"
        },
        "document_video_stream": {
          "href": "https://api.ragie.ai/documents/docId/content?media_type=video/mp4",
          "type": "video/mp4"
        },
        "document_video_download": {
          "href": "https://api.ragie.ai/documents/docId/content?media_type=video/mp4&download=true",
          "type": "video/mp4"
        }
      }
    }
  ]
}
```

## 🔄 **Integration Flow**

### 1. **Query Processing**
```typescript
// User submits query
const query = "Secretary-General Opening Message";

// Ragie retrieval request
const response = await fetch('https://api.ragie.ai/retrievals', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: query,
    top_k: 5,
    include_metadata: true,
    partition: "default"
  })
});
```

### 2. **URL Prioritization Logic**

Our implementation follows this priority order:

1. **Primary**: `self_video_stream` / `self_audio_stream` (chunk-level)
2. **Fallback**: `document_video_stream` / `document_audio_stream` (document-level)
3. **Legacy**: Metadata URLs (`video_url`, `audio_url`)

```typescript
function createVideoPlayerUrl(videoStreamUrl: string, metadata?: any, documentContent?: string, links?: any) {
  // Prefer chunk-level streaming URL from links if available
  const chunkVideoUrl = links?.self_video_stream?.href;
  const streamUrl = chunkVideoUrl || videoStreamUrl;
  
  if (!streamUrl) return null;
  
  const encodedUrl = encodeURIComponent(streamUrl);
  let playerUrl = `/api/player/video?stream=${encodedUrl}`;
  
  // Add chunk metadata if available (from Ragie API structure)
  if (metadata?.start_time !== undefined && metadata?.end_time !== undefined) {
    playerUrl += `&start_time=${metadata.start_time}&end_time=${metadata.end_time}`;
    
    if (documentContent) {
      const cleanText = documentContent.replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/\*([^*]+)\*/g, '$1')
        .replace(/\\u[\da-f]{4}/gi, '')
        .replace(/\s+/g, ' ')
        .trim();
      playerUrl += `&chunk_text=${encodeURIComponent(cleanText)}`;
    }
  }
  
  return playerUrl;
}
```

### 3. **Proxy Integration**

All Ragie streaming URLs are routed through our proxy to handle authentication:

```
Original Ragie URL:
https://api.ragie.ai/documents/docId/chunks/chunkId/content?media_type=video/mp4

Proxied URL:
/api/proxy/ragie/stream?url=https%3A%2F%2Fapi.ragie.ai%2Fdocuments%2FdocId%2Fchunks%2FchunkId%2Fcontent%3Fmedia_type%3Dvideo%2Fmp4
```

### 4. **Video Player Implementation**

The video player receives enhanced metadata:

```html
<!-- Video Player URL with chunk metadata -->
/api/player/video?stream=ENCODED_URL&start_time=115.37&end_time=167.5&chunk_text=ENCODED_TEXT

<!-- Player displays chunk information -->
<div class="chunk-info">
  <h3>📹 Video Segment</h3>
  <p><strong>Duration:</strong> 52.1 seconds</p>
  <p><strong>Timing:</strong> 115.4s - 167.5s</p>
  <p><strong>Content:</strong> Secretary-General Opening Message excerpt...</p>
</div>
```

## 🎬 **Video Player Features**

### Chunk-Specific Display
- **Segment Duration**: Calculated from `end_time - start_time`
- **Timing Information**: Shows exact position in original video
- **Content Preview**: Displays chunk text content
- **Professional UI**: Dark theme with proper controls

### JavaScript Implementation
```javascript
// Enhanced metadata handling
const chunkInfo = {chunk_info_json};
if (chunkInfo) {
  showStatus(`Chunk loaded: ${chunkInfo.duration.toFixed(1)}s segment (${video.videoWidth}x${video.videoHeight})`);
} else {
  showStatus(`Video loaded: ${video.videoWidth}x${video.videoHeight}, Duration: ${Math.round(video.duration)}s`);
}
```

## 🎵 **Audio Player Features**

Similar to video, but optimized for audio content:
- **Compact UI**: Smaller footprint for audio-only content
- **Transcript Display**: Shows audio transcript with timing
- **Playback Controls**: Play, pause, restart functionality

## 🔗 **URL Structure Examples**

### Chunk-Level Streaming (Preferred)
```
Video: https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/chunks/c0a7dd52-5de8-4eff-a0cc-0d150ffc2b4e/content?media_type=video/mp4

Audio: https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/chunks/c0a7dd52-5de8-4eff-a0cc-0d150ffc2b4e/content?media_type=audio/mpeg
```

### Document-Level Streaming (Fallback)
```
Video: https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/content?media_type=video/mp4

Audio: https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/content?media_type=audio/mpeg
```

## 🛡️ **Security & Authentication**

### Proxy Authentication
```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "*/*",
    "User-Agent": "OpenWebUI-Ragie-Proxy/1.0"
}
```

### Security Headers
```python
response_headers = {
    "Content-Security-Policy": "default-src 'self'; media-src 'self' data: blob:",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN"
}
```

## 📊 **Request Flow Example**

### Complete Request Chain
```
1. User Query → Ragie Retrieval API
   POST https://api.ragie.ai/retrievals
   
2. Extract Links → Create Player URL
   self_video_stream.href → /api/player/video?stream=...
   
3. Player Requests → Proxy Stream
   GET /api/player/video → GET /api/proxy/ragie/stream
   
4. Proxy Forwards → Ragie Content API
   GET /api/proxy/ragie/stream → GET https://api.ragie.ai/documents/.../content
   
5. Stream Response → Browser Player
   206 Partial Content → HTML5 Video Element
```

## 🔧 **Implementation Checklist**

### ✅ **Completed Features**
- [x] Chunk-level URL prioritization
- [x] Metadata extraction (start_time, end_time)
- [x] Enhanced player UI with chunk information
- [x] Proxy authentication handling
- [x] Fallback to document-level streaming
- [x] Professional video/audio player styling
- [x] Debug logging and error handling

### 🎯 **Key Benefits**

1. **Precise Streaming**: Users get exact video/audio segments matching their queries
2. **BaseChat Compatibility**: Follows Ragie's official BaseChat implementation patterns
3. **Robust Fallbacks**: Multiple URL sources ensure reliability
4. **Professional UI**: Clean, informative player interface
5. **Security**: Proper authentication and content security policies

## 📝 **Testing Scenarios**

### Test Cases
1. **Chunk Streaming**: Verify `self_video_stream` URLs work correctly
2. **Fallback Logic**: Test document-level streaming when chunk URLs fail
3. **Metadata Display**: Confirm timing and content information appears
4. **Authentication**: Ensure proxy properly forwards API keys
5. **Error Handling**: Test behavior with invalid or missing URLs

## 🚀 **Next Steps**

1. **Performance Optimization**: Implement video preloading strategies
2. **Enhanced Navigation**: Add chapter/segment navigation
3. **Accessibility**: Improve screen reader support
4. **Analytics**: Track streaming usage and performance metrics

---

*This integration follows Ragie's official API documentation and BaseChat implementation patterns for optimal compatibility and user experience.*
