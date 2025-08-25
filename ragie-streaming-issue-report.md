# Ragie API Streaming URLs Issue Report

## Summary
I'm implementing audio/video streaming functionality using Ragie's retrieval API, but the streaming URLs provided in your API responses are returning 404 "Document not found" errors when I try to access them.

## What I'm Building

### Implementation Details
1. **Querying your `/retrievals` endpoint** for video content (e.g., "Secretary-General Opening Message.mp4")
2. **Extracting streaming URLs** from the `links` section of chunk responses
3. **Using your provided URLs** exactly as returned by your API
4. **Routing through a proxy** with proper authentication headers (`Authorization: Bearer <API_KEY>`)

### My API Response Analysis
Your API correctly returns streaming links in the chunk responses:

```json
{
  "links": {
    "self_audio_stream": {"href": "https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/chunks/c0a7dd52-5de8-4eff-a0cc-0d150ffc2b4e/content?media_type=audio/mpeg"},
    "self_video_stream": {"href": "https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/chunks/c0a7dd52-5de8-4eff-a0cc-0d150ffc2b4e/content?media_type=video/mp4"},
    "document_audio_stream": {"href": "https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/content?media_type=audio/mpeg"},
    "document_video_stream": {"href": "https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/content?media_type=video/mp4"}
  }
}
```

## The Problem I'm Experiencing

### 404 Errors on All Streaming URLs
When I make GET requests to **any** of the streaming URLs you provide, I consistently receive:

```
HTTP/1.1 404 Not Found
{"detail": "Document not found"}
```

### URLs I've Tested
- **Chunk-level**: `https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/chunks/c0a7dd52-5de8-4eff-a0cc-0d150ffc2b4e/content?media_type=video/mp4`
- **Document-level**: `https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/content?media_type=video/mp4`

### Authentication is Working
- ✅ Same API key works perfectly for your `/retrievals` endpoint
- ✅ I'm including proper `Authorization: Bearer <API_KEY>` header
- ✅ Test call to `https://api.ragie.ai/partitions` succeeds with same credentials

## My Technical Investigation

### What I've Tried
1. **URL Encoding**: Tested with and without URL encoding - no difference
2. **HTTP Methods**: Tried GET and HEAD requests - both return 404
3. **Document ID Verification**: Used exact document IDs from your API responses
4. **Chunk ID Verification**: Used exact chunk IDs from your API responses
5. **Manual URL Construction**: Built URLs manually vs. using your provided URLs - same result

### The Pattern I'm Seeing
- ✅ **Your Retrieval API works perfectly** - returns chunks with streaming links
- ❌ **All streaming URLs return 404** - regardless of format or construction method
- ✅ **My authentication is valid** - works for other endpoints

## Example from My Logs
```
2025-08-25 22:28:47.246 | INFO | Available links: ['self_audio_stream', 'document_audio_stream', 'self_video_stream', 'document_video_stream']

2025-08-25 22:28:47.246 | INFO | document_video_stream: {'href': 'https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/content?media_type=video/mp4', 'type': 'video/mp4'}

2025-08-25 22:27:12.739 | INFO | HTTP Request: GET https://api.ragie.ai/documents/ac0e7711-6c1f-4743-94d8-a2fb76de2583/content?media_type=video/mp4 "HTTP/1.1 404 Not Found"

2025-08-25 22:27:12.740 | ERROR | Ragie stream failed: 404 - {"detail":"Document not found"}
```

## What I Need Help With

### My Questions
1. **Is there a known issue** with streaming URLs for video/audio content?
2. **Do I need additional permissions** for streaming endpoints beyond what works for retrieval?
3. **Is the document ID format different** for streaming vs. retrieval endpoints?
4. **Are there rate limits** or other restrictions on streaming endpoints I should know about?

### Document Information I'm Working With
- **Document Name**: "Secretary-General Opening Message.mp4"
- **Document ID**: `ac0e7711-6c1f-4743-94d8-a2fb76de2583`
- **Source**: Google Drive integration
- **Content Type**: Video file (MP4)

### What I Expected vs. What I'm Getting
**Expected**: The streaming URLs provided in your API responses should return the actual media content (audio/video streams) when I access them with proper authentication.

**Actual**: All streaming URLs return 404 "Document not found" errors.

### My Current Workaround
I've implemented a fallback mechanism using the original Google Drive URLs, but I'd prefer to use your streaming infrastructure for better performance and user experience.

---

**I'm happy to provide additional debugging information, logs, or API call examples if that would help troubleshoot this issue.**
