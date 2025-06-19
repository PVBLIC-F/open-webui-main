# OpenWebUI with Google Drive Cloud Sync

This is a custom fork of OpenWebUI that adds **Google Drive Cloud Sync** functionality to Knowledge Bases. This feature allows you to automatically sync files from Google Drive folders into your OpenWebUI Knowledge Bases, with real-time notifications and automatic updates when files are added, modified, or removed.

## 🌟 Features

### **Google Drive Integration**
- **Folder Sync**: Connect Google Drive folders to Knowledge Bases
- **Real-time Sync**: Automatic background synchronization every 30 seconds
- **File Type Support**: PDF, Word docs, spreadsheets, presentations, text files, markdown
- **Google Workspace Conversion**: Automatically converts Google Docs/Sheets/Slides to standard formats
- **Bidirectional Sync**: Handles both file additions and deletions

### **User Experience**
- **OAuth Authentication**: Secure Google Drive access with refresh tokens
- **Folder Picker**: Visual folder selection interface with subfolder support
- **Real-time Notifications**: Live updates when files are processed
- **Progress Tracking**: Visual feedback during sync operations
- **Auto-refresh**: Knowledge Base automatically updates when sync completes

### **Technical Features**
- **Background Processing**: Non-blocking sync operations
- **Conflict Resolution**: Handles duplicate files and hash conflicts
- **Error Handling**: Comprehensive error reporting and retry logic
- **Database Integration**: Full PostgreSQL support with proper schema
- **Socket Notifications**: Real-time UI updates via WebSocket

## 🚀 Quick Start

### Prerequisites
- OpenWebUI installation (compatible with latest version)
- Google Cloud Project with Drive API enabled
- PostgreSQL database (required for cloud sync tables)

### 1. Google Cloud Setup

1. **Create Google Cloud Project**
   ```bash
   # Go to https://console.cloud.google.com/
   # Create new project or select existing one
   ```

2. **Enable Google Drive API**
   ```bash
   # In Google Cloud Console:
   # APIs & Services > Library > Search "Google Drive API" > Enable
   ```

3. **Create OAuth 2.0 Credentials**
   ```bash
   # APIs & Services > Credentials > Create Credentials > OAuth 2.0 Client ID
   # Application type: Web application
   # Authorized redirect URIs: http://localhost:8080/oauth/google-drive/callback
   # (Replace localhost:8080 with your domain)
   ```

4. **Get Client ID and Secret**
   ```bash
   # Download the JSON credentials file
   # Note the client_id and client_secret values
   ```

### 2. Environment Configuration

Add these environment variables to your OpenWebUI installation:

```bash
# Google Drive Integration
ENABLE_GOOGLE_DRIVE_INTEGRATION=true
GOOGLE_DRIVE_CLIENT_ID=your_client_id_here
GOOGLE_DRIVE_CLIENT_SECRET=your_client_secret_here

# Required for cloud sync
DATABASE_URL=postgresql://user:password@host:port/database
```

### 3. Installation

#### Option A: Direct Installation (Recommended)

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/open-webui-google-drive-sync.git
   cd open-webui-google-drive-sync
   ```

2. **Install dependencies**
   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt

   # Frontend
   cd ../
   npm install
   ```

3. **Run the application**
   ```bash
   # Development mode
   npm run dev

   # Production mode
   npm run build
   npm run start
   ```

#### Option B: Add to Existing OpenWebUI

1. **Copy the files from this repository to your existing OpenWebUI installation:**

   **Backend Files:**
   ```bash
   # Core sync functionality
   backend/open_webui/models/cloud_sync.py
   backend/open_webui/models/cloud_tokens.py
   backend/open_webui/routers/cloud_sync.py
   backend/open_webui/routers/oauth.py
   backend/open_webui/sync/providers.py
   backend/open_webui/sync/worker.py
   backend/open_webui/utils/cloud_sync_utils.py
   backend/open_webui/utils/knowledge.py

   # Modified existing files
   backend/open_webui/config.py
   backend/open_webui/main.py
   backend/open_webui/socket/main.py
   backend/open_webui/routers/knowledge.py
   backend/open_webui/routers/users.py
   backend/open_webui/utils/oauth.py
   ```

   **Frontend Files:**
   ```bash
   # New components
   src/lib/components/drive/GoogleDrivePicker.svelte
   src/lib/components/workspace/Knowledge/KnowledgeBase/FilesWithFolders.svelte
   src/lib/components/workspace/Knowledge/KnowledgeBase/FolderCard.svelte
   src/lib/components/workspace/Knowledge/KnowledgeBase/GroupedFiles.svelte
   src/lib/utils/config.ts
   src/lib/utils/google-drive-api.ts

   # Modified existing files
   src/lib/components/workspace/Knowledge/KnowledgeBase.svelte
   src/lib/components/workspace/Knowledge/KnowledgeBase/AddContentMenu.svelte
   src/lib/constants.ts
   src/lib/utils/google-drive-picker.ts
   ```

2. **Install additional Python dependencies:**
   ```bash
   pip install aiohttp
   ```

3. **Create local vite.config.ts (for development):**
   ```typescript
   import { sveltekit } from '@sveltejs/kit/vite';
   import { defineConfig } from 'vite';

   export default defineConfig({
     plugins: [sveltekit()],
     server: {
       proxy: {
         '/api': {
           target: 'http://localhost:8080',
           changeOrigin: true
         },
         '/oauth': {
           target: 'http://localhost:8080',
           changeOrigin: true
         }
       }
     }
   });
   ```

## 📖 Usage Guide

### 1. First-Time Setup

1. **Access Knowledge Base**
   - Go to Workspace > Knowledge
   - Create or select a Knowledge Base

2. **Connect Google Drive**
   - Click "Add Content" > "Google Drive"
   - Authenticate with your Google account
   - Grant necessary permissions

3. **Select Folders**
   - Use the folder picker to select Google Drive folders
   - You can select multiple folders and subfolders
   - Click "Save" to start syncing

### 2. Managing Sync

**View Connected Folders:**
- Connected folders appear as cards in the Knowledge Base
- Each card shows folder name and file count

**Sync Status:**
- Real-time notifications show sync progress
- Blue notifications indicate processing
- Green notifications indicate completion
- Red notifications indicate errors

**Manual Sync:**
- Sync happens automatically every 30 seconds
- Manual sync triggers when connecting new folders

### 3. File Management

**Supported File Types:**
- PDF documents
- Microsoft Office files (Word, Excel, PowerPoint)
- Google Workspace files (Docs, Sheets, Slides)
- Text files (.txt, .md)

**File Processing:**
- Files are automatically converted to searchable text
- Content is indexed for semantic search
- Metadata is preserved (filename, size, modification date)

**File Updates:**
- Modified files are automatically re-synced
- Deleted files are removed from Knowledge Base
- New files are added automatically

## 🔧 Configuration

### Environment Variables

```bash
# Google Drive Integration (Required)
ENABLE_GOOGLE_DRIVE_INTEGRATION=true
GOOGLE_DRIVE_CLIENT_ID=your_google_client_id
GOOGLE_DRIVE_CLIENT_SECRET=your_google_client_secret

# Sync Settings (Optional)
CLOUD_SYNC_INTERVAL=30  # Sync interval in seconds (default: 30)
CLOUD_SYNC_BATCH_SIZE=10  # Files to process per batch (default: 10)

# Database (Required for cloud sync)
DATABASE_URL=postgresql://user:password@host:port/database

# OAuth Redirect URL (Optional - auto-detected)
OAUTH_REDIRECT_URL=https://yourdomain.com/oauth/google-drive/callback
```

### Database Schema

The following tables are automatically created:

```sql
-- OAuth tokens storage
CREATE TABLE cloud_tokens (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    access_token VARCHAR NOT NULL,
    refresh_token VARCHAR,
    expires_at INTEGER,
    created_at TIMESTAMP,
    UNIQUE(user_id, provider)
);

-- Folder connections
CREATE TABLE cloud_sync_folder (
    id VARCHAR PRIMARY KEY,
    knowledge_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    cloud_folder_id VARCHAR NOT NULL,
    folder_name VARCHAR,
    last_successful_sync BIGINT,
    last_sync_attempt BIGINT,
    sync_enabled BOOLEAN DEFAULT true,
    folder_token VARCHAR,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

-- File sync tracking
CREATE TABLE cloud_sync_file (
    id VARCHAR PRIMARY KEY,
    knowledge_id VARCHAR NOT NULL,
    file_id VARCHAR,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    cloud_file_id VARCHAR NOT NULL,
    cloud_folder_id VARCHAR NOT NULL,
    cloud_folder_name VARCHAR,
    filename VARCHAR NOT NULL,
    mime_type VARCHAR,
    file_size INTEGER,
    cloud_modified_time VARCHAR,
    sync_status VARCHAR NOT NULL DEFAULT 'pending',
    last_sync_time BIGINT,
    meta TEXT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    content_hash VARCHAR,
    cloud_version VARCHAR,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    next_retry_at BIGINT
);

-- Sync operations tracking
CREATE TABLE cloud_sync_operation (
    id VARCHAR PRIMARY KEY,
    knowledge_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    operation_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'pending',
    started_at BIGINT,
    finished_at BIGINT,
    duration_ms INTEGER,
    files_processed INTEGER DEFAULT 0,
    files_skipped INTEGER DEFAULT 0,
    files_failed INTEGER DEFAULT 0,
    stats_json TEXT,
    data TEXT,
    meta TEXT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);
```

## 🛠️ Development

### Project Structure

```
backend/
├── open_webui/
│   ├── models/
│   │   ├── cloud_sync.py          # Database models for sync
│   │   └── cloud_tokens.py        # OAuth token management
│   ├── routers/
│   │   ├── cloud_sync.py          # Sync API endpoints
│   │   └── oauth.py               # OAuth endpoints
│   ├── sync/
│   │   ├── providers.py           # Google Drive sync logic
│   │   └── worker.py              # Background sync worker
│   └── utils/
│       ├── cloud_sync_utils.py    # Utility functions
│       └── knowledge.py           # Knowledge Base utilities

src/lib/
├── components/
│   ├── drive/
│   │   └── GoogleDrivePicker.svelte    # Folder selection UI
│   └── workspace/Knowledge/KnowledgeBase/
│       ├── AddContentMenu.svelte       # Google Drive option
│       ├── FilesWithFolders.svelte     # File display with folders
│       ├── FolderCard.svelte           # Folder card component
│       └── GroupedFiles.svelte         # File grouping component
└── utils/
    ├── google-drive-api.ts             # Google Drive API client
    └── config.ts                       # Configuration utilities
```

### API Endpoints

**OAuth Endpoints:**
- `GET /oauth/google-drive/login` - Initiate OAuth flow
- `GET /oauth/google-drive/callback` - OAuth callback handler
- `GET /api/oauth/google-drive/status` - Check token status

**Sync Endpoints:**
- `POST /api/v1/cloud-sync/connect-folders` - Connect folders to KB
- `POST /api/v1/cloud-sync/disconnect-folders` - Disconnect folders
- `GET /api/v1/cloud-sync/connected-folders/{kb_id}` - List connected folders
- `GET /api/v1/cloud-sync/status/{kb_id}` - Get sync status
- `POST /api/v1/cloud-sync/manual-sync/{kb_id}` - Trigger manual sync

### Socket Events

**Sync Notifications:**
```javascript
// File processing events
socket.on('sync_notification', (data) => {
  switch(data.event_type) {
    case 'file_processing':
      // File is being processed
      break;
    case 'file_completed':
      // File processing completed
      break;
    case 'file_error':
      // File processing failed
      break;
    case 'sync_completed':
      // Entire sync operation completed
      break;
    case 'sync_error':
      // Sync operation failed
      break;
  }
});
```

### Testing

**Backend Tests:**
```bash
cd backend
python -m pytest tests/
```

**Frontend Tests:**
```bash
npm run test
```

**Integration Tests:**
```bash
# Test Google Drive connection
curl -X GET "http://localhost:8080/api/oauth/google-drive/status" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Test folder connection
curl -X POST "http://localhost:8080/api/v1/cloud-sync/connect-folders" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "knowledge_id": "kb_id",
    "folders": [{"id": "folder_id", "name": "My Folder"}],
    "provider": "google_drive"
  }'
```

## 🔒 Security

### OAuth Security
- Secure token storage with encryption
- Refresh token rotation
- Scope limitation (read-only access to Drive files)
- CSRF protection on OAuth callbacks

### Data Privacy
- Files are processed locally (not sent to external services)
- Only metadata is stored in database
- User consent required for each folder connection
- Easy disconnection and data cleanup

### Access Control
- User-specific folder connections
- Knowledge Base permissions respected
- Admin can disable Google Drive integration
- Rate limiting on API endpoints

## 🐛 Troubleshooting

### Common Issues

**"Import Error: cannot import name 'get_active_user_ids'"**
- Ensure you've copied the correct version of `backend/open_webui/routers/users.py`
- This file has compatible imports for the socket functions

**"404 Not Found on /api/oauth/google-drive/status"**
- Check that `vite.config.ts` exists locally with proxy configuration
- Ensure backend server is running on port 8080
- Verify OAuth router is registered in `main.py`

**"Google Drive authentication fails"**
- Verify `GOOGLE_DRIVE_CLIENT_ID` and `GOOGLE_DRIVE_CLIENT_SECRET` are set
- Check OAuth redirect URL matches Google Cloud Console settings
- Ensure Google Drive API is enabled in Google Cloud Console

**"Sync notifications not appearing"**
- Verify WebSocket connection is established
- Check browser console for socket errors
- Ensure `backend/open_webui/socket/main.py` has EventEmitter functionality

**"Files not syncing"**
- Check sync worker is running (logs should show periodic sync attempts)
- Verify folder permissions in Google Drive
- Check database for sync operation records
- Look for error messages in backend logs

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export SRC_LOG_LEVELS='{"MODELS": "DEBUG", "SOCKET": "DEBUG"}'
```

### Database Cleanup

If you need to reset sync data:
```sql
-- Remove all sync data (keeps refresh tokens)
DELETE FROM cloud_sync_file;
DELETE FROM cloud_sync_operation;
DELETE FROM cloud_sync_folder;

-- Remove everything including tokens
DELETE FROM cloud_sync_file;
DELETE FROM cloud_sync_operation;
DELETE FROM cloud_sync_folder;
DELETE FROM cloud_tokens WHERE provider = 'google_drive';
```

## 🤝 Contributing

### Development Setup

1. Fork this repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style

- Follow existing code patterns
- Use TypeScript for frontend
- Use Python type hints for backend
- Add comprehensive error handling
- Include unit tests for new features

### Pull Request Guidelines

- Describe the feature/fix in detail
- Include screenshots for UI changes
- Test with multiple Google accounts
- Verify database migrations work
- Check that existing functionality isn't broken

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenWebUI](https://github.com/open-webui/open-webui) - The amazing base platform
- [Google Drive API](https://developers.google.com/drive) - For providing the sync capabilities
- [SvelteKit](https://kit.svelte.dev/) - For the frontend framework
- [FastAPI](https://fastapi.tiangolo.com/) - For the backend framework

## 📞 Support

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/your-username/open-webui-google-drive-sync/issues)
3. Create a new issue with:
   - Detailed description of the problem
   - Steps to reproduce
   - Environment details (OS, browser, OpenWebUI version)
   - Relevant log messages
   - Screenshots if applicable

---

**Happy syncing! 🚀**
