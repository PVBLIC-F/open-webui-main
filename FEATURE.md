# Google Drive Cloud Sync Integration

## 🌟 Overview

This document describes the comprehensive **Google Drive Cloud Sync** functionality added to OpenWebUI Knowledge Bases, enabling automatic synchronization of files from Google Drive folders with real-time updates and background processing.

## ✨ Features

### **Core Functionality**
- **📁 Folder Sync**: Connect Google Drive folders to Knowledge Bases
- **🔄 Real-time Sync**: Automatic background synchronization every 30 seconds
- **🔐 OAuth Authentication**: Secure Google Drive access with refresh tokens
- **📄 File Type Support**: PDF, Word docs, spreadsheets, presentations, text files, markdown
- **🔄 Google Workspace Conversion**: Automatically converts Google Docs/Sheets/Slides to standard formats
- **↔️ Bidirectional Sync**: Handles both file additions and deletions

### **User Experience**
- **🎯 Visual Folder Picker**: Intuitive folder selection interface with subfolder support
- **📊 Progress Tracking**: Real-time notifications and visual feedback during sync operations
- **🔔 Live Updates**: WebSocket-powered notifications when files are processed
- **♻️ Auto-refresh**: Knowledge Base automatically updates when sync completes
- **📋 Folder Management**: Easy connection/disconnection of folders with visual cards

### **Technical Features**
- **⚡ Background Processing**: Non-blocking sync operations
- **🛡️ Error Handling**: Comprehensive error reporting and retry logic with exponential backoff
- **🗄️ Database Integration**: Full PostgreSQL support with proper schema and indexes
- **🔌 Socket Notifications**: Real-time UI updates via WebSocket
- **⚖️ Conflict Resolution**: Handles duplicate files and hash conflicts

## 🏗️ Architecture

### **Backend Components**
- **Models**: Database schema for sync operations, file tracking, and OAuth tokens
- **Routers**: API endpoints for OAuth flow and sync management
- **Sync Worker**: Background service for continuous synchronization
- **Providers**: Google Drive integration with API handling and file processing
- **Utilities**: Helper functions for timestamps, cleanup, and metadata handling

### **Frontend Components**
- **GoogleDrivePicker**: Hierarchical folder selection with multi-select capability
- **File Display**: Enhanced file grouping with folder-based organization
- **Folder Cards**: Visual representation of connected folders with management controls
- **Real-time Notifications**: Live sync status updates and progress indicators

## 📊 Database Schema

Adds 4 new tables with optimized indexes:

### **cloud_tokens**
Stores OAuth tokens for cloud providers with automatic refresh capability:
```sql
CREATE TABLE cloud_tokens (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    access_token VARCHAR NOT NULL,
    refresh_token VARCHAR,
    expires_at BIGINT,
    created_at TIMESTAMP,
    UNIQUE(user_id, provider)
);
```

### **cloud_sync_file** 
Tracks individual file sync status with metadata and error handling:
```sql
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
```

### **cloud_sync_operation**
Logs sync operations with performance metrics and audit trail:
```sql
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

### **cloud_sync_folder**
Manages folder connections and sync configuration:
```sql
CREATE TABLE cloud_sync_folder (
    id VARCHAR PRIMARY KEY,
    knowledge_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    cloud_folder_id VARCHAR NOT NULL,
    folder_name VARCHAR,
    last_successful_sync BIGINT,
    last_sync_attempt BIGINT,
    sync_enabled BOOLEAN DEFAULT TRUE,
    folder_token VARCHAR,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);
```
### Database SQL Script

If the tables aren't automatically created in your hosted PostgreSQL database, run this SQL script:

```sql
-- Cloud Sync Database Tables and Indexes
-- Run this SQL script in your PostgreSQL database to create all required tables

-- ============================================================================
-- Cloud Tokens Table
-- Stores OAuth tokens for cloud providers (Google Drive, OneDrive, etc.)
-- ============================================================================

CREATE TABLE IF NOT EXISTS cloud_tokens (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    access_token VARCHAR NOT NULL,
    refresh_token VARCHAR,
    expires_at BIGINT,
    created_at TIMESTAMP,
    
    -- Ensure only one token per user per provider
    CONSTRAINT uq_user_provider UNIQUE (user_id, provider)
);

-- Indexes for cloud_tokens
CREATE INDEX IF NOT EXISTS idx_cloud_tokens_user_provider ON cloud_tokens (user_id, provider);
CREATE INDEX IF NOT EXISTS idx_cloud_tokens_provider ON cloud_tokens (provider);
CREATE INDEX IF NOT EXISTS idx_cloud_tokens_expires_at ON cloud_tokens (expires_at);

-- ============================================================================
-- Cloud Sync File Table
-- Tracks individual cloud files and their sync status
-- ============================================================================

CREATE TABLE IF NOT EXISTS cloud_sync_file (
    id VARCHAR PRIMARY KEY,
    knowledge_id VARCHAR NOT NULL,
    file_id VARCHAR,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    
    -- Cloud provider metadata
    cloud_file_id VARCHAR NOT NULL,
    cloud_folder_id VARCHAR NOT NULL,
    cloud_folder_name VARCHAR,
    filename VARCHAR NOT NULL,
    mime_type VARCHAR,
    file_size INTEGER,
    cloud_modified_time VARCHAR,
    
    -- Sync status and timing
    sync_status VARCHAR NOT NULL DEFAULT 'pending',
    last_sync_time BIGINT,
    meta TEXT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    
    -- Enhanced tracking for reliability
    content_hash VARCHAR,
    cloud_version VARCHAR,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    next_retry_at BIGINT
);

-- Indexes for cloud_sync_file
CREATE INDEX IF NOT EXISTS idx_cloud_sync_file_knowledge_id ON cloud_sync_file (knowledge_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_file_user_id ON cloud_sync_file (user_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_file_provider ON cloud_sync_file (provider);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_file_sync_status ON cloud_sync_file (sync_status);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_file_cloud_file_id ON cloud_sync_file (cloud_file_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_file_cloud_folder_id ON cloud_sync_file (cloud_folder_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_file_knowledge_cloud_file ON cloud_sync_file (knowledge_id, cloud_file_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_file_next_retry_at ON cloud_sync_file (next_retry_at);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_file_error_count ON cloud_sync_file (error_count);

-- ============================================================================
-- Cloud Sync Operation Table
-- Tracks sync operations and their performance metrics
-- ============================================================================

CREATE TABLE IF NOT EXISTS cloud_sync_operation (
    id VARCHAR PRIMARY KEY,
    knowledge_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    operation_type VARCHAR NOT NULL,
    
    -- Operation status and timing
    status VARCHAR NOT NULL DEFAULT 'pending',
    started_at BIGINT,
    finished_at BIGINT,
    duration_ms INTEGER,
    
    -- Performance metrics
    files_processed INTEGER DEFAULT 0,
    files_skipped INTEGER DEFAULT 0,
    files_failed INTEGER DEFAULT 0,
    
    -- Additional data
    stats_json TEXT,
    data TEXT,
    meta TEXT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

-- Indexes for cloud_sync_operation
CREATE INDEX IF NOT EXISTS idx_cloud_sync_operation_knowledge_id ON cloud_sync_operation (knowledge_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_operation_user_id ON cloud_sync_operation (user_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_operation_provider ON cloud_sync_operation (provider);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_operation_status ON cloud_sync_operation (status);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_operation_operation_type ON cloud_sync_operation (operation_type);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_operation_created_at ON cloud_sync_operation (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_operation_status_created ON cloud_sync_operation (status, created_at ASC);

-- ============================================================================
-- Cloud Sync Folder Table
-- Tracks folder-level sync configuration and state
-- ============================================================================

CREATE TABLE IF NOT EXISTS cloud_sync_folder (
    id VARCHAR PRIMARY KEY,
    knowledge_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    cloud_folder_id VARCHAR NOT NULL,
    folder_name VARCHAR,
    
    -- Sync status and preferences
    last_successful_sync BIGINT,
    last_sync_attempt BIGINT,
    sync_enabled BOOLEAN DEFAULT TRUE,
    folder_token VARCHAR,
    
    -- Timestamps
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

-- Indexes for cloud_sync_folder
CREATE INDEX IF NOT EXISTS idx_cloud_sync_folder_knowledge_id ON cloud_sync_folder (knowledge_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_folder_user_id ON cloud_sync_folder (user_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_folder_provider ON cloud_sync_folder (provider);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_folder_cloud_folder_id ON cloud_sync_folder (cloud_folder_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_folder_knowledge_cloud_folder ON cloud_sync_folder (knowledge_id, cloud_folder_id);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_folder_sync_enabled ON cloud_sync_folder (sync_enabled);
CREATE INDEX IF NOT EXISTS idx_cloud_sync_folder_last_successful_sync ON cloud_sync_folder (last_successful_sync);

-- ============================================================================
-- Verify Tables Created
-- ============================================================================

-- Check if all tables were created successfully
SELECT 
    tablename,
    schemaname
FROM pg_tables 
WHERE tablename IN ('cloud_tokens', 'cloud_sync_file', 'cloud_sync_operation', 'cloud_sync_folder')
ORDER BY tablename;

-- Check indexes created
SELECT 
    indexname,
    tablename,
    indexdef
FROM pg_indexes 
WHERE tablename IN ('cloud_tokens', 'cloud_sync_file', 'cloud_sync_operation', 'cloud_sync_folder')
ORDER BY tablename, indexname;
```

## 🔧 Configuration

### **Required Environment Variables**
```bash
ENABLE_GOOGLE_DRIVE_INTEGRATION=true
GOOGLE_DRIVE_CLIENT_ID=your_google_client_id
GOOGLE_DRIVE_CLIENT_SECRET=your_google_client_secret
DATABASE_URL=postgresql://user:password@host:port/database
```

### **Optional Settings**
```bash
CLOUD_SYNC_INTERVAL=30  # Sync interval in seconds (default: 30)
CLOUD_SYNC_BATCH_SIZE=10  # Files to process per batch (default: 10)
```

### **Google Cloud Setup**
1. Create Google Cloud Project
2. Enable Google Drive API
3. Create OAuth 2.0 credentials (Web application)
4. Add authorized redirect URI: `https://yourdomain.com/oauth/google-drive/callback`
5. Copy Client ID and Client Secret to environment variables

## 🚀 Usage

### **Initial Setup**
1. **Configure Environment**: Set required environment variables
2. **Database Setup**: Ensure PostgreSQL is configured and tables are created
3. **Google Cloud**: Complete OAuth credential setup

### **User Workflow**
1. **Access Knowledge Base**: Navigate to Workspace → Knowledge
2. **Connect Google Drive**: Click "Add Content" → "Google Drive"
3. **Authenticate**: Complete OAuth flow with Google account
4. **Select Folders**: Use hierarchical folder picker to choose directories
5. **Automatic Sync**: Background synchronization begins immediately

### **Folder Management**
- **View Connected Folders**: Folders appear as cards with file counts
- **Disconnect Folders**: Remove folder connections and clean up files
- **Manual Sync**: Trigger immediate synchronization
- **Monitor Progress**: Real-time notifications show sync status

## 🔒 Security

### **OAuth Security**
- **OAuth 2.0 Flow**: Standard authorization code flow with PKCE
- **Token Management**: Secure storage with automatic refresh
- **Scope Limitation**: Read-only access to Google Drive files
- **User Isolation**: All tokens and connections are user-specific

### **Data Privacy**
- **Local Processing**: Files processed locally, not sent to external services
- **Metadata Only**: Only file metadata stored in database
- **User Consent**: Explicit permission required for each folder connection
- **Easy Cleanup**: Complete data removal when disconnecting

### **Access Control**
- **User Permissions**: Respects existing Knowledge Base access controls
- **Admin Controls**: Can disable Google Drive integration globally
- **Rate Limiting**: API request throttling to respect Google limits

## 📁 Implementation Details

### **New Backend Files**
```
backend/open_webui/
├── models/
│   ├── cloud_sync.py          # Database models for sync operations
│   └── cloud_tokens.py        # OAuth token management
├── routers/
│   ├── cloud_sync.py          # Sync API endpoints
│   └── oauth.py               # OAuth authentication flow
├── sync/
│   ├── providers.py           # Google Drive integration
│   └── worker.py              # Background sync worker
└── utils/
    ├── cloud_sync_utils.py    # Utility functions
    └── knowledge.py           # Knowledge Base utilities
```

### **New Frontend Files**
```
src/lib/
├── components/
│   ├── drive/
│   │   └── GoogleDrivePicker.svelte    # Folder selection UI
│   └── workspace/Knowledge/KnowledgeBase/
│       ├── FilesWithFolders.svelte     # Enhanced file display
│       ├── FolderCard.svelte           # Folder management
│       └── GroupedFiles.svelte         # File grouping
└── utils/
    ├── google-drive-api.ts             # Google Drive API utilities
    └── config.ts                       # Configuration helpers
```

### **Modified Files**
- Enhanced existing Knowledge Base components for cloud sync integration
- Updated main application files for OAuth router registration
- Extended socket handling for real-time sync notifications

## 🔄 Sync Process

### **File Processing Pipeline**
1. **Discovery**: Scan Google Drive folders for files
2. **Change Detection**: Compare modification times and versions
3. **Download**: Fetch file content from Google Drive
4. **Conversion**: Transform Google Workspace files to standard formats
5. **Processing**: Extract text content and generate embeddings
6. **Storage**: Save to Knowledge Base with metadata
7. **Indexing**: Update vector database for search

### **Error Handling**
- **Retry Logic**: Exponential backoff for failed operations
- **Error Tracking**: Detailed logging of failures with context
- **Recovery**: Automatic retry of failed files on next sync
- **Monitoring**: Real-time error notifications to users

### **Performance Optimization**
- **Batch Processing**: Handle multiple files efficiently
- **Incremental Sync**: Only process changed files
- **Background Operations**: Non-blocking sync execution
- **Database Indexing**: Optimized queries for large datasets

## 🧪 Testing

### **Test Coverage**
- **OAuth Flow**: Token exchange and refresh scenarios
- **File Processing**: Multiple file types and formats
- **Error Scenarios**: Network failures, API limits, invalid files
- **Database Operations**: CRUD operations and cleanup
- **Real-time Updates**: WebSocket notification delivery

### **Manual Testing Checklist**
- [ ] Google OAuth authentication flow
- [ ] Folder selection and connection
- [ ] File upload and processing
- [ ] Real-time sync notifications
- [ ] Folder disconnection and cleanup
- [ ] Error handling and recovery
- [ ] Performance with large folders

## 🚨 Troubleshooting

### **Common Issues**

**OAuth Authentication Fails**
- Verify Google Cloud credentials are correct
- Check redirect URI matches Google Cloud Console
- Ensure Google Drive API is enabled

**Files Not Syncing**
- Check sync worker is running (background process)
- Verify folder permissions in Google Drive
- Review sync operation logs for errors

**Database Errors**
- Ensure PostgreSQL is running and accessible
- Verify database tables are created
- Check connection string format

**Performance Issues**
- Monitor sync batch size configuration
- Check database index performance
- Review network connectivity to Google APIs

### **Debug Mode**
Enable detailed logging for troubleshooting:
```bash
export LOG_LEVEL=DEBUG
export SRC_LOG_LEVELS='{"MODELS": "DEBUG", "SOCKET": "DEBUG"}'
```

## 📈 Monitoring

### **Key Metrics**
- **Sync Success Rate**: Percentage of successful file operations
- **Processing Time**: Average time per file and operation
- **Error Frequency**: Rate of failed operations by type
- **User Adoption**: Number of connected folders and active users

### **Health Checks**
- **Token Validity**: Monitor OAuth token expiration
- **API Limits**: Track Google Drive API usage
- **Database Performance**: Monitor query execution times
- **Background Worker**: Ensure sync worker is running

## 🔮 Future Enhancements

### **Planned Features**
- **OneDrive Integration**: Extend to Microsoft OneDrive
- **Dropbox Support**: Add Dropbox cloud sync
- **Selective Sync**: Choose specific file types to sync
- **Sync Scheduling**: Custom sync intervals per folder
- **Advanced Filtering**: Exclude files by pattern or size

### **Performance Improvements**
- **Parallel Processing**: Multi-threaded file processing
- **Smart Caching**: Cache file metadata for faster sync
- **Delta Sync**: Only sync file changes, not entire files
- **Compression**: Optimize storage and transfer

## 📄 License

This feature is part of OpenWebUI and follows the same licensing terms.

## 🤝 Contributing

### **Development Setup**
1. Fork the OpenWebUI repository
2. Set up development environment with Google Cloud credentials
3. Create feature branch for changes
4. Test thoroughly with multiple file types
5. Submit pull request with detailed description

### **Code Guidelines**
- Follow existing OpenWebUI code patterns
- Add comprehensive error handling
- Include unit tests for new functionality
- Document API changes and new endpoints
- Ensure backward compatibility

---

**Ready for production deployment with full Google Drive synchronization capabilities!** 🎉 
