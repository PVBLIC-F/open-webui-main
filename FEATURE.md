
Google Drive Cloud Sync Integration
🌟 Overview
This PR adds comprehensive Google Drive Cloud Sync functionality to OpenWebUI Knowledge Bases, enabling automatic synchronization of files from Google Drive folders with real-time updates and background processing.
✨ Features
Core Functionality
📁 Folder Sync: Connect Google Drive folders to Knowledge Bases
🔄 Real-time Sync: Automatic background synchronization every 30 seconds
🔐 OAuth Authentication: Secure Google Drive access with refresh tokens
📄 File Type Support: PDF, Word docs, spreadsheets, presentations, text files, markdown
🔄 Google Workspace Conversion: Automatically converts Google Docs/Sheets/Slides to standard formats
↔️ Bidirectional Sync: Handles both file additions and deletions
User Experience
🎯 Visual Folder Picker: Intuitive folder selection interface with subfolder support
📊 Progress Tracking: Real-time notifications and visual feedback during sync operations
🔔 Live Updates: WebSocket-powered notifications when files are processed
♻️ Auto-refresh: Knowledge Base automatically updates when sync completes
📋 Folder Management: Easy connection/disconnection of folders with visual cards
Technical Features
⚡ Background Processing: Non-blocking sync operations
🛡️ Error Handling: Comprehensive error reporting and retry logic with exponential backoff
🗄️ Database Integration: Full PostgreSQL support with proper schema and indexes
🔌 Socket Notifications: Real-time UI updates via WebSocket
⚖️ Conflict Resolution: Handles duplicate files and hash conflicts
🏗️ Architecture
Backend Components
Models: Database schema for sync operations, file tracking, and OAuth tokens
Routers: API endpoints for OAuth flow and sync management
Sync Worker: Background service for continuous synchronization
Providers: Google Drive integration with API handling and file processing
Utilities: Helper functions for timestamps, cleanup, and metadata handling
Frontend Components
GoogleDrivePicker: Hierarchical folder selection with multi-select capability
File Display: Enhanced file grouping with folder-based organization
Folder Cards: Visual representation of connected folders with management controls
Real-time Notifications: Live sync status updates and progress indicators
📊 Database Schema
Adds 4 new tables with optimized indexes:
cloud_tokens: OAuth token storage and management
cloud_sync_file: Individual file sync tracking with metadata
cloud_sync_operation: Sync operation history and performance metrics
cloud_sync_folder: Folder connection configuration and state
🔧 Configuration
Required Environment Variables
Apply
Run
database
Optional Settings
Apply
Run
batch
🚀 Usage
Setup: Configure Google Cloud OAuth credentials and environment variables
Connect: Navigate to Knowledge Base → Add Content → Google Drive
Authenticate: Complete OAuth flow with Google account
Select Folders: Use folder picker to choose Google Drive folders
Sync: Automatic background synchronization begins immediately
🔒 Security
OAuth 2.0: Secure authentication with refresh token rotation
User Isolation: All tokens and folder connections are user-specific
Read-only Access: Limited scope for Google Drive file access
Local Processing: Files processed locally, not sent to external services
📁 Files Added/Modified
New Backend Files
backend/open_webui/models/cloud_sync.py - Database models for sync operations
backend/open_webui/models/cloud_tokens.py - OAuth token management
backend/open_webui/routers/cloud_sync.py - Sync API endpoints
backend/open_webui/routers/oauth.py - OAuth authentication flow
backend/open_webui/sync/worker.py - Background sync worker
backend/open_webui/sync/providers.py - Google Drive integration
backend/open_webui/utils/cloud_sync_utils.py - Utility functions
New Frontend Files
src/lib/components/drive/GoogleDrivePicker.svelte - Folder selection UI
src/lib/components/workspace/Knowledge/KnowledgeBase/FilesWithFolders.svelte - Enhanced file display
src/lib/components/workspace/Knowledge/KnowledgeBase/FolderCard.svelte - Folder management
src/lib/components/workspace/Knowledge/KnowledgeBase/GroupedFiles.svelte - File grouping
src/lib/utils/google-drive-api.ts - Google Drive API utilities
src/lib/utils/config.ts - Configuration helpers
Modified Files
Enhanced existing knowledge base components for cloud sync integration
Updated main application files for OAuth router registration
Extended socket handling for real-time sync notifications
🧪 Testing
Comprehensive error handling and retry mechanisms
Real-time sync validation with multiple file types
OAuth flow testing with token refresh scenarios
Database cleanup and orphaned file handling
