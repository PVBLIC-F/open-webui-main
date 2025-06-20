"""
Cloud Provider Sync Implementation

This module implements cloud provider-specific synchronization logic for integrating
external cloud storage with OpenWebUI knowledge bases. It handles file discovery,
download, processing, and change detection for various cloud providers.

Key Features:
- Multi-provider architecture with pluggable provider system
- Google Drive integration with OAuth 2.0 authentication
- Automatic file type conversion (Google Docs → Office formats)
- Incremental sync with change detection
- Real-time notifications for sync progress
- Comprehensive error handling and retry logic
- Support for both sync-managed and manually uploaded files

Architecture:
- Base CloudProvider class for extensibility
- GoogleDriveProvider implements Google Drive specific logic
- File processing pipeline: download → convert → extract → store → index
- Change detection compares cloud state with local database
- Notification system for real-time UI updates

Supported File Types:
- Google Workspace documents (Docs, Sheets, Slides)
- Standard documents (PDF, TXT, Markdown)
- Automatic conversion to compatible formats

Security:
- OAuth 2.0 token management with automatic refresh
- User isolation - each user's files are processed independently
- Secure file storage with proper metadata tracking
"""

import aiohttp
import io
import json
import logging
import os
import tempfile
import uuid
from typing import Dict, List, Any, Optional


from open_webui.models.cloud_sync import CloudSyncFiles
from open_webui.models.cloud_tokens import CloudTokens
from open_webui.models.knowledge import Knowledges
from open_webui.models.files import Files, FileForm
from open_webui.models.users import Users
from open_webui.utils.misc import calculate_sha256_string
from open_webui.utils.cloud_sync_utils import (
    parse_google_drive_timestamp,
    format_google_drive_timestamp,
    get_current_timestamp,
)

from open_webui.retrieval.loaders.main import Loader
from open_webui.storage.provider import Storage
from open_webui.config import (
    GOOGLE_DRIVE_CLIENT_ID,
    GOOGLE_DRIVE_CLIENT_SECRET,
)

# Use the same cloud_sync logger as the worker
logger = logging.getLogger("cloud_sync")


class CloudProvider:
    """
    Base class for cloud provider implementations.

    This abstract base class defines the interface that all cloud providers
    must implement. It provides a consistent API for the sync worker to
    interact with different cloud storage services.
    """

    def sync(self):
        """
        Main sync entry point for the provider.

        This method should be overridden by concrete provider implementations
        to perform their specific synchronization logic.
        """
        logger.info(f"[CloudProvider] sync() called (should be overridden)")


class GoogleDriveProvider(CloudProvider):
    """
    Google Drive cloud provider implementation.

    This class handles synchronization with Google Drive, including OAuth
    authentication, file discovery, download, and processing. It supports
    both Google Workspace documents and standard file types.

    Features:
    - OAuth 2.0 authentication with automatic token refresh
    - Support for Google Workspace document conversion
    - Incremental sync with change detection
    - Real-time progress notifications
    - Comprehensive error handling
    """

    def __init__(self):
        """
        Initialize Google Drive provider with supported file type mappings.

        The supported_mime_types dictionary maps source MIME types to target
        MIME types for file conversion. Google Workspace documents are
        converted to Office formats for compatibility.
        """
        # Maps source MIME type to target MIME type for conversion
        self.supported_mime_types = {
            # Google Workspace documents (convert to Office formats)
            "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            # Standard file types (no conversion needed)
            "application/pdf": "application/pdf",
            "text/plain": "text/plain",
            "text/markdown": "text/markdown",
        }

    async def sync(self):
        """
        Main sync entry point - run async sync for all connected users.

        This is the primary method called by the sync worker to perform
        automatic synchronization for all users with Google Drive connections.
        """
        try:
            await self._async_sync()
        except Exception as e:
            logger.error(f"[GoogleDriveProvider] Error in sync: {e}")

    async def _async_sync(self):
        """
        Async sync implementation for all knowledge bases with Google Drive connections.

        This method orchestrates the complete sync process:
        1. Discovers all knowledge bases with Google Drive folder connections
        2. Cleans up orphaned sync records from disconnected folders
        3. Groups folders by knowledge base for efficient processing
        4. Syncs each knowledge base with consolidated notifications

        The grouping approach ensures users receive a single notification per
        knowledge base rather than separate notifications for each folder.
        """
        try:
            logger.info("[GoogleDriveProvider] Starting async sync...")

            # Get all knowledge bases with Google Drive connections
            knowledge_bases = await self._get_knowledge_bases_with_google_drive()

            # Clean up sync records for knowledge bases that no longer have Google Drive connections
            await self._cleanup_orphaned_sync_records(knowledge_bases)

            if not knowledge_bases:
                logger.info(
                    "[GoogleDriveProvider] No knowledge bases with Google Drive connections found"
                )
                return

            logger.info(
                f"[GoogleDriveProvider] Found {len(knowledge_bases)} knowledge bases with Google Drive connections"
            )

            # Group knowledge bases by ID to aggregate results
            kb_groups = {}
            for kb in knowledge_bases:
                kb_id = kb["id"]
                if kb_id not in kb_groups:
                    kb_groups[kb_id] = {
                        "kb_info": {
                            "id": kb["id"],
                            "name": kb["name"],
                            "user_id": kb["user_id"],
                        },
                        "folders": [],
                    }
                kb_groups[kb_id]["folders"].append(kb)

            # Process each knowledge base (all folders together)
            for kb_id, kb_group in kb_groups.items():
                try:
                    await self._sync_knowledge_base_group(kb_group)
                except Exception as e:
                    logger.error(f"[GoogleDriveProvider] Error syncing KB {kb_id}: {e}")

            logger.info("[GoogleDriveProvider] Sync completed")

        except Exception as e:
            logger.error(f"[GoogleDriveProvider] Error in async sync: {e}")

    async def _get_knowledge_bases_with_google_drive(self) -> List[Dict[str, Any]]:
        """
        Get knowledge bases that have Google Drive folder connections and no active operations.

        This method discovers all knowledge bases with Google Drive connections that are
        ready for synchronization. It excludes knowledge bases with pending or running
        operations to prevent conflicts and ensure data consistency.

        Returns:
            List[Dict[str, Any]]: List of knowledge base dictionaries with folder info

        Coordination Logic:
        - Skips knowledge bases with active sync operations
        - Only includes enabled folder connections
        - Returns detailed folder information for each connection
        """
        try:
            from open_webui.models.cloud_sync import (
                CloudSyncFolders,
                CloudSyncOperations,
            )

            knowledge_bases = []

            # Get all knowledge bases from the database
            all_knowledges = Knowledges.get_knowledge_bases()

            # For each knowledge base, check if it has folder connections
            for knowledge in all_knowledges:
                # Skip knowledge bases with pending or running operations to prevent conflicts
                active_operations = CloudSyncOperations.get_active_operations_for_kb(
                    knowledge.id
                )
                if active_operations:
                    logger.debug(
                        f"[GoogleDriveProvider] Skipping KB {knowledge.id} - has {len(active_operations)} active operations"
                    )
                    continue

                # Get connected folders for this knowledge base
                connected_folders = CloudSyncFolders.get_connected_folders(knowledge.id)

                # Add knowledge base entry for each connected folder
                for folder in connected_folders:
                    if folder.sync_enabled:  # Only include enabled folders
                        knowledge_bases.append(
                            {
                                "id": knowledge.id,
                                "name": knowledge.name,
                                "user_id": knowledge.user_id,
                                "google_drive_folder_id": folder.cloud_folder_id,
                                "google_drive_folder_name": folder.folder_name,
                            }
                        )

            logger.info(
                f"[GoogleDriveProvider] Found {len(knowledge_bases)} knowledge base folder connections ready for sync"
            )
            return knowledge_bases

        except Exception as e:
            logger.error(f"Error getting knowledge bases with Google Drive: {e}")
            return []

    async def _get_knowledge_bases_with_google_drive_no_coordination(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Get knowledge bases with Google Drive connections WITHOUT coordination checks.

        This method is used for manual operations where immediate execution is required
        regardless of other pending operations. It bypasses the coordination logic
        used in automatic sync to ensure manual user actions are processed promptly.

        Returns:
            List[Dict[str, Any]]: All knowledge bases with Google Drive connections

        Use Cases:
        - Manual sync triggers from the UI
        - First-time folder connection sync
        - User-initiated operations that should execute immediately
        """
        try:
            from open_webui.models.cloud_sync import CloudSyncFolders

            knowledge_bases = []

            # Get all knowledge bases from the database
            all_knowledges = Knowledges.get_knowledge_bases()

            # For each knowledge base, check if it has folder connections (no coordination checks)
            for knowledge in all_knowledges:
                # Get connected folders for this knowledge base
                connected_folders = CloudSyncFolders.get_connected_folders(knowledge.id)

                # Add knowledge base entry for each connected folder
                for folder in connected_folders:
                    if folder.sync_enabled:  # Only include enabled folders
                        knowledge_bases.append(
                            {
                                "id": knowledge.id,
                                "name": knowledge.name,
                                "user_id": knowledge.user_id,
                                "google_drive_folder_id": folder.cloud_folder_id,
                                "google_drive_folder_name": folder.folder_name,
                            }
                        )

            logger.info(
                f"[GoogleDriveProvider] Found {len(knowledge_bases)} knowledge base folder connections for manual sync"
            )
            return knowledge_bases

        except Exception as e:
            logger.error(
                f"Error getting knowledge bases with Google Drive (no coordination): {e}"
            )
            return []

    async def _cleanup_orphaned_sync_records(
        self, active_knowledge_bases: List[Dict[str, Any]]
    ):
        """
        Clean up sync records AND files for knowledge bases that no longer have Google Drive connections.

        This method performs comprehensive cleanup of orphaned data when users disconnect
        folders or delete knowledge bases. It removes both database records and physical
        files to prevent data inconsistencies and storage bloat.

        Cleanup Process:
        1. Identifies knowledge bases with sync files but no active connections
        2. Removes all sync-managed files from the knowledge base
        3. Cleans up orphaned Google Drive files
        4. Removes database sync records

        Args:
            active_knowledge_bases: List of currently connected knowledge bases
        """
        try:
            # Get IDs of knowledge bases that currently have Google Drive connections
            active_kb_ids = [kb["id"] for kb in active_knowledge_bases]

            # Get all knowledge bases to find orphaned ones
            all_knowledges = Knowledges.get_knowledge_bases()

            # Find knowledge bases that had sync files but no longer have folder connections
            orphaned_kb_ids = []
            for knowledge in all_knowledges:
                if knowledge.id not in active_kb_ids:
                    # Check if this KB has any sync files
                    sync_files = CloudSyncFiles.get_files_by_knowledge(knowledge.id)
                    if sync_files:
                        orphaned_kb_ids.append(knowledge.id)

                        # Remove all sync-managed files from this knowledge base
                        logger.info(
                            f"[GoogleDriveProvider] Cleaning up {len(sync_files)} sync-tracked files from orphaned KB {knowledge.id}"
                        )

                        for sync_file in sync_files:
                            if sync_file.file_id:
                                try:
                                    await self._remove_file_from_knowledge_base(
                                        knowledge.id, sync_file.file_id
                                    )
                                    logger.info(
                                        f"[GoogleDriveProvider] Removed sync-tracked file {sync_file.filename} from KB {knowledge.id}"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"[GoogleDriveProvider] Error removing sync-tracked file {sync_file.file_id}: {e}"
                                    )

                        # Also clean up any orphaned Google Drive files in this knowledge base
                        # Use helper function for consistent cleanup behavior
                        from open_webui.utils.cloud_sync_utils import (
                            cleanup_google_drive_files_from_knowledge_base,
                        )

                        try:
                            cleaned_count = (
                                cleanup_google_drive_files_from_knowledge_base(
                                    knowledge.id
                                )
                            )
                            if cleaned_count > 0:
                                logger.info(
                                    f"[GoogleDriveProvider] Cleaned up {cleaned_count} orphaned Google Drive files from KB {knowledge.id}"
                                )
                        except Exception as e:
                            logger.error(
                                f"[GoogleDriveProvider] Error cleaning up orphaned files from KB {knowledge.id}: {e}"
                            )

            # Clean up orphaned sync records after removing files
            deleted_count = CloudSyncFiles.cleanup_orphaned_records(active_kb_ids)

            if deleted_count > 0:
                logger.info(
                    f"[GoogleDriveProvider] Cleaned up {deleted_count} orphaned sync records from {len(orphaned_kb_ids)} knowledge bases"
                )

        except Exception as e:
            logger.error(
                f"[GoogleDriveProvider] Error cleaning up orphaned sync records: {e}"
            )

    async def _sync_knowledge_base_group(self, kb_group: Dict[str, Any]):
        """
        Sync all folders for a knowledge base and emit a single summary notification.

        This method coordinates the synchronization of all folders connected to a
        single knowledge base. It aggregates results from all folders and sends
        a consolidated notification to provide a better user experience.

        Benefits of Grouping:
        - Single notification per knowledge base instead of per folder
        - Aggregated statistics across all folders
        - Better error reporting with context
        - Reduced notification noise for users with multiple folders

        Args:
            kb_group: Dictionary containing knowledge base info and list of folders
        """
        kb_info = kb_group["kb_info"]
        folders = kb_group["folders"]

        kb_id = kb_info["id"]
        user_id = kb_info["user_id"]
        kb_name = kb_info["name"]

        logger.info(
            f"[GoogleDriveProvider] Starting sync for KB {kb_name} (ID: {kb_id}) with {len(folders)} folders"
        )

        total_new_files = 0
        total_deleted_files = 0
        folder_names = []
        folders_with_changes = []
        sync_errors = []

        # Process each folder for this knowledge base
        for folder_kb in folders:
            folder_name = folder_kb["google_drive_folder_name"]
            folder_names.append(folder_name)

            try:
                # Sync this folder (without emitting completion notification)
                new_count, deleted_count = await self._sync_knowledge_base_folder(
                    folder_kb
                )
                total_new_files += new_count
                total_deleted_files += deleted_count

                # Track folders that had changes
                if new_count > 0 or deleted_count > 0:
                    folders_with_changes.append(folder_name)

            except Exception as e:
                logger.error(
                    f"[GoogleDriveProvider] Error syncing folder {folder_name} in KB {kb_id}: {e}"
                )
                sync_errors.append(f"{folder_name}: {str(e)}")

        # Only emit notification if there were actual changes or errors
        if sync_errors:
            # If there were errors, emit error notification
            await self._emit_sync_notification(
                "sync_error",
                kb_id,
                user_id,
                {"folder_names": folder_names, "errors": sync_errors},
            )
        elif total_new_files > 0 or total_deleted_files > 0:
            # Only emit completion notification if files were actually processed
            # Use the count of folders that actually had changes
            await self._emit_sync_notification(
                "sync_completed",
                kb_id,
                user_id,
                {
                    "folder_names": folders_with_changes,
                    "new_files_count": total_new_files,
                    "deleted_files_count": total_deleted_files,
                    "total_files_processed": total_new_files + total_deleted_files,
                    "folders_count": len(folders_with_changes),
                },
            )

        logger.info(
            f"[GoogleDriveProvider] Completed sync for KB {kb_name}: {total_new_files} new, {total_deleted_files} deleted across {len(folders)} folders"
        )

    async def _sync_knowledge_base_folder(self, kb: Dict[str, Any]) -> tuple[int, int]:
        """
        Sync a specific folder within a knowledge base and return file counts.

        This method handles the synchronization of a single Google Drive folder,
        including file discovery, change detection, and processing. It returns
        counts for aggregation at the knowledge base level.

        Process:
        1. Authenticate with Google Drive using stored tokens
        2. Fetch current files from the cloud folder
        3. Compare with stored files in the database
        4. Process new and deleted files
        5. Return counts for parent aggregation

        Args:
            kb: Knowledge base dictionary with folder information

        Returns:
            tuple[int, int]: (new_files_count, deleted_files_count)
        """
        kb_id = kb["id"]
        user_id = kb["user_id"]
        folder_id = kb["google_drive_folder_id"]
        folder_name = kb["google_drive_folder_name"]

        logger.info(
            f"[GoogleDriveProvider] Syncing folder {folder_name} in KB {kb['name']}"
        )

        # Get user's Google Drive token
        token = CloudTokens.get_token(user_id, "google_drive")
        if not token:
            logger.warning(f"No Google Drive token found for user {user_id}")
            return 0, 0

        # Ensure token is valid
        access_token = await self._get_valid_access_token(token)
        if not access_token:
            logger.warning(f"Could not get valid access token for user {user_id}")
            return 0, 0

        # Get current files from Google Drive
        current_files = await self._get_folder_files(folder_id, access_token)
        logger.info(
            f"[GoogleDriveProvider] Found {len(current_files)} files in Google Drive folder {folder_name}"
        )

        # Get stored files from database
        stored_files = await self._get_stored_files(kb_id, folder_id)
        logger.info(
            f"[GoogleDriveProvider] Found {len(stored_files)} stored files in database for folder {folder_name}"
        )

        # Detect changes
        changes = self._detect_changes(current_files, stored_files)

        # Process new files
        for file_data in changes["new"]:
            await self._process_new_file(
                kb_id, user_id, folder_id, folder_name, file_data, access_token
            )

        # Process deleted files
        for cloud_file_id in changes["deleted"]:
            await self._process_deleted_file(kb_id, folder_id, cloud_file_id)

        # Log the result for this folder
        new_count = len(changes["new"])
        deleted_count = len(changes["deleted"])

        if new_count or deleted_count:
            logger.info(
                f"[GoogleDriveProvider] Folder {folder_name}: {new_count} new, {deleted_count} deleted"
            )
        else:
            logger.info(
                f"[GoogleDriveProvider] Folder {folder_name}: No changes detected - all files up to date"
            )

        return new_count, deleted_count

    async def _sync_knowledge_base(self, kb: Dict[str, Any]):
        """
        Sync a specific knowledge base with its Google Drive folder.

        This is the legacy sync method that handles individual knowledge base
        synchronization. It's still used for compatibility but the grouped
        approach (_sync_knowledge_base_group) is preferred for better UX.

        Args:
            kb: Knowledge base dictionary with Google Drive folder information
        """
        kb_id = kb["id"]
        user_id = kb["user_id"]
        folder_id = kb["google_drive_folder_id"]
        folder_name = kb["google_drive_folder_name"]

        logger.info(
            f"[GoogleDriveProvider] Syncing KB {kb['name']} (folder: {folder_name})"
        )

        # Get user's Google Drive token
        token = CloudTokens.get_token(user_id, "google_drive")
        if not token:
            logger.warning(f"No Google Drive token found for user {user_id}")
            return

        # Ensure token is valid
        access_token = await self._get_valid_access_token(token)
        if not access_token:
            logger.warning(f"Could not get valid access token for user {user_id}")
            return

        try:
            # Get current files from Google Drive
            current_files = await self._get_folder_files(folder_id, access_token)
            logger.info(
                f"[GoogleDriveProvider] Found {len(current_files)} files in Google Drive folder"
            )

            # Get stored files from database
            stored_files = await self._get_stored_files(kb_id, folder_id)
            logger.info(
                f"[GoogleDriveProvider] Found {len(stored_files)} stored files in database"
            )

            # Detect changes
            changes = self._detect_changes(current_files, stored_files)

            # Process new files
            for file_data in changes["new"]:
                await self._process_new_file(
                    kb_id, user_id, folder_id, folder_name, file_data, access_token
                )

            # Process deleted files
            for cloud_file_id in changes["deleted"]:
                await self._process_deleted_file(kb_id, folder_id, cloud_file_id)

            # Always log the result
            if changes["new"] or changes["deleted"]:
                logger.info(
                    f"[GoogleDriveProvider] KB {kb['name']}: {len(changes['new'])} new, {len(changes['deleted'])} deleted"
                )
            else:
                logger.info(
                    f"[GoogleDriveProvider] KB {kb['name']}: No changes detected - all files up to date"
                )

            # NOTE: Sync completion notifications are now handled by _sync_knowledge_base_group
            # to prevent duplicate notifications when a KB has multiple folders

        except Exception as e:
            logger.error(f"Error syncing knowledge base {kb_id}: {e}")
            # NOTE: Sync error notifications are now handled by _sync_knowledge_base_group
            raise  # Re-raise to be caught by the group method

    async def _get_valid_access_token(self, token) -> Optional[str]:
        """
        Get a valid access token, refreshing if necessary.

        This method ensures we always have a valid access token for Google Drive API
        calls. It automatically refreshes expired tokens using the stored refresh token.

        Token Refresh Process:
        1. Check if current token is still valid (with buffer time)
        2. If expired, use refresh token to get new access token
        3. Update stored token in database
        4. Return fresh access token

        Args:
            token: CloudTokenModel with access and refresh tokens

        Returns:
            Optional[str]: Valid access token, or None if refresh failed
        """
        try:
            # Check if token is still valid
            if CloudTokens.is_token_valid(token):
                return token.access_token

            # Try to refresh the token
            if not token.refresh_token:
                logger.error("No refresh token available")
                return None

            refresh_data = {
                "client_id": GOOGLE_DRIVE_CLIENT_ID,
                "client_secret": GOOGLE_DRIVE_CLIENT_SECRET,
                "refresh_token": token.refresh_token,
                "grant_type": "refresh_token",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://oauth2.googleapis.com/token", data=refresh_data
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()

                        # Update the stored token
                        CloudTokens.update_token(
                            token.user_id,
                            "google_drive",
                            token_data["access_token"],
                            token.refresh_token,
                            token_data.get("expires_in", 3600),
                        )

                        return token_data["access_token"]
                    else:
                        logger.error(f"Failed to refresh token: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error getting valid access token: {e}")
            return None

    async def _get_folder_files(
        self, folder_id: str, access_token: str
    ) -> List[Dict[str, Any]]:
        """
        Get all supported files in a Google Drive folder.

        This method fetches all files from a Google Drive folder and filters them
        to only include file types that can be processed by the system.

        Process:
        1. Fetch all files from Google Drive API (with pagination)
        2. Filter to supported MIME types
        3. Return list of file metadata

        Args:
            folder_id: Google Drive folder ID
            access_token: Valid access token for API calls

        Returns:
            List[Dict[str, Any]]: List of supported file metadata
        """
        try:
            # Fetch all files from Google Drive API
            all_files = await self._fetch_all_files_from_drive(folder_id, access_token)

            # Filter to supported file types
            supported_files = self._filter_supported_files(all_files)

            logger.debug(
                f"Found {len(supported_files)} supported files in folder {folder_id}"
            )
            return supported_files

        except Exception as e:
            logger.error(f"Error getting folder files: {e}")
            return []

    async def _fetch_all_files_from_drive(
        self, folder_id: str, access_token: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch all files from Google Drive API with pagination support.

        This method handles the Google Drive API pagination to ensure all files
        in a folder are retrieved, regardless of folder size.

        Features:
        - Automatic pagination handling
        - Excludes trashed files
        - Fetches essential metadata for sync processing

        Args:
            folder_id: Google Drive folder ID
            access_token: Valid access token for API calls

        Returns:
            List[Dict[str, Any]]: Complete list of file metadata from the folder
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        files = []
        page_token = None

        async with aiohttp.ClientSession() as session:
            while True:
                params = {
                    "q": f"'{folder_id}' in parents and trashed=false",
                    "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,parents)",
                    "pageSize": 100,
                }

                if page_token:
                    params["pageToken"] = page_token

                async with session.get(
                    "https://www.googleapis.com/drive/v3/files",
                    headers=headers,
                    params=params,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        files.extend(data.get("files", []))

                        page_token = data.get("nextPageToken")
                        if not page_token:
                            break
                    else:
                        logger.error(f"Failed to get folder files: {response.status}")
                        break

        return files

    def _filter_supported_files(
        self, files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter files to only include supported MIME types.

        This method filters the raw file list from Google Drive to only include
        file types that can be processed by the system. Unsupported files are
        ignored to prevent processing errors.

        Args:
            files: Raw list of file metadata from Google Drive

        Returns:
            List[Dict[str, Any]]: Filtered list containing only supported files
        """
        supported_files = []
        for file_data in files:
            mime_type = file_data.get("mimeType", "")
            if mime_type in self.supported_mime_types:
                supported_files.append(file_data)
        return supported_files

    async def _get_stored_files(
        self, kb_id: str, folder_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get stored files for a folder - both sync-managed and manually uploaded.

        This method builds a comprehensive view of all files currently stored
        in the system for a specific folder. It includes both files managed
        by the sync system and files manually uploaded by users.

        File Categories:
        - Sync-managed: Files automatically synced from cloud storage
        - Manual uploads: Files uploaded directly by users but with cloud metadata

        Args:
            kb_id: Knowledge base ID
            folder_id: Google Drive folder ID

        Returns:
            Dict[str, Dict[str, Any]]: Map of cloud file ID to file metadata
        """
        stored_files = {}

        # Add sync-managed files
        self._add_sync_managed_files(stored_files, kb_id, folder_id)

        # Add manually uploaded Google Drive files
        self._add_manually_uploaded_files(stored_files, kb_id, folder_id)

        return stored_files

    def _add_sync_managed_files(
        self, stored_files: Dict[str, Dict[str, Any]], kb_id: str, folder_id: str
    ):
        """
        Add sync-managed files to the stored files dictionary.

        This method adds files that are managed by the sync system to the
        stored files collection. These files have complete sync metadata
        and are tracked in the cloud_sync_file table.

        Args:
            stored_files: Dictionary to add files to
            kb_id: Knowledge base ID
            folder_id: Google Drive folder ID
        """
        sync_files = CloudSyncFiles.get_files_by_knowledge(kb_id)
        for sync_file in sync_files:
            if (
                sync_file.cloud_folder_id == folder_id
                and sync_file.sync_status != "deleted"
            ):
                stored_files[sync_file.cloud_file_id] = {
                    "id": sync_file.cloud_file_id,
                    "name": sync_file.filename,
                    "mimeType": sync_file.mime_type,
                    "size": str(sync_file.file_size) if sync_file.file_size else "0",
                    "modifiedTime": sync_file.cloud_modified_time,
                    "sync_status": "sync_managed",
                }

    def _add_manually_uploaded_files(
        self, stored_files: Dict[str, Dict[str, Any]], kb_id: str, folder_id: str
    ):
        """
        Add manually uploaded Google Drive files to the stored files dictionary.

        This method adds files that were manually uploaded by users but contain
        Google Drive metadata. These files are not managed by the sync system
        but should be considered when detecting changes.

        Args:
            stored_files: Dictionary to add files to
            kb_id: Knowledge base ID
            folder_id: Google Drive folder ID
        """
        from open_webui.utils.cloud_sync_utils import get_google_drive_files_by_folder

        google_drive_files = get_google_drive_files_by_folder(kb_id, folder_id)
        for gd_file in google_drive_files:
            # Only include manually uploaded files (not sync-managed)
            if not gd_file["is_sync_managed"]:
                drive_file_id = gd_file["google_drive_file_id"]

                # Only add if not already present (avoid duplicates)
                if drive_file_id not in stored_files:
                    # Get the actual file record for additional metadata
                    file = Files.get_file_by_id(gd_file["file_id"])
                    if file:
                        stored_files[drive_file_id] = {
                            "id": drive_file_id,
                            "name": file.filename,
                            "mimeType": file.meta.get("content_type", ""),
                            "size": str(file.meta.get("size", 0)),
                            "modifiedTime": file.meta["data"].get("modified_time", ""),
                            "sync_status": "manual_upload",
                        }

    def _detect_changes(
        self,
        current_files: List[Dict[str, Any]],
        stored_files: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List]:
        """
        Detect changes between current cloud files and stored files.

        This method compares the current state of files in the cloud with the
        files stored in the local database to identify what needs to be synced.

        Change Detection Logic:
        - New files: Present in cloud but not in local storage
        - Deleted files: Present in local storage but not in cloud
        - Modified files: Present in both but with different modification times

        Args:
            current_files: List of files currently in the cloud folder
            stored_files: Dictionary of files currently stored locally

        Returns:
            Dict[str, List]: Dictionary with 'new' and 'deleted' file lists
        """
        current_file_ids = {f["id"] for f in current_files}
        stored_file_ids = set(stored_files.keys())

        # New files: in current but not in stored
        new_file_ids = current_file_ids - stored_file_ids
        new_files = [f for f in current_files if f["id"] in new_file_ids]

        # Deleted files: in stored but not in current
        deleted_file_ids = stored_file_ids - current_file_ids

        # Modified files: check modification time
        modified_files = []
        for file_data in current_files:
            file_id = file_data["id"]
            if file_id in stored_files:
                stored_file = stored_files[file_id]
                current_mod_time = file_data.get("modifiedTime")
                stored_mod_time = stored_file.get("modifiedTime")

                # Skip modification check if stored file has no modification time (manual upload)
                if not stored_mod_time:
                    continue

                if current_mod_time != stored_mod_time:
                    modified_files.append(file_data)

        # Log detailed change detection
        logger.debug(
            f"[GoogleDriveProvider] Change detection: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_file_ids)} deleted"
        )

        if new_files or modified_files or deleted_file_ids:
            logger.info(
                f"[GoogleDriveProvider] Changes detected: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_file_ids)} deleted"
            )
        else:
            logger.debug(
                f"[GoogleDriveProvider] No changes detected: all {len(current_files)} files are up to date"
            )

        return {
            "new": new_files
            + modified_files,  # Treat modified as new for re-processing
            "deleted": list(deleted_file_ids),
        }

    async def _process_new_file(
        self,
        kb_id: str,
        user_id: str,
        folder_id: str,
        folder_name: str,
        file_data: Dict[str, Any],
        access_token: str,
    ):
        """
        Process a new file using existing OpenWebUI pipeline.

        This method handles the complete processing of a new or modified file from
        Google Drive, including download, conversion, text extraction, storage,
        and indexing into the knowledge base.

        Processing Pipeline:
        1. Download file content from Google Drive
        2. Create file record in OpenWebUI storage
        3. Extract text content for indexing
        4. Add file to knowledge base
        5. Record sync metadata

        Args:
            kb_id: Knowledge base ID
            user_id: User ID who owns the file
            folder_id: Google Drive folder ID
            folder_name: Human-readable folder name
            file_data: File metadata from Google Drive
            access_token: Valid access token for downloads
        """
        filename = file_data["name"]

        try:
            # Download file content
            file_content = await self._download_file(
                file_data["id"], file_data.get("mimeType", ""), access_token
            )
            if not file_content:
                logger.error(f"Failed to download file content for {filename}")
                raise Exception("Failed to download file content")

            # Create file record using existing patterns
            file_id = await self._create_file_record(
                user_id, filename, file_content, file_data, folder_name, folder_id
            )
            if not file_id:
                logger.error(f"Failed to create file record for {filename}")
                raise Exception("Failed to create file record")

            # Add to knowledge base using existing functions
            await self._add_file_to_knowledge_base(kb_id, file_id, user_id)

            # Record sync metadata
            await self._record_sync_file(
                kb_id, user_id, folder_id, folder_name, file_data, file_id
            )

            logger.info(
                f"[GoogleDriveProvider] Successfully processed file {filename} from folder {folder_name}"
            )

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            # Re-raise to be handled by the group method
            raise

    async def _download_file(
        self, file_id: str, mime_type: str, access_token: str
    ) -> Optional[bytes]:
        """
        Download a file from Google Drive.

        This method handles the download of a file from Google Drive using the
        provided access token. It determines the appropriate download URL based
        on the file type and returns the file content as bytes.

        Args:
            file_id: Google Drive file ID
            mime_type: File MIME type
            access_token: Valid access token for API calls

        Returns:
            Optional[bytes]: File content as bytes, or None if download failed
        """
        try:
            headers = {"Authorization": f"Bearer {access_token}"}

            # Determine export format for Google Workspace files
            if mime_type in self.supported_mime_types:
                export_mime_type = self.supported_mime_types[mime_type]
                if mime_type.startswith("application/vnd.google-apps"):
                    # Export Google Workspace document
                    url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
                    params = {"mimeType": export_mime_type}
                else:
                    # Download regular file
                    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
                    params = {"alt": "media"}
            else:
                logger.warning(f"Unsupported MIME type: {mime_type}")
                return None

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.error(
                            f"Failed to download file {file_id}: {response.status}"
                        )
                        return None

        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return None

    async def _create_file_record(
        self,
        user_id: str,
        filename: str,
        file_content: bytes,
        file_data: Dict[str, Any],
        folder_name: str,
        folder_id: str,
    ) -> Optional[str]:
        """
        Create file record using existing OpenWebUI patterns.

        This method handles the creation of a new file record in the system using
        existing patterns and functions. It returns the file ID if successful, or None if it fails.

        Args:
            user_id: User ID who owns the file
            filename: File name
            file_content: File content as bytes
            file_data: File metadata
            folder_name: Human-readable folder name
            folder_id: Google Drive folder ID

        Returns:
            Optional[str]: File ID if successful, or None if it fails
        """
        try:
            file_id = str(uuid.uuid4())

            # Upload file to storage
            file_path = await self._upload_file_to_storage(
                file_id, filename, file_content, user_id
            )
            if not file_path:
                return None

            # Extract text content
            text_content = self._extract_text_content(
                filename, file_content, file_data.get("mimeType", "")
            )

            # Create and insert file record
            file_form = self._build_file_form(
                file_id,
                filename,
                file_path,
                text_content,
                file_content,
                file_data,
                folder_name,
                folder_id,
            )
            file_item = Files.insert_new_file(user_id, file_form)
            if not file_item:
                return None

            # Update file hash if we have text content
            if text_content:
                Files.update_file_hash_by_id(
                    file_id, calculate_sha256_string(text_content)
                )

            return file_id

        except Exception as e:
            logger.error(f"Error creating file record: {e}")
            return None

    async def _upload_file_to_storage(
        self, file_id: str, filename: str, file_content: bytes, user_id: str
    ) -> Optional[str]:
        """
        Upload file content to storage system.

        This method handles the upload of file content to the storage system. It returns
        the file path if successful, or None if it fails.

        Args:
            file_id: File ID
            filename: File name
            file_content: File content as bytes
            user_id: User ID who owns the file

        Returns:
            Optional[str]: File path if successful, or None if it fails
        """
        try:
            file_stream = io.BytesIO(file_content)
            file_stream.name = filename
            _, file_path = Storage.upload_file(
                file_stream,
                f"{file_id}_{filename}",
                {
                    "OpenWebUI-User-Id": user_id,
                    "OpenWebUI-File-Id": file_id,
                    "OpenWebUI-Source": "google_drive_sync",
                },
            )
            return file_path
        except Exception as e:
            logger.error(f"Error uploading file to storage: {e}")
            return None

    def _build_file_form(
        self,
        file_id: str,
        filename: str,
        file_path: str,
        text_content: str,
        file_content: bytes,
        file_data: Dict[str, Any],
        folder_name: str,
        folder_id: str,
    ) -> FileForm:
        """
        Build FileForm object with Google Drive metadata.

        This method constructs a FileForm object with Google Drive metadata for a new file.

        Args:
            file_id: File ID
            filename: File name
            file_path: File path in storage
            text_content: Extracted text content
            file_content: File content as bytes
            file_data: File metadata
            folder_name: Human-readable folder name
            folder_id: Google Drive folder ID

        Returns:
            FileForm: Constructed FileForm object
        """
        return FileForm(
            id=file_id,
            filename=filename,
            path=file_path,
            data={"content": text_content} if text_content else {},
            meta={
                "name": filename,
                "content_type": file_data.get("mimeType", "application/octet-stream"),
                "size": len(file_content),
                "data": {
                    "source": "google_drive",
                    "google_drive_file_id": file_data["id"],
                    "google_drive_folder_id": folder_id,
                    "google_drive_folder_name": folder_name,
                    "original_filename": filename,
                    "mime_type": file_data.get("mimeType", ""),
                    "file_size": file_data.get("size", 0),
                    "modified_time": file_data.get("modifiedTime", ""),
                },
            },
        )

    def _extract_text_content(
        self, filename: str, file_content: bytes, mime_type: str
    ) -> str:
        """
        Extract text content using existing loader.

        This method extracts text content from a file using the existing loader.

        Args:
            filename: File name
            file_content: File content as bytes
            mime_type: File MIME type

        Returns:
            str: Extracted text content
        """
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_{filename}"
            ) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            loader = Loader(engine="")
            docs = loader.load(filename, mime_type, temp_file_path)
            text_content = " ".join(
                [doc.page_content for doc in docs if doc.page_content]
            )

            os.unlink(temp_file_path)
            return text_content

        except Exception as e:
            logger.warning(f"Content extraction failed for {filename}: {e}")
            return ""

    async def _add_file_to_knowledge_base(self, kb_id: str, file_id: str, user_id: str):
        """
        Add file to knowledge base using the shared utility function.

        This method uses the shared utility function to add a file to a knowledge base.

        Args:
            kb_id: Knowledge base ID
            file_id: File ID
            user_id: User ID who owns the file
        """
        try:
            # Use the shared utility function for consistent behavior
            from open_webui.utils.knowledge import add_file_to_knowledge_base

            # Get user
            user = Users.get_user_by_id(user_id)
            if not user:
                raise Exception(f"User {user_id} not found")

            # Create a proper request object with app state
            request = self._create_real_request()

            # Use the shared utility function
            add_file_to_knowledge_base(request, kb_id, file_id, user)

            logger.info(
                f"[GoogleDriveProvider] Successfully added file {file_id} to KB {kb_id}"
            )

        except Exception as e:
            logger.error(f"[GoogleDriveProvider] Error adding file to KB: {e}")
            raise

    def _create_real_request(self):
        """
        Create a real request object with proper app state from the running application.

        This method creates a real request object with proper app state from the running application.

        Returns:
            Request: Real request object
        """
        try:
            app = self._get_fastapi_app()
            scope = self._build_request_scope(app)
            request = self._create_request_from_scope(scope)
            return request

        except Exception as e:
            logger.error(f"[GoogleDriveProvider] Could not create real request: {e}")
            raise

    def _get_fastapi_app(self):
        """
        Get the actual FastAPI app instance from the running application.

        This method gets the actual FastAPI app instance from the running application.

        Returns:
            FastAPI: Actual FastAPI app instance
        """
        import open_webui.main as main_module

        return main_module.app

    def _build_request_scope(self, app):
        """
        Build a minimal request scope with the app included.

        This method builds a minimal request scope with the app included.

        Args:
            app: FastAPI app instance

        Returns:
            dict: Minimal request scope
        """
        return {
            "type": "http",
            "method": "POST",
            "path": "/sync-worker",
            "headers": [],
            "query_string": b"",
            "app": app,
        }

    def _create_request_from_scope(self, scope):
        """
        Create a FastAPI Request object from the given scope.

        This method creates a FastAPI Request object from the given scope.

        Args:
            scope: Request scope

        Returns:
            Request: FastAPI Request object
        """
        from fastapi import Request

        return Request(scope)

    async def _record_sync_file(
        self,
        kb_id: str,
        user_id: str,
        folder_id: str,
        folder_name: str,
        file_data: Dict[str, Any],
        file_id: str,
    ):
        """
        Record sync file in database.

        This method handles the recording of sync file information in the database.

        Args:
            kb_id: Knowledge base ID
            user_id: User ID who owns the file
            folder_id: Google Drive folder ID
            folder_name: Human-readable folder name
            file_data: File metadata
            file_id: File ID
        """
        try:
            # Create or update sync record with all information at once
            CloudSyncFiles.upsert_and_complete_sync(
                user_id=user_id,
                knowledge_id=kb_id,
                provider="google_drive",
                cloud_file_data=file_data,
                folder_name=folder_name,
                file_id=file_id,
                cloud_version=file_data.get("modifiedTime"),
            )

        except Exception as e:
            logger.error(f"Error recording sync file: {e}")

    async def _process_deleted_file(
        self, kb_id: str, folder_id: str, cloud_file_id: str
    ):
        """
        Process deleted file using existing stored files lookup.

        This method handles the processing of a deleted file using the existing stored files lookup.

        Args:
            kb_id: Knowledge base ID
            folder_id: Google Drive folder ID
            cloud_file_id: Cloud file ID
        """
        try:
            # Use existing function to get all stored files (both sync-managed and manual)
            stored_files = await self._get_stored_files(kb_id, folder_id)

            if cloud_file_id not in stored_files:
                logger.warning(
                    f"[GoogleDriveProvider] File {cloud_file_id} not found in stored files"
                )
                return

            stored_file = stored_files[cloud_file_id]

            # Handle sync-managed files
            if stored_file.get("sync_status") == "sync_managed":
                sync_files = CloudSyncFiles.get_files_by_knowledge(kb_id)
                for sync_file in sync_files:
                    if sync_file.cloud_file_id == cloud_file_id and sync_file.file_id:
                        await self._remove_file_from_knowledge_base(
                            kb_id, sync_file.file_id
                        )
                        CloudSyncFiles.delete_sync_record(sync_file.id)
                        logger.info(
                            f"[GoogleDriveProvider] Removed sync-managed file {cloud_file_id}"
                        )
                        return

            # Handle manually uploaded files
            elif stored_file.get("sync_status") == "manual_upload":
                # Use helper functions for consistent metadata access
                from open_webui.utils.cloud_sync_utils import (
                    get_google_drive_files_by_folder,
                )

                # Find the specific file by Google Drive file ID
                google_drive_files = get_google_drive_files_by_folder(kb_id)
                for gd_file in google_drive_files:
                    if (
                        gd_file["google_drive_file_id"] == cloud_file_id
                        and not gd_file["is_sync_managed"]
                    ):
                        await self._remove_file_from_knowledge_base(
                            kb_id, gd_file["file_id"]
                        )
                        logger.info(
                            f"[GoogleDriveProvider] Removed manually uploaded file {cloud_file_id}"
                        )
                        return

        except Exception as e:
            logger.error(f"Error processing deleted file {cloud_file_id}: {e}")

    async def _remove_file_from_knowledge_base(self, kb_id: str, file_id: str):
        """
        Remove file using the shared utility function.

        This method uses the shared utility function to remove a file from a knowledge base.

        Args:
            kb_id: Knowledge base ID
            file_id: File ID
        """
        try:
            # Use the shared utility function for consistent behavior
            from open_webui.utils.knowledge import remove_file_from_knowledge_base

            # Use the shared utility function
            remove_file_from_knowledge_base(kb_id, file_id)

            logger.info(
                f"[GoogleDriveProvider] Successfully removed file {file_id} from KB {kb_id}"
            )

        except Exception as e:
            logger.error(f"[GoogleDriveProvider] Error removing file from KB: {e}")
            raise

    async def sync_knowledge_base_by_id(self, knowledge_id: str):
        """
        Sync a specific knowledge base by ID - used for manual sync operations.

        This method handles the synchronization of a specific knowledge base by ID.

        Args:
            knowledge_id: Knowledge base ID
        """
        try:
            logger.info(
                f"[GoogleDriveProvider] Starting manual sync for KB {knowledge_id}"
            )

            # For manual sync, get knowledge bases WITHOUT coordination checks
            # Manual operations should execute immediately regardless of pending operations
            all_knowledge_bases = (
                await self._get_knowledge_bases_with_google_drive_no_coordination()
            )

            # Filter to the specific knowledge base and group by ID
            target_kbs = [kb for kb in all_knowledge_bases if kb["id"] == knowledge_id]

            if not target_kbs:
                logger.warning(
                    f"[GoogleDriveProvider] No Google Drive connection found for KB {knowledge_id}"
                )
                return

            # Group folders for this knowledge base (same pattern as automatic sync)
            kb_group = {
                "kb_info": {
                    "id": target_kbs[0]["id"],
                    "name": target_kbs[0]["name"],
                    "user_id": target_kbs[0]["user_id"],
                },
                "folders": target_kbs,
            }

            # Use the same grouped sync method to ensure single notification
            await self._sync_knowledge_base_group(kb_group)
            logger.info(
                f"[GoogleDriveProvider] Manual sync completed for KB {knowledge_id}"
            )

        except Exception as e:
            logger.error(
                f"[GoogleDriveProvider] Error in sync_knowledge_base_by_id: {e}"
            )
            raise

    async def _emit_sync_notification(
        self, event_type: str, knowledge_id: str, user_id: str, data: Dict[str, Any]
    ):
        """
        Emit sync notification events for real-time UI updates.

        This method handles the emission of sync notification events for real-time UI updates.

        Args:
            event_type: Event type
            knowledge_id: Knowledge base ID
            user_id: User ID who owns the file
            data: Notification data
        """
        try:
            # Import here to avoid circular imports
            from open_webui.socket.main import get_global_event_emitter

            # Create notification event
            notification_data = {
                "type": "sync_notification",
                "event_type": event_type,  # 'file_processing', 'file_completed', 'file_error', 'sync_completed'
                "knowledge_id": knowledge_id,
                "user_id": user_id,
                "timestamp": get_current_timestamp(),
                **data,
            }

            # Emit to specific user using the global event emitter
            event_emitter = get_global_event_emitter()
            if event_emitter:
                await event_emitter.emit_to_user(
                    user_id, "sync_notification", notification_data
                )
                logger.debug(
                    f"[GoogleDriveProvider] Emitted {event_type} notification for user {user_id}"
                )
            else:
                logger.warning(
                    f"[GoogleDriveProvider] No event emitter available for {event_type}"
                )

        except Exception as e:
            logger.warning(f"[GoogleDriveProvider] Failed to emit notification: {e}")


# Provider registry for future extensibility
PROVIDERS = {
    "google_drive": GoogleDriveProvider(),
    # 'dropbox': DropboxProvider(),
    # 'onedrive': OneDriveProvider(),
}
