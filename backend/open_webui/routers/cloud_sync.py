"""
Cloud Sync API Endpoints

This module provides REST API endpoints for managing cloud provider synchronization
with knowledge bases. It handles folder connections, sync operations, status monitoring,
and file management for cloud storage integration.

Key Features:
- Manual and automatic sync triggering
- Folder connection/disconnection management
- Real-time sync status and progress monitoring
- File-level sync tracking and statistics
- Operation history and audit trail
- User authorization and access control

Supported Operations:
- Connect/disconnect cloud folders to knowledge bases
- Trigger manual sync operations
- Monitor sync status and file statistics
- View sync operation history
- Manage individual file sync status

Security:
- All operations require user authentication
- Knowledge base access validation on every request
- User isolation - users can only access their own data
- Comprehensive error handling with appropriate HTTP status codes
"""

import logging
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from open_webui.models.cloud_sync import (
    CloudSyncOperations,
    CloudSyncFiles,
    CloudSyncFolders,
    CloudSyncOperationModel,
    CloudSyncFileModel,
)
from open_webui.models.knowledge import Knowledges
from open_webui.utils.auth import get_verified_user
from open_webui.models.users import UserModel
from open_webui.sync.worker import execute_sync_operation
from open_webui.utils.cloud_sync_utils import (
    get_time_ago_description,
    format_timestamp_for_display,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/cloud-sync", tags=["cloud-sync"])

####################
# Helper Functions
####################


def _validate_knowledge_base_access(knowledge_id: str, user: UserModel):
    """
    Validate that user has access to the knowledge base.

    This security function ensures users can only access their own knowledge bases
    and provides consistent error handling across all endpoints.

    Args:
        knowledge_id: ID of the knowledge base to validate
        user: Authenticated user making the request

    Returns:
        Knowledge base object if access is valid

    Raises:
        HTTPException: 404 if knowledge base not found, 403 if access denied
    """
    kb = Knowledges.get_knowledge_by_id(knowledge_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found"
        )

    if kb.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this knowledge base",
        )

    return kb


####################
# Request/Response Models
####################


class TriggerSyncRequest(BaseModel):
    """Request model for triggering manual sync operations"""

    knowledge_id: str
    provider: str = "google_drive"
    operation_type: str = "manual_sync"


class FirstTimeSyncRequest(BaseModel):
    """Request model for first-time folder sync setup"""

    knowledge_id: str
    folder_id: str
    folder_name: str
    provider: str = "google_drive"


class FolderInfo(BaseModel):
    """Model representing a cloud folder for connection"""

    id: str
    name: str


class ConnectFoldersRequest(BaseModel):
    """Request model for connecting multiple folders to a knowledge base"""

    knowledge_id: str
    folders: List[FolderInfo]
    provider: str = "google_drive"


class SyncStatusResponse(BaseModel):
    """Response model for sync status information"""

    knowledge_id: str
    sync_enabled: bool
    last_sync: Optional[int] = None
    total_files: int = 0
    synced_files: int = 0
    pending_files: int = 0
    failed_files: int = 0
    recent_operations: List[CloudSyncOperationModel] = []


class DisconnectFoldersRequest(BaseModel):
    """Request model for disconnecting folders from a knowledge base"""

    knowledge_id: str
    folder_ids: List[str]
    provider: str = "google_drive"


####################
# Sync Operation Endpoints
####################


@router.post("/trigger", response_model=CloudSyncOperationModel)
async def trigger_sync(
    request: TriggerSyncRequest, user: UserModel = Depends(get_verified_user)
):
    """
    Trigger a manual sync operation for a knowledge base.

    This endpoint allows users to manually initiate synchronization of their
    connected cloud folders. Useful for immediate sync needs or testing.

    Args:
        request: Sync trigger request with knowledge base and provider info
        user: Authenticated user from dependency injection

    Returns:
        CloudSyncOperationModel: Created operation with status and metrics

    Raises:
        HTTPException: If knowledge base access denied or sync operation fails
    """
    try:
        # Verify user owns the knowledge base
        kb = _validate_knowledge_base_access(request.knowledge_id, user)

        # Create sync operation
        operation = CloudSyncOperations.create_operation(
            user_id=user.id,
            knowledge_id=request.knowledge_id,
            provider=request.provider,
            operation_type=request.operation_type,
        )

        log.info(
            f"Manual sync triggered for KB {request.knowledge_id} by user {user.id}"
        )

        # Execute sync using shared function - eliminates code duplication
        operation = await execute_sync_operation(
            operation.id, request.knowledge_id, request.provider
        )

        return operation

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error triggering sync: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/first-time-sync", response_model=CloudSyncOperationModel)
async def first_time_sync(
    request: FirstTimeSyncRequest, user: UserModel = Depends(get_verified_user)
):
    """
    Trigger first-time sync for a newly connected Google Drive folder.

    This endpoint is called when a user first connects a folder to establish
    the initial sync state and download existing files.

    Args:
        request: First-time sync request with folder details
        user: Authenticated user from dependency injection

    Returns:
        CloudSyncOperationModel: Created operation tracking the sync progress
    """
    try:
        # Verify user owns the knowledge base
        kb = _validate_knowledge_base_access(request.knowledge_id, user)

        # Create first-time sync operation
        operation = CloudSyncOperations.create_operation(
            user_id=user.id,
            knowledge_id=request.knowledge_id,
            provider=request.provider,
            operation_type="first_time_sync",
        )

        log.info(
            f"First-time sync triggered for KB {request.knowledge_id}, folder {request.folder_id}"
        )

        # Execute sync using shared function - eliminates code duplication
        operation = await execute_sync_operation(
            operation.id, request.knowledge_id, request.provider
        )

        return operation

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error triggering first-time sync: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/connect-folders", response_model=List[CloudSyncOperationModel])
async def connect_folders(
    request: ConnectFoldersRequest, user: UserModel = Depends(get_verified_user)
):
    """
    Connect multiple Google Drive folders to a knowledge base for sync.

    This endpoint establishes connections between cloud folders and a knowledge base,
    creating database records and triggering initial sync operations for each folder.

    Args:
        request: Connection request with knowledge base and folder list
        user: Authenticated user from dependency injection

    Returns:
        List[CloudSyncOperationModel]: Operations created for each folder connection
    """
    try:
        # Verify user owns the knowledge base
        kb = _validate_knowledge_base_access(request.knowledge_id, user)

        operations = []

        # Create sync operation for each folder
        for folder in request.folders:
            # Create folder connection record in database
            CloudSyncFolders.create_folder_connection(
                user_id=user.id,
                knowledge_id=request.knowledge_id,
                provider=request.provider,
                cloud_folder_id=folder.id,
                folder_name=folder.name,
            )

            operation = CloudSyncOperations.create_operation(
                user_id=user.id,
                knowledge_id=request.knowledge_id,
                provider=request.provider,
                operation_type="folder_connection",
            )

            log.info(
                f"Folder connection triggered for KB {request.knowledge_id}, folder {folder.id}"
            )

            # Execute sync using existing function
            operation = await execute_sync_operation(
                operation.id, request.knowledge_id, request.provider
            )

            operations.append(operation)

        return operations

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error connecting folders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/disconnect-folders")
async def disconnect_folders(
    request: DisconnectFoldersRequest, user: UserModel = Depends(get_verified_user)
):
    """
    Disconnect Google Drive folders from a knowledge base.

    This endpoint removes folder connections and cleans up associated files,
    including both sync-managed files and orphaned Google Drive files.
    Provides comprehensive cleanup to prevent data inconsistencies.

    Args:
        request: Disconnection request with folder IDs to remove
        user: Authenticated user from dependency injection

    Returns:
        dict: Summary of disconnection results including counts of folders
              and files cleaned up
    """
    try:
        # Verify user owns the knowledge base
        kb = _validate_knowledge_base_access(request.knowledge_id, user)

        disconnected_count = 0
        files_cleaned_up = 0

        # Remove each folder connection and clean up associated files
        for folder_id in request.folder_ids:
            # Remove folder connection
            success = CloudSyncFolders.remove_folder_connection(
                knowledge_id=request.knowledge_id, cloud_folder_id=folder_id
            )

            if success:
                disconnected_count += 1
                log.info(
                    f"Disconnected folder {folder_id} from KB {request.knowledge_id}"
                )

                # Use helper functions to clean up files from this folder
                from open_webui.utils.cloud_sync_utils import (
                    cleanup_sync_managed_files_from_folder,
                    cleanup_google_drive_files_from_knowledge_base,
                )

                # Clean up sync-managed files
                sync_cleaned = cleanup_sync_managed_files_from_folder(
                    request.knowledge_id, folder_id
                )
                files_cleaned_up += sync_cleaned

                # Clean up orphaned Google Drive files from this folder
                orphaned_cleaned = cleanup_google_drive_files_from_knowledge_base(
                    request.knowledge_id, folder_id=folder_id, exclude_sync_managed=True
                )
                files_cleaned_up += orphaned_cleaned

                log.info(
                    f"Cleaned up {sync_cleaned} sync-managed and {orphaned_cleaned} orphaned files from folder {folder_id}"
                )

            else:
                log.warning(
                    f"Failed to disconnect folder {folder_id} from KB {request.knowledge_id}"
                )

        return {
            "success": True,
            "disconnected_count": disconnected_count,
            "files_cleaned_up": files_cleaned_up,
            "total_requested": len(request.folder_ids),
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error disconnecting folders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/connected-folders/{knowledge_id}")
async def get_connected_folders(
    knowledge_id: str, user: UserModel = Depends(get_verified_user)
):
    """Get connected Google Drive folders for a knowledge base"""
    try:
        # Verify user owns the knowledge base
        kb = _validate_knowledge_base_access(knowledge_id, user)

        # Get connected folders
        folders = CloudSyncFolders.get_connected_folders(knowledge_id)

        # Return simplified folder info
        return {
            "folders": [
                {
                    "id": folder.cloud_folder_id,
                    "name": folder.folder_name,
                    "provider": folder.provider,
                }
                for folder in folders
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting connected folders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


####################
# Status and Monitoring Endpoints
####################


@router.get("/status/{knowledge_id}", response_model=SyncStatusResponse)
async def get_sync_status(
    knowledge_id: str, user: UserModel = Depends(get_verified_user)
):
    """Get sync status for a knowledge base"""
    try:
        # Verify user owns the knowledge base
        kb = _validate_knowledge_base_access(knowledge_id, user)

        # Get sync data
        sync_files = CloudSyncFiles.get_files_by_knowledge(knowledge_id)
        recent_operations = CloudSyncOperations.get_recent_operations(knowledge_id, 5)

        # Calculate file statistics
        file_stats = _calculate_file_statistics(sync_files)

        # Get last successful sync time
        last_sync = _get_last_successful_sync_time(recent_operations)

        return SyncStatusResponse(
            knowledge_id=knowledge_id,
            sync_enabled=True,
            last_sync=last_sync,
            **file_stats,
            recent_operations=recent_operations,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting sync status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


def _calculate_file_statistics(sync_files):
    """Calculate file statistics from sync files"""
    total_files = len(sync_files)
    synced_files = len([f for f in sync_files if f.sync_status == "synced"])
    pending_files = len([f for f in sync_files if f.sync_status == "pending"])
    failed_files = len([f for f in sync_files if f.sync_status == "failed"])

    return {
        "total_files": total_files,
        "synced_files": synced_files,
        "pending_files": pending_files,
        "failed_files": failed_files,
    }


def _get_last_successful_sync_time(recent_operations):
    """Get the timestamp of the last successful sync operation"""
    if not recent_operations:
        return None

    completed_ops = [op for op in recent_operations if op.status == "completed"]
    return completed_ops[0].finished_at if completed_ops else None


@router.get("/files/{knowledge_id}", response_model=List[CloudSyncFileModel])
async def get_sync_files(
    knowledge_id: str,
    status_filter: Optional[str] = None,
    user: UserModel = Depends(get_verified_user),
):
    """Get sync files for a knowledge base"""
    try:
        # Verify user owns the knowledge base
        kb = _validate_knowledge_base_access(knowledge_id, user)

        # Get files based on filter
        if status_filter == "pending":
            files = CloudSyncFiles.get_pending_files(knowledge_id)
        else:
            files = CloudSyncFiles.get_files_by_knowledge(knowledge_id)

            # Apply status filter if specified
            if status_filter:
                files = [f for f in files if f.sync_status == status_filter]

        return files

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting sync files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/operations/{knowledge_id}", response_model=List[CloudSyncOperationModel])
async def get_sync_operations(
    knowledge_id: str, limit: int = 10, user: UserModel = Depends(get_verified_user)
):
    """Get recent sync operations for a knowledge base"""
    try:
        # Verify user owns the knowledge base
        kb = _validate_knowledge_base_access(knowledge_id, user)

        operations = CloudSyncOperations.get_recent_operations(knowledge_id, limit)
        return operations

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting sync operations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
