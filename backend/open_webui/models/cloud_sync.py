"""
Cloud Sync Database Models

This module defines the database schema and operations for cloud provider synchronization.
It tracks files, operations, and folder connections for services like Google Drive.

Key Features:
- File-level sync tracking with status, metadata, and error handling
- Operation logging with detailed metrics and performance data
- Folder connection management with sync preferences
- Automatic retry logic for failed operations
- Comprehensive audit trail for troubleshooting

Database Tables:
- cloud_sync_file: Individual file sync records
- cloud_sync_operation: Sync operation history and metrics
- cloud_sync_folder: Folder connection configuration
"""

import logging
import time
import uuid
from typing import Optional, List, Dict, Any

from open_webui.internal.db import Base, get_db
from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.cloud_sync_utils import (
    get_current_timestamp,
    create_operation_timestamps,
    update_operation_timestamp,
    calculate_duration_ms,
)
from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text, Integer, Boolean

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

####################
# Cloud Sync DB Schema
####################


class CloudSyncFile(Base):
    """
    Track individual cloud files and their sync status.

    This table maintains a record of every file that has been synced from cloud
    providers, including metadata, sync status, and error tracking for failed operations.

    Key fields:
    - cloud_file_id: Unique identifier from the cloud provider
    - sync_status: Current status (pending, synced, failed, deleted)
    - error_count: Number of failed sync attempts (for retry logic)
    - content_hash: File content hash for change detection
    """

    __tablename__ = "cloud_sync_file"

    # Primary identifiers
    id = Column(String, primary_key=True)
    knowledge_id = Column(String, nullable=False)  # Parent knowledge base
    file_id = Column(String, nullable=True)  # Local OpenWebUI file ID
    user_id = Column(String, nullable=False)
    provider = Column(String, nullable=False)  # 'google_drive', 'dropbox', etc.

    # Cloud provider metadata
    cloud_file_id = Column(String, nullable=False)  # Provider's file ID
    cloud_folder_id = Column(String, nullable=False)  # Provider's folder ID
    cloud_folder_name = Column(String, nullable=True)  # Human-readable folder name
    filename = Column(String, nullable=False)
    mime_type = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    cloud_modified_time = Column(String, nullable=True)  # Provider's modification time

    # Sync status and timing
    sync_status = Column(
        String, nullable=False, default="pending"
    )  # pending, synced, failed, deleted
    last_sync_time = Column(BigInteger, nullable=True)
    meta = Column(Text, nullable=True)  # JSON metadata
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)

    # Enhanced tracking for reliability
    content_hash = Column(String, nullable=True)  # For change detection
    cloud_version = Column(String, nullable=True)  # Provider's version identifier
    error_count = Column(Integer, default=0)  # Failed sync attempts
    last_error = Column(Text, nullable=True)  # Last error message
    next_retry_at = Column(BigInteger, nullable=True)  # Exponential backoff


class CloudSyncOperation(Base):
    """
    Track sync operations and their performance metrics.

    This table provides an audit trail of all sync operations, including timing,
    success rates, and detailed statistics for monitoring and troubleshooting.

    Operation types:
    - first_time_sync: Initial folder connection
    - folder_connection: Adding new folders
    - manual_sync: User-triggered sync
    - automatic_sync: Background scheduled sync
    """

    __tablename__ = "cloud_sync_operation"

    # Operation identifiers
    id = Column(String, primary_key=True)
    knowledge_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    operation_type = Column(String, nullable=False)  # Type of sync operation

    # Operation status and timing
    status = Column(
        String, nullable=False, default="pending"
    )  # pending, running, completed, failed
    started_at = Column(BigInteger, nullable=True)
    finished_at = Column(BigInteger, nullable=True)
    duration_ms = Column(Integer, nullable=True)  # Operation duration

    # Performance metrics
    files_processed = Column(Integer, default=0)  # Successfully processed files
    files_skipped = Column(Integer, default=0)  # Already up-to-date files
    files_failed = Column(Integer, default=0)  # Failed to process

    # Additional data
    stats_json = Column(Text, nullable=True)  # Detailed statistics as JSON
    data = Column(Text, nullable=True)  # Operation-specific data
    meta = Column(Text, nullable=True)  # Additional metadata
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)


class CloudSyncFolder(Base):
    """
    Track folder-level sync configuration and state.

    This table manages the connection between knowledge bases and cloud folders,
    including sync preferences and last successful sync timestamps.

    Key features:
    - Folder connection management
    - Sync scheduling preferences
    - Last sync tracking for incremental updates
    """

    __tablename__ = "cloud_sync_folder"

    # Folder identifiers
    id = Column(String, primary_key=True)
    knowledge_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    cloud_folder_id = Column(String, nullable=False)  # Provider's folder ID
    folder_name = Column(String, nullable=True)  # Human-readable name

    # Sync status and preferences
    last_successful_sync = Column(BigInteger, nullable=True)
    last_sync_attempt = Column(BigInteger, nullable=True)
    sync_enabled = Column(Boolean, default=True)  # Can disable individual folders
    folder_token = Column(String, nullable=True)  # Provider-specific token

    # Timestamps
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)


####################
# Pydantic Models
####################


class CloudSyncFileModel(BaseModel):
    """Pydantic model for cloud sync file records"""

    id: str
    knowledge_id: str
    file_id: Optional[str] = None
    user_id: str
    provider: str
    cloud_file_id: str
    cloud_folder_id: str
    cloud_folder_name: Optional[str] = None
    filename: str
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    cloud_modified_time: Optional[str] = None
    sync_status: str
    last_sync_time: Optional[int] = None
    meta: Optional[str] = None
    created_at: int
    updated_at: int
    content_hash: Optional[str] = None
    cloud_version: Optional[str] = None
    error_count: int = 0
    last_error: Optional[str] = None
    next_retry_at: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class CloudSyncOperationModel(BaseModel):
    """Pydantic model for cloud sync operation records"""

    id: str
    knowledge_id: str
    user_id: str
    provider: str
    operation_type: str
    status: str
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    duration_ms: Optional[int] = None
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    stats_json: Optional[str] = None
    data: Optional[str] = None
    meta: Optional[str] = None
    created_at: int
    updated_at: int

    model_config = ConfigDict(from_attributes=True)


####################
# Table Operations
####################


class CloudSyncFilesTable:
    """Database operations for cloud sync file records"""

    def upsert_and_complete_sync(
        self,
        user_id: str,
        knowledge_id: str,
        provider: str,
        cloud_file_data: Dict[str, Any],
        folder_name: str,
        file_id: str,
        cloud_version: Optional[str] = None,
    ) -> Optional[CloudSyncFileModel]:
        """
        Insert or update a cloud sync file record with complete sync information.

        This method handles both new files and updates to existing files,
        marking them as successfully synced with all metadata populated.
        """
        with get_db() as db:
            current_time = get_current_timestamp()

            # Check if file already exists
            existing = (
                db.query(CloudSyncFile)
                .filter_by(
                    knowledge_id=knowledge_id, cloud_file_id=cloud_file_data["id"]
                )
                .first()
            )

            if existing:
                # Update existing record with all sync information
                existing.filename = cloud_file_data["name"]
                existing.mime_type = cloud_file_data.get("mimeType")
                existing.file_size = int(cloud_file_data.get("size", 0))
                existing.cloud_modified_time = cloud_file_data.get("modifiedTime")
                existing.cloud_folder_name = folder_name
                existing.file_id = file_id
                existing.sync_status = "synced"
                existing.last_sync_time = current_time
                existing.cloud_version = cloud_version
                existing.updated_at = current_time

                db.commit()
                db.refresh(existing)
                return CloudSyncFileModel.model_validate(existing)
            else:
                # Create new record with complete sync information
                timestamps = create_operation_timestamps()
                new_file = CloudSyncFile(
                    id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    user_id=user_id,
                    provider=provider,
                    cloud_file_id=cloud_file_data["id"],
                    cloud_folder_id=cloud_file_data.get("parents", [""])[0],
                    cloud_folder_name=folder_name,
                    filename=cloud_file_data["name"],
                    mime_type=cloud_file_data.get("mimeType"),
                    file_size=int(cloud_file_data.get("size", 0)),
                    cloud_modified_time=cloud_file_data.get("modifiedTime"),
                    file_id=file_id,
                    sync_status="synced",
                    last_sync_time=current_time,
                    cloud_version=cloud_version,
                    **timestamps,
                )

                db.add(new_file)
                db.commit()
                db.refresh(new_file)
                return CloudSyncFileModel.model_validate(new_file)

    def get_files_by_knowledge(self, knowledge_id: str) -> List[CloudSyncFileModel]:
        """Get all sync files for a knowledge base"""
        with get_db() as db:
            files = db.query(CloudSyncFile).filter_by(knowledge_id=knowledge_id).all()
            return [CloudSyncFileModel.model_validate(f) for f in files]

    def get_pending_files(self, knowledge_id: str) -> List[CloudSyncFileModel]:
        """Get pending sync files for a knowledge base"""
        with get_db() as db:
            files = (
                db.query(CloudSyncFile)
                .filter_by(knowledge_id=knowledge_id, sync_status="pending")
                .all()
            )
            return [CloudSyncFileModel.model_validate(f) for f in files]

    def delete_sync_record(self, sync_file_id: str) -> bool:
        """Remove a sync record"""
        with get_db() as db:
            sync_file = db.query(CloudSyncFile).filter_by(id=sync_file_id).first()
            if not sync_file:
                return False

            db.delete(sync_file)
            db.commit()
            return True

    def cleanup_orphaned_records(self, active_knowledge_base_ids: List[str]) -> int:
        """Remove sync records for knowledge bases that no longer have Google Drive connections"""
        with get_db() as db:
            # Single efficient query to delete orphaned records
            count = (
                db.query(CloudSyncFile)
                .filter(~CloudSyncFile.knowledge_id.in_(active_knowledge_base_ids))
                .count()
            )

            if count > 0:
                db.query(CloudSyncFile).filter(
                    ~CloudSyncFile.knowledge_id.in_(active_knowledge_base_ids)
                ).delete(synchronize_session=False)
                db.commit()

            return count


class CloudSyncOperationsTable:
    """Database operations for cloud sync operation records"""

    def create_operation(
        self, user_id: str, knowledge_id: str, provider: str, operation_type: str
    ) -> CloudSyncOperationModel:
        """Create a new sync operation record"""
        with get_db() as db:
            timestamps = create_operation_timestamps()

            operation = CloudSyncOperation(
                id=str(uuid.uuid4()),
                knowledge_id=knowledge_id,
                user_id=user_id,
                provider=provider,
                operation_type=operation_type,
                status="pending",
                **timestamps,
            )

            db.add(operation)
            db.commit()
            db.refresh(operation)
            return CloudSyncOperationModel.model_validate(operation)

    def update_operation(
        self, operation_id: str, **updates
    ) -> Optional[CloudSyncOperationModel]:
        """Update an existing sync operation with new status, metrics, or timestamps"""
        with get_db() as db:
            operation = db.query(CloudSyncOperation).filter_by(id=operation_id).first()
            if not operation:
                return None

            # Apply updates
            for key, value in updates.items():
                if hasattr(operation, key):
                    setattr(operation, key, value)

            # Calculate duration if both start and end times are available
            if (
                hasattr(operation, "started_at")
                and hasattr(operation, "finished_at")
                and operation.started_at
                and operation.finished_at
            ):
                operation.duration_ms = calculate_duration_ms(
                    operation.started_at, operation.finished_at
                )

            # Update timestamp
            timestamp_update = update_operation_timestamp()
            operation.updated_at = timestamp_update["updated_at"]

            db.commit()
            db.refresh(operation)
            return CloudSyncOperationModel.model_validate(operation)

    def get_recent_operations(
        self, knowledge_id: str, limit: int = 10
    ) -> List[CloudSyncOperationModel]:
        """Get recent sync operations for a knowledge base"""
        with get_db() as db:
            operations = (
                db.query(CloudSyncOperation)
                .filter_by(knowledge_id=knowledge_id)
                .order_by(CloudSyncOperation.created_at.desc())
                .limit(limit)
                .all()
            )

            return [CloudSyncOperationModel.model_validate(op) for op in operations]

    def get_pending_operations(self) -> List[CloudSyncOperationModel]:
        """Get all pending sync operations across all knowledge bases"""
        with get_db() as db:
            operations = (
                db.query(CloudSyncOperation)
                .filter_by(status="pending")
                .order_by(CloudSyncOperation.created_at.asc())
                .all()
            )

            return [CloudSyncOperationModel.model_validate(op) for op in operations]

    def get_active_operations_for_kb(
        self, knowledge_id: str
    ) -> List[CloudSyncOperationModel]:
        """Get active (pending or running) operations for a specific knowledge base"""
        with get_db() as db:
            operations = (
                db.query(CloudSyncOperation)
                .filter(
                    CloudSyncOperation.knowledge_id == knowledge_id,
                    CloudSyncOperation.status.in_(["pending", "running"]),
                )
                .all()
            )

            return [CloudSyncOperationModel.model_validate(op) for op in operations]


class CloudSyncFoldersTable:
    """Database operations for cloud sync folder records"""

    def create_folder_connection(
        self,
        user_id: str,
        knowledge_id: str,
        provider: str,
        cloud_folder_id: str,
        folder_name: str,
    ) -> Optional[CloudSyncFolder]:
        """
        Create or update a folder connection record.

        This establishes the link between a knowledge base and a cloud folder,
        enabling automatic synchronization of files.
        """
        with get_db() as db:
            current_time = get_current_timestamp()

            # Check if folder connection already exists
            existing = (
                db.query(CloudSyncFolder)
                .filter_by(knowledge_id=knowledge_id, cloud_folder_id=cloud_folder_id)
                .first()
            )

            if existing:
                # Update existing record
                existing.folder_name = folder_name
                existing.sync_enabled = True
                existing.updated_at = current_time
                db.commit()
                db.refresh(existing)
                return existing
            else:
                # Create new record
                timestamps = create_operation_timestamps()
                new_folder = CloudSyncFolder(
                    id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    user_id=user_id,
                    provider=provider,
                    cloud_folder_id=cloud_folder_id,
                    folder_name=folder_name,
                    sync_enabled=True,
                    **timestamps,
                )

                db.add(new_folder)
                db.commit()
                db.refresh(new_folder)
                return new_folder

    def get_connected_folders(self, knowledge_id: str) -> List[CloudSyncFolder]:
        """Get all active folder connections for a knowledge base"""
        with get_db() as db:
            folders = (
                db.query(CloudSyncFolder)
                .filter_by(knowledge_id=knowledge_id, sync_enabled=True)
                .all()
            )
            return folders

    def remove_folder_connection(self, knowledge_id: str, cloud_folder_id: str) -> bool:
        """Remove a folder connection from the database"""
        with get_db() as db:
            try:
                # Find the folder connection
                folder = (
                    db.query(CloudSyncFolder)
                    .filter_by(
                        knowledge_id=knowledge_id, cloud_folder_id=cloud_folder_id
                    )
                    .first()
                )

                if folder:
                    # Delete the folder connection record
                    db.delete(folder)
                    db.commit()
                    return True
                else:
                    # Folder connection not found
                    return False

            except Exception as e:
                db.rollback()
                raise e


# Global instances for easy access throughout the application
CloudSyncFiles = CloudSyncFilesTable()
CloudSyncOperations = CloudSyncOperationsTable()
CloudSyncFolders = CloudSyncFoldersTable()
