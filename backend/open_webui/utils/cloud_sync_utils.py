"""
Cloud Sync Utility Functions

This module provides centralized utility functions for cloud synchronization operations.
It eliminates code duplication across sync components and provides consistent behavior
for common operations like file cleanup, metadata access, and timestamp handling.

Key Features:
- Timestamp utilities for sync operations and display
- File cleanup functions for disconnected folders and orphaned files
- Metadata access helpers for Google Drive file information
- Batch operations for processing multiple files
- Comprehensive error handling and logging

Architecture:
- Pure utility functions with no side effects (where possible)
- Centralized error handling and logging
- Consistent return types and error conditions
- Helper functions designed for reuse across sync components

Categories:
- Timestamp Utilities: Time handling, formatting, and calculations
- File Cleanup Utilities: Removing files and sync records
- Metadata Access Utilities: Extracting Google Drive metadata
- Batch Operations: Processing multiple files efficiently
"""

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set
from open_webui.models.files import Files
from open_webui.models.knowledge import Knowledges

logger = logging.getLogger(__name__)

####################
# Timestamp Utilities
####################


def get_current_timestamp() -> int:
    """
    Get current Unix timestamp as integer.

    This is the standard timestamp function used throughout the sync system
    to ensure consistent time representation across all components.

    Returns:
        int: Current Unix timestamp in seconds
    """
    return int(time.time())


def get_timestamp_ms() -> int:
    """
    Get current Unix timestamp in milliseconds.

    Used for high-precision timing measurements and performance monitoring.

    Returns:
        int: Current Unix timestamp in milliseconds
    """
    return int(time.time() * 1000)


def calculate_duration_ms(
    start_timestamp: Optional[int], end_timestamp: Optional[int] = None
) -> Optional[int]:
    """
    Calculate duration in milliseconds between two timestamps.

    This function is used for measuring operation durations and performance
    metrics. It handles None values gracefully for incomplete operations.

    Args:
        start_timestamp: Start time as Unix timestamp in seconds
        end_timestamp: End time as Unix timestamp in seconds (defaults to current time)

    Returns:
        Optional[int]: Duration in milliseconds, None if start_timestamp is None
    """
    if start_timestamp is None:
        return None

    end_time = end_timestamp if end_timestamp is not None else get_current_timestamp()
    return (end_time - start_timestamp) * 1000


def format_timestamp_for_display(
    timestamp: Optional[int], format_str: str = "%Y-%m-%d %H:%M:%S"
) -> Optional[str]:
    """
    Format Unix timestamp for human-readable display.

    Converts Unix timestamps to user-friendly date/time strings for UI display.
    Handles timezone conversion and formatting errors gracefully.

    Args:
        timestamp: Unix timestamp in seconds
        format_str: strftime format string for output formatting

    Returns:
        Optional[str]: Formatted timestamp string, None if timestamp is None
    """
    if timestamp is None:
        return None

    try:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime(format_str)
    except (ValueError, OSError) as e:
        logger.warning(f"Failed to format timestamp {timestamp}: {e}")
        return None


def parse_google_drive_timestamp(gd_timestamp: Optional[str]) -> Optional[int]:
    """
    Parse Google Drive timestamp string to Unix timestamp.

    Google Drive uses RFC 3339 format (e.g., '2023-12-01T10:30:00.000Z').
    This function handles various timestamp formats and timezone representations
    that may be returned by the Google Drive API.

    Supported Formats:
    - RFC 3339 with Z suffix: '2023-12-01T10:30:00.000Z'
    - RFC 3339 with timezone: '2023-12-01T10:30:00.000+05:00'
    - ISO format without timezone (assumes UTC)

    Args:
        gd_timestamp: Google Drive timestamp string

    Returns:
        Optional[int]: Unix timestamp in seconds, None if parsing fails
    """
    if not gd_timestamp:
        return None

    try:
        # Handle various Google Drive timestamp formats
        if gd_timestamp.endswith("Z"):
            # RFC 3339 format with Z suffix
            dt = datetime.fromisoformat(gd_timestamp.replace("Z", "+00:00"))
        elif "+" in gd_timestamp or gd_timestamp.count("-") > 2:
            # RFC 3339 format with timezone offset
            dt = datetime.fromisoformat(gd_timestamp)
        else:
            # Fallback: assume UTC if no timezone info
            dt = datetime.fromisoformat(gd_timestamp).replace(tzinfo=timezone.utc)

        return int(dt.timestamp())
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse Google Drive timestamp '{gd_timestamp}': {e}")
        return None


def format_google_drive_timestamp(timestamp: Optional[int]) -> Optional[str]:
    """
    Format Unix timestamp to Google Drive compatible format.

    Converts Unix timestamps to RFC 3339 format expected by Google Drive API.
    Used when sending timestamp data to Google Drive for filtering or updates.

    Args:
        timestamp: Unix timestamp in seconds

    Returns:
        Optional[str]: RFC 3339 formatted timestamp, None if timestamp is None
    """
    if timestamp is None:
        return None

    try:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except (ValueError, OSError) as e:
        logger.warning(f"Failed to format timestamp {timestamp} for Google Drive: {e}")
        return None


def calculate_retry_timestamp(
    base_delay_seconds: int = 300, attempt_count: int = 0, max_delay_seconds: int = 3600
) -> int:
    """
    Calculate next retry timestamp using exponential backoff.

    This function implements exponential backoff for retry logic, helping to
    avoid overwhelming external services with repeated failed requests.

    Backoff Formula: base_delay * (2 ^ attempt_count), capped at max_delay

    Args:
        base_delay_seconds: Base delay in seconds (default: 5 minutes)
        attempt_count: Number of previous attempts (for exponential backoff)
        max_delay_seconds: Maximum delay in seconds (default: 1 hour)

    Returns:
        int: Unix timestamp for next retry attempt
    """
    # Exponential backoff: base_delay * (2 ^ attempt_count)
    delay = min(base_delay_seconds * (2**attempt_count), max_delay_seconds)
    return get_current_timestamp() + delay


def is_timestamp_expired(timestamp: Optional[int], expiry_seconds: int = 3600) -> bool:
    """
    Check if a timestamp has expired.

    Used for validating cached data, token expiration, and cleanup operations.
    Treats None timestamps as expired for safety.

    Args:
        timestamp: Unix timestamp to check
        expiry_seconds: Expiry duration in seconds (default: 1 hour)

    Returns:
        bool: True if timestamp is None or expired, False otherwise
    """
    if timestamp is None:
        return True

    return (get_current_timestamp() - timestamp) > expiry_seconds


def get_time_ago_description(timestamp: Optional[int]) -> str:
    """
    Get human-readable "time ago" description.

    Converts Unix timestamps to user-friendly relative time descriptions
    for UI display (e.g., "2 hours ago", "just now").

    Time Ranges:
    - < 1 minute: "just now"
    - < 1 hour: "X minutes ago"
    - < 1 day: "X hours ago"
    - < 30 days: "X days ago"
    - >= 30 days: "X months ago"

    Args:
        timestamp: Unix timestamp in seconds

    Returns:
        str: Human-readable time description
    """
    if timestamp is None:
        return "never"

    current_time = get_current_timestamp()
    diff = current_time - timestamp

    if diff < 0:
        return "in the future"
    elif diff < 60:
        return "just now"
    elif diff < 3600:  # Less than 1 hour
        minutes = diff // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif diff < 86400:  # Less than 1 day
        hours = diff // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff < 2592000:  # Less than 30 days
        days = diff // 86400
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        months = diff // 2592000
        return f"{months} month{'s' if months != 1 else ''} ago"


def create_operation_timestamps() -> Dict[str, int]:
    """
    Create standard timestamp dictionary for new operations.

    Provides consistent timestamp initialization for database records
    that track creation and update times.

    Returns:
        Dict[str, int]: Dictionary with created_at and updated_at timestamps
    """
    current_time = get_current_timestamp()
    return {"created_at": current_time, "updated_at": current_time}


def update_operation_timestamp() -> Dict[str, int]:
    """
    Create timestamp dictionary for operation updates.

    Provides consistent timestamp updates for database records when
    modifying existing operations.

    Returns:
        Dict[str, int]: Dictionary with updated_at timestamp
    """
    return {"updated_at": get_current_timestamp()}


####################
# File Cleanup Utilities
####################


def cleanup_google_drive_files_from_knowledge_base(
    knowledge_id: str,
    folder_id: Optional[str] = None,
    exclude_sync_managed: bool = False,
) -> int:
    """
    Clean up Google Drive files from a knowledge base.

    This function provides comprehensive cleanup of Google Drive files from a
    knowledge base. It can target specific folders or clean all Google Drive
    files, and can exclude sync-managed files to only clean manual uploads.

    Cleanup Process:
    1. Identifies Google Drive files in the knowledge base
    2. Applies folder and sync-managed filters as specified
    3. Removes files from knowledge base (vector DB + file records)
    4. Logs cleanup progress and results

    Args:
        knowledge_id: ID of the knowledge base to clean
        folder_id: Optional folder ID to filter files (if None, cleans all Google Drive files)
        exclude_sync_managed: If True, only clean up manually uploaded files

    Returns:
        int: Number of files successfully cleaned up
    """
    try:
        from open_webui.utils.knowledge import remove_file_from_knowledge_base

        cleaned_count = 0

        # Get knowledge base
        knowledge = Knowledges.get_knowledge_by_id(knowledge_id)
        if not knowledge or not knowledge.data or not knowledge.data.get("file_ids"):
            return 0

        # Get all files in the knowledge base
        files = Files.get_files_by_ids(knowledge.data["file_ids"])

        # Filter Google Drive files
        google_drive_files = []
        for file in files:
            if is_google_drive_file(file):
                file_folder_id = get_google_drive_folder_id(file)

                # Apply folder filter if specified
                if folder_id and file_folder_id != folder_id:
                    continue

                # Apply sync-managed filter if specified
                if exclude_sync_managed and is_sync_managed_file(knowledge_id, file):
                    continue

                google_drive_files.append(file)

        # Remove each Google Drive file
        for file in google_drive_files:
            try:
                remove_file_from_knowledge_base(knowledge_id, file.id)
                cleaned_count += 1
                logger.info(
                    f"Cleaned up Google Drive file {file.filename} from KB {knowledge_id}"
                )

            except Exception as e:
                logger.error(f"Error cleaning up file {file.id}: {e}")

        return cleaned_count

    except Exception as e:
        logger.error(
            f"Error cleaning up Google Drive files from KB {knowledge_id}: {e}"
        )
        return 0


def cleanup_sync_managed_files_from_folder(knowledge_id: str, folder_id: str) -> int:
    """
    Clean up sync-managed files from a specific folder.

    This function removes all files that were automatically synced from a
    specific Google Drive folder. It performs complete cleanup including
    vector database records, file records, and sync tracking records.

    Cleanup Steps:
    1. Identifies sync-managed files for the specified folder
    2. Removes files from knowledge base (vector DB + file records)
    3. Removes sync tracking records
    4. Logs cleanup progress

    Args:
        knowledge_id: ID of the knowledge base
        folder_id: Google Drive folder ID to clean

    Returns:
        int: Number of files successfully cleaned up
    """
    try:
        from open_webui.utils.knowledge import remove_file_from_knowledge_base
        from open_webui.models.cloud_sync import CloudSyncFiles

        cleaned_count = 0

        # Get sync files for this folder
        sync_files = CloudSyncFiles.get_files_by_knowledge(knowledge_id)
        folder_files = [f for f in sync_files if f.cloud_folder_id == folder_id]

        # Clean up each sync-managed file
        for sync_file in folder_files:
            if sync_file.file_id:
                try:
                    # Remove file from knowledge base (vector DB + file DB)
                    remove_file_from_knowledge_base(knowledge_id, sync_file.file_id)

                    # Remove sync record
                    CloudSyncFiles.delete_sync_record(sync_file.id)

                    cleaned_count += 1
                    logger.info(
                        f"Cleaned up sync-managed file {sync_file.filename} from KB {knowledge_id}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error cleaning up sync-managed file {sync_file.file_id}: {e}"
                    )

        return cleaned_count

    except Exception as e:
        logger.error(
            f"Error cleaning up sync-managed files from folder {folder_id}: {e}"
        )
        return 0


####################
# Metadata Access Utilities
####################


def is_google_drive_file(file) -> bool:
    """
    Check if a file is from Google Drive.

    This function examines file metadata to determine if a file originated
    from Google Drive. Used throughout the sync system to identify Google
    Drive files for special handling.

    Detection Logic:
    - Checks for presence of Google Drive file ID in metadata
    - Validates metadata structure is correct
    - Returns False for files without proper metadata

    Args:
        file: File object with meta attribute

    Returns:
        bool: True if file is from Google Drive, False otherwise
    """
    return (
        file.meta
        and file.meta.get("data")
        and file.meta["data"].get("google_drive_file_id") is not None
    )


def get_google_drive_file_id(file) -> Optional[str]:
    """
    Get Google Drive file ID from file metadata.

    Extracts the Google Drive file ID from OpenWebUI file metadata.
    This ID is used for API calls and sync tracking.

    Args:
        file: File object with meta attribute

    Returns:
        Optional[str]: Google Drive file ID or None if not a Google Drive file
    """
    if not is_google_drive_file(file):
        return None

    return file.meta["data"].get("google_drive_file_id")


def get_google_drive_folder_id(file) -> Optional[str]:
    """
    Get Google Drive folder ID from file metadata.

    Extracts the Google Drive folder ID from file metadata. Used for
    organizing files by folder and cleanup operations.

    Args:
        file: File object with meta attribute

    Returns:
        Optional[str]: Google Drive folder ID or None if not available
    """
    if not is_google_drive_file(file):
        return None

    return file.meta["data"].get("google_drive_folder_id")


def get_google_drive_folder_name(file) -> Optional[str]:
    """
    Get Google Drive folder name from file metadata.

    Extracts the human-readable folder name from file metadata.
    Used for display purposes and logging.

    Args:
        file: File object with meta attribute

    Returns:
        Optional[str]: Google Drive folder name or None if not available
    """
    if not is_google_drive_file(file):
        return None

    return file.meta["data"].get("google_drive_folder_name")


def is_sync_managed_file(knowledge_id: str, file) -> bool:
    """
    Check if a Google Drive file is managed by sync (vs manually uploaded).

    This function determines whether a Google Drive file is managed by the
    automatic sync system or was manually uploaded by a user. This distinction
    is important for cleanup operations and change detection.

    Detection Method:
    - Checks for existence of sync record in cloud_sync_file table
    - Matches Google Drive file ID with sync records
    - Returns False for non-Google Drive files

    Args:
        knowledge_id: ID of the knowledge base
        file: File object to check

    Returns:
        bool: True if file is sync-managed, False otherwise
    """
    if not is_google_drive_file(file):
        return False

    google_drive_file_id = get_google_drive_file_id(file)
    if not google_drive_file_id:
        return False

    # Check if there's a sync record for this file
    from open_webui.models.cloud_sync import CloudSyncFiles

    sync_files = CloudSyncFiles.get_files_by_knowledge(knowledge_id)
    for sync_file in sync_files:
        if sync_file.cloud_file_id == google_drive_file_id:
            return True

    return False


####################
# Batch Operations
####################


def get_google_drive_files_by_folder(
    knowledge_id: str, folder_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all Google Drive files in a knowledge base, optionally filtered by folder.

    This function provides a comprehensive view of all Google Drive files in a
    knowledge base, with optional filtering by folder. It includes both sync-managed
    and manually uploaded files with detailed metadata.

    Returned Information:
    - File ID and filename
    - Google Drive file and folder IDs
    - Folder name for display
    - Sync management status

    Args:
        knowledge_id: ID of the knowledge base to query
        folder_id: Optional folder ID to filter by (if None, returns all Google Drive files)

    Returns:
        List[Dict[str, Any]]: List of file info dictionaries with Google Drive metadata
    """
    try:
        # Get knowledge base
        knowledge = Knowledges.get_knowledge_by_id(knowledge_id)
        if not knowledge or not knowledge.data or not knowledge.data.get("file_ids"):
            return []

        # Get all files in the knowledge base
        files = Files.get_files_by_ids(knowledge.data["file_ids"])

        # Filter and collect Google Drive files
        google_drive_files = []
        for file in files:
            if is_google_drive_file(file):
                file_folder_id = get_google_drive_folder_id(file)

                # Apply folder filter if specified
                if folder_id and file_folder_id != folder_id:
                    continue

                google_drive_files.append(
                    {
                        "file_id": file.id,
                        "filename": file.filename,
                        "google_drive_file_id": get_google_drive_file_id(file),
                        "google_drive_folder_id": file_folder_id,
                        "google_drive_folder_name": get_google_drive_folder_name(file),
                        "is_sync_managed": is_sync_managed_file(knowledge_id, file),
                    }
                )

        return google_drive_files

    except Exception as e:
        logger.error(f"Error getting Google Drive files for KB {knowledge_id}: {e}")
        return []


def get_orphaned_google_drive_files(knowledge_id: str) -> List[Dict[str, Any]]:
    """
    Get Google Drive files that have metadata but no sync tracking.

    This function identifies "orphaned" Google Drive files - files that have
    Google Drive metadata but are not tracked by the sync system. These are
    typically files that were manually uploaded by users.

    Use Cases:
    - Cleanup operations for disconnected folders
    - Identifying manually uploaded files
    - Troubleshooting sync issues

    Args:
        knowledge_id: ID of the knowledge base to check

    Returns:
        List[Dict[str, Any]]: List of orphaned file info dictionaries
    """
    all_gd_files = get_google_drive_files_by_folder(knowledge_id)
    return [f for f in all_gd_files if not f["is_sync_managed"]]
