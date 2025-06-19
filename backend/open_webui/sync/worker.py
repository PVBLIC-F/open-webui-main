"""
Cloud Sync Background Worker

This module provides the background worker that handles automatic and manual cloud
synchronization operations. It runs continuously to process sync requests and
maintain up-to-date content from connected cloud providers.

Key Features:
- Continuous background processing with configurable intervals
- Manual sync operation queue processing
- Automatic sync for all connected users
- Dedicated logging with file rotation
- Robust error handling and recovery
- Shared execution logic to eliminate code duplication

Architecture:
- Main worker loop runs every SYNC_INTERVAL seconds
- Processes pending manual operations first (priority)
- Then runs automatic sync for all users with valid tokens
- Uses provider-specific sync implementations
- All operations are tracked in the database for monitoring

Error Handling:
- Individual operation failures don't stop the worker
- Comprehensive logging for troubleshooting
- Graceful degradation when providers unavailable
- Automatic retry through normal sync cycles
"""

import asyncio
import logging
import time
from logging.handlers import RotatingFileHandler
from os import getenv
from pathlib import Path
from .providers import PROVIDERS
from open_webui.models.cloud_tokens import CloudTokens
from open_webui.models.cloud_sync import CloudSyncOperations
from open_webui.utils.cloud_sync_utils import get_current_timestamp
from open_webui.env import DATA_DIR


# Setup dedicated sync worker logging
def setup_sync_logging():
    """
    Setup dedicated logging for sync worker with file rotation.

    Creates a separate log file for sync operations to make troubleshooting
    easier and prevent sync logs from cluttering the main application logs.

    Features:
    - Separate cloud_sync.log file in the data directory
    - 10MB file size limit with 5 backup files
    - Both file and console output
    - Consistent timestamp formatting

    Returns:
        logging.Logger: Configured logger for sync operations
    """
    logs_dir = Path(DATA_DIR) / "logs"
    logs_dir.mkdir(exist_ok=True)

    sync_log_file = logs_dir / "cloud_sync.log"

    # Create sync worker logger
    sync_logger = logging.getLogger("cloud_sync")
    sync_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in sync_logger.handlers[:]:
        sync_logger.removeHandler(handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        sync_log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    sync_logger.addHandler(file_handler)
    sync_logger.addHandler(console_handler)

    return sync_logger


logger = setup_sync_logging()
SYNC_INTERVAL = int(getenv("CLOUD_SYNC_INTERVAL", "30"))  # Default 30 seconds


async def execute_sync_operation(
    operation_id: str, knowledge_id: str, provider_name: str = "google_drive"
):
    """
    Shared function to execute any sync operation - used by both manual and automatic sync.

    This function provides a unified execution path for all sync operations,
    eliminating code duplication and ensuring consistent behavior between
    manual triggers and automatic background sync.

    Features:
    - Updates operation status in real-time (pending -> running -> completed/failed)
    - Tracks timing and duration for performance monitoring
    - Uses existing provider infrastructure for actual sync work
    - Comprehensive error handling with database updates

    Args:
        operation_id: Database ID of the operation to execute
        knowledge_id: ID of the knowledge base to sync
        provider_name: Cloud provider name (default: 'google_drive')

    Returns:
        CloudSyncOperationModel: Updated operation with final status and timing

    Raises:
        Exception: If provider not found or sync operation fails
    """
    try:
        # Update operation status to running
        CloudSyncOperations.update_operation(
            operation_id, status="running", started_at=get_current_timestamp()
        )

        # Get provider and execute sync using existing infrastructure
        provider = PROVIDERS.get(provider_name)
        if not provider:
            raise Exception(f"Provider {provider_name} not found")

        # Use existing sync method - this already handles all the complexity
        await provider.sync_knowledge_base_by_id(knowledge_id)

        # Update operation status to completed
        return CloudSyncOperations.update_operation(
            operation_id, status="completed", finished_at=get_current_timestamp()
        )

    except Exception as e:
        logger.error(f"[SyncWorker] Error executing operation {operation_id}: {e}")

        # Update operation status to failed
        return CloudSyncOperations.update_operation(
            operation_id, status="failed", finished_at=get_current_timestamp()
        )


async def sync_worker():
    """
    Background sync worker that runs continuously.

    This is the main worker loop that handles both manual and automatic
    synchronization operations. It runs indefinitely, processing work
    at regular intervals defined by SYNC_INTERVAL.

    Processing Order:
    1. Manual operations (higher priority for user responsiveness)
    2. Automatic sync for all connected users
    3. Sleep for SYNC_INTERVAL seconds
    4. Repeat

    Error Resilience:
    - Individual operation failures don't stop the worker
    - Worker continues running even if entire sync cycles fail
    - All errors are logged for debugging
    """
    logger.info(f"[SyncWorker] Starting with {SYNC_INTERVAL}s interval")

    while True:
        try:
            # Process pending manual operations first
            await process_pending_operations()

            # Then run automatic sync for all users
            await run_automatic_sync()

        except Exception as e:
            logger.error(f"[SyncWorker] Error in sync worker: {e}")

        await asyncio.sleep(SYNC_INTERVAL)


async def process_pending_operations():
    """
    Process pending manual sync operations using shared execution function.

    This function handles the queue of manual sync operations triggered by users
    through the API. Manual operations are processed with higher priority than
    automatic sync to provide responsive user experience.

    Features:
    - Processes all pending operations in creation order
    - Uses shared execution logic for consistency
    - Individual operation failures don't stop processing
    - Comprehensive logging for each operation

    Operation Types Handled:
    - manual_sync: User-triggered sync
    - first_time_sync: Initial folder connection sync
    - folder_connection: New folder addition sync
    """
    try:
        pending_ops = CloudSyncOperations.get_pending_operations()

        if not pending_ops:
            return

        logger.info(f"[SyncWorker] Processing {len(pending_ops)} pending operations")

        for operation in pending_ops:
            try:
                await execute_sync_operation(
                    operation.id, operation.knowledge_id, operation.provider
                )
                logger.info(
                    f"[SyncWorker] Completed operation {operation.id} for KB {operation.knowledge_id}"
                )

            except Exception as e:
                logger.error(
                    f"[SyncWorker] Error processing operation {operation.id}: {e}"
                )

    except Exception as e:
        logger.error(f"[SyncWorker] Error processing pending operations: {e}")


async def run_automatic_sync():
    """
    Run automatic sync for all users with valid cloud provider tokens.

    This function performs background synchronization for all users who have
    connected cloud storage accounts. It runs after manual operations to
    ensure all connected folders stay up-to-date automatically.

    Features:
    - Processes all users with valid tokens
    - Uses existing provider sync infrastructure
    - Graceful handling when no users connected
    - Comprehensive error logging

    Token Validation:
    - Only processes users with valid, non-expired tokens
    - Token refresh is handled automatically by providers
    - Invalid tokens are logged but don't stop other users' sync
    """
    try:
        provider = PROVIDERS.get("google_drive")
        if not provider:
            logger.error("[SyncWorker] Google Drive provider not found")
            return

        tokens = CloudTokens.get_tokens_by_provider("google_drive")

        if not tokens:
            logger.debug(
                "[SyncWorker] No Google Drive tokens found, skipping automatic sync"
            )
            return

        logger.info(
            f"[SyncWorker] Running automatic sync for {len(tokens)} Google Drive users"
        )

        # Use existing provider sync method - it already handles all users
        await provider.sync()

        logger.info("[SyncWorker] Automatic sync completed successfully")

    except Exception as e:
        logger.error(f"[SyncWorker] Error in automatic sync: {e}")


def start_sync_worker():
    """
    Start the sync worker as a background task.

    This function initializes and starts the background sync worker as an
    asyncio task. It's called during application startup to begin continuous
    synchronization operations.

    Features:
    - Creates asyncio task for non-blocking operation
    - Returns task handle for management (stop, status check)
    - Graceful error handling during startup

    Returns:
        asyncio.Task: Background task handle, or None if startup failed

    Usage:
        Called once during application initialization to start background sync.
        The returned task runs until the application shuts down.
    """
    try:
        logger.info("[SyncWorker] Starting background sync worker...")
        return asyncio.create_task(sync_worker())
    except Exception as e:
        logger.error(f"[SyncWorker] Error starting sync worker: {e}")
        return None
