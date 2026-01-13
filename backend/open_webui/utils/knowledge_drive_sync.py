"""
Knowledge Drive Sync Service

Orchestrates syncing files from Google Drive to Knowledge bases.
Uses existing RAG pipeline for processing.

Flow:
1. User connects a Drive folder to a Knowledge base
2. Sync service downloads files from Drive
3. Files are saved to Open WebUI storage
4. Files are processed through existing RAG pipeline
5. Changes are tracked for incremental sync
"""

import logging
import asyncio
import time
import uuid
import io
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime

from open_webui.utils.google_drive_client import (
    GoogleDriveClient,
    DriveFile,
    SUPPORTED_MIME_TYPES,
    GOOGLE_EXPORT_FORMATS,
    create_drive_client_for_user,
)
from open_webui.models.knowledge_drive import (
    KnowledgeDriveSources,
    KnowledgeDriveFiles,
    KnowledgeDriveSourceModel,
    KnowledgeDriveFileModel,
)
from open_webui.models.knowledge import Knowledges, KnowledgeModel
from open_webui.models.files import Files, FileForm
from open_webui.storage.provider import Storage
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class KnowledgeDriveSyncService:
    """
    Service for syncing Google Drive folders to Knowledge bases.
    
    Supports:
    - Initial full sync of all files in a folder
    - Incremental sync using Drive change tokens
    - Change detection via MD5 checksums
    - Automatic deletion handling
    """

    def __init__(self, app_state=None):
        """
        Initialize sync service.
        
        Args:
            app_state: FastAPI app state for accessing config and services
        """
        self.app_state = app_state
        self.active_syncs: Dict[str, dict] = {}

    async def sync_drive_source(
        self,
        source: KnowledgeDriveSourceModel,
        drive_client: GoogleDriveClient,
        user_id: str,
        force_full_sync: bool = False,
    ) -> Dict[str, Any]:
        """
        Sync a Drive source to its Knowledge base.
        
        Args:
            source: Drive source to sync
            drive_client: Authenticated Drive client
            user_id: User ID for file ownership
            force_full_sync: If True, resync all files regardless of changes
            
        Returns:
            Dict with sync statistics
        """
        source_id = source.id
        knowledge_id = source.knowledge_id

        log.info("=" * 70)
        log.info(f"🚀 DRIVE SYNC STARTED")
        log.info(f"   Source: {source_id[:8]}...")
        log.info(f"   Knowledge: {knowledge_id[:8]}...")
        log.info(f"   Folder: {source.drive_folder_name or source.drive_folder_id}")
        log.info(f"   Force full sync: {force_full_sync}")
        log.info("=" * 70)

        # Initialize sync tracking
        self.active_syncs[source_id] = {
            "status": "running",
            "start_time": time.time(),
            "files_found": 0,
            "files_synced": 0,
            "files_updated": 0,
            "files_deleted": 0,
            "errors": 0,
        }

        # Mark sync as starting
        KnowledgeDriveSources.mark_sync_start(source_id)

        try:
            # Get Knowledge base
            knowledge = Knowledges.get_knowledge_by_id(knowledge_id)
            if not knowledge:
                raise ValueError(f"Knowledge base {knowledge_id} not found")

            # Detect Shared Drive ID if not already stored (for backward compatibility)
            if not source.shared_drive_id:
                try:
                    folder_info = await drive_client.get_folder_info(source.drive_folder_id)
                    if folder_info.drive_id:
                        log.info(f"🔍 Detected Shared Drive ID: {folder_info.drive_id}")
                        # Update source with the drive ID
                        KnowledgeDriveSources.update_drive_source(
                            source_id, shared_drive_id=folder_info.drive_id
                        )
                        # Update local source object
                        source = KnowledgeDriveSources.get_drive_source_by_id(source_id)
                except Exception as e:
                    log.warning(f"Could not detect Shared Drive ID: {e}")

            # Determine sync type
            is_incremental = (
                not force_full_sync
                and source.last_sync_change_token
                and source.last_sync_timestamp
            )

            if is_incremental:
                log.info("📊 Performing INCREMENTAL sync using change tracking")
                result = await self._sync_incremental(
                    source, drive_client, user_id, knowledge
                )
            else:
                log.info("📊 Performing FULL sync of all files")
                result = await self._sync_full(
                    source, drive_client, user_id, knowledge
                )

            # Update sync status
            self.active_syncs[source_id]["status"] = "completed"
            self.active_syncs[source_id].update(result)

            # Get new change token for next incremental sync
            new_token = await drive_client.get_start_page_token()

            # Mark sync complete
            KnowledgeDriveSources.mark_sync_complete(
                source_id,
                files_synced=result.get("files_synced", 0),
                change_token=new_token,
            )

            duration = time.time() - self.active_syncs[source_id]["start_time"]

            log.info("")
            log.info("=" * 70)
            log.info("🎉 DRIVE SYNC COMPLETED")
            log.info(f"   Files found: {result.get('files_found', 0)}")
            log.info(f"   Files synced: {result.get('files_synced', 0)}")
            log.info(f"   Files updated: {result.get('files_updated', 0)}")
            log.info(f"   Files deleted: {result.get('files_deleted', 0)}")
            log.info(f"   Errors: {result.get('errors', 0)}")
            log.info(f"   Duration: {duration:.1f}s")
            log.info("=" * 70)

            return {
                **result,
                "duration": duration,
                "status": "completed",
            }

        except Exception as e:
            log.error(f"❌ Drive sync failed: {e}", exc_info=True)
            self.active_syncs[source_id]["status"] = "failed"
            self.active_syncs[source_id]["error"] = str(e)

            KnowledgeDriveSources.mark_sync_error(source_id, str(e))
            raise

        finally:
            await drive_client.close()

    async def _sync_full(
        self,
        source: KnowledgeDriveSourceModel,
        drive_client: GoogleDriveClient,
        user_id: str,
        knowledge: KnowledgeModel,
    ) -> Dict[str, Any]:
        """
        Perform a full sync of all files in the Drive folder.
        """
        # Get max files from config, default to 1000
        max_files = 1000
        if self.app_state and hasattr(self.app_state, "config"):
            max_files = getattr(
                self.app_state.config, "KNOWLEDGE_DRIVE_MAX_FILES", 1000
            )

        # Detect if folder is in a Shared Drive (need drive_id for API)
        folder_info = await drive_client.get_folder_info(source.drive_folder_id)
        drive_id = folder_info.drive_id
        
        # List all files in folder
        log.info(f"📂 Listing files in folder {source.drive_folder_id}...")
        if drive_id:
            log.info(f"   Folder is in Shared Drive: {drive_id}")
        drive_files = await drive_client.list_all_folder_files(
            source.drive_folder_id,
            max_files=max_files,
            drive_id=drive_id,  # Required for Shared Drives
        )

        # Filter to supported file types
        supported_files = [f for f in drive_files if f.is_supported()]
        log.info(
            f"   Found {len(drive_files)} files, {len(supported_files)} supported"
        )

        self.active_syncs[source.id]["files_found"] = len(supported_files)

        # Get existing tracked files
        existing_tracked = {
            f.drive_file_id: f
            for f in KnowledgeDriveFiles.get_drive_files_by_source_id(source.id)
        }

        # Track current Drive file IDs
        current_drive_ids = set()

        files_synced = 0
        files_updated = 0
        errors = 0

        # Process each file
        for i, drive_file in enumerate(supported_files):
            try:
                log.info(
                    f"📄 [{i+1}/{len(supported_files)}] Processing: {drive_file.name}"
                )
                current_drive_ids.add(drive_file.id)

                # Check if file exists and has changed
                existing = existing_tracked.get(drive_file.id)
                if existing:
                    # Check if file has changed (by MD5 or modified time)
                    if (
                        existing.drive_file_md5 == drive_file.md5_checksum
                        and existing.sync_status == "synced"
                    ):
                        log.info(f"   ⏭️  Unchanged, skipping")
                        continue

                    log.info(f"   🔄 File changed, updating")
                    was_update = True
                else:
                    log.info(f"   ➕ New file, downloading")
                    was_update = False

                # Download and process file
                file_id = await self._download_and_process_file(
                    drive_client,
                    drive_file,
                    source,
                    user_id,
                    knowledge,
                )

                if file_id:
                    # Track the file - create_or_update handles both cases
                    tracked_file = KnowledgeDriveFiles.create_or_update_drive_file(
                        drive_source_id=source.id,
                        knowledge_id=source.knowledge_id,
                        drive_file_id=drive_file.id,
                        drive_file_name=drive_file.name,
                        drive_file_mime_type=drive_file.mime_type,
                        drive_file_size=drive_file.size,
                        drive_file_md5=drive_file.md5_checksum,
                        drive_file_modified_time=drive_file.modified_time,
                        file_id=file_id,
                    )

                    # Mark as synced with timestamp
                    if tracked_file:
                        KnowledgeDriveFiles.mark_file_synced(
                            tracked_file.id, file_id
                        )

                    if was_update:
                        files_updated += 1
                    else:
                        files_synced += 1

                    log.info(f"   ✅ Synced successfully")
                else:
                    errors += 1
                    log.error(f"   ❌ Failed to sync")

                # Yield to event loop
                await asyncio.sleep(0)

            except Exception as e:
                log.error(f"   ❌ Error processing {drive_file.name}: {e}")
                errors += 1
                continue

        # Handle deleted files (in Drive but no longer present)
        deleted_ids = set(existing_tracked.keys()) - current_drive_ids
        files_deleted = 0

        for drive_id in deleted_ids:
            tracked = existing_tracked[drive_id]
            log.info(f"🗑️  File removed from Drive: {tracked.drive_file_name}")

            # Mark as deleted (don't remove from knowledge yet - just track status)
            KnowledgeDriveFiles.mark_file_deleted(tracked.id)
            files_deleted += 1

        return {
            "files_found": len(supported_files),
            "files_synced": files_synced,
            "files_updated": files_updated,
            "files_deleted": files_deleted,
            "errors": errors,
        }

    async def _sync_incremental(
        self,
        source: KnowledgeDriveSourceModel,
        drive_client: GoogleDriveClient,
        user_id: str,
        knowledge: KnowledgeModel,
    ) -> Dict[str, Any]:
        """
        Perform incremental sync using Drive change tracking.
        """
        log.info(f"📊 Getting changes since token: {source.last_sync_change_token[:20]}...")

        # Get all changes since last sync
        changes, new_token = await drive_client.get_all_changes(
            source.last_sync_change_token,
            folder_id=source.drive_folder_id,
        )

        log.info(f"   Found {len(changes)} changes")

        files_synced = 0
        files_updated = 0
        files_deleted = 0
        errors = 0

        for change in changes:
            try:
                file_data = change.get("file", {})
                file_id = change.get("fileId")
                removed = change.get("removed", False)

                if removed or file_data.get("trashed"):
                    # File was deleted or trashed
                    tracked = KnowledgeDriveFiles.get_drive_file_by_drive_id(
                        source.id, file_id
                    )
                    if tracked:
                        log.info(f"🗑️  File deleted: {tracked.drive_file_name}")
                        KnowledgeDriveFiles.mark_file_deleted(tracked.id)
                        files_deleted += 1
                    continue

                # Check if file is in our folder
                parents = file_data.get("parents", [])
                if source.drive_folder_id not in parents:
                    continue

                # Create DriveFile object
                drive_file = DriveFile.from_api_response(file_data)

                if not drive_file.is_supported():
                    continue

                log.info(f"📄 Change detected: {drive_file.name}")

                # Check if we already have this file
                existing = KnowledgeDriveFiles.get_drive_file_by_drive_id(
                    source.id, drive_file.id
                )

                # Download and process
                new_file_id = await self._download_and_process_file(
                    drive_client,
                    drive_file,
                    source,
                    user_id,
                    knowledge,
                )

                if new_file_id:
                    KnowledgeDriveFiles.create_or_update_drive_file(
                        drive_source_id=source.id,
                        knowledge_id=source.knowledge_id,
                        drive_file_id=drive_file.id,
                        drive_file_name=drive_file.name,
                        drive_file_mime_type=drive_file.mime_type,
                        drive_file_size=drive_file.size,
                        drive_file_md5=drive_file.md5_checksum,
                        drive_file_modified_time=drive_file.modified_time,
                        file_id=new_file_id,
                    )

                    if existing:
                        files_updated += 1
                    else:
                        files_synced += 1

                    log.info(f"   ✅ Synced")
                else:
                    errors += 1

                await asyncio.sleep(0)

            except Exception as e:
                log.error(f"Error processing change: {e}")
                errors += 1
                continue

        return {
            "files_found": len(changes),
            "files_synced": files_synced,
            "files_updated": files_updated,
            "files_deleted": files_deleted,
            "errors": errors,
        }

    async def _download_and_process_file(
        self,
        drive_client: GoogleDriveClient,
        drive_file: DriveFile,
        source: KnowledgeDriveSourceModel,
        user_id: str,
        knowledge: KnowledgeModel,
    ) -> Optional[str]:
        """
        Download a file from Drive and process it through the RAG pipeline.
        
        Returns:
            Open WebUI file ID if successful, None otherwise
        """
        try:
            # Download file content
            log.info(f"   ⬇️  Downloading {drive_file.name}...")
            content = await drive_client.download_file(drive_file)

            if not content:
                log.error(f"   ❌ Empty content for {drive_file.name}")
                return None

            # Determine filename and extension
            filename = drive_file.name
            if drive_file.is_google_workspace():
                # Add proper extension for exported files
                ext = SUPPORTED_MIME_TYPES.get(
                    GOOGLE_EXPORT_FORMATS.get(drive_file.mime_type, drive_file.mime_type),
                    ""
                )
                if ext and not filename.endswith(ext):
                    filename = f"{filename}{ext}"

            # Calculate hash for deduplication
            content_hash = hashlib.md5(content).hexdigest()

            # Generate file ID
            file_id = str(uuid.uuid4())
            storage_filename = f"{file_id}_{filename}"

            # Upload to Open WebUI storage
            log.info(f"   📤 Uploading to storage...")
            file_io = io.BytesIO(content)
            _, file_path = Storage.upload_file(
                file_io,
                storage_filename,
                {
                    "OpenWebUI-User-Id": user_id,
                    "OpenWebUI-File-Id": file_id,
                    "OpenWebUI-Drive-Source": source.id,
                    "OpenWebUI-Drive-File-Id": drive_file.id,
                },
            )

            # Determine content type
            content_type = drive_file.mime_type
            if drive_file.is_google_workspace():
                content_type = GOOGLE_EXPORT_FORMATS.get(
                    drive_file.mime_type, drive_file.mime_type
                )

            # Create file record
            log.info(f"   📝 Creating file record...")
            file_record = Files.insert_new_file(
                user_id,
                FileForm(
                    id=file_id,
                    hash=content_hash,
                    filename=filename,
                    path=file_path,
                    data={"status": "pending"},
                    meta={
                        "name": filename,
                        "content_type": content_type,
                        "size": len(content),
                        "source": "google_drive",
                        "drive_file_id": drive_file.id,
                        "drive_folder_id": source.drive_folder_id,
                    },
                ),
            )

            if not file_record:
                log.error(f"   ❌ Failed to create file record")
                return None

            # Add file to knowledge base
            log.info(f"   📚 Adding to Knowledge base...")
            Knowledges.add_file_to_knowledge_by_id(
                knowledge.id, file_id, user_id
            )

            # Process file through RAG pipeline
            log.info(f"   ⚙️  Processing through RAG pipeline...")
            await self._process_file_rag(file_id, knowledge.id, user_id)

            return file_id

        except Exception as e:
            log.error(f"   ❌ Error downloading/processing {drive_file.name}: {e}")
            return None

    async def _process_file_rag(
        self,
        file_id: str,
        knowledge_id: str,
        user_id: str,
    ):
        """
        Process a file through the RAG pipeline.
        
        Reuses existing Open WebUI file processing infrastructure.
        """
        try:
            # Import here to avoid circular imports
            from open_webui.routers.retrieval import process_file, ProcessFileForm
            from open_webui.models.users import Users

            # Get user for auth
            user = Users.get_user_by_id(user_id)
            if not user:
                raise ValueError(f"User {user_id} not found")

            # Create a mock request object for process_file
            # This is a workaround since process_file expects a Request
            class MockRequest:
                def __init__(self, app_state):
                    self.app = type("App", (), {"state": app_state})()

            if self.app_state:
                request = MockRequest(self.app_state)

                # Process file using existing infrastructure
                # Note: process_file is synchronous, run in executor
                # 
                # IMPORTANT: First call WITHOUT collection_name to trigger
                # the Loader to extract content from the file and create
                # vectors in file-{file_id} collection
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    process_file,
                    request,
                    ProcessFileForm(file_id=file_id),  # No collection_name!
                    user,
                    None,  # db session
                )
                
                # Now call WITH collection_name to copy vectors to knowledge base
                # This will find existing vectors in file-{file_id} and reuse them
                await loop.run_in_executor(
                    None,
                    process_file,
                    request,
                    ProcessFileForm(
                        file_id=file_id,
                        collection_name=knowledge_id,
                    ),
                    user,
                    None,  # db session
                )

                log.info(f"   ✅ RAG processing complete for {file_id[:8]}...")
            else:
                log.warning("   ⚠️  No app_state available, skipping RAG processing")

        except Exception as e:
            log.error(f"   ❌ RAG processing error for {file_id}: {e}")
            # Don't raise - file was synced, just RAG failed
            # Mark file with error status
            Files.update_file_data_by_id(file_id, {"status": "error", "error": str(e)})

    def get_sync_status(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get current sync status for a source"""
        return self.active_syncs.get(source_id)


# =============================================================================
# Background Sync Functions
# =============================================================================


async def sync_knowledge_drive_source(
    source_id: str,
    app_state,
    force_full_sync: bool = False,
):
    """
    Sync a single Drive source.
    
    Called from API endpoints or background scheduler.
    """
    log.info(f"🔄 Starting Drive sync for source {source_id[:8]}...")

    # Get source
    source = KnowledgeDriveSources.get_drive_source_by_id(source_id)
    if not source:
        log.error(f"Drive source {source_id} not found")
        return

    if not source.sync_enabled:
        log.info(f"Drive source {source_id} sync is disabled")
        return

    # Get user's Drive client
    drive_client = await create_drive_client_for_user(
        app_state.oauth_manager,
        source.user_id,
    )

    if not drive_client:
        log.error(f"Cannot create Drive client for user {source.user_id}")
        KnowledgeDriveSources.mark_sync_error(
            source_id,
            "Cannot access Google Drive. Please reconnect your Google account.",
        )
        return

    # Create sync service and run
    sync_service = KnowledgeDriveSyncService(app_state)

    try:
        await sync_service.sync_drive_source(
            source,
            drive_client,
            source.user_id,
            force_full_sync=force_full_sync,
        )
    except Exception as e:
        log.error(f"Drive sync failed for source {source_id}: {e}")
        raise


async def periodic_drive_sync_scheduler(app_state):
    """
    Periodic scheduler for Drive syncs.
    
    Runs as a background task, checking for sources that need syncing.
    Similar to Gmail periodic sync scheduler.
    """
    log.info("🔄 Drive Periodic Sync Scheduler starting...")

    # Startup delay
    await asyncio.sleep(120)

    log.info("✅ Drive Periodic Sync Scheduler ready")

    check_interval_minutes = 5

    while True:
        try:
            # Get sources needing sync
            # Check sources where last sync was > their configured interval ago
            sources = KnowledgeDriveSources.get_sources_needing_sync(
                max_hours_since_sync=1  # Default check hourly
            )

            if sources:
                log.info(f"📂 Found {len(sources)} Drive source(s) needing sync")

                for source in sources:
                    # Check if source's specific interval has passed
                    if source.last_sync_timestamp:
                        hours_since = (
                            time.time() - source.last_sync_timestamp
                        ) / 3600
                        if hours_since < source.auto_sync_interval_hours:
                            continue

                    try:
                        await sync_knowledge_drive_source(
                            source.id,
                            app_state,
                        )
                    except Exception as e:
                        log.error(f"Periodic sync failed for {source.id}: {e}")
                        continue

                    # Delay between syncs
                    await asyncio.sleep(5)

            # Wait before next check
            await asyncio.sleep(check_interval_minutes * 60)

        except asyncio.CancelledError:
            log.info("🛑 Drive Periodic Sync Scheduler shutting down")
            raise
        except Exception as e:
            log.error(f"Error in Drive sync scheduler: {e}")
            await asyncio.sleep(60)


async def trigger_drive_sync_for_knowledge(
    knowledge_id: str,
    app_state,
    force_full_sync: bool = False,
):
    """
    Trigger sync for all Drive sources connected to a Knowledge base.
    """
    sources = KnowledgeDriveSources.get_drive_sources_by_knowledge_id(knowledge_id)

    if not sources:
        log.info(f"No Drive sources connected to knowledge {knowledge_id}")
        return

    log.info(f"🔄 Triggering sync for {len(sources)} Drive source(s)")

    for source in sources:
        try:
            await sync_knowledge_drive_source(
                source.id,
                app_state,
                force_full_sync=force_full_sync,
            )
        except Exception as e:
            log.error(f"Sync failed for source {source.id}: {e}")
            continue
