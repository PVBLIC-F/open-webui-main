"""
Knowledge Base Utilities

This module provides shared business logic for knowledge base file operations.
It centralizes common file management tasks that are used by both the knowledge
router and cloud sync system, ensuring consistent behavior across components.

Key Features:
- File addition to knowledge bases with vector indexing
- File removal with comprehensive cleanup (vector DB + physical files + database)
- Error handling and logging for troubleshooting
- Integration with vector database and storage systems

Architecture:
- Shared utility functions to eliminate code duplication
- Consistent error handling and logging patterns
- Integration with OpenWebUI's storage and vector database systems
- Support for both manual and automated file operations

Usage:
- Knowledge router: Direct file management operations
- Cloud sync system: Automated file lifecycle management
- Admin operations: Bulk file operations and cleanup

Critical Notes:
- Physical file deletion was added to prevent orphaned files
- Vector database cleanup includes both file_id and hash filters
- File collection cleanup prevents vector database bloat
- Database record cleanup maintains referential integrity
"""

import logging
from typing import Optional
from open_webui.models.knowledge import Knowledges
from open_webui.models.files import Files
from open_webui.models.users import Users
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.routers.retrieval import process_file, ProcessFileForm
from open_webui.storage.provider import Storage

logger = logging.getLogger(__name__)


def add_file_to_knowledge_base(request, kb_id: str, file_id: str, user) -> None:
    """
    Add a file to a knowledge base with full processing and indexing.

    This function performs the complete workflow for adding a file to a knowledge base:
    1. Validates the file exists and has been processed
    2. Processes the file content for vector indexing
    3. Adds the file to the knowledge base's file list

    The function ensures that only properly processed files are added to knowledge
    bases and that the vector database is properly updated for search functionality.

    Error Handling:
    - Validates file existence and processing status
    - Propagates processing errors with descriptive messages
    - Maintains data consistency if processing fails

    Args:
        request: FastAPI request object (required for process_file function)
        kb_id: Knowledge base ID to add the file to
        file_id: File ID to add to the knowledge base
        user: User object for authorization and processing context

    Raises:
        Exception: If file not found, not processed, or vector processing fails
    """
    # Validate file exists and has data
    file = Files.get_file_by_id(file_id)
    if not file:
        raise Exception(f"File {file_id} not found")
    if not file.data:
        raise Exception(f"File {file_id} not processed")

    # Add content to the vector database
    try:
        process_file(
            request, ProcessFileForm(file_id=file_id, collection_name=kb_id), user=user
        )
    except Exception as e:
        raise Exception(str(e))

    # Add to knowledge base file list
    knowledge = Knowledges.get_knowledge_by_id(kb_id)
    if knowledge:
        data = knowledge.data or {}
        file_ids = data.get("file_ids", [])

        if file_id not in file_ids:
            file_ids.append(file_id)
            data["file_ids"] = file_ids
            Knowledges.update_knowledge_data_by_id(kb_id, data)


def remove_file_from_knowledge_base(kb_id: str, file_id: str) -> None:
    """
    Remove a file from a knowledge base with comprehensive cleanup.

    This function performs complete file removal including:
    1. Vector database cleanup (by file_id and hash)
    2. File collection cleanup
    3. Physical file deletion from storage
    4. Database record removal
    5. Knowledge base file list update

    The comprehensive cleanup prevents:
    - Orphaned vector records that consume storage
    - Orphaned physical files in data/uploads/
    - Hash conflicts when re-uploading files
    - Stale references in knowledge base metadata

    Critical Enhancement:
    Physical file deletion was added to fix a major issue where files were
    accumulating in data/uploads/ without being cleaned up, causing storage
    bloat and hash conflicts during Google Drive sync operations.

    Error Handling:
    - Continues cleanup even if individual steps fail
    - Logs errors for troubleshooting
    - Gracefully handles missing files or collections

    Args:
        kb_id: Knowledge base ID to remove the file from
        file_id: File ID to remove from the knowledge base
    """
    file = Files.get_file_by_id(file_id)
    if not file:
        return

    # Remove content from the vector database
    try:
        VECTOR_DB_CLIENT.delete(collection_name=kb_id, filter={"file_id": file_id})
    except Exception as e:
        logger.debug("This was most likely caused by bypassing embedding processing")
        logger.debug(e)
        pass

    # Also remove by hash to prevent duplicate content errors on re-upload
    try:
        if file.hash:
            VECTOR_DB_CLIENT.delete(collection_name=kb_id, filter={"hash": file.hash})
    except Exception as e:
        logger.debug("Error removing vector records by hash")
        logger.debug(e)
        pass

    try:
        # Remove the file's collection from vector database
        file_collection = f"file-{file_id}"
        if VECTOR_DB_CLIENT.has_collection(collection_name=file_collection):
            VECTOR_DB_CLIENT.delete_collection(collection_name=file_collection)
    except Exception as e:
        logger.debug("This was most likely caused by bypassing embedding processing")
        logger.debug(e)
        pass

    # Delete physical file from storage (this was missing!)
    # This critical addition prevents orphaned files from accumulating
    try:
        if file.path:
            Storage.delete_file(file.path)
            logger.debug(f"Deleted physical file: {file.path}")
    except Exception as e:
        logger.error(f"Error deleting physical file {file.path}: {e}")
        # Continue with database cleanup even if physical file deletion fails

    # Delete file from database
    Files.delete_file_by_id(file_id)

    # Remove from knowledge base file list
    knowledge = Knowledges.get_knowledge_by_id(kb_id)
    if knowledge:
        data = knowledge.data or {}
        file_ids = data.get("file_ids", [])

        if file_id in file_ids:
            file_ids.remove(file_id)
            data["file_ids"] = file_ids
            Knowledges.update_knowledge_data_by_id(kb_id, data)
