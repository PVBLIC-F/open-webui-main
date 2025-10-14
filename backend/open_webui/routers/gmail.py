"""
Gmail Sync API Router

Provides admin endpoints for managing user-level Gmail sync settings and operations.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status

from open_webui.models.users import Users
from open_webui.models.oauth_sessions import OAuthSessions
from open_webui.utils.auth import get_admin_user
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.tasks import create_task

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()


@router.get("/api/users/{user_id}/gmail/status")
async def get_gmail_status(
    user_id: str,
    request: Request,
    admin=Depends(get_admin_user),
):
    """
    Get Gmail sync status for a user.
    
    Returns sync status, last sync time, email counts, etc.
    """
    
    user = Users.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get Gmail settings from user settings
    gmail_settings = {}
    if user.settings:
        settings_dict = user.settings.model_dump() if hasattr(user.settings, 'model_dump') else user.settings
        gmail_settings = settings_dict.get("gmail", {}) if isinstance(settings_dict, dict) else {}
    
    # Check if user has Google OAuth session
    oauth_session = OAuthSessions.get_session_by_provider_and_user_id("google", user_id)
    has_gmail_oauth = oauth_session is not None
    
    # Check if OAuth token has Gmail scopes
    has_gmail_scopes = False
    if oauth_session:
        token_scope = oauth_session.token.get("scope", "")
        has_gmail_scopes = "gmail" in token_scope
    
    return {
        "sync_enabled": gmail_settings.get("sync_enabled", False),
        "sync_status": gmail_settings.get("sync_status", "not_connected"),
        "last_synced_at": gmail_settings.get("last_synced_at"),
        "total_emails_indexed": gmail_settings.get("total_emails_indexed", 0),
        "total_vectors": gmail_settings.get("total_vectors", 0),
        "has_gmail_oauth": has_gmail_oauth,
        "has_gmail_scopes": has_gmail_scopes,
    }


@router.post("/api/users/{user_id}/gmail/enable")
async def enable_gmail_sync(
    user_id: str,
    request: Request,
    admin=Depends(get_admin_user),
):
    """
    Enable Gmail sync for a user.
    
    Sets sync_enabled to True in user settings.
    Does NOT trigger sync - user must click "Sync Now" separately.
    """
    
    user = Users.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if user has Google OAuth with Gmail scopes
    oauth_session = OAuthSessions.get_session_by_provider_and_user_id("google", user_id)
    if not oauth_session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must log in with Google OAuth first"
        )
    
    token_scope = oauth_session.token.get("scope", "")
    if "gmail" not in token_scope:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User's Google OAuth token doesn't include Gmail scopes"
        )
    
    # Update user settings
    user_settings = user.settings.model_dump() if user.settings and hasattr(user.settings, 'model_dump') else (user.settings if isinstance(user.settings, dict) else {})
    existing_gmail = user_settings.get("gmail", {}) if isinstance(user_settings, dict) else {}
    
    user_settings["gmail"] = {
        "sync_enabled": True,
        "sync_status": "ready",
        "last_synced_at": existing_gmail.get("last_synced_at"),
        "total_emails_indexed": existing_gmail.get("total_emails_indexed", 0),
        "total_vectors": existing_gmail.get("total_vectors", 0),
    }
    
    Users.update_user_by_id(user_id, {"settings": user_settings})
    
    logger.info(f"✅ Gmail sync enabled for user {user_id}")
    
    return {
        "status": "enabled",
        "message": "Gmail sync enabled. Click 'Sync Now' to start indexing emails.",
        "settings": user_settings["gmail"]
    }


@router.post("/api/users/{user_id}/gmail/sync-now")
async def trigger_gmail_sync(
    user_id: str,
    request: Request,
    admin=Depends(get_admin_user),
):
    """
    Manually trigger Gmail sync for a user.
    
    Requires:
    - User has Gmail sync enabled
    - User has valid Google OAuth session with Gmail scopes
    """
    
    user = Users.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if Gmail sync is enabled
    settings_dict = user.settings.model_dump() if user.settings and hasattr(user.settings, 'model_dump') else (user.settings if isinstance(user.settings, dict) else {})
    gmail_settings = settings_dict.get("gmail", {}) if isinstance(settings_dict, dict) else {}
    if not gmail_settings.get("sync_enabled", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gmail sync is not enabled for this user"
        )
    
    # Get OAuth session
    oauth_session = OAuthSessions.get_session_by_provider_and_user_id("google", user_id)
    if not oauth_session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Google OAuth session found. User must log in with Google first."
        )
    
    # Validate Gmail scopes
    token_scope = oauth_session.token.get("scope", "")
    if "gmail" not in token_scope:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth token doesn't include Gmail scopes"
        )
    
    # Update status to syncing
    user_settings = user.settings.model_dump() if user.settings and hasattr(user.settings, 'model_dump') else (user.settings if isinstance(user.settings, dict) else {})
    user_settings["gmail"] = {
        **gmail_settings,
        "sync_status": "syncing",
    }
    Users.update_user_by_id(user_id, {"settings": user_settings})
    
    # Trigger background sync task
    try:
        from open_webui.utils.gmail_auto_sync import _background_gmail_sync
        
        task_id, task = await create_task(
            request.app.state.redis,
            _background_gmail_sync(request, user_id, oauth_session.token),
            id=f"gmail_sync_{user_id}"
        )
        
        logger.info(f"🚀 Manual Gmail sync triggered for user {user_id}, task_id={task_id}")
        
        return {
            "status": "syncing",
            "message": f"Gmail sync started in background (task: {task_id})",
            "task_id": task_id
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger Gmail sync for user {user_id}: {e}")
        
        # Reset status to ready on error
        user_settings["gmail"]["sync_status"] = "error"
        Users.update_user_by_id(user_id, {"settings": user_settings})
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start Gmail sync: {str(e)}"
        )


@router.post("/api/users/{user_id}/gmail/disable")
async def disable_gmail_sync(
    user_id: str,
    request: Request,
    admin=Depends(get_admin_user),
):
    """
    Disable Gmail sync for a user.
    
    Sets sync_enabled to False. Does not delete existing data.
    """
    
    user = Users.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update user settings
    user_settings = user.settings.model_dump() if user.settings and hasattr(user.settings, 'model_dump') else (user.settings if isinstance(user.settings, dict) else {})
    gmail_settings = user_settings.get("gmail", {}) if isinstance(user_settings, dict) else {}
    
    user_settings["gmail"] = {
        **gmail_settings,
        "sync_enabled": False,
        "sync_status": "disabled",
    }
    
    Users.update_user_by_id(user_id, {"settings": user_settings})
    
    logger.info(f"🛑 Gmail sync disabled for user {user_id}")
    
    return {
        "status": "disabled",
        "message": "Gmail sync disabled. Existing email data is preserved."
    }


@router.delete("/api/users/{user_id}/gmail/data")
async def delete_gmail_data(
    user_id: str,
    request: Request,
    admin=Depends(get_admin_user),
):
    """
    Delete all Gmail data for a user from Pinecone.
    
    Removes all vectors and disables sync.
    """
    
    user = Users.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    collection_name = f"gmail_{user_id}"
    
    try:
        # Delete all email vectors for this user
        VECTOR_DB_CLIENT.delete(
            collection_name=collection_name,
            filter={"type": "email", "user_id": user_id}
        )
        
        logger.info(f"🗑️ Deleted all Gmail data for user {user_id} from collection {collection_name}")
        
        # Update user settings
        user_settings = user.settings.model_dump() if user.settings and hasattr(user.settings, 'model_dump') else (user.settings if isinstance(user.settings, dict) else {})
        user_settings["gmail"] = {
            "sync_enabled": False,
            "sync_status": "not_connected",
            "last_synced_at": None,
            "total_emails_indexed": 0,
            "total_vectors": 0,
        }
        
        Users.update_user_by_id(user_id, {"settings": user_settings})
        
        return {
            "status": "deleted",
            "message": "All Gmail data has been deleted from Pinecone"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete Gmail data for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete Gmail data: {str(e)}"
        )

