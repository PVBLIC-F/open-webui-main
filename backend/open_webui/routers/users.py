import logging
from typing import Optional
import base64
import io


from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse, FileResponse
from pydantic import BaseModel


from open_webui.models.auths import Auths
from open_webui.models.oauth_sessions import OAuthSessions

from open_webui.models.groups import Groups
from open_webui.models.chats import Chats
from open_webui.models.users import (
    UserModel,
    UserListResponse,
    UserInfoListResponse,
    UserIdNameListResponse,
    UserRoleUpdateForm,
    Users,
    UserSettings,
    UserUpdateForm,
)


from open_webui.socket.main import (
    get_active_status_by_user_id,
    get_active_user_ids,
    get_user_active_status,
)
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS, STATIC_DIR


from open_webui.utils.auth import get_admin_user, get_password_hash, get_verified_user
from open_webui.utils.access_control import get_permissions, has_permission


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()


############################
# GetActiveUsers
############################


@router.get("/active")
async def get_active_users(
    user=Depends(get_verified_user),
):
    """
    Get a list of active users.
    """
    return {
        "user_ids": get_active_user_ids(),
    }


############################
# GetUsers
############################


PAGE_ITEM_COUNT = 30


@router.get("/", response_model=UserListResponse)
async def get_users(
    query: Optional[str] = None,
    order_by: Optional[str] = None,
    direction: Optional[str] = None,
    page: Optional[int] = 1,
    user=Depends(get_admin_user),
):
    limit = PAGE_ITEM_COUNT

    page = max(1, page)
    skip = (page - 1) * limit

    filter = {}
    if query:
        filter["query"] = query
    if order_by:
        filter["order_by"] = order_by
    if direction:
        filter["direction"] = direction

    return Users.get_users(filter=filter, skip=skip, limit=limit)


@router.get("/all", response_model=UserInfoListResponse)
async def get_all_users(
    user=Depends(get_admin_user),
):
    return Users.get_users()


@router.get("/search", response_model=UserIdNameListResponse)
async def search_users(
    query: Optional[str] = None,
    user=Depends(get_verified_user),
):
    limit = PAGE_ITEM_COUNT

    page = 1  # Always return the first page for search
    skip = (page - 1) * limit

    filter = {}
    if query:
        filter["query"] = query

    return Users.get_users(filter=filter, skip=skip, limit=limit)


############################
# User Groups
############################


@router.get("/groups")
async def get_user_groups(user=Depends(get_verified_user)):
    return Groups.get_groups_by_member_id(user.id)


############################
# User Permissions
############################


@router.get("/permissions")
async def get_user_permissisions(request: Request, user=Depends(get_verified_user)):
    user_permissions = get_permissions(
        user.id, request.app.state.config.USER_PERMISSIONS
    )

    return user_permissions


############################
# User Default Permissions
############################
class WorkspacePermissions(BaseModel):
    models: bool = False
    knowledge: bool = False
    prompts: bool = False
    tools: bool = False


class SharingPermissions(BaseModel):
    public_models: bool = True
    public_knowledge: bool = True
    public_prompts: bool = True
    public_tools: bool = True
    public_notes: bool = True


class ChatPermissions(BaseModel):
    controls: bool = True
    valves: bool = True
    system_prompt: bool = True
    params: bool = True
    file_upload: bool = True
    delete: bool = True
    delete_message: bool = True
    continue_response: bool = True
    regenerate_response: bool = True
    rate_response: bool = True
    edit: bool = True
    share: bool = True
    export: bool = True
    stt: bool = True
    tts: bool = True
    call: bool = True
    multiple_models: bool = True
    temporary: bool = True
    temporary_enforced: bool = False


class FeaturesPermissions(BaseModel):
    direct_tool_servers: bool = False
    web_search: bool = True
    image_generation: bool = True
    code_interpreter: bool = True
    notes: bool = True


class UserPermissions(BaseModel):
    workspace: WorkspacePermissions
    sharing: SharingPermissions
    chat: ChatPermissions
    features: FeaturesPermissions


@router.get("/default/permissions", response_model=UserPermissions)
async def get_default_user_permissions(request: Request, user=Depends(get_admin_user)):
    return {
        "workspace": WorkspacePermissions(
            **request.app.state.config.USER_PERMISSIONS.get("workspace", {})
        ),
        "sharing": SharingPermissions(
            **request.app.state.config.USER_PERMISSIONS.get("sharing", {})
        ),
        "chat": ChatPermissions(
            **request.app.state.config.USER_PERMISSIONS.get("chat", {})
        ),
        "features": FeaturesPermissions(
            **request.app.state.config.USER_PERMISSIONS.get("features", {})
        ),
    }


@router.post("/default/permissions")
async def update_default_user_permissions(
    request: Request, form_data: UserPermissions, user=Depends(get_admin_user)
):
    request.app.state.config.USER_PERMISSIONS = form_data.model_dump()
    return request.app.state.config.USER_PERMISSIONS


############################
# GetUserSettingsBySessionUser
############################


@router.get("/user/settings", response_model=Optional[UserSettings])
async def get_user_settings_by_session_user(user=Depends(get_verified_user)):
    user = Users.get_user_by_id(user.id)
    if user:
        return user.settings
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.USER_NOT_FOUND,
        )


############################
# UpdateUserSettingsBySessionUser
############################


@router.post("/user/settings/update", response_model=UserSettings)
async def update_user_settings_by_session_user(
    request: Request, form_data: UserSettings, user=Depends(get_verified_user)
):
    updated_user_settings = form_data.model_dump()
    if (
        user.role != "admin"
        and "toolServers" in updated_user_settings.get("ui").keys()
        and not has_permission(
            user.id,
            "features.direct_tool_servers",
            request.app.state.config.USER_PERMISSIONS,
        )
    ):
        # If the user is not an admin and does not have permission to use tool servers, remove the key
        updated_user_settings["ui"].pop("toolServers", None)

    user = Users.update_user_settings_by_id(user.id, updated_user_settings)
    if user:
        return user.settings
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.USER_NOT_FOUND,
        )


############################
# GetUserInfoBySessionUser
############################


@router.get("/user/info", response_model=Optional[dict])
async def get_user_info_by_session_user(user=Depends(get_verified_user)):
    user = Users.get_user_by_id(user.id)
    if user:
        return user.info
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.USER_NOT_FOUND,
        )


############################
# UpdateUserInfoBySessionUser
############################


@router.post("/user/info/update", response_model=Optional[dict])
async def update_user_info_by_session_user(
    form_data: dict, user=Depends(get_verified_user)
):
    user = Users.get_user_by_id(user.id)
    if user:
        if user.info is None:
            user.info = {}

        user = Users.update_user_by_id(user.id, {"info": {**user.info, **form_data}})
        if user:
            return user.info
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.USER_NOT_FOUND,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.USER_NOT_FOUND,
        )


############################
# GetUserById
############################


class UserResponse(BaseModel):
    name: str
    profile_image_url: str
    active: Optional[bool] = None


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(user_id: str, user=Depends(get_verified_user)):
    # Check if user_id is a shared chat
    # If it is, get the user_id from the chat
    if user_id.startswith("shared-"):
        chat_id = user_id.replace("shared-", "")
        chat = Chats.get_chat_by_id(chat_id)
        if chat:
            user_id = chat.user_id
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.USER_NOT_FOUND,
            )

    user = Users.get_user_by_id(user_id)

    if user:
        return UserResponse(
            **{
                "name": user.name,
                "profile_image_url": user.profile_image_url,
                "active": get_active_status_by_user_id(user_id),
            }
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.USER_NOT_FOUND,
        )


@router.get("/{user_id}/oauth/sessions", response_model=Optional[dict])
async def get_user_oauth_sessions_by_id(user_id: str, user=Depends(get_admin_user)):
    sessions = OAuthSessions.get_sessions_by_user_id(user_id)
    if sessions and len(sessions) > 0:
        return sessions
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.USER_NOT_FOUND,
        )


############################
# GetUserProfileImageById
############################


@router.get("/{user_id}/profile/image")
async def get_user_profile_image_by_id(user_id: str, user=Depends(get_verified_user)):
    user = Users.get_user_by_id(user_id)
    if user:
        if user.profile_image_url:
            # check if it's url or base64
            if user.profile_image_url.startswith("http"):
                return Response(
                    status_code=status.HTTP_302_FOUND,
                    headers={"Location": user.profile_image_url},
                )
            elif user.profile_image_url.startswith("data:image"):
                try:
                    header, base64_data = user.profile_image_url.split(",", 1)
                    image_data = base64.b64decode(base64_data)
                    image_buffer = io.BytesIO(image_data)

                    return StreamingResponse(
                        image_buffer,
                        media_type="image/png",
                        headers={"Content-Disposition": "inline; filename=image.png"},
                    )
                except Exception as e:
                    pass
        return FileResponse(f"{STATIC_DIR}/user.png")
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.USER_NOT_FOUND,
        )


############################
# GetUserActiveStatusById
############################


@router.get("/{user_id}/active", response_model=dict)
async def get_user_active_status_by_id(user_id: str, user=Depends(get_verified_user)):
    return {
        "active": get_user_active_status(user_id),
    }


############################
# UpdateUserById
############################


@router.post("/{user_id}/update", response_model=Optional[UserModel])
async def update_user_by_id(
    user_id: str,
    form_data: UserUpdateForm,
    session_user=Depends(get_admin_user),
):
    log.info(f"Updating user {user_id} with data: {form_data.model_dump()}")
    log.info(f"Gmail sync enabled value: {form_data.gmail_sync_enabled}")
    # Prevent modification of the primary admin user by other admins
    try:
        first_user = Users.get_first_user()
        if first_user:
            if user_id == first_user.id:
                if session_user.id != user_id:
                    # If the user trying to update is the primary admin, and they are not the primary admin themselves
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=ERROR_MESSAGES.ACTION_PROHIBITED,
                    )

                if form_data.role != "admin":
                    # If the primary admin is trying to change their own role, prevent it
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=ERROR_MESSAGES.ACTION_PROHIBITED,
                    )

    except Exception as e:
        log.error(f"Error checking primary admin status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not verify primary admin status.",
        )

    user = Users.get_user_by_id(user_id)

    if user:
        if form_data.email.lower() != user.email:
            email_user = Users.get_user_by_email(form_data.email.lower())
            if email_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.EMAIL_TAKEN,
                )

        if form_data.password:
            hashed = get_password_hash(form_data.password)
            log.debug(f"hashed: {hashed}")
            Auths.update_user_password_by_id(user_id, hashed)

        Auths.update_email_by_id(user_id, form_data.email.lower())
        # Prepare update data
        update_data = {
            "role": form_data.role,
            "name": form_data.name,
            "email": form_data.email.lower(),
            "profile_image_url": form_data.profile_image_url,
        }
        
        # Add gmail_sync_enabled if provided
        if form_data.gmail_sync_enabled is not None:
            update_data["gmail_sync_enabled"] = form_data.gmail_sync_enabled
            log.info(f"Updating gmail_sync_enabled for user {user_id}: {form_data.gmail_sync_enabled}")
        
        updated_user = Users.update_user_by_id(user_id, update_data)

        if updated_user:
            return updated_user

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(),
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=ERROR_MESSAGES.USER_NOT_FOUND,
    )


############################
# DeleteUserById
############################


@router.delete("/{user_id}", response_model=bool)
async def delete_user_by_id(user_id: str, user=Depends(get_admin_user)):
    # Prevent deletion of the primary admin user
    try:
        first_user = Users.get_first_user()
        if first_user and user_id == first_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=ERROR_MESSAGES.ACTION_PROHIBITED,
            )
    except Exception as e:
        log.error(f"Error checking primary admin status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not verify primary admin status.",
        )

    if user.id != user_id:
        result = Auths.delete_auth_by_id(user_id)

        if result:
            return True

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DELETE_USER_ERROR,
        )

    # Prevent self-deletion
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=ERROR_MESSAGES.ACTION_PROHIBITED,
    )


############################
# GetUserGroupsById
############################


@router.get("/{user_id}/groups")
async def get_user_groups_by_id(user_id: str, user=Depends(get_admin_user)):
    return Groups.get_groups_by_member_id(user_id)


@router.post("/{user_id}/gmail-sync/reset")
async def reset_user_gmail_sync(user_id: str, user=Depends(get_admin_user)):
    """
    Reset Gmail sync status for a user (admin only).
    
    This will trigger a full sync on the next periodic sync cycle.
    Use this when a user has deleted their Pinecone records or needs to rebuild their index.
    """
    from open_webui.models.gmail_sync import gmail_sync_status
    
    sync_status = gmail_sync_status.get_sync_status(user_id)
    if not sync_status:
        return {
            "success": False,
            "message": "No sync status found for this user"
        }
    
    # Reset sync status to trigger full sync
    updated = gmail_sync_status.update_sync_status(
        user_id=user_id,
        last_sync_timestamp=None,
        sync_status="never",
        total_emails_synced=0,
        last_sync_count=0,
        error_count=0,
        last_error=None
    )
    
    if updated:
        return {
            "success": True,
            "message": f"Gmail sync reset for user {user_id}. Full sync will occur on next cycle.",
            "user_id": user_id,
            "next_sync": "Will be triggered by periodic scheduler (within 30 minutes)"
        }
    else:
        return {
            "success": False,
            "message": "Failed to reset sync status"
        }


@router.get("/{user_id}/gmail-sync-status")
async def get_user_gmail_sync_status(user_id: str, user=Depends(get_admin_user)):
    """
    Get Gmail sync status and metrics for a user (admin only).
    
    Returns comprehensive sync information including:
    - Current sync status and configuration
    - Performance metrics (emails synced, duration)
    - Error tracking and diagnostics
    - Next sync prediction
    """
    from open_webui.models.gmail_sync import gmail_sync_status
    from datetime import datetime, timedelta
    import time
    
    sync_status = gmail_sync_status.get_sync_status(user_id)
    if not sync_status:
        return {
            "user_id": user_id,
            "sync_enabled": False,
            "status": "never",
            "message": "No sync status found - user may not have Gmail connected",
            "last_sync": None,
            "next_sync_estimate": None
        }
    
    # Calculate human-readable last sync time
    last_sync_human = None
    next_sync_estimate = None
    if sync_status.last_sync_timestamp:
        last_sync_dt = datetime.fromtimestamp(sync_status.last_sync_timestamp)
        last_sync_human = last_sync_dt.isoformat()
        
        # Estimate next sync time based on frequency
        next_sync_dt = last_sync_dt + timedelta(hours=sync_status.sync_frequency_hours)
        next_sync_estimate = next_sync_dt.isoformat()
        time_until_sync = (next_sync_dt - datetime.now()).total_seconds()
        hours_until = max(0, time_until_sync / 3600)
    else:
        hours_until = 0
    
    # Calculate sync health score (0-100)
    health_score = 100
    if sync_status.error_count > 0:
        health_score -= min(50, sync_status.error_count * 10)  # -10 per error, max -50
    if sync_status.sync_status == "error":
        health_score -= 30
    if sync_status.sync_status == "paused":
        health_score = 0
    
    return {
        "user_id": user_id,
        "sync_enabled": sync_status.sync_enabled,
        "auto_sync_enabled": sync_status.auto_sync_enabled,
        "status": sync_status.sync_status,
        
        # Timestamps
        "last_sync_timestamp": sync_status.last_sync_timestamp,
        "last_sync": last_sync_human,
        "next_sync_estimate": next_sync_estimate,
        "hours_until_next_sync": round(hours_until, 1) if next_sync_estimate else None,
        
        # Performance metrics
        "last_sync_count": sync_status.last_sync_count,
        "last_sync_duration_seconds": sync_status.last_sync_duration,
        "total_emails_synced": sync_status.total_emails_synced,
        
        # Error tracking
        "error_count": sync_status.error_count,
        "last_error": sync_status.last_error,
        "health_score": max(0, health_score),
        
        # Configuration
        "sync_frequency_hours": sync_status.sync_frequency_hours,
        "max_emails_per_sync": sync_status.max_emails_per_sync,
        
        # Diagnostics
        "created_at": sync_status.created_at,
        "updated_at": sync_status.updated_at
    }


@router.post("/{user_id}/gmail-sync/force")
async def force_gmail_sync(user_id: str, request: Request, user=Depends(get_admin_user)):
    """
    Force a full Gmail sync for a user (admin only).
    
    This will:
    1. Reset the sync status to trigger a full sync
    2. Start a background sync task immediately
    3. Sync all emails (not just new ones)
    
    Use this when:
    - User wants to re-index all their emails
    - Sync status is stuck or corrupted
    - User deleted their vector DB data
    """
    from open_webui.models.gmail_sync import gmail_sync_status
    from open_webui.models.oauth_sessions import OAuthSessions
    from open_webui.utils.gmail_auto_sync import trigger_gmail_sync_if_needed
    
    # Validate user exists
    target_user = Users.get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if user has Gmail sync enabled
    if not getattr(target_user, 'gmail_sync_enabled', 0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gmail sync is not enabled for this user"
        )
    
    # Check if user has Google OAuth session
    oauth_session = OAuthSessions.get_session_by_provider_and_user_id("google", user_id)
    if not oauth_session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has not connected their Google account"
        )
    
    # Validate that token exists and has required fields
    if not oauth_session.token or not oauth_session.token.get("access_token"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth session is missing access token"
        )
    
    # Reset sync status to force full sync
    sync_status = gmail_sync_status.get_sync_status(user_id)
    if sync_status:
        gmail_sync_status.update_sync_status(
            user_id=user_id,
            last_sync_timestamp=None,  # Reset to trigger full sync
            last_sync_history_id=None,
            sync_status="pending",
            last_sync_count=0,
        )
        log.info(f"Reset Gmail sync status for user {user_id} - will do full sync")
    
    # Prepare OAuth token from session (token is a dict with access_token, refresh_token, etc.)
    oauth_token = oauth_session.token
    
    # Trigger sync immediately in background
    try:
        await trigger_gmail_sync_if_needed(
            request=request,
            user_id=user_id,
            provider="google",
            token=oauth_token,
            is_new_user=False
        )
        
        return {
            "success": True,
            "message": "Full Gmail sync started in background",
            "user_id": user_id,
            "sync_type": "full",
            "note": "This may take several minutes. Check sync status for progress."
        }
    except Exception as e:
        log.error(f"Error triggering Gmail sync for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start Gmail sync: {str(e)}"
        )
