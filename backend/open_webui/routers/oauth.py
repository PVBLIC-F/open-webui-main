"""
Google Drive OAuth Router

This module handles OAuth 2.0 authentication flow for Google Drive integration.
It provides endpoints for exchanging authorization codes for tokens, checking token status,
and refreshing expired tokens for background synchronization.

Key Features:
- OAuth 2.0 authorization code exchange
- Secure token storage with refresh capability
- Token validation and expiration checking
- Automatic token refresh for background operations
- Debug endpoints for troubleshooting

Security Notes:
- Refresh tokens are captured and stored for background sync
- All tokens are associated with specific users for isolation
- Tokens are automatically refreshed before expiration
- Debug endpoints include detailed logging for troubleshooting

Endpoints:
- POST /api/oauth/google-drive/exchange: Exchange auth code for tokens
- GET /api/oauth/google-drive/status: Check current token status
- POST /api/oauth/google-drive/refresh: Manually refresh tokens
"""

import logging
import aiohttp
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from open_webui.models.cloud_tokens import CloudTokens, CloudTokenForm
from open_webui.utils.auth import get_verified_user
from open_webui.config import GOOGLE_DRIVE_CLIENT_ID, GOOGLE_DRIVE_CLIENT_SECRET

log = logging.getLogger(__name__)
router = APIRouter()


class CodeExchangeRequest(BaseModel):
    """Request model for OAuth code exchange"""

    code: str


@router.post("/api/oauth/google-drive/exchange")
async def exchange_google_drive_code(
    request: CodeExchangeRequest, user=Depends(get_verified_user)
):
    """
    Exchange Google authorization code for access token and refresh token.

    This is the critical step that captures the refresh token needed for background sync.
    The refresh token allows the system to obtain new access tokens without user interaction.

    Args:
        request: Contains the authorization code from Google OAuth flow
        user: Authenticated user from dependency injection

    Returns:
        dict: Contains the access token for immediate use

    Raises:
        HTTPException: If OAuth credentials not configured, code exchange fails,
                      or token storage fails
    """
    try:
        log.info(
            f"🔍 DEBUG: Starting token exchange for user {user.id}, code length: {len(request.code) if request.code else 0}"
        )

        if not GOOGLE_DRIVE_CLIENT_ID.value or not GOOGLE_DRIVE_CLIENT_SECRET.value:
            log.error("🔍 DEBUG: Google Drive credentials not configured")
            raise HTTPException(
                status_code=500, detail="Google Drive OAuth not configured"
            )

        log.info(f"🔍 DEBUG: Using client_id: {GOOGLE_DRIVE_CLIENT_ID.value[:20]}...")

        # Exchange authorization code for tokens
        data = {
            "client_id": GOOGLE_DRIVE_CLIENT_ID.value,
            "client_secret": GOOGLE_DRIVE_CLIENT_SECRET.value,
            "code": request.code,
            "grant_type": "authorization_code",
            "redirect_uri": "postmessage",  # Special value for popup flows
        }

        log.info("🔍 DEBUG: Making request to Google token endpoint...")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://oauth2.googleapis.com/token", data=data
            ) as response:
                response_text = await response.text()
                log.info(f"🔍 DEBUG: Google response status: {response.status}")
                log.info(f"🔍 DEBUG: Google response text: {response_text[:500]}...")

                if response.status != 200:
                    log.error(
                        f"🔍 DEBUG: Google token exchange failed: {response.status} - {response_text}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to exchange authorization code: {response_text}",
                    )

                try:
                    token_data = await response.json()
                except:
                    # Already read as text, parse manually
                    import json

                    token_data = json.loads(response_text)

        log.info(f"🔍 DEBUG: Token data keys: {list(token_data.keys())}")

        # Extract tokens
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        log.info(
            f"🔍 DEBUG: Extracted tokens - access_token: {bool(access_token)}, refresh_token: {bool(refresh_token)}, expires_in: {expires_in}"
        )

        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")

        log.info(f"🔍 DEBUG: Storing tokens in database...")

        # Store tokens in database
        token = CloudTokens.insert_token(
            user_id=user.id,
            form_data=CloudTokenForm(
                provider="google_drive",
                access_token=access_token,
                refresh_token=refresh_token,  # This is the key - capturing refresh token!
                expires_in=expires_in,  # Use expires_in, not expires_at
            ),
        )

        if token:
            log.info(
                f"🔍 DEBUG: Successfully stored Google Drive tokens for user {user.id} (refresh_token: {'Yes' if refresh_token else 'No'})"
            )
            return {"access_token": access_token}
        else:
            log.error("🔍 DEBUG: Failed to store tokens in database")
            raise HTTPException(status_code=500, detail="Failed to store tokens")

    except HTTPException:
        raise
    except Exception as e:
        log.error(
            f"🔍 DEBUG: Unexpected error exchanging Google Drive code for user {user.id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to exchange code: {str(e)}"
        )


@router.get("/api/oauth/google-drive/status")
async def get_google_drive_token_status(user=Depends(get_verified_user)):
    """
    Debug endpoint to check Google Drive token status for current user.

    This endpoint provides detailed information about the user's stored tokens,
    including expiration status and refresh token availability. Useful for
    troubleshooting sync issues and verifying OAuth setup.

    Args:
        user: Authenticated user from dependency injection

    Returns:
        dict: Detailed token status information including:
            - has_token: Whether user has any stored token
            - has_refresh_token: Whether refresh token is available
            - expires_at: Token expiration timestamp
            - is_valid: Whether token is currently valid
            - seconds_until_expiry: Time until token expires
    """
    try:
        token = CloudTokens.get_token(user.id, "google_drive")
        if not token:
            return {
                "has_token": False,
                "user_id": user.id,
                "message": "No Google Drive token found for user",
            }

        import time

        current_time = int(time.time())
        is_valid = CloudTokens.is_token_valid(token)

        return {
            "has_token": True,
            "user_id": user.id,
            "has_refresh_token": bool(token.refresh_token),
            "expires_at": token.expires_at,
            "current_time": current_time,
            "seconds_until_expiry": (
                token.expires_at - current_time if token.expires_at else None
            ),
            "is_valid": is_valid,
            "created_at": token.created_at.isoformat() if token.created_at else None,
        }

    except Exception as e:
        log.error(
            f"Error checking Google Drive token status for user {user.id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to check token status: {str(e)}"
        )


@router.post("/api/oauth/google-drive/refresh")
async def refresh_google_drive_token(user=Depends(get_verified_user)):
    """
    Manually refresh a Google Drive token using the stored refresh token.

    This endpoint allows manual token refresh for testing and troubleshooting.
    In normal operation, tokens are automatically refreshed by the sync worker
    when they approach expiration.

    Args:
        user: Authenticated user from dependency injection

    Returns:
        dict: Contains the new access token

    Raises:
        HTTPException: If no token found, no refresh token available,
                      or refresh request fails
    """
    try:
        token = CloudTokens.get_token(user.id, "google_drive")
        if not token:
            raise HTTPException(
                status_code=404, detail="No Google Drive token found for user"
            )

        if not token.refresh_token:
            raise HTTPException(status_code=400, detail="No refresh token available")

        # Use the existing refresh logic from sync providers
        refresh_data = {
            "client_id": GOOGLE_DRIVE_CLIENT_ID.value,
            "client_secret": GOOGLE_DRIVE_CLIENT_SECRET.value,
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
                        user.id,
                        "google_drive",
                        token_data["access_token"],
                        token.refresh_token,
                        token_data.get("expires_in", 3600),
                    )

                    log.info(
                        f"Successfully refreshed Google Drive token for user {user.id}"
                    )
                    return {"access_token": token_data["access_token"]}
                else:
                    log.error(
                        f"Failed to refresh Google Drive token: {response.status}"
                    )
                    raise HTTPException(
                        status_code=400, detail="Failed to refresh token"
                    )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error refreshing Google Drive token for user {user.id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to refresh token: {str(e)}"
        )
