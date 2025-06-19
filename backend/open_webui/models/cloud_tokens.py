"""
Cloud Provider OAuth Token Management

This module handles secure storage and management of OAuth tokens for cloud providers.
It provides token storage, validation, refresh capabilities, and automatic expiration handling.

Key Features:
- Secure token storage with user isolation
- Automatic token refresh using refresh tokens
- Token expiration validation with buffer time
- Support for multiple cloud providers per user
- Thread-safe database operations

Supported Providers:
- Google Drive
- (Future: OneDrive, Dropbox, Box, etc.)
"""

import logging
import time
from typing import Optional
from datetime import datetime

from open_webui.internal.db import Base, get_db
from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.cloud_sync_utils import get_current_timestamp

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Integer, Column, String, DateTime, UniqueConstraint

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

####################
# Cloud Token DB Schema
####################


class CloudToken(Base):
    """
    Store cloud provider OAuth tokens for users.

    This table securely stores OAuth access and refresh tokens that allow
    the system to access cloud provider APIs on behalf of users for
    background synchronization operations.

    Security considerations:
    - Tokens are stored per user to maintain isolation
    - Unique constraint prevents duplicate tokens per user/provider
    - Expiration times are tracked for automatic refresh
    """

    __tablename__ = "cloud_tokens"

    # Primary key and user association
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False)
    provider = Column(String, nullable=False)  # 'google_drive', 'onedrive', 'dropbox'

    # OAuth token data
    access_token = Column(String, nullable=False)  # Short-lived access token
    refresh_token = Column(String, nullable=True)  # Long-lived refresh token

    # Token expiration management
    expires_at = Column(Integer, nullable=True)  # Unix timestamp

    # Audit trail
    created_at = Column(DateTime, nullable=True)

    # Ensure only one token per user per provider
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_user_provider"),)


####################
# Pydantic Models
####################


class CloudTokenModel(BaseModel):
    """Pydantic model for cloud token records"""

    id: int
    user_id: str
    provider: str
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class CloudTokenForm(BaseModel):
    """Form model for creating/updating cloud tokens"""

    provider: str
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None  # Seconds until expiration


####################
# Table Operations
####################


class CloudTokensTable:
    """Database operations for cloud provider OAuth tokens"""

    def get_token(self, user_id: str, provider: str) -> Optional[CloudTokenModel]:
        """
        Retrieve a cloud token for a specific user and provider.

        Args:
            user_id: User identifier
            provider: Cloud provider name (e.g., 'google_drive')

        Returns:
            CloudTokenModel if found, None otherwise
        """
        with get_db() as db:
            token = (
                db.query(CloudToken)
                .filter_by(user_id=user_id, provider=provider)
                .first()
            )

            if token:
                return CloudTokenModel.model_validate(token)
            return None

    def insert_token(
        self, user_id: str, form_data: CloudTokenForm
    ) -> Optional[CloudTokenModel]:
        """
        Insert or update a cloud token for a user.

        This method handles both new token creation and updates to existing tokens.
        It automatically calculates expiration times and maintains the unique
        constraint of one token per user per provider.

        Args:
            user_id: User identifier
            form_data: Token form data with access/refresh tokens

        Returns:
            CloudTokenModel if successful, None otherwise
        """
        with get_db() as db:
            # Calculate expiration timestamp
            expires_at = None
            if form_data.expires_in:
                expires_at = get_current_timestamp() + form_data.expires_in

            # Check if token already exists for this user/provider
            existing = (
                db.query(CloudToken)
                .filter_by(user_id=user_id, provider=form_data.provider)
                .first()
            )

            if existing:
                # Update existing token
                existing.access_token = form_data.access_token
                existing.refresh_token = form_data.refresh_token
                existing.expires_at = expires_at

                db.commit()
                db.refresh(existing)
                return CloudTokenModel.model_validate(existing)
            else:
                # Create new token
                new_token = CloudToken(
                    user_id=user_id,
                    provider=form_data.provider,
                    access_token=form_data.access_token,
                    refresh_token=form_data.refresh_token,
                    expires_at=expires_at,
                    created_at=datetime.now(),
                )

                db.add(new_token)
                db.commit()
                db.refresh(new_token)
                return CloudTokenModel.model_validate(new_token)

    def update_token(
        self,
        user_id: str,
        provider: str,
        access_token: str,
        refresh_token: str,
        expires_in: int,
    ) -> bool:
        """
        Update an existing token with fresh credentials.

        This method is typically called after a successful token refresh
        to store the new access token and updated expiration time.

        Args:
            user_id: User identifier
            provider: Cloud provider name
            access_token: New access token
            refresh_token: Refresh token (may be unchanged)
            expires_in: Seconds until new token expires

        Returns:
            True if update successful, False if token not found
        """
        with get_db() as db:
            token = (
                db.query(CloudToken)
                .filter_by(user_id=user_id, provider=provider)
                .first()
            )

            if token:
                token.access_token = access_token
                if refresh_token:
                    token.refresh_token = refresh_token
                token.expires_at = get_current_timestamp() + expires_in

                db.commit()
                return True
            return False

    def is_token_valid(self, token: CloudTokenModel) -> bool:
        """
        Check if a token is still valid (not expired).

        This method includes a 5-minute buffer to account for clock skew
        and network delays, ensuring tokens are refreshed before they
        actually expire.

        Args:
            token: Token to validate

        Returns:
            True if token is valid, False if expired or expiring soon
        """
        if not token.expires_at:
            return True  # No expiration set, assume valid

        current_time = get_current_timestamp()
        # Add 5 minute buffer to account for clock skew
        return token.expires_at > (current_time + 300)

    def delete_token(self, user_id: str, provider: str) -> bool:
        """
        Delete a cloud token for a user and provider.

        This is typically called when a user disconnects their cloud account
        or when tokens become permanently invalid.

        Args:
            user_id: User identifier
            provider: Cloud provider name

        Returns:
            True if deletion successful, False if token not found
        """
        with get_db() as db:
            token = (
                db.query(CloudToken)
                .filter_by(user_id=user_id, provider=provider)
                .first()
            )

            if token:
                db.delete(token)
                db.commit()
                return True
            return False

    def get_tokens_by_provider(self, provider: str):
        """
        Get all tokens for a specific provider across all users.

        This method is useful for background sync operations that need
        to process all users who have connected a specific cloud provider.

        Args:
            provider: Cloud provider name

        Returns:
            List of CloudTokenModel objects
        """
        with get_db() as db:
            tokens = db.query(CloudToken).filter_by(provider=provider).all()
            return [CloudTokenModel.model_validate(token) for token in tokens]


# Global instance for easy access throughout the application
CloudTokens = CloudTokensTable()
