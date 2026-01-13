"""
Google Drive API Client

Provides methods to interact with Google Drive API for Knowledge base sync.
Reuses existing OAuth infrastructure from Open WebUI.
"""

import logging
import aiohttp
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from open_webui.env import SRC_LOG_LEVELS, AIOHTTP_CLIENT_SESSION_SSL

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

# Google Drive API endpoints
DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"

# Supported file types for RAG processing
SUPPORTED_MIME_TYPES = {
    # Documents
    "application/pdf": ".pdf",
    "application/vnd.google-apps.document": ".docx",  # Google Docs -> export as docx
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    # Spreadsheets
    "application/vnd.google-apps.spreadsheet": ".xlsx",  # Google Sheets -> export as xlsx
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    # Presentations
    "application/vnd.google-apps.presentation": ".pptx",  # Google Slides -> export as pptx
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    # Text
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/markdown": ".md",
    "text/html": ".html",
    # Other
    "application/json": ".json",
    "application/xml": ".xml",
    "text/xml": ".xml",
}

# Google Workspace export formats
GOOGLE_EXPORT_FORMATS = {
    "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


@dataclass
class DriveFile:
    """Represents a file from Google Drive"""

    id: str
    name: str
    mime_type: str
    size: Optional[int] = None
    md5_checksum: Optional[str] = None
    modified_time: Optional[str] = None
    parents: Optional[List[str]] = None
    trashed: bool = False

    @classmethod
    def from_api_response(cls, data: dict) -> "DriveFile":
        """Create DriveFile from Google Drive API response"""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            mime_type=data.get("mimeType", ""),
            size=int(data.get("size", 0)) if data.get("size") else None,
            md5_checksum=data.get("md5Checksum"),
            modified_time=data.get("modifiedTime"),
            parents=data.get("parents"),
            trashed=data.get("trashed", False),
        )

    def is_supported(self) -> bool:
        """Check if file type is supported for RAG processing"""
        return self.mime_type in SUPPORTED_MIME_TYPES

    def is_google_workspace(self) -> bool:
        """Check if file is a Google Workspace file (Docs, Sheets, Slides)"""
        return self.mime_type.startswith("application/vnd.google-apps.")


@dataclass
class DriveFolder:
    """Represents a folder from Google Drive"""

    id: str
    name: str
    parents: Optional[List[str]] = None

    @classmethod
    def from_api_response(cls, data: dict) -> "DriveFolder":
        """Create DriveFolder from Google Drive API response"""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            parents=data.get("parents"),
        )


class GoogleDriveClient:
    """
    Google Drive API client for Knowledge base sync.
    
    Reuses Open WebUI's OAuth infrastructure for token management.
    Supports listing folders, downloading files, and tracking changes.
    """

    def __init__(
        self,
        access_token: str,
        timeout: int = 30,
        max_retries: int = 3,
        token_refresh_callback=None,
    ):
        """
        Initialize Google Drive client.
        
        Args:
            access_token: OAuth2 access token with Drive read scope
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            token_refresh_callback: Async callback to refresh token when expired
                                   Should return new access_token or None
        """
        self.access_token = access_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.token_refresh_callback = token_refresh_callback
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                trust_env=True,
            )
        return self._session

    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_headers(self) -> dict:
        """Get request headers with authorization"""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        url: str,
        params: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """
        Make an authenticated request to Google Drive API.
        
        Handles token refresh and retries automatically.
        """
        session = await self._get_session()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                async with session.request(
                    method,
                    url,
                    headers=self._get_headers(),
                    params=params,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                    **kwargs,
                ) as response:
                    # Handle token expiry
                    if response.status == 401:
                        if self.token_refresh_callback and attempt < self.max_retries - 1:
                            log.info("Token expired, attempting refresh...")
                            new_token = await self.token_refresh_callback()
                            if new_token:
                                self.access_token = new_token
                                continue
                        raise Exception("OAuth token expired or invalid")

                    if response.status == 403:
                        error_data = await response.json()
                        error_msg = error_data.get("error", {}).get("message", "Access denied")
                        raise Exception(f"Drive API access denied: {error_msg}")

                    if response.status == 404:
                        raise Exception("Resource not found")

                    if response.status >= 400:
                        error_text = await response.text()
                        raise Exception(f"Drive API error {response.status}: {error_text[:200]}")

                    return await response.json()

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise

        raise last_error or Exception("Request failed after retries")

    async def _download_request(
        self,
        url: str,
        params: Optional[dict] = None,
    ) -> bytes:
        """Download binary content from Google Drive"""
        session = await self._get_session()

        for attempt in range(self.max_retries):
            try:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as response:
                    if response.status == 401:
                        if self.token_refresh_callback and attempt < self.max_retries - 1:
                            new_token = await self.token_refresh_callback()
                            if new_token:
                                self.access_token = new_token
                                continue
                        raise Exception("OAuth token expired")

                    if response.status >= 400:
                        raise Exception(f"Download failed: {response.status}")

                    return await response.read()

            except aiohttp.ClientError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        raise Exception("Download failed after retries")

    async def get_folder_info(self, folder_id: str) -> DriveFolder:
        """
        Get information about a folder.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            DriveFolder with folder details
        """
        url = f"{DRIVE_API_BASE}/files/{folder_id}"
        params = {"fields": "id,name,parents"}

        data = await self._request("GET", url, params=params)
        return DriveFolder.from_api_response(data)

    async def list_folder_contents(
        self,
        folder_id: str,
        include_trashed: bool = False,
        page_size: int = 100,
        page_token: Optional[str] = None,
    ) -> tuple[List[DriveFile], Optional[str]]:
        """
        List all files in a folder (non-recursive).
        
        Args:
            folder_id: Google Drive folder ID
            include_trashed: Include trashed files
            page_size: Number of files per page
            page_token: Token for pagination
            
        Returns:
            Tuple of (list of DriveFile, next_page_token or None)
        """
        url = f"{DRIVE_API_BASE}/files"

        # Build query for files in folder
        query = f"'{folder_id}' in parents"
        if not include_trashed:
            query += " and trashed = false"

        params = {
            "q": query,
            "fields": "nextPageToken,files(id,name,mimeType,size,md5Checksum,modifiedTime,parents,trashed)",
            "pageSize": page_size,
            "orderBy": "modifiedTime desc",
        }
        if page_token:
            params["pageToken"] = page_token

        data = await self._request("GET", url, params=params)

        files = [
            DriveFile.from_api_response(f)
            for f in data.get("files", [])
            # Skip folders, only return files
            if f.get("mimeType") != "application/vnd.google-apps.folder"
        ]

        return files, data.get("nextPageToken")

    async def list_all_folder_files(
        self,
        folder_id: str,
        include_trashed: bool = False,
        max_files: int = 1000,
    ) -> List[DriveFile]:
        """
        List all files in a folder (handles pagination).
        
        Args:
            folder_id: Google Drive folder ID
            include_trashed: Include trashed files
            max_files: Maximum files to return
            
        Returns:
            List of DriveFile objects
        """
        all_files = []
        page_token = None

        while len(all_files) < max_files:
            files, page_token = await self.list_folder_contents(
                folder_id,
                include_trashed=include_trashed,
                page_token=page_token,
            )
            all_files.extend(files)

            if not page_token:
                break

        return all_files[:max_files]

    async def get_file_metadata(self, file_id: str) -> DriveFile:
        """
        Get metadata for a specific file.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            DriveFile with file details
        """
        url = f"{DRIVE_API_BASE}/files/{file_id}"
        params = {
            "fields": "id,name,mimeType,size,md5Checksum,modifiedTime,parents,trashed"
        }

        data = await self._request("GET", url, params=params)
        return DriveFile.from_api_response(data)

    async def download_file(self, file: DriveFile) -> bytes:
        """
        Download a file's content.
        
        For Google Workspace files (Docs, Sheets, Slides), exports to
        Office format (docx, xlsx, pptx).
        
        Args:
            file: DriveFile to download
            
        Returns:
            File content as bytes
        """
        if file.is_google_workspace():
            # Export Google Workspace file
            export_mime = GOOGLE_EXPORT_FORMATS.get(file.mime_type)
            if not export_mime:
                raise Exception(f"Unsupported Google Workspace type: {file.mime_type}")

            url = f"{DRIVE_API_BASE}/files/{file.id}/export"
            params = {"mimeType": export_mime}
        else:
            # Download regular file
            url = f"{DRIVE_API_BASE}/files/{file.id}"
            params = {"alt": "media"}

        return await self._download_request(url, params=params)

    async def get_start_page_token(self) -> str:
        """
        Get the start page token for tracking changes.
        
        Call this after initial sync to get a token for incremental updates.
        
        Returns:
            Start page token string
        """
        url = f"{DRIVE_API_BASE}/changes/startPageToken"
        data = await self._request("GET", url)
        return data.get("startPageToken", "")

    async def list_changes(
        self,
        page_token: str,
        page_size: int = 100,
    ) -> tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
        """
        List changes since the given page token.
        
        Used for incremental sync - only fetches files that have changed.
        
        Args:
            page_token: Start page token from previous sync
            page_size: Number of changes per page
            
        Returns:
            Tuple of (changes list, next_page_token or None, new_start_page_token or None)
            - next_page_token: Used for pagination (None when on last page)
            - new_start_page_token: Token for next sync (only on last page)
        """
        url = f"{DRIVE_API_BASE}/changes"
        params = {
            "pageToken": page_token,
            "pageSize": page_size,
            "fields": "nextPageToken,newStartPageToken,changes(fileId,removed,file(id,name,mimeType,size,md5Checksum,modifiedTime,parents,trashed))",
            "includeRemoved": "true",
            "spaces": "drive",
        }

        data = await self._request("GET", url, params=params)

        return (
            data.get("changes", []),
            data.get("nextPageToken"),  # For pagination
            data.get("newStartPageToken"),  # For next sync (only on last page)
        )

    async def get_all_changes(
        self,
        page_token: str,
        folder_id: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Get all changes since page token (handles pagination).
        
        Args:
            page_token: Start page token
            folder_id: Optional folder ID to filter changes
            
        Returns:
            Tuple of (list of changes, new start page token)
        """
        all_changes = []
        current_token = page_token
        new_start_token = page_token

        while True:
            changes, next_page_token, new_start_page_token = await self.list_changes(
                current_token
            )
            
            # Filter changes to only include files in the specified folder
            if folder_id:
                filtered_changes = []
                for change in changes:
                    file_data = change.get("file", {})
                    parents = file_data.get("parents", [])
                    if folder_id in parents or change.get("removed"):
                        filtered_changes.append(change)
                all_changes.extend(filtered_changes)
            else:
                all_changes.extend(changes)

            # Store the new start token when we reach the last page
            if new_start_page_token:
                new_start_token = new_start_page_token

            # Continue pagination if there's a next page
            if not next_page_token:
                break

            current_token = next_page_token

        return all_changes, new_start_token

    async def check_folder_access(self, folder_id: str) -> bool:
        """
        Check if we have access to a folder.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            True if accessible, False otherwise
        """
        try:
            await self.get_folder_info(folder_id)
            return True
        except Exception as e:
            log.warning(f"Cannot access folder {folder_id}: {e}")
            return False


async def create_drive_client_for_user(
    oauth_manager,
    user_id: str,
) -> Optional[GoogleDriveClient]:
    """
    Create a Google Drive client for a user using their OAuth session.
    
    Args:
        oauth_manager: Open WebUI's OAuthManager instance
        user_id: User ID
        
    Returns:
        GoogleDriveClient or None if no valid session
    """
    from open_webui.models.oauth_sessions import OAuthSessions

    # Get Google OAuth session
    oauth_session = OAuthSessions.get_session_by_provider_and_user_id("google", user_id)
    if not oauth_session:
        log.warning(f"No Google OAuth session for user {user_id}")
        return None

    # Get token (auto-refreshes if needed)
    try:
        token = await oauth_manager.get_oauth_token(
            user_id=user_id,
            session_id=oauth_session.id,
        )
    except Exception as e:
        log.error(f"Failed to get OAuth token for user {user_id}: {e}")
        return None

    if not token or not token.get("access_token"):
        log.warning(f"Invalid OAuth token for user {user_id}")
        return None

    # Check for Drive scope
    scope = token.get("scope", "")
    if "drive" not in scope.lower():
        log.warning(f"User {user_id} OAuth token missing Drive scope")
        return None

    # Create token refresh callback
    async def refresh_callback():
        try:
            refreshed = await oauth_manager.get_oauth_token(
                user_id=user_id,
                session_id=oauth_session.id,
                force_refresh=True,
            )
            if refreshed:
                return refreshed.get("access_token")
        except Exception as e:
            log.error(f"Token refresh failed: {e}")
        return None

    return GoogleDriveClient(
        access_token=token.get("access_token"),
        token_refresh_callback=refresh_callback,
    )
