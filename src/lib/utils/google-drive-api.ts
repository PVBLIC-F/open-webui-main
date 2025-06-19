/**
 * Google Drive API Utilities
 * 
 * This module provides client-side utilities for interacting with the Google Drive API.
 * It handles folder browsing and OAuth URL generation for the cloud sync feature.
 * 
 * Key Features:
 * - Type-safe Google Drive item representation
 * - Efficient folder content listing with proper error handling
 * - OAuth URL generation with offline access for token refresh
 * - Consistent API response handling and error propagation
 * 
 * Usage:
 * Used by GoogleDrivePicker component to browse and select folders for synchronization.
 * Integrates with backend OAuth flow for secure API access.
 */

/**
 * Google Drive Item Interface
 * 
 * Represents a file or folder item from Google Drive API responses.
 * Includes essential metadata needed for folder picker functionality.
 */
export interface DriveItem {
  id: string;           // Unique Google Drive file/folder ID
  name: string;         // Display name of the item
  mimeType: string;     // MIME type (used to identify folders vs files)
  iconLink?: string;    // Optional icon URL from Google Drive
  parents?: string[];   // Parent folder IDs (for hierarchy navigation)
}

/**
 * List Google Drive Items
 * 
 * Fetches the contents of a specified Google Drive folder using the Drive API v3.
 * Filters out trashed items and returns only active files and folders.
 * 
 * @param token - OAuth access token for API authentication
 * @param folderId - Google Drive folder ID to list contents of (defaults to 'root')
 * @returns Promise resolving to array of DriveItem objects
 * @throws Error if API request fails or token is invalid
 */
export async function listDriveItems(token: string, folderId: string = 'root'): Promise<DriveItem[]> {
  // Build query to get non-trashed items in the specified folder
  const q = `'${folderId}' in parents and trashed = false`;
  const url = `https://www.googleapis.com/drive/v3/files?fields=files(id,name,mimeType,iconLink,parents)&q=${encodeURIComponent(q)}`;
  
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` }
  });
  
  if (!res.ok) throw new Error('Failed to list Google Drive items');
  
  const data = await res.json();
  return data.files;
}

/**
 * Generate Google OAuth URL
 * 
 * Creates a properly formatted Google OAuth 2.0 authorization URL with all
 * necessary parameters for cloud sync integration. Includes offline access
 * for refresh token capability.
 * 
 * OAuth Flow Configuration:
 * - Requests offline access for refresh tokens
 * - Forces consent prompt to ensure fresh permissions
 * - Includes granted scopes for incremental authorization
 * - Uses authorization code flow for secure token exchange
 * 
 * @param clientId - Google OAuth client ID from cloud console
 * @param redirectUri - Callback URL for OAuth flow completion
 * @param scope - Array of OAuth scopes to request (e.g., ['https://www.googleapis.com/auth/drive.readonly'])
 * @returns Complete OAuth authorization URL for user redirection
 */
export function getGoogleOAuthUrl(clientId: string, redirectUri: string, scope: string[]): string {
  const params = new URLSearchParams({
    client_id: clientId,
    redirect_uri: redirectUri,
    response_type: 'code',        // Authorization code flow
    access_type: 'offline',       // Request refresh token
    prompt: 'consent',            // Force consent screen for fresh permissions
    scope: scope.join(' '),       // Space-separated scope list
    include_granted_scopes: 'true', // Allow incremental authorization
  });
  
  return `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`;
} 