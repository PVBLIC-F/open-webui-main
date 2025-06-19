/**
 * Application Configuration Utilities
 * 
 * This module provides centralized configuration management for cloud sync integrations.
 * It handles fetching and validating configuration from the backend API, ensuring
 * consistent access to cloud provider credentials across the application.
 * 
 * Key Features:
 * - Centralized configuration fetching to prevent duplication
 * - Type-safe configuration interfaces with validation
 * - Provider-specific configuration helpers (Google Drive, OneDrive)
 * - Error handling for missing or invalid configurations
 * - Credential validation before component initialization
 * 
 * Architecture:
 * - Single source of truth for cloud provider configurations
 * - Lazy loading of configuration data when needed
 * - Consistent error handling across all config operations
 * - Extensible design for adding new cloud providers
 * 
 * Usage:
 * Used by cloud sync components to obtain API credentials and configuration
 * before initializing OAuth flows or making API calls.
 */

/**
 * Application Configuration Interface
 * 
 * Defines the structure of the application configuration object returned
 * by the backend API. Supports multiple cloud providers with optional
 * configurations for flexible deployment scenarios.
 */
export interface AppConfig {
	// Google Drive Integration Configuration
	google_drive?: {
		client_id: string;    // OAuth client ID from Google Cloud Console
		api_key: string;      // Google Drive API key for public access
	};
	
	// OneDrive/SharePoint Integration Configuration
	onedrive?: {
		client_id: string;              // Azure app registration client ID
		sharepoint_url?: string;        // Optional SharePoint site URL
		sharepoint_tenant_id?: string;  // Optional tenant ID for SharePoint
	};
	
	// Extensible design for future cloud providers
	// Add other config types as needed
}

/**
 * Fetch Application Configuration
 * 
 * Retrieves the complete application configuration from the backend API.
 * This is the primary method for accessing cloud provider configurations
 * and should be used by provider-specific helper functions.
 * 
 * API Integration:
 * - Uses credentials: 'include' for authenticated requests
 * - Handles HTTP errors with descriptive error messages
 * - Returns type-safe configuration object
 * 
 * @returns Promise<AppConfig> The complete application configuration
 * @throws Error if the config cannot be fetched or parsed
 */
export async function fetchAppConfig(): Promise<AppConfig> {
	const response = await fetch('/api/config', {
		credentials: 'include'  // Include authentication cookies
	});
	
	if (!response.ok) {
		throw new Error(`Failed to fetch app configuration: ${response.status} ${response.statusText}`);
	}
	
	return response.json();
}

/**
 * Fetch Google Drive Configuration
 * 
 * Retrieves and validates Google Drive-specific configuration from the backend.
 * Ensures both client ID and API key are present before allowing Google Drive
 * integration to proceed.
 * 
 * Validation:
 * - Checks for presence of both client_id and api_key
 * - Throws descriptive error if configuration is incomplete
 * - Returns validated credentials ready for OAuth flow
 * 
 * @returns Promise with validated Google Drive client ID and API key
 * @throws Error if Google Drive is not configured or incomplete
 */
export async function fetchGoogleDriveConfig(): Promise<{ clientId: string; apiKey: string }> {
	const config = await fetchAppConfig();
	
	const clientId = config.google_drive?.client_id;
	const apiKey = config.google_drive?.api_key;
	
	// Validate required Google Drive credentials
	if (!clientId || !apiKey) {
		throw new Error('Google Drive API credentials not configured');
	}
	
	return { clientId, apiKey };
}

/**
 * Fetch OneDrive Configuration
 * 
 * Retrieves and validates OneDrive/SharePoint configuration from the backend.
 * Supports both personal OneDrive and SharePoint integrations with optional
 * SharePoint-specific configuration.
 * 
 * Configuration Support:
 * - Required: client_id for basic OneDrive access
 * - Optional: sharepoint_url for SharePoint site access
 * - Optional: sharepoint_tenant_id for multi-tenant scenarios
 * 
 * @returns Promise with validated OneDrive configuration object
 * @throws Error if OneDrive configuration is incomplete
 */
export async function fetchOneDriveConfig(): Promise<{
	clientId: string;
	sharepointUrl?: string;
	sharepointTenantId?: string;
}> {
	const config = await fetchAppConfig();
	
	const clientId = config.onedrive?.client_id;
	
	// Validate minimum required OneDrive configuration
	if (!clientId) {
		throw new Error('OneDrive configuration is incomplete');
	}
	
	return {
		clientId,
		sharepointUrl: config.onedrive?.sharepoint_url,
		sharepointTenantId: config.onedrive?.sharepoint_tenant_id
	};
} 