/**
 * Google Drive Picker API Integration
 * 
 * This module provides comprehensive Google Drive integration for both chat file picking
 * and knowledge base folder synchronization. It handles OAuth flows, API loading, and
 * file operations with proper authentication and scope management.
 * 
 * Key Features:
 * - Dual OAuth flow support (implicit for chat, authorization code for KB sync)
 * - Dynamic API loading with proper error handling
 * - Scope-based authentication for different use cases
 * - File picker integration with download capabilities
 * - Token management and refresh for background operations
 * 
 * Architecture:
 * - Lazy loading of Google APIs to reduce initial bundle size
 * - Separation of concerns between chat and knowledge base workflows
 * - Centralized credential management from backend configuration
 * - Comprehensive error handling and logging for debugging
 * 
 * OAuth Flows:
 * - Chat: Implicit flow for temporary file access (no refresh tokens)
 * - Knowledge Base: Authorization code flow for background sync (includes refresh tokens)
 * 
 * Usage:
 * - Chat: createPicker() for one-time file selection and download
 * - Knowledge Base: getAuthTokenForKnowledgeBase() for folder sync setup
 */

import { fetchGoogleDriveConfig } from './config';

// TypeScript Global Declarations: Define Google APIs for type safety
declare global {
	interface Window {
		gapi: any;      // Google API loader
		google: any;    // Google Identity Services and Picker
	}
}

// Configuration State: Dynamically loaded from backend
let API_KEY = '';
let CLIENT_ID = '';

// OAuth Scopes: Different permissions for different use cases
// Chat scope for temporary file access (implicit flow)
const SCOPE = [
	'https://www.googleapis.com/auth/drive.readonly',  // Read access to files
	'https://www.googleapis.com/auth/drive.file'       // Access to app-created files
];

// KB scope includes metadata for background sync capabilities (authorization code flow)
const KB_SCOPE = [
	...SCOPE,
	'https://www.googleapis.com/auth/drive.metadata.readonly'  // Metadata access for sync
];

// Initialization State: Track API loading and authentication status
let pickerApiLoaded = false;
let authApiLoaded = false;
let oauthToken: string | null = null;
let initialized = false;

/**
 * Credential Management: Fetch Google Drive API credentials from backend
 * 
 * Retrieves client ID and API key from the backend configuration service.
 * This ensures credentials are centrally managed and not hardcoded in frontend.
 * 
 * @throws Error if credentials are not configured in backend
 */
export async function getCredentials(): Promise<void> {
	const { clientId, apiKey } = await fetchGoogleDriveConfig();
	CLIENT_ID = clientId;
	API_KEY = apiKey;
}

/**
 * Google Drive Picker API Loader
 * 
 * Dynamically loads the Google Drive Picker API with proper error handling.
 * Uses lazy loading to reduce initial bundle size and only load when needed.
 * 
 * Loading Strategy:
 * - Checks if API is already loaded to avoid duplicate requests
 * - Creates script element for dynamic loading
 * - Handles both load success and error scenarios
 * 
 * @returns Promise<boolean> Resolves to true when API is successfully loaded
 */
export const loadGoogleDriveApi = (): Promise<boolean> => {
	return new Promise((resolve, reject) => {
		if (typeof window.gapi === 'undefined') {
			// Dynamic Script Loading: Load Google API script
			const script = document.createElement('script');
			script.src = 'https://apis.google.com/js/api.js';
			script.onload = () => {
				// Load Picker Module: Initialize picker functionality
				window.gapi.load('picker', () => {
					pickerApiLoaded = true;
					resolve(true);
				});
			};
			script.onerror = reject;
			document.body.appendChild(script);
		} else {
			// API Already Loaded: Just initialize picker module
			window.gapi.load('picker', () => {
				pickerApiLoaded = true;
				resolve(true);
			});
		}
	});
};

/**
 * Google Auth API Loader
 * 
 * Dynamically loads the Google Identity Services API for OAuth operations.
 * This API handles both implicit and authorization code flows.
 * 
 * @returns Promise<boolean> Resolves to true when API is successfully loaded
 */
export const loadGoogleAuthApi = (): Promise<boolean> => {
	return new Promise((resolve, reject) => {
		if (typeof window.google === 'undefined') {
			// Dynamic Script Loading: Load Google Identity Services
			const script = document.createElement('script');
			script.src = 'https://accounts.google.com/gsi/client';
			script.onload = () => {
				authApiLoaded = true;
				resolve(true);
			};
			script.onerror = reject;
			document.body.appendChild(script);
		} else {
			// API Already Loaded: Mark as ready
			authApiLoaded = true;
			resolve(true);
		}
	});
};

/**
 * Chat OAuth Token: Get temporary access token for file picking
 * 
 * Uses implicit OAuth flow to get a temporary access token for chat file operations.
 * This flow is simpler but doesn't provide refresh tokens, making it suitable
 * for one-time file access scenarios.
 * 
 * Flow Characteristics:
 * - Implicit flow (no refresh tokens)
 * - Temporary access (token expires)
 * - Suitable for immediate file operations
 * 
 * @returns Promise<string> The OAuth access token for immediate use
 */
export const getAuthToken = async (): Promise<string> => {
	// Credential Initialization: Ensure we have API credentials
	if (!CLIENT_ID || !API_KEY) {
		await getCredentials();
	}
	
	// API Loading: Ensure auth API is available
	if (!authApiLoaded) {
		await loadGoogleAuthApi();
	}

	// Token Management: Get or reuse existing token
	if (!oauthToken) {
		const token = await new Promise<string>((resolve, reject) => {
			// Implicit Flow: Request immediate access token
			const tokenClient = window.google.accounts.oauth2.initTokenClient({
				client_id: CLIENT_ID,
				scope: SCOPE.join(' '),
				callback: (response: any) => {
					if (response.access_token) {
						oauthToken = response.access_token;
						resolve(response.access_token);
					} else {
						reject(new Error('Failed to get access token'));
					}
				},
				error_callback: (error: any) => {
					reject(new Error(error.message || 'OAuth error occurred'));
				}
			});
			tokenClient.requestAccessToken();
		});
		return token;
	}
	return oauthToken;
};

/**
 * Knowledge Base OAuth Token: Get access token with refresh capability
 * 
 * Uses authorization code flow to obtain tokens with refresh capability for
 * background synchronization operations. This flow provides both access and
 * refresh tokens, enabling long-term background sync without user interaction.
 * 
 * Flow Characteristics:
 * - Authorization code flow (includes refresh tokens)
 * - Long-term access (refresh tokens for background operations)
 * - Suitable for automated sync operations
 * - Enhanced scope for metadata access
 * 
 * Backend Integration:
 * - Exchanges authorization code for tokens via backend API
 * - Backend stores refresh tokens securely for background sync
 * - Provides access token for immediate frontend operations
 * 
 * @returns Promise<string> The OAuth access token for knowledge base operations
 */
export const getAuthTokenForKnowledgeBase = async (): Promise<string> => {
	console.log('🔍 DEBUG: getAuthTokenForKnowledgeBase started');
	
	// Credential Initialization: Ensure we have API credentials
	if (!CLIENT_ID || !API_KEY) {
		console.log('🔍 DEBUG: Getting credentials...');
		await getCredentials();
		console.log('🔍 DEBUG: Credentials obtained:', { CLIENT_ID: CLIENT_ID?.substring(0, 20) + '...', API_KEY: API_KEY?.substring(0, 20) + '...' });
	}
	
	// API Loading: Ensure auth API is available
	if (!authApiLoaded) {
		console.log('🔍 DEBUG: Loading Google Auth API...');
		await loadGoogleAuthApi();
		console.log('🔍 DEBUG: Auth API loaded');
	}

	console.log('🔍 DEBUG: Creating code client with scope:', KB_SCOPE);
	
	// Authorization Code Flow: Request code for token exchange
	const token = await new Promise<string>((resolve, reject) => {
		const codeClient = window.google.accounts.oauth2.initCodeClient({
			client_id: CLIENT_ID,
			scope: KB_SCOPE.join(' '),  // Enhanced scope for metadata access
			ux_mode: 'popup',           // Popup for better UX
			callback: async (response: any) => {
				console.log('🔍 DEBUG: OAuth callback received:', { 
					hasCode: !!response.code, 
					codeLength: response.code?.length,
					responseKeys: Object.keys(response)
				});
				
				if (response.code) {
					try {
						console.log('🔍 DEBUG: Exchanging code for tokens...');
						// Backend Token Exchange: Convert code to tokens with refresh capability
						const tokenResponse = await exchangeCodeForTokens(response.code);
						console.log('🔍 DEBUG: Token exchange successful, access token length:', tokenResponse.access_token?.length);
						resolve(tokenResponse.access_token);
					} catch (error) {
						console.error('🔍 DEBUG: Token exchange failed:', error);
						reject(error);
					}
				} else {
					console.error('🔍 DEBUG: No authorization code in response');
					reject(new Error('Failed to get authorization code'));
				}
			},
			error_callback: (error: any) => {
				console.error('🔍 DEBUG: OAuth error callback:', error);
				reject(new Error(error.message || 'OAuth error occurred'));
			}
		});
		
		console.log('🔍 DEBUG: Requesting authorization code...');
		codeClient.requestCode();
	});
	
	console.log('🔍 DEBUG: getAuthTokenForKnowledgeBase completed');
	return token;
};

/**
 * Backend Token Exchange: Convert authorization code to access/refresh tokens
 * 
 * Exchanges the authorization code received from Google OAuth for actual tokens.
 * This operation is performed by the backend to securely handle refresh tokens
 * and maintain long-term sync capabilities.
 * 
 * Security Considerations:
 * - Authorization code is single-use and short-lived
 * - Backend securely stores refresh tokens
 * - Access tokens are returned for immediate frontend use
 * - All communication uses HTTPS and includes credentials
 * 
 * @param code Authorization code from Google OAuth flow
 * @returns Promise with access token for immediate use
 */
async function exchangeCodeForTokens(code: string): Promise<{access_token: string}> {
	console.log('🔍 DEBUG: exchangeCodeForTokens called with code length:', code?.length);
	
	// Backend API Call: Exchange code for tokens
	const response = await fetch('/api/oauth/google-drive/exchange', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		credentials: 'include',  // Include authentication cookies
		body: JSON.stringify({ code })
	});
	
	console.log('🔍 DEBUG: Backend response status:', response.status);
	
	// Error Handling: Provide detailed error information
	if (!response.ok) {
		const errorText = await response.text();
		console.error('🔍 DEBUG: Backend error response:', errorText);
		throw new Error(`Failed to exchange code: ${response.status} - ${errorText}`);
	}
	
	const result = await response.json();
	console.log('🔍 DEBUG: Backend success response keys:', Object.keys(result));
	
	return result;
}

/**
 * API Initialization: Prepare Google Drive APIs for use
 * 
 * Ensures all required APIs and credentials are loaded before attempting
 * to use Google Drive functionality. This prevents race conditions and
 * provides better error handling.
 * 
 * Initialization Steps:
 * 1. Load credentials from backend configuration
 * 2. Load Google Drive Picker API
 * 3. Load Google Auth API
 * 4. Mark system as initialized
 * 
 * @returns Promise that resolves when initialization is complete
 */
const initialize = async (): Promise<void> => {
	if (!initialized) {
		await getCredentials();
		await Promise.all([loadGoogleDriveApi(), loadGoogleAuthApi()]);
		initialized = true;
	}
};

/**
 * Google Drive File Picker: Create interactive file selection interface
 * 
 * Creates and displays a Google Drive file picker for chat file operations.
 * Handles the complete workflow from authentication to file download,
 * providing a seamless user experience for accessing Google Drive files.
 * 
 * Picker Configuration:
 * - Multi-select enabled for batch operations
 * - Filtered MIME types for supported file formats
 * - Hidden navigation for cleaner interface
 * - Automatic file download after selection
 * 
 * File Processing:
 * - Handles both regular files and Google Workspace documents
 * - Automatic format conversion for Google Docs/Sheets/Slides
 * - Direct download with proper authentication headers
 * - Blob creation for frontend file handling
 * 
 * Supported Formats:
 * - PDF documents
 * - Plain text files
 * - Microsoft Word documents
 * - Google Docs, Sheets, and Presentations
 * 
 * @returns Promise that resolves with selected file data or null if cancelled
 */
export const createPicker = (): Promise<any> => {
	return new Promise(async (resolve, reject) => {
		try {
			console.log('Initializing Google Drive Picker...');
			await initialize();
			
			console.log('Getting auth token...');
			// Authentication: Get token for file access
			const token = await getAuthToken();
			if (!token) {
				console.error('Failed to get OAuth token');
				throw new Error('Unable to get OAuth token');
			}
			console.log('Auth token obtained successfully');

			// Picker Configuration: Set up file picker with proper settings
			const picker = new window.google.picker.PickerBuilder()
				.enableFeature(window.google.picker.Feature.NAV_HIDDEN)        // Clean interface
				.enableFeature(window.google.picker.Feature.MULTISELECT_ENABLED) // Batch selection
				.addView(
					new window.google.picker.DocsView()
						.setIncludeFolders(false)      // Files only, no folders
						.setSelectFolderEnabled(false) // Disable folder selection
						.setMimeTypes(                 // Supported file formats
							'application/pdf,text/plain,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.google-apps.document,application/vnd.google-apps.spreadsheet,application/vnd.google-apps.presentation'
						)
				)
				.setOAuthToken(token)      // Authentication token
				.setDeveloperKey(API_KEY)  // API key for requests
				.setCallback(async (data: any) => {
					// File Selection Handler: Process selected files
					if (data[window.google.picker.Response.ACTION] === window.google.picker.Action.PICKED) {
						try {
							// File Metadata Extraction
							const doc = data[window.google.picker.Response.DOCUMENTS][0];
							const fileId = doc[window.google.picker.Document.ID];
							const fileName = doc[window.google.picker.Document.NAME];
							const fileUrl = doc[window.google.picker.Document.URL];

							if (!fileId || !fileName) {
								throw new Error('Required file details missing');
							}

							// Download URL Construction: Handle different file types
							const mimeType = doc[window.google.picker.Document.MIME_TYPE];

							let downloadUrl: string;
							let exportFormat: string;

							// Google Workspace Files: Require format conversion
							if (mimeType.includes('google-apps')) {
								// Format Selection: Choose appropriate export format
								if (mimeType.includes('document')) {
									exportFormat = 'text/plain';
								} else if (mimeType.includes('spreadsheet')) {
									exportFormat = 'text/csv';
								} else if (mimeType.includes('presentation')) {
									exportFormat = 'text/plain';
								} else {
									exportFormat = 'application/pdf';
								}
								downloadUrl = `https://www.googleapis.com/drive/v3/files/${fileId}/export?mimeType=${encodeURIComponent(exportFormat)}`;
							} else {
								// Regular Files: Direct download
								downloadUrl = `https://www.googleapis.com/drive/v3/files/${fileId}?alt=media`;
							}

							// File Download: Fetch file content with authentication
							const response = await fetch(downloadUrl, {
								headers: {
									Authorization: `Bearer ${token}`,
									Accept: '*/*'
								}
							});

							// Download Error Handling
							if (!response.ok) {
								const errorText = await response.text();
								console.error('Download failed:', {
									status: response.status,
									statusText: response.statusText,
									error: errorText
								});
								throw new Error(`Failed to download file (${response.status}): ${errorText}`);
							}

							// Blob Creation: Convert response to usable file object
							const blob = await response.blob();
							const result = {
								id: fileId,
								name: fileName,
								url: downloadUrl,
								blob: blob,
								headers: {
									Authorization: `Bearer ${token}`,
									Accept: '*/*'
								}
							};
							resolve(result);
						} catch (error) {
							reject(error);
						}
					} else if (data[window.google.picker.Response.ACTION] === window.google.picker.Action.CANCEL) {
						// User Cancellation: Handle graceful cancellation
						resolve(null);
					}
				})
				.build();
			
			// Display Picker: Show the file selection interface
			picker.setVisible(true);
		} catch (error) {
			console.error('Google Drive Picker error:', error);
			reject(error);
		}
	});
};
