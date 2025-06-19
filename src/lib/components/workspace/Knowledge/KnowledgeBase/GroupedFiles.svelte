<!--
GroupedFiles Component

An advanced file display component that organizes files by Google Drive folders with
comprehensive interaction capabilities. This component serves as the main file browser
for knowledge bases with cloud sync integration, providing folder-level operations
and clear visual separation between synced and regular content.

Key Features:
- Hierarchical display with Google Drive folders prominently featured
- Folder-level operations (click to view, delete entire folders)
- Processing state indicators for ongoing sync operations
- File count badges for quick folder content overview
- Consistent file interaction patterns across all file types
- Visual separation between synced folders and regular files

Cloud Sync Integration:
- Displays Google Drive folders with official branding and icons
- Shows real-time processing states during sync operations
- Provides folder-level delete operations for bulk cleanup
- Handles mixed content scenarios (synced + manual uploads)
- External link indicators for Google Drive navigation

Architecture:
- Type-safe interfaces for better code maintainability
- Event-driven communication for parent component integration
- Helper functions for consistent file item properties
- Reactive processing state management
- Clean separation between folder and file handling logic

User Experience:
- Intuitive folder browsing with visual feedback
- Processing indicators to show sync progress
- Hover states and transitions for smooth interactions
- Consistent styling with application design system
-->

<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import FileItem from '$lib/components/common/FileItem.svelte';

	const dispatch = createEventDispatcher();

	// Type Definitions: Ensure type safety and code clarity
	interface GoogleDriveFolder {
		id: string;      // Google Drive folder ID
		name: string;    // Display name of the folder
		files: any[];    // Array of files in this folder
	}

	interface GroupedFiles {
		googleDriveFolders: Record<string, GoogleDriveFolder>;  // Folders organized by ID
		regularFiles: any[];                                    // Non-synced files
	}

	// Component Props: Configure display behavior and data
	export let selectedFileId: string | null = null;
	export let groupedFiles: GroupedFiles = { googleDriveFolders: {}, regularFiles: [] };
	export let small: boolean = false;
	export let processingFolders: Set<string> = new Set();  // Folders currently being processed

	// Event Handlers: Manage user interactions with files and folders
	function handleFileClick(file: any) {
		// Upload State Protection: Prevent interaction during upload
		if (file.status === 'uploading') return;
		dispatch('click', file.id);
	}

	function handleFileDelete(file: any) {
		// Upload State Protection: Prevent deletion during upload
		if (file.status === 'uploading') return;
		dispatch('delete', file.id);
	}

	function handleFolderClick(folder: GoogleDriveFolder) {
		// Folder Navigation: Allow parent to handle folder viewing
		dispatch('folder-click', folder);
	}

	function handleFolderDelete(folder: GoogleDriveFolder) {
		// Folder Deletion: Bulk delete operation for entire folders
		dispatch('folder-delete', folder);
	}

	// UI Helper: Generate consistent file item properties
	// File item properties helper
	function getFileItemProps(file: any) {
		const isSelected = selectedFileId === file.id;
		const isUploading = file.status === 'uploading';
		
		return {
			className: "w-full",
			colorClassName: `${isSelected ? ' bg-gray-50 dark:bg-gray-850' : 'bg-transparent'} hover:bg-gray-50 dark:hover:bg-gray-850 transition`,
			small,
			item: file,
			name: file?.name ?? file?.meta?.name,
			type: "file",
			size: file?.size ?? file?.meta?.size ?? '',
			loading: isUploading,
			dismissible: true
		};
	}
</script>

<!-- Layout: Two-tier display with folders first, then individual files -->
<div class="max-h-full flex flex-col w-full">
	<!-- Google Drive Folders Section: Prominent display of synced folders -->
	{#each Object.values(groupedFiles.googleDriveFolders) as folder}
		<div class="mt-2">
			<!-- Folder Button: Interactive folder representation with full feature set -->
			<button
				class="w-full px-2 py-1.5 text-left text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg transition-colors duration-200 flex items-center gap-2"
				on:click={() => handleFolderClick(folder)}
			>
				<!-- Google Drive Branding: Official Google Drive icon for recognition -->
				<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 87.3 78" class="w-4 h-4 flex-shrink-0">
					<path d="m6.6 66.85 3.85 6.65c.8 1.4 1.95 2.5 3.3 3.3l13.75-23.8h-27.5c0 1.55.4 3.1 1.2 4.5z" fill="#0066da"/>
					<path d="m43.65 25-13.75-23.8c-1.35.8-2.5 1.9-3.3 3.3l-25.4 44a9.06 9.06 0 0 0 -1.2 4.5h27.5z" fill="#00ac47"/>
					<path d="m73.55 76.8c1.35-.8 2.5-1.9 3.3-3.3l1.6-2.75 7.65-13.25c.8-1.4 1.2-2.95 1.2-4.5h-27.502l5.852 11.5z" fill="#ea4335"/>
					<path d="m43.65 25 13.75-23.8c-1.35-.8-2.9-1.2-4.5-1.2h-18.5c-1.6 0-3.15.45-4.5 1.2z" fill="#00832d"/>
					<path d="m59.8 53h-32.3l-13.75 23.8c1.35.8 2.9 1.2 4.5 1.2h50.8c1.6 0 3.15-.45 4.5-1.2z" fill="#2684fc"/>
					<path d="m73.4 26.5-12.7-22c-.8-1.4-1.95-2.5-3.3-3.3l-13.75 23.8 16.15 28h27.45c0-1.55-.4-3.1-1.2-4.5z" fill="#ffba00"/>
				</svg>
				<!-- Folder Name: Truncated display for long names -->
				<span class="flex-1 truncate">{folder.name}</span>
				
				<!-- Dynamic Folder Status: Processing indicator or action buttons -->
				{#if processingFolders.has(folder.id)}
					<!-- Processing State: Spinner for ongoing sync operations -->
					<svg class="animate-spin h-4 w-4 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
					</svg>
				{:else}
					<!-- Ready State: File count and action buttons -->
					<!-- File Count Badge: Shows number of files in folder -->
					<span class="text-xs text-gray-500">({folder.files.length})</span>
					<!-- Delete Action: Bulk folder deletion -->
					<button
						class="p-1 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
						on:click|stopPropagation={() => handleFolderDelete(folder)}
						title="Delete folder and all files"
					>
						<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
						</svg>
					</button>
					<!-- External Link Indicator: Shows this links to Google Drive -->
					<svg class="w-3 h-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
					</svg>
				{/if}
			</button>
		</div>
	{/each}

	<!-- Regular Files Section: Individual files not in Google Drive folders -->
	{#if groupedFiles.regularFiles.length > 0}
		<div class="mt-1">
			<!-- Section Separator: Only show when both folders and files exist -->
			{#if Object.keys(groupedFiles.googleDriveFolders).length > 0}
				<div class="px-2 py-0.5 text-xs font-medium text-gray-500 dark:text-gray-400 border-t border-gray-200 dark:border-gray-700 mt-2 pt-2">
					Other Files
				</div>
			{/if}
			
			<!-- File List: Individual file items with consistent interaction patterns -->
			{#each groupedFiles.regularFiles as file}
				<div class="mt-0.5 px-2">
					<FileItem
						{...getFileItemProps(file)}
						on:click={() => handleFileClick(file)}
						on:dismiss={() => handleFileDelete(file)}
					/>
				</div>
			{/each}
		</div>
	{/if}
</div> 