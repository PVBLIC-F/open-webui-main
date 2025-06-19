<!--
FilesWithFolders Component

A smart file organization component that groups files by their Google Drive folder origin
while handling orphaned files gracefully. This component is key to the cloud sync user
experience, providing clear visual organization of synced vs manually uploaded content.

Key Features:
- Intelligent file grouping by Google Drive folder metadata
- Orphaned file handling for disconnected sync folders
- Connected folder filtering to show only active sync relationships
- Consistent file interaction patterns (click, delete) across all file types
- Visual distinction between synced folders and regular files

Cloud Sync Integration:
- Reads Google Drive metadata from file.meta.data to determine folder relationships
- Filters files based on connectedFolderIds to show only active sync connections
- Treats files from disconnected folders as regular files for cleanup clarity
- Provides seamless user experience for mixed content (synced + manual uploads)

Architecture:
- Reactive file grouping with automatic updates when files or connections change
- Event-driven communication for file interactions
- Reusable components (FolderCard, FileItem) for consistent UI patterns
- Separation of concerns between grouping logic and display components
-->

<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import FileItem from '$lib/components/common/FileItem.svelte';
	import FolderCard from './FolderCard.svelte';

	const dispatch = createEventDispatcher();

	// Component Props: Configure display behavior and data
	export let selectedFileId: string | null = null;
	export let files: any[] = [];
	export let small: boolean = false;
	export let connectedFolderIds: string[] = []; // Only show folders that are actually connected

	// Smart File Grouping: Organize files by Google Drive folders with orphan handling
	// Group files by Google Drive folders and regular files
	$: groupedFiles = files.reduce((acc, file) => {
		// Cloud Sync Integration: Check if file has Google Drive metadata AND the folder is actually connected
		if (file.meta?.data?.google_drive_folder_id && file.meta?.data?.google_drive_folder_name) {
			const folderId = file.meta.data.google_drive_folder_id;
			const folderName = file.meta.data.google_drive_folder_name;
			
			// Folder Connection Validation: Only group into folder if the folder is actually connected
			if (connectedFolderIds.includes(folderId)) {
				if (!acc.folders[folderId]) {
					acc.folders[folderId] = {
						id: folderId,
						name: folderName,
						files: []
					};
				}
				acc.folders[folderId].files.push(file);
			} else {
				// Orphaned File Handling: Treat orphaned Google Drive files as regular files
				// This provides clear visual feedback when folders are disconnected
				acc.regularFiles.push(file);
			}
		} else {
			// Regular File Processing: Handle manually uploaded files
			acc.regularFiles.push(file);
		}
		return acc;
	}, { folders: {}, regularFiles: [] });

	// Reactive Folder Array: Convert folder object to array for template iteration
	$: folderArray = Object.values(groupedFiles.folders) as { id: string; name: string; files: any[] }[];
</script>

<!-- Layout: Two-section display with folders first, then individual files -->
<div class="max-h-full flex flex-col w-full">
	<!-- Google Drive Folders Section: Display connected sync folders -->
	{#each folderArray as folder}
		<FolderCard {folder} />
	{/each}

	<!-- Regular Files Section: Individual files (manual uploads + orphaned sync files) -->
	{#each groupedFiles.regularFiles as file}
		<div class="mt-1 px-2">
			<FileItem
				className="w-full"
				colorClassName="{selectedFileId === file.id
					? ' bg-gray-50 dark:bg-gray-850'
					: 'bg-transparent'} hover:bg-gray-50 dark:hover:bg-gray-850 transition"
				{small}
				item={file}
				name={file?.name ?? file?.meta?.name}
				type="file"
				size={file?.size ?? file?.meta?.size ?? ''}
				loading={file.status === 'uploading'}
				dismissible
				on:click={() => {
					// Upload State Handling: Prevent interaction during upload
					if (file.status === 'uploading') {
						return;
					}

					dispatch('click', file.id);
				}}
				on:dismiss={() => {
					// Upload State Handling: Prevent deletion during upload
					if (file.status === 'uploading') {
						return;
					}

					dispatch('delete', file.id);
				}}
			/>
		</div>
	{/each}
</div> 