<!--
FolderCard Component

A visual representation of a Google Drive folder containing synced files in a knowledge base.
This component provides a clear folder-based organization view with direct links to the
source Google Drive folder for user convenience.

Key Features:
- Clean folder display with name, file count, and visual folder icon
- Direct link to open the folder in Google Drive (external navigation)
- Responsive design with hover states for better user interaction
- File count display with proper pluralization
- Consistent styling with the overall application theme

Cloud Sync Integration:
- Displays folders that contain files synced from Google Drive
- Provides quick access to source folder for verification or manual management
- Shows real-time file count based on currently synced files
- Integrates seamlessly with FilesWithFolders parent component

User Experience:
- Hover effects for interactive feedback
- External link icon to indicate Google Drive navigation
- Truncated folder names to handle long folder names gracefully
- Clear visual hierarchy with folder icon and metadata
-->

<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Link from '$lib/components/icons/Link.svelte';
	import FolderOpen from '$lib/components/icons/FolderOpen.svelte';

	const dispatch = createEventDispatcher();

	// Folder Data: Information about the Google Drive folder and its synced files
	export let folder: {
		id: string;        // Google Drive folder ID for direct linking
		name: string;      // Display name of the folder
		files: any[];      // Array of files currently synced from this folder
	};
</script>

<!-- Folder Card Layout: Compact folder representation with external link -->
<div class="mt-1 px-2">
	<div class="w-full bg-transparent hover:bg-gray-50 dark:hover:bg-gray-850 transition">
		<div class="flex items-center justify-between">
			<!-- Folder Information: Icon, name, and file count -->
			<div class="flex items-center space-x-2 flex-1">
				<!-- Folder Icon: Visual indicator for folder type -->
				<FolderOpen className="w-4 h-4 text-gray-900 dark:text-gray-100" />
				<div class="flex-1">
					<!-- Folder Name: Truncated for long names -->
					<div class="font-medium text-sm text-gray-900 dark:text-gray-100 truncate">
						{folder.name}
					</div>
					<!-- File Count: Shows number of synced files with proper pluralization -->
					<div class="text-xs text-gray-500 dark:text-gray-400">
						{folder.files.length} file{folder.files.length === 1 ? '' : 's'}
					</div>
				</div>
			</div>
			
			<!-- Actions: External link to Google Drive folder -->
			<div class="flex items-center space-x-2">
				<!-- Google Drive Link: Direct navigation to source folder -->
				<a
					href="https://drive.google.com/drive/folders/{folder.id}"
					target="_blank"
					rel="noopener noreferrer"
					class="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
					title="Open in Google Drive"
				>
					<Link className="w-3.5 h-3.5 text-gray-600 dark:text-gray-300" />
				</a>
			</div>
		</div>
	</div>
</div> 