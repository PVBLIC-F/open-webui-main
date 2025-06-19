<script lang="ts">
	import { DropdownMenu } from 'bits-ui';
	import { flyAndScale } from '$lib/utils/transitions';
	import { getContext, createEventDispatcher } from 'svelte';
	const dispatch = createEventDispatcher();

	import Dropdown from '$lib/components/common/Dropdown.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import ArrowUpCircle from '$lib/components/icons/ArrowUpCircle.svelte';
	import BarsArrowUp from '$lib/components/icons/BarsArrowUp.svelte';
	import FolderOpen from '$lib/components/icons/FolderOpen.svelte';
	import ArrowPath from '$lib/components/icons/ArrowPath.svelte';
	import { getAuthTokenForKnowledgeBase } from '$lib/utils/google-drive-picker';
	import { toast } from 'svelte-sonner';
	import GoogleDrivePicker from '$lib/components/drive/GoogleDrivePicker.svelte';
	import Modal from '$lib/components/common/Modal.svelte';

	const i18n = getContext('i18n');

	export let onClose: Function = () => {};
	export let connectedFolderIds: string[] = [];

	let show = false;
	let showCustomFolderPicker = false;
	let driveToken = '';
	let currentTab = 'folders'; // Track current tab for dynamic title
	let selectionState: { hasSelection: boolean, selectionInfo: any } = { hasSelection: false, selectionInfo: null }; // Track selection state
	let pickerComponent: any;

	/**
	 * Handles Google Drive authentication and shows folder picker
	 * First checks for existing valid token, only prompts for auth if needed
	 */
	async function startGoogleDriveKBFlow() {
		try {
			// First, check if we already have a valid token
			console.log('🔍 Checking for existing Google Drive token...');
			const existingTokenResponse = await fetch('/api/oauth/google-drive/status', {
				credentials: 'include'
			});
			
			console.log('🔍 Token status response:', existingTokenResponse.status, existingTokenResponse.ok);
			
			if (existingTokenResponse.ok) {
				const tokenStatus = await existingTokenResponse.json();
				console.log('🔍 Token status:', tokenStatus);
				
				if (tokenStatus.has_token && tokenStatus.is_valid) {
					console.log('🔍 Using existing valid Google Drive token');
					// We have a valid token, get a fresh access token from the backend
					const refreshResponse = await fetch('/api/oauth/google-drive/refresh', {
						method: 'POST',
						credentials: 'include'
					});
					
					console.log('🔍 Refresh response:', refreshResponse.status, refreshResponse.ok);
					
					if (refreshResponse.ok) {
						const refreshData = await refreshResponse.json();
						console.log('🔍 Got fresh access token');
						driveToken = refreshData.access_token;
						showCustomFolderPicker = true;
						return;
					} else {
						const errorText = await refreshResponse.text();
						console.error('🔍 Refresh failed:', errorText);
					}
				} else {
					console.log('🔍 No valid token found in status:', tokenStatus);
				}
			} else {
				const errorText = await existingTokenResponse.text();
				console.error('🔍 Status check failed:', existingTokenResponse.status, errorText);
			}
			
			console.log('🔍 No valid token found, starting OAuth flow');
			// No valid token, start OAuth flow
			const token = await getAuthTokenForKnowledgeBase();
			
			if (token) {
				// Use the token with our custom folder picker
				driveToken = token;
				showCustomFolderPicker = true;
			} else {
				throw new Error('Failed to get access token from Google Drive');
			}
		} catch (error) {
			console.error('Error accessing Google Drive:', error);
			const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
			toast.error(`Error accessing Google Drive: ${errorMessage}`);
		}
	}

	function handleGoogleDriveSelection(event: { detail: any }) {
		const selection = event.detail;
		
		if (!selection || !driveToken) {
			return;
		}

		// Handle different selection types
		if (selection.type === 'multiple-folders') {
			// Multiple folders selected
			dispatch('google-drive-folders-selected', { 
				folders: selection.folders,
				access_token: driveToken 
			});
		} else if (selection.type === 'multiple-files') {
			// Multiple files selected
			dispatch('google-drive-files-selected', { 
				files: selection.files,
				access_token: driveToken 
			});
		} else {
			// Single item selected (backward compatibility)
			dispatch('google-drive-folder-selected', { 
				folder: selection,
				access_token: driveToken 
			});
		}
		
		showCustomFolderPicker = false;
	}

	function handleTabChange(event: { detail: { activeTab: string } }) {
		currentTab = event.detail.activeTab;
	}

	function handleSelectionState(event: { detail: { hasSelection: boolean, selectionInfo: any } }) {
		selectionState = event.detail;
	}

	function handlePickerCancel() {
		// Restore original selection state in the picker
		if (pickerComponent && pickerComponent.cancelSelection) {
			pickerComponent.cancelSelection();
		}
		showCustomFolderPicker = false;
	}
</script>

<Dropdown
	bind:show
	on:change={(e) => {
		if (e.detail === false) {
			onClose();
		}
	}}
	align="end"
>
	<Tooltip content={$i18n.t('Add Content')}>
		<button
			class=" p-1.5 rounded-xl hover:bg-gray-100 dark:bg-gray-850 dark:hover:bg-gray-800 transition font-medium text-sm flex items-center space-x-1"
			on:click={(e) => {
				e.stopPropagation();
				show = true;
			}}
		>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				viewBox="0 0 16 16"
				fill="currentColor"
				class="w-4 h-4"
			>
				<path
					d="M8.75 3.75a.75.75 0 0 0-1.5 0v3.5h-3.5a.75.75 0 0 0 0 1.5h3.5v3.5a.75.75 0 0 0 1.5 0v-3.5h3.5a.75.75 0 0 0 0-1.5h-3.5v-3.5Z"
				/>
			</svg>
		</button>
	</Tooltip>

	<div slot="content">
		<DropdownMenu.Content
			class="w-full max-w-44 rounded-xl p-1 z-50 bg-white dark:bg-gray-850 dark:text-white shadow-sm"
			sideOffset={4}
			side="bottom"
			align="end"
			transition={flyAndScale}
		>
			<DropdownMenu.Item
				class="flex  gap-2  items-center px-3 py-2 text-sm  cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded-md"
				on:click={() => {
					dispatch('upload', { type: 'files' });
				}}
			>
				<ArrowUpCircle strokeWidth="2" />
				<div class="flex items-center">{$i18n.t('Upload files')}</div>
			</DropdownMenu.Item>

			<DropdownMenu.Item
				class="flex  gap-2  items-center px-3 py-2 text-sm  cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded-md"
				on:click={() => {
					dispatch('upload', { type: 'directory' });
				}}
			>
				<FolderOpen strokeWidth="2" />
				<div class="flex items-center">{$i18n.t('Upload directory')}</div>
			</DropdownMenu.Item>

			<Tooltip
				content={$i18n.t(
					'This option will delete all existing files in the collection and replace them with newly uploaded files.'
				)}
				className="w-full"
			>
				<DropdownMenu.Item
					class="flex  gap-2  items-center px-3 py-2 text-sm  cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded-md"
					on:click={() => {
						dispatch('sync', { type: 'directory' });
					}}
				>
					<ArrowPath strokeWidth="2" />
					<div class="flex items-center">{$i18n.t('Sync directory')}</div>
				</DropdownMenu.Item>
			</Tooltip>

			<DropdownMenu.Item
				class="flex  gap-2  items-center px-3 py-2 text-sm  cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded-md"
				on:click={() => {
					dispatch('upload', { type: 'text' });
				}}
			>
				<BarsArrowUp strokeWidth="2" />
				<div class="flex items-center">{$i18n.t('Add text content')}</div>
			</DropdownMenu.Item>

			<DropdownMenu.Item
				class="flex  gap-2  items-center px-3 py-2 text-sm  cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded-md"
				on:click={startGoogleDriveKBFlow}
			>
				<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 87.3 78" class="w-5 h-5">
					<path d="m6.6 66.85 3.85 6.65c.8 1.4 1.95 2.5 3.3 3.3l13.75-23.8h-27.5c0 1.55.4 3.1 1.2 4.5z" fill="#0066da"/>
					<path d="m43.65 25-13.75-23.8c-1.35.8-2.5 1.9-3.3 3.3l-25.4 44a9.06 9.06 0 0 0 -1.2 4.5h27.5z" fill="#00ac47"/>
					<path d="m73.55 76.8c1.35-.8 2.5-1.9 3.3-3.3l1.6-2.75 7.65-13.25c.8-1.4 1.2-2.95 1.2-4.5h-27.502l5.852 11.5z" fill="#ea4335"/>
					<path d="m43.65 25 13.75-23.8c-1.35-.8-2.9-1.2-4.5-1.2h-18.5c-1.6 0-3.15.45-4.5 1.2z" fill="#00832d"/>
					<path d="m59.8 53h-32.3l-13.75 23.8c1.35.8 2.9 1.2 4.5 1.2h50.8c1.6 0 3.15-.45 4.5-1.2z" fill="#2684fc"/>
					<path d="m73.4 26.5-12.7-22c-.8-1.4-1.95-2.5-3.3-3.3l-13.75 23.8 16.15 28h27.45c0-1.55-.4-3.1-1.2-4.5z" fill="#ffba00"/>
				</svg>
				<div class="flex items-center">Google Drive</div>
			</DropdownMenu.Item>
		</DropdownMenu.Content>
	</div>
</Dropdown>

{#if showCustomFolderPicker}
<Modal size="md" bind:show={showCustomFolderPicker} on:close={handlePickerCancel}>
	<div>
		<div class="flex justify-between items-center dark:text-gray-100 px-5 pt-4 pb-1">
			<div class="text-lg font-medium font-primary">
				{currentTab === 'files' ? 'Select File(s)' : 'Select Folder(s)'}
			</div>
			
			<!-- Selection Controls in Header -->
			<div class="flex items-center space-x-3">
				{#if selectionState.selectionInfo}
					<div class="flex items-center space-x-2">
						<span class="text-xs text-blue-700 dark:text-blue-300 font-medium">
							{selectionState.selectionInfo.label}
						</span>
						{#if selectionState.hasSelection}
							<div class="flex space-x-1">
								<button
									on:click={selectionState.selectionInfo.clearFn}
									class="px-2 py-1 text-xs text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
								>
									Clear
								</button>
								<button
									on:click={selectionState.selectionInfo.confirmFn}
									class="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
								>
									{selectionState.selectionInfo.confirmLabel}
								</button>
							</div>
						{/if}
					</div>
				{/if}
				
				<button
					class="self-center"
					on:click={handlePickerCancel}
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 20 20"
						fill="currentColor"
						class="w-5 h-5"
					>
						<path
							d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z"
						/>
					</svg>
				</button>
			</div>
		</div>
		<div class="w-full px-5 pb-4 dark:text-white">
			<GoogleDrivePicker 
				bind:this={pickerComponent}
				token={driveToken || ''} 
				preSelectedFolderIds={connectedFolderIds}
				on:select={handleGoogleDriveSelection}
				on:tab-change={handleTabChange}
				on:selection-state={handleSelectionState}
			/>
		</div>
	</div>
</Modal>
{/if}
