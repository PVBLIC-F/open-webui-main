<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { onMount, getContext, createEventDispatcher } from 'svelte';
	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	import Modal from '$lib/components/common/Modal.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import GoogleDrive from '$lib/components/icons/GoogleDrive.svelte';
	import { createFolderPicker } from '$lib/utils/google-drive-picker';
	import { connectDriveFolder } from '$lib/apis/knowledge';
	import { config } from '$lib/stores';

	export let show = false;
	export let knowledgeId: string;

	let loading = false;
	let selectedFolder: { id: string; name: string } | null = null;
	let syncInterval = 1;

	const handleSelectFolder = async () => {
		try {
			const folder = await createFolderPicker();
			if (folder) {
				selectedFolder = folder as { id: string; name: string };
			}
		} catch (error) {
			console.error('Folder picker error:', error);
			const errorMessage = error instanceof Error ? error.message : String(error);
			toast.error($i18n.t('Error accessing Google Drive: {{error}}', { error: errorMessage }));
		}
	};

	const handleConnect = async () => {
		if (!selectedFolder) {
			toast.error($i18n.t('Please select a folder first'));
			return;
		}

		loading = true;
		try {
			const result = await connectDriveFolder(localStorage.token, knowledgeId, {
				drive_folder_id: selectedFolder.id,
				drive_folder_name: selectedFolder.name,
				auto_sync_interval_hours: syncInterval
			});

			if (result) {
				toast.success($i18n.t('Google Drive folder connected successfully'));
				dispatch('connected', result);
				show = false;
				selectedFolder = null;
			}
		} catch (error) {
			console.error('Connect error:', error);
			const errorMessage = error instanceof Error ? error.message : String(error);
			toast.error($i18n.t('Failed to connect folder: {{error}}', { error: errorMessage }));
		} finally {
			loading = false;
		}
	};

	const handleClose = () => {
		show = false;
		selectedFolder = null;
	};
</script>

<Modal size="sm" bind:show>
	<div class="px-6 py-5">
		<div class="flex items-center gap-3 mb-4">
			<div class="p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
				<GoogleDrive className="w-6 h-6 text-blue-600 dark:text-blue-400" />
			</div>
			<div>
				<h3 class="text-lg font-semibold dark:text-white">
					{$i18n.t('Connect Google Drive')}
				</h3>
				<p class="text-sm text-gray-500 dark:text-gray-400">
					{$i18n.t('Sync files from a Drive folder to this Knowledge base')}
				</p>
			</div>
		</div>

		{#if !$config?.features?.enable_google_drive_integration}
			<div class="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg mb-4">
				<p class="text-sm text-yellow-800 dark:text-yellow-200">
					{$i18n.t('Google Drive integration is not enabled. Please contact your administrator.')}
				</p>
			</div>
		{:else}
			<div class="space-y-4">
				<!-- Folder Selection -->
				<div>
					<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
						{$i18n.t('Drive Folder')}
					</label>
					{#if selectedFolder}
						<div
							class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
						>
							<div class="flex items-center gap-2">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									class="w-5 h-5 text-yellow-500"
									fill="currentColor"
									viewBox="0 0 24 24"
								>
									<path
										d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"
									/>
								</svg>
								<span class="text-sm font-medium dark:text-white">{selectedFolder.name}</span>
							</div>
							<button
								type="button"
								class="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400"
								on:click={handleSelectFolder}
							>
								{$i18n.t('Change')}
							</button>
						</div>
					{:else}
						<button
							type="button"
							class="w-full p-4 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg hover:border-blue-500 dark:hover:border-blue-400 transition-colors"
							on:click={handleSelectFolder}
						>
							<div class="flex flex-col items-center gap-2 text-gray-500 dark:text-gray-400">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									class="w-8 h-8"
									fill="none"
									viewBox="0 0 24 24"
									stroke="currentColor"
								>
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										stroke-width="2"
										d="M12 4v16m8-8H4"
									/>
								</svg>
								<span class="text-sm font-medium">{$i18n.t('Select a folder from Google Drive')}</span>
							</div>
						</button>
					{/if}
				</div>

				<!-- Sync Interval -->
				<div>
					<label
						for="sync-interval"
						class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
					>
						{$i18n.t('Auto-sync Interval')}
					</label>
					<select
						id="sync-interval"
						bind:value={syncInterval}
						class="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
					>
						<option value={1}>{$i18n.t('Every hour')}</option>
						<option value={6}>{$i18n.t('Every 6 hours')}</option>
						<option value={12}>{$i18n.t('Every 12 hours')}</option>
						<option value={24}>{$i18n.t('Every day')}</option>
					</select>
					<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
						{$i18n.t('How often to check for new or updated files')}
					</p>
				</div>

				<!-- Info -->
				<div class="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
					<p class="text-xs text-blue-800 dark:text-blue-200">
						{$i18n.t(
							'Files will be automatically synced from this folder. Changes in Drive (new files, updates, deletions) will be reflected in this Knowledge base.'
						)}
					</p>
				</div>
			</div>
		{/if}

		<!-- Actions -->
		<div class="flex justify-end gap-3 mt-6">
			<button
				type="button"
				class="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
				on:click={handleClose}
			>
				{$i18n.t('Cancel')}
			</button>
			{#if $config?.features?.enable_google_drive_integration}
				<button
					type="button"
					class="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
					disabled={!selectedFolder || loading}
					on:click={handleConnect}
				>
					{#if loading}
						<Spinner className="w-4 h-4" />
					{/if}
					{$i18n.t('Connect')}
				</button>
			{/if}
		</div>
	</div>
</Modal>
