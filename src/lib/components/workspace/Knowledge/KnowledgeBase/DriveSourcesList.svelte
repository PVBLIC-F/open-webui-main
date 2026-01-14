<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { onMount, getContext, createEventDispatcher } from 'svelte';
	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import GoogleDrive from '$lib/components/icons/GoogleDrive.svelte';
	import { config } from '$lib/stores';
	import {
		getDriveSources,
		disconnectDriveFolder,
		syncDriveSource,
		updateDriveSource,
		type DriveSource
	} from '$lib/apis/knowledge';

	export let knowledgeId: string;
	export let writeAccess: boolean = false;

	let sources: DriveSource[] = [];
	let loading = true;
	let syncingSourceId: string | null = null;
	let showDisconnectConfirm = false;
	let disconnectTarget: { id: string; name: string } | null = null;

	const loadSources = async () => {
		loading = true;
		try {
			const result = await getDriveSources(localStorage.token, knowledgeId);
			sources = result?.sources || [];
		} catch (error) {
			console.error('Error loading Drive sources:', error);
			toast.error($i18n.t('Failed to load Drive sources'));
		} finally {
			loading = false;
		}
	};

	const handleSync = async (sourceId: string, forceFull: boolean = false) => {
		syncingSourceId = sourceId;
		try {
			const result = await syncDriveSource(localStorage.token, knowledgeId, sourceId, forceFull);
			if (result) {
				toast.success(forceFull ? $i18n.t('Full resync started') : $i18n.t('Sync started'));
				// Reload after a delay to show updated status
				setTimeout(loadSources, 2000);
			}
		} catch (error) {
			console.error('Sync error:', error);
			const errorMessage = error instanceof Error ? error.message : String(error);
			toast.error($i18n.t('Failed to start sync: {{error}}', { error: errorMessage }));
		} finally {
			syncingSourceId = null;
		}
	};

	const confirmDisconnect = (sourceId: string, folderName: string) => {
		disconnectTarget = { id: sourceId, name: folderName };
		showDisconnectConfirm = true;
	};

	const handleDisconnect = async () => {
		if (!disconnectTarget) return;

		const { id: sourceId } = disconnectTarget;
		try {
			const result = await disconnectDriveFolder(localStorage.token, knowledgeId, sourceId);
			if (result) {
				toast.success($i18n.t('Folder disconnected'));
				sources = sources.filter((s) => s.id !== sourceId);
				dispatch('disconnected', sourceId);
			}
		} catch (error) {
			console.error('Disconnect error:', error);
			const errorMessage = error instanceof Error ? error.message : String(error);
			toast.error($i18n.t('Failed to disconnect: {{error}}', { error: errorMessage }));
		} finally {
			showDisconnectConfirm = false;
			disconnectTarget = null;
		}
	};

	const handleToggleSync = async (source: DriveSource) => {
		try {
			const result = await updateDriveSource(localStorage.token, knowledgeId, source.id, {
				sync_enabled: !source.sync_enabled
			});
			if (result) {
				sources = sources.map((s) =>
					s.id === source.id ? { ...s, sync_enabled: !s.sync_enabled } : s
				);
				toast.success(
					source.sync_enabled
						? $i18n.t('Auto-sync disabled')
						: $i18n.t('Auto-sync enabled')
				);
			}
		} catch (error) {
			console.error('Toggle sync error:', error);
			const errorMessage = error instanceof Error ? error.message : String(error);
			toast.error($i18n.t('Failed to update settings: {{error}}', { error: errorMessage }));
		}
	};

	const handleToggleRecursive = async (source: DriveSource) => {
		try {
			const result = await updateDriveSource(localStorage.token, knowledgeId, source.id, {
				recursive: !source.recursive
			});
			if (result) {
				sources = sources.map((s) =>
					s.id === source.id ? { ...s, recursive: !s.recursive } : s
				);
				toast.success(
					source.recursive
						? $i18n.t('Recursive sync disabled')
						: $i18n.t('Recursive sync enabled - will include subfolders on next sync')
				);
			}
		} catch (error) {
			console.error('Toggle recursive error:', error);
			const errorMessage = error instanceof Error ? error.message : String(error);
			toast.error($i18n.t('Failed to update settings: {{error}}', { error: errorMessage }));
		}
	};

	const formatTimestamp = (timestamp: number | null) => {
		if (!timestamp) return $i18n.t('Never');
		const date = new Date(timestamp * 1000);
		return date.toLocaleString();
	};

	const getStatusColor = (status: string) => {
		switch (status) {
			case 'active':
				return 'text-blue-600 bg-blue-100 dark:text-blue-400 dark:bg-blue-900/30';
			case 'completed':
				return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30';
			case 'error':
				return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30';
			default:
				return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-800';
		}
	};

	const getStatusLabel = (status: string) => {
		switch (status) {
			case 'active':
				return $i18n.t('Syncing');
			case 'completed':
				return $i18n.t('Synced');
			case 'error':
				return $i18n.t('Error');
			case 'never':
				return $i18n.t('Not synced');
			default:
				return status;
		}
	};

	onMount(loadSources);

	// Expose reload method for parent components
	export const reload = loadSources;
</script>

<ConfirmDialog
	bind:show={showDisconnectConfirm}
	message={$i18n.t('Disconnect "{{name}}"? Synced files will remain in the Knowledge base.', { name: disconnectTarget?.name || 'folder' })}
	on:confirm={handleDisconnect}
/>

{#if !$config?.features?.enable_google_drive_integration}
	<!-- Drive integration disabled - show nothing -->
{:else if loading}
	<div class="flex justify-center py-4">
		<Spinner className="w-5 h-5" />
	</div>
{:else if sources.length === 0}
	<!-- No sources connected - show nothing to keep UI clean -->
{:else}
	<div class="space-y-2">
		{#each sources as source (source.id)}
			<div
				class="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700"
			>
				<div class="flex items-start justify-between gap-3">
					<div class="flex items-start gap-3 flex-1 min-w-0">
						<div class="p-1.5 bg-white dark:bg-gray-700 rounded-lg shrink-0">
							<GoogleDrive className="w-4 h-4 text-blue-600 dark:text-blue-400" />
						</div>
						<div class="min-w-0 flex-1">
							<div class="flex items-center gap-2 flex-wrap">
								<span class="font-medium text-sm dark:text-white truncate">
									{source.drive_folder_name || source.drive_folder_id}
								</span>
								{#if source.last_sync_file_count > 0}
									<span class="px-1.5 py-0.5 rounded text-xs font-medium text-gray-600 bg-gray-200 dark:text-gray-300 dark:bg-gray-700">
										{source.last_sync_file_count} {source.last_sync_file_count === 1 ? 'file' : 'files'}
									</span>
								{/if}
								{#if source.recursive}
									<Tooltip content={$i18n.t('Syncs all subfolders')}>
										<span class="px-1.5 py-0.5 rounded text-xs font-medium text-purple-600 bg-purple-100 dark:text-purple-400 dark:bg-purple-900/30">
											recursive
										</span>
									</Tooltip>
								{/if}
								<span
									class="px-2 py-0.5 rounded-full text-xs font-medium {getStatusColor(
										source.sync_status
									)}"
								>
									{getStatusLabel(source.sync_status)}
								</span>
							</div>
							<div class="flex items-center gap-3 mt-1 text-xs text-gray-500 dark:text-gray-400">
								<span>
									{$i18n.t('Last sync')}: {formatTimestamp(source.last_sync_timestamp)}
								</span>
							</div>
							{#if source.last_error}
								<div class="mt-1 text-xs text-red-600 dark:text-red-400 truncate">
									{source.last_error}
								</div>
							{/if}
						</div>
					</div>

					{#if writeAccess}
						<div class="flex items-center gap-1 shrink-0">
							<!-- Sync Button (incremental) -->
							<Tooltip content={$i18n.t('Sync changes')}>
								<button
									class="p-1.5 text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
									disabled={syncingSourceId === source.id || source.sync_status === 'active'}
									on:click={() => handleSync(source.id, false)}
								>
									{#if syncingSourceId === source.id || source.sync_status === 'active'}
										<Spinner className="w-4 h-4" />
									{:else}
										<svg
											xmlns="http://www.w3.org/2000/svg"
											class="w-4 h-4"
											fill="none"
											viewBox="0 0 24 24"
											stroke="currentColor"
										>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
											/>
										</svg>
									{/if}
								</button>
							</Tooltip>

							<!-- Full Resync Button (reprocess all files) -->
							<Tooltip content={$i18n.t('Full resync (reprocess all files)')}>
								<button
									class="p-1.5 text-gray-400 hover:text-orange-600 dark:text-gray-500 dark:hover:text-orange-400 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
									disabled={syncingSourceId === source.id || source.sync_status === 'active'}
									on:click={() => handleSync(source.id, true)}
								>
									<!-- Database refresh icon -->
									<svg
										xmlns="http://www.w3.org/2000/svg"
										class="w-4 h-4"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											stroke-width="2"
											d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"
										/>
									</svg>
								</button>
							</Tooltip>

							<!-- Toggle Recursive -->
							<Tooltip
								content={source.recursive
									? $i18n.t('Disable recursive sync (subfolders)')
									: $i18n.t('Enable recursive sync (include subfolders)')}
							>
								<button
									class="p-1.5 rounded-lg transition-colors {source.recursive
										? 'text-purple-600 hover:text-purple-700 dark:text-purple-400 hover:bg-gray-200 dark:hover:bg-gray-700'
										: 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'}"
									on:click={() => handleToggleRecursive(source)}
								>
									<!-- Folder tree icon -->
									<svg
										xmlns="http://www.w3.org/2000/svg"
										class="w-4 h-4"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											stroke-width="2"
											d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
										/>
									</svg>
								</button>
							</Tooltip>

							<!-- Toggle Auto-sync -->
							<Tooltip
								content={source.sync_enabled
									? $i18n.t('Disable auto-sync')
									: $i18n.t('Enable auto-sync')}
							>
								<button
									class="p-1.5 rounded-lg transition-colors {source.sync_enabled
										? 'text-green-600 hover:text-green-700 dark:text-green-400 hover:bg-gray-200 dark:hover:bg-gray-700'
										: 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'}"
									on:click={() => handleToggleSync(source)}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										class="w-4 h-4"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											stroke-width="2"
											d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
										/>
									</svg>
								</button>
							</Tooltip>

							<!-- Disconnect Button -->
							<Tooltip content={$i18n.t('Disconnect')}>
								<button
									class="p-1.5 text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
									on:click={() =>
										confirmDisconnect(source.id, source.drive_folder_name || 'folder')}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										class="w-4 h-4"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											stroke-width="2"
											d="M6 18L18 6M6 6l12 12"
										/>
									</svg>
								</button>
							</Tooltip>
						</div>
					{/if}
				</div>
			</div>
		{/each}
	</div>
{/if}
