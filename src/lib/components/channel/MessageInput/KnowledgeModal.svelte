<script lang="ts">
	import { getContext } from 'svelte';
	import { knowledge } from '$lib/stores';
	import Modal from '$lib/components/common/Modal.svelte';
	import Database from '$lib/components/icons/Database.svelte';
	import DocumentPage from '$lib/components/icons/DocumentPage.svelte';
	import Folder from '$lib/components/icons/Folder.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	const i18n = getContext('i18n');

	export let show = false;
	export let onSelect: Function = () => {};

	let searchQuery = '';
	let selectedItems = [];

	// Auto-reset state when modal closes
	$: if (!show) {
		selectedItems = [];
		searchQuery = '';
	}

	// Reactive filtered list
	$: filteredKnowledge = searchQuery
		? ($knowledge ?? []).filter(
				(item) =>
					item.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
					item.description?.toLowerCase().includes(searchQuery.toLowerCase())
			)
		: ($knowledge ?? []);

	const handleSelect = (item) => {
		const index = selectedItems.findIndex((i) => i.id === item.id);
		selectedItems =
			index >= 0
				? selectedItems.filter((i) => i.id !== item.id)
				: [...selectedItems, item];
	};

	const handleConfirm = () => {
		selectedItems.forEach(onSelect);
		show = false;
	};
</script>

<Modal bind:show size="sm">
	<div slot="title">
		<div class="flex items-center gap-2">
			<Database className="size-5" />
			<div>{$i18n.t('Select Knowledge')}</div>
		</div>
	</div>

	<div class="flex flex-col gap-3">
		<!-- Search Input -->
		<input
			type="text"
			bind:value={searchQuery}
			placeholder={$i18n.t('Search knowledge...')}
			class="w-full px-3 py-2 rounded-xl bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-sm outline-none focus:ring-2 focus:ring-gray-200 dark:focus:ring-gray-700"
		/>

		<!-- Knowledge List -->
		<div class="flex flex-col gap-1 max-h-96 overflow-y-auto">
			{#if filteredKnowledge.length > 0}
				{#each filteredKnowledge as item}
					{@const isSelected = selectedItems.find((i) => i.id === item.id)}
					<button
						type="button"
						class="flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-800 transition {isSelected
							? 'bg-gray-100 dark:bg-gray-800'
							: ''}"
						on:click={() => handleSelect(item)}
					>
						<div
							class="flex items-center justify-center size-4 rounded {isSelected
								? 'bg-black dark:bg-white'
								: 'border-2 border-gray-300 dark:border-gray-600'}"
						>
							{#if isSelected}
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 20 20"
									fill="currentColor"
									class="size-3 text-white dark:text-black"
								>
									<path
										fill-rule="evenodd"
										d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"
										clip-rule="evenodd"
									/>
								</svg>
							{/if}
						</div>

						<div class="flex-1 flex items-center gap-2 text-left">
							{#if item?.files && item.files.length > 0}
								<Database className="size-4" />
							{:else if item?.type === 'folder'}
								<Folder className="size-4" />
							{:else}
								<DocumentPage className="size-4" />
							{/if}

							<div class="flex-1 min-w-0">
								<div class="text-sm font-medium truncate">{item.name}</div>
								{#if item.description}
									<div class="text-xs text-gray-500 dark:text-gray-400 truncate">
										{item.description}
									</div>
								{/if}
							</div>
						</div>
					</button>
				{/each}
			{:else}
				<div class="text-center py-8 text-gray-500 dark:text-gray-400 text-sm">
					{searchQuery ? $i18n.t('No knowledge bases found') : $i18n.t('No knowledge bases available')}
				</div>
			{/if}
		</div>
	</div>

	<div slot="footer" class="flex justify-end gap-2">
		<button
			type="button"
			class="px-4 py-2 rounded-xl text-sm hover:bg-gray-50 dark:hover:bg-gray-800 transition"
			on:click={() => (show = false)}
		>
			{$i18n.t('Cancel')}
		</button>
		<button
			type="button"
			class="px-4 py-2 rounded-xl text-sm bg-black dark:bg-white text-white dark:text-black hover:bg-gray-900 dark:hover:bg-gray-100 transition disabled:opacity-50 disabled:cursor-not-allowed"
			disabled={selectedItems.length === 0}
			on:click={handleConfirm}
		>
			{$i18n.t('Add')} {selectedItems.length > 0 ? `(${selectedItems.length})` : ''}
		</button>
	</div>
</Modal>

