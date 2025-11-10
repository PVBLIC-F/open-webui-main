<script lang="ts">
	import { onMount, tick, getContext } from 'svelte';

	import { decodeString } from '$lib/utils';
	import { knowledge } from '$lib/stores';

	import { getKnowledgeBases } from '$lib/apis/knowledge';

	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Database from '$lib/components/icons/Database.svelte';
	import DocumentPage from '$lib/components/icons/DocumentPage.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import ChevronDown from '$lib/components/icons/ChevronDown.svelte';
	import ChevronRight from '$lib/components/icons/ChevronRight.svelte';

	const i18n = getContext('i18n');

	export let onSelect = (e) => {};

	let loaded = false;
	let collections: any[] = [];
	let legacy_collections: any[] = [];
	let legacy_documents: any[] = [];
	let selectedIdx = 0;
	let expandedCollections: Set<string> = new Set(); // Track which collections are expanded
	
	const toggleCollection = (collectionId: string, event: Event) => {
		event.stopPropagation(); // Prevent selecting the collection
		if (expandedCollections.has(collectionId)) {
			expandedCollections.delete(collectionId);
		} else {
			expandedCollections.add(collectionId);
		}
		expandedCollections = expandedCollections; // Trigger reactivity
	};
	
	// Reactive statement to rebuild items when expandedCollections changes
	$: items = (() => {
		let result = [];
		
		// Add collections with files nested
		for (const collection of collections) {
			result.push(collection);
			
			// Add files if collection is expanded
			if (expandedCollections.has(collection.id) && collection.files?.length > 0) {
				for (const file of collection.files) {
					result.push({
						...file,
						isNested: true, // Mark as nested for styling
					});
				}
			}
		}
		
		// Add legacy collections and documents
		result = [...result, ...legacy_collections, ...legacy_documents];
		
		return result;
	})();

	onMount(async () => {
		if ($knowledge === null) {
			await knowledge.set(await getKnowledgeBases(localStorage.token));
		}

		let _legacy_documents = $knowledge
			.filter((item) => item?.meta?.document)
			.map((item) => ({
				...item,
				type: 'file',
				legacy: true
			}));

		legacy_collections =
			_legacy_documents.length > 0
				? [
						{
							name: 'All Documents',
							legacy: true,
							type: 'collection',
							description: 'Deprecated (legacy collection), please create a new knowledge base.',
							title: $i18n.t('All Documents'),
							collection_names: _legacy_documents.map((item) => item.id)
						},

						..._legacy_documents
							.reduce((a, item) => {
								return [...new Set([...a, ...(item?.meta?.tags ?? []).map((tag) => tag.name)])];
							}, [])
							.map((tag) => ({
								name: tag,
								legacy: true,
								type: 'collection',
								description: 'Deprecated (legacy collection), please create a new knowledge base.',
								collection_names: _legacy_documents
									.filter((item) => (item?.meta?.tags ?? []).map((tag) => tag.name).includes(tag))
									.map((item) => item.id)
							}))
					]
				: [];
		
		legacy_documents = _legacy_documents;

		// Build collections with their nested files
		collections = $knowledge
			.filter((item) => !item?.meta?.document)
			.map((item) => ({
				...item,
				type: 'collection',
				// Map files for this collection
				files: (item?.files ?? []).map((file) => ({
					...file,
					name: file?.meta?.name,
					parent_collection_id: item.id,
					parent_collection_name: item.name,
					knowledge: true, // DO NOT REMOVE, USED TO INDICATE KNOWLEDGE BASE FILE
					type: 'file'
				}))
			}));

		await tick();

		loaded = true;
	});
</script>

{#if loaded}
	<div class="flex flex-col gap-0.5">
		{#each items as item, idx}
			<button
				class="px-2.5 py-1 rounded-xl w-full text-left flex justify-between items-center text-sm {item.isNested ? 'pl-8' : ''} {idx ===
				selectedIdx
					? ' bg-gray-50 dark:bg-gray-800 dark:text-gray-100 selected-command-option-button'
					: ''}"
				type="button"
				on:click={() => {
					console.log(item);
					onSelect(item);
				}}
				on:mousemove={() => {
					selectedIdx = idx;
				}}
				on:mouseleave={() => {
					if (idx === 0) {
						selectedIdx = -1;
					}
				}}
				data-selected={idx === selectedIdx}
			>
				<div class="text-black dark:text-gray-100 flex items-center gap-1 flex-1 min-w-0">
					<!-- Expand/Collapse icon for collections with files -->
					{#if item?.type === 'collection' && item?.files?.length > 0}
						<button
							class="shrink-0"
							on:click={(e) => toggleCollection(item.id, e)}
							type="button"
						>
							{#if expandedCollections.has(item.id)}
								<ChevronDown className="size-4" />
							{:else}
								<ChevronRight className="size-4" />
							{/if}
						</button>
					{:else if item?.type === 'collection'}
						<!-- Empty space for collections without files to maintain alignment -->
						<div class="size-4 shrink-0"></div>
					{/if}
					
					<Tooltip
						content={item?.legacy
							? $i18n.t('Legacy')
							: item?.type === 'file'
								? $i18n.t('File')
								: item?.type === 'collection'
									? $i18n.t('Collection')
									: ''}
						placement="top"
					>
						{#if item?.type === 'collection'}
							<Database className="size-4 shrink-0" />
						{:else}
							<DocumentPage className="size-4 shrink-0" />
						{/if}
					</Tooltip>

					<Tooltip content={item.description || decodeString(item?.name)} placement="top-start">
						<div class="line-clamp-1 flex-1 min-w-0">
							{decodeString(item?.name)}
							{#if item?.type === 'collection' && item?.files?.length > 0}
								<span class="text-xs text-gray-500 ml-1">({item.files.length})</span>
							{/if}
						</div>
					</Tooltip>
				</div>
			</button>
		{/each}
	</div>
{:else}
	<div class="py-4.5">
		<Spinner />
	</div>
{/if}
