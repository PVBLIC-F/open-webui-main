<script lang="ts">
	import { getContext, onMount, tick } from 'svelte';
	import Modal from '$lib/components/common/Modal.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import { WEBUI_API_BASE_URL } from '$lib/constants';

	import XMark from '$lib/components/icons/XMark.svelte';
	import Textarea from '$lib/components/common/Textarea.svelte';

	interface CitationDocument {
		source: any;
		document: string;
		metadata?: any;
		distance?: number;
	}

	interface Citation {
		id: string;
		source: any;
		document: string[];
		metadata: any[];
		distances?: number[];
	}

	const i18n = getContext('i18n');

	export let show = false;
	export let citation: Citation | null = null;
	export let showPercentage = false;
	export let showRelevance = true;

	let mergedDocuments: CitationDocument[] = [];

	function calculatePercentage(distance: number) {
		if (typeof distance !== 'number') return null;
		if (distance < 0) return 0;
		if (distance > 1) return 100;
		return Math.round(distance * 10000) / 100;
	}

	function getRelevanceColor(percentage: number) {
		if (percentage >= 80)
			return 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200';
		if (percentage >= 60)
			return 'bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200';
		if (percentage >= 40)
			return 'bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200';
		return 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200';
	}

	$: if (citation) {
		mergedDocuments = citation.document?.map((c: string, i: number): CitationDocument => {
			return {
				source: citation.source,
				document: c,
				metadata: citation.metadata?.[i],
				distance: citation.distances?.[i]
			};
		}) || [];
		if (mergedDocuments.every((doc: CitationDocument) => doc.distance !== undefined)) {
			mergedDocuments = mergedDocuments.sort(
				(a: CitationDocument, b: CitationDocument) => (b.distance ?? Infinity) - (a.distance ?? Infinity)
			);
		}
	}

	const decodeString = (str: string) => {
		try {
			return decodeURIComponent(str);
		} catch (e) {
			return str;
		}
	};

	// Function to parse and format citation content for better readability
	function formatCitationContent(content: string): { formatted: string; isStructured: boolean } {
		if (!content) return { formatted: '', isStructured: false };

		try {
			// Try to parse as JSON first
			const parsed = JSON.parse(content);
			
			// Handle video_description object
			if (parsed.video_description && typeof parsed.video_description === 'string') {
				return { formatted: parsed.video_description, isStructured: true };
			}
			
			// Handle other structured content
			if (typeof parsed === 'object') {
				return { formatted: formatObjectContent(parsed), isStructured: true };
			}
		} catch (e) {
			// Not JSON, check if it looks like structured text
			if (content.includes('**') || content.includes('*') || content.includes('\n')) {
				return { formatted: content, isStructured: true };
			}
		}

		return { formatted: content, isStructured: false };
	}

	// Format object content into readable text
	function formatObjectContent(obj: any): string {
		if (typeof obj !== 'object' || obj === null) return String(obj);
		
		let result = '';
		
		for (const [key, value] of Object.entries(obj)) {
			// Format key as header
			const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
			result += `**${formattedKey}:**\n`;
			
			if (typeof value === 'string') {
				result += `${value}\n\n`;
			} else if (Array.isArray(value)) {
				value.forEach(item => {
					result += `• ${item}\n`;
				});
				result += '\n';
			} else if (typeof value === 'object') {
				result += `${JSON.stringify(value, null, 2)}\n\n`;
			} else {
				result += `${value}\n\n`;
			}
		}
		
		return result.trim();
	}

	// Function to render markdown-like text
	function renderFormattedText(text: string): string {
		return text
			// Bold text
			.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
			// Italic text  
			.replace(/\*(.*?)\*/g, '<em>$1</em>')
			// Line breaks
			.replace(/\n/g, '<br>')
			// Bullet points
			.replace(/^• /gm, '&bull; ');
	}
</script>

<Modal size="lg" bind:show>
	<div>
		<div class=" flex justify-between dark:text-gray-300 px-5 pt-4 pb-2">
			<div class=" text-lg font-medium self-center capitalize">
				{$i18n.t('Citation')}
			</div>
			<button
				class="self-center"
				on:click={() => {
					show = false;
				}}
			>
				<XMark className={'size-5'} />
			</button>
		</div>

		<div class="flex flex-col md:flex-row w-full px-6 pb-5 md:space-x-4">
			<div
				class="flex flex-col w-full dark:text-gray-200 overflow-y-scroll max-h-[22rem] scrollbar-hidden"
			>
				{#each mergedDocuments as document, documentIdx}
					<div class="flex flex-col w-full">
						<div class="text-sm font-medium dark:text-gray-300">
							{$i18n.t('Source')}
						</div>

						{#if document.source?.name}
							<Tooltip
								className="w-fit"
								content={$i18n.t('Open file')}
								placement="top-start"
								tippyOptions={{ duration: [500, 0] }}
							>
								<div class="text-sm dark:text-gray-400 flex items-center gap-2 w-fit">
									<a
										class="hover:text-gray-500 dark:hover:text-gray-100 underline grow"
										href={document?.metadata?.file_id
											? `${WEBUI_API_BASE_URL}/files/${document?.metadata?.file_id}/content${document?.metadata?.page !== undefined ? `#page=${document.metadata.page + 1}` : ''}`
											: document.source?.url?.includes('http')
												? document.source.url
												: `#`}
										target="_blank"
									>
										{decodeString(document?.metadata?.name ?? document.source.name)}
									</a>
									{#if Number.isInteger(document?.metadata?.page)}
										<span class="text-xs text-gray-500 dark:text-gray-400">
											({$i18n.t('page')}
											{document.metadata.page + 1})
										</span>
									{/if}
								</div>
							</Tooltip>
							{#if document.metadata?.parameters}
								<div class="text-sm font-medium dark:text-gray-300 mt-2 mb-0.5">
									{$i18n.t('Parameters')}
								</div>

								<Textarea readonly value={JSON.stringify(document.metadata.parameters, null, 2)}
								></Textarea>
							{/if}
							{#if showRelevance}
								<div class="text-sm font-medium dark:text-gray-300 mt-2">
									{$i18n.t('Relevance')}
								</div>
								{#if document.distance !== undefined}
									<Tooltip
										className="w-fit"
										content={$i18n.t('Semantic distance to query')}
										placement="top-start"
										tippyOptions={{ duration: [500, 0] }}
									>
										<div class="text-sm my-1 dark:text-gray-400 flex items-center gap-2 w-fit">
											{#if showPercentage}
												{@const percentage = calculatePercentage(document.distance)}

												{#if typeof percentage === 'number'}
													<span
														class={`px-1 rounded-sm font-medium ${getRelevanceColor(percentage)}`}
													>
														{percentage.toFixed(2)}%
													</span>
												{/if}

												{#if typeof document?.distance === 'number'}
													<span class="text-gray-500 dark:text-gray-500">
														({(document?.distance ?? 0).toFixed(4)})
													</span>
												{/if}
											{:else if typeof document?.distance === 'number'}
												<span class="text-gray-500 dark:text-gray-500">
													({(document?.distance ?? 0).toFixed(4)})
												</span>
											{/if}
										</div>
									</Tooltip>
								{:else}
									<div class="text-sm dark:text-gray-400">
										{$i18n.t('No distance available')}
									</div>
								{/if}
							{/if}
						{:else}
							<div class="text-sm dark:text-gray-400">
								{$i18n.t('No source available')}
							</div>
						{/if}
					</div>
					<div class="flex flex-col w-full">
						<div class=" text-sm font-medium dark:text-gray-300 mt-2">
							{$i18n.t('Content')}
						</div>
						{#if document.metadata?.html}
							<iframe
								class="w-full border-0 h-auto rounded-none"
								sandbox="allow-scripts allow-forms allow-same-origin"
								srcdoc={document.document}
								title={$i18n.t('Content')}
							></iframe>
						{:else}
							{@const contentFormat = formatCitationContent(document.document)}
							{#if contentFormat.isStructured}
								<div class="text-sm dark:text-gray-400 bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2">
									{@html renderFormattedText(contentFormat.formatted)}
								</div>
							{:else}
								<pre class="text-sm dark:text-gray-400 whitespace-pre-line bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2">
{contentFormat.formatted}
								</pre>
							{/if}
						{/if}
					</div>

					{#if documentIdx !== mergedDocuments.length - 1}
						<hr class="border-gray-100 dark:border-gray-850 my-3" />
					{/if}
				{/each}
			</div>
		</div>
	</div>
</Modal>
