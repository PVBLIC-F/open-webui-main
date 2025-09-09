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

		// First, try to clean up any obvious JSON wrapper artifacts in the raw content
		let cleanContent = content.trim();
		
		// Handle case where content might be showing raw JSON structure
		if (cleanContent.startsWith('{"video_description":') || cleanContent.includes('"video_description":')) {
			try {
				const parsed = JSON.parse(cleanContent);
				if (parsed.video_description && typeof parsed.video_description === 'string') {
					return { formatted: formatVideoDescription(parsed.video_description), isStructured: true };
				}
			} catch (e) {
				// If JSON parsing fails, try to extract video_description manually
				const match = cleanContent.match(/"video_description":\s*"([^"]+(?:\\.[^"]*)*)"/) || 
							 cleanContent.match(/"video_description":\s*'([^']+(?:\\.[^']*)*)'/) ||
							 cleanContent.match(/video_description['":\s]*([^}]+)/i);
				if (match && match[1]) {
					return { formatted: formatVideoDescription(match[1].replace(/\\"/g, '"')), isStructured: true };
				}
			}
		}

		try {
			// Try to parse as JSON
			const parsed = JSON.parse(cleanContent);
			
			// Handle video_description object
			if (parsed.video_description && typeof parsed.video_description === 'string') {
				return { formatted: formatVideoDescription(parsed.video_description), isStructured: true };
			}
			
			// Handle other structured content
			if (typeof parsed === 'object') {
				return { formatted: formatObjectContent(parsed), isStructured: true };
			}
		} catch (e) {
			// Not valid JSON - check if it looks like video content anyway
			if (cleanContent.toLowerCase().includes('video') || cleanContent.includes('**') || cleanContent.includes('Visuals:')) {
				return { formatted: formatVideoDescription(cleanContent), isStructured: true };
			}
			// Format as plain text with better paragraph breaks
			return { formatted: formatTextContent(cleanContent), isStructured: true };
		}

		return { formatted: formatTextContent(cleanContent), isStructured: true };
	}

	// Format video description content with proper structure and sections
	function formatVideoDescription(text: string): string {
		if (!text) return '';
		
		return text
			// Remove comprehensive wrapper patterns - be more aggressive
			.replace(/^"?(?:Here is a (?:detailed )?description of the (?:video|audio)(?: provided)?:?|Sure!? Here is a (?:detailed )?description of the (?:video|audio):?)\s*/i, '')
			.replace(/^"?(?:Here is (?:a )?(?:detailed )?description of the (?:video|audio):?)\s*/i, '')
			.replace(/^"?(?:Sure! Here is (?:a )?(?:detailed )?description of the (?:video|audio):?)\s*/i, '')
			// Remove any remaining JSON artifacts
			.replace(/^[{"]?video_description["}]?\s*:\s*["{]?\s*/i, '')
			.replace(/^["{]?\s*/, '')
			// Clean up excessive whitespace
			.replace(/\s+/g, ' ')
			// Format section headers (text followed by colons)
			.replace(/\*\*([^*]+):\*\*/g, '\n\n**$1:**\n')
			// Add breaks before bullet points and list items
			.replace(/(\* \*\*[^*]+\*\*)/g, '\n$1')
			// Add breaks after periods when followed by section markers
			.replace(/\.\s+(\*\*[^*]+\*\*)/g, '.\n\n$1')
			// Add breaks before "Content of the Discussion" and similar major sections
			.replace(/(\*\*Content of the Discussion:\*\*)/g, '\n\n$1')
			.replace(/(\*\*Visuals:\*\*)/g, '\n\n$1')
			.replace(/(\*\*Audio:\*\*)/g, '\n\n$1')
			.replace(/(\*\*Overall Impression:\*\*)/g, '\n\n$1')
			.replace(/(\*\*Setting and People:\*\*)/g, '\n\n$1')
			// Add proper spacing after bullet points
			.replace(/(\* \*\*[^*]+\*\* [^*]+)/g, '$1\n')
			// Clean up multiple line breaks
			.replace(/\n{3,}/g, '\n\n')
			.trim();
	}

	// Format plain text content with proper paragraph breaks
	function formatTextContent(text: string): string {
		if (!text) return '';
		
		// Clean up the text and add proper paragraph breaks
		return text
			// Remove excessive whitespace but preserve intentional breaks
			.replace(/\s+/g, ' ')
			// Split into sentences and add breaks after periods followed by capital letters
			.replace(/\.\s+([A-Z])/g, '.\n\n$1')
			// Add breaks after colons when followed by capital letters (likely new sections)
			.replace(/:\s+([A-Z][^.]*?\.)/g, ':\n\n$1')
			// Add breaks before common section starters
			.replace(/\b(The [A-Z][A-Za-z\s]+(?:Zone|Media|SDG)[^.]*?\.)/g, '\n\n$1')
			// Add breaks before sentences that start with "Our" or "The" and seem like new paragraphs
			.replace(/\.\s+((?:Our|The)\s+[A-Z][^.]*?(?:includes?|provides?|aims?|makes?|hosts?)[^.]*?\.)/g, '.\n\n$1')
			// Clean up multiple line breaks
			.replace(/\n{3,}/g, '\n\n')
			.trim();
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
			// Convert double line breaks to paragraph breaks
			.replace(/\n\n/g, '</p><p class="mb-3">')
			// Convert single line breaks to <br>
			.replace(/\n/g, '<br>')
			// Wrap in paragraph tags
			.replace(/^/, '<p class="mb-3">')
			.replace(/$/, '</p>')
			// Clean up empty paragraphs
			.replace(/<p class="mb-3"><\/p>/g, '')
			// Bullet points
			.replace(/^• /gm, '&bull; ');
	}
</script>

<Modal size="lg" bind:show>
	<div>
		<div class=" flex justify-between dark:text-gray-300 px-5 pt-4 pb-2">
			<div class=" text-lg font-medium self-center">
				{#if citation?.source?.name}
					{@const document = mergedDocuments?.[0]}
					{#if document?.metadata?.file_id || document.source?.url?.includes('http')}
						<Tooltip
							className="w-fit"
							content={$i18n.t('Open file')}
							placement="top-start"
							tippyOptions={{ duration: [500, 0] }}
						>
							<a
								class="hover:text-gray-500 dark:hover:text-gray-100 underline grow"
								href={document?.metadata?.file_id
									? `${WEBUI_API_BASE_URL}/files/${document?.metadata?.file_id}/content${document?.metadata?.page !== undefined ? `#page=${document.metadata.page + 1}` : ''}`
									: document.source?.url?.includes('http')
										? document.source.url
										: `#`}
								target="_blank"
							>
								{decodeString(citation?.source?.name)}
							</a>
						</Tooltip>
					{:else}
						{decodeString(citation?.source?.name)}
					{/if}
				{:else}
					{$i18n.t('Citation')}
				{/if}
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
				class="flex flex-col w-full dark:text-gray-200 overflow-y-scroll max-h-[22rem] scrollbar-hidden gap-1"
			>
				{#each mergedDocuments as document, documentIdx}
					<div class="flex flex-col w-full gap-2">
						{#if document.metadata?.parameters}
							<div>
								<div class="text-sm font-medium dark:text-gray-300 mb-1">
									{$i18n.t('Parameters')}
								</div>

								<Textarea readonly value={JSON.stringify(document.metadata.parameters, null, 2)}
								></Textarea>
							</div>
						{/if}

						<div>
							<div
								class=" text-sm font-medium dark:text-gray-300 flex items-center gap-2 w-fit mb-1"
							>
								{$i18n.t('Content')}

								{#if showRelevance && document.distance !== undefined}
									<Tooltip
										className="w-fit"
										content={$i18n.t('Relevance')}
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
											{:else if typeof document?.distance === 'number'}
												<span class="text-gray-500 dark:text-gray-500">
													({(document?.distance ?? 0).toFixed(4)})
												</span>
											{/if}
										</div>
									</Tooltip>
								{/if}

								{#if Number.isInteger(document?.metadata?.page)}
									<span class="text-sm text-gray-500 dark:text-gray-400">
										({$i18n.t('page')}
										{document.metadata.page + 1})
									</span>
								{/if}
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
									<div class="text-sm text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2 leading-relaxed">
										<div class="whitespace-pre-line">
											{@html renderFormattedText(contentFormat.formatted)}
										</div>
									</div>
								{:else}
									<pre class="text-sm dark:text-gray-400 whitespace-pre-line bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2 leading-relaxed">
{contentFormat.formatted}
									</pre>
								{/if}
							{/if}
						</div>
					</div>
				{/each}
			</div>
		</div>
	</div>
</Modal>
