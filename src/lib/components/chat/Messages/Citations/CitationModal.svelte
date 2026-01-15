<script lang="ts">
	import { getContext } from 'svelte';
	import Modal from '$lib/components/common/Modal.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import { WEBUI_API_BASE_URL, WEBUI_BASE_URL } from '$lib/constants';
	import { settings } from '$lib/stores';

	import XMark from '$lib/components/icons/XMark.svelte';
	import Textarea from '$lib/components/common/Textarea.svelte';
	import Markdown from '$lib/components/chat/Messages/Markdown.svelte';

	const i18n = getContext('i18n');
	
	// Shared class for markdown prose styling with tighter spacing
	const PROSE_CLASS = 'prose dark:prose-invert prose-sm max-w-full prose-headings:mt-4 prose-headings:mb-2 prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 prose-hr:my-3';
	
	/**
	 * Detect if content contains markdown formatting.
	 * Checks for common markdown patterns like headers, bold, lists, etc.
	 */
	function isMarkdownContent(text: string): boolean {
		if (!text) return false;
		const markdownPatterns = [
			/^#{1,6}\s/m,           // Headers
			/\*\*[^*]+\*\*/,        // Bold
			/\*[^*]+\*/,            // Italic
			/^\s*[-*+]\s/m,         // Unordered lists
			/^\s*\d+\.\s/m,         // Ordered lists
			/^>\s/m,                // Blockquotes
			/```[\s\S]*?```/,       // Code blocks
			/`[^`]+`/,              // Inline code
			/\[.+\]\(.+\)/,         // Links
			/^---+$/m,              // Horizontal rules
			/^\|.+\|$/m,            // Tables
		];
		return markdownPatterns.some(pattern => pattern.test(text));
	}
	
	// Audio blob URLs for authenticated requests
	let audioBlobUrls = {};

	export let show = false;
	export let citation;
	export let showPercentage = false;
	export let showRelevance = true;

	let mergedDocuments = [];

	function calculatePercentage(distance: number) {
		if (typeof distance !== 'number') return null;
		if (distance < 0) return 0;
		if (distance > 1) return 100;
		return Math.round(distance * 10000) / 100;
	}

	function getRelevanceColor(percentage: number) {
		// Lower percentage = higher relevance (inverted scale)
		if (percentage <= 20)
			return 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200';
		if (percentage <= 40)
			return 'bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200';
		if (percentage <= 60)
			return 'bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200';
		return 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200';
	}
	
	function getRelevanceLabel(percentage: number) {
		// Lower percentage = higher relevance
		if (percentage <= 20) return 'Highly Relevant';
		if (percentage <= 40) return 'Relevant';
		if (percentage <= 60) return 'Moderately Relevant';
		return 'Low Relevance';
	}

	$: if (citation) {
		mergedDocuments = citation.document?.map((c, i) => {
			return {
				source: citation.source,
				document: c,
				metadata: citation.metadata?.[i],
				distance: citation.distances?.[i]
			};
		});
		if (mergedDocuments.every((doc) => doc.distance !== undefined)) {
			mergedDocuments = mergedDocuments.sort(
				(a, b) => (b.distance ?? Infinity) - (a.distance ?? Infinity)
			);
		}
	}

	const decodeString = (str: string) => {
		try {
			return decodeURIComponent(str);
		} catch {
			return str;
		}
	};

    // Fetch media (audio/video) with authentication and create blob URL
	const loadMediaBlob = async (mediaUrl: string) => {
		if (audioBlobUrls[mediaUrl]) {
			return audioBlobUrls[mediaUrl];
		}
		
		try {
			const token = localStorage.getItem('token');
			const response = await fetch(`${WEBUI_BASE_URL}${mediaUrl}`, {
				method: 'GET',
				headers: {
					'Authorization': `Bearer ${token}`
				},
				credentials: 'include'
			});
			
			if (!response.ok) {
				throw new Error(`Failed to load media: ${response.statusText}`);
			}
			
			const blob = await response.blob();
			const blobUrl = URL.createObjectURL(blob);
			audioBlobUrls[mediaUrl] = blobUrl;
			return blobUrl;
		} catch (error) {
			console.error('Error loading media:', error);
			return null;
		}
    };

    // Programmatic seek for full video playback
    function onVideoMeta(e: Event, start?: number) {
        const el = e.currentTarget as HTMLVideoElement;
        if (!el || start == null) return;
        try {
            el.currentTime = start;
            // Attempt autoplay (will be ignored if not allowed)
            el.play().catch(() => {});
        } catch (_) {}
    }

	const getTextFragmentUrl = (doc: any): string | null => {
		const { metadata, source, document: content } = doc ?? {};
		const { file_id, page } = metadata ?? {};
		const sourceUrl = source?.url;

		const baseUrl = file_id
			? `${WEBUI_API_BASE_URL}/files/${file_id}/content${page !== undefined ? `#page=${page + 1}` : ''}`
			: sourceUrl?.includes('http')
				? sourceUrl
				: null;

		if (!baseUrl || !content) return baseUrl;

		// Extract first and last words for text fragment, filtering out URLs and emojis
		const words = content
			.trim()
			.replace(/\s+/g, ' ')
			.split(' ')
			.filter((w: string) => w.length > 0 && !/https?:\/\/|[\u{1F300}-\u{1F9FF}]/u.test(w));

		if (words.length === 0) return baseUrl;

		const clean = (w: string) => w.replace(/[^\w]/g, '');
		const first = clean(words[0]);
		const last = clean(words.at(-1));
		const fragment = words.length === 1 ? first : `${first},${last}`;

		return fragment ? `${baseUrl}#:~:text=${fragment}` : baseUrl;
	};
</script>

<Modal size="xl" bind:show>
	<div>
		<div class=" flex justify-between dark:text-gray-300 px-4.5 pt-3 pb-2">
			<div class=" text-lg font-medium self-center flex items-center">
				{#if citation?.source?.name}
					{@const document = mergedDocuments?.[0]}
					{#if document?.metadata?.file_id || document.source?.url?.includes('http')}
						<Tooltip
							className="w-fit"
							content={document.source?.url?.includes('http')
								? $i18n.t('Open link')
								: $i18n.t('Open file')}
							placement="top-start"
							tippyOptions={{ duration: [500, 0] }}
						>
							<a
								class="hover:text-gray-500 dark:hover:text-gray-100 underline grow line-clamp-1"
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

		<div class="flex flex-col md:flex-row w-full px-5 pb-5 md:space-x-4">
			<div
				class="flex flex-col w-full dark:text-gray-200 overflow-y-scroll max-h-[36rem] scrollbar-thin gap-1"
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
								{#if document.source?.url?.includes('http')}
									{@const snippetUrl = getTextFragmentUrl(document)}
									{#if snippetUrl}
										<a
											href={snippetUrl}
											target="_blank"
											class="underline hover:text-gray-500 dark:hover:text-gray-100"
											>{$i18n.t('Content')}</a
										>
									{:else}
										{$i18n.t('Content')}
									{/if}
								{:else}
									{$i18n.t('Content')}
								{/if}

								{#if showRelevance && document.distance !== undefined}
									<Tooltip
										className="w-fit"
										content="Relevance score: Lower percentage = more relevant to your query"
										placement="top-start"
										tippyOptions={{ duration: [500, 0] }}
									>
										<div class="text-sm my-1 dark:text-gray-400 flex items-center gap-2 w-fit">
											{#if showPercentage}
												{@const percentage = calculatePercentage(document.distance)}

												{#if typeof percentage === 'number'}
													<span
														class={`px-2 py-0.5 rounded-md font-medium text-xs flex items-center gap-1.5 ${getRelevanceColor(percentage)}`}
													>
														<span>{getRelevanceLabel(percentage)}</span>
														<span class="opacity-75">({percentage.toFixed(1)}%)</span>
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

							{#if document.metadata?.video_segment_url || document.metadata?.audio_segment_url}
								<!-- Video/Audio segment player for multimedia content -->
								<div class="flex flex-col gap-2 p-3 bg-gray-50 dark:bg-gray-850 rounded-lg">
									<div class="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
										<svg
											xmlns="http://www.w3.org/2000/svg"
											fill="none"
											viewBox="0 0 24 24"
											stroke-width="1.5"
											stroke="currentColor"
											class="size-4"
										>
											{#if document.metadata?.video_segment_url}
												<!-- Video icon -->
												<path
													stroke-linecap="round"
													stroke-linejoin="round"
													d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z"
												/>
											{:else}
												<!-- Audio icon -->
												<path
													stroke-linecap="round"
													stroke-linejoin="round"
													d="M19.114 5.636a9 9 0 0 1 0 12.728M16.463 8.288a5.25 5.25 0 0 1 0 7.424M6.75 8.25l4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z"
												/>
											{/if}
										</svg>
										{#if document.metadata?.timestamp_start !== undefined && document.metadata?.timestamp_end !== undefined}
											<span>
												{Math.floor(document.metadata.timestamp_start / 60)}:{String(Math.floor(document.metadata.timestamp_start % 60)).padStart(2, '0')}
												-
												{Math.floor(document.metadata.timestamp_end / 60)}:{String(Math.floor(document.metadata.timestamp_end % 60)).padStart(2, '0')}
											</span>
										{/if}
										{#if document.metadata?.duration !== undefined}
											<span class="text-gray-500 dark:text-gray-500">
												({Math.floor(document.metadata.duration)}s)
											</span>
										{/if}
									</div>
									
                            {#if document.metadata?.video_url}
                                <!-- Stream full video and seek to start timestamp -->
                                <video
                                    controls
                                    preload="metadata"
                                    class="w-full max-h-[32rem] rounded"
                                    style="max-width: 100%; object-fit: contain;"
                                    src={`${WEBUI_BASE_URL}${document.metadata.video_url}`}
                                    on:loadedmetadata={(e) => onVideoMeta(e, document.metadata?.timestamp_start)}
                                    playsinline
                                >
                                    Your browser does not support video playback.
                                </video>
                            {:else if document.metadata?.video_segment_url}
                                <!-- Legacy segment fallback -->
                                {#await loadMediaBlob(document.metadata.video_segment_url)}
                                    <div class="flex items-center justify-center py-4 text-gray-500">
                                        <span class="text-sm">Loading video...</span>
                                    </div>
                                {:then videoBlobUrl}
                                    {#if videoBlobUrl}
                                        <video
                                            controls
                                            preload="metadata"
                                            class="w-full max-h-[32rem] rounded"
                                            style="max-width: 100%; object-fit: contain;"
                                            src={videoBlobUrl}
                                        >
                                            Your browser does not support video playback.
                                        </video>
                                    {:else}
                                        <div class="text-sm text-red-500">Failed to load video</div>
                                    {/if}
                                {:catch}
                                    <div class="text-sm text-red-500">Failed to load video</div>
                                {/await}
									{:else}
										<!-- Audio-only player -->
										{#await loadMediaBlob(document.metadata.audio_segment_url)}
											<div class="flex items-center justify-center py-4 text-gray-500">
												<span class="text-sm">Loading audio...</span>
											</div>
										{:then blobUrl}
											{#if blobUrl}
												<audio
													controls
													preload="metadata"
													class="w-full"
													src={blobUrl}
												>
													Your browser does not support audio playback.
												</audio>
											{:else}
												<div class="text-sm text-red-500">Failed to load audio</div>
											{/if}
										{:catch error}
											<div class="text-sm text-red-500">Error: {error.message}</div>
										{/await}
									{/if}
								</div>
								<!-- Text content below media -->
								{#if isMarkdownContent(document.document)}
									<div class="{PROSE_CLASS} mt-2">
										<Markdown content={document.document} id={`citation-media-${documentIdx}`} />
									</div>
								{:else}
									<pre class="text-sm dark:text-gray-400 whitespace-pre-line mt-2">{document.document}</pre>
								{/if}
							{:else if document.metadata?.html}
								<iframe
									class="w-full border-0 h-auto rounded-none"
									sandbox="allow-scripts allow-forms{($settings?.iframeSandboxAllowSameOrigin ??
									false)
										? ' allow-same-origin'
										: ''}"
									srcdoc={document.document}
									title={$i18n.t('Content')}
								></iframe>
							{:else if isMarkdownContent(document.document)}
								<div class={PROSE_CLASS}>
									<Markdown content={document.document} id={`citation-${documentIdx}`} />
								</div>
							{:else}
								<pre class="text-sm dark:text-gray-400 whitespace-pre-line">{document.document
										.trim()
										.replace(/\n\n+/g, '\n\n')}</pre>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		</div>
	</div>
</Modal>
