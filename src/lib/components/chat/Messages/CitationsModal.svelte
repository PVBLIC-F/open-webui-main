<script lang="ts">
	import { getContext, onMount, tick } from 'svelte';
	import Modal from '$lib/components/common/Modal.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import { WEBUI_API_BASE_URL } from '$lib/constants';

	import XMark from '$lib/components/icons/XMark.svelte';
	import Textarea from '$lib/components/common/Textarea.svelte';

	const i18n = getContext('i18n');

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
		if (percentage >= 80)
			return 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200';
		if (percentage >= 60)
			return 'bg-yellow-200 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-200';
		if (percentage >= 40)
			return 'bg-orange-200 dark:bg-orange-800 text-orange-800 dark:text-orange-200';
		return 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200';
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
		} catch (e) {
			return str;
		}
	};
</script>

<Modal size="xl" bind:show>
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
				class="flex flex-col w-full dark:text-gray-200 overflow-y-auto max-h-[32rem] scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600 scrollbar-track-transparent"
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
							{@const isJsonContent = document.document.trim().startsWith('{')}
							{#if isJsonContent}
								{@const parsedContent = (() => {
									try {
										return JSON.parse(document.document);
									} catch {
										return null;
									}
								})()}
								{#if parsedContent && (parsedContent.video_description || parsedContent.audio_transcript)}
									<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2 space-y-4">
										{#if parsedContent.video_description}
											<div>
												<div class="flex items-center gap-2 mb-3">
													<span class="text-lg">🎬</span>
													<span class="text-sm font-semibold text-gray-700 dark:text-gray-300">Video Content</span>
												</div>
												{@const cleanDescription = parsedContent.video_description
													.replace(/\*\*([^*]+)\*\*/g, '$1')
													.replace(/\*([^*]+)\*/g, '$1')
													.replace(/\\u[\da-f]{4}/gi, '')
													.replace(/\s+/g, ' ')
													.trim()}
												<div class="text-sm dark:text-gray-200 leading-relaxed pl-6 border-l-2 border-blue-200 dark:border-blue-600">
													{cleanDescription}
												</div>
												
												<!-- Show video streaming link if available -->
												{#if document.metadata?.video_url}
													<div class="mt-3 pl-6">
														<a 
															href={document.metadata.video_url} 
															class="inline-flex items-center gap-2 px-3 py-2 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors text-sm"
															target="_blank"
														>
															<span>🎬</span>
															<span>Play Video</span>
														</a>
													</div>
												{/if}
											</div>
										{/if}
										
										{#if parsedContent.audio_transcript}
											<div>
												<div class="flex items-center gap-2 mb-3">
													<span class="text-lg">🎵</span>
													<span class="text-sm font-semibold text-gray-700 dark:text-gray-300">Audio Transcript</span>
												</div>
												{@const cleanTranscript = parsedContent.audio_transcript
													.replace(/\*\*([^*]+)\*\*/g, '$1')
													.replace(/\*([^*]+)\*/g, '$1')
													.replace(/\\u[\da-f]{4}/gi, '')
													.replace(/\s+/g, ' ')
													.trim()}
												<div class="text-sm dark:text-gray-200 leading-relaxed pl-6 border-l-2 border-green-200 dark:border-green-600">
													{cleanTranscript}
												</div>
												
												<!-- Show audio streaming link if available and no video -->
												{#if document.metadata?.audio_url && !document.metadata?.video_url}
													<div class="mt-3 pl-6">
														<a 
															href={document.metadata.audio_url} 
															class="inline-flex items-center gap-2 px-3 py-2 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-200 dark:hover:bg-green-800 transition-colors text-sm"
															target="_blank"
														>
															<span>🎵</span>
															<span>Play Audio</span>
														</a>
													</div>
												{/if}
											</div>
										{/if}
										
										<!-- Show both links if both are available -->
										{#if document.metadata?.video_url && document.metadata?.audio_url}
											<div class="flex gap-2 pl-6">
												<a 
													href={document.metadata.video_url} 
													class="inline-flex items-center gap-2 px-3 py-2 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors text-sm"
													target="_blank"
												>
													<span>🎬</span>
													<span>Video</span>
												</a>
												<a 
													href={document.metadata.audio_url} 
													class="inline-flex items-center gap-2 px-3 py-2 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-200 dark:hover:bg-green-800 transition-colors text-sm"
													target="_blank"
												>
													<span>🎵</span>
													<span>Audio</span>
												</a>
											</div>
										{/if}
									</div>
								{:else}
									<!-- Fallback for non-media JSON or failed parsing -->
									<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2">
										<pre class="text-sm dark:text-gray-300 whitespace-pre-wrap leading-relaxed overflow-x-auto">
{document.document}
										</pre>
									</div>
								{/if}
							{:else}
								<!-- Regular text content -->
								<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2">
									<div class="text-sm dark:text-gray-300 whitespace-pre-wrap leading-relaxed">
										{document.document}
									</div>
								</div>
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
