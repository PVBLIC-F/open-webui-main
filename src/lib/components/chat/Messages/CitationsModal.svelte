<script lang="ts">
	import { getContext } from 'svelte';
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
	let documentsArray = [];

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

	// Helper functions for constructing proxy URLs
	function constructProxyUrl(docId: string, mediaType: string) {
		if (!docId) return null;
		const baseUrl = `/api/proxy/ragie/stream`;
		const ragieUrl = `https://api.ragie.ai/documents/${docId}/content?media_type=${encodeURIComponent(mediaType)}`;
		return `${baseUrl}?url=${encodeURIComponent(ragieUrl)}`;
	}
	
	function normalizeProxyUrl(url: string) {
		if (!url) return null;
		// If it's already a relative URL, return as-is
		if (url.startsWith('/api/proxy/ragie/stream')) {
			return url;
		}
		// If it's an absolute URL with our proxy path, make it relative
		if (url.includes('/api/proxy/ragie/stream')) {
			const match = url.match(/\/api\/proxy\/ragie\/stream\?.*$/);
			if (match) {
				return match[0];
			}
		}
		return url;
	}
	
	function extractDocumentId(source: any) {
		// Try to extract document ID from various possible locations
		if (source?.id) return source.id;
		if (source?.document_id) return source.document_id;
		// Try to extract from URL if present
		if (source?.url) {
			const match = source.url.match(/documents\/([^\/]+)/);
			if (match) return match[1];
		}
		// Try to extract from source string if it's a document ID
		if (typeof source === 'string' && source.match(/^[a-f0-9-]{36}$/)) {
			return source;
		}
		// Try metadata.source if it's a document ID
		if (source?.source && typeof source.source === 'string' && source.source.match(/^[a-f0-9-]{36}$/)) {
			return source.source;
		}
		return null;
	}

	$: if (citation) {
		console.log('=== CitationsModal Debug ===');
		console.log('Citation prop received:', citation);
		console.log('Citation type:', typeof citation);
		console.log('Citation keys:', Object.keys(citation || {}));
		console.log('Citation.document:', citation.document);
		console.log('Citation.document type:', typeof citation.document);
		console.log('Citation.document length:', Array.isArray(citation.document) ? citation.document.length : 'Not an array');
		
		// Convert object to array if needed
		documentsArray = citation.document;
		console.log('Initial documentsArray:', documentsArray);
		console.log('Initial documentsArray type:', typeof documentsArray);
		console.log('Initial documentsArray isArray:', Array.isArray(documentsArray));
		
		if (citation.document && typeof citation.document === 'object' && !Array.isArray(citation.document)) {
			console.log('Detected object, checking for length property...');
			// If it's an object with numeric keys, convert to array
			if (citation.document.length && typeof citation.document.length === 'number') {
				console.log('Object has length property:', citation.document.length);
				console.log('Object keys:', Object.keys(citation.document));
				documentsArray = Array.from({ length: citation.document.length }, (_, i) => {
					const item = citation.document[i];
					console.log(`Item ${i}:`, item);
					return item;
				});
				console.log('Converted object to array:', documentsArray);
			} else {
				// If it's a single object, wrap it in an array
				documentsArray = [citation.document];
				console.log('Wrapped single object in array:', documentsArray);
			}
		}
		
		console.log('Final documentsArray:', documentsArray);
		console.log('Final documentsArray type:', typeof documentsArray);
		console.log('Final documentsArray isArray:', Array.isArray(documentsArray));
		console.log('Final documentsArray length:', documentsArray?.length);
		
		mergedDocuments = documentsArray?.map((c, i) => {
			const doc = {
				source: citation.source,
				document: c,
				metadata: citation.metadata?.[i],
				distance: citation.distances?.[i]
			};
			console.log(`Document ${i}:`, doc);
			return doc;
		});
		
		console.log('Final mergedDocuments:', mergedDocuments);
		console.log('mergedDocuments length:', mergedDocuments?.length || 0);
		
		if (mergedDocuments?.every((doc) => doc.distance !== undefined)) {
			mergedDocuments = mergedDocuments.sort(
				(a, b) => (b.distance ?? Infinity) - (a.distance ?? Infinity)
			);
		}
		console.log('=== End CitationsModal Debug ===');
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
		<div class="flex justify-between dark:text-gray-300 px-5 pt-4 pb-2">
			<div class="text-lg font-medium self-center capitalize">
				{$i18n.t('Citation')} {mergedDocuments.length > 1 ? `(${mergedDocuments.length} sources)` : ''}
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
				<!-- Debug Info -->
				<div class="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded p-2 mb-2 text-xs">
					<strong>Debug:</strong> mergedDocuments.length = {mergedDocuments?.length || 0} | 
					citation.document type = {typeof citation?.document} | 
					citation.document length = {citation?.document?.length || 'No length'} | 
					documentsArray type = {typeof documentsArray} | 
					documentsArray length = {Array.isArray(documentsArray) ? documentsArray?.length : 'Not array'}
				</div>
				
				{#each mergedDocuments as document, documentIdx}
					<div class="flex flex-col w-full">
						<div class="text-sm font-medium dark:text-gray-300">
							{$i18n.t('Source')} #{documentIdx + 1}
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
						<div class="text-sm font-medium dark:text-gray-300 mt-2">
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
							{@const isJsonContent = document.document && typeof document.document === 'string' && (document.document.trim().startsWith('{') || document.document.includes('"video_description"') || document.document.includes('"audio_transcript"'))}
							{@const debugContent = (() => {
								console.log('=== Content Debug ===');
								console.log('document.document:', document.document);
								console.log('document.document type:', typeof document.document);
								console.log('document.document starts with {:', document.document?.trim().startsWith('{'));
								console.log('isJsonContent:', isJsonContent);
								return true;
							})()}
							{#if isJsonContent}
								{@const parsedContent = (() => {
									try {
										console.log('Attempting to parse JSON:', document.document);
										const parsed = JSON.parse(document.document);
										console.log('Parsed content:', parsed);
										// Validate that it's an object with expected properties
										return typeof parsed === 'object' && parsed !== null ? parsed : null;
									} catch (error) {
										console.warn('Failed to parse JSON content:', error);
										// Try to extract content from malformed JSON using more robust regex patterns
										const content = document.document;
										console.log('Attempting to extract from malformed JSON:', content);
										
										// More robust extraction patterns
										const extracted = {};
										
										// Extract video_description - handle various formats including truncated JSON
										const videoPatterns = [
											/"video_description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"/,
											/"video_description":\s*"([^"]+)/,
											/video_description['":\s]+([^"'\n]+)/i
										];
										
										for (const pattern of videoPatterns) {
											const match = content.match(pattern);
											if (match) {
												extracted.video_description = match[1].replace(/\\"/g, '"').replace(/\\n/g, '\n');
												console.log('Extracted video_description:', extracted.video_description);
												break;
											}
										}
										
										// Extract audio_transcript - handle various formats including truncated JSON
										const audioPatterns = [
											/"audio_transcript"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"/,
											/"audio_transcript":\s*"([^"]+)/,
											/audio_transcript['":\s]+([^"'\n]+)/i
										];
										
										for (const pattern of audioPatterns) {
											const match = content.match(pattern);
											if (match) {
												extracted.audio_transcript = match[1].replace(/\\"/g, '"').replace(/\\n/g, '\n');
												console.log('Extracted audio_transcript:', extracted.audio_transcript);
												break;
											}
										}
										
										// If we found anything, return it
										if (Object.keys(extracted).length > 0) {
											return extracted;
										}
										
										// Final fallback: if content looks like it contains relevant data, show it as-is
										if (content.includes('video_description') || content.includes('audio_transcript')) {
											console.log('Showing raw content as fallback');
											return { raw_content: content };
										}
										
										return null;
									}
								})()}
								{#if parsedContent && (parsedContent.video_description || parsedContent.audio_transcript || parsedContent.raw_content)}
									<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2 space-y-4">
										<!-- Debug: Show available metadata -->
										<div class="bg-yellow-100 dark:bg-yellow-900 p-3 rounded-lg mb-4 text-xs">
											<strong>Debug - Available URLs:</strong><br/>
											metadata.video_url: {document.metadata?.video_url || 'NOT SET'}<br/>
											metadata.audio_url: {document.metadata?.audio_url || 'NOT SET'}<br/>
											metadata.stream_url: {document.metadata?.stream_url || 'NOT SET'}<br/>
											source.url: {document.source?.url || 'NOT SET'}<br/>
											source.video_url: {document.source?.video_url || 'NOT SET'}<br/>
											source.audio_url: {document.source?.audio_url || 'NOT SET'}<br/>
											source.fallback_url: {document.source?.fallback_url || 'NOT SET'}<br/>
											<br/>
											<strong>Full document structure:</strong><br/>
											<pre>{JSON.stringify(document, null, 2)}</pre>
										</div>
										
										<!-- Content Type Indicator -->
										<div class="flex items-center gap-2 mb-2 pb-2 border-b border-gray-200 dark:border-gray-700">
											<span class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
												{parsedContent.video_description && parsedContent.audio_transcript ? 'Multimedia Content' : 
												 parsedContent.video_description ? 'Video Content' : 'Audio Content'}
											</span>
										</div>
										
										{#if parsedContent.video_description}
											{@const cleanDescription = parsedContent.video_description
												.replace(/\*\*([^*]+)\*\*/g, '$1')
												.replace(/\*([^*]+)\*/g, '$1')
												.replace(/\\u[\da-f]{4}/gi, '')
												.replace(/\s+/g, ' ')
												.trim()}
											{@const docId = extractDocumentId(document.source) || extractDocumentId(document.metadata) || extractDocumentId(document)}
											{@const rawVideoUrl = document.source?.video_url || document.metadata?.video_url || (docId && parsedContent.video_description ? constructProxyUrl(docId, 'video/mp4') : null)}
											{@const videoUrl = normalizeProxyUrl(rawVideoUrl)}
											{@const fallbackUrl = document.source?.fallback_url || document.metadata?.fallback_url}
											<div>
												<div class="flex items-center gap-2 mb-3">
													<span class="text-lg">🎬</span>
													<span class="text-sm font-semibold text-gray-700 dark:text-gray-300">Video Content</span>
												</div>
												<div class="text-sm dark:text-gray-200 leading-relaxed pl-6 border-l-2 border-blue-200 dark:border-blue-600">
													<div class="whitespace-pre-wrap break-words">
														{cleanDescription}
													</div>
												</div>
												
												<!-- Embedded Video Player -->
												{#if videoUrl || fallbackUrl}
													{@const videoDebug = (() => {
														console.log('=== Video URL Debug ===');
														console.log('Full document:', document);
														console.log('document.source:', document.source);
														console.log('document.metadata:', document.metadata);
														console.log('Extracted docId:', docId);
														console.log('document.source?.video_url:', document.source?.video_url);
														console.log('document.metadata?.video_url:', document.metadata?.video_url);
														console.log('Raw videoUrl:', rawVideoUrl);
														console.log('Normalized videoUrl:', videoUrl);
														console.log('Final fallbackUrl:', fallbackUrl);
														console.log('parsedContent.video_description exists:', !!parsedContent.video_description);
														return true;
													})()}
													<div class="mt-4 pl-6">
														<div class="bg-black rounded-lg overflow-hidden shadow-lg">
															<video 
																controls
																preload="metadata"
																class="w-full max-h-96"
																poster=""
															>
																{#if videoUrl}
																	<source src={videoUrl} type="video/mp4" />
																{/if}
																{#if fallbackUrl}
																	<source src={fallbackUrl} type="video/mp4" />
																{/if}
																<!-- Add empty track for accessibility -->
																<track kind="captions" />
																<p class="text-sm text-gray-500 p-4">
																	Your browser doesn't support embedded video. 
																	<a href={fallbackUrl || videoUrl} class="text-blue-500 hover:underline" target="_blank">
																		Click here to view the video.
																	</a>
																</p>
															</video>
														</div>
													</div>
												{/if}
												
												<!-- Video time info if available -->
												{#if document.metadata?.start_time && document.metadata?.end_time}
													<div class="mt-2 text-xs text-gray-500 dark:text-gray-400 pl-6">
														Segment: {document.metadata.start_time}s - {document.metadata.end_time}s 
														({document.metadata.duration || `${(document.metadata.end_time - document.metadata.start_time).toFixed(1)}s`})
													</div>
												{/if}
												
												<!-- Show video streaming links -->
												<div class="mt-3 pl-6 space-y-2">
													<!-- External streaming link (if proxy fails) -->
													{#if videoUrl}
														<a 
															href={videoUrl} 
															class="inline-flex items-center gap-2 px-3 py-2 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors text-sm font-medium"
															target="_blank"
														>
															<span>🎬</span>
															<span>Open Video Stream</span>
														</a>
													{/if}
													
													<!-- Google Drive fallback -->
													{#if fallbackUrl}
														<a 
															href={fallbackUrl} 
															class="inline-flex items-center gap-2 px-3 py-2 bg-blue-50 dark:bg-blue-800 text-blue-600 dark:text-blue-200 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-700 transition-colors text-sm border border-blue-200 dark:border-blue-600"
															target="_blank"
														>
															<span>📁</span>
															<span>Open in Google Drive</span>
														</a>
													{/if}
													
													<!-- Fallback video link from source file -->
													{#if document.source?.name && (document.source.name.toLowerCase().includes('.mp4') || document.source.name.toLowerCase().includes('.avi') || document.source.name.toLowerCase().includes('.mov') || document.source.name.toLowerCase().includes('.mkv'))}
														<a 
															href={document.source.url || `#`}
															class="inline-flex items-center gap-2 px-3 py-2 bg-blue-50 dark:bg-blue-800 text-blue-600 dark:text-blue-200 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-700 transition-colors text-sm border border-blue-200 dark:border-blue-600"
															target="_blank"
														>
															<span>📁</span>
															<span>Open Video File</span>
														</a>
													{/if}
												</div>
											</div>
										{/if}
										
										{#if parsedContent.audio_transcript}
											{@const cleanTranscript = parsedContent.audio_transcript
												.replace(/\*\*([^*]+)\*\*/g, '$1')
												.replace(/\*([^*]+)\*/g, '$1')
												.replace(/\\u[\da-f]{4}/gi, '')
												.replace(/\s+/g, ' ')
												.trim()}
											{@const audioDocId = extractDocumentId(document.source) || extractDocumentId(document.metadata) || extractDocumentId(document)}
											{@const rawAudioUrl = document.source?.audio_url || document.metadata?.audio_url || (audioDocId && parsedContent.audio_transcript ? constructProxyUrl(audioDocId, 'audio/mpeg') : null)}
											{@const audioUrl = normalizeProxyUrl(rawAudioUrl)}
											{@const audioFallbackUrl = document.source?.fallback_url || document.metadata?.fallback_url}
											<div>
												<div class="flex items-center gap-2 mb-3">
													<span class="text-lg">🎵</span>
													<span class="text-sm font-semibold text-gray-700 dark:text-gray-300">Audio Transcript</span>
												</div>
												<div class="text-sm dark:text-gray-200 leading-relaxed pl-6 border-l-2 border-green-200 dark:border-green-600">
													<div class="whitespace-pre-wrap break-words">
														{cleanTranscript}
													</div>
												</div>
												
												<!-- Embedded Audio Player -->
												{#if audioUrl || audioFallbackUrl}
													{@const audioDebug = (() => {
														console.log('=== Audio URL Debug ===');
														console.log('Full document:', document);
														console.log('document.source:', document.source);
														console.log('document.metadata:', document.metadata);
														console.log('Extracted audioDocId:', audioDocId);
														console.log('document.source?.audio_url:', document.source?.audio_url);
														console.log('document.metadata?.audio_url:', document.metadata?.audio_url);
														console.log('Raw audioUrl:', rawAudioUrl);
														console.log('Normalized audioUrl:', audioUrl);
														console.log('Final audioFallbackUrl:', audioFallbackUrl);
														console.log('parsedContent.audio_transcript exists:', !!parsedContent.audio_transcript);
														return true;
													})()}
													<div class="mt-4 pl-6">
														<div class="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 shadow-inner">
															<audio 
																controls
																preload="metadata"
																class="w-full"
															>
																{#if audioUrl}
																	<source src={audioUrl} type="audio/mpeg" />
																{/if}
																{#if audioFallbackUrl}
																	<source src={audioFallbackUrl} />
																{/if}
																<p class="text-sm text-gray-500">
																	Your browser doesn't support embedded audio. 
																	<a href={audioFallbackUrl || audioUrl} class="text-blue-500 hover:underline" target="_blank">
																		Click here to play the audio.
																	</a>
																</p>
															</audio>
														</div>
													</div>
												{/if}
												
												<!-- Audio time info if available -->
												{#if document.metadata?.start_time && document.metadata?.end_time}
													<div class="mt-2 text-xs text-gray-500 dark:text-gray-400 pl-6">
														Segment: {document.metadata.start_time}s - {document.metadata.end_time}s 
														({document.metadata.duration || `${(document.metadata.end_time - document.metadata.start_time).toFixed(1)}s`})
													</div>
												{/if}
												
												<!-- Show audio streaming links -->
												<div class="mt-3 pl-6 space-y-2">
													<!-- External streaming link (if proxy fails) -->
													{#if audioUrl}
														<a 
															href={audioUrl} 
															class="inline-flex items-center gap-2 px-3 py-2 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-200 dark:hover:bg-green-800 transition-colors text-sm font-medium"
															target="_blank"
														>
															<span>🎵</span>
															<span>Open Audio Stream</span>
														</a>
													{/if}
													
													<!-- Google Drive fallback -->
													{#if audioFallbackUrl}
														<a 
															href={audioFallbackUrl} 
															class="inline-flex items-center gap-2 px-3 py-2 bg-green-50 dark:bg-green-800 text-green-600 dark:text-green-200 rounded-lg hover:bg-green-100 dark:hover:bg-green-700 transition-colors text-sm border border-green-200 dark:border-green-600"
															target="_blank"
														>
															<span>📁</span>
															<span>Open in Google Drive</span>
														</a>
													{/if}
													
													<!-- Fallback audio link from source file -->
													{#if document.source?.name && (document.source.name.toLowerCase().includes('.mp3') || document.source.name.toLowerCase().includes('.wav') || document.source.name.toLowerCase().includes('.m4a') || document.source.name.toLowerCase().includes('.aac'))}
														<a 
															href={document.source.url || `#`}
															class="inline-flex items-center gap-2 px-3 py-2 bg-green-50 dark:bg-green-800 text-green-600 dark:text-green-200 rounded-lg hover:bg-green-100 dark:hover:bg-green-700 transition-colors text-sm border border-green-200 dark:border-green-600"
															target="_blank"
														>
															<span>📁</span>
															<span>Open Audio File</span>
														</a>
													{/if}
												</div>
											</div>
										{/if}
										
										{#if parsedContent.raw_content && !(parsedContent.video_description || parsedContent.audio_transcript)}
											<div>
												<div class="flex items-center gap-2 mb-3">
													<span class="text-lg">📄</span>
													<span class="text-sm font-semibold text-gray-700 dark:text-gray-300">Content</span>
												</div>
												<div class="text-sm dark:text-gray-200 leading-relaxed pl-6 border-l-2 border-gray-300 dark:border-gray-600">
													<pre class="whitespace-pre-wrap break-words text-xs font-mono bg-gray-100 dark:bg-gray-900 p-3 rounded overflow-x-auto">{parsedContent.raw_content}</pre>
												</div>
											</div>
										{/if}
									</div>
								{:else if parsedContent && parsedContent.raw_content}
									<!-- Raw content extracted from malformed JSON -->
									<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2">
										<div class="flex items-center gap-2 mb-3">
											<span class="text-lg">📄</span>
											<span class="text-sm font-semibold text-gray-700 dark:text-gray-300">Content (Extracted)</span>
										</div>
										<pre class="whitespace-pre-wrap break-words text-xs font-mono bg-gray-100 dark:bg-gray-900 p-3 rounded overflow-x-auto">{parsedContent.raw_content}</pre>

									</div>
								{:else}
									<!-- Fallback for non-media JSON or failed parsing -->
									{@const fallbackDebug = (() => {
										console.log('=== Fallback Debug ===');
										console.log('Content fell through to fallback section');
										console.log('document.document:', document.document);
										console.log('parsedContent was falsy or missing video_description/audio_transcript');
										return true;
									})()}
									<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2">
										<div class="flex items-center gap-2 mb-3">
											<span class="text-lg">📄</span>
											<span class="text-sm font-semibold text-gray-700 dark:text-gray-300">Raw Content</span>
										</div>
										<div class="text-sm dark:text-gray-300 leading-relaxed pl-6 border-l-2 border-gray-200 dark:border-gray-600">
											<div class="whitespace-pre-wrap break-words">
												{document.document}
											</div>
										</div>
									</div>
								{/if}
							{:else}
								<!-- Regular text content -->
								<div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mt-2">
									<div class="flex items-center gap-2 mb-3">
										<span class="text-lg">📝</span>
										<span class="text-sm font-semibold text-gray-700 dark:text-gray-300">Text Content</span>
									</div>
									<div class="text-sm dark:text-gray-300 leading-relaxed pl-6 border-l-2 border-gray-200 dark:border-gray-600">
										<div class="whitespace-pre-wrap break-words">
											{document.document}
										</div>
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