<script lang="ts">
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	
	export let src: string;
	export let startTime: number = 0;
	export let endTime: number | undefined = undefined;
	export let totalDuration: number | undefined = undefined;
	export let chapters: Array<{
		index: number;
		title: string;
		summary?: string;
		start: number;
		end: number;
		keywords?: string[];
	}> = [];
	
	let videoElement: HTMLVideoElement;
	let currentTime = 0;
	let duration = 0;
	let isPlaying = false;
	let loopSegment = false;
	let volume = 1;
	let isMuted = false;
	let previousVolume = 1;
	let isFullscreen = false;
	let containerElement: HTMLDivElement;
	let showChapterList = false;
	
	// Computed values
	$: segmentStart = startTime || 0;
	$: segmentEnd = endTime || duration;
	$: segmentDuration = segmentEnd - segmentStart;
	$: videoDuration = totalDuration || duration;
	
	// Calculate highlight position as percentage
	$: highlightLeft = videoDuration > 0 ? (segmentStart / videoDuration) * 100 : 0;
	$: highlightWidth = videoDuration > 0 ? (segmentDuration / videoDuration) * 100 : 100;
	
	// Check if current time is within the relevant segment
	$: isInSegment = currentTime >= segmentStart && currentTime <= segmentEnd;
	
	// Find current chapter based on playback position
	$: currentChapter = chapters.find(ch => currentTime >= ch.start && currentTime < ch.end) || null;
	$: hasChapters = chapters && chapters.length > 0;
	
	function formatTime(seconds: number): string {
		if (!isFinite(seconds)) return '0:00';
		const mins = Math.floor(seconds / 60);
		const secs = Math.floor(seconds % 60);
		return `${mins}:${secs.toString().padStart(2, '0')}`;
	}
	
	function onLoadedMetadata() {
		duration = videoElement.duration;
		// Seek to start of relevant segment
		if (segmentStart > 0) {
			videoElement.currentTime = segmentStart;
		}
	}
	
	function onTimeUpdate() {
		currentTime = videoElement.currentTime;
		
		// Loop within segment if enabled
		if (loopSegment && currentTime >= segmentEnd) {
			videoElement.currentTime = segmentStart;
			videoElement.play();
		}
	}
	
	function onPlay() {
		isPlaying = true;
	}
	
	function onPause() {
		isPlaying = false;
	}
	
	function togglePlayPause() {
		if (isPlaying) {
			videoElement.pause();
		} else {
			videoElement.play();
		}
	}
	
	function seekToPosition(event: MouseEvent | KeyboardEvent) {
		if (event instanceof KeyboardEvent) {
			// Handle keyboard navigation
			const step = videoDuration * 0.05; // 5% jump
			if (event.key === 'ArrowRight') {
				videoElement.currentTime = Math.min(currentTime + step, videoDuration);
			} else if (event.key === 'ArrowLeft') {
				videoElement.currentTime = Math.max(currentTime - step, 0);
			} else if (event.key === 'Home') {
				videoElement.currentTime = segmentStart;
			} else if (event.key === 'End') {
				videoElement.currentTime = segmentEnd;
			} else if (event.key === ' ') {
				event.preventDefault();
				togglePlayPause();
			}
			return;
		}
		
		const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
		const percent = (event.clientX - rect.left) / rect.width;
		const newTime = percent * videoDuration;
		videoElement.currentTime = newTime;
	}
	
	function jumpToSegmentStart() {
		videoElement.currentTime = segmentStart;
		videoElement.play();
	}
	
	function toggleLoop() {
		loopSegment = !loopSegment;
	}
	
	function jumpToChapter(chapter: typeof chapters[0]) {
		videoElement.currentTime = chapter.start;
		videoElement.play();
		showChapterList = false;
	}
	
	function toggleChapterList() {
		showChapterList = !showChapterList;
	}
	
	function toggleMute() {
		if (isMuted) {
			videoElement.volume = previousVolume;
			volume = previousVolume;
			isMuted = false;
		} else {
			previousVolume = volume;
			videoElement.volume = 0;
			volume = 0;
			isMuted = true;
		}
	}
	
	function onVolumeChange(event: Event) {
		const target = event.target as HTMLInputElement;
		volume = parseFloat(target.value);
		videoElement.volume = volume;
		isMuted = volume === 0;
		if (volume > 0) {
			previousVolume = volume;
		}
	}
	
	function skipBack() {
		videoElement.currentTime = Math.max(currentTime - 10, 0);
	}
	
	function skipForward() {
		videoElement.currentTime = Math.min(currentTime + 10, videoDuration);
	}
	
	async function toggleFullscreen() {
		if (!document.fullscreenElement) {
			try {
				await containerElement.requestFullscreen();
				isFullscreen = true;
			} catch (err) {
				console.error('Fullscreen error:', err);
			}
		} else {
			await document.exitFullscreen();
			isFullscreen = false;
		}
	}
	
	function onFullscreenChange() {
		isFullscreen = !!document.fullscreenElement;
	}
</script>

<svelte:document on:fullscreenchange={onFullscreenChange} />

<div 
	bind:this={containerElement}
	class="video-player-container w-full bg-black {isFullscreen ? 'fullscreen-container' : ''}"
>
	<!-- Video Element (no native controls) -->
	<div class="relative" on:click={togglePlayPause} on:keydown={(e) => e.key === ' ' && togglePlayPause()} role="button" tabindex="-1">
		<video
			bind:this={videoElement}
			{src}
			controls={false}
			preload="metadata"
			class="w-full {isFullscreen ? 'max-h-[calc(100vh-120px)]' : 'max-h-[28rem]'} rounded-t-lg"
			style="max-width: 100%; object-fit: contain;"
			on:loadedmetadata={onLoadedMetadata}
			on:timeupdate={onTimeUpdate}
			on:play={onPlay}
			on:pause={onPause}
			playsinline
		>
			Your browser does not support video playback.
		</video>
		
		<!-- Play/Pause Overlay (shown when paused) -->
		{#if !isPlaying}
			<div class="absolute inset-0 flex items-center justify-center bg-black/20 pointer-events-none">
				<div class="bg-black/60 rounded-full p-4">
					<svg xmlns="http://www.w3.org/2000/svg" fill="white" viewBox="0 0 24 24" class="size-12">
						<path d="M8 5v14l11-7z"/>
					</svg>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Custom Controls -->
	<div class="controls-container bg-gray-100 dark:bg-gray-800 {isFullscreen ? 'rounded-none' : 'rounded-b-lg'} p-3">
		<!-- Main Controls Row -->
		<div class="flex items-center gap-3 mb-2">
			<!-- Play/Pause Button -->
			<Tooltip content={isPlaying ? "Pause" : "Play"}>
				<button
					class="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
					on:click={togglePlayPause}
				>
					{#if isPlaying}
						<svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24" class="size-5">
							<path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
						</svg>
					{:else}
						<svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24" class="size-5">
							<path d="M8 5v14l11-7z"/>
						</svg>
					{/if}
				</button>
			</Tooltip>
			
			<!-- Skip Back 10s -->
			<Tooltip content="Back 10s">
				<button
					class="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
					on:click={skipBack}
				>
					<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-4">
						<path stroke-linecap="round" stroke-linejoin="round" d="M9 15 3 9m0 0 6-6M3 9h12a6 6 0 0 1 0 12h-3" />
					</svg>
				</button>
			</Tooltip>
			
			<!-- Skip Forward 10s -->
			<Tooltip content="Forward 10s">
				<button
					class="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
					on:click={skipForward}
				>
					<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-4">
						<path stroke-linecap="round" stroke-linejoin="round" d="m15 15 6-6m0 0-6-6m6 6H9a6 6 0 0 0 0 12h3" />
					</svg>
				</button>
			</Tooltip>
			
			<!-- Time Display -->
			<span class="text-sm text-gray-600 dark:text-gray-300 font-mono min-w-[90px]">
				{formatTime(currentTime)} / {formatTime(videoDuration)}
			</span>
			
			<!-- Spacer -->
			<div class="flex-1"></div>
			
			<!-- Volume Control -->
			<div class="flex items-center gap-1">
				<Tooltip content={isMuted ? "Unmute" : "Mute"}>
					<button
						class="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
						on:click={toggleMute}
					>
						{#if isMuted || volume === 0}
							<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-4">
								<path stroke-linecap="round" stroke-linejoin="round" d="M17.25 9.75 19.5 12m0 0 2.25 2.25M19.5 12l2.25-2.25M19.5 12l-2.25 2.25m-10.5-6 4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" />
							</svg>
						{:else if volume < 0.5}
							<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-4">
								<path stroke-linecap="round" stroke-linejoin="round" d="M19.114 5.636a9 9 0 0 1 0 12.728M6.75 8.25l4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" />
							</svg>
						{:else}
							<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-4">
								<path stroke-linecap="round" stroke-linejoin="round" d="M19.114 5.636a9 9 0 0 1 0 12.728M16.463 8.288a5.25 5.25 0 0 1 0 7.424M6.75 8.25l4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z" />
							</svg>
						{/if}
					</button>
				</Tooltip>
				<input
					type="range"
					min="0"
					max="1"
					step="0.05"
					value={volume}
					on:input={onVolumeChange}
					class="w-16 h-1 accent-blue-500 cursor-pointer"
				/>
			</div>
			
			<!-- Jump to Segment Button -->
			<Tooltip content="Jump to relevant section">
				<button
					class="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors text-blue-600 dark:text-blue-400"
					on:click={jumpToSegmentStart}
				>
					<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-4">
						<path stroke-linecap="round" stroke-linejoin="round" d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
						<path stroke-linecap="round" stroke-linejoin="round" d="M15.91 11.672a.375.375 0 0 1 0 .656l-5.603 3.113a.375.375 0 0 1-.557-.328V8.887c0-.286.307-.466.557-.327l5.603 3.112Z" />
					</svg>
				</button>
			</Tooltip>
			
			<!-- Loop Toggle -->
			<Tooltip content={loopSegment ? "Disable loop" : "Loop relevant section"}>
				<button
					class="p-1.5 rounded transition-colors {loopSegment 
						? 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400' 
						: 'hover:bg-gray-200 dark:hover:bg-gray-700'}"
					on:click={toggleLoop}
				>
					<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-4">
						<path stroke-linecap="round" stroke-linejoin="round" d="M19.5 12c0-1.232-.046-2.453-.138-3.662a4.006 4.006 0 0 0-3.7-3.7 48.678 48.678 0 0 0-7.324 0 4.006 4.006 0 0 0-3.7 3.7c-.017.22-.032.441-.046.662M19.5 12l3-3m-3 3-3-3m-12 3c0 1.232.046 2.453.138 3.662a4.006 4.006 0 0 0 3.7 3.7 48.656 48.656 0 0 0 7.324 0 4.006 4.006 0 0 0 3.7-3.7c.017-.22.032-.441.046-.662M4.5 12l3 3m-3-3-3 3" />
					</svg>
				</button>
			</Tooltip>
			
			<!-- Fullscreen Button -->
			<Tooltip content={isFullscreen ? "Exit fullscreen" : "Fullscreen"}>
				<button
					class="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
					on:click={toggleFullscreen}
				>
					{#if isFullscreen}
						<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-4">
							<path stroke-linecap="round" stroke-linejoin="round" d="M9 9V4.5M9 9H4.5M9 9 3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5 5.25 5.25" />
						</svg>
					{:else}
						<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-4">
							<path stroke-linecap="round" stroke-linejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
						</svg>
					{/if}
				</button>
			</Tooltip>
		</div>
		
		<!-- Segment Info -->
		<div class="flex items-center gap-2 mb-2 text-xs">
			<span class="text-gray-500 dark:text-gray-400">Relevant section:</span>
			<span class="font-medium text-blue-600 dark:text-blue-400">
				{formatTime(segmentStart)} - {formatTime(segmentEnd)}
			</span>
			<span class="text-gray-400 dark:text-gray-500">
				({formatTime(segmentDuration)})
			</span>
			<span 
				class="ml-auto px-1.5 py-0.5 rounded text-xs {isInSegment 
					? 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 font-medium' 
					: 'text-gray-400 dark:text-gray-500'}"
			>
				{isInSegment ? '● In section' : '○ Outside'}
			</span>
		</div>
		
		<!-- Chapter Info (if chapters available) -->
		{#if hasChapters}
			<div class="flex items-center gap-2 mb-2 text-xs relative">
				<span class="text-gray-500 dark:text-gray-400">Chapter:</span>
				<button 
					class="flex items-center gap-1 px-2 py-0.5 rounded bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 hover:bg-purple-200 dark:hover:bg-purple-800/50 transition-colors"
					on:click={toggleChapterList}
				>
					<span class="font-medium truncate max-w-[200px]">
						{currentChapter?.title || 'Select chapter'}
					</span>
					<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3">
						<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
					</svg>
				</button>
				<span class="text-gray-400 dark:text-gray-500">
					{chapters.length} chapters
				</span>
				
				<!-- Chapter dropdown -->
				{#if showChapterList}
					<div class="absolute bottom-full left-0 mb-1 w-72 max-h-48 overflow-y-auto bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-10">
						{#each chapters as chapter}
							<button
								class="w-full text-left px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-start gap-2 {currentChapter?.index === chapter.index ? 'bg-purple-50 dark:bg-purple-900/30' : ''}"
								on:click={() => jumpToChapter(chapter)}
							>
								<span class="text-xs text-gray-400 dark:text-gray-500 min-w-[40px]">
									{formatTime(chapter.start)}
								</span>
								<div class="flex-1 min-w-0">
									<div class="font-medium text-sm truncate text-gray-800 dark:text-gray-200">
										{chapter.title}
									</div>
									{#if chapter.summary}
										<div class="text-xs text-gray-500 dark:text-gray-400 truncate">
											{chapter.summary}
										</div>
									{/if}
								</div>
							</button>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
		
		<!-- Visual Timeline -->
		<div 
			class="timeline relative h-2 bg-gray-300 dark:bg-gray-600 rounded-full cursor-pointer overflow-hidden"
			on:click={seekToPosition}
			on:keydown={seekToPosition}
			role="slider"
			tabindex="0"
			aria-label="Video timeline - Use arrow keys to seek, Space to play/pause"
			aria-valuemin="0"
			aria-valuemax={videoDuration}
			aria-valuenow={currentTime}
		>
			<!-- Chapter Markers (behind everything) -->
			{#if hasChapters}
				{#each chapters as chapter, idx}
					{#if idx > 0}
						<Tooltip content={chapter.title}>
							<div 
								class="absolute top-0 h-full w-0.5 bg-purple-500 dark:bg-purple-400 z-[1]"
								style="left: {videoDuration > 0 ? (chapter.start / videoDuration) * 100 : 0}%;"
							></div>
						</Tooltip>
					{/if}
				{/each}
			{/if}
			
			<!-- Highlighted Segment Region -->
			<div 
				class="absolute top-0 h-full bg-blue-200 dark:bg-blue-800 opacity-60"
				style="left: {highlightLeft}%; width: {highlightWidth}%;"
			></div>
			
			<!-- Segment Boundaries (vertical lines) -->
			<div 
				class="absolute top-0 h-full w-0.5 bg-blue-500 dark:bg-blue-400"
				style="left: {highlightLeft}%;"
			></div>
			<div 
				class="absolute top-0 h-full w-0.5 bg-blue-500 dark:bg-blue-400"
				style="left: {highlightLeft + highlightWidth}%;"
			></div>
			
			<!-- Progress Indicator -->
			<div 
				class="absolute top-0 h-full bg-blue-500 dark:bg-blue-400 transition-all duration-100"
				style="width: {videoDuration > 0 ? (currentTime / videoDuration) * 100 : 0}%;"
			></div>
			
			<!-- Current Position Marker -->
			<div 
				class="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white border-2 border-blue-500 rounded-full shadow-md transition-all duration-100"
				style="left: calc({videoDuration > 0 ? (currentTime / videoDuration) * 100 : 0}% - 6px);"
			></div>
		</div>
	</div>
</div>

<style>
	/* Timeline interaction styles */
	.timeline:hover {
		transform: scaleY(1.3);
	}
	
	.timeline:focus {
		outline: 2px solid rgb(59, 130, 246);
		outline-offset: 2px;
	}
	
	/* Fullscreen layout */
	.fullscreen-container {
		display: flex;
		flex-direction: column;
		height: 100vh;
		background: black;
	}
	
	.fullscreen-container video {
		flex: 1;
		max-height: calc(100vh - 100px);
	}
	
	.fullscreen-container .controls-container {
		flex-shrink: 0;
	}
	
	/* Volume slider - cross-browser styling */
	input[type="range"] {
		-webkit-appearance: none;
		appearance: none;
		background: transparent;
	}
	
	input[type="range"]::-webkit-slider-runnable-track {
		height: 4px;
		background: #d1d5db;
		border-radius: 2px;
	}
	
	input[type="range"]::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 12px;
		height: 12px;
		background: #3b82f6;
		border-radius: 50%;
		cursor: pointer;
		margin-top: -4px;
	}
	
	input[type="range"]::-moz-range-track {
		height: 4px;
		background: #d1d5db;
		border-radius: 2px;
	}
	
	input[type="range"]::-moz-range-thumb {
		width: 12px;
		height: 12px;
		background: #3b82f6;
		border-radius: 50%;
		cursor: pointer;
		border: none;
	}
	
	:global(.dark) input[type="range"]::-webkit-slider-runnable-track {
		background: #4b5563;
	}
	
	:global(.dark) input[type="range"]::-moz-range-track {
		background: #4b5563;
	}
</style>
