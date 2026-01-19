<script lang="ts">
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	
	export let src: string;
	export let startTime: number = 0;
	export let endTime: number | undefined = undefined;
	export let totalDuration: number | undefined = undefined;
	
	let videoElement: HTMLVideoElement;
	let currentTime = 0;
	let duration = 0;
	let isPlaying = false;
	let loopSegment = false;
	let showControls = true;
	
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
</script>

<div class="video-player-container w-full">
	<!-- Video Element -->
	<video
		bind:this={videoElement}
		{src}
		controls={showControls}
		preload="metadata"
		class="w-full max-h-[28rem] rounded-t-lg"
		style="max-width: 100%; object-fit: contain;"
		on:loadedmetadata={onLoadedMetadata}
		on:timeupdate={onTimeUpdate}
		on:play={onPlay}
		on:pause={onPause}
		playsinline
	>
		Your browser does not support video playback.
	</video>
	
	<!-- Custom Timeline with Highlight -->
	<div class="timeline-container bg-gray-100 dark:bg-gray-800 rounded-b-lg p-3">
		<!-- Segment Info Bar -->
		<div class="flex items-center justify-between mb-2 text-xs">
			<div class="flex items-center gap-2">
				<span class="text-gray-500 dark:text-gray-400">Relevant section:</span>
				<span class="font-medium text-blue-600 dark:text-blue-400">
					{formatTime(segmentStart)} - {formatTime(segmentEnd)}
				</span>
				<span class="text-gray-400 dark:text-gray-500">
					({formatTime(segmentDuration)})
				</span>
			</div>
			
			<div class="flex items-center gap-2">
				<!-- Jump to Segment Button -->
				<Tooltip content="Jump to relevant section">
					<button
						class="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
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
			</div>
		</div>
		
		<!-- Visual Timeline -->
		<div 
			class="timeline relative h-3 bg-gray-300 dark:bg-gray-600 rounded-full cursor-pointer overflow-hidden"
			on:click={seekToPosition}
			on:keydown={seekToPosition}
			role="slider"
			tabindex="0"
			aria-label="Video timeline - Use arrow keys to seek, Home to jump to relevant section start"
			aria-valuemin="0"
			aria-valuemax={videoDuration}
			aria-valuenow={currentTime}
		>
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
		
		<!-- Time Labels -->
		<div class="flex justify-between mt-1 text-xs text-gray-500 dark:text-gray-400">
			<span>{formatTime(currentTime)}</span>
			<span 
				class="px-1.5 py-0.5 rounded text-xs {isInSegment 
					? 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 font-medium' 
					: 'text-gray-400 dark:text-gray-500'}"
			>
				{isInSegment ? '● In relevant section' : '○ Outside section'}
			</span>
			<span>{formatTime(videoDuration)}</span>
		</div>
	</div>
</div>

<style>
	.timeline:hover {
		transform: scaleY(1.2);
	}
	
	.timeline:focus {
		outline: 2px solid rgb(59, 130, 246);
		outline-offset: 2px;
	}
</style>
