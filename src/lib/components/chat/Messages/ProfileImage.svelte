<script lang="ts">
	import { WEBUI_BASE_URL } from '$lib/constants';

	export let className = 'size-8';
	export let src = `${WEBUI_BASE_URL}/static/favicon.png`;
	export let modelId: string | undefined = undefined;
	export let modelName: string | undefined = undefined;

	// Handle image load errors - fallback to default favicon
	const handleImageError = (event) => {
		const img = event.target;
		if (img.src !== `${WEBUI_BASE_URL}/static/favicon.png`) {
			img.src = `${WEBUI_BASE_URL}/static/favicon.png`;
		}
	};
</script>

<img
	crossorigin="anonymous"
	src={src === ''
		? `${WEBUI_BASE_URL}/static/favicon.png`
		: src.startsWith(WEBUI_BASE_URL) ||
			  src.startsWith('https://www.gravatar.com/avatar/') ||
			  src.startsWith('data:') ||
			  src.startsWith('/')
			? src
			: `/user.png`}
	class="{className} object-cover rounded-full"
	alt="profile"
	draggable="false"
	on:error={handleImageError}
/>
