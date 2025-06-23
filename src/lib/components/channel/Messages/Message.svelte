<script lang="ts">
	import dayjs from 'dayjs';
	import relativeTime from 'dayjs/plugin/relativeTime';
	import isToday from 'dayjs/plugin/isToday';
	import isYesterday from 'dayjs/plugin/isYesterday';
	import localizedFormat from 'dayjs/plugin/localizedFormat';

	dayjs.extend(relativeTime);
	dayjs.extend(isToday);
	dayjs.extend(isYesterday);
	dayjs.extend(localizedFormat);

	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	const i18n = getContext<Writable<any>>('i18n');

	import { settings, user, shortCodesToEmojis } from '$lib/stores';

	import { WEBUI_BASE_URL } from '$lib/constants';

	import Markdown from '$lib/components/chat/Messages/Markdown.svelte';
	import ContentRenderer from '$lib/components/chat/Messages/ContentRenderer.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import GarbageBin from '$lib/components/icons/GarbageBin.svelte';
	import Pencil from '$lib/components/icons/Pencil.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import ChatBubbleOvalEllipsis from '$lib/components/icons/ChatBubbleOvalEllipsis.svelte';
	import FaceSmile from '$lib/components/icons/FaceSmile.svelte';
	import ReactionPicker from './Message/ReactionPicker.svelte';
	import ChevronRight from '$lib/components/icons/ChevronRight.svelte';
	import { formatDate } from '$lib/utils';

	export let message;
	export let showUserProfile = true;
	export let thread = false;

	export let onDelete: Function = () => {};
	export let onEdit: Function = () => {};
	export let onThread: Function = () => {};
	export let onReaction: Function = () => {};

	let showButtons = false;

	let edit = false;
	let editedContent: string | null = null;
	let showDeleteConfirmDialog = false;

	// Check if this is a bot message using metadata
	$: isBot = message?.data?.bot === true;
	$: botName = message?.data?.bot_name || 'AI Assistant';
	$: aiProvider = message?.data?.ai_provider;
	$: aiModel = message?.data?.ai_model;
	
	// Create display name with model info
	$: displayName = isBot && aiModel 
		? botName
		: botName;

	// Generate user initials for user messages
	$: userInitials = message?.user?.name ? message.user.name.split(' ').map((n: string) => n[0]).join('').toUpperCase().slice(0, 2) : 'U';
</script>

<ConfirmDialog
	bind:show={showDeleteConfirmDialog}
	title={$i18n.t('Delete Message')}
	message={$i18n.t('Are you sure you want to delete this message?')}
	onConfirm={async () => {
		await onDelete();
	}}
/>

{#if message}
	<div
		class="flex flex-col justify-between px-5 {showUserProfile
			? 'pt-1.5 pb-0.5'
			: ''} w-full max-w-5xl mx-auto group hover:bg-gray-300/5 dark:hover:bg-gray-700/5 transition relative"
	>
		{#if !edit}
			<div
				class=" absolute {showButtons ? '' : 'invisible group-hover:visible'} right-1 -top-2 z-10"
			>
				<div
					class="flex gap-1 rounded-lg bg-white dark:bg-gray-850 shadow-md p-0.5 border border-gray-100 dark:border-gray-850"
				>
					<ReactionPicker
						onClose={() => (showButtons = false)}
						onSubmit={(name) => {
							showButtons = false;
							onReaction(name);
						}}
					>
						<Tooltip content={$i18n.t('Add Reaction')}>
							<button
								class="hover:bg-gray-100 dark:hover:bg-gray-800 transition rounded-lg p-1"
								on:click={() => {
									showButtons = true;
								}}
							>
								<FaceSmile />
							</button>
						</Tooltip>
					</ReactionPicker>

					{#if !thread}
						<Tooltip content={$i18n.t('Reply in Thread')}>
							<button
								class="hover:bg-gray-100 dark:hover:bg-gray-800 transition rounded-lg p-1"
								on:click={() => {
									onThread(message.id);
								}}
							>
								<ChatBubbleOvalEllipsis />
							</button>
						</Tooltip>
					{/if}

					{#if message.user_id === $user?.id || $user?.role === 'admin'}
						<Tooltip content={$i18n.t('Edit')}>
							<button
								class="hover:bg-gray-100 dark:hover:bg-gray-800 transition rounded-lg p-1"
								on:click={() => {
									edit = true;
									editedContent = message.content;
								}}
							>
								<Pencil />
							</button>
						</Tooltip>

						<Tooltip content={$i18n.t('Delete')}>
							<button
								class="hover:bg-gray-100 dark:hover:bg-gray-800 transition rounded-lg p-1"
								on:click={() => (showDeleteConfirmDialog = true)}
							>
								<GarbageBin />
							</button>
						</Tooltip>
					{/if}
				</div>
			</div>
		{/if}

		{#if isBot}
			<!-- Bot message - LEFT ALIGNED -->
			<div class="flex items-start gap-3 mb-4">
				<!-- Bot avatar circle -->
				<div class="size-8 rounded-full flex items-center justify-center text-white font-medium text-sm shrink-0 bg-blue-500">
					🤖
				</div>
				
				<!-- Bot message content -->
				<div class="flex-1">
					<div class="font-medium text-gray-900 dark:text-gray-100 mb-1">
						{displayName}
						{#if isBot && aiModel}
							<span class="text-xs text-gray-500 dark:text-gray-400 font-normal ml-1">({aiModel})</span>
						{/if}
					</div>
					<div class="chat-assistant w-full min-w-full markdown-prose">
						<Markdown id={message.id} content={message.content} />
					</div>
				</div>
			</div>
		{:else}
			<!-- User message - RIGHT ALIGNED -->
			<div class="flex items-start gap-3 mb-4 flex-row-reverse">
				<!-- User avatar circle -->
				<div class="size-8 rounded-full flex items-center justify-center text-white font-medium text-sm shrink-0 bg-gray-500">
					{userInitials}
				</div>
				
				<!-- User message content -->
				<div class="flex-1 text-right">
					<div class="font-medium text-gray-900 dark:text-gray-100 mb-1">
						{message?.user?.name}
					</div>
					<div class="text-gray-700 dark:text-gray-300 prose prose-sm max-w-none dark:prose-invert prose-headings:font-semibold prose-headings:text-gray-900 dark:prose-headings:text-gray-100 prose-p:mb-2 prose-ul:space-y-1 prose-li:text-gray-700 dark:prose-li:text-gray-300">
						<Markdown id={message.id} content={message.content} />
					</div>
				</div>
			</div>
		{/if}

		{#if (message?.reactions ?? []).length > 0}
			<div>
				<div class="flex items-center flex-wrap gap-y-1.5 gap-1 mt-1 mb-2">
					{#each message.reactions as reaction}
						<Tooltip content={`:${reaction.name}:`}>
							<button
								class="flex items-center gap-1.5 transition rounded-xl px-2 py-1 cursor-pointer {reaction.user_ids.includes(
									$user?.id
								)
									? ' bg-blue-300/10 outline outline-blue-500/50 outline-1'
									: 'bg-gray-300/10 dark:bg-gray-500/10 hover:outline hover:outline-gray-700/30 dark:hover:outline-gray-300/30 hover:outline-1'}"
								on:click={() => {
									onReaction(reaction.name);
								}}
							>
								{#if $shortCodesToEmojis && $shortCodesToEmojis[reaction.name]}
									<img
										src="/assets/emojis/{$shortCodesToEmojis[
											reaction.name
										]?.toLowerCase()}.svg"
										alt={reaction.name}
										class=" size-4"
										loading="lazy"
									/>
								{:else}
									<div>
										{reaction.name}
									</div>
								{/if}

								{#if reaction.user_ids.length > 0}
									<div class="text-xs font-medium text-gray-500 dark:text-gray-400">
										{reaction.user_ids?.length}
									</div>
								{/if}
							</button>
						</Tooltip>
					{/each}

					<ReactionPicker
						onSubmit={(name) => {
							onReaction(name);
						}}
					>
						<Tooltip content={$i18n.t('Add Reaction')}>
							<div
								class="flex items-center gap-1.5 bg-gray-500/10 hover:outline hover:outline-gray-700/30 dark:hover:outline-gray-300/30 hover:outline-1 transition rounded-xl px-1 py-1 cursor-pointer text-gray-500 dark:text-gray-400"
							>
								<FaceSmile />
							</div>
						</Tooltip>
					</ReactionPicker>
				</div>
			</div>
		{/if}

		{#if !thread && message.reply_count > 0}
			<div class="flex items-center gap-1.5 -mt-0.5 mb-1.5">
				<button
					class="flex items-center text-xs py-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition"
					on:click={() => {
						onThread(message.id);
					}}
				>
					<span class="font-medium mr-1">
						{$i18n.t('{{COUNT}} Replies', { COUNT: message.reply_count })}</span
					><span>
						{' - '}{$i18n.t('Last reply')}
						{dayjs.unix(message.latest_reply_at / 1000000000).fromNow()}</span
					>

					<span class="ml-1">
						<ChevronRight className="size-2.5" strokeWidth="3" />
					</span>
				</button>
			</div>
		{/if}
	</div>
{/if}
