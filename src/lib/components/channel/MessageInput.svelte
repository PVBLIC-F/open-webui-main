<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { v4 as uuidv4 } from 'uuid';
	import { tick, getContext, onMount, onDestroy } from 'svelte';

	const i18n = getContext('i18n');

	import { config, mobile, settings, socket, user } from '$lib/stores';
	import { uploadFile } from '$lib/apis/files';
	import { updateChannelById, sendMessage } from '$lib/apis/channels';
	import { WEBUI_API_BASE_URL } from '$lib/constants';
	import { compressImage, removeLastWordFromString } from '$lib/utils';

	// Components
	import Tooltip from '../common/Tooltip.svelte';
	import RichTextInput from '../common/RichTextInput.svelte';
	import VoiceRecording from '../chat/MessageInput/VoiceRecording.svelte';
	import Commands from '../chat/MessageInput/Commands.svelte';
	import InputMenu from './MessageInput/InputMenu.svelte';
	import FileItem from '../common/FileItem.svelte';
	import Image from '../common/Image.svelte';
	import FilesOverlay from '../chat/MessageInput/FilesOverlay.svelte';

	export let placeholder = $i18n.t('Send a Message (use @ to select AI model)');
	export let id = null;
	export let channel = null;
	export let typingUsers = [];
	export let onSubmit: Function;
	export let onChange: Function;
	export let scrollEnd = true;
	export let scrollToBottom: Function = () => {};

	let draggedOver = false;
	let recording = false;
	let content = '';
	let files = [];
	let filesInputElement;
	let inputFiles;
	let commandsElement;
	let atSelectedModel = undefined;
	let aiEnabled = false;

	// Computed mobile detection to avoid duplication
	$: isDesktop = !$mobile || !('ontouchstart' in window || navigator.maxTouchPoints > 0);

	// Model selection persistence - unique per channel and user
	const getModelStorageKey = () => `channel-${id}-selected-model`;

	const saveModelSelection = (model) => {
		try {
			localStorage.setItem(getModelStorageKey(), JSON.stringify(model));
			console.log('💾 Saved model selection:', model.name);
		} catch (error) {
			console.error('Failed to save model selection:', error);
		}
	};

	const loadModelSelection = () => {
		try {
			const savedModel = localStorage.getItem(getModelStorageKey());
			if (savedModel) {
				const model = JSON.parse(savedModel);
				atSelectedModel = model;
				console.log('📂 Loaded model selection:', model.name);
				return model;
			}
		} catch (error) {
			console.error('Failed to load model selection:', error);
			localStorage.removeItem(getModelStorageKey());
		}
		return null;
	};

	// Shared file upload logic
	const inputFilesHandler = async (inputFiles) => {
		if (
			($config?.file?.max_count ?? null) !== null &&
			files.length + inputFiles.length > $config?.file?.max_count
		) {
			toast.error(
				$i18n.t(`You can only upload a maximum of {{maxCount}} file(s) at a time.`, {
					maxCount: $config?.file?.max_count
				})
			);
			return;
		}

		for (const file of inputFiles) {
			if (
				($config?.file?.max_size ?? null) !== null &&
				file.size > ($config?.file?.max_size ?? 0) * 1024 * 1024
			) {
				toast.error(
					$i18n.t(`File size should not exceed {{maxSize}} MB.`, {
						maxSize: $config?.file?.max_size
					})
				);
				continue;
			}

			if (['image/gif', 'image/webp', 'image/jpeg', 'image/png', 'image/avif'].includes(file.type)) {
				const reader = new FileReader();
				reader.onload = async (event) => {
					let imageUrl = event.target.result;

					if (
						($settings?.imageCompression ?? false) ||
						($config?.file?.image_compression?.width ?? null) ||
						($config?.file?.image_compression?.height ?? null)
					) {
						let width = $settings?.imageCompressionSize?.width ?? null;
						let height = $settings?.imageCompressionSize?.height ?? null;

						if ($config?.file?.image_compression?.width && width > $config.file.image_compression.width) {
							width = $config.file.image_compression.width;
						}
						if ($config?.file?.image_compression?.height && height > $config.file.image_compression.height) {
							height = $config.file.image_compression.height;
						}

						if (width || height) {
							imageUrl = await compressImage(imageUrl, width, height);
						}
					}

					files = [...files, { type: 'image', url: imageUrl }];
				};
				reader.readAsDataURL(file);
			} else {
				await uploadFileHandler(file);
			}
		}
	};

	const uploadFileHandler = async (file) => {
		const tempItemId = uuidv4();
		const fileItem = {
			type: 'file',
			file: '',
			id: null,
			url: '',
			name: file.name,
			collection_name: '',
			status: 'uploading',
			size: file.size,
			error: '',
			itemId: tempItemId
		};

		if (fileItem.size === 0) {
			toast.error($i18n.t('You cannot upload an empty file.'));
			return;
		}

		files = [...files, fileItem];

		try {
			let metadata = null;
			if (
				(file.type.startsWith('audio/') || file.type.startsWith('video/')) &&
				$settings?.audio?.stt?.language
			) {
				metadata = { language: $settings.audio.stt.language };
			}

			const uploadedFile = await uploadFile(localStorage.token, file, metadata);

			if (uploadedFile) {
				if (uploadedFile.error) {
					toast.warning(uploadedFile.error);
				}

				fileItem.status = 'uploaded';
				fileItem.file = uploadedFile;
				fileItem.id = uploadedFile.id;
				fileItem.collection_name = uploadedFile?.meta?.collection_name || uploadedFile?.collection_name;
				fileItem.url = `${WEBUI_API_BASE_URL}/files/${uploadedFile.id}`;
				files = files;
			} else {
				files = files.filter((item) => item?.itemId !== tempItemId);
			}
		} catch (e) {
			toast.error(`${e}`);
			files = files.filter((item) => item?.itemId !== tempItemId);
		}
	};

	// Event handlers
	const handleKeyDown = (event) => {
		if (event.key === 'Escape') {
			draggedOver = false;
		}
	};

	const handleClickOutside = (event) => {
		const commandsContainer = document.getElementById('commands-container');
		const chatInput = document.getElementById(`chat-input-${id}`);
		
		if (commandsContainer && !commandsContainer.contains(event.target) && !chatInput.contains(event.target)) {
			// Get the current command to remove
			const currentCommand = content?.split('\n').pop()?.split(' ')?.pop() ?? '';
			if (currentCommand && ['@', '#', '/'].includes(currentCommand.charAt(0))) {
				content = removeLastWordFromString(content, currentCommand);
			}
		}
	};

	const onDragOver = (e) => {
		e.preventDefault();
		draggedOver = e.dataTransfer?.types?.includes('Files');
	};

	const onDragLeave = () => {
		draggedOver = false;
	};

	const onDrop = async (e) => {
		e.preventDefault();
		if (e.dataTransfer?.files) {
			const droppedFiles = Array.from(e.dataTransfer.files);
			if (droppedFiles.length > 0) {
				await inputFilesHandler(droppedFiles);
			}
		}
		draggedOver = false;
	};

	const submitHandler = async () => {
		if (content === '' && files.length === 0) return;

		const messageData = { files };
		if (atSelectedModel && aiEnabled) {
			messageData.atSelectedModel = atSelectedModel;
		}

		onSubmit({ content, data: messageData });

		content = '';
		files = [];
		// Keep atSelectedModel persistent - don't reset it!
		// atSelectedModel = undefined;

		await tick();
		document.getElementById(`chat-input-${id}`)?.focus();
	};

	$: if (content) onChange();

	// Load saved model selection when component mounts
	onMount(async () => {
		// Load the saved model selection
		loadModelSelection();

		setTimeout(() => {
			document.getElementById(`chat-input-${id}`)?.focus();
		}, 0);

		window.addEventListener('keydown', handleKeyDown);
		document.addEventListener('click', handleClickOutside);

		const dropzoneElement = document.getElementById('channel-container');
		if (dropzoneElement) {
			dropzoneElement.addEventListener('dragover', onDragOver);
			dropzoneElement.addEventListener('drop', onDrop);
			dropzoneElement.addEventListener('dragleave', onDragLeave);
		}
	});

	onDestroy(() => {
		window.removeEventListener('keydown', handleKeyDown);
		document.removeEventListener('click', handleClickOutside);
		const dropzoneElement = document.getElementById('channel-container');
		if (dropzoneElement) {
			dropzoneElement.removeEventListener('dragover', onDragOver);
			dropzoneElement.removeEventListener('drop', onDrop);
			dropzoneElement.removeEventListener('dragleave', onDragLeave);
		}
	});
</script>

<FilesOverlay show={draggedOver} />

<input
	bind:this={filesInputElement}
	bind:files={inputFiles}
	type="file"
	hidden
	multiple
	on:change={async () => {
		if (inputFiles?.length > 0) {
			await inputFilesHandler(Array.from(inputFiles));
		} else {
			toast.error($i18n.t('File not found.'));
		}
		filesInputElement.value = '';
	}}
/>

<div class="bg-transparent">
	<div class="{($settings?.widescreenMode ?? null) ? 'max-w-full' : 'max-w-6xl'} px-2.5 mx-auto inset-x-0 relative">
		<div class="absolute top-0 left-0 right-0 mx-auto inset-x-0 bg-transparent flex justify-center">
			<div class="flex flex-col px-3 w-full">
				<div class="relative">
					{#if scrollEnd === false}
						<div class="absolute -top-12 left-0 right-0 flex justify-center z-30 pointer-events-none">
							<button
								class="bg-white border border-gray-100 dark:border-none dark:bg-white/20 p-1.5 rounded-full pointer-events-auto"
								on:click={() => {
									scrollEnd = true;
									scrollToBottom();
								}}
							>
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 20 20"
									fill="currentColor"
									class="w-5 h-5"
								>
									<path
										fill-rule="evenodd"
										d="M10 3a.75.75 0 01.75.75v10.638l3.96-4.158a.75.75 0 111.08 1.04l-5.25 5.5a.75.75 0 01-1.08 0l-5.25-5.5a.75.75 0 111.08-1.04l3.96 4.158V3.75A.75.75 0 0110 3z"
										clip-rule="evenodd"
									/>
								</svg>
							</button>
						</div>
					{/if}
				</div>

				<div class="relative">
					<div class="-mt-5">
						{#if typingUsers.length > 0}
							<div class="text-xs px-4 mb-1">
								<span class="font-normal text-black dark:text-white">
									{typingUsers.map((user) => user.name).join(', ')}
								</span>
								{$i18n.t('is typing...')}
							</div>
						{/if}
					</div>
				</div>
			</div>
		</div>

		<div class="">
			{#if recording}
				<VoiceRecording
					bind:recording
					onCancel={async () => {
						recording = false;
						await tick();
						document.getElementById(`chat-input-${id}`)?.focus();
					}}
					onConfirm={async (data) => {
						const { text } = data;
						content = `${content}${text} `;
						recording = false;
						await tick();
						document.getElementById(`chat-input-${id}`)?.focus();
					}}
				/>
			{:else}
				<form class="w-full flex gap-1.5" on:submit|preventDefault={submitHandler}>
					<div
						class="flex-1 flex flex-col relative w-full rounded-3xl px-1 bg-gray-600/5 dark:bg-gray-400/5 dark:text-gray-100"
						dir={$settings?.chatDirection ?? 'auto'}
					>
						{#if files.length > 0}
							<div class="mx-2 mt-2.5 -mb-1 flex flex-wrap gap-2">
								{#each files as file, fileIdx}
									{#if file.type === 'image'}
										<div class="relative group">
											<div class="relative">
												<Image
													src={file.url}
													alt="input"
													imageClassName="h-16 w-16 rounded-xl object-cover"
												/>
											</div>
											<div class="absolute -top-1 -right-1">
												<button
													class="bg-white text-black border border-white rounded-full group-hover:visible invisible transition"
													type="button"
													on:click={() => {
														files.splice(fileIdx, 1);
														files = files;
													}}
												>
													<svg
														xmlns="http://www.w3.org/2000/svg"
														viewBox="0 0 20 20"
														fill="currentColor"
														class="w-4 h-4"
													>
														<path
															d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z"
														/>
													</svg>
												</button>
											</div>
										</div>
									{:else}
										<FileItem
											item={file}
											name={file.name}
											type={file.type}
											size={file?.size}
											loading={file.status === 'uploading'}
											dismissible={true}
											edit={true}
											on:dismiss={() => {
												files.splice(fileIdx, 1);
												files = files;
											}}
										/>
									{/if}
								{/each}
							</div>
						{/if}

						<div class="px-2.5">
							<div class="scrollbar-hidden font-primary text-left bg-transparent dark:text-gray-100 outline-hidden w-full pt-3 px-1 rounded-xl resize-none h-fit max-h-80 overflow-auto">
								<!-- Simple model indicator with AI toggle -->
								{#if atSelectedModel}
									<div class="text-xs text-gray-500 dark:text-gray-400 mb-2 px-1 flex items-center gap-2">
										<input
											type="checkbox"
											bind:checked={aiEnabled}
											class="w-3 h-3"
										/>
										<span>AI: {atSelectedModel.name}</span>
									</div>
								{/if}
								
								<RichTextInput
									bind:value={content}
									id={`chat-input-${id}`}
									messageInput={true}
									shiftEnter={isDesktop}
									{placeholder}
									largeTextAsFile={$settings?.largeTextAsFile ?? false}
									on:keydown={async (e) => {
										e = e.detail.event;
										const commandsContainerElement = document.getElementById('commands-container');

										if (commandsContainerElement) {
											if (e.key === 'ArrowUp') {
												e.preventDefault();
												commandsElement.selectUp();
												return;
											}
											if (e.key === 'ArrowDown') {
												e.preventDefault();
												commandsElement.selectDown();
												return;
											}
											if (e.key === 'Tab') {
												e.preventDefault();
												const commandOptionButton = document.querySelector('.selected-command-option-button');
												commandOptionButton?.click();
												return;
											}
											if (e.key === 'Enter') {
												e.preventDefault();
												const commandOptionButton = document.querySelector('.selected-command-option-button');
												if (commandOptionButton) {
													commandOptionButton.click();
												} else {
													submitHandler();
												}
												return;
											}
											if (e.key === 'Escape') {
												e.preventDefault();
												// Get the current command to remove
												const currentCommand = content?.split('\n').pop()?.split(' ')?.pop() ?? '';
												if (currentCommand && ['@', '#', '/'].includes(currentCommand.charAt(0))) {
													content = removeLastWordFromString(content, currentCommand);
												}
												return;
											}
										}

										if (isDesktop) {
											if (e.keyCode === 13 && !e.shiftKey) {
												e.preventDefault();
											}
											if (content !== '' && e.keyCode === 13 && !e.shiftKey) {
												submitHandler();
											}
										}
									}}
								/>
							</div>
						</div>

						<div class="flex justify-between mb-2.5 mt-1.5 mx-0.5">
							<div class="ml-1 self-end flex space-x-1">
								<InputMenu uploadFilesHandler={() => filesInputElement.click()} />
							</div>

							<div class="self-end flex space-x-1 mr-1">
								{#if content === ''}
									<Tooltip content={$i18n.t('Record voice')}>
										<button
											class="text-gray-600 dark:text-gray-300 hover:text-gray-700 dark:hover:text-gray-200 transition rounded-full p-1.5 mr-0.5 self-center"
											type="button"
											on:click={async () => {
												try {
													const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
													if (stream) {
														recording = true;
														stream.getTracks().forEach((track) => track.stop());
													}
												} catch {
													toast.error($i18n.t('Permission denied when accessing microphone'));
												}
											}}
										>
											<svg
												xmlns="http://www.w3.org/2000/svg"
												viewBox="0 0 20 20"
												fill="currentColor"
												class="w-5 h-5 translate-y-[0.5px]"
											>
												<path d="M7 4a3 3 0 016 0v6a3 3 0 11-6 0V4z" />
												<path
													d="M5.5 9.643a.75.75 0 00-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-1.5v-1.546A6.001 6.001 0 0016 10v-.357a.75.75 0 00-1.5 0V10a4.5 4.5 0 01-9 0v-.357z"
												/>
											</svg>
										</button>
									</Tooltip>
								{/if}

								<div class="flex items-center">
										<Tooltip content={$i18n.t('Send message')}>
											<button
												class="{content !== '' || files.length !== 0
												? 'bg-black text-white hover:bg-gray-900 dark:bg-white dark:text-black dark:hover:bg-gray-100'
													: 'text-white bg-gray-200 dark:text-gray-900 dark:bg-gray-700 disabled'} transition rounded-full p-1.5 self-center"
												type="submit"
												disabled={content === '' && files.length === 0}
											>
												<svg
													xmlns="http://www.w3.org/2000/svg"
													viewBox="0 0 16 16"
													fill="currentColor"
													class="size-5"
												>
													<path
														fill-rule="evenodd"
														d="M8 14a.75.75 0 0 1-.75-.75V4.56L4.03 7.78a.75.75 0 0 1-1.06-1.06l4.5-4.5a.75.75 0 0 1 1.06 0l4.5 4.5a.75.75 0 0 1-1.06 1.06L8.75 4.56v8.69A.75.75 0 0 1 8 14Z"
														clip-rule="evenodd"
													/>
												</svg>
											</button>
										</Tooltip>
								</div>
							</div>
						</div>
					</div>
				</form>
			{/if}

			<Commands
				bind:this={commandsElement}
				bind:prompt={content}
				bind:files
				on:upload={(e) => {
					console.log('Upload event:', e.detail);
				}}
				on:select={(e) => {
					const data = e.detail;
					if (data?.type === 'model') {
						atSelectedModel = data.data;
						// Save the model selection to localStorage for persistence
						saveModelSelection(data.data);
					}
					document.getElementById(`chat-input-${id}`)?.focus();
				}}
			/>
		</div>
	</div>
</div>
