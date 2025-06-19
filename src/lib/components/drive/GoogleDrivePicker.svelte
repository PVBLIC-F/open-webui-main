<!--
GoogleDrivePicker Component

A sophisticated folder selection interface for Google Drive cloud sync integration.
This component provides hierarchical folder browsing with multi-selection capabilities,
designed specifically for knowledge base synchronization workflows.

Key Features:
- Hierarchical folder tree navigation with expand/collapse functionality
- Multi-folder selection with visual feedback and selection state management
- Search functionality for quick folder discovery
- Breadcrumb navigation for easy traversal
- Pre-selection support for existing sync configurations
- Real-time selection count and change detection
- Responsive design with dark mode support

Architecture:
- Event-driven communication with parent components
- Reactive state management for folder expansion and selection
- Efficient loading with on-demand subfolder content fetching
- Clean separation of concerns between UI and data management

Usage:
Used in knowledge base settings to allow users to select Google Drive folders
for automatic synchronization. Integrates with cloud sync backend APIs.
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { listDriveItems, type DriveItem } from '$lib/utils/google-drive-api';
  import { createEventDispatcher } from 'svelte';
  import { GOOGLE_DRIVE_FOLDER_MIME_TYPE } from '$lib/constants';

  const dispatch = createEventDispatcher();

  // Component Props: Configure picker behavior and initial state
  export let token: string = '';
  export let initialFolderId: string = 'root';
  export let preSelectedFolderIds: string[] = []; // Folders that should be pre-selected

  // Core State: Manage folder data and UI state
  let items: DriveItem[] = [];
  let filteredItems: DriveItem[] = [];
  let loading = false;
  let error: string | null = null;
  let path: { id: string; name: string }[] = [];
  let searchQuery = '';
  
  // Selection State: Track folder expansion and selection for multi-folder picking
  // Always folders-only for knowledge base sync
  let expandedFolders = new Set<string>();
  let selectedFolders = new Set<string>();
  let folderContents = new Map<string, DriveItem[]>();

  // Change Detection: Compare current selection with initial state
  // This enables smart UI updates and prevents unnecessary API calls
  function hasChanges() {
    const currentSelection = new Set(selectedFolders);
    const originalSelection = new Set(preSelectedFolderIds);
    
    // Compare sets - return true if they're different
    const hasChanges = currentSelection.size !== originalSelection.size || 
           ![...currentSelection].every(id => originalSelection.has(id));
    
    return hasChanges;
  }

  // Event Communication: Notify parent component of selection state changes
  // Provides selection count, labels, and action callbacks for parent UI
  function emitSelectionState() {
    const count = selectedFolders.size;
    const changesDetected = hasChanges();
    
    const selectionInfo = {
      count,
      label: `${count} folder${count === 1 ? '' : 's'} selected`,
      clearFn: deselectAllFolders,
      confirmFn: confirmFolderSelection,
      confirmLabel: 'Save'
    };
    
    dispatch('selection-state', {
      hasSelection: changesDetected, // Only show buttons when there are changes
      selectionInfo: selectionInfo // Always send selection info for count display
    });
  }

  // Reactive Filtering: Auto-filter folders based on search query
  // Simplified filtering logic - folders only for cloud sync use case
  $: {
    let baseItems = items.filter(item => item.mimeType === GOOGLE_DRIVE_FOLDER_MIME_TYPE);
    
    // Apply search filtering
    filteredItems = searchQuery.trim() 
      ? baseItems.filter(item => item.name.toLowerCase().includes(searchQuery.toLowerCase()))
      : baseItems;
  }

  // Folder Loading: Fetch folder contents from Google Drive API
  // Handles breadcrumb updates and pre-selection initialization
  async function load(folderId: string, folderName = 'My Drive') {
    loading = true;
    error = null;
    searchQuery = '';
    
    try {
      const rawItems = await listDriveItems(token, folderId);
      items = rawItems.sort((a, b) => a.name.localeCompare(b.name));
      
      // Update breadcrumb path
      if (path.length === 0 || path[path.length - 1].id !== folderId) {
        path = [...path, { id: folderId, name: folderName }];
      }
      
      // Optimization: Pre-load subfolder contents for badge display
      // If we have preselected folders, proactively load folder contents to show badges
      if (preSelectedFolderIds.length > 0) {
        const folderItems = rawItems.filter(item => item.mimeType === GOOGLE_DRIVE_FOLDER_MIME_TYPE);
        // Load contents for all folders to check for selected subfolders
        folderItems.forEach(folder => loadFolderContents(folder.id));
      }
      
      // Emit selection state after items are loaded
      emitSelectionState();
    } catch (e: any) {
      error = e.message || 'Failed to load items';
    } finally {
      loading = false;
    }
  }

  // Breadcrumb Navigation: Handle breadcrumb clicks for quick navigation
  function goToBreadcrumb(idx: number) {
    const crumb = path[idx];
    path = path.slice(0, idx + 1);
    load(crumb.id, crumb.name);
  }

  // Content Loading: Shared function to load folder contents with caching
  // Used for both expansion and badge checking to avoid duplicate API calls
  async function loadFolderContents(folderId: string): Promise<DriveItem[]> {
    try {
      const contents = await listDriveItems(token, folderId);
      const filteredContents = contents
        .filter(item => item.mimeType === GOOGLE_DRIVE_FOLDER_MIME_TYPE)
        .sort((a, b) => a.name.localeCompare(b.name));
      
      folderContents.set(folderId, filteredContents);
      folderContents = new Map(folderContents); // Trigger reactivity
      return filteredContents;
    } catch (e: any) {
      console.error('Failed to load folder contents:', e);
      return [];
    }
  }

  // Tree Expansion: Toggle folder expansion with lazy loading
  // Folder expansion functions (only for full mode)
  async function toggleFolderExpansion(folder: DriveItem) {
    if (!expandedFolders || !folderContents) return;
    
    if (expandedFolders.has(folder.id)) {
      expandedFolders.delete(folder.id);
      expandedFolders = new Set(expandedFolders);
    } else {
      await loadFolderContents(folder.id);
      expandedFolders.add(folder.id);
      expandedFolders = new Set(expandedFolders);
    }
  }

  // Selection Management: Handle folder selection state changes
  // Folder selection functions
  function toggleFolderSelection(folder: DriveItem) {
    if (selectedFolders.has(folder.id)) {
      selectedFolders.delete(folder.id);
    } else {
      selectedFolders.add(folder.id);
    }
    selectedFolders = new Set(selectedFolders);
    emitSelectionState();
  }

  // Badge Display: Count selected subfolders for visual feedback
  // Helper function to count selected subfolders for a parent folder
  function getSelectedSubfolderCount(parentFolderId: string): number {
    const subfolders = folderContents.get(parentFolderId) || [];
    const selectedCount = subfolders.filter(subfolder => selectedFolders.has(subfolder.id)).length;
    
    // Lazy Loading Optimization: Load contents on-demand for badge display
    // If no contents loaded yet and we have preselected folders, trigger background loading
    if (!folderContents.has(parentFolderId) && preSelectedFolderIds.length > 0) {
      loadFolderContents(parentFolderId);
    }
    
    return selectedCount;
  }

  // Selection Actions: Provide user actions for selection management
  function deselectAllFolders() {
    selectedFolders.clear();
    selectedFolders = new Set(selectedFolders);
    emitSelectionState();
  }

  function cancelSelection() {
    // Restore to original preSelected state
    selectedFolders = new Set(preSelectedFolderIds);
    emitSelectionState();
  }

  function confirmFolderSelection() {
    const selectedFolderItems = getAllVisibleFolders().filter(folder => selectedFolders.has(folder.id));
    dispatch('select', { type: 'multiple-folders', folders: selectedFolderItems });
  }

  // Folder Collection: Gather all visible folders for selection confirmation
  function getAllVisibleFolders(): DriveItem[] {
    const folders: DriveItem[] = [];
    const rootFolders = items.filter(item => item.mimeType === GOOGLE_DRIVE_FOLDER_MIME_TYPE);
    folders.push(...rootFolders);
    
    // Recursively collect expanded subfolder contents
    function addExpandedContents(folderId: string) {
      const contents = folderContents!.get(folderId);
      if (contents && expandedFolders!.has(folderId)) {
        const subFolders = contents.filter(item => item.mimeType === GOOGLE_DRIVE_FOLDER_MIME_TYPE);
        folders.push(...subFolders);
        subFolders.forEach(subFolder => addExpandedContents(subFolder.id));
      }
    }
    
    rootFolders.forEach(folder => addExpandedContents(folder.id));
    return folders;
  }

  // Reactive Initialization: Set up pre-selected folders and emit initial state
  // Initialize pre-selected folders
  $: {
    selectedFolders = new Set(preSelectedFolderIds);
    emitSelectionState();
  }

  // Component Lifecycle: Initialize the picker with root folder
  onMount(() => {
    load(initialFolderId);
  });
</script>

<!-- UI Layout: Responsive folder picker interface with header, search, and content areas -->
<div class="h-[400px] flex flex-col">
<!-- Header Section: Breadcrumbs and search functionality -->
<div class="border-b border-gray-200 dark:border-gray-700 pb-3 mb-4 px-5">
  <!-- Breadcrumbs and Selection Summary Row -->
  <div class="flex items-center justify-between mb-3">
    <!-- Breadcrumb Navigation: Show current path with clickable navigation -->
    <div class="flex items-center space-x-1 text-sm text-gray-600 dark:text-gray-400">
      {#each path.slice(1) as crumb, idx}
        <button 
          class="hover:text-gray-900 dark:hover:text-gray-200 transition-colors duration-200 font-medium"
          on:click={() => goToBreadcrumb(idx + 1)}
        >
          {crumb.name}
        </button>
        {#if idx < path.length - 2}
          <span class="text-gray-400">/</span>
        {/if}
      {/each}
    </div>
  </div>

  <!-- Search Input: Filter folders by name -->
  <div class="flex-1 max-w-xs">
    <div class="relative">
      <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <svg class="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
        </svg>
      </div>
      <input
        type="text"
        bind:value={searchQuery}
        placeholder="Search..."
        class="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
      />
    </div>
  </div>
</div>

<!-- Content Area: Main folder display with loading, error, and empty states -->
<div class="flex-1 overflow-y-auto px-5 pb-4">
  {#if loading}
    <!-- Loading State: Spinner with loading message -->
    <div class="flex items-center justify-center py-16">
      <div class="flex items-center space-x-3 text-gray-500 dark:text-gray-400">
        <svg class="animate-spin h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <span class="text-sm font-medium">Loading...</span>
      </div>
    </div>
  {:else if error}
    <!-- Error State: Display error message with icon -->
    <div class="flex items-center justify-center py-16">
      <div class="text-center">
        <div class="text-red-500 dark:text-red-400 mb-3">
          <svg class="w-10 h-10 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
        </div>
        <p class="text-red-600 dark:text-red-400 font-medium">{error}</p>
      </div>
    </div>
      {:else}
      <div class="space-y-0">
        <!-- Folder Tree View: Hierarchical folder display with selection -->
        {#each filteredItems as folder}
          {@const isExpanded = expandedFolders?.has(folder.id) ?? false}
          {@const isSelected = selectedFolders?.has(folder.id) ?? false}
          {@const selectedSubCount = getSelectedSubfolderCount(folder.id)}
          
          <div class="folder-item">
            <!-- Root folder: Main folder item with expand/collapse and selection -->
            <div class="flex items-center space-x-2 p-1 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors duration-200 border border-transparent hover:border-gray-200 dark:hover:border-gray-700">
            <!-- Expand/Collapse button: Tree navigation control -->
            <button
              on:click={() => toggleFolderExpansion(folder)}
              class="flex-shrink-0 w-6 h-6 flex items-center justify-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors rounded"
            >
              {#if isExpanded}
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                </svg>
              {:else}
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                </svg>
              {/if}
            </button>

            <!-- Selection Checkbox: Multi-select functionality -->
            <input
              type="checkbox"
              checked={isSelected}
              on:change={() => toggleFolderSelection(folder)}
              class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
            />

            <!-- Folder Display: Name and selection count badge -->
            <div class="flex items-center space-x-2 flex-1 min-w-0">
              <span class="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                {folder.name}
              </span>
              <!-- Selection Badge: Show count of selected subfolders -->
              {#if selectedSubCount > 0}
                <span class="inline-flex items-center justify-center px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full min-w-[1.25rem] h-5">
                  {selectedSubCount}
                </span>
              {/if}
            </div>
          </div>

          <!-- Subfolder Display: Expanded subfolder contents with indentation -->
          <!-- Subfolders (if expanded) -->
            {#if isExpanded && folderContents?.has(folder.id)}
              <div class="ml-9 mt-0 space-y-0 border-l border-gray-200 dark:border-gray-700 pl-4">
                {#each (folderContents.get(folder.id) || []) as subfolder}
                  {@const isSubSelected = selectedFolders?.has(subfolder.id) ?? false}
                  
                  <!-- Subfolder Item: Nested folder with selection -->
                  <div class="flex items-center space-x-2 py-0.5 px-1 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors duration-200">
                  <!-- Spacer for alignment with parent folder controls -->
                  <div class="w-6 h-6"></div>

                  <!-- Subfolder Checkbox: Selection control -->
                  <input
                    type="checkbox"
                    checked={isSubSelected}
                    on:change={() => toggleFolderSelection(subfolder)}
                    class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                  />

                  <!-- Subfolder Name: Display with truncation -->
                  <div class="flex items-center space-x-3 flex-1 min-w-0">
                    <span class="text-sm text-gray-900 dark:text-gray-100 truncate">
                      {subfolder.name}
                    </span>
                  </div>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/each}

      <!-- Empty State: No folders found message -->
      {#if filteredItems.length === 0 && !loading}
        <div class="flex items-center justify-center py-16">
          <div class="text-center">
            <div class="text-gray-400 dark:text-gray-500 mb-3">
              {#if searchQuery.trim() !== ''}
                <!-- Search Empty State -->
                <svg class="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                </svg>
              {:else}
                <!-- General Empty State -->
                <svg class="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"></path>
                </svg>
              {/if}
            </div>
            <p class="text-gray-500 dark:text-gray-400 text-sm font-medium">
              {#if searchQuery.trim() !== ''}
                No folders found matching "{searchQuery}"
              {:else}
                No folders in this location
              {/if}
            </p>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>
</div>

<style>
/* Styling: Using Tailwind CSS classes for consistent design system integration */
/* No custom styles needed - using Tailwind CSS classes */
</style> 