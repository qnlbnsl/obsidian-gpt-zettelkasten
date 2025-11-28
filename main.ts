import {
  App,
  MarkdownView,
  Modal,
  Plugin,
  PluginSettingTab,
  Setting,
  TFile,
  TFolder,
  WorkspaceLeaf,
  Events,
  EventRef,
  Notice,
  DropdownComponent,
} from 'obsidian';
import {
  OpenAIClient,
  unlabelledEmbeddingModel,
  availableEmbeddingModels,
  AnthropicClient,
  availableChatModels,
  CLAUDE_3_5_HAIKU,
  ChatModel,
  EmbeddingModel,
  OPENAI_PROVIDER,
  CustomChatModel,
  CustomEmbeddingModel,
} from './src/llm_client';
import { generateAndStoreEmbeddings, FileFilter } from './src/semantic_search';
import { VectorStore, StoredVector } from './src/vector_storage';
import SemanticSearchModal from './src/semantic_search_modal';
import CopilotTab from './src/zettelkasten_ai_tab';
import BatchVectorStorageModal from './src/batch_vector_storage_modal';
import { VIEW_TYPE_AI_COPILOT, VIEW_TYPE_AI_SEARCH } from './src/constants';
import SemanticSearchTab from 'src/semantic_search_tab';
import { DEFAULT_NOTE_GROUPS, NoteGroup, filesInGroupFolder } from 'src/note_group';
import EmbeddingsOverwriteConfirmModal from 'src/embeddings_overwrite_confirm_modal';

const IDLE_STATUS = 'idle';
const INDEXING_STATUS = 'indexing';

interface CustomProvider {
  id: string;
  name: string;
  apiKey: string;
  baseUrl: string;
  enabled: boolean;
}

interface ProviderModels {
  chat: ChatModel[];
  embedding: EmbeddingModel[];
}

interface ZettelkastenLLMToolsPluginSettings {
  openaiAPIKey: string;
  openaiBaseUrl: string;
  anthropicAPIKey: string;
  gcpAPIKey: string;
  vectors: Array<StoredVector>;
  noteGroups: Array<NoteGroup>;
  embeddingsModelVersion?: string;
  embeddingsModelProviderId?: string;
  embeddingsEnabled: boolean;
  indexedNoteGroup: number;
  copilotModel: string;
  copilotModelProviderId?: string;
  customChatModels: Array<CustomChatModel>;
  customEmbeddingModels: Array<CustomEmbeddingModel>;
  customProviders: Array<CustomProvider>;
  providerModels: Record<string, ProviderModels>;
};

// TODO: when removing a note group, remove the vectors associated with it

const DEFAULT_SETTINGS: ZettelkastenLLMToolsPluginSettings = {
  openaiAPIKey: '',
  openaiBaseUrl: '',
  anthropicAPIKey: '',
  gcpAPIKey: '',
  vectors: [],
  noteGroups: DEFAULT_NOTE_GROUPS.map(grp => ({ ...grp })), // deep copy
  embeddingsEnabled: false,
  indexedNoteGroup: 0,
  copilotModel: CLAUDE_3_5_HAIKU,
  customChatModels: [],
  customEmbeddingModels: [],
  customProviders: [],
  providerModels: {},
};

// Helper functions for model value format (providerId:modelName)
function formatModelValue(providerId: string, modelName: string): string {
  return `${providerId}:${modelName}`;
}

function parseModelValue(value: string): { providerId: string; modelName: string } {
  const colonIndex = value.indexOf(':');
  if (colonIndex === -1) {
    // Old format - just model name, default to openai
    return { providerId: 'openai', modelName: value };
  }
  return {
    providerId: value.substring(0, colonIndex),
    modelName: value.substring(colonIndex + 1)
  };
}

function migrateModelValue<T extends { name: string; providerId?: string }>(
  oldValue: string,
  availableModels: T[],
  currentProviderId?: string
): string {
  // If already in new format, return as-is
  if (oldValue.includes(':')) {
    return oldValue;
  }
  
  // Find matching models by name
  const matchingModels = availableModels.filter(m => m.name === oldValue);
  
  if (matchingModels.length === 0) {
    // No match found, default to openai
    return formatModelValue('openai', oldValue);
  }
  
  if (matchingModels.length === 1) {
    // Single match, use its providerId
    return formatModelValue(matchingModels[0].providerId || 'openai', oldValue);
  }
  
  // Multiple matches - prefer current providerId if set
  if (currentProviderId) {
    const preferredMatch = matchingModels.find(m => m.providerId === currentProviderId);
    if (preferredMatch) {
      return formatModelValue(preferredMatch.providerId || 'openai', oldValue);
    }
  }
  
  // Default to first match
  return formatModelValue(matchingModels[0].providerId || 'openai', oldValue);
}

function detectDuplicateModelNames<T extends { name: string; providerId?: string; displayName: string }>(
  models: T[]
): Map<string, T[]> {
  const nameToModels = new Map<string, T[]>();
  
  for (const model of models) {
    const existing = nameToModels.get(model.name) || [];
    existing.push(model);
    nameToModels.set(model.name, existing);
  }
  
  // Filter to only duplicates
  const duplicates = new Map<string, T[]>();
  for (const [name, modelList] of nameToModels) {
    if (modelList.length > 1) {
      duplicates.set(name, modelList);
    }
  }
  
  return duplicates;
}

export default class ZettelkastenLLMToolsPlugin extends Plugin {
  app: App & {
    workspace: WorkspaceWithCustomEvents;
  };
  settings: ZettelkastenLLMToolsPluginSettings;
  vectorStore: VectorStore;
  fileFilter: FileFilter;
  copilotTab: CopilotTab;
  semanticSearchTab: SemanticSearchTab;
  openaiClient: OpenAIClient;
  anthropicClient: AnthropicClient;
  indexingStatus: typeof IDLE_STATUS | typeof INDEXING_STATUS;
  lastIndexedCount: number;
  private events: Events;
  private clientCache: Map<string, OpenAIClient> = new Map();

  async onload() {
    this.events = new Events();
    this.fileFilter = new FileFilter();
    await this.loadSettings();
    this.vectorStore = new VectorStore(this);
    this.indexingStatus = IDLE_STATUS;
    this.lastIndexedCount = this.settings.vectors.length;
    // this.indexVectorStores();

    // Generate embeddings for current note command
    this.addCommand({
      id: 'generate-embeddings-current-note',
      name: 'Generate embeddings for current note',
      callback: async () => {
        const activeView = this.app.workspace.getActiveViewOfType(MarkdownView);
        if (activeView) {
          const activeFile = this.app.workspace.getActiveFile();
          if (!activeFile) { return; }

          try {
            this.indexingStatus = INDEXING_STATUS;
            await this.saveSettings();
            
            // Get the client for the configured embedding model
            const providerId = this.settings.embeddingsModelProviderId;
            const client = this.getOpenAIClient(providerId);

            const concurrencyManager = await generateAndStoreEmbeddings({
              vectorStore: this.vectorStore,
              files: [activeFile],
              app: this.app,
              openaiClient: client,
              notify: (numCompleted: number) => {
                this.lastIndexedCount = numCompleted;
              }
            });
            await concurrencyManager.done();
          } finally {
            this.indexingStatus = IDLE_STATUS;
            await this.saveSettings();
          }
        }
      }
    });

    // Semantic search command
    this.addCommand({
      id: 'open-semantic-search-modal',
      name: 'Semantic Search for notes similar to current note',
      callback: () => {
        new SemanticSearchModal(this.app, this).open();
      }
    });

    this.registerView(VIEW_TYPE_AI_COPILOT, (leaf: WorkspaceLeaf) => {
      this.copilotTab = new CopilotTab(leaf, this);
      return this.copilotTab;
    });

    this.registerView(VIEW_TYPE_AI_SEARCH, (leaf: WorkspaceLeaf) => {
      this.semanticSearchTab = new SemanticSearchTab(leaf, this);
      return this.semanticSearchTab;
    });

    let batchModel: BatchVectorStorageModal | undefined;
    this.addCommand({
      id: 'open-batch-generate-embeddings-modal',
      name: 'Open batch generate embeddings modal',
      callback: () => {
        if (batchModel === undefined) {
          batchModel = new BatchVectorStorageModal(this.app, this, this.settings.noteGroups);
          batchModel.open();
        }
      }
    });

    this.addSettingTab(new ZettelkastenLLMToolsPluginSettingTab(this.app, this));

    this.app.workspace.onLayoutReady(() => {
      this.initLeaf();
      if (this.copilotTab) {
        this.copilotTab.render();
      }
      if (this.semanticSearchTab) {
        this.semanticSearchTab.render();
      }
    });
  }

  onunload(): void {
    this.app.workspace.getLeavesOfType(VIEW_TYPE_AI_SEARCH).forEach((leaf) => leaf.detach());
    this.app.workspace.getLeavesOfType(VIEW_TYPE_AI_COPILOT).forEach((leaf) => leaf.detach());
  }

  initLeaf(): void {
    // Check if both leaves already exist
    const hasSearchLeaf = this.app.workspace.getLeavesOfType(VIEW_TYPE_AI_SEARCH).length > 0;
    const hasCopilotLeaf = this.app.workspace.getLeavesOfType(VIEW_TYPE_AI_COPILOT).length > 0;

    if (hasSearchLeaf && hasCopilotLeaf) {
      return;
    }

    // Create search leaf if missing
    if (!hasSearchLeaf) {
      const rightLeaf = this.app.workspace.getRightLeaf(false);
      if (rightLeaf) {
        rightLeaf.setViewState({
          type: VIEW_TYPE_AI_SEARCH,
        });
      }
    }

    // Create copilot leaf if missing
    if (!hasCopilotLeaf) {
      const rightLeaf = this.app.workspace.getRightLeaf(false);
      if (rightLeaf) {
        rightLeaf.setViewState({
          type: VIEW_TYPE_AI_COPILOT,
        });
      }
    }
  }

  getOpenAIClient(providerId?: string): OpenAIClient {
    // Default to 'openai' if no providerId specified
    const id = providerId || 'openai';

    // Anthropic uses a different client (AnthropicClient), not OpenAI-compatible
    if (id === 'anthropic') {
      throw new Error('Anthropic provider is not supported for OpenAI-compatible operations. Use AnthropicClient for Anthropic models.');
    }

    if (this.clientCache.has(id)) {
      return this.clientCache.get(id)!;
    }

    let apiKey = this.settings.openaiAPIKey;
    let baseURL = this.settings.openaiBaseUrl;

    if (id === 'gcp') {
      if (!this.settings.gcpAPIKey) {
        console.warn(`GCP provider requested but no API key is configured. Falling back to OpenAI.`);
      } else {
        apiKey = this.settings.gcpAPIKey;
        baseURL = 'https://generativelanguage.googleapis.com/v1beta/openai/';
      }
    } else if (id !== 'openai') {
      // Check custom providers
      const customProvider = this.settings.customProviders.find(p => p.id === id);
      if (customProvider) {
        apiKey = customProvider.apiKey;
        baseURL = customProvider.baseUrl;
      } else {
        console.warn(`Custom provider "${id}" not found. It may have been deleted. Falling back to OpenAI.`);
      }
    }

    // Parse embeddings model version to extract just the model name (new format is providerId:modelName)
    let embeddingsModel = this.settings.embeddingsModelVersion || unlabelledEmbeddingModel;
    if (embeddingsModel.includes(':')) {
      const { modelName } = parseModelValue(embeddingsModel);
      embeddingsModel = modelName;
    }

    const client = new OpenAIClient(apiKey, {
      embeddings_model: embeddingsModel,
      baseURL: baseURL || undefined,
    });

    this.clientCache.set(id, client);
    return client;
  }

  async loadSettings() {
    const loadedSettings = await this.loadData();
    this.settings = Object.assign({}, DEFAULT_SETTINGS, loadedSettings);
    if (!loadedSettings?.embeddingsModelVersion && this.settings.vectors.length !== 0) {
      // if the model version was not set in settings, but vectors exist
      this.settings.embeddingsModelVersion = unlabelledEmbeddingModel;
      this.settings.embeddingsEnabled = true;
    }

    if (this.settings.vectors.length !== 0) {
      this.lastIndexedCount = this.settings.vectors.length;
    }

    // Migrate old format model values to new format (providerId:modelName)
    this.migrateModelSettings();

    // Initialize clients
    this.clientCache.clear();
    this.openaiClient = this.getOpenAIClient(); // Default client
    this.anthropicClient = new AnthropicClient(this.settings.anthropicAPIKey);
  }

  private migrateModelSettings() {
    // Migrate embeddingsModelVersion if in old format
    if (this.settings.embeddingsModelVersion && !this.settings.embeddingsModelVersion.includes(':')) {
      const availableModels = availableEmbeddingModels(
        this.settings.openaiAPIKey,
        this.settings.customEmbeddingModels,
        this.settings.providerModels
      );
      const migratedValue = migrateModelValue(
        this.settings.embeddingsModelVersion,
        availableModels,
        this.settings.embeddingsModelProviderId
      );
      console.log(`Migrating embeddingsModelVersion from "${this.settings.embeddingsModelVersion}" to "${migratedValue}"`);
      this.settings.embeddingsModelVersion = migratedValue;
      
      // Extract and set providerId from migrated value
      const { providerId } = parseModelValue(migratedValue);
      this.settings.embeddingsModelProviderId = providerId;
    }

    // Migrate copilotModel if in old format
    if (this.settings.copilotModel && !this.settings.copilotModel.includes(':')) {
      const availableModels = availableChatModels(
        this.settings.openaiAPIKey,
        this.settings.anthropicAPIKey,
        this.settings.customChatModels,
        this.settings.providerModels
      );
      const migratedValue = migrateModelValue(
        this.settings.copilotModel,
        availableModels,
        this.settings.copilotModelProviderId
      );
      console.log(`Migrating copilotModel from "${this.settings.copilotModel}" to "${migratedValue}"`);
      this.settings.copilotModel = migratedValue;
      
      // Extract and set providerId from migrated value
      const { providerId } = parseModelValue(migratedValue);
      this.settings.copilotModelProviderId = providerId;
    }
  }

  clearVectorArray() {
    this.settings.vectors = [];
    this.vectorStore = new VectorStore(this);
    this.lastIndexedCount = 0;
  }

  async saveSettings() {
    console.log('saving...');
    await this.saveData(this.settings);
    this.clientCache.clear();
    this.openaiClient = this.getOpenAIClient();
    this.anthropicClient = new AnthropicClient(this.settings.anthropicAPIKey);
    console.log('done');
  }

  linkTextForFile(file: TFile): { linktext: string, path: string } {
    return {
      linktext: this.app.metadataCache.fileToLinktext(file, file.path),
      path: file.path
    };
  }

  async indexVectorStores() {
    if (!this.settings.embeddingsEnabled || this.indexingStatus === INDEXING_STATUS) {
      return;
    }

    // for now only the first note group will have a vector store
    this.indexingStatus = INDEXING_STATUS;
    this.lastIndexedCount = 0;
    this.trigger('zettelkasten-llm-tools:index-updated');
    await this.saveSettings(); // Save immediately to update UI

    try {
      const noteGroup = this.settings.noteGroups[this.settings.indexedNoteGroup];
      const filesForNoteGroup = filesInGroupFolder(this.app, noteGroup);
      
      // Get the client for the configured embedding model
      const providerId = this.settings.embeddingsModelProviderId;
      const client = this.getOpenAIClient(providerId);

      const concurrencyManager = await generateAndStoreEmbeddings({
        files: filesForNoteGroup,
        app: this.app,
        vectorStore: this.vectorStore,
        openaiClient: client,
        notify: (numCompleted: number) => {
          this.lastIndexedCount = numCompleted;
          this.trigger('zettelkasten-llm-tools:index-updated');
        }
      });
      await concurrencyManager.done();
    } catch (error) {
      console.error('Error during indexing:', error);
    } finally {
      this.indexingStatus = IDLE_STATUS;
      this.trigger('zettelkasten-llm-tools:index-updated');
      await this.saveSettings();
      new Notice(`Indexed ${this.lastIndexedCount} notes`);
    }
  }

  on(name: 'zettelkasten-llm-tools:index-updated' | 'zettelkasten-llm-tools:api-keys-updated', callback: () => void): EventRef {
    return this.events.on(name, callback);
  }

  off(name: 'zettelkasten-llm-tools:index-updated' | 'zettelkasten-llm-tools:api-keys-updated', callback: () => void): void {
    this.events.off(name, callback);
  }

  trigger(name: 'zettelkasten-llm-tools:index-updated' | 'zettelkasten-llm-tools:api-keys-updated'): void {
    this.events.trigger(name);
  }
}

interface WorkspaceWithCustomEvents extends Events {
  on(name: 'zettelkasten-llm-tools:index-updated' | 'zettelkasten-llm-tools:api-keys-updated', callback: () => void): EventRef;
  trigger(name: 'zettelkasten-llm-tools:index-updated' | 'zettelkasten-llm-tools:api-keys-updated'): void;
}

class ZettelkastenLLMToolsPluginSettingTab extends PluginSettingTab {
  plugin: ZettelkastenLLMToolsPlugin;
  private eventListeners: Array<{ name: string; callback: () => void }> = [];

  constructor(app: App, plugin: ZettelkastenLLMToolsPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  hide(): void {
    // Clean up event listeners to prevent memory leaks
    this.eventListeners.forEach(({ name, callback }) => {
      this.plugin.off(name as 'zettelkasten-llm-tools:index-updated' | 'zettelkasten-llm-tools:api-keys-updated', callback);
    });
    this.eventListeners = [];
    
    this.plugin.loadSettings();
  }

  display(): void {
    const {containerEl} = this;

    // Clean up any existing event listeners before registering new ones
    // This prevents accumulation when display() is called multiple times
    this.eventListeners.forEach(({ name, callback }) => {
      this.plugin.off(name as 'zettelkasten-llm-tools:index-updated' | 'zettelkasten-llm-tools:api-keys-updated', callback);
    });
    this.eventListeners = [];

    containerEl.empty();
    new Setting(containerEl)
      .setName('OpenAI API Key')
      .setDesc('Paste your OpenAI API key here.')
      .addText(text => text
        .setPlaceholder('Enter your API key')
        .setValue(this.plugin.settings.openaiAPIKey)
        .then(text => { text.inputEl.type = 'password'; })
        .onChange(async (value: string) => {
          this.plugin.settings.openaiAPIKey = value;
          await this.plugin.saveSettings();
          this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
        }));

    new Setting(containerEl)
      .setName('OpenAI Base URL')
      .setDesc('Custom base URL for OpenAI-compatible endpoints (e.g., for GCP Vertex AI or other providers). Leave empty for default OpenAI API.')
      .addText(text => text
        .setPlaceholder('https://api.openai.com/v1')
        .setValue(this.plugin.settings.openaiBaseUrl)
        .onChange(async (value: string) => {
          this.plugin.settings.openaiBaseUrl = value;
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName('Anthropic API Key')
      .setDesc('Paste your Anthropic API key here.')
      .addText(text => text
        .setPlaceholder('Enter your API key')
        .setValue(this.plugin.settings.anthropicAPIKey)
        .then(text => { text.inputEl.type = 'password'; })
        .onChange(async (value: string) => {
          this.plugin.settings.anthropicAPIKey = value;
          await this.plugin.saveSettings();
          this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
        }));

    new Setting(containerEl)
      .setName('GCP API Key')
      .setDesc('Paste your Google Cloud Vertex AI / Gemini API key here.')
      .addText(text => text
        .setPlaceholder('Enter your API key')
        .setValue(this.plugin.settings.gcpAPIKey)
        .then(text => { text.inputEl.type = 'password'; })
        .onChange(async (value: string) => {
          this.plugin.settings.gcpAPIKey = value;
          
          if (value) {
            // Auto-enumerate GCP models
            try {
              new Notice('Enumerating GCP models...');
              const client = new OpenAIClient(value, {
                baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai/'
              });
              
              const models = await client.openai.models.list();
              const chatModels: ChatModel[] = [];
              const embeddingModels: EmbeddingModel[] = [];

              for (const model of models.data) {
                if (model.id.includes('embedding')) {
                  embeddingModels.push({
                    provider: OPENAI_PROVIDER,
                    name: model.id,
                    displayName: `GCP: ${model.id}`,
                    available: true,
                    providerId: 'gcp'
                  });
                } else if (model.id.includes('gemini')) {
                  chatModels.push({
                    provider: OPENAI_PROVIDER, // Use OpenAI provider interface for GCP
                    name: model.id,
                    displayName: `GCP: ${model.id}`,
                    available: true,
                    providerId: 'gcp'
                  });
                }
              }

              this.plugin.settings.providerModels['gcp'] = {
                chat: chatModels,
                embedding: embeddingModels
              };
              
              new Notice(`Found ${chatModels.length} chat models and ${embeddingModels.length} embedding models from GCP`);
            } catch (error) {
              console.error('Failed to enumerate GCP models:', error);
              new Notice('Failed to enumerate GCP models. Please check your API key.');
            }
          } else {
            // GCP API key was cleared - remove stale GCP models
            if (this.plugin.settings.providerModels['gcp']) {
              delete this.plugin.settings.providerModels['gcp'];
              new Notice('GCP models removed.');
            }
          }

          await this.plugin.saveSettings();
          this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
        }));

    // Custom Providers section
    const customProvidersContainer = containerEl.createDiv('custom-providers-container');
    customProvidersContainer.style.border = '1px solid var(--background-modifier-border)';
    customProvidersContainer.style.padding = '10px';
    customProvidersContainer.style.marginBottom = '20px';
    customProvidersContainer.style.borderRadius = '5px';

    const customProvidersHeading = customProvidersContainer.createEl('h3');
    customProvidersHeading.setText('Custom Providers');
    customProvidersHeading.style.marginTop = '0';
    customProvidersHeading.style.marginBottom = '10px';

    const customProvidersDesc = customProvidersContainer.createEl('p');
    customProvidersDesc.setText('Add custom OpenAI-compatible providers (e.g., local LLMs, other services).');
    customProvidersDesc.style.marginBottom = '10px';
    customProvidersDesc.style.color = 'var(--text-muted)';

    this.plugin.settings.customProviders.forEach((provider, i) => {
      const providerRow = customProvidersContainer.createDiv('custom-provider-row');
      providerRow.style.marginBottom = '10px';
      providerRow.style.padding = '10px';
      providerRow.style.border = '1px solid var(--background-modifier-border)';
      providerRow.style.borderRadius = '4px';

      const topRow = providerRow.createDiv();
      topRow.style.display = 'flex';
      topRow.style.gap = '8px';
      topRow.style.marginBottom = '8px';
      topRow.style.alignItems = 'center';

      const nameInput = topRow.createEl('input', { type: 'text', placeholder: 'Provider Name' });
      nameInput.value = provider.name;
      nameInput.style.flex = '1';
      nameInput.onchange = async () => {
        this.plugin.settings.customProviders[i].name = nameInput.value;
        await this.plugin.saveSettings();
        this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
      };

      const refreshButton = topRow.createEl('button', { text: '↻ Models' });
      refreshButton.onclick = async () => {
        if (!provider.apiKey || !provider.baseUrl) {
          new Notice('Please set API Key and Base URL first');
          return;
        }

        try {
          new Notice(`Enumerating models for ${provider.name}...`);
          const client = new OpenAIClient(provider.apiKey, {
            baseURL: provider.baseUrl
          });
          
          const models = await client.openai.models.list();
          const chatModels: ChatModel[] = [];
          const embeddingModels: EmbeddingModel[] = [];

          for (const model of models.data) {
             // Assuming anything can be chat/embedding, we might need heuristics or just add to both
             // For now, let's look for keywords, or default to chat
             if (model.id.includes('embedding') || model.id.includes('embed')) {
               embeddingModels.push({
                 provider: OPENAI_PROVIDER,
                 name: model.id,
                 displayName: `${provider.name}: ${model.id}`,
                 available: true,
                 providerId: provider.id
               });
             } else {
               chatModels.push({
                 provider: OPENAI_PROVIDER,
                 name: model.id,
                 displayName: `${provider.name}: ${model.id}`,
                 available: true,
                 providerId: provider.id
               });
             }
          }

          this.plugin.settings.providerModels[provider.id] = {
            chat: chatModels,
            embedding: embeddingModels
          };
          
          await this.plugin.saveSettings();
          this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
          new Notice(`Found ${chatModels.length} chat and ${embeddingModels.length} embedding models`);
        } catch (error) {
          console.error('Failed to enumerate models:', error);
          new Notice('Failed to enumerate models. Check settings and network.');
        }
      };

      const deleteButton = topRow.createEl('button', { text: '✕' });
      deleteButton.onclick = async () => {
        const id = this.plugin.settings.customProviders[i].id;
        delete this.plugin.settings.providerModels[id];
        this.plugin.settings.customProviders.splice(i, 1);
        await this.plugin.saveSettings();
        this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
        this.display();
      };

      const bottomRow = providerRow.createDiv();
      bottomRow.style.display = 'flex';
      bottomRow.style.gap = '8px';

      const baseUrlInput = bottomRow.createEl('input', { type: 'text', placeholder: 'Base URL (e.g. http://localhost:11434/v1/)' });
      baseUrlInput.value = provider.baseUrl;
      baseUrlInput.style.flex = '2';
      baseUrlInput.onchange = async () => {
        this.plugin.settings.customProviders[i].baseUrl = baseUrlInput.value;
        await this.plugin.saveSettings();
      };

      const apiKeyInput = bottomRow.createEl('input', { type: 'password', placeholder: 'API Key (optional)' });
      apiKeyInput.value = provider.apiKey;
      apiKeyInput.style.flex = '1';
      apiKeyInput.onchange = async () => {
        this.plugin.settings.customProviders[i].apiKey = apiKeyInput.value;
        await this.plugin.saveSettings();
      };
    });

    const addProviderButton = customProvidersContainer.createEl('button', { text: 'Add Custom Provider' });
    addProviderButton.onclick = async () => {
      this.plugin.settings.customProviders.push({
        id: crypto.randomUUID(),
        name: 'New Provider',
        apiKey: '',
        baseUrl: '',
        enabled: true
      });
      await this.plugin.saveSettings();
      this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
      this.display();
    };

    const statusEl = containerEl.createEl('div', { cls: 'embedding-status' });

    new Setting(statusEl)
      .setName('Enable Embeddings')
      .setDesc('Toggle embeddings functionality on/off')
      .addToggle(toggle => {
        toggle
          .setValue(this.plugin.settings.embeddingsEnabled)
          .onChange(async (value) => {
            this.plugin.settings.embeddingsEnabled = value;
            await this.plugin.saveSettings();
            // Refresh the settings display to update status
            this.display();
          });
      });

    // Helper to get current embedding model value in new format
    const getCurrentEmbeddingModelValue = (): string => {
      const modelVersion = this.plugin.settings.embeddingsModelVersion;
      if (!modelVersion) return '';
      
      // Check if already in new format
      if (modelVersion.includes(':')) {
        return modelVersion;
      }
      
      // Migrate from old format
      const availableModels = availableEmbeddingModels(
        this.plugin.settings.openaiAPIKey,
        this.plugin.settings.customEmbeddingModels,
        this.plugin.settings.providerModels
      );
      return migrateModelValue(modelVersion, availableModels, this.plugin.settings.embeddingsModelProviderId);
    };

    let embeddingModelSettingDropdown: DropdownComponent;
    const embeddingModelSetting = new Setting(statusEl)
      .setName('Model version for embeddings')
      .setDesc('Select the model version you want to use for vector embeddings.')
      .addDropdown(dropdown => {
        embeddingModelSettingDropdown = dropdown;
        const availableModels = availableEmbeddingModels(
          this.plugin.settings.openaiAPIKey,
          this.plugin.settings.customEmbeddingModels,
          this.plugin.settings.providerModels
        );

        // Detect duplicates and show warning if any
        const duplicates = detectDuplicateModelNames(availableModels);
        if (duplicates.size > 0) {
          const duplicateNames = Array.from(duplicates.keys()).join(', ');
          embeddingModelSetting.setDesc(`Select the model version you want to use for vector embeddings. Note: Some model names appear in multiple providers (${duplicateNames}). Models are prefixed with their provider to avoid conflicts.`);
        }

        availableModels.forEach(model => {
          if (model.available) {
            const value = formatModelValue(model.providerId || 'openai', model.name);
            dropdown.addOption(value, model.displayName);
          }
        });

        dropdown.setValue(getCurrentEmbeddingModelValue());
        dropdown.onChange(async (value) => {
          const confirmModal = new EmbeddingsOverwriteConfirmModal(
            this.app,
            this.plugin,
            async (confirmWasClicked) => {
              if (!confirmWasClicked) {
                dropdown.setValue(getCurrentEmbeddingModelValue());
                return;
              }
              const currentValue = getCurrentEmbeddingModelValue();
              if (value !== '' && value !== currentValue) {
                const { providerId, modelName } = parseModelValue(value);
                this.plugin.settings.embeddingsModelVersion = value;
                this.plugin.settings.embeddingsModelProviderId = providerId;

                this.plugin.clearVectorArray();
                await this.plugin.saveSettings();
                await this.plugin.indexVectorStores();
              }
            }
          );
          confirmModal.open();
        });
      });

    // Event listener to refresh embedding dropdown when models change
    const embeddingModelsUpdateCallback = () => {
      const availableModels = availableEmbeddingModels(
        this.plugin.settings.openaiAPIKey,
        this.plugin.settings.customEmbeddingModels,
        this.plugin.settings.providerModels
      );
      
      // Clear existing options
      while (embeddingModelSettingDropdown.selectEl.options.length > 0) {
        embeddingModelSettingDropdown.selectEl.remove(0);
      }
      
      // Repopulate with new models
      availableModels.forEach(model => {
        if (model.available) {
          const value = formatModelValue(model.providerId || 'openai', model.name);
          embeddingModelSettingDropdown.addOption(value, model.displayName);
        }
      });
      
      // Update description if duplicates exist
      const duplicates = detectDuplicateModelNames(availableModels);
      if (duplicates.size > 0) {
        const duplicateNames = Array.from(duplicates.keys()).join(', ');
        embeddingModelSetting.setDesc(`Select the model version you want to use for vector embeddings. Note: Some model names appear in multiple providers (${duplicateNames}). Models are prefixed with their provider to avoid conflicts.`);
      } else {
        embeddingModelSetting.setDesc('Select the model version you want to use for vector embeddings.');
      }
      
      embeddingModelSettingDropdown.setValue(getCurrentEmbeddingModelValue());
    };
    this.plugin.on('zettelkasten-llm-tools:api-keys-updated', embeddingModelsUpdateCallback);
    this.eventListeners.push({ name: 'zettelkasten-llm-tools:api-keys-updated', callback: embeddingModelsUpdateCallback });

    new Setting(statusEl)
      .setName('Note group to index')
      .setDesc('Select which note group to index with embeddings. (Only one note group can be indexed.)')
      .addDropdown(dropdown => {
        this.plugin.settings.noteGroups.forEach((group, index) => {
          dropdown.addOption(index.toString(), group.name);
        });

        dropdown.setValue(this.plugin.settings.indexedNoteGroup.toString());
        dropdown.onChange(async (value) => {
          const confirmModal = new EmbeddingsOverwriteConfirmModal(
            this.app,
            this.plugin,
            async (confirmWasClicked) => {
              if (!confirmWasClicked) {
                dropdown.setValue(this.plugin.settings.indexedNoteGroup.toString());
                return;
              }
              const newIndex = parseInt(value);
              if (newIndex !== this.plugin.settings.indexedNoteGroup) {
                this.plugin.settings.indexedNoteGroup = newIndex;
                this.plugin.clearVectorArray();
                await this.plugin.saveSettings();
                await this.plugin.indexVectorStores();
              }
            }
          );
          confirmModal.open();
        });
      });

    if (this.plugin.settings.embeddingsEnabled) {
      const status = this.plugin.indexingStatus === INDEXING_STATUS
        ? '🔄 Indexing...'
        : '✓ Indexed';

      const statusIndicatorEl = statusEl.createEl('div', {
        text: `Status: ${status}`,
        cls: this.plugin.indexingStatus === INDEXING_STATUS ? 'status-indexing' : 'status-ready'
      });

      const indexedCountEl = statusEl.createEl('div', {
        text: `Indexed notes: ${this.plugin.lastIndexedCount}`,
        cls: 'indexed-count'
      });

      const indexUpdateCallback = () => {
        indexedCountEl.setText(`Indexed notes: ${this.plugin.lastIndexedCount}`);
        const newStatus = this.plugin.indexingStatus === INDEXING_STATUS
          ? '🔄 Indexing...'
          : '✓ Indexed';
        statusIndicatorEl.setText(`Status: ${newStatus}`);
        statusIndicatorEl.classList.toggle('status-indexing', this.plugin.indexingStatus === INDEXING_STATUS);
        statusIndicatorEl.classList.toggle('status-ready', this.plugin.indexingStatus === IDLE_STATUS);
      };
      this.plugin.on('zettelkasten-llm-tools:index-updated', indexUpdateCallback);
      this.eventListeners.push({ name: 'zettelkasten-llm-tools:index-updated', callback: indexUpdateCallback });

      const buttonContainer = statusEl.createEl('div', { cls: 'button-container' });
      buttonContainer.style.marginTop = '0.5em';

      const indexButton = buttonContainer.createEl('button', {
        text: 'Index Notes',
        cls: 'mod-cta',
      });
      indexButton.onclick = async () => {
        await this.plugin.indexVectorStores();
        this.plugin.trigger('zettelkasten-llm-tools:index-updated');
      };
    } else {
      statusEl.createEl('div', {
        text: 'Embeddings are disabled',
        cls: 'status-disabled'
      });
    }

    // Add some basic styles
    statusEl.style.marginTop = '1em';
    statusEl.style.marginBottom = '1em';
    statusEl.style.padding = '1em';
    statusEl.style.backgroundColor = 'var(--background-secondary)';
    statusEl.style.borderRadius = '4px';

    // Helper to get current copilot model value in new format
    const getCurrentCopilotModelValue = (): string => {
      const copilotModel = this.plugin.settings.copilotModel;
      if (!copilotModel) return '';
      
      // Check if already in new format
      if (copilotModel.includes(':')) {
        return copilotModel;
      }
      
      // Migrate from old format
      const availableModels = availableChatModels(
        this.plugin.settings.openaiAPIKey,
        this.plugin.settings.anthropicAPIKey,
        this.plugin.settings.customChatModels,
        this.plugin.settings.providerModels
      );
      return migrateModelValue(copilotModel, availableModels, this.plugin.settings.copilotModelProviderId);
    };

    let copilotModelSettingDropdown: DropdownComponent;
    const copilotModelSetting = new Setting(containerEl)
      .setName('Copilot Model')
      .setDesc('Select which model to use for the AI Copilot')
      .addDropdown(dropdown => {
        copilotModelSettingDropdown = dropdown;
        const availableModels = availableChatModels(
          this.plugin.settings.openaiAPIKey,
          this.plugin.settings.anthropicAPIKey,
          this.plugin.settings.customChatModels,
          this.plugin.settings.providerModels
        );

        // Detect duplicates and show warning if any
        const duplicates = detectDuplicateModelNames(availableModels);
        if (duplicates.size > 0) {
          const duplicateNames = Array.from(duplicates.keys()).join(', ');
          copilotModelSetting.setDesc(`Select which model to use for the AI Copilot. Note: Some model names appear in multiple providers (${duplicateNames}). Models are prefixed with their provider to avoid conflicts.`);
        }

        availableModels.forEach(model => {
          if (model.available) {
            const value = formatModelValue(model.providerId || 'openai', model.name);
            dropdown.addOption(value, model.displayName);
          }
        });

        dropdown.setValue(getCurrentCopilotModelValue())
          .onChange(async (value) => {
            const { providerId, modelName } = parseModelValue(value);
            this.plugin.settings.copilotModel = value;
            this.plugin.settings.copilotModelProviderId = providerId;
            await this.plugin.saveSettings();
          });
      });

    const copilotModelsUpdateCallback = () => {
      const availableModels = availableChatModels(
        this.plugin.settings.openaiAPIKey,
        this.plugin.settings.anthropicAPIKey,
        this.plugin.settings.customChatModels,
        this.plugin.settings.providerModels
      );
      
      // Clear existing options
      while (copilotModelSettingDropdown.selectEl.options.length > 0) {
        copilotModelSettingDropdown.selectEl.remove(0);
      }
      
      // Repopulate with new models
      availableModels.forEach(model => {
        if (model.available) {
          const value = formatModelValue(model.providerId || 'openai', model.name);
          copilotModelSettingDropdown.addOption(value, model.displayName);
        }
      });
      
      // Update description if duplicates exist
      const duplicates = detectDuplicateModelNames(availableModels);
      if (duplicates.size > 0) {
        const duplicateNames = Array.from(duplicates.keys()).join(', ');
        copilotModelSetting.setDesc(`Select which model to use for the AI Copilot. Note: Some model names appear in multiple providers (${duplicateNames}). Models are prefixed with their provider to avoid conflicts.`);
      } else {
        copilotModelSetting.setDesc('Select which model to use for the AI Copilot');
      }
      
      copilotModelSettingDropdown.setValue(getCurrentCopilotModelValue());
    };
    this.plugin.on('zettelkasten-llm-tools:api-keys-updated', copilotModelsUpdateCallback);
    this.eventListeners.push({ name: 'zettelkasten-llm-tools:api-keys-updated', callback: copilotModelsUpdateCallback });

    // Custom Chat Models section
    const customChatModelsContainer = containerEl.createDiv('custom-chat-models-container');
    customChatModelsContainer.style.border = '1px solid var(--background-modifier-border)';
    customChatModelsContainer.style.padding = '10px';
    customChatModelsContainer.style.marginBottom = '20px';
    customChatModelsContainer.style.borderRadius = '5px';

    const customChatModelsHeading = customChatModelsContainer.createEl('h3');
    customChatModelsHeading.setText('Custom Chat Models');
    customChatModelsHeading.style.marginTop = '0';
    customChatModelsHeading.style.marginBottom = '10px';

    const customChatModelsDesc = customChatModelsContainer.createEl('p');
    customChatModelsDesc.setText('Add custom chat models for OpenAI-compatible endpoints.');
    customChatModelsDesc.style.marginBottom = '10px';
    customChatModelsDesc.style.color = 'var(--text-muted)';

    this.plugin.settings.customChatModels.forEach((model, i) => {
      const modelRow = customChatModelsContainer.createDiv('custom-model-row');
      modelRow.style.display = 'flex';
      modelRow.style.gap = '8px';
      modelRow.style.marginBottom = '8px';
      modelRow.style.alignItems = 'center';

      const nameInput = modelRow.createEl('input', { type: 'text', placeholder: 'Model name (e.g., gpt-4)' });
      nameInput.value = model.name;
      nameInput.style.flex = '1';
      nameInput.onchange = async () => {
        this.plugin.settings.customChatModels[i].name = nameInput.value;
        await this.plugin.saveSettings();
        this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
      };

      const displayNameInput = modelRow.createEl('input', { type: 'text', placeholder: 'Display name' });
      displayNameInput.value = model.displayName;
      displayNameInput.style.flex = '1';
      displayNameInput.onchange = async () => {
        this.plugin.settings.customChatModels[i].displayName = displayNameInput.value;
        await this.plugin.saveSettings();
        this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
      };

      const providerSelect = modelRow.createEl('select');
      providerSelect.createEl('option', { value: 'openai', text: 'OpenAI' });
      providerSelect.createEl('option', { value: 'anthropic', text: 'Anthropic' });
      providerSelect.value = model.provider;
      providerSelect.onchange = async () => {
        this.plugin.settings.customChatModels[i].provider = providerSelect.value as 'openai' | 'anthropic';
        await this.plugin.saveSettings();
        this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
      };

      const deleteButton = modelRow.createEl('button', { text: '✕' });
      deleteButton.style.padding = '4px 8px';
      deleteButton.onclick = async () => {
        this.plugin.settings.customChatModels.splice(i, 1);
        await this.plugin.saveSettings();
        this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
        this.display();
      };
    });

    const addChatModelButton = customChatModelsContainer.createEl('button', { text: 'Add Custom Chat Model' });
    addChatModelButton.onclick = async () => {
      this.plugin.settings.customChatModels.push({
        name: '',
        displayName: '',
        provider: 'openai'
      });
      await this.plugin.saveSettings();
      this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
      this.display();
    };

    // Custom Embedding Models section
    const customEmbeddingModelsContainer = containerEl.createDiv('custom-embedding-models-container');
    customEmbeddingModelsContainer.style.border = '1px solid var(--background-modifier-border)';
    customEmbeddingModelsContainer.style.padding = '10px';
    customEmbeddingModelsContainer.style.marginBottom = '20px';
    customEmbeddingModelsContainer.style.borderRadius = '5px';

    const customEmbeddingModelsHeading = customEmbeddingModelsContainer.createEl('h3');
    customEmbeddingModelsHeading.setText('Custom Embedding Models');
    customEmbeddingModelsHeading.style.marginTop = '0';
    customEmbeddingModelsHeading.style.marginBottom = '10px';

    const customEmbeddingModelsDesc = customEmbeddingModelsContainer.createEl('p');
    customEmbeddingModelsDesc.setText('Add custom embedding models for OpenAI-compatible endpoints.');
    customEmbeddingModelsDesc.style.marginBottom = '10px';
    customEmbeddingModelsDesc.style.color = 'var(--text-muted)';

    this.plugin.settings.customEmbeddingModels.forEach((model, i) => {
      const modelRow = customEmbeddingModelsContainer.createDiv('custom-model-row');
      modelRow.style.display = 'flex';
      modelRow.style.gap = '8px';
      modelRow.style.marginBottom = '8px';
      modelRow.style.alignItems = 'center';

      const nameInput = modelRow.createEl('input', { type: 'text', placeholder: 'Model name (e.g., text-embedding-ada-002)' });
      nameInput.value = model.name;
      nameInput.style.flex = '1';
      nameInput.onchange = async () => {
        this.plugin.settings.customEmbeddingModels[i].name = nameInput.value;
        await this.plugin.saveSettings();
        this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
      };

      const displayNameInput = modelRow.createEl('input', { type: 'text', placeholder: 'Display name' });
      displayNameInput.value = model.displayName;
      displayNameInput.style.flex = '1';
      displayNameInput.onchange = async () => {
        this.plugin.settings.customEmbeddingModels[i].displayName = displayNameInput.value;
        await this.plugin.saveSettings();
        this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
      };

      const deleteButton = modelRow.createEl('button', { text: '✕' });
      deleteButton.style.padding = '4px 8px';
      deleteButton.onclick = async () => {
        this.plugin.settings.customEmbeddingModels.splice(i, 1);
        await this.plugin.saveSettings();
        this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
        this.display();
      };
    });

    const addEmbeddingModelButton = customEmbeddingModelsContainer.createEl('button', { text: 'Add Custom Embedding Model' });
    addEmbeddingModelButton.onclick = async () => {
      this.plugin.settings.customEmbeddingModels.push({
        name: '',
        displayName: ''
      });
      await this.plugin.saveSettings();
      this.plugin.trigger('zettelkasten-llm-tools:api-keys-updated');
      this.display();
    };

    this.plugin.settings.noteGroups.forEach((noteGroup, i) => {
      // Create container div for this note group
      const groupContainer = containerEl.createDiv('note-group-container');
      groupContainer.style.border = '1px solid var(--background-modifier-border)';
      groupContainer.style.padding = '10px';
      groupContainer.style.marginBottom = '20px';
      groupContainer.style.borderRadius = '5px';

      // Add heading for group number
      const groupHeading = groupContainer.createEl('h3');
      groupHeading.setText(`${i + 1}. ${noteGroup.name}`);
      groupHeading.style.marginTop = '0';
      groupHeading.style.marginBottom = '10px';

      // group name
      new Setting(groupContainer)
        .setName('Group Name')
        .setDesc('Name of the group of notes')
        .addText(text => text
          .setPlaceholder('Permanent Notes')
          .setValue(noteGroup.name)
          .onChange(async (value) => {
            this.plugin.settings.noteGroups[i].name = value;
            await this.plugin.saveSettings();
          }));

      // select folder
      new Setting(groupContainer)
        .setName('Note group folder')
        .setDesc('Select folder containing notes to index')
        .addDropdown(dropdown => {
          const NO_FOLDER_SELECTED = '(none selected)';

          // Get all folders in vault
          const allFolders = this.app.vault.getAllLoadedFiles()
            .filter((f): f is TFolder => f instanceof TFolder)
            .map(f => f.path);

          // Filter out folders that are already used by other note groups
          const usedFolders = new Set(
            this.plugin.settings.noteGroups
              .filter((g, idx) => idx !== i && g.notesFolder) // Exclude current group
              .map(g => g.notesFolder!)
          );

          const selectableFolders = allFolders.filter(folder => {
            // Keep folder if it's not used and none of its parent folders are used
            return !Array.from(usedFolders).some(usedFolder =>
              folder === usedFolder || folder.startsWith(usedFolder + '/')
            );
          });

          // Add "none selected" option
          selectableFolders.unshift(NO_FOLDER_SELECTED);
          selectableFolders.sort();

          // Populate dropdown with folder paths
          selectableFolders.forEach(folder => {
            dropdown.addOption(folder, folder);
          });

          dropdown.setValue(noteGroup.notesFolder ?? NO_FOLDER_SELECTED);
          dropdown.onChange(async (value) => {
            this.plugin.settings.noteGroups[i].notesFolder = value === NO_FOLDER_SELECTED ? null : value;
            await this.plugin.saveSettings();
          });
        });

      // prompt write
      new Setting(groupContainer)
        .setName('Copilot Prompt')
        .setDesc('System prompt used by this note group')
        .addTextArea(text => {
          text.inputEl.style.width = '100%';
          text.inputEl.style.height = '150px';

          let saveTimeout: NodeJS.Timeout;

          return text
            .setPlaceholder('')
            .setValue(noteGroup.copilotPrompt)
            .onChange(async (value) => {
              // Clear existing timeout
              if (saveTimeout) clearTimeout(saveTimeout);

              // Set new timeout to save after 2 seconds of no typing
              saveTimeout = setTimeout(async () => {
                this.plugin.settings.noteGroups[i].copilotPrompt = value;
                await this.plugin.saveSettings();
              }, 2000);
            });
        });

      if (i !== 0 && this.plugin.settings.noteGroups.length > 1) {
        new Setting(groupContainer)
          .setName('Delete Note Group')
          .setDesc('Remove this note group.')
          .addButton(button => {
            button.setButtonText('Delete')
              .setWarning()
              .onClick(async () => {
                const modal = new Modal(this.app);
                modal.contentEl.createEl("h3", { text: "Delete Note Group" });
                modal.contentEl.createEl("p", { text: "Are you sure you want to delete this note group? This will only remove the group's settings. Your notes and folders will not be affected." });

                const buttonContainer = modal.contentEl.createDiv();
                buttonContainer.style.display = "flex";
                buttonContainer.style.justifyContent = "flex-end";
                buttonContainer.style.gap = "10px";

                const confirmButton = buttonContainer.createEl("button", { text: "Delete", cls: "mod-warning" });
                confirmButton.addEventListener("click", async () => {
                  this.plugin.settings.noteGroups.splice(i, 1);
                  await this.plugin.saveSettings();
                  this.display();
                  modal.close();
                });

                const cancelButton = buttonContainer.createEl("button", { text: "Cancel" });
                cancelButton.addEventListener("click", () => modal.close());

                modal.open();
              });
          });
      }
    });

    new Setting(containerEl)
      .setName('Create New Note Group')
      .setDesc('Add a new note group to the settings.')
      .addButton(button => {
        button.setButtonText('Add Note Group')
          .setCta()
          .onClick(async () => {
            const newNoteGroup = {
              name: `New Note Group ${this.plugin.settings.noteGroups.length + 1}`,
              notesFolder: null,
              copilotPrompt: '',
            };
            this.plugin.settings.noteGroups.push(newNoteGroup);
            await this.plugin.saveSettings();
            this.display(); // Refresh the settings display to show the new group
          });
      });
  }
}