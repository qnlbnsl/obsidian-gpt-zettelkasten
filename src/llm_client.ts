import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { MessageParam } from '@anthropic-ai/sdk/resources';
import type { ChatCompletionMessageParam } from 'openai/resources';

type OpenAIChatMessage = ChatCompletionMessageParam;
type AnthropicChatMessage = MessageParam;

export interface OpenAIClientConfig {
  embeddings_model: EmbeddingModelNames | string;
  quantization_decimals: number;
  baseURL?: string;
};

export const OPENAI_PROVIDER = 'openai';
export const ANTHROPIC_PROVIDER = 'anthropic';

export type EmbeddingsProvider = typeof OPENAI_PROVIDER;

export const OPENAI_EMBEDDING_3_SMALL = 'text-embedding-3-small';
export const OPENAI_EMBEDDING_3_LARGE = 'text-embedding-3-large';
export type EmbeddingModelNames = typeof OPENAI_EMBEDDING_3_SMALL | typeof OPENAI_EMBEDDING_3_LARGE | string;

export interface EmbeddingModel {
  provider: EmbeddingsProvider;
  name: string;
  displayName: string;
  available: boolean;
  providerId?: string;
}

export interface CustomEmbeddingModel {
  name: string;
  displayName: string;
}

export function availableEmbeddingModels(
  openAIKey: string,
  customModels?: CustomEmbeddingModel[],
  providerModels?: Record<string, { embedding: EmbeddingModel[] }>
): EmbeddingModel[] {
  const builtInModels: EmbeddingModel[] = [
    {
      provider: OPENAI_PROVIDER,
      name: OPENAI_EMBEDDING_3_SMALL,
      displayName: 'OpenAI: text-embedding-3-small',
      available: !!openAIKey,
      providerId: 'openai'
    },
    {
      provider: OPENAI_PROVIDER,
      name: OPENAI_EMBEDDING_3_LARGE,
      displayName: 'OpenAI: text-embedding-3-large',
      available: !!openAIKey,
      providerId: 'openai'
    },
  ];

  // Custom models (manual entry) - assume OpenAI provider unless specified otherwise
  const customEmbeddingModels: EmbeddingModel[] = (customModels || [])
    .filter(model => model.name)
    .map(model => ({
      provider: OPENAI_PROVIDER,
      name: model.name,
      displayName: model.displayName || model.name,
      available: true,
      providerId: 'openai' // Legacy custom models default to openai client
    }));

  // Auto-discovered provider models
  const discoveredModels: EmbeddingModel[] = [];
  if (providerModels) {
    Object.values(providerModels).forEach(models => {
      if (models.embedding) {
        discoveredModels.push(...models.embedding);
      }
    });
  }

  return [...builtInModels, ...customEmbeddingModels, ...discoveredModels];
}

export const unlabelledEmbeddingModel = OPENAI_EMBEDDING_3_SMALL;
export const quantizationDecimals = 3;

const defaultOpenAIConfig: OpenAIClientConfig = {
  embeddings_model: OPENAI_EMBEDDING_3_SMALL,
  quantization_decimals: quantizationDecimals,
};

export const CLAUDE_3_5_SONNET = 'claude-3-5-sonnet-latest';
export const CLAUDE_3_5_HAIKU = 'claude-3-5-haiku-latest';

export class AnthropicClient {
  anthropic: Anthropic;
  defaultModel = CLAUDE_3_5_SONNET;

  constructor(apiKey: string) {
    this.anthropic = new Anthropic({
      apiKey: apiKey,
      dangerouslyAllowBrowser: true, // for obsidian, all API keys are provided by the user
    });
  }

  async createMessage(system_prompt: string, msgs: ChatMessage[], modelName?: string) {
    const model = modelName || this.defaultModel;
    let formattedMessages: AnthropicChatMessage[];
    formattedMessages = msgs.map(msg => ({
      role: msg.role === 'user' ? 'user' : 'assistant',
      content: msg.content
    }));
    const msg = await this.anthropic.messages.create({
      model: model,
      max_tokens: 1024,
      messages: formattedMessages,
      system: system_prompt,
    });
    return msg;
  }
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export class OpenAIClient {
  openai: OpenAI;
  config: OpenAIClientConfig;

  constructor(apiKey: string, config?: Partial<OpenAIClientConfig>) {
    this.config = { ...defaultOpenAIConfig, ...config };
    const openaiOptions: { apiKey: string; dangerouslyAllowBrowser: boolean; baseURL?: string } = {
      apiKey: apiKey,
      dangerouslyAllowBrowser: true, // for obsidian, all API keys are provided by the user
    };
    if (this.config.baseURL) {
      openaiOptions.baseURL = this.config.baseURL;
    }
    this.openai = new OpenAI(openaiOptions);
  }

  async createMessage(system_prompt: string, msgs: ChatMessage[], modelName?: string) {
    const model = modelName || OPENAI_GPT4o_MINI;

    // Convert messages to OpenAI format
    const formattedMessages: OpenAIChatMessage[] = [
      { role: 'system', content: system_prompt },
      ...msgs.map(msg => ({
        role: msg.role === 'user' ? 'user' : 'assistant',
        content: msg.content
      } as OpenAIChatMessage))
    ];

    const response = await this.openai.chat.completions.create({
      model: model,
      messages: formattedMessages,
      max_tokens: 1024,
    });

    return response.choices[0].message.content || '';
  }

  async generateOpenAiEmbeddings(docs: Array<string>) {
    const model = this.config.embeddings_model;
    let dimensions;
    if (model === OPENAI_EMBEDDING_3_SMALL) {
      dimensions = 256;
    }
    const embeddings = await this.openai.embeddings.create({
      model,
      input: docs,
      dimensions
    });
    return embeddings.data.map((entry: any) =>
      entry.embedding.map((value: number) =>
        Number(value.toFixed(this.config.quantization_decimals))
      )
    )[0];
  };
}

export async function generateEmbeddings(
  text: string,
  modelName: EmbeddingModelNames,
  openaiClient?: OpenAIClient,
): Promise<number[]> {
  if (!text) {
    throw new Error('No text provided for embedding generation');
  }

  switch (modelName) {
    case OPENAI_EMBEDDING_3_SMALL:
    case OPENAI_EMBEDDING_3_LARGE:
      if (!openaiClient) throw new Error('OpenAI client not initialized');
      return await openaiClient.generateOpenAiEmbeddings([text]);
    default:
      throw new Error(`Unknown embedding model: ${modelName}`);
  }
}

export const OPENAI_GPT4o = 'gpt-4o';
export const OPENAI_GPT4o_MINI = 'gpt-4o-mini';
export const OPENAI_GPT35 = 'gpt-3.5-turbo';
export type ChatModelNames =
  | typeof OPENAI_GPT4o
  | typeof OPENAI_GPT4o_MINI
  | typeof OPENAI_GPT35
  | typeof CLAUDE_3_5_SONNET
  | typeof CLAUDE_3_5_HAIKU
  | string;

export interface ChatModel {
  provider: typeof OPENAI_PROVIDER | typeof ANTHROPIC_PROVIDER;
  name: string;
  displayName: string;
  available: boolean;
  providerId?: string;
}

export interface CustomChatModel {
  name: string;
  displayName: string;
  provider: 'openai' | 'anthropic';
}

export function availableChatModels(
  openAIKey: string,
  anthropicKey: string,
  customModels?: CustomChatModel[],
  providerModels?: Record<string, { chat: ChatModel[] }>
): ChatModel[] {
  const builtInModels: ChatModel[] = [
    {
      provider: OPENAI_PROVIDER,
      name: OPENAI_GPT4o,
      displayName: 'GPT-4o',
      available: !!openAIKey,
      providerId: 'openai'
    },
    {
      provider: OPENAI_PROVIDER,
      name: OPENAI_GPT4o_MINI,
      displayName: 'GPT-4o Mini',
      available: !!openAIKey,
      providerId: 'openai'
    },
    {
      provider: OPENAI_PROVIDER,
      name: OPENAI_GPT35,
      displayName: 'GPT-3.5 Turbo',
      available: !!openAIKey,
      providerId: 'openai'
    },
    {
      provider: ANTHROPIC_PROVIDER,
      name: CLAUDE_3_5_SONNET,
      displayName: 'Claude 3.5 Sonnet',
      available: !!anthropicKey,
      providerId: 'anthropic'
    },
    {
      provider: ANTHROPIC_PROVIDER,
      name: CLAUDE_3_5_HAIKU,
      displayName: 'Claude 3.5 Haiku',
      available: !!anthropicKey,
      providerId: 'anthropic'
    },
  ];

  const customChatModels: ChatModel[] = (customModels || [])
    .filter(model => model.name)
    .map(model => ({
      provider: model.provider === 'anthropic' ? ANTHROPIC_PROVIDER : OPENAI_PROVIDER,
      name: model.name,
      displayName: model.displayName || model.name,
      available: true,
      providerId: model.provider // 'openai' or 'anthropic' from manual entry
    }));

  // Auto-discovered provider models
  const discoveredModels: ChatModel[] = [];
  if (providerModels) {
    Object.values(providerModels).forEach(models => {
      if (models.chat) {
        discoveredModels.push(...models.chat);
      }
    });
  }

  return [...builtInModels, ...customChatModels, ...discoveredModels];
}
