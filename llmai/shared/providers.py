from enum import Enum


class LLMProvider(Enum):
    CHATGPT = "chatgpt"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    CEREBRAS = "cerebras"
    AZURE = "azure"
    VERTEX = "vertex"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
