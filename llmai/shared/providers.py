from enum import Enum


class LLMProvider(Enum):
    CHATGPT = "chatgpt"
    OPENAI = "openai"
    AZURE = "azure"
    VERTEX = "vertex"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
