from enum import Enum


class LLMProvider(Enum):
    CHATGPT = "chatgpt"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
