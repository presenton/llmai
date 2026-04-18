from enum import Enum


class LLMProvider(Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
