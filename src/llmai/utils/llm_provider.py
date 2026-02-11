from typing import Optional
from fastapi import HTTPException

from constants.llm import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GOOGLE_MODEL,
    DEFAULT_OPENAI_MODEL,
)
from enums.llm_provider import LLMProvider
from utils.get_env import (
    get_anthropic_model_env,
    get_custom_model_env,
    get_google_model_env,
    get_llm_provider_env,
    get_ollama_model_env,
    get_openai_model_env,
)


def get_llm_provider(provider: Optional[str] = None):
    try:
        return LLMProvider(provider or get_llm_provider_env())
    except Exception as _:
        raise HTTPException(
            status_code=500,
            detail="Invalid LLM provider. Please select one of: openai, google, anthropic, ollama, custom",
        )


def is_openai_selected():
    return get_llm_provider() == LLMProvider.OPENAI


def is_google_selected():
    return get_llm_provider() == LLMProvider.GOOGLE


def is_anthropic_selected():
    return get_llm_provider() == LLMProvider.ANTHROPIC


def is_ollama_selected():
    return get_llm_provider() == LLMProvider.OLLAMA


def is_custom_llm_selected():
    return get_llm_provider() == LLMProvider.CUSTOM


def get_model(llm_provider: Optional[LLMProvider] = None):
    selected_llm = llm_provider or get_llm_provider()
    if selected_llm == LLMProvider.OPENAI:
        return get_openai_model_env() or DEFAULT_OPENAI_MODEL
    elif selected_llm == LLMProvider.GOOGLE:
        return get_google_model_env() or DEFAULT_GOOGLE_MODEL
    elif selected_llm == LLMProvider.ANTHROPIC:
        return get_anthropic_model_env() or DEFAULT_ANTHROPIC_MODEL
    elif selected_llm == LLMProvider.OLLAMA:
        return get_ollama_model_env()
    elif selected_llm == LLMProvider.CUSTOM:
        return get_custom_model_env()
    else:
        raise HTTPException(
            status_code=500,
            detail="Invalid LLM provider. Please select one of: openai, google, anthropic, ollama, custom",
        )
