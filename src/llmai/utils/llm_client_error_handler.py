from fastapi import HTTPException
from anthropic import APIError as AnthropicAPIError
from openai import APIError as OpenAIAPIError
from google.genai.errors import APIError as GoogleAPIError


def handle_llm_client_exceptions(e: Exception) -> HTTPException:
    print(e)
    if isinstance(e, OpenAIAPIError):
        return HTTPException(status_code=500, detail="OpenAI API error")
    if isinstance(e, GoogleAPIError):
        return HTTPException(status_code=500, detail="Google API error")
    if isinstance(e, AnthropicAPIError):
        return HTTPException(status_code=500, detail="Anthropic API error")
    return HTTPException(status_code=500, detail="LLM API error")
