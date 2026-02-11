from logging import Logger
from typing import Optional

from core.client import LLMClient
from models.llm_provider import LLMProvider


class GoogleClient(LLMClient):
    def __init__(self, logger: Optional[Logger] = None):
        super().__init__(llm_provider=LLMProvider.GOOGLE, logger=logger)
