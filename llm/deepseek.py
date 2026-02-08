"""DeepSeek client via OpenAI-compatible endpoint."""

from __future__ import annotations

import config
from llm.base import OpenAICompatibleClient


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek (深度求索)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or config.DEEPSEEK_API_KEY,
            base_url=base_url or config.DEEPSEEK_BASE_URL,
            model=model or config.DEEPSEEK_MODEL,
        )
