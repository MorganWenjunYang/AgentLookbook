"""Qwen client via DashScope OpenAI-compatible endpoint."""

from __future__ import annotations

import config
from llm.base import OpenAICompatibleClient


class QwenClient(OpenAICompatibleClient):
    """Qwen (通义千问) via DashScope."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or config.QWEN_API_KEY,
            base_url=base_url or config.QWEN_BASE_URL,
            model=model or config.QWEN_MODEL,
        )
