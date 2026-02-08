"""GLM client via 智谱 BigModel OpenAI-compatible endpoint."""

from __future__ import annotations

import config
from llm.base import OpenAICompatibleClient


class GLMClient(OpenAICompatibleClient):
    """GLM (智谱清言) via BigModel API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or config.GLM_API_KEY,
            base_url=base_url or config.GLM_BASE_URL,
            model=model or config.GLM_MODEL,
        )
