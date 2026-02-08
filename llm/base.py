"""LLM client abstraction -- raw HTTP only, no SDK."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

import requests


class LLMClient(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        """Send a list of messages and return the assistant's text reply."""


class OpenAICompatibleClient(LLMClient):
    """Shared implementation for any OpenAI-compatible chat/completions API.

    Qwen (DashScope), GLM (智谱), DeepSeek all expose this format.
    """

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    # ── public interface ──────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # Standard OpenAI-compatible response shape
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Unexpected response structure: {json.dumps(data, ensure_ascii=False)[:500]}"
            ) from exc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, base_url={self.base_url!r})"
