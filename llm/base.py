"""LLM client abstraction -- raw HTTP only, no SDK."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

import requests

logger = logging.getLogger(__name__)


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
        # Metrics tracking
        self.call_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

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

        logger.debug(f"[LLM] Calling {self.model} (call #{self.call_count + 1})...")
        start = __import__("time").time()
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        api_time = __import__("time").time() - start

        # Track metrics from usage field
        self.call_count += 1
        logger.debug(f"[LLM] Call #{self.call_count} completed in {api_time:.2f}s")
        if "usage" in data:
            usage = data["usage"]
            self.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.total_completion_tokens += usage.get("completion_tokens", 0)

        # Standard OpenAI-compatible response shape
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Unexpected response structure: {json.dumps(data, ensure_ascii=False)[:500]}"
            ) from exc

    def get_stats(self) -> dict:
        """Return aggregated metrics for this client instance."""
        return {
            "call_count": self.call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, base_url={self.base_url!r})"
