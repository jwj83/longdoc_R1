"""Lightweight OpenAI-compatible chat completion client."""

from __future__ import annotations

import os
from typing import Dict, List

import requests


class OpenAICompatibleClient:
    """Simple HTTP client for OpenAI-compatible chat completion APIs."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model_name: str | None = None):
        self.base_url = (base_url or os.getenv("BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.model_name = model_name or os.getenv("MODEL_NAME", "")

        if not self.base_url:
            raise ValueError("BASE_URL is not set.")
        if not self.api_key:
            raise ValueError("API_KEY is not set.")
        if not self.model_name:
            raise ValueError("MODEL_NAME is not set.")

    def _candidate_urls(self) -> List[str]:
        """Build candidate completion URLs from BASE_URL.

        Supports BASE_URL as either:
        - API root: https://host or https://host/v1
        - Full endpoint: https://host/v1/chat/completions
        """
        base = self.base_url.rstrip("/")

        # If user already passed a full endpoint, use it directly.
        if base.endswith("/chat/completions"):
            return [base]

        # Common OpenAI-compatible roots.
        if base.endswith("/v1"):
            return [f"{base}/chat/completions"]

        # Try both styles for maximum compatibility.
        return [f"{base}/chat/completions", f"{base}/v1/chat/completions"]

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 120,
    ) -> str:
        """Generate one completion and return plain text content."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        errors: List[str] = []
        for url in self._candidate_urls():
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except requests.RequestException as e:
                errors.append(f"{url}: request exception: {e}")
                continue

            if resp.status_code >= 400:
                errors.append(f"{url}: HTTP {resp.status_code}: {resp.text[:500]}")
                continue

            try:
                data = resp.json()
            except ValueError:
                content_type = resp.headers.get("Content-Type", "")
                preview = resp.text[:300]
                errors.append(f"{url}: non-JSON response (Content-Type={content_type}): {preview}")
                continue

            choices = data.get("choices", [])
            if not choices:
                return ""
            return str(choices[0].get("message", {}).get("content", "")).strip()

        joined = " | ".join(errors) if errors else "unknown error"
        raise RuntimeError(f"LLM request failed on all candidate endpoints: {joined}")
