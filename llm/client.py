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

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 120,
    ) -> str:
        """Generate one completion and return plain text content."""
        url = f"{self.base_url}/chat/completions"
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
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM request failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()
