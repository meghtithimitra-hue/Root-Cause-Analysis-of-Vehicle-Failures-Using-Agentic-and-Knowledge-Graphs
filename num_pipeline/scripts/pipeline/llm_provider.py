"""
LLM Provider — abstraction layer for LLM calls.

Supports Ollama (local) with fallback to None if unavailable.
This module is used ONLY by explanation_generator.py.
The reasoning engine does NOT use LLM — it uses deterministic logic.
"""

import json
import subprocess
from typing import Optional


class LLMProvider:
    """
    Abstraction for LLM inference.

    Supports:
    - Ollama (local, via HTTP API or CLI)
    - Fallback: returns None if LLM unavailable
    """

    def __init__(self, model: str = "llama3.1:8b", provider: str = "ollama"):
        self.model = model
        self.provider = provider
        self._available = None  # lazy check

    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        if self._available is not None:
            return self._available

        if self.provider == "ollama":
            self._available = self._check_ollama()
        else:
            self._available = False

        return self._available

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text or None if LLM unavailable
        """
        if not self.is_available():
            return None

        if self.provider == "ollama":
            return self._generate_ollama(prompt, system_prompt)
        return None

    def _check_ollama(self) -> bool:
        """Check if Ollama is running and has the model."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return self.model in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return False

    def _generate_ollama(
        self, prompt: str, system_prompt: str
    ) -> Optional[str]:
        """Generate using Ollama API."""
        try:
            import requests

            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
        except Exception:
            pass

        return None


def get_llm_provider(
    model: str = "llama3.1:8b",
    provider: str = "ollama"
) -> LLMProvider:
    """
    Factory function to get an LLM provider instance.
    """
    return LLMProvider(model=model, provider=provider)
