import requests
from langchain_ollama import ChatOllama
from .base_provider import LLMProvider
from typing import List

class OllamaProvider(LLMProvider):
    def create_llm(self):
        base_url = self.config_manager.get_provider_url("ollama")
        return ChatOllama(
            base_url=base_url,
            model=self.model_name,
            callbacks=self.callbacks
        )

    def get_available_models(self) -> List[str]:
        base_url = self.config_manager.get_provider_url("ollama")
        try:
            response = requests.get(f"{base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.exceptions.RequestException:
            return []
