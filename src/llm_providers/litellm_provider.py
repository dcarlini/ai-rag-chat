import os
import requests
from langchain_litellm import ChatLiteLLM
from .base_provider import LLMProvider
from typing import List

class LiteLLMProvider(LLMProvider):
    def create_llm(self):
        # Extract provider from model_name (e.g., "openai/gpt-4" -> "openai")
        provider_name = self.model_name.split('/')[0] if '/' in self.model_name else None
        provider_api_key = None
        if provider_name:
            provider_api_key = self.config_manager.get_api_key(provider_name)

        # Get LiteLLM proxy API key from the standard api_keys section
        litellm_proxy_key = self.config_manager.get_api_key("litellm")
        if litellm_proxy_key:
            os.environ["LITELLM_API_KEY"] = litellm_proxy_key

        llm_args = {
            "model": self.model_name,
            "base_url": self.config_manager.get_provider_url("litellm"),
            "streaming": True,
            "callbacks": self.callbacks
        }

        if provider_api_key:
            llm_args["api_key"] = provider_api_key

        return ChatLiteLLM(**llm_args)

    def get_available_models(self) -> List[str]:
        base_url = self.config_manager.get_provider_url("litellm")
        proxy_key = self.config_manager.get_api_key("litellm")
        try:
            headers = {}
            if proxy_key:
                headers["Authorization"] = f"Bearer {proxy_key}"
            response = requests.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            models = response.json().get("data", [])
            return [model["id"] for model in models]
        except requests.exceptions.RequestException:
            return []
