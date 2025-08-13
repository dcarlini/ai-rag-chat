import requests
from langchain_openai import ChatOpenAI
from .base_provider import LLMProvider
from typing import List

class LMStudioProvider(LLMProvider):
    def create_llm(self):
        base_url = self.config_manager.get_provider_url("lm_studio")
        return ChatOpenAI(
            base_url=f"{base_url}/v1",
            api_key="lm-studio",
            model=self.model_name,
            temperature=0.1,
            streaming=True,
            callbacks=self.callbacks
        )

    def get_available_models(self) -> List[str]:
        base_url = self.config_manager.get_provider_url("lm_studio")
        try:
            response = requests.get(f"{base_url}/v1/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            return [model["id"] for model in models]
        except requests.exceptions.RequestException:
            return []
