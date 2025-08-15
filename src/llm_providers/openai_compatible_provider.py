import requests
from langchain_openai import ChatOpenAI
from .base_provider import LLMProvider
from typing import List

class OpenAICompatibleProvider(LLMProvider):
    """
    Generic provider for OpenAI-compatible APIs.
    This can be used for LM Studio, OpenRouter, and other OpenAI-compatible services.
    """
    
    def __init__(self, config_manager, model_name: str, callbacks, provider_name: str):
        super().__init__(config_manager, model_name, callbacks)
        self.provider_name = provider_name
    
    def get_base_url(self) -> str:
        """Get the base URL for this provider from config"""
        return self.config_manager.get_provider_url(self.provider_name)
    
    def get_api_key(self) -> str:
        """Get the API key for this provider. Override in subclasses if needed."""
        # Default to provider name as API key (like LM Studio)
        return self.provider_name.replace("_", "-")
    
    def get_api_endpoint(self) -> str:
        """Get the full API endpoint URL"""
        base_url = self.get_base_url()
        return f"{base_url}/v1"
    
    def create_llm(self):
        """Create the ChatOpenAI instance with provider-specific configuration"""
        return ChatOpenAI(
            base_url=self.get_api_endpoint(),
            api_key=self.get_api_key(),
            model=self.model_name,
            temperature=0.1,
            streaming=True,
            callbacks=self.callbacks
        )

    def get_available_models(self) -> List[str]:
        """Fetch available models from the provider's /models endpoint"""
        try:
            models_url = f"{self.get_api_endpoint()}/models"
            headers = {"Authorization": f"Bearer {self.get_api_key()}"}
            
            response = requests.get(models_url, headers=headers)
            response.raise_for_status()
            models = response.json().get("data", [])
            return [model["id"] for model in models]
        except requests.exceptions.RequestException:
            return []
