from .openai_compatible_provider import OpenAICompatibleProvider

class OpenRouterProvider(OpenAICompatibleProvider):
    """
    OpenRouter provider implementation.
    Uses OpenRouter's API which is OpenAI-compatible.
    """
    
    def __init__(self, config_manager, model_name: str, callbacks):
        super().__init__(config_manager, model_name, callbacks, "openrouter")
    
    def get_api_key(self) -> str:
        """Get the OpenRouter API key from config"""
        config = self.config_manager.get_config()
        api_keys = config.get("api_keys", {})
        return api_keys.get("openrouter", "")
    
    def get_base_url(self) -> str:
        """Get OpenRouter base URL from config, with fallback to default"""
        try:
            return self.config_manager.get_provider_url("openrouter")
        except:
            # Fallback to default OpenRouter URL if not configured
            return "https://openrouter.ai/api"
    
    def supports_embeddings(self) -> bool:
        """OpenRouter does not support embedding models"""
        return False
