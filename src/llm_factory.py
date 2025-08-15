from config_manager import ConfigManager
from llm_providers import LMStudioProvider, LiteLLMProvider, OllamaProvider, OpenRouterProvider

class LLMFactory:
    _providers = {
        "lm_studio": LMStudioProvider,
        "litellm": LiteLLMProvider,
        "ollama": OllamaProvider,
        "openrouter": OpenRouterProvider
    }

    @staticmethod
    def create_llm(mode, model_name, callbacks):
        config_manager = ConfigManager()
        
        provider_class = LLMFactory._providers.get(mode)
        if not provider_class:
            raise ValueError(f"Unsupported mode: {mode}")

        print(f"\nUsing {mode} with model: {model_name}")
        provider = provider_class(config_manager, model_name, callbacks)
        return provider.create_llm()

    @staticmethod
    def get_available_models(mode):
        config_manager = ConfigManager()
        
        provider_class = LLMFactory._providers.get(mode)
        if not provider_class:
            raise ValueError(f"Unsupported mode: {mode}")

        provider = provider_class(config_manager, "", [])
        return provider.get_available_models()
