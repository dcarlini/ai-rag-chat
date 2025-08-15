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
    def get_available_providers():
        """Return a list of available provider names."""
        return list(LLMFactory._providers.keys())

    @staticmethod
    def get_embedding_providers():
        """Return a list of providers that support embedding models."""
        config_manager = ConfigManager()
        embedding_providers = []
        
        for provider_name, provider_class in LLMFactory._providers.items():
            # Create an instance of the provider to check if it supports embeddings
            try:
                provider = provider_class(config_manager, "", [])
                if provider.supports_embeddings():
                    embedding_providers.append(provider_name)
            except Exception:
                # If we can't create the provider, we'll skip it
                pass
                
        return embedding_providers

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
    def get_available_models(mode, model_type="chat"):
        config_manager = ConfigManager()
        
        provider_class = LLMFactory._providers.get(mode)
        if not provider_class:
            raise ValueError(f"Unsupported mode: {mode}")

        provider = provider_class(config_manager, "", [])
        all_models = provider.get_available_models()

        if not all_models:
            return []

        embedding_models_whitelist = config_manager.get_config().get("embedding_models_whitelist", [])

        if model_type == "embedding":
            return [m for m in all_models if "embed" in m.lower() or m in embedding_models_whitelist]
        else: # chat
            return [m for m in all_models if "embed" not in m.lower() and m not in embedding_models_whitelist]
