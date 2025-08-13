from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_litellm import ChatLiteLLM
from streaming_handler import StreamingHandler
import os
from config_manager import ConfigManager # Import the new ConfigManager

class LLMFactory:
    @staticmethod
    def create_llm(mode, model_name, callbacks): # Modified to take mode, model_name, callbacks directly
        config_manager = ConfigManager() # Instantiate ConfigManager

        if mode == "lm_studio":
            print(f"\nUsing LM Studio with model: {model_name}")
            base_url = config_manager.get_provider_url("lm_studio")
            return ChatOpenAI(
                base_url=f"{base_url}/v1",
                api_key="lm-studio",
                model=model_name,
                temperature=0.1,
                streaming=True,
                callbacks=callbacks
            )
        elif mode == "litellm":
            print(f"\nUsing LiteLLM with model: {model_name}")
            
            litellm_proxy_key = config_manager.get_litellm_proxy_key()
            if litellm_proxy_key:
                os.environ["LITELLM_API_KEY"] = litellm_proxy_key
            
            # Extract provider from model_name (e.g., "openai/gpt-4" -> "openai")
            provider_name = model_name.split('/')[0] if '/' in model_name else None
            provider_api_key = None
            if provider_name:
                provider_api_key = config_manager.get_api_key(provider_name)
            
            llm_args = {
                "model": model_name,
                "base_url": config_manager.get_provider_url("litellm"),
                "streaming": True,
                "callbacks": callbacks
            }
            
            if provider_api_key:
                llm_args["api_key"] = provider_api_key
                
            return ChatLiteLLM(**llm_args)
        elif mode == "ollama":
            print(f"\nUsing Ollama with model: {model_name}")
            base_url = config_manager.get_provider_url("ollama")
            return ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
                streaming=True,
                callbacks=callbacks
            )
        else:
            raise ValueError(f"Unknown LLM provider: {mode}")