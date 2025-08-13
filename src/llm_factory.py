from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_litellm import ChatLiteLLM
from streaming_handler import StreamingHandler
import os

class LLMFactory:
    @staticmethod
    def create_llm(config):
        mode = config["mode"]
        model_name = config["model_name"]
        callbacks = config.get("callbacks", [StreamingHandler()])

        if mode == "lm_studio":
            print(f"\nUsing LM Studio with model: {model_name}")
            return ChatOpenAI(
                base_url=f"{config['model_base_url']}/v1",
                api_key="lm-studio",
                model=model_name,
                temperature=0.1,
                streaming=True,
                callbacks=callbacks
            )
        elif mode == "litellm":
            print(f"\nUsing LiteLLM with model: {model_name}")
            proxy_key = config.get("litellm_proxy_key")
            if proxy_key:
                os.environ["LITELLM_API_KEY"] = proxy_key
            
            provider_api_key = config.get("provider_api_key")
            
            llm_args = {
                "model": model_name,
                "base_url": config.get("model_base_url"),
                "streaming": True,
                "callbacks": callbacks
            }
            
            if provider_api_key:
                llm_args["api_key"] = provider_api_key
                
            return ChatLiteLLM(**llm_args)
        elif mode == "ollama":
            print(f"\nUsing Ollama with model: {model_name}")
            return ChatOllama(
                model=model_name,
                base_url=config["model_base_url"],
                temperature=0.1,
                streaming=True,
                callbacks=callbacks
            )
        else:
            raise ValueError(f"Unknown LLM provider: {mode}")
