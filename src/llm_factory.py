from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_litellm import ChatLiteLLM
from StreamingHandler import StreamingHandler

class LLMFactory:
    @staticmethod
    def create_llm(config):
        mode = config["mode"]
        model_name = config["model_name"]
        callbacks = [StreamingHandler()]

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
            return ChatLiteLLM(
                model=model_name,
                openai_api_key=config.get("litellm_api_key"),
                streaming=True,
                callbacks=callbacks
            )
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
