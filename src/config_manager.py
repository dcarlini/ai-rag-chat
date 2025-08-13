import os
import sys
import argparse
from dotenv import load_dotenv


class ConfigManager:
    def __init__(self):
        parser = argparse.ArgumentParser(description="RAG Chat with your documents")
        parser.add_argument("--env", help="Path to .env file")
        args, _ = parser.parse_known_args()

        env_path = args.env
        if env_path:
            if os.path.exists(env_path):
                load_dotenv(dotenv_path=env_path)
                print(f"Loaded environment variables from {env_path}")
            else:
                print(f"Warning: specified --env file '{env_path}' not found.")
        elif os.path.exists(".env"):
            load_dotenv()
            print("Loaded environment variables from default .env file")

        self.config = self._setupconfig()

    def getconfig(self):
        return self.config

    def _setupconfig(self):
        ingest_docs_str = os.getenv("INGEST_DOCS")
        config = {
            "ingest_docs": ingest_docs_str.split(',') if ingest_docs_str else [],
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER"),
            "embedding_base_url": os.getenv("EMBEDDING_BASE_URL"),
            "embedding_openai_api_key": os.getenv("EMBEDDING_OPENAI_API_KEY"),
            "mode": os.getenv("MODE"),
            "model_name": os.getenv("MODEL_NAME"),
            "model_base_url": os.getenv("MODEL_BASE_URL"),
            "litellm_proxy_key": os.getenv("LITELLM_PROXY_KEY"),
            "provider_api_key": os.getenv("PROVIDER_API_KEY"),
        }

        print("\n--- Loaded Configuration from .env ---")
        for key, value in config.items():
            if value:
                if key in ['embedding_openai_api_key', 'litellm_proxy_key', 'provider_api_key']:
                    print(f"{key}: ************")
                else:
                    print(f"{key}: {value}")
        print("-------------------------------------\n")

        if not config["ingest_docs"]:
            config["ingest_docs"] = self._get_doc_paths()

        if config["ingest_docs"]:
            if not config["embedding_provider"]:
                config["embedding_provider"] = self._get_embedding_provider()
            
            embedding_config = self._get_embedding_config(config)
            config.update(embedding_config)

        if not config["mode"]:
            config["mode"] = self._get_llm_provider()
        
        llm_config = self._get_llm_config(config)
        config.update(llm_config)

        return config

    def _get_doc_paths(self):
        while True:
            doc_paths = input(
                "Enter the paths to your documents or directories (comma-separated, or press Enter to skip): "
            ).strip()
            if not doc_paths:
                return []
            paths = [p.strip() for p in doc_paths.split(",")]
            if all(os.path.exists(p) for p in paths):
                return paths
            else:
                print(
                    f"Error: One or more paths do not exist. Please try again."
                )

    def _get_embedding_provider(self):
        print("\nSelect your embedding provider:")
        print("1. Ollama")
        print("2. OpenAI-compatible (e.g., LM Studio)")
        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                return "ollama"
            elif choice == "2":
                return "openai"
            else:
                print("Please enter 1 or 2")

    def _get_embedding_config(self, config):
        provider = config.get("embedding_provider")
        base_url = config.get("embedding_base_url")
        api_key = config.get("embedding_openai_api_key")

        if not base_url:
            if provider == "ollama":
                base_url = (
                    input(
                        "Enter Ollama server IP and port for embeddings (e.g., http://localhost:11434) or press Enter for local: "
                    ).strip()
                    or "http://localhost:11434"
                )
            elif provider == "openai":
                base_url = (
                    input(
                        "Enter OpenAI-compatible server IP and port for embeddings (e.g., http://localhost:1234): "
                    ).strip()
                    or "http://localhost:1234"
                )
        
        new_config = {"embedding_base_url": base_url}
        if provider == "openai":
            if not api_key:
                api_key = input(
                    "Enter your OpenAI API key (or press Enter for none): "
                ).strip()
            new_config["embedding_openai_api_key"] = api_key
        return new_config

    def _get_llm_provider(self):
        print("\nSelect your LLM provider:")
        print("1. Ollama (local or remote)")
        print("2. LM Studio")
        print("3. LiteLLM")
        while True:
            llm_choice = input("Enter choice (1, 2, or 3): ").strip()
            if llm_choice == "1":
                return "ollama"
            elif llm_choice == "2":
                return "lm_studio"
            elif llm_choice == "3":
                return "litellm"
            else:
                print("Please enter 1, 2, or 3")

    def _get_llm_config(self, config):
        provider = config.get("mode")
        if provider == "ollama":
            return self._get_ollama_config(config)
        elif provider == "lm_studio":
            return self._get_lm_studio_config(config)
        elif provider == "litellm":
            return self._get_litellm_config(config)
        return {}

    def _get_ollama_config(self, config):
        base_url = config.get("model_base_url")
        model_name = config.get("model_name")

        if not base_url:
            base_url = (
                input(
                    "Enter Ollama server IP and port (e.g., http://localhost:11434) or press Enter for local: "
                ).strip()
                or "http://localhost:11434"
            )
        
        if not model_name:
            models = self._get_ollama_models(base_url)
            model_name = (
                self._select_model(models)
                if models
                else self._prompt_for_model_name("Ollama")
            )
        return {"model_base_url": base_url, "model_name": model_name}

    def _get_lm_studio_config(self, config):
        base_url = config.get("model_base_url")
        model_name = config.get("model_name")

        if not base_url:
            base_url = (
                input(
                    "Enter LM Studio server IP and port (e.g., http://localhost:1234): "
                ).strip()
                or "http://localhost:1234"
            )
        
        if not model_name:
            models = self._get_lm_studio_models(base_url)
            if models:
                model_name = self._select_model(models)
            else:
                print("Could not fetch models from LM Studio. Make sure a model is loaded.")
                sys.exit(1)
        return {"model_base_url": base_url, "model_name": model_name}

    def _get_litellm_config(self, config):
        base_url = config.get("model_base_url")
        model_name = config.get("model_name")
        proxy_key = config.get("litellm_proxy_key")
        provider_key = config.get("provider_api_key")

        if not base_url:
            base_url = (
                input(
                    "Enter LiteLLM proxy server IP and port (e.g., http://localhost:4000): "
                ).strip()
                or "http://localhost:4000"
            )

        if not proxy_key:
            proxy_key = self._get_litellm_proxy_key()

        if not provider_key:
            provider_key = self._get_provider_api_key()

        if not model_name:
            models = self._get_litellm_models(base_url, proxy_key)
            if models:
                model_name = self._select_model(models)
            else:
                model_name = self._prompt_for_model_name("LiteLLM")

        return {"model_base_url": base_url, "model_name": model_name, "litellm_proxy_key": proxy_key, "provider_api_key": provider_key}

    def _get_litellm_proxy_key(self):
        return input("Enter your LiteLLM Proxy key (or press Enter for none): ").strip()

    def _get_provider_api_key(self):
        return input("Enter your Provider API key (e.g., OpenAI key) or press Enter for none: ").strip()

    def _prompt_for_model_name(self, provider):
        return (
            input(f"Enter {provider} model name (e.g., llama3): ").strip() or "llama3"
        )

    def _get_ollama_models(self, base_url):
        try:
            import requests

            response = requests.get(f"{base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return None

    def _get_lm_studio_models(self, base_url):
        try:
            import requests

            response = requests.get(f"{base_url}/v1/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            return [model["id"] for model in models]
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to LM Studio: {e}")
            return None

    def _get_litellm_models(self, base_url, api_key=None):
        try:
            import requests
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            response = requests.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            models = response.json().get("data", [])
            return [model["id"] for model in models]
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to LiteLLM: {e}")
            return None

    def _select_model(self, models):
        print("\nSelect a model:")
        for i, model in enumerate(models):
            print(f"{i + 1}. {model}")
        while True:
            try:
                model_index = (
                    int(input(f"Enter choice (1-{len(models)}): ").strip()) - 1
                )
                if 0 <= model_index < len(models):
                    return models[model_index]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
