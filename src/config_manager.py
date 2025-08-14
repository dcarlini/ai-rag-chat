import yaml
import os

class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls, config_path="config/config.yml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, 'r') as f:
            ConfigManager._config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")

    def get_config(self):
        return ConfigManager._config

    def get_provider_url(self, provider_name):
        return ConfigManager._config.get("providers", {}).get(provider_name, {}).get("url")

    def get_litellm_proxy_key(self):
        return ConfigManager._config.get("providers", {}).get("litellm", {}).get("api_key")

    def get_api_key(self, key_name):
        return ConfigManager._config.get("api_keys", {}).get(key_name)

    def get_embedding_config(self):
        return ConfigManager._config.get("embedding", {})