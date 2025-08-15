from .openai_compatible_provider import OpenAICompatibleProvider

class LMStudioProvider(OpenAICompatibleProvider):
    """
    LM Studio provider implementation.
    Uses LM Studio's OpenAI-compatible API.
    """
    
    def __init__(self, config_manager, model_name: str, callbacks):
        super().__init__(config_manager, model_name, callbacks, "lm_studio")
    
    def get_api_key(self) -> str:
        """LM Studio uses a fixed API key"""
        return "lm-studio"
