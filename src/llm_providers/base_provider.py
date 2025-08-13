from abc import ABC, abstractmethod
from typing import List, Optional
from langchain.callbacks.base import BaseCallbackHandler

class LLMProvider(ABC):
    def __init__(self, config_manager, model_name: str, callbacks: List[BaseCallbackHandler]):
        self.config_manager = config_manager
        self.model_name = model_name
        self.callbacks = callbacks

    @abstractmethod
    def create_llm(self):
        """Create and return the LLM instance"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return a list of available models for this provider"""
        pass
