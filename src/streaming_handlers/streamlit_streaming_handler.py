from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

class StreamlitStreamingHandler(BaseCallbackHandler):
    def __init__(self, container=None):
        self.container = container
        self.text = ""
        self._text_placeholder = None

    def set_container(self, container):
        self.container = container
        self.text = ""  # Reset text for new conversation
        self._text_placeholder = None  # Reset placeholder as well

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        if self.container:
            if not self._text_placeholder:
                self._text_placeholder = self.container.empty()
            self._text_placeholder.markdown(self.text)
