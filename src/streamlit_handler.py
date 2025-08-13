from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

class StreamlitStreamingHandler(BaseCallbackHandler):
    def __init__(self, container=None, initial_text=""):
        self.container = container
        self.text = initial_text

    def set_container(self, container):
        self.container = container

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        if self.container:
            self.container.markdown(self.text)

    def on_llm_end(self, response, **kwargs):
        if self.container:
            self.container.markdown(self.text)
