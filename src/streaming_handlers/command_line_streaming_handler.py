from langchain.callbacks.base import BaseCallbackHandler

class CommandLineStreamingHandler(BaseCallbackHandler):
    def __init__(self):
        self.buffer = ""

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)  # prints tokens live

    def on_llm_end(self, response, **kwargs):
        print("\n" + "-" * 30)  # optional line after complete output

