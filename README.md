# AI RAG Chat

This project is a command-line application that allows you to chat with your documents using the power of Retrieval-Augmented Generation (RAG). You can also use it as a general-purpose chatbot. It supports various LLM providers and is designed to be extensible.

## Features

- Chat with your documents (PDF, Markdown, text).
- General-purpose chatbot mode.
- Support for multiple LLM providers: Ollama, LM Studio, LiteLLM.
- Command-line interface.
- Graphical user interface (GUI) using Streamlit.
- Easy configuration using `.env` files.
- Extensible architecture.

## Installation

1.  Clone the repository.
2.  Create the virtual environment:
    ```bash
    python3 -m venv .venv
    ```
3.  Activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```
4.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  The application can be configured using `.env` files. There are three sample files in the `config` directory: `ollama.env`, `lm_studio.env`, and `litellm.env`.
2.  You can create your own `.env` file in the root directory or use the `--env` command-line argument to specify the path to your configuration file.
3.  The following variables can be set in the `.env` file:
    - `INGEST_DOCS`: Comma-separated paths to your documents.
    - `EMBEDDING_PROVIDER`: The embedding provider (`ollama` or `openai`).
    - `EMBEDDING_BASE_URL`: The URL for the embedding server.
    - `EMBEDDING_OPENAI_API_KEY`: Your API key for OpenAI embeddings.
    - `MODE`: The LLM provider to use (`ollama`, `lm_studio`, or `litellm`).
    - `MODEL_NAME`: The specific model name to use.
    - `MODEL_BASE_URL`: The URL for the LLM server.
    - `LITELLM_API_KEY`: Your API key for LiteLLM.

## Usage

After activating the virtual environment (as described in the Installation section):

### Command-Line Interface

- To start the application, run the following command:
  ```bash
  python src/app_cli.py
  ```
- You can also specify the path to your configuration file:
  ```bash
  python src/app_cli.py --env config/ollama.env
  ```

### Graphical User Interface

- To start the Streamlit GUI, run the following command:
  ```bash
  streamlit run src/app_gui.py --server.headless true
  ```

## Future Work

- Improve the GUI with more features.
