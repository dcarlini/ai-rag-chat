# AI RAG Chat

This project is a command-line or GUI application that allows you to chat with your documents using the power of Retrieval-Augmented Generation (RAG). You can also use it as a general-purpose chatbot. It supports various LLM providers and is designed to be extensible.

## Features

-   **Chat with your documents:** Supports PDF, Markdown, and text files.
-   **General-purpose chatbot mode:** Use it as a regular chatbot without document ingestion.
-   **Multiple LLM Providers:** Supports a variety of LLM providers, including:
    -   Ollama
    -   LM Studio
    -   LiteLLM
    -   OpenAI-compatible servers
-   **Flexible Embedding Models:** Select from different embedding providers (Ollama, OpenAI) and models.
-   **Command-Line Interface (CLI):** An interactive CLI for console-based usage.
-   **Graphical User Interface (GUI):** A user-friendly GUI built with Streamlit.
-   **Easy Configuration:** Configure the application using a simple `config.yml` file.
-   **Extensible Architecture:** The modular provider system makes it easy to add new LLM providers.

## Architecture

The project is designed with a clear separation of concerns:

-   **`config_manager.py`:** Handles loading and managing the `config.yml` file.
-   **`rag_pipeline.py`:** The core of the application, responsible for creating and managing the RAG chain.
-   **`llm_factory.py` and `llm_providers/`:** Manage the creation of LLM instances for different providers.
-   **`document_processor.py`:** Handles document loading, splitting, and vector store creation.
-   **`app_cli.py` and `app_gui.py`:** Implement the command-line and graphical user interfaces.
-   **`streaming_handlers/`:** Manage streaming responses for the different interfaces.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ai-rag-chat.git
    cd ai-rag-chat
    ```
2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    ```
3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```
4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Set up your configuration:**
    ```bash
    cp config/config.yml.example config/config.yml
    ```
    Then, edit `config.yml` with your settings.

## Configuration

The application is configured using `config.yml`.

### Defaults

The `defaults` section specifies the default models to use in the UI.

```yaml
defaults:
  chat_model_provider: ollama
  chat_model: llama3
  embedding_model_provider: ollama
  embedding_model: nomic-embed-text
```

### Provider Configuration

Configure your LLM providers in the `providers` section. Make sure the URLs and API keys are correct.

```yaml
providers:
  ollama:
    url: http://localhost:11434
  lm_studio:
    url: http://localhost:1234
  litellm:
    url: http://localhost:4000
    api_key: your_litellm_proxy_api_key  # Optional
  openrouter:
    url: https://openrouter.ai/api
```

### API Keys

For providers that require API keys (like LiteLLM or OpenRouter), you can configure them in the `api_keys` section:

```yaml
api_keys:
  openai: your_openai_api_key
  anthropic: your_anthropic_api_key
  openrouter: sk-or-your_openrouter_api_key
  # Add other provider keys as needed
```

### Document Ingestion

For RAG functionality, configure the `ingest_docs` section:

```yaml
ingest_docs:
  - path/to/your/documents
  - another/document/path
```

## Usage

After activating the virtual environment, you can run the application in either CLI or GUI mode.

### Command-Line Interface

To start the application in CLI mode, run:

```bash
python src/app_cli.py
```

The CLI will prompt you to:

1.  Select an LLM provider.
2.  Choose an available model.
3.  Select an embedding provider.
4.  Choose an available embedding model.

**Available commands in the chat:**

-   `/restart`: Switch to a different provider or model.
-   `/quit`: Exit the application.

### Graphical User Interface

To start the Streamlit GUI, run:

```bash
streamlit run src/app_gui.py --server.headless true
```

The GUI provides a sidebar for:

-   Selecting the LLM provider and model.
-   Selecting the embedding provider and model.
-   Uploading documents for the RAG pipeline.

The RAG pipeline will automatically re-initialize when you change the configuration in the sidebar.

## Future Work

-   Improve the GUI with more features.
-   Add support for more document types.
-   Implement conversation history.