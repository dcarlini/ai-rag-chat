# AI RAG Chat

This project is a command-line or GUI application that allows you to chat with your documents using the power of Retrieval-Augmented Generation (RAG). You can also use it as a general-purpose chatbot. It supports various LLM providers and is designed to be extensible.

## Features

- Chat with your documents (PDF, Markdown, text).
- General-purpose chatbot mode.
- Support for multiple LLM providers: Ollama, LM Studio, LiteLLM.
- Command-line interface.
- Graphical user interface (GUI) using Streamlit.
- Easy configuration using YAML.
- Extensible architecture with modular provider system.

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
5.  Set up your configuration:
    ```bash
    cp config/config.yml.example config/config.yml
    ```
    Then edit `config.yml` with your settings:
    - Set the appropriate URLs for your LLM providers
    - Add any necessary API keys for LiteLLM providers
    - Configure other options as needed

## Configuration

The application is configured using `config.yml`. A sample configuration file is provided at `config/config.yml.example`.

### Provider Configuration

Configure your LLM providers in the `providers` section:

```yaml
providers:
  ollama:
    url: http://localhost:11434
  lm_studio:
    url: http://localhost:1234
  litellm:
    url: http://localhost:4000
    api_key: your_litellm_proxy_api_key  # Optional
```

### API Keys

For LiteLLM provider, you can configure API keys for different services:

```yaml
api_keys:
  openai: your_openai_api_key
  anthropic: your_anthropic_api_key
  # Add other provider keys as needed
```

### Document Processing

For RAG functionality, configure document processing settings:

```yaml
embedding:
  provider: ollama  # or openai
  model: nomic-embed-text
  
ingest_docs:
  - path/to/your/documents
  - another/document/path
```

Copy `config.yml.example` to `config.yml` and adjust the settings according to your needs. The configuration file is automatically loaded from the `config` directory.

## Usage

After activating the virtual environment (as described in the Installation section):

### Command-Line Interface

- To start the application in CLI mode:
  ```bash
  python src/app_cli.py
  ```
- The CLI will prompt you to:
  1. Select an LLM provider (Ollama, LM Studio, or LiteLLM)
  2. Choose from available models for the selected provider
  
Available commands in chat:
- `/restart` - Switch to a different provider or model
- `/quit` - Exit the application

### Graphical User Interface

- To start the Streamlit GUI, run the following command:
  ```bash
  streamlit run src/app_gui.py --server.headless true
  ```

## Future Work

- Improve the GUI with more features.
