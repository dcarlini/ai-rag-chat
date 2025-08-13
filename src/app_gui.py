
import streamlit as st
import os
import tempfile
import requests
from rag_pipeline import RAGPipeline
from llm_factory import LLMFactory
from document_processor import DocumentProcessor
from streamlit_handler import StreamlitStreamingHandler

def get_ollama_models(base_url):
    try:
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models]
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Ollama: {e}")
        return None

def get_lm_studio_models(base_url):
    try:
        response = requests.get(f"{base_url}/v1/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        return [model["id"] for model in models]
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to LM Studio: {e}")
        return None

def get_litellm_models(base_url, api_key=None):
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        models = response.json().get("data", [])
        return [model["id"] for model in models]
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to LiteLLM: {e}")
        if e.response and e.response.status_code == 401:
            st.error("Authentication error. Please check your API key.")
        return None

def setup_pipeline(config, handler):
    """Sets up the RAG pipeline and returns the chain."""
    try:
        config["callbacks"] = [handler]
        
        # Create a temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded files to the temporary directory
        saved_files = []
        if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
            for uploaded_file in st.session_state.uploaded_files:
                path = os.path.join(temp_dir, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                saved_files.append(path)
        
        config["ingest_docs"] = saved_files
        
        pipeline = RAGPipeline(config)
        pipeline.setup()
        return pipeline.chain
            
    except Exception as e:
        st.error(f"Error setting up RAG pipeline: {e}")
        return None

def main():
    st.title("AI RAG Chat")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "config" not in st.session_state:
        st.session_state.config = {}
    if "handler" not in st.session_state:
        st.session_state.handler = StreamlitStreamingHandler()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # LLM Provider
        st.session_state.config["mode"] = st.selectbox("LLM Provider", ["ollama", "lm_studio", "litellm"])
        
        # Base URL
        if st.session_state.config["mode"] in ["ollama", "lm_studio", "litellm"]:
            default_url = {
                "ollama": "http://localhost:11434",
                "lm_studio": "http://localhost:1234",
                "litellm": "http://localhost:4000"
            }
            st.session_state.config["model_base_url"] = st.text_input("Base URL", default_url[st.session_state.config["mode"]])
        
        # API Keys
        if st.session_state.config["mode"] == "litellm":
            st.session_state.config["litellm_proxy_key"] = st.text_input("LiteLLM Proxy Key (optional)", type="password")
            st.session_state.config["provider_api_key"] = st.text_input("Provider API Key (optional)", type="password")

        # Model Name
        if st.session_state.config["mode"] == "ollama":
            models = get_ollama_models(st.session_state.config["model_base_url"])
            if models:
                st.session_state.config["model_name"] = st.selectbox("Model Name", models)
            else:
                st.session_state.config["model_name"] = st.text_input("Model Name", "llama3")
        elif st.session_state.config["mode"] == "lm_studio":
            models = get_lm_studio_models(st.session_state.config["model_base_url"])
            if models:
                st.session_state.config["model_name"] = st.selectbox("Model Name", models)
            else:
                st.session_state.config["model_name"] = st.text_input("Model Name", "Not available - check LM Studio")
        elif st.session_state.config["mode"] == "litellm":
            proxy_key = st.session_state.config.get("litellm_proxy_key")
            models = get_litellm_models(st.session_state.config["model_base_url"], proxy_key)
            if models:
                st.session_state.config["model_name"] = st.selectbox("Model Name", models)
            else:
                st.session_state.config["model_name"] = st.text_input("Model Name", "llama3")

        # Document Uploader
        st.session_state.uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True)

        # Embedding Provider
        if st.session_state.uploaded_files:
            st.session_state.config["embedding_provider"] = st.selectbox("Embedding Provider", ["ollama", "openai"])
            if st.session_state.config["embedding_provider"] == "ollama":
                st.session_state.config["embedding_base_url"] = st.text_input("Embedding Base URL", "http://localhost:11434")
            elif st.session_state.config["embedding_provider"] == "openai":
                st.session_state.config["embedding_base_url"] = st.text_input("Embedding Base URL", "http://localhost:1234")
                st.session_state.config["embedding_openai_api_key"] = st.text_input("Embedding OpenAI API Key", type="password")


        if st.button("Apply Configuration"):
            with st.spinner("Setting up RAG Pipeline..."):
                st.session_state.rag_chain = setup_pipeline(st.session_state.config, st.session_state.handler)
                if st.session_state.rag_chain:
                    st.success("Configuration applied successfully!")
                    st.session_state.messages = [] # Clear messages on re-config

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.rag_chain:
                message_placeholder = st.empty()
                st.session_state.handler.set_container(message_placeholder)
                
                response = st.session_state.rag_chain.invoke({"query": prompt})
                
                if isinstance(response, dict) and "result" in response:
                    message_placeholder.markdown(response["result"])
                    st.session_state.messages.append({"role": "assistant", "content": response["result"]})
                else:
                     # The handler will have already updated the UI
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.handler.text})

            else:
                st.warning("Please configure the RAG pipeline in the sidebar first.")

if __name__ == "__main__":
    main()
