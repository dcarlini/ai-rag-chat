import streamlit as st
import os
import tempfile
import requests
from rag_pipeline import RAGPipeline
from llm_factory import LLMFactory
from document_processor import DocumentProcessor
from streamlit_handler import StreamlitStreamingHandler
from config_manager import ConfigManager # Import the new ConfigManager

# Instantiate ConfigManager globally or pass it around
config_manager = ConfigManager()

def get_available_models(mode):
    """Gets the available models for the selected mode using LLMFactory."""
    try:
        return LLMFactory.get_available_models(mode)
    except Exception as e:
        st.error(f"Error fetching models for {mode}: {e}")
        return None

def setup_pipeline(selected_mode, selected_model_name, selected_embedding_provider, selected_embedding_model, uploaded_files, handler):
    """Sets up the RAG pipeline and returns the chain."""
    try:
        # Create a temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded files to the temporary directory
        saved_files = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                path = os.path.join(temp_dir, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                saved_files.append(path)
        
        embedding_config = {
            "provider": selected_embedding_provider,
            "model": selected_embedding_model,
            "openai_api_key": config_manager.get_config().get("embedding", {}).get("openai_api_key")
        }

        pipeline_config = {
            "mode": selected_mode,
            "model_name": selected_model_name,
            "callbacks": [handler],
            "ingest_docs": saved_files,
            "embedding": embedding_config,
        }
        
        pipeline = RAGPipeline(pipeline_config, handler=handler)
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
    if "handler" not in st.session_state:
        st.session_state.handler = StreamlitStreamingHandler()
    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = "ollama" # Default
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = "" # Default
    if "selected_embedding_provider" not in st.session_state:
        st.session_state.selected_embedding_provider = "ollama" # Default
    if "selected_embedding_model" not in st.session_state:
        st.session_state.selected_embedding_model = "" # Default
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    with st.sidebar:
        st.header("Configuration")
        
        # LLM Provider Selection
        st.session_state.selected_mode = st.selectbox("LLM Provider", ["ollama", "lm_studio", "litellm"])
        
        # Model Name Selection based on selected provider
        models = get_available_models(st.session_state.selected_mode)

        if models:
            st.session_state.selected_model_name = st.selectbox("Model Name", models)
        else:
            st.session_state.selected_model_name = st.text_input("Model Name (could not fetch, enter manually)", "llama3")

        st.markdown("---")
        
        # Embedding Provider Selection
        st.session_state.selected_embedding_provider = st.selectbox("Embedding Provider", ["ollama", "openai"])

        # Embedding Model Selection based on selected provider
        embedding_models = get_available_models(st.session_state.selected_embedding_provider)

        if embedding_models:
            st.session_state.selected_embedding_model = st.selectbox("Embedding Model", embedding_models)
        else:
            st.session_state.selected_embedding_model = st.text_input("Embedding Model (could not fetch, enter manually)", "nomic-embed-text")


        # Document Uploader
        st.session_state.uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True)

        st.markdown("--- ")
        st.markdown("Configuration loaded from `config/config.yml`")
        st.markdown("Please ensure your LLM servers are running and `config.yml` is correctly set up.")


        if st.button("Apply Configuration"):
            with st.spinner("Setting up RAG Pipeline..."):
                st.session_state.rag_chain = setup_pipeline(
                    st.session_state.selected_mode,
                    st.session_state.selected_model_name,
                    st.session_state.selected_embedding_provider,
                    st.session_state.selected_embedding_model,
                    st.session_state.uploaded_files,
                    st.session_state.handler
                )
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
