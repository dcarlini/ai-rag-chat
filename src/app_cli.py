import sys
from config_manager import ConfigManager
from rag_pipeline import RAGPipeline
from streaming_handlers.command_line_streaming_handler import CommandLineStreamingHandler
from llm_factory import LLMFactory

def get_mode_selection():
    # Get available providers from LLMFactory
    providers = LLMFactory.get_available_providers()
    
    # Create a mapping of numbers to providers
    modes = {str(i+1): provider for i, provider in enumerate(providers)}
    
    print("\nAvailable modes:")
    for key, mode in modes.items():
        print(f"{key}. {mode}")
    
    while True:
        choice = input(f"\nSelect mode (1-{len(modes)}): ")
        if choice in modes:
            return modes[choice]
        print("Invalid choice. Please try again.")

def get_model_selection(mode):
    print(f"\nFetching available models for {mode}...")
    try:
        models = LLMFactory.get_available_models(mode)
        if not models:
            print(f"No models found for {mode}. Please check your configuration and connection.")
            return None
        
        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = int(input(f"\nSelect model (1-{len(models)}): "))
                if 1 <= choice <= len(models):
                    return models[choice - 1]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    except Exception as e:
        print(f"Error fetching models: {e}")
        return None

def get_embedding_provider_selection():
    # Get embedding providers from LLMFactory
    embedding_providers = LLMFactory.get_embedding_providers()
    
    # Create a mapping of numbers to providers
    providers = {str(i+1): provider for i, provider in enumerate(embedding_providers)}
    
    print("\nAvailable embedding providers:")
    for key, provider in providers.items():
        print(f"{key}. {provider}")
    
    while True:
        choice = input(f"\nSelect embedding provider (1-{len(providers)}): ")
        if choice in providers:
            return providers[choice]
        print("Invalid choice. Please try again.")

def get_embedding_model_selection(provider):
    print(f"\nFetching available models for {provider} embeddings...")
    try:
        models = LLMFactory.get_available_models(provider, model_type="embedding")
        if not models:
            print(f"No models found for {provider}. Please check your configuration and connection.")
            return None
        
        print("\nAvailable embedding models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = int(input(f"\nSelect embedding model (1-{len(models)}): "))
                if 1 <= choice <= len(models):
                    return models[choice - 1]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    except Exception as e:
        print(f"Error fetching models: {e}")
        return None

def setup_providers(config):
    # Get mode selection
    mode = get_mode_selection()
    if not mode:
        print("Mode selection failed.")
        return None, None, None, None

    # Get model selection
    model = get_model_selection(mode)
    if not model:
        print("Model selection failed.")
        return None, None, None, None

    # Get embedding provider selection
    embedding_provider = get_embedding_provider_selection()
    if not embedding_provider:
        print("Embedding provider selection failed.")
        return None, None, None, None

    # Get embedding model selection
    embedding_model = get_embedding_model_selection(embedding_provider)
    if not embedding_model:
        print("Embedding model selection failed.")
        return None, None, None, None

    return mode, model, embedding_provider, embedding_model

def run_chat(config, handler, rag_pipeline):
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Check for special commands
            if user_input.lower() == '/quit':
                print("\nExiting chat...")
                return False  # Signal to exit the program
            elif user_input.lower() == '/restart':
                print("\nRestarting provider selection...")
                return True  # Signal to restart provider selection
            
            # Process normal chat input
            if user_input:
                rag_pipeline.process_input(user_input)
                
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            return False

def main():
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        while True:
            # Get mode and model selections
            mode, model, embedding_provider, embedding_model = setup_providers(config)
            if mode is None or model is None or embedding_provider is None or embedding_model is None:
                print("Setup failed. Exiting...")
                return

            # Update config with selected mode and model
            config["mode"] = mode
            config["model_name"] = model
            if "embedding" not in config:
                config["embedding"] = {}
            config["embedding"]["provider"] = embedding_provider
            config["embedding"]["model"] = embedding_model

            print(f"\nInitializing chat with {mode} using model: {model}")
            print(f"Using {embedding_provider} for embeddings with model: {embedding_model}")
            print("\nAvailable commands:")
            print("/restart - Restart provider selection")
            print("/quit    - Exit the program")
            
            # Initialize RAG pipeline with streaming handler
            handler = CommandLineStreamingHandler()
            rag_pipeline = RAGPipeline(config, handler=handler)
            rag_pipeline.setup()

            # Start chat loop
            should_restart = run_chat(config, handler, rag_pipeline)
            if not should_restart:
                break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
