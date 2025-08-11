import sys
from config_manager import ConfigManager
from rag_pipeline import RAGPipeline

def main():
    try:
        config_manager = ConfigManager()
        config = config_manager.getconfig()

        rag_pipeline = RAGPipeline(config)
        rag_pipeline.setup()
        rag_pipeline.chat()
    except (KeyboardInterrupt, SystemExit):
        print("\nExiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
