import os
import json
import glob
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

CHROMA_PATH = "chromadb"

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.ingest_docs = self.config["ingest_docs"]
        self.chroma_path = CHROMA_PATH
        self.embeddings = self._create_embeddings()
        self.file_loaders = {
            ".pdf": PyPDFLoader,
            ".md": UnstructuredMarkdownLoader,
            ".txt": TextLoader,
        }
        self.vector_store = self._setup_vector_store()

    def _setup_vector_store(self):
        from datetime import datetime

        metadata_file = os.path.join(self.chroma_path, "doc_metadata.json")

        old_metadata = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    old_metadata = json.load(f)
            except Exception:
                old_metadata = {}

        if os.path.exists(self.chroma_path) and os.listdir(self.chroma_path):
            print(f"Loading existing ChromaDB from: {self.chroma_path}")
            vector_store = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings
            )
        else:
            print("No existing ChromaDB found. Creating a new one.")
            vector_store = None

        to_process = []
        new_metadata = {}

        for path in self.ingest_docs:
            path = os.path.expanduser(path)
            if os.path.isfile(path):
                files = [path]
            else:
                files = []
                for file_type in self.file_loaders.keys():
                    files.extend(glob.glob(os.path.join(path, f"**/*{file_type}"), recursive=True))
            
            for file in files:
                mtime = os.path.getmtime(file)
                new_metadata[file] = mtime
                if file not in old_metadata or old_metadata[file] != mtime:
                    to_process.append(file)

        if to_process:
            print(f"Processing {len(to_process)} new/updated files...")
            documents = []
            for file in to_process:
                file_extension = os.path.splitext(file)[1]
                loader_class = self.file_loaders.get(file_extension)
                if loader_class:
                    loader = loader_class(file)
                    documents.extend(loader.load())

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            if vector_store:
                vector_store.add_documents(docs)
            else:
                vector_store = Chroma.from_documents(
                    docs, self.embeddings, persist_directory=self.chroma_path
                )

            print("Vector store updated with new documents.")
        else:
            print("No new or updated files found. Using existing vector store.")

        os.makedirs(self.chroma_path, exist_ok=True)
        with open(metadata_file, "w") as f:
            json.dump(new_metadata, f)

        return vector_store
    
    def _create_embeddings(self):
        embedding_provider = self.config["embedding_provider"]
        if embedding_provider == "ollama":
            return OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=self.config["embedding_base_url"],
            )
        elif embedding_provider == "openai":
            return OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_base=self.config["embedding_base_url"],
                openai_api_key=self.config["embedding_openai_api_key"],
            )
        else:
            raise ValueError("Invalid embedding provider specified")

    # get method for embeddings
    def get_embeddings(self):
        return self.embeddings  
    
    # get method for vector store
    def get_vector_store(self):
        return self.vector_store
