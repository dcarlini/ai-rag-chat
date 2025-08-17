from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_strategy import BaseChunkingStrategy

class SlidingWindowChunking(BaseChunkingStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.chunk_size = config.get("size")
        self.chunk_overlap = config.get("overlap")
        if self.chunk_size is None or self.chunk_overlap is None:
            raise ValueError("SlidingWindowChunking requires 'size' and 'overlap' in its configuration.")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)
