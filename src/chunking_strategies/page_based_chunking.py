from typing import List, Dict, Any
from langchain.docstore.document import Document
from src.chunking_strategies.base_strategy import BaseChunkingStrategy

class PageBasedChunking(BaseChunkingStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            # Each document is a page, so we just add it to the list of chunks
            chunks.append(doc)
        return chunks
