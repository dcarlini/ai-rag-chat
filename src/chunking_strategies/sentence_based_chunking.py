from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_text_splitters import NLTKTextSplitter
from .base_strategy import BaseChunkingStrategy

class SentenceBasedChunking(BaseChunkingStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = NLTKTextSplitter()
        return text_splitter.split_documents(documents)
