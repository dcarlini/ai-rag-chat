from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain.docstore.document import Document

class BaseChunkingStrategy(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        pass
