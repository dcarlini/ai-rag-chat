from typing import Dict, Any
from chunking_strategies import (
    FixedSizeChunking,
    SlidingWindowChunking,
    SentenceBasedChunking,
    ParagraphBasedChunking,
    PageBasedChunking,
)

class ChunkingStrategyFactory:
    _strategies = {
        "fixed_size": FixedSizeChunking,
        "sliding_window": SlidingWindowChunking,
        "sentence_based": SentenceBasedChunking,
        "paragraph_based": ParagraphBasedChunking,
        "page_based": PageBasedChunking,
    }

    @staticmethod
    def create_strategy(strategy_name: str, strategy_config: Dict[str, Any]):
        strategy_class = ChunkingStrategyFactory._strategies.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Unsupported chunking strategy: {strategy_name}")
        
        return strategy_class(strategy_config)
