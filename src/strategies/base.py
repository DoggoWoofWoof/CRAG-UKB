from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import time

from src.pipeline.standardizer import StandardNode
from src.core.engine import CoreEngine

@dataclass
class RetrievalResult:
    query: str
    nodes: List[StandardNode]
    answer: str
    latency: float
    metadata: Dict[str, Any]

class BaseRetriever(ABC):
    def __init__(self, engine: CoreEngine):
        self.engine = engine
        
    @abstractmethod
    def retrieve(self, query: str) -> RetrievalResult:
        pass
        
    def _format_context(self, nodes: List[StandardNode]) -> List[str]:
        return [f"[{n.node_id}] {n.content}" for n in nodes]
