import json
from typing import List, Dict, Any, Optional

class StandardNode:
    def __init__(self, node_id: str, content: str, metadata: Dict[str, Any], neighbors: Optional[List[str]] = None):
        self.node_id = node_id
        self.content = content  # Raw text
        self.metadata = metadata # Source dataset, partition_id, type, etc.
        self.neighbors = neighbors or [] # List of neighboring node_ids
        
    def to_dict(self):
        return {
            "node_id": self.node_id,
            "content": self.content,
            "metadata": self.metadata,
            "neighbors": self.neighbors
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            node_id=data["node_id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            neighbors=data.get("neighbors", [])
        )

def save_nodes(nodes: List[StandardNode], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([n.to_dict() for n in nodes], f, ensure_ascii=False, indent=2)

def load_nodes(path: str) -> List[StandardNode]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [StandardNode.from_dict(d) for d in data]
