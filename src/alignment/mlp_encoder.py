"""
Text-to-Partition MLP Encoder — Maps BERT query embeddings to partition-aligned space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPartitionMLP(nn.Module):
    """
    Lightweight MLP that projects BERT query embeddings (384/768-dim)
    into a partition-aligned embedding space for FAISS centroid search.
    
    Architecture: Linear → ReLU → Dropout → Linear → L2-normalize
    
    Used as an alternative to raw FAISS centroid search to test whether
    a learned projection improves partition selection recall.
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256,
                 output_dim: int = 384, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize."""
        projected = self.net(x)
        return F.normalize(projected, p=2, dim=-1)
