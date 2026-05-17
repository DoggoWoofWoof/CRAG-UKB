import torch
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class DenseEncoder:
    def __init__(self, model_name: str = 'multi-qa-MiniLM-L6-cos-v1', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

# Note: For ColBERT, we will use Ragatouille or a custom wrap if needed,
# but for now we'll define a placeholder or interface if required.
