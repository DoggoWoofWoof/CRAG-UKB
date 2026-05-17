import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Sequential, Linear, ReLU, Dropout, LayerNorm

class BaseAlignmentEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.input_proj = Linear(in_channels, hidden_channels)
        
        # Readout from 6 layers * 3 pooling types
        self.readout_dim = hidden_channels * num_layers * 3
        
        self.readout_proj = Sequential(
            Linear(self.readout_dim, hidden_channels * 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels * 2, hidden_channels),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, out_channels)
        )
        self.readout_skip = Linear(self.readout_dim, out_channels)

    def pool(self, h_list, batch):
        pooled = []
        for h in h_list:
            pooled.extend([
                global_mean_pool(h, batch),
                global_max_pool(h, batch),
                global_add_pool(h, batch)
            ])
        return torch.cat(pooled, dim=1)

    def project_text(self, text_emb, device):
        """Helper to project a raw text embedding (BERT) into the same space."""
        # Treat text as a single-node graph for the projector
        batch = torch.arange(text_emb.size(0), dtype=torch.long, device=device)
        
        # Auto-pad text queries since they lack topological/structural metrics naturally
        if text_emb.size(1) < self.input_proj.in_features:
            padding_dim = self.input_proj.in_features - text_emb.size(1)
            padding = torch.zeros((text_emb.size(0), padding_dim), dtype=text_emb.dtype, device=device)
            text_emb = torch.cat([text_emb, padding], dim=1)

        h = F.relu(self.input_proj(text_emb))
        # Since there's no graph structure for pure text, we just replicate it 
        # to match the expected readout dimension (6 layers * 3 pooling)
        h_list = [h] * self.num_layers
        pooled = self.pool(h_list, batch)
        return F.normalize(self.readout_proj(pooled) + self.readout_skip(pooled), p=2, dim=-1)

class GINAlignmentEncoder(BaseAlignmentEncoder):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.3):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        
        for _ in range(num_layers):
            nn_gin = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Dropout(dropout),
                Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(nn_gin))
            self.lns.append(LayerNorm(hidden_channels))

    def forward(self, x, edge_index, batch):
        h = F.relu(self.input_proj(x))
        h_list = []
        
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index)
            h = F.relu(self.lns[i](h_new + h))
            h_list.append(h)
            
        pooled = self.pool(h_list, batch)
        return F.normalize(self.readout_proj(pooled) + self.readout_skip(pooled), p=2, dim=-1)

class GCNAlignmentEncoder(BaseAlignmentEncoder):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.3):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.lns.append(LayerNorm(hidden_channels))

    def forward(self, x, edge_index, batch):
        h = F.relu(self.input_proj(x))
        h_list = []
        
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index)
            h = F.relu(self.lns[i](h_new + h))
            h_list.append(h)
            
        pooled = self.pool(h_list, batch)
        return F.normalize(self.readout_proj(pooled) + self.readout_skip(pooled), p=2, dim=-1)

class SAGEAlignmentEncoder(BaseAlignmentEncoder):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.3):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.lns.append(LayerNorm(hidden_channels))

    def forward(self, x, edge_index, batch):
        h = F.relu(self.input_proj(x))
        h_list = []
        
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index)
            h = F.relu(self.lns[i](h_new + h))
            h_list.append(h)
            
        pooled = self.pool(h_list, batch)
        return F.normalize(self.readout_proj(pooled) + self.readout_skip(pooled), p=2, dim=-1)
