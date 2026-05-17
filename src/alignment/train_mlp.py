"""
Pure MLP Loss Ablation Pipeline.
=================================
Trains the TextPartitionMLP directly against FAISS centroids using mathematically
restricted ablation losses: info_nce_single, info_nce_multi, kl_div, bce.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import pickle
import logging
import argparse
import numpy as np
import faiss

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.pipeline.standardizer import load_nodes
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP
from src.alignment.train_alignment import get_split_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
class PartitionQueryDataset(Dataset):
    def __init__(self, pairs, encoder, num_partitions, dataset_name="2wiki", split="train"):
        self.samples = []
        ds_nodes = []
        for node, pids in pairs:
            valid_pids = set()
            for pid in pids:
                if int(pid) < num_partitions:
                    valid_pids.add(int(pid))
            if valid_pids:
                ds_nodes.append((node, list(valid_pids)))

        if not ds_nodes:
            log.warning(f"No {dataset_name} [{split}] queries found.")
            return

        log.info(f"Pre-encoding {len(ds_nodes)} {dataset_name} [{split}] query-GT pairs...")
        texts = [n.content for n, _ in ds_nodes]
        embeddings = encoder.encode(texts)
        faiss.normalize_L2(embeddings)

        for i, (node, pids) in enumerate(ds_nodes):
            self.samples.append((embeddings[i], pids))

        self.num_partitions = num_partitions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emb, pids = self.samples[idx]
        return torch.tensor(emb, dtype=torch.float32), pids

def collate_fn(batch):
    embs, pids_list = zip(*batch)
    return torch.stack(embs), list(pids_list)

# ═══════════════════════════════════════════════════════════════════
# Loss Functions Engine
# ═══════════════════════════════════════════════════════════════════

def info_nce_single_loss(projected, pids_list, all_centroids, temperature=0.07):
    device = projected.device
    pos_targets = torch.tensor([p[0] for p in pids_list], dtype=torch.long, device=device)
    logits = torch.matmul(projected, all_centroids.T) / temperature
    return F.cross_entropy(logits, pos_targets)

def info_nce_multi_loss(projected, pids_list, all_centroids, temperature=0.07, hn_k=0):
    sim = torch.matmul(projected, all_centroids.T) / temperature
    B, num_coarse = sim.shape

    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    for i, pids in enumerate(pids_list):
        for pid in pids:
            if pid < num_coarse: pos_mask[i, pid] = True

    valid = pos_mask.any(dim=1)
    if valid.sum() == 0: return torch.tensor(0.0, device=projected.device, requires_grad=True)
    
    sim = sim[valid]
    pos_mask = pos_mask[valid]
    
    if hn_k > 0:
        # zero out positives so they cannot be selected as hard negatives explicitly
        neg_sim = sim.masked_fill(pos_mask, -1e9)
        # select strictly top-k hardest false positive targets
        k = min(hn_k, num_coarse - 1)
        _, hard_idx = neg_sim.topk(k, dim=1)
        
        keep_mask = pos_mask.clone()
        keep_mask.scatter_(1, hard_idx, True)
        
        # evaporate everything globally outside {positives U top-k hard negatives}
        sim = sim.masked_fill(~keep_mask, -1e9)
        
    log_probs = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # Extract only valid positive log probabilities to prevent -inf * 0.0 NaN destruction
    loss = -log_probs[pos_mask].sum() / pos_mask.sum()
    return loss

def kl_div_loss(projected, pids_list, all_centroids, temperature=0.07, hn_k=0):
    sim = torch.matmul(projected, all_centroids.T) / temperature
    B, num_coarse = sim.shape
    
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    for i, pids in enumerate(pids_list):
        for pid in pids:
            if pid < num_coarse: pos_mask[i, pid] = True
            
    # Filter empty queries
    valid = pos_mask.any(dim=1)
    if valid.sum() == 0: return torch.tensor(0.0, device=projected.device, requires_grad=True)
    
    sim = sim[valid]
    pos_mask = pos_mask[valid]
    
    if hn_k > 0:
        neg_sim = sim.masked_fill(pos_mask, -1e9)
        k = min(hn_k, num_coarse - 1)
        _, hard_idx = neg_sim.topk(k, dim=1)
        
        keep_mask = pos_mask.clone()
        keep_mask.scatter_(1, hard_idx, True)
        
        # mask both student logits geometrically identically
        sim = sim.masked_fill(~keep_mask, -1e9)
        
    student = F.log_softmax(sim, dim=1)
    
    # Teacher target explicitly isolated and cleanly renormalized natively
    teacher = pos_mask.float()
    if hn_k > 0:
        teacher = teacher.masked_fill(~keep_mask, 0.0)
    teacher = teacher / teacher.sum(dim=1, keepdim=True)

    return F.kl_div(student, teacher, reduction='batchmean')

def bce_multi_label_loss(projected, pids_list, all_centroids, temperature=0.07):
    logits = torch.matmul(projected, all_centroids.T) / temperature

    target_labels = torch.zeros_like(logits)
    for i, pids in enumerate(pids_list):
        target_labels[i, pids] = 1.0

    return F.binary_cross_entropy_with_logits(logits, target_labels)


# ═══════════════════════════════════════════════════════════════════
# Execution Loop
# ═══════════════════════════════════════════════════════════════════

def _run_epoch(model, dataloader, centroids_gpu, loss_type, tau=0.07, hn_k=0, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0
    num_batches = 0
    with torch.set_grad_enabled(optimizer is not None):
        for embs, pids_list in dataloader:
            embs = embs.to(centroids_gpu.device)
            
            projected = model(embs)
            projected = F.normalize(projected, dim=-1)
            
            if loss_type == "info_nce_single": loss = info_nce_single_loss(projected, pids_list, centroids_gpu, temperature=tau)
            elif loss_type == "info_nce_multi": loss = info_nce_multi_loss(projected, pids_list, centroids_gpu, temperature=tau, hn_k=hn_k)
            elif loss_type == "kl_div": loss = kl_div_loss(projected, pids_list, centroids_gpu, temperature=tau, hn_k=hn_k)
            elif loss_type == "bce": loss = bce_multi_label_loss(projected, pids_list, centroids_gpu, temperature=tau)
            else: raise ValueError("Unknown loss configuration.")
                
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / max(num_batches, 1)


GLOBAL_LOADER_CACHE = {}
GLOBAL_CENTROID_CACHE = {}

def train(dataset_name="2wiki", loss_type="info_nce_single", epochs=100, batch_size=64, lr=1e-4, tau=0.07, hn_k=0):
    nodes_path = "data/processed/master_nodes.json"
    partition_map_path = f"data/ukb_storage/{dataset_name}/partition_map.json"
    centroids_path = f"data/ukb_storage/{dataset_name}/centroids.index"
    
    # Target nested ablation cleanly avoiding pipeline collisions organically
    ckpt_dir = f"checkpoints/{dataset_name}/hnm_ablation"
    os.makedirs(ckpt_dir, exist_ok=True)
    output_path = os.path.join(ckpt_dir, f"alignment_mlp_{loss_type}_tau_{tau:g}_hnm_{hn_k}.pth")

    log.info(f"Loading data for {dataset_name} | Target Loss: {loss_type} | Target Tau: {tau:g} | Target HNM: {hn_k}")
    if not os.path.exists(partition_map_path):
        log.warning(f"Partition map missing for {dataset_name}. Skipping ablation.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    centroid_cache_key = dataset_name
    if centroid_cache_key not in GLOBAL_CENTROID_CACHE:
        all_nodes = load_nodes(nodes_path)
        with open(partition_map_path, 'r') as f:
            partition_map = json.load(f)
            
        centroid_index = faiss.read_index(centroids_path)
        num_parts = centroid_index.ntotal
        embed_dim = centroid_index.d
        centroids_np = np.array([centroid_index.reconstruct(i) for i in range(num_parts)])
        centroids_gpu = torch.tensor(centroids_np, dtype=torch.float32).to(device)
        centroids_gpu = F.normalize(centroids_gpu, dim=-1)
        GLOBAL_CENTROID_CACHE[centroid_cache_key] = (centroids_gpu, num_parts, embed_dim, all_nodes, partition_map)

    centroids_gpu, num_parts, embed_dim, all_nodes, partition_map = GLOBAL_CENTROID_CACHE[centroid_cache_key]

    # Cache keyed on dataset+batch only — data is invariant across loss/tau/hn_k
    cache_key = f"{dataset_name}_{batch_size}"
    
    if GLOBAL_LOADER_CACHE.get(cache_key) == "empty":
        log.warning(f"Skipping {dataset_name} — previously found empty.")
        return
        
    if cache_key not in GLOBAL_LOADER_CACHE:
        encoder = DenseEncoder()
        splits = get_split_pairs(all_nodes, partition_map, dataset_name)
        
        trn_ds = PartitionQueryDataset(splits["train"], encoder, num_parts, dataset_name, split="train")
        val_ds = PartitionQueryDataset(splits["val"], encoder, num_parts, dataset_name, split="val")
        tst_ds = PartitionQueryDataset(splits["test"], encoder, num_parts, dataset_name, split="test")
        
        if len(trn_ds) == 0:
            GLOBAL_LOADER_CACHE[cache_key] = "empty"
            return
        
        trn_ldr = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        val_ldr = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
        tst_ldr = DataLoader(tst_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
        
        GLOBAL_LOADER_CACHE[cache_key] = (trn_ldr, val_ldr, tst_ldr)
        
    trn_ldr, val_ldr, tst_ldr = GLOBAL_LOADER_CACHE[cache_key]
    
    model = TextPartitionMLP(input_dim=embed_dim, hidden_dim=512, output_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)
    
    best_val = float('inf')
    best_state = None
    epochs_without_improvement = 0
    
    with tqdm(total=epochs, desc=f"Training MLP ({loss_type} | τ={tau:g} | hn={hn_k})") as pbar:
        for epoch in range(epochs):
            trl = _run_epoch(model, trn_ldr, centroids_gpu, loss_type, tau=tau, hn_k=hn_k, optimizer=optimizer)
            val = _run_epoch(model, val_ldr, centroids_gpu, loss_type, tau=tau, hn_k=hn_k, optimizer=None)
            tel = _run_epoch(model, tst_ldr, centroids_gpu, loss_type, tau=tau, hn_k=hn_k, optimizer=None)
            
            scheduler.step(val)
            
            is_best = ""
            if val < best_val:
                best_val = val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                is_best = " \u2605=best"
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            pbar.set_postfix_str(f"TrL={trl:.4f}, VaL={val:.4f}, TeL={tel:.4f}{is_best}")
            pbar.update(1)
            
            if epochs_without_improvement >= 20:
                log.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
    if best_state is None:
        log.error(f"No valid checkpoint state found for {dataset_name} | {loss_type} | tau={tau} | hn_k={hn_k}. Skipping save.")
        return

    torch.save({
        'model_state_dict': best_state,
        'loss_type': loss_type,
        'tau': tau,
        'hn_k': hn_k,
        'input_dim': embed_dim,
        'hidden_dim': 512,
        'output_dim': embed_dim,
    }, output_path)
    log.info(f"Strict Checkpoint successfully mapped at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2wiki")
    parser.add_argument("--loss_type", type=str, default="info_nce_single", choices=["info_nce_single", "info_nce_multi", "kl_div", "bce"])
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    if args.dataset == "all":
        for ds in ["squad", "metaqa", "musique", "2wiki"]:
            train(dataset_name=ds, loss_type=args.loss_type, epochs=args.epochs)
    else:
        train(dataset_name=args.dataset, loss_type=args.loss_type, epochs=args.epochs)