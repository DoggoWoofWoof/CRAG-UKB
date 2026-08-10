"""
Pure MLP Loss Ablation Pipeline.
=================================
Trains the TextPartitionMLP directly against FAISS centroids using mathematically
restricted ablation losses: info_nce_single, info_nce_multi, kl_div, bce.

Coverage-aware losses (Jigsaw FullCov transfer): coverage_kl, coverage_infonce,
coverage. These add a CVaR-over-positives + FullCov@K top-K barrier term
(src/alignment/coverage_losses.py) on top of a base contrastive objective so
training optimizes retrieval of the WEAKEST required partition (all golds in
top-K), not just average positive mass. Coverage checkpoints are named with an
extra `_lam_{lambda_cov}` suffix so they never collide with the base ablations.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import csv
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
from src.alignment.coverage_losses import partition_coverage_loss

# Coverage-loss default hyperparameters (mirrors Jigsaw's paper-final config).
COVERAGE_DEFAULTS = {
    "lambda_cov": 0.5,
    "cov_temperature": 0.05,
    "target_topk": 20,
    "topk_weight": 0.35,
    "topk_margin": 0.0,
    "margin_weight": 0.25,
    "cvar_fraction": 0.25,
    "positive_aggregation": "cvar",
}

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
# Coverage-aware losses (Jigsaw FullCov transfer)
# ═══════════════════════════════════════════════════════════════════
# Each combines a base contrastive objective with lambda_cov * coverage_term,
# where the coverage term is CVaR-over-positives(CE + margin) + topk_weight *
# CVaR(FullCov@K barrier). The coverage term operates on its own temperature
# (cov_temperature) so it is decoupled from the base loss temperature.


def _coverage_term(projected, pids_list, all_centroids, cov_temperature,
                   target_topk, topk_weight, topk_margin, margin_weight,
                   cvar_fraction, positive_aggregation):
    return partition_coverage_loss(
        projected, pids_list, all_centroids,
        temperature=cov_temperature,
        target_topk=target_topk,
        topk_weight=topk_weight,
        topk_margin=topk_margin,
        margin_weight=margin_weight,
        cvar_fraction=cvar_fraction,
        positive_aggregation=positive_aggregation,
    )


def coverage_kl_loss(projected, pids_list, all_centroids, temperature=0.07, hn_k=0,
                     lambda_cov=0.5, cov_temperature=0.05, target_topk=20,
                     topk_weight=0.35, topk_margin=0.0, margin_weight=0.25,
                     cvar_fraction=0.25, positive_aggregation="cvar"):
    """Primary method: KL(+HNM) base + lambda_cov * FullCov coverage term."""
    base = kl_div_loss(projected, pids_list, all_centroids, temperature=temperature, hn_k=hn_k)
    cov = _coverage_term(projected, pids_list, all_centroids, cov_temperature,
                         target_topk, topk_weight, topk_margin, margin_weight,
                         cvar_fraction, positive_aggregation)
    return base + lambda_cov * cov


def coverage_infonce_loss(projected, pids_list, all_centroids, temperature=0.07, hn_k=0,
                          lambda_cov=0.5, cov_temperature=0.05, target_topk=20,
                          topk_weight=0.35, topk_margin=0.0, margin_weight=0.25,
                          cvar_fraction=0.25, positive_aggregation="cvar"):
    """Ablation: multi-positive InfoNCE(+HNM) base + lambda_cov * coverage term."""
    base = info_nce_multi_loss(projected, pids_list, all_centroids, temperature=temperature, hn_k=hn_k)
    cov = _coverage_term(projected, pids_list, all_centroids, cov_temperature,
                         target_topk, topk_weight, topk_margin, margin_weight,
                         cvar_fraction, positive_aggregation)
    return base + lambda_cov * cov


def coverage_only_loss(projected, pids_list, all_centroids, temperature=0.07, hn_k=0,
                       lambda_cov=0.5, cov_temperature=0.05, target_topk=20,
                       topk_weight=0.35, topk_margin=0.0, margin_weight=0.25,
                       cvar_fraction=0.25, positive_aggregation="cvar"):
    """Ablation: coverage term alone (no base contrastive loss).

    temperature/hn_k are accepted for a uniform call signature but unused here.
    lambda_cov IS applied so a lambda sweep on this variant is meaningful and
    consistent with coverage_kl/coverage_infonce (and so distinct _lam_ checkpoints
    are not misleadingly identical).
    """
    return lambda_cov * _coverage_term(projected, pids_list, all_centroids, cov_temperature,
                                       target_topk, topk_weight, topk_margin, margin_weight,
                                       cvar_fraction, positive_aggregation)


COVERAGE_LOSSES = {
    "coverage_kl": coverage_kl_loss,
    "coverage_infonce": coverage_infonce_loss,
    "coverage": coverage_only_loss,
}


# ═══════════════════════════════════════════════════════════════════
# Execution Loop
# ═══════════════════════════════════════════════════════════════════

def _run_epoch(model, dataloader, centroids_gpu, loss_type, tau=0.07, hn_k=0, optimizer=None,
               cov_params=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    cov_params = cov_params or {}

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
            elif loss_type in COVERAGE_LOSSES:
                loss = COVERAGE_LOSSES[loss_type](
                    projected, pids_list, centroids_gpu, temperature=tau, hn_k=hn_k, **cov_params)
            else: raise ValueError(f"Unknown loss configuration: {loss_type}")
                
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / max(num_batches, 1)


def _val_coverage(model, loader, centroids_gpu, k=20, max_q=2000):
    """Cheap per-epoch FullCov@k + gt_recall@k on the val set, computed by
    ranking the frozen centroids directly (no engine needed). Capped to max_q
    queries so it stays fast even for large val splits — this is a training
    curve, not the final eval (which the benchmarks compute on full test)."""
    model.eval()
    device = centroids_gpu.device
    fc_sum = gtr_sum = 0.0
    n = 0
    with torch.no_grad():
        for embs, pids_list in loader:
            if n >= max_q:
                break
            proj = F.normalize(model(embs.to(device)), dim=-1)
            sims = torch.matmul(proj, centroids_gpu.T)
            topk = torch.topk(sims, min(k, sims.shape[1]), dim=1).indices.cpu().tolist()
            for i, pids in enumerate(pids_list):
                gt = {int(p) for p in pids}
                if not gt:
                    continue
                tk = set(topk[i])
                fc_sum += 1.0 if gt.issubset(tk) else 0.0
                gtr_sum += len(gt & tk) / len(gt)
                n += 1
                if n >= max_q:
                    break
    if n == 0:
        return 0.0, 0.0
    return fc_sum / n, gtr_sum / n


GLOBAL_LOADER_CACHE = {}
GLOBAL_CENTROID_CACHE = {}

def train(dataset_name="2wiki", loss_type="info_nce_single", epochs=100, batch_size=64, lr=1e-4, tau=0.07, hn_k=0,
          limit=0, tag="", **coverage_kwargs):
    nodes_path = "data/processed/master_nodes.json"
    partition_map_path = f"data/ukb_storage/{dataset_name}/partition_map.json"
    centroids_path = f"data/ukb_storage/{dataset_name}/centroids.index"

    is_coverage = loss_type in COVERAGE_LOSSES
    # Coverage hyperparameters: fall back to Jigsaw-final defaults for any unset key.
    cov_params = {**COVERAGE_DEFAULTS, **{k: v for k, v in coverage_kwargs.items() if v is not None}}
    lambda_cov = cov_params["lambda_cov"]

    # Target nested ablation cleanly avoiding pipeline collisions organically.
    # Coverage runs get an extra _lam_{lambda} suffix so they never overwrite the
    # base kl_div/info_nce_multi HNM checkpoints (which SuperModel indexes by name).
    ckpt_dir = f"checkpoints/{dataset_name}/hnm_ablation"
    os.makedirs(ckpt_dir, exist_ok=True)
    # A `_lim{N}` suffix isolates limited smoke-run checkpoints from full-data
    # production checkpoints so a quick test never overwrites a real model.
    lim_suffix = f"_lim{limit}" if limit and limit > 0 else ""
    tag_suffix = f"_{tag}" if tag else ""   # non-clobbering marker (e.g. covbase, encft)
    if is_coverage:
        output_path = os.path.join(
            ckpt_dir,
            f"alignment_mlp_{loss_type}_tau_{tau:g}_hnm_{hn_k}_lam_{lambda_cov:g}{tag_suffix}{lim_suffix}.pth")
    else:
        output_path = os.path.join(
            ckpt_dir, f"alignment_mlp_{loss_type}_tau_{tau:g}_hnm_{hn_k}{tag_suffix}{lim_suffix}.pth")

    log.info(
        f"Loading data for {dataset_name} | Target Loss: {loss_type} | Target Tau: {tau:g} | "
        f"Target HNM: {hn_k}"
        + (f" | lambda_cov={lambda_cov:g} | cov_tau={cov_params['cov_temperature']:g} | "
           f"topk={cov_params['target_topk']} (w={cov_params['topk_weight']:g}) | "
           f"agg={cov_params['positive_aggregation']}" if is_coverage else ""))
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

    # Cache keyed on dataset+batch+limit — data is invariant across loss/tau/hn_k
    cache_key = f"{dataset_name}_{batch_size}_{limit}"
    
    if GLOBAL_LOADER_CACHE.get(cache_key) == "empty":
        log.warning(f"Skipping {dataset_name} — previously found empty.")
        return
        
    if cache_key not in GLOBAL_LOADER_CACHE:
        encoder = DenseEncoder()
        splits = get_split_pairs(all_nodes, partition_map, dataset_name)

        if limit and limit > 0:
            # Fast smoke run: cap each split. Full-data runs use limit=0.
            splits = {s: pairs[:limit] for s, pairs in splits.items()}

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
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    # Crash-safe per-epoch log: written incrementally (each epoch appended +
    # flushed) so a mid-training crash (OOM/kill) still leaves the curve on disk.
    # Per-dataset/per-model layout, mirroring how the benchmarks are stored.
    stem = os.path.splitext(os.path.basename(output_path))[0]
    logs_dir = os.path.join("logs", dataset_name, stem)
    os.makedirs(logs_dir, exist_ok=True)
    hist_path = os.path.join(logs_dir, "history.csv")
    _hist_cols = ["epoch", "train_loss", "val_loss", "test_loss",
                  "val_full_coverage@20", "val_gt_recall@20", "lr", "is_best"]
    with open(hist_path, "w", newline="", encoding="utf-8") as _hf:
        csv.DictWriter(_hf, fieldnames=_hist_cols).writeheader()
    
    with tqdm(total=epochs, desc=f"Training MLP ({loss_type} | τ={tau:g} | hn={hn_k})") as pbar:
        for epoch in range(epochs):
            _cov = cov_params if is_coverage else None
            trl = _run_epoch(model, trn_ldr, centroids_gpu, loss_type, tau=tau, hn_k=hn_k, optimizer=optimizer, cov_params=_cov)
            val = _run_epoch(model, val_ldr, centroids_gpu, loss_type, tau=tau, hn_k=hn_k, optimizer=None, cov_params=_cov)
            tel = _run_epoch(model, tst_ldr, centroids_gpu, loss_type, tau=tau, hn_k=hn_k, optimizer=None, cov_params=_cov)
            
            scheduler.step(val)
            
            improved = val < best_val
            if improved:
                best_val = val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            val_fc, val_gtr = _val_coverage(model, val_ldr, centroids_gpu, k=20)
            row = {
                "epoch": epoch + 1, "train_loss": round(trl, 6), "val_loss": round(val, 6),
                "test_loss": round(tel, 6), "val_full_coverage@20": round(val_fc * 100, 2),
                "val_gt_recall@20": round(val_gtr * 100, 2),
                "lr": optimizer.param_groups[0]["lr"], "is_best": improved,
            }
            history.append(row)
            with open(hist_path, "a", newline="", encoding="utf-8") as _hf:   # incremental, crash-safe
                csv.DictWriter(_hf, fieldnames=_hist_cols).writerow(row)
                
            pbar.set_postfix_str(f"TrL={trl:.4f}, VaL={val:.4f}, TeL={tel:.4f}, vFC@20={val_fc*100:.1f}%" + (" ★=best" if improved else ""))
            pbar.update(1)
            
            if epochs_without_improvement >= 20:
                log.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
    if best_state is None:
        log.error(f"No valid checkpoint state found for {dataset_name} | {loss_type} | tau={tau} | hn_k={hn_k}. Skipping save.")
        return

    # Final-epoch weights (model was trained in place; best_state is the
    # lowest-val-loss snapshot). We save BOTH so downstream can benchmark
    # best-by-val vs final-epoch — the val objective doesn't perfectly track
    # FullCov, so the two can differ.
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ckpt_payload = {
        'model_state_dict': best_state,       # best by validation loss (early-stopping pick)
        'final_state_dict': final_state,      # last-epoch weights
        'best_val': float(best_val),
        'loss_type': loss_type,
        'tau': tau,
        'hn_k': hn_k,
        'input_dim': embed_dim,
        'hidden_dim': 512,
        'output_dim': embed_dim,
    }
    if is_coverage:
        ckpt_payload['coverage_params'] = cov_params
        ckpt_payload['lambda_cov'] = lambda_cov
    torch.save(ckpt_payload, output_path)
    log.info(f"Strict Checkpoint (best+final) mapped at: {output_path}")

    # history.csv was written incrementally during training (crash-safe above).
    # Write the run summary (best vs final metrics) alongside it.
    summary = {
        "dataset": dataset_name,
        "loss_type": loss_type,
        "tau": tau,
        "hn_k": hn_k,
        "limit": limit,
        "checkpoint": output_path,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val), 6),
        "early_stopped": epochs_without_improvement >= 20,
        "best_epoch_metrics": history[best_epoch - 1] if history and best_epoch >= 1 else {},
        "final_epoch_metrics": history[-1] if history else {},
    }
    if is_coverage:
        summary["lambda_cov"] = lambda_cov
        summary["coverage_params"] = cov_params
    with open(os.path.join(logs_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Training log saved -> {logs_dir}/ (history.csv + summary.json)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2wiki")
    parser.add_argument("--loss_type", type=str, default="info_nce_single",
                        choices=["info_nce_single", "info_nce_multi", "kl_div", "bce",
                                 "coverage_kl", "coverage_infonce", "coverage"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--hn_k", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap queries per split for a fast smoke run (0 = full corpus).")
    # Coverage-loss knobs (only used when --loss_type starts with "coverage").
    parser.add_argument("--lambda_cov", type=float, default=None,
                        help="Weight of the coverage term (default 0.5).")
    parser.add_argument("--cov_temperature", type=float, default=None,
                        help="Temperature of the coverage logits (default 0.05).")
    parser.add_argument("--target_topk", type=int, default=None,
                        help="K for the FullCov@K barrier (default 20).")
    parser.add_argument("--topk_weight", type=float, default=None,
                        help="Weight of the FullCov@K barrier term (default 0.35; 0 disables).")
    parser.add_argument("--topk_margin", type=float, default=None)
    parser.add_argument("--margin_weight", type=float, default=None)
    parser.add_argument("--cvar_fraction", type=float, default=None)
    parser.add_argument("--positive_aggregation", type=str, default=None,
                        choices=["cvar", "mean", "smoothmax"])
    args = parser.parse_args()

    cov_kwargs = dict(
        lambda_cov=args.lambda_cov,
        cov_temperature=args.cov_temperature,
        target_topk=args.target_topk,
        topk_weight=args.topk_weight,
        topk_margin=args.topk_margin,
        margin_weight=args.margin_weight,
        cvar_fraction=args.cvar_fraction,
        positive_aggregation=args.positive_aggregation,
    )

    datasets = ["squad", "metaqa", "musique", "2wiki"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        train(dataset_name=ds, loss_type=args.loss_type, epochs=args.epochs,
              tau=args.tau, hn_k=args.hn_k, limit=args.limit, **cov_kwargs)