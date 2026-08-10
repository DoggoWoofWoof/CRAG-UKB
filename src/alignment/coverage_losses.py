"""
Coverage-focused Level-1 partition-routing losses.
====================================================
Ported from Jigsaw (scripts/coverage_losses.py) and adapted to CRAG's setup:
a small MLP projects a frozen query embedding onto FROZEN FAISS partition
centroids. Because the centroids are not trained, the live-positive
re-encoding machinery from Jigsaw is unnecessary — we only backprop into the
query projection. The transferable innovation is preserved verbatim:

  1. CVaR-over-positives aggregation (worst-required-partition pressure).
     For a query with p gold partitions, CVaR_rho keeps the mean of the
     top-ceil(rho*p) LARGEST per-positive losses. For p<=4 and rho=0.25 this
     degenerates to pure min-over-positives (the single weakest gold), which
     is exactly the FullCov failure mode: one missed required partition sinks
     the whole answer.

  2. FullCov@K "top-K barrier": pushes every gold partition above the
     (K - p + 1)-th highest NEGATIVE logit, i.e. directly optimizes whether
     all required partitions land inside the top-K retrieval budget. This is
     the piece plain KL / mean-over-positives objectives lack.

Reference: Jigsaw FullCov@100 66% -> 82% (locked OGBN-Arxiv, McNemar p=4e-10).
See docs jigsaw_loss_and_retrieval_method.md and jigsaw_to_rag_transfer_playbook.md.
"""
import math

import torch
import torch.nn.functional as F


def _aggregate_positive_terms(
    per_positive_terms,
    positive_mask,
    mode="cvar",
    cvar_fraction=0.25,
    smoothmax_temperature=0.1,
):
    """Aggregate variable-count per-positive terms one query at a time.

    mode="mean"     -> mean over positives (the Jigsaw CONTROL baseline that
                       *loses* the ablation; provided only for parity/ablation).
    mode="cvar"     -> mean of the top-ceil(cvar_fraction*p) largest positive
                       terms (worst-positive pressure; the winning method).
    mode="smoothmax"-> temperature-smoothed max over positives (soft worst).
    """
    aggregated = []
    for row_terms, row_mask in zip(per_positive_terms, positive_mask):
        values = row_terms[row_mask]
        if values.numel() == 0:
            continue
        if mode == "mean":
            aggregated.append(values.mean())
        elif mode == "cvar":
            count = max(1, math.ceil(values.numel() * float(cvar_fraction)))
            aggregated.append(torch.topk(values, count).values.mean())
        elif mode == "smoothmax":
            temperature = max(1e-6, float(smoothmax_temperature))
            aggregated.append(
                temperature * torch.logsumexp(values / temperature, dim=0)
                - temperature * math.log(values.numel())
            )
        else:
            raise ValueError(
                f"Unknown positive aggregation {mode!r}; use mean, cvar, or smoothmax"
            )
    if not aggregated:
        return per_positive_terms.new_tensor(0.0)
    return torch.stack(aggregated).mean()


def partition_coverage_loss(
    projected,
    pids_list,
    all_centroids,
    temperature=0.05,
    target_topk=20,
    topk_bucket_size=10,
    topk_weight=0.35,
    topk_margin=0.0,
    margin_weight=0.25,
    positive_aggregation="cvar",
    cvar_fraction=0.25,
    smoothmax_temperature=0.1,
):
    """Multi-label coverage loss aligned with FullCov@K (worst-positive pressure).

    Args:
        projected:      (B, D) L2-normalized query projections.
        pids_list:      list (len B) of lists of gold partition ids per query.
        all_centroids:  (num_partitions, D) L2-normalized frozen partition centroids.
        temperature:    coverage-logit temperature (tau_cov; Jigsaw uses 0.05).
        target_topk:    the K in FullCov@K the barrier optimizes for.
        topk_bucket_size: bucket for queries with more golds than target_topk.
        topk_weight:    weight of the top-K barrier term (0.0 disables it).
        topk_margin:    additive slack on the barrier threshold.
        margin_weight:  weight of the hardest-negative softplus margin (Jigsaw 0.25).
        positive_aggregation: 'cvar' (default), 'mean' (control), or 'smoothmax'.
        cvar_fraction:  rho for CVaR aggregation.

    Returns:
        Scalar coverage loss (0.0 if no query in the batch has a valid positive).
    """
    device = projected.device
    batch_size = projected.shape[0]
    num_partitions = all_centroids.shape[0]

    logits = torch.matmul(projected, all_centroids.T) / temperature

    positive_mask = torch.zeros(
        batch_size, num_partitions, dtype=torch.bool, device=device
    )
    for row, partition_ids in enumerate(pids_list):
        for partition_id in partition_ids:
            if 0 <= int(partition_id) < num_partitions:
                positive_mask[row, int(partition_id)] = True

    valid = positive_mask.any(dim=1)
    if not valid.any():
        return projected.new_tensor(0.0, requires_grad=True)
    logits = logits[valid]
    positive_mask = positive_mask[valid]

    # ── Per-positive base term: CE + margin against the hardest negative ──
    log_probs = F.log_softmax(logits, dim=1)
    per_positive_ce = -log_probs

    negative_logits = logits.masked_fill(positive_mask, float("-inf"))
    hardest_negative = negative_logits.max(dim=1).values
    has_negative = torch.isfinite(hardest_negative)
    per_positive_margin = F.softplus(hardest_negative.unsqueeze(1) - logits)
    per_positive_margin = torch.where(
        has_negative.unsqueeze(1),
        per_positive_margin,
        torch.zeros_like(per_positive_margin),
    )
    loss = _aggregate_positive_terms(
        per_positive_ce + margin_weight * per_positive_margin,
        positive_mask,
        mode=positive_aggregation,
        cvar_fraction=cvar_fraction,
        smoothmax_temperature=smoothmax_temperature,
    )

    # ── FullCov@K top-K barrier ──
    if target_topk and target_topk > 0 and topk_weight > 0.0:
        barrier_rows = []
        barrier_masks = []
        for row_logits, row_positive_mask in zip(logits, positive_mask):
            positive_count = int(row_positive_mask.sum().item())
            if positive_count == 0:
                continue
            bucket_size = max(1, int(topk_bucket_size))
            effective_topk = (
                int(target_topk)
                if positive_count <= target_topk
                else ((positive_count + bucket_size - 1) // bucket_size) * bucket_size
            )
            negative_row = row_logits[~row_positive_mask]
            if negative_row.numel() == 0:
                continue
            kth_negative = min(
                max(1, effective_topk - positive_count + 1),
                negative_row.numel(),
            )
            threshold = torch.topk(negative_row, kth_negative).values[-1]
            barrier_rows.append(F.softplus(threshold + topk_margin - row_logits))
            barrier_masks.append(row_positive_mask)
        if barrier_rows:
            barrier_loss = _aggregate_positive_terms(
                torch.stack(barrier_rows),
                torch.stack(barrier_masks),
                mode=positive_aggregation,
                cvar_fraction=cvar_fraction,
                smoothmax_temperature=smoothmax_temperature,
            )
            loss = loss + topk_weight * barrier_loss

    return loss
