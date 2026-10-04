"""C12 — LIST-CONTEXT / DEEPSETS RESIDUAL MLP (residual on C8c). Reuses the C11a local query-candidate
interaction to produce z_i, then adds a PERMUTATION-INVARIANT leave-one-out set context (mean [+ max]) over
the top-20 window, a candidate x list interaction, an optional query condition hq, and a compact expert-set
summary. Pure MLP + pooling: NO attention, NO transformer, NO GNN, NO recurrence, NO new encoder/LLM.
Residual S_i = S_C8c_i + beta*delta_i with zero-init output (init ranking == C8c). C12a = mean context;
C12b = mean + max context."""
import torch, torch.nn as nn, torch.nn.functional as F

CONTRIB = slice(3, 8)   # E columns c_dense..c_relation (EFEAT order from _run_c11_prep)


def zero_init(layer):
    nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias); return layer


def set_context(feat, ridx, slot, nq, cnt, Nmax):
    """Leave-one-out mean & max of `feat` (M,F) grouped by query. ridx=row->query idx, slot=row->pos in query,
    cnt=(nq,) window sizes. Returns (loo_mean, loo_max) each (M,F). Permutation-invariant; no self-leakage."""
    F_ = feat.shape[1]
    ssum = feat.new_zeros(nq, F_).index_add_(0, ridx, feat)          # per-query sum
    denom = (cnt - 1).clamp(min=1).float().unsqueeze(1)              # leave-one-out count
    loo_mean = (ssum[ridx] - feat) / denom[ridx]
    pad = feat.new_full((nq, Nmax, F_), float("-inf"))
    pad[ridx, slot] = feat
    m1, a1 = pad.max(1)                                              # (nq,F)
    pad2 = pad.scatter(1, a1.unsqueeze(1), float("-inf"))
    m2 = pad2.max(1).values                                         # 2nd max
    is_self = a1[ridx] == slot.unsqueeze(1)
    loo_max = torch.where(is_self, m2[ridx], m1[ridx])
    loo_max = torch.where(torch.isinf(loo_max), torch.zeros_like(loo_max), loo_max)  # g==1 safety
    return loo_mean, loo_max


def list_summary(feat, ridx, slot, nq, cnt, Nmax):
    """Full-set mean/max/std of `feat` (M,F) per query (a list-level summary, same for all candidates)."""
    F_ = feat.shape[1]
    ssum = feat.new_zeros(nq, F_).index_add_(0, ridx, feat)
    ssq = feat.new_zeros(nq, F_).index_add_(0, ridx, feat * feat)
    c = cnt.float().clamp(min=1).unsqueeze(1)
    mean = ssum / c; var = (ssq / c - mean * mean).clamp(min=0); std = var.sqrt()
    pad = feat.new_full((nq, Nmax, F_), float("-inf")); pad[ridx, slot] = feat
    mx = pad.max(1).values; mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
    return torch.cat([mean, mx, std], dim=1)                        # (nq, 3F)


class C12(nn.Module):
    def __init__(self, D=1536, H=256, Z=128, ne=18, use_max=True, p=0.1):
        super().__init__()
        self.use_max = use_max; self.Z = Z; self.H = H
        self.Pq = nn.Linear(D, H); self.Pd = nn.Linear(D, H)
        lin = 4 * H + ne
        self.local = nn.Sequential(nn.Linear(lin, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(p),
                                   nn.Linear(256, Z), nn.GELU())     # z_i (C11a-style local interaction)
        ctx_mult = 3 if not use_max else 6                          # [mean, z*mean, |z-mean|] (+ max trio)
        u = Z + ctx_mult * Z + H + 3 * 5                            # z + interactions + hq + expert-set(15)
        self.ctx = nn.Sequential(nn.Linear(u, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(p),
                                 nn.Linear(256, 128), nn.GELU(), nn.Dropout(p))
        self.out = zero_init(nn.Linear(128, 1))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def z_of(self, hq, hd, E):
        return self.local(torch.cat([hq, hd, hq * hd, torch.abs(hq - hd), E], dim=-1))

    def delta(self, qv_u, ridx, slot, cnt, d, E, Nmax):
        """Grouped forward. qv_u=(nq,D) unique query vecs; ridx/slot/cnt group rows into query windows."""
        hq = F.gelu(self.Pq(qv_u))[ridx]                            # (M,H) compute per query, expand
        hd = F.gelu(self.Pd(d)); z = self.z_of(hq, hd, E)           # (M,Z)
        nq = qv_u.shape[0]
        loo_mean, loo_max = set_context(z, ridx, slot, nq, cnt, Nmax)
        ectx = list_summary(E[:, CONTRIB], ridx, slot, nq, cnt, Nmax)[ridx]   # (M,15)
        parts = [z, loo_mean, z * loo_mean, torch.abs(z - loo_mean)]
        if self.use_max:
            parts += [loo_max, z * loo_max, torch.abs(z - loo_max)]
        parts += [hq, ectx]
        return self.out(self.ctx(torch.cat(parts, dim=-1))).squeeze(-1)

    def self_leak(self, qv_u, ridx, slot, cnt, d, E, Nmax):
        """Diagnostic: fraction of (row,dim) where ordinary max pooling would pick the candidate itself."""
        with torch.no_grad():
            hq = F.gelu(self.Pq(qv_u))[ridx]; hd = F.gelu(self.Pd(d)); z = self.z_of(hq, hd, E)
            nq = qv_u.shape[0]; pad = z.new_full((nq, Nmax, z.shape[1]), float("-inf")); pad[ridx, slot] = z
            a1 = pad.max(1).indices
            return float((a1[ridx] == slot.unsqueeze(1)).float().mean())
