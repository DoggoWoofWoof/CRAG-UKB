"""C11 — LEARNED QUERY-CANDIDATE INTERACTION MLP (residual on C8c). Pure MLP: no attention, no transformer, no
GNN, no recurrence, no new encoder/LLM/cross-encoder. Reuses frozen gte_qwen query/node embeddings + compact
proven retrieval evidence. Residual scoring S = S_C8c + beta*Delta_MLP with the final layer zero-initialized so
that at init the ranking == C8c (avoids the C10b relearn-from-scratch collapse). Pairwise top-5-focused ranking
loss, multi-positive. C11a = single interaction MLP; C11b = K-offset mixture interaction MLP.
NEW_ENCODER_FORWARD_PASSES = 0, NEW_LLM_COMPONENTS = 0."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
EPS = 1e-9


def zero_init(layer):
    nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias); return layer


class C11a(nn.Module):
    """Projected interaction MLP: hq=Pq(q), hd=Pd(d); x=[hq,hd,hq*hd,|hq-hd|,E] -> scorer -> Delta."""
    def __init__(self, D=1536, H=256, ne=18, p=0.1):
        super().__init__()
        self.Pq = nn.Linear(D, H); self.Pd = nn.Linear(D, H)
        din = 4 * H + ne
        self.f1 = nn.Linear(din, 512); self.ln = nn.LayerNorm(512); self.f2 = nn.Linear(512, 128)
        self.out = zero_init(nn.Linear(128, 1)); self.drop = nn.Dropout(p)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def _score(self, hq, hd, E):
        x = torch.cat([hq, hd, hq * hd, torch.abs(hq - hd), E], dim=-1)
        h = self.drop(F.gelu(self.ln(self.f1(x)))); h = self.drop(F.gelu(self.f2(h)))
        return self.out(h).squeeze(-1)

    def delta(self, q, d, E):
        return self._score(F.gelu(self.Pq(q)), F.gelu(self.Pd(d)), E)

    def delta_grouped(self, qv_u, ridx, d, E):
        """qv_u=(nq,D) unique query vecs; ridx=(m,) query index per row. Compute hq once per query, expand."""
        hq = F.gelu(self.Pq(qv_u))[ridx]
        return self._score(hq, F.gelu(self.Pd(d)), E)

    def forward(self, q, d, E, s_c8c):
        return s_c8c + self.beta * self.delta(q, d, E)


class C11b(nn.Module):
    """Multi-offset interaction MLP. From hq predict K latent offsets o_k(q); u_k=hq+o_k; compare each u_k to hd
    via m_k=[u_k*hd, |u_k-hd|]; f(m_k) small MLP -> vector; query-dependent softmax mixture a_k(q) aggregates
    (mixture path) PLUS an elementwise max-over-k specialist path. Concatenate with hq/hd interaction + E."""
    def __init__(self, D=1536, H=256, K=4, ne=18, mf=64, p=0.1):
        super().__init__()
        self.K = K; self.H = H
        self.Pq = nn.Linear(D, H); self.Pd = nn.Linear(D, H)
        self.off = nn.Linear(H, K * H)          # K latent offsets from hq
        self.mix = nn.Linear(H, K)              # query-dependent mixture logits
        self.fm = nn.Sequential(nn.Linear(2 * H, mf), nn.GELU(), nn.Linear(mf, mf))  # f(m_k)
        din = 2 * mf + 4 * H + ne               # [mixture(mf), max-specialist(mf), hq,hd,hq*hd,|hq-hd|, E]
        self.f1 = nn.Linear(din, 512); self.ln = nn.LayerNorm(512); self.f2 = nn.Linear(512, 128)
        self.out = zero_init(nn.Linear(128, 1)); self.drop = nn.Dropout(p)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self._last_off = None

    def _score(self, hq, hd, E):
        B = hq.shape[0]; o = self.off(hq).view(B, self.K, self.H)     # (B,K,H)
        self._last_off = o
        u = hq.unsqueeze(1) + o                                       # (B,K,H)
        hd_e = hd.unsqueeze(1)
        m = torch.cat([u * hd_e, torch.abs(u - hd_e)], dim=-1)        # (B,K,2H)
        fmk = self.fm(m)                                              # (B,K,mf)
        self._last_fmk = fmk                                         # per-candidate per-offset match (for specialization diag)
        a = torch.softmax(self.mix(hq), dim=-1).unsqueeze(-1)         # (B,K,1)
        mix = (a * fmk).sum(1)                                        # (B,mf)
        mx = fmk.max(1).values                                       # (B,mf) specialist path
        x = torch.cat([mix, mx, hq, hd, hq * hd, torch.abs(hq - hd), E], dim=-1)
        h = self.drop(F.gelu(self.ln(self.f1(x)))); h = self.drop(F.gelu(self.f2(h)))
        return self.out(h).squeeze(-1)

    def delta(self, q, d, E):
        return self._score(F.gelu(self.Pq(q)), F.gelu(self.Pd(d)), E)

    def delta_grouped(self, qv_u, ridx, d, E):
        hq = F.gelu(self.Pq(qv_u))[ridx]
        return self._score(hq, F.gelu(self.Pd(d)), E)

    def offset_reg(self):
        o = self._last_off                                           # (B,K,H)
        return (o ** 2).sum(-1).mean()

    def offset_pairwise_cos(self):
        o = self._last_off                                          # (B,K,H)
        on = F.normalize(o, dim=-1); C = torch.bmm(on, on.transpose(1, 2))  # (B,K,K)
        K = self.K; iu = torch.triu_indices(K, K, 1)
        return C[:, iu[0], iu[1]].mean()

    def forward(self, q, d, E, s_c8c):
        return s_c8c + self.beta * self.delta(q, d, E)


# ------------------------------------------------------------------ pairwise top-5-focused ranking loss
def pair_loss(scores, y, base_rank_in_win, groups_idx):
    """scores,(m) final scores for one batch of queries flattened; y gold; base_rank_in_win = C8c within-window
    rank; groups_idx list of (start,end) per query. RankNet with pair weights up-weighting the exact C11 repair
    case (neg currently rank<5 AND pos currently rank 5-19). Multi-positive: every gold paired vs every non-gold."""
    total = scores.new_zeros(()); npair = 0
    for (a, b) in groups_idx:
        ys = y[a:b]; ss = scores[a:b]; br = base_rank_in_win[a:b]
        pos = torch.where(ys == 1)[0]; neg = torch.where(ys == 0)[0]
        if len(pos) == 0 or len(neg) == 0: continue
        sp = ss[pos].unsqueeze(1); sn = ss[neg].unsqueeze(0)                     # (P,N)
        diff = sp - sn
        w = torch.ones_like(diff)
        neg_top5 = (br[neg] < 5).float().unsqueeze(0); pos_mid = ((br[pos] >= 5) & (br[pos] < 20)).float().unsqueeze(1)
        w = w + 3.0 * (neg_top5 * pos_mid)                                      # up-weight the repair pairs
        total = total + (w * F.softplus(-diff)).sum(); npair += diff.numel()
    return total / max(npair, 1)
