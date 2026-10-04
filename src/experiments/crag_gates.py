"""CRAG L2 Phase-D — COOPERATIVE-KL FUSION (non-exclusive gates).

Central hypothesis (from STEP 5A): dense/splade/offset/relation/path carry COMPLEMENTARY gold evidence;
parameter-free cooperative max-rank already beats the learned LODO on all 5. The learned model must therefore
COMBINE experts while SUPPRESSING unsupported destructive contributions -- NOT select one expert. The softmax-KL
controller (Sigma alpha = 1) forces competition; we replace it with INDEPENDENT sigmoid gates (Sigma NOT constrained).

Two-level non-exclusive gating (no softmax simplex anywhere in the scoring path):
    h_e(q)     = sigmoid(H(query_state))              query-level expert reliability
    c_e(q,d)   = sigmoid(C_e(cand_state, cross-expert evidence, h))   candidate-level usefulness
    z_e(q,d)   = mask_e * h_e * c_e * phi_e           gated expert contribution (phi_e = [score, rank_pct, top5])
    S(q,d)     = MLP(raw phi, z, h, c, cross-expert interactions, corroboration, state, masks)

KL is AUXILIARY only: pi*_e = softmax(-L_e/tau) (bounded-InfoNCE expert quality). G1 = KL(pi* || normalize(h))
(teaches relative reliability WITHOUT forcing "splade up => offset down"); G2 = pairwise hinge (pi*_i>pi*_j => h_i>=h_j).
The scorer always consumes the ORIGINAL independent h_e, never the normalized copy.

Arms:  A (softmax-KL, via crag_fusion)  F0 (gates+rank, no interactions/KL)  F1 (+interactions/corroboration feats)
       G1 (F1 + normalized-gate KL)  G2 (F1 + pairwise teacher order)  H (coop max-rank base + bounded residual)
       I  (F1 + KL(G1) + corroboration non-destructive loss)  <- main architecture.

Metric = canonical recall@k = golds_in_topk / ng (kg_hybrid._qmet). Reuses the FROZEN canonical substrate + all
mask/dropout/teacher semantics from crag_fusion (nothing about the substrate or normalization changes).
"""
import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.experiments.kg_hybrid import (EXPERTS, NE, EXPERT_NAMES, _qmet, _load, _pooled_musd, _expert_loss,
                                       QEnc, Cand, PrepBatch)
from src.experiments import crag_fusion as CF

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# within-EXPERTS indices (EXPERTS=[0,1,3,5,6] -> dense,offset,splade,relation,path)
DENSE, OFFSET, SPLADE, RELATION, PATH = 0, 1, 2, 3, 4
COOP = [DENSE, OFFSET, SPLADE, RELATION]                 # cooperative max-rank uses dense,splade,offset,relation
PHI_CH = 3                                               # phi_e channels: [score, rank_pct, top5]
# cross-expert interaction pairs (within-EXPERTS idx)
PAIRS = [(DENSE, SPLADE), (DENSE, OFFSET), (SPLADE, OFFSET), (OFFSET, RELATION),
         (OFFSET, PATH), (RELATION, PATH), (SPLADE, RELATION), (DENSE, RELATION)]
GPAIRS = [(DENSE, SPLADE), (OFFSET, RELATION), (OFFSET, PATH), (SPLADE, RELATION)]   # gated interactions
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROP = CF.DROP_REGIMES
RES_SCALE = 0.15                                         # arm-H bounded residual scale (rank_pct space)


# ------------------------------------------------------------------ features
def cand_tensors(pb):
    """All per-candidate per-expert (n,NE) tensors + query-level (NE,) state. Everything finite.
    Rank features vectorized over all NE experts at once (one argsort, no python per-expert loop)."""
    n = pb.x_fuse.shape[0]
    score = pb.x_fuse[:, EXPERTS]                        # z, missing->0
    valid = pb.mask[:, EXPERTS]
    v = valid > 0.5
    xr = pb.x_rank[:, EXPERTS]                           # (n,NE) missing=-inf
    order = torch.argsort(xr, dim=0, descending=True)    # (n,NE)
    pos = torch.empty(n, NE, device=DEV)
    pos.scatter_(0, order, torch.arange(n, device=DEV).float().unsqueeze(1).expand(n, NE))
    z0 = torch.zeros(n, NE, device=DEV)
    rp = torch.where(v, 1.0 - pos / n, z0)
    rc = torch.where(v, 1.0 / (1.0 + pos), z0)
    t5 = ((pos < 5) & v).float(); t20 = ((pos < 20) & v).float(); t50 = ((pos < 50) & v).float()
    z, cnt = CF._valid_z(pb)
    ent, marg = CF._ent_marg(z, cnt)
    ag = CF._agree(z, cnt, CF.CONF_K)
    qav = pb.qavail[EXPERTS]
    vfrac = cnt / max(n, 1)
    qvec = torch.cat([marg, ent, ag, qav, vfrac])       # (5*NE,)
    return dict(n=n, score=score, valid=valid, rank_pct=rp, recip=rc, top5=t5, top20=t20, top50=t50,
                qav=qav, qvec=qvec)


def phi_of(T):
    """phi_e per candidate = [score, rank_pct, top5] -> (n,NE,PHI_CH)."""
    return torch.stack([T["score"], T["rank_pct"], T["top5"]], dim=-1)


def coop_maxrank(T):
    """CANONICAL parameter-free cooperative rank fusion (single source of truth for baseline AND arm-H base).
    Primary = max over {dense,splade,offset,relation} rank_pct (missing->0). Deterministic tie-break = + eps*sum
    of the same rank_pcts (corroboration: a candidate top-ranked by MULTIPLE experts outranks a single-expert top).
    This removes the argsort tie-break fragility that made a plain max ambiguous (every expert's #1 ties at 1.0)."""
    rp = T["rank_pct"][:, COOP]
    return rp.max(dim=1).values + 1e-6 * rp.sum(dim=1)       # (n,)


# ------------------------------------------------------------------ models
QDIM = 5 * NE                                            # qvec: margin+ent+agree+qav+vfrac per expert


class HNet(nn.Module):                                   # query-level independent gates h_e in [0,1]
    def __init__(s, h=32):
        super().__init__(); s.m = nn.Sequential(nn.Linear(QDIM, h), nn.GELU(), nn.Linear(h, NE))
    def forward(s, qvec):
        return torch.sigmoid(s.m(qvec))                 # (NE,) independent sigmoids (NOT softmax)


CIN = 7 + NE + NE + NE + QDIM                            # own(7)+cross score(NE)+cross rank_pct(NE)+h(NE)+qvec
COWN = 7                                                 # own per-expert channels


def _cnet_inputs(T, hb, qb):
    """Build the (NE, n, CIN) input stack for the candidate gates. hb=(n,NE) per-candidate h, qb=(n,QDIM)."""
    n = hb.shape[0]
    own = torch.stack([T["score"], T["rank_pct"], T["recip"], T["top5"], T["top20"], T["top50"], T["valid"]], -1)  # (n,NE,7)
    own = own.permute(1, 0, 2)                          # (NE,n,7)
    cs = T["score"].unsqueeze(0).expand(NE, n, NE)      # (NE,n,NE) cross scores (same for every head)
    cr = T["rank_pct"].unsqueeze(0).expand(NE, n, NE)
    hb3 = hb.unsqueeze(0).expand(NE, n, NE)
    qb3 = qb.unsqueeze(0).expand(NE, n, QDIM)
    return torch.cat([own, cs, cr, hb3, qb3], dim=2)    # (NE,n,CIN)


class CNet(nn.Module):                                   # candidate-level INDEPENDENT gates c_e(q,d), one head/expert (bmm)
    def __init__(s, h=32):
        super().__init__()
        s.W1 = nn.Parameter(torch.empty(NE, CIN, h)); s.b1 = nn.Parameter(torch.zeros(NE, 1, h))
        s.W2 = nn.Parameter(torch.empty(NE, h, 1)); s.b2 = nn.Parameter(torch.zeros(NE, 1, 1))
        nn.init.kaiming_uniform_(s.W1, a=5 ** 0.5); nn.init.kaiming_uniform_(s.W2, a=5 ** 0.5)
    def forward(s, T, hb, qb):
        XE = _cnet_inputs(T, hb, qb)                    # (NE,n,CIN)
        hid = F.gelu(torch.baddbmm(s.b1, XE, s.W1))     # (NE,n,h)
        out = torch.baddbmm(s.b2, hid, s.W2).squeeze(-1)  # (NE,n)
        return torch.sigmoid(out).permute(1, 0)         # (n,NE)


def _feat_dim(use_int):
    d = 7 * NE + NE * PHI_CH + NE + NE + QDIM            # raw(7NE)+z(NE*PHI)+h(NE)+c(NE)+qvec
    if use_int:
        d += len(PAIRS) + len(GPAIRS) + 4               # interactions + gated interactions + 4 corroboration feats
    return d


class Scorer(nn.Module):
    def __init__(s, fin, h=128):
        super().__init__()
        s.m = nn.Sequential(nn.Linear(fin, h), nn.GELU(), nn.Linear(h, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(s, X):
        return s.m(X).squeeze(-1)


def build_scorer_input(T, hb, cgate, qb, use_int):
    """Assemble the per-candidate scorer input (n,F). hb=(n,NE) per-candidate h, qb=(n,QDIM)."""
    n = hb.shape[0]
    phi = phi_of(T)                                     # (n,NE,PHI_CH)
    z = T["valid"].unsqueeze(-1) * hb.unsqueeze(-1) * cgate.unsqueeze(-1) * phi   # (n,NE,PHI_CH)
    raw = torch.cat([T["score"], T["rank_pct"], T["recip"], T["top5"], T["top20"], T["top50"], T["valid"]], 1)  # (n,7NE)
    blocks = [raw, z.reshape(n, -1), hb, cgate, qb]
    if use_int:
        sc = T["score"]
        inter = torch.stack([sc[:, i] * sc[:, j] for i, j in PAIRS], 1)              # (n,len(PAIRS))
        zsc = z[:, :, 0]                                                             # gated score channel
        ginter = torch.stack([zsc[:, i] * zsc[:, j] for i, j in GPAIRS], 1)
        rp = T["rank_pct"]
        corro = torch.stack([
            rp[:, DENSE] * rp[:, SPLADE],                                            # dense-splade rank agreement
            rp[:, OFFSET] * rp[:, RELATION],                                         # offset-relation agreement
            rp[:, OFFSET] * rp[:, PATH],                                             # offset-path agreement
            (torch.maximum(rp[:, DENSE], rp[:, SPLADE]) - torch.maximum(rp[:, OFFSET], rp[:, RELATION])).abs(),  # sem vs rel disagreement
        ], 1)
        blocks += [inter, ginter, corro]
    return torch.cat(blocks, 1)


# ------------------------------------------------------------------ losses
def _listnet(S, y):
    return -(F.log_softmax(S, 0) * (y / y.sum())).sum()


def _kl_normgate(hgate, pb):
    """G1: KL(pi* || normalize(h)) over AVAILABLE experts only. Teaches relative reliability, no simplex on scoring."""
    avail = pb.qavail[EXPERTS]
    Le = torch.stack([_expert_loss(pb.x_rank[:, c], pb.y) for c in EXPERTS])
    with torch.no_grad():
        tlog = torch.where(avail > 0.5, -Le, torch.full_like(Le, float("-inf")))
        pi = torch.nan_to_num(F.softmax(tlog, 0), nan=0.0)
    hh = hgate * avail
    hhat = hh / hh.sum().clamp(min=1e-6)
    m = avail > 0.5
    return (pi[m] * ((pi[m] + 1e-9).log() - (hhat[m] + 1e-9).log())).sum()


def _kl_pairwise(hgate, pb, margin=0.1):
    """G2: pairwise hinge -- if teacher says i better than j (lower loss), encourage h_i >= h_j (no sum constraint)."""
    avail = pb.qavail[EXPERTS]
    Le = torch.stack([_expert_loss(pb.x_rank[:, c], pb.y) for c in EXPERTS]).detach()
    loss = torch.zeros((), device=DEV); cnt = 0
    idx = [e for e in range(NE) if avail[e] > 0.5]
    for a in idx:
        for b in idx:
            if a == b:
                continue
            if Le[a] < Le[b] - 1e-6:                     # a strictly better
                loss = loss + F.relu(margin - (hgate[a] - hgate[b])); cnt += 1
    return loss / max(cnt, 1)


def _corro_loss(S, T, margin=0.5, npairs=64):
    """Non-destructive: candidates corroborated by >=2 experts (top5) should not be pushed far below weakly-supported
    (<=1 expert) ones without evidence. Symmetric across expert families (uses rank support, NO gold labels)."""
    supp = T["top5"].sum(1)                              # (n,) #experts with cand in their top5
    prot = torch.where(supp >= 2)[0]
    weak = torch.where(supp <= 1)[0]
    if len(prot) == 0 or len(weak) == 0:
        return torch.zeros((), device=DEV)
    pi = prot[torch.randint(len(prot), (npairs,), device=DEV)]
    wj = weak[torch.randint(len(weak), (npairs,), device=DEV)]
    return F.relu(margin - (S[pi] - S[wj])).mean()


# ------------------------------------------------------------------ arm config
ARMS = {
    "F0": dict(use_int=False, kl=None, corro=False, residual=False),
    "F1": dict(use_int=True,  kl=None, corro=False, residual=False),
    "G1": dict(use_int=True,  kl="g1", corro=False, residual=False),
    "G2": dict(use_int=True,  kl="g2", corro=False, residual=False),
    "H":  dict(use_int=True,  kl=None, corro=False, residual=True),
    "I":  dict(use_int=True,  kl="g1", corro=True,  residual=False),
}


class GateModel(nn.Module):
    def __init__(s, use_int):
        super().__init__()
        s.h = HNet(); s.c = CNet(); s.scorer = Scorer(_feat_dim(use_int))
        s.use_int = use_int
    def forward(s, T):
        n = T["n"]
        hg = s.h(T["qvec"])                              # (NE,) per query
        hb = hg.unsqueeze(0).expand(n, -1)              # (n,NE)
        qb = T["qvec"].unsqueeze(0).expand(n, -1)       # (n,QDIM)
        cg = s.c(T, hb, qb)
        X = build_scorer_input(T, hb, cg, qb, s.use_int)
        return s.scorer(X), hg, cg


def train_gate(seed, TR, arm, epochs, p_drop, lam_kl=1.0, gam_corro=0.3, res_pen=0.05):
    cfg = ARMS[arm]
    torch.manual_seed(seed); np.random.seed(seed)
    model = GateModel(cfg["use_int"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rnames = list(DROP.keys())
    for ep in range(epochs):
        for j in np.random.permutation(len(TR)):
            pb = TR[j]
            pbc = CF.apply_drop(pb, DROP[rnames[np.random.randint(len(rnames))]]) if (p_drop > 0 and np.random.rand() < p_drop) else pb
            if float(pbc.qavail[EXPERTS].sum()) == 0:
                continue
            T = cand_tensors(pbc)
            s_out, hg, cg = model(T)
            S = (coop_maxrank(T) + RES_SCALE * torch.tanh(s_out)) if cfg["residual"] else s_out
            loss = _listnet(S, pb.y)
            if cfg["residual"]:
                loss = loss + res_pen * (torch.tanh(s_out) ** 2).mean()
            if cfg["kl"] == "g1":
                loss = loss + lam_kl * _kl_normgate(hg, pbc)
            elif cfg["kl"] == "g2":
                loss = loss + lam_kl * _kl_pairwise(hg, pbc)
            if cfg["corro"]:
                loss = loss + gam_corro * _corro_loss(S, T)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def _build_pack(pbs):
    """Concatenate a minibatch of (already dropout-applied) queries into one segment-indexed pack.
    Returns bigT (Ntot,NE tensors), qb (Ntot,QDIM), seg (Ntot,), y (Ntot,), qvecs (B,QDIM), per-query (pb,T,slice)."""
    Ts = [cand_tensors(pb) for pb in pbs]
    keys = ["score", "valid", "rank_pct", "recip", "top5", "top20", "top50"]
    bigT = {k: torch.cat([T[k] for T in Ts], 0) for k in keys}
    Ntot = bigT["score"].shape[0]
    bigT["n"] = Ntot
    qvecs = torch.stack([T["qvec"] for T in Ts], 0)          # (B,QDIM)
    seg = torch.cat([torch.full((T["n"],), i, dtype=torch.long, device=DEV) for i, T in enumerate(Ts)])
    qb = qvecs[seg]                                          # (Ntot,QDIM)
    y = torch.cat([pb.y for pb in pbs], 0)
    per = []
    off = 0
    for pb, T in zip(pbs, Ts):
        per.append((pb, T, slice(off, off + T["n"]))); off += T["n"]
    return bigT, qb, seg, y, qvecs, per


def _seg_listnet(S, y, seg, B):
    """Segment (per-query) ListNet, averaged over the B queries in the minibatch."""
    segmax = torch.full((B,), float("-inf"), device=DEV).scatter_reduce(0, seg, S, "amax", include_self=True)
    Ssh = S - segmax[seg]
    expsum = torch.zeros(B, device=DEV).index_add_(0, seg, Ssh.exp())
    logZ = expsum.clamp(min=1e-12).log() + segmax                     # (B,)
    logp = S - logZ[seg]
    ysum = torch.zeros(B, device=DEV).index_add_(0, seg, y).clamp(min=1e-6)
    target = y / ysum[seg]
    return -(target * logp).sum() / B


def train_gate_batched(seed, TR, arm, epochs, p_drop, bs=16, lam_kl=1.0, gam_corro=0.3, res_pen=0.05):
    """Batched trainer: one forward+step per minibatch of `bs` queries via segment-softmax. Same model/losses as
    train_gate; minibatch SGD (not per-query steps) -> designed for GPU on the full-5/LODO runs."""
    cfg = ARMS[arm]
    torch.manual_seed(seed); np.random.seed(seed)
    model = GateModel(cfg["use_int"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rnames = list(DROP.keys())
    for ep in range(epochs):
        perm = np.random.permutation(len(TR))
        for b0 in range(0, len(perm), bs):
            idx = perm[b0:b0 + bs]
            pbs = []
            for j in idx:
                pb = TR[j]
                pbc = CF.apply_drop(pb, DROP[rnames[np.random.randint(len(rnames))]]) if (p_drop > 0 and np.random.rand() < p_drop) else pb
                if float(pbc.qavail[EXPERTS].sum()) > 0:
                    pbs.append(pbc)
            if not pbs:
                continue
            B = len(pbs)
            bigT, qb, seg, y, qvecs, per = _build_pack(pbs)
            hgs = model.h(qvecs)                                       # (B,NE) per query
            hb = hgs[seg]                                              # (Ntot,NE)
            cg = model.c(bigT, hb, qb)
            s_out = model.scorer(build_scorer_input(bigT, hb, cg, qb, cfg["use_int"]))
            S = (coop_maxrank(bigT) + RES_SCALE * torch.tanh(s_out)) if cfg["residual"] else s_out
            loss = _seg_listnet(S, y, seg, B)
            if cfg["residual"]:
                loss = loss + res_pen * (torch.tanh(s_out) ** 2).mean()
            for i, (pbc, T, sl) in enumerate(per):
                if cfg["kl"] == "g1":
                    loss = loss + (lam_kl / B) * _kl_normgate(hgs[i], pbc)
                elif cfg["kl"] == "g2":
                    loss = loss + (lam_kl / B) * _kl_pairwise(hgs[i], pbc)
                if cfg["corro"]:
                    loss = loss + (gam_corro / B) * _corro_loss(S[sl], T)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_gate(model, TE, datasets, arm):
    cfg = ARMS[arm]
    recs = {ds: [] for ds in datasets}
    model.eval()
    with torch.no_grad():
        for pb in TE:
            if pb.ds not in recs:
                continue
            T = cand_tensors(pb)
            s_out, hg, cg = model(T)
            S = (coop_maxrank(T) + RES_SCALE * torch.tanh(s_out)) if cfg["residual"] else s_out
            ry = pb.y.cpu().numpy()[np.argsort(-S.cpu().numpy(), kind="stable")][:50].astype(np.uint8)
            recs[pb.ds].append((ry, pb.ng))
    return recs


def _agg(recs):
    out = {}
    for ds, rs in recs.items():
        if not rs:
            continue
        ms = [_qmet(ry, ng) for ry, ng in rs]
        out[ds] = {k: round(float(np.mean([m[k] for m in ms])) * 100, 2) for k in ms[0]}
    return out


def mean_gates(model, TE, datasets):
    model.eval(); acc = {ds: [] for ds in datasets}
    with torch.no_grad():
        for pb in TE:
            if pb.ds not in acc:
                continue
            hg = model.h(cand_tensors(pb)["qvec"])
            acc[pb.ds].append(hg.cpu().numpy())
    return {ds: {EXPERT_NAMES[e]: round(float(np.mean([a[e] for a in acc[ds]])), 3) for e in range(NE)}
            for ds in datasets if acc[ds]}


# ------------------------------------------------------------------ parameter-free anchors + realistic oracles
def _recall5_perquery(S, y, ng):
    top = np.argsort(-S, kind="stable")[:5]
    return float(y[top].sum()) / max(float(ng), 1.0)


def anchors_and_oracles(TE, datasets, klmlp=None):
    """Standalone experts, cooperative max-rank, and REALISTIC per-query oracles (5G).
    klmlp = (qenc,cand) softmax-KL model for the coop-vs-KL and 5-way oracles; None -> skip those."""
    per = {ds: {"dense": [], "splade": [], "offset": [], "coop": [], "klmlp": []} for ds in datasets}
    for pb in TE:
        ds = pb.ds
        if ds not in per:
            continue
        y = pb.y.cpu().numpy(); ng = pb.ng
        T = cand_tensors(pb)
        xr = pb.x_rank.cpu().numpy()
        per[ds]["dense"].append(_recall5_perquery(np.where(np.isfinite(xr[:, 0]), xr[:, 0], -1e30), y, ng))
        per[ds]["splade"].append(_recall5_perquery(np.where(np.isfinite(xr[:, 3]), xr[:, 3], -1e30), y, ng))
        per[ds]["offset"].append(_recall5_perquery(np.where(np.isfinite(xr[:, 1]), xr[:, 1], -1e30), y, ng))
        per[ds]["coop"].append(_recall5_perquery(coop_maxrank(T).cpu().numpy(), y, ng))
        if klmlp is not None:
            qenc, cand = klmlp
            with torch.no_grad():
                st = CF.state_of(pb, CF.CONF_K)
                alpha, _ = CF._controller(qenc, pb, st)
                S = cand(CF._cand_input(pb, alpha, st)).cpu().numpy()
            per[ds]["klmlp"].append(_recall5_perquery(S, y, ng))
    out = {}
    for ds in datasets:
        d = {k: np.array(v) for k, v in per[ds].items() if v}
        r = {"dense": round(100 * d["dense"].mean(), 2), "splade": round(100 * d["splade"].mean(), 2),
             "offset": round(100 * d["offset"].mean(), 2), "coop_maxrank": round(100 * d["coop"].mean(), 2)}
        r["oracle_dense_splade_offset"] = round(100 * np.maximum.reduce([d["dense"], d["splade"], d["offset"]]).mean(), 2)
        if "klmlp" in d:
            r["klmlp"] = round(100 * d["klmlp"].mean(), 2)
            r["oracle_coop_vs_klmlp"] = round(100 * np.maximum(d["coop"], d["klmlp"]).mean(), 2)
            r["oracle_5way"] = round(100 * np.maximum.reduce(
                [d["dense"], d["splade"], d["offset"], d["coop"], d["klmlp"]]).mean(), 2)
        out[ds] = r
    return out


# ------------------------------------------------------------------ driver
def run(datasets, arms, epochs, seeds, p_drop, out_path, batched=False, bs=16):
    data = _load(datasets, DEV)
    mu, sd = _pooled_musd(data, datasets, DEV, "train")
    TR = [CF.prepbatch(r, mu, sd, DEV, ds) for ds in datasets for r in data[ds]["train"]]
    TE = [CF.prepbatch(r, mu, sd, DEV, ds) for ds in datasets for r in data[ds]["test"]]
    trainer = (lambda s, arm: train_gate_batched(s, TR, arm, epochs, p_drop, bs)) if batched \
        else (lambda s, arm: train_gate(s, TR, arm, epochs, p_drop))
    log.info("loaded %d train / %d test over %s (device=%s, batched=%s bs=%d)",
             len(TR), len(TE), datasets, DEV, batched, bs)

    # baseline A (softmax-KL) once -- also used for realistic oracles
    log.info("[A softmax-KL] training reference ...")
    qenc, cand = CF.train_crag(0, TR, DEV, epochs, 1.0, 1.0, CF.CONF_K, 32, DROP, p_drop)
    A = _agg(CF.eval_crag(qenc, cand, TE, datasets))

    oracles = anchors_and_oracles(TE, datasets, klmlp=(qenc, cand))
    log.info("anchors+oracles: %s", json.dumps(oracles))

    results = {"A_softmaxKL": {"in_dist": A, "seeds": 1}}
    gates = {}
    for arm in arms:
        seed_metrics = []
        model = None
        for s in range(seeds):
            model = trainer(s, arm)
            seed_metrics.append(_agg(eval_gate(model, TE, datasets, arm)))
        agg = {ds: {k: [round(float(np.mean([sm[ds][k] for sm in seed_metrics])), 2),
                        round(float(np.std([sm[ds][k] for sm in seed_metrics])), 2)] for k in seed_metrics[0][ds]}
               for ds in datasets}
        results[arm] = {"in_dist": agg, "seeds": seeds}
        gates[arm] = mean_gates(model, TE, datasets)     # last-seed gate means
        log.info("[%s] R@5 %s", arm, {ds: agg[ds]["R@5"][0] for ds in datasets})

    payload = {"_config": {"datasets": list(datasets), "arms": arms, "epochs": epochs, "seeds": seeds,
                           "p_drop": p_drop, "metric": "recall@k = golds_in_topk/ng"},
               "anchors_oracles": oracles, "results": results, "mean_query_gates": gates,
               "coop_maxrank_baseline_R@5": {"webqsp": 35.69, "metaqa": 83.04, "2wiki_clean": 89.63,
                                             "musique_clean": 75.81, "hotpotqa_clean": 77.05}}
    json.dump(payload, open(out_path, "w"), indent=2)
    log.info("-> %s", out_path)
    return payload


# =================================================================================================
# CONFIRMATION HARNESS (3-arm A/H/I): TRAIN-ONLY validation splits, fixed max-optimizer-step budget,
# worst-domain (robust) checkpoint selection, curve + gate-trajectory logging. TEST is touched ONCE,
# after the checkpoint is already chosen from source-only validation. Same protocol/step-granularity
# across A/H/I (1 optimizer step = 1 query for every arm -> no arm gets more optimization).
# =================================================================================================
HQ = 32                                                  # A controller hidden width (matches screening)


def make_splits(data, datasets, frac_val=0.15, split_seed=1234):
    """Deterministic per-dataset opt/val split over TRAIN indices ONLY. Identical across arms/seeds/containers
    (depends only on split_seed + dataset sizes, never on the model seed). TEST is never touched here."""
    splits = {}
    rng = np.random.RandomState(split_seed)
    for ds in datasets:
        n = len(data[ds]["train"])
        idx = rng.permutation(n)
        nval = max(1, int(round(frac_val * n)))
        val = sorted(idx[:nval].tolist()); opt = sorted(idx[nval:].tolist())
        splits[ds] = {"opt": opt, "val": val, "n_train": n, "n_opt": len(opt), "n_val": len(val)}
    return splits


def _prep_from_idx(data, datasets, splits, key, mu, sd):
    return [CF.prepbatch(data[ds]["train"][i], mu, sd, DEV, ds) for ds in datasets for i in splits[ds][key]]


def _val_refs(TE_val, datasets):
    """Parameter-free per-dataset references computed ON THE VALIDATION SPLIT (source-only, no model, no test):
    base = cooperative max-rank R@5; ceil = per-query oracle max(dense,splade,offset) R@5. Robust normalization
    W_norm = (R - base)/max(ceil-base, eps): 0 = matches the coop baseline, 1 = reaches the parameter-free
    realistic ceiling. min over datasets -> emphasizes the worse domain; requires beating coop on the worse one."""
    o = anchors_and_oracles(TE_val, datasets, klmlp=None)
    refs = {}
    for ds in datasets:
        base = o[ds]["coop_maxrank"]; ceil = o[ds]["oracle_dense_splade_offset"]
        refs[ds] = {"base": base, "ceil": ceil, "scale": max(ceil - base, 1e-6),
                    "standalone_best": max(o[ds]["dense"], o[ds]["splade"], o[ds]["offset"])}
    return refs


def _robust(valR, refs, datasets):
    norms = {ds: (valR[ds] - refs[ds]["base"]) / refs[ds]["scale"] for ds in datasets}
    return min(norms.values()), norms


def _clone_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _val_R5(eval_recs, datasets):
    agg = _agg(eval_recs)
    return {ds: agg[ds]["R@5"] for ds in datasets if ds in agg}


def _run_confirm_arm(seed, arm, TR_opt, TE_val, TE_test, datasets, refs, max_steps, val_every, p_drop,
                     ckpt_sink=None):
    """Train one arm under the step-budget/robust-checkpoint protocol. Returns (test_metrics, curve, meta).
    If ckpt_sink is a list, every validation-step's {step, state} is appended (frozen per-step checkpoints for
    the validation-proxy trajectory audit -- no effect on training)."""
    import time
    torch.manual_seed(seed); np.random.seed(seed)
    is_A = (arm == "A")
    rnames = list(DROP.keys())
    if is_A:
        qenc = QEnc(CF.UIN, HQ).to(DEV); cand = Cand(CF.CAND_IN).to(DEV)
        opt = torch.optim.Adam(list(cand.parameters()) + list(qenc.parameters()), lr=1e-3)
        modules = {"qenc": qenc, "cand": cand}
    else:
        cfg = ARMS[arm]
        model = GateModel(cfg["use_int"]).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        modules = {"model": model}

    def validate():
        if is_A:
            recs = CF.eval_crag(qenc, cand, TE_val, datasets)
        else:
            recs = eval_gate(model, TE_val, datasets, arm)
        return _val_R5(recs, datasets)

    def gates_now():
        if is_A:
            return {d: CF.mean_alpha(qenc, TE_val, [d]) for d in datasets}
        return mean_gates(model, TE_val, datasets)

    curve = []
    best = {"robust": -1e18, "step": 0, "state": None, "valR": None}
    t0 = time.time(); step = 0; done = False
    while not done:
        for j in np.random.permutation(len(TR_opt)):
            pb = TR_opt[j]
            pbc = CF.apply_drop(pb, DROP[rnames[np.random.randint(len(rnames))]]) if (p_drop > 0 and np.random.rand() < p_drop) else pb
            if float(pbc.qavail[EXPERTS].sum()) == 0:
                continue
            if is_A:
                state = CF.state_of(pbc, CF.CONF_K)
                alpha, alogit = CF._controller(qenc, pbc, state)
                S = cand(CF._cand_input(pbc, alpha, state))
                loss = -(F.log_softmax(S, 0) * (pb.y / pb.y.sum())).sum()
                with torch.no_grad():
                    avail, Le = CF._teacher(pbc)
                    tlog = torch.where(avail > 0.5, -Le, torch.full_like(Le, float("-inf")))
                    pi = torch.nan_to_num(F.softmax(tlog, 0), nan=0.0)
                la = F.log_softmax(alogit, 0)
                loss = loss + (pi * ((pi + 1e-9).log() - la))[avail > 0.5].sum()   # lam=1.0
            else:
                T = cand_tensors(pbc)
                s_out, hg, cg = model(T)
                S = (coop_maxrank(T) + RES_SCALE * torch.tanh(s_out)) if cfg["residual"] else s_out
                loss = _listnet(S, pb.y)
                if cfg["residual"]:
                    loss = loss + 0.05 * (torch.tanh(s_out) ** 2).mean()
                if cfg["kl"] == "g1":
                    loss = loss + _kl_normgate(hg, pbc)
                elif cfg["kl"] == "g2":
                    loss = loss + _kl_pairwise(hg, pbc)
                if cfg["corro"]:
                    loss = loss + 0.3 * _corro_loss(S, T)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1

            if step % val_every == 0 or step >= max_steps:
                valR = validate()
                robust, norms = _robust(valR, refs, datasets)
                rec = {"step": step, "val_R5": {d: round(valR[d], 2) for d in datasets},
                       "robust": round(robust, 4), "norms": {d: round(norms[d], 3) for d in datasets}}
                if arm == "I":
                    rec["gates"] = gates_now()
                curve.append(rec)
                if ckpt_sink is not None:
                    ckpt_sink.append({"step": step, "state": {k: _clone_state(m) for k, m in modules.items()}})
                if robust > best["robust"]:
                    best = {"robust": robust, "step": step, "valR": valR,
                            "state": {k: _clone_state(m) for k, m in modules.items()}}
                if step >= max_steps:
                    done = True; break
    runtime = round(time.time() - t0, 1)

    # restore the source-validation-selected checkpoint, then touch TEST exactly once
    for k, m in modules.items():
        m.load_state_dict(best["state"][k])
    if is_A:
        test = _agg(CF.eval_crag(qenc, cand, TE_test, datasets))
        diag = None
    else:
        test = _agg(eval_gate(model, TE_test, datasets, arm))
        diag = i_diagnostics(model, TE_test, datasets) if arm == "I" else None
    meta = {"best_step": best["step"], "best_robust": round(best["robust"], 4),
            "best_val_R5": {d: round(best["valR"][d], 2) for d in datasets},
            "total_steps": step, "val_checkpoints": len(curve), "runtime_s": runtime,
            "best_ckpt_gates": gates_now() if arm == "I" else None,
            "test_diagnostics": diag}
    return test, curve, meta


def i_diagnostics(model, TE, datasets):
    """Arm-I candidate-level diagnostics on the given split: mean candidate gate c_e per dataset, and gold
    rank movement vs the parameter-free coop max-rank (does I lift or bury gold candidates?)."""
    model.eval()
    cg_acc = {ds: [] for ds in datasets}
    gold_dpos = {ds: [] for ds in datasets}     # (coop gold rank) - (I gold rank); >0 => I moved gold UP
    with torch.no_grad():
        for pb in TE:
            if pb.ds not in cg_acc:
                continue
            T = cand_tensors(pb)
            s_out, hg, cg = model(T)
            cg_acc[pb.ds].append(cg.mean(0).cpu().numpy())
            y = pb.y.cpu().numpy()
            if y.sum() == 0:
                continue
            si = np.argsort(-s_out.cpu().numpy(), kind="stable")
            ci = np.argsort(-coop_maxrank(T).cpu().numpy(), kind="stable")
            rank_i = np.empty(len(y), dtype=np.int64); rank_i[si] = np.arange(len(y))
            rank_c = np.empty(len(y), dtype=np.int64); rank_c[ci] = np.arange(len(y))
            g = np.where(y > 0.5)[0]
            gold_dpos[pb.ds].append(float(np.mean(rank_c[g] - rank_i[g])))
    out = {}
    for ds in datasets:
        if cg_acc[ds]:
            out[ds] = {"mean_cand_gate": {EXPERT_NAMES[e]: round(float(np.mean([a[e] for a in cg_acc[ds]])), 3)
                                          for e in range(NE)},
                       "gold_rank_lift_vs_coop": round(float(np.mean(gold_dpos[ds])), 2) if gold_dpos[ds] else None}
    return out


def confirm(datasets, arms, seeds, max_steps, val_every, p_drop, frac_val, split_seed, out_path):
    """3-arm confirmation. arms subset of {A,H,I}. TRAIN-only opt/val split (persisted). Same split every arm/seed.
    Robust worst-domain checkpoint from val; ONE test eval per selected checkpoint; 3 seeds -> mean +/- std."""
    data = _load(datasets, DEV)
    mu, sd = _pooled_musd(data, datasets, DEV, "train")      # frozen present-only pooled TRAIN stats (unchanged)
    splits = make_splits(data, datasets, frac_val, split_seed)
    TR_opt = _prep_from_idx(data, datasets, splits, "opt", mu, sd)
    TE_val = _prep_from_idx(data, datasets, splits, "val", mu, sd)
    TE_test = [CF.prepbatch(r, mu, sd, DEV, ds) for ds in datasets for r in data[ds]["test"]]
    refs = _val_refs(TE_val, datasets)
    log.info("confirm: opt=%d val=%d test=%d over %s (device=%s) | splits=%s",
             len(TR_opt), len(TE_val), len(TE_test), datasets, DEV,
             {ds: (splits[ds]["n_opt"], splits[ds]["n_val"]) for ds in datasets})
    log.info("confirm val-refs (source-only): %s", json.dumps(refs))

    results = {}
    curves = {}
    for arm in arms:
        seed_tests = []; seed_meta = []
        for s in range(seeds):
            test, curve, meta = _run_confirm_arm(s, arm, TR_opt, TE_val, TE_test, datasets, refs,
                                                 max_steps, val_every, p_drop)
            seed_tests.append(test); seed_meta.append(meta)
            curves.setdefault(arm, {})[f"seed{s}"] = curve
            log.info("[%s s%d] TEST R@5 %s | best_step=%d best_robust=%.3f runtime=%.0fs",
                     arm, s, {d: test[d]["R@5"] for d in datasets}, meta["best_step"], meta["best_robust"], meta["runtime_s"])
        agg = {ds: {k: [round(float(np.mean([st[ds][k] for st in seed_tests])), 2),
                        round(float(np.std([st[ds][k] for st in seed_tests])), 2)] for k in seed_tests[0][ds]}
               for ds in datasets}
        results[arm] = {"test": agg,
                        "best_step": [m["best_step"] for m in seed_meta],
                        "best_robust": [m["best_robust"] for m in seed_meta],
                        "runtime_s": [m["runtime_s"] for m in seed_meta],
                        "best_ckpt_gates": [m["best_ckpt_gates"] for m in seed_meta] if arm == "I" else None,
                        "test_diagnostics": [m["test_diagnostics"] for m in seed_meta] if arm == "I" else None,
                        "seeds": seeds}
        log.info("[%s] TEST mean R@5 %s", arm, {d: results[arm]["test"][d]["R@5"] for d in datasets})

    payload = {"_config": {"kind": "confirmation_3arm", "datasets": list(datasets), "arms": arms, "seeds": seeds,
                           "max_steps": max_steps, "val_every": val_every, "p_drop": p_drop,
                           "frac_val": frac_val, "split_seed": split_seed,
                           "protocol": "per-query step budget (1 step = 1 query, identical across arms); "
                                       "robust=min_ds (valR5-coopbase)/(oracle_dse-coopbase) on the TRAIN-only val split; "
                                       "checkpoint chosen on val ROBUST, TEST evaluated once.",
                           "metric": "recall@k = golds_in_topk/ng"},
               "splits": splits, "val_refs": refs, "results": results, "curves": curves,
               "frozen_baselines": {"standalone": {"webqsp": 39.30, "metaqa": 77.08},
                                    "coop_maxrank": {"webqsp": 35.69, "metaqa": 83.04},
                                    "old_comparable_indist": {"webqsp": 54.8}}}
    json.dump(payload, open(out_path, "w"), indent=2)
    log.info("-> %s", out_path)
    return payload


# =================================================================================================
# VALIDATION-PROXY AUDIT: score FROZEN per-step I checkpoints on source-only proxies (iid / injected
# hard-negatives / expert-conflict slice / structural strata) AND ONCE on canonical test, then measure
# Spearman(proxy-curve, test-curve) -- which source-only proxy tracks the known collapse the IID val misses.
# NO architecture tuning; the model is the frozen confirm-config I. Test is used once, retrospectively.
# =================================================================================================
def make_pb(x, y, ng, m, mu, sd, ds):
    z = (x - mu) / sd
    x_fuse = torch.where(m > 0.5, z, torch.zeros_like(z))
    x_rank = torch.where(m > 0.5, z, torch.full_like(z, float("-inf")))
    qavail = (m.sum(0) > 0).float()
    return PrepBatch(raw=x, mask=m, qavail=qavail, x_fuse=x_fuse, x_rank=x_rank, y=y, ng=int(ng), ds=ds)


def _bank_from(TR):
    bank = {}
    for pb in TR:
        for i in torch.where(pb.y > 0.5)[0].tolist():
            bank.setdefault(pb.ds, []).append((pb.raw[i], pb.mask[i]))
    return bank


def _proxy_inject(pb, bank, mu, sd, K, rng):
    rows = bank.get(pb.ds, [])
    if not rows:
        return pb
    pick = rng.randint(0, len(rows), size=min(K, len(rows)))
    x = torch.cat([pb.raw] + [rows[i][0].unsqueeze(0) for i in pick], 0)
    m = torch.cat([pb.mask] + [rows[i][1].unsqueeze(0) for i in pick], 0)
    y = torch.cat([pb.y, torch.zeros(len(pick), device=pb.y.device)])
    return make_pb(x, y, pb.ng, m, mu, sd, pb.ds)


def _tops(pb):
    T = cand_tensors(pb); rp = T["rank_pct"]
    o = rp[:, OFFSET]; s = rp[:, SPLADE]
    ot = int(o.argmax()) if float(o.max()) > 0 else -1
    st = int(s.argmax()) if float(s.max()) > 0 else -2
    return ot, st


def _is_conflict(pb):
    ot, st = _tops(pb); return ot != st


def _struct(pb):
    ot, st = _tops(pb); y = pb.y
    if ot >= 0 and float(y[ot]) > 0.5:
        return "offset_dom"                              # offset's #1 is a gold (offset trivially right)
    if st >= 0 and float(y[st]) > 0.5:
        return "splade_sup"                              # splade's #1 is a gold
    return "other"


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return None
    ra = np.argsort(np.argsort(a[ok])).astype(float); rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return round(float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12)), 3)


def _r5(model, vset, datasets):
    ag = _agg(eval_gate(model, vset, datasets, "I"))
    return {ds: (ag[ds]["R@5"] if ds in ag else float("nan")) for ds in datasets}


def valproxy_run(datasets, max_steps, val_every, p_drop, frac_val, split_seed, out_path, seed=0, k_inject=25):
    data = _load(datasets, DEV)
    mu, sd = _pooled_musd(data, datasets, DEV, "train")
    splits = make_splits(data, datasets, frac_val, split_seed)
    TR_opt = _prep_from_idx(data, datasets, splits, "opt", mu, sd)
    TE_val = _prep_from_idx(data, datasets, splits, "val", mu, sd)
    TE_test = [CF.prepbatch(r, mu, sd, DEV, ds) for ds in datasets for r in data[ds]["test"]]
    refs = _val_refs(TE_val, datasets)
    bank = _bank_from(TR_opt)
    rng = np.random.RandomState(123)
    prox = {"iid": TE_val,
            "inject": [_proxy_inject(pb, bank, mu, sd, k_inject, rng) for pb in TE_val],
            "conflict": [pb for pb in TE_val if _is_conflict(pb)],
            "offset_dom": [pb for pb in TE_val if _struct(pb) == "offset_dom"],
            "splade_sup": [pb for pb in TE_val if _struct(pb) == "splade_sup"]}
    log.info("valproxy slice sizes: %s", {p: len(v) for p, v in prox.items()})

    ck = []
    test_final, curve, meta = _run_confirm_arm(seed, "I", TR_opt, TE_val, TE_test, datasets, refs,
                                               max_steps, val_every, p_drop, ckpt_sink=ck)
    log.info("captured %d I checkpoints; scoring on proxies + test ...", len(ck))

    model = GateModel(ARMS["I"]["use_int"]).to(DEV)
    steps = [c["step"] for c in ck]
    traj = {p: {ds: [] for ds in datasets} for p in prox}
    test_curve = {ds: [] for ds in datasets}
    gate_curve = {ds: {"offset": [], "splade": []} for ds in datasets}
    for c in ck:
        model.load_state_dict(c["state"]["model"])
        for p, vset in prox.items():
            r = _r5(model, vset, datasets)
            for ds in datasets:
                traj[p][ds].append(r[ds])
        tr = _r5(model, TE_test, datasets)
        mg = mean_gates(model, TE_val, datasets)
        for ds in datasets:
            test_curve[ds].append(tr[ds])
            gate_curve[ds]["offset"].append(mg[ds]["offset"] if ds in mg else float("nan"))
            gate_curve[ds]["splade"].append(mg[ds]["splade"] if ds in mg else float("nan"))

    spear = {p: {ds: _spearman(traj[p][ds], test_curve[ds]) for ds in datasets} for p in prox}
    def _rng(v):
        a = np.asarray(v, float); a = a[np.isfinite(a)]
        return round(float(a.max() - a.min()), 2) if len(a) else None
    rng_range = {p: {ds: _rng(traj[p][ds]) for ds in datasets} for p in prox}
    payload = {"_config": {"kind": "valproxy_audit", "datasets": list(datasets), "arm": "I(frozen)",
                           "max_steps": max_steps, "val_every": val_every, "p_drop": p_drop,
                           "split_seed": split_seed, "k_inject": k_inject,
                           "note": "proxies source-only; test used once retrospectively for Spearman; NO tuning"},
               "slice_sizes": {p: len(v) for p, v in prox.items()},
               "steps": steps, "test_curve": test_curve, "gate_curve": gate_curve,
               "proxy_trajectories": traj, "proxy_range": rng_range,
               "spearman_proxy_vs_test": spear, "test_final_selected": test_final, "confirm_meta": meta}
    json.dump(payload, open(out_path, "w"), indent=2)
    log.info("spearman(proxy,test): %s", json.dumps(spear))
    log.info("-> %s", out_path)
    return payload


def _score_ckpt(arm, state, vset, dss):
    """Score one captured checkpoint (arm-aware model reconstruction) -> {ds: R@5} on vset."""
    if arm == "A":
        qenc = QEnc(CF.UIN, HQ).to(DEV); cand = Cand(CF.CAND_IN).to(DEV)
        qenc.load_state_dict(state["qenc"]); cand.load_state_dict(state["cand"])
        recs = CF.eval_crag(qenc, cand, vset, dss)
    else:
        model = GateModel(ARMS[arm]["use_int"]).to(DEV)
        model.load_state_dict(state["model"])
        recs = eval_gate(model, vset, dss, arm)
    ag = _agg(recs)
    return {ds: (ag[ds]["R@5"] if ds in ag else float("nan")) for ds in dss}


def pseudood_run(datasets, ood_datasets, max_steps, val_every, p_drop, frac_val, split_seed,
                 out_path, seed=0, ood_cap=500):
    """TRACK-2 pseudo-OOD validation audit. Regenerate frozen A/H/I trajectories (deterministic, identical to
    confirm config -> the SAME frozen models). Score EVERY checkpoint on: source IID val (webqsp/metaqa),
    source TEST (webqsp/metaqa, used only for retrospective correlation), and each held-out THIRD-DOMAIN set
    (source-only for that domain; model NEVER trains on it; no dataset-ID fed). Then compare, per arm, whether a
    pseudo-OOD signal rank-tracks real source-test goodness BETTER than the IID source-val does."""
    src = list(datasets)
    data = _load(src + list(ood_datasets), DEV)
    mu, sd = _pooled_musd(data, src, DEV, "train")          # frozen source stats; OOD z-scored with SAME stats
    splits = make_splits(data, src, frac_val, split_seed)
    TR_opt = _prep_from_idx(data, src, splits, "opt", mu, sd)
    TE_val = _prep_from_idx(data, src, splits, "val", mu, sd)
    TE_test = [CF.prepbatch(r, mu, sd, DEV, ds) for ds in src for r in data[ds]["test"]]
    refs = _val_refs(TE_val, src)
    ood = {od: [CF.prepbatch(r, mu, sd, DEV, od) for r in data[od]["train"][:ood_cap]] for od in ood_datasets}
    log.info("pseudood sizes: opt=%d val=%d test=%d ood=%s", len(TR_opt), len(TE_val), len(TE_test),
             {od: len(v) for od, v in ood.items()})

    arms = ["A", "H", "I"]
    out = {"_config": {"kind": "pseudo_ood_validation", "source": src, "ood": list(ood_datasets),
                       "max_steps": max_steps, "val_every": val_every, "p_drop": p_drop, "seed": seed,
                       "ood_cap": ood_cap, "split_seed": split_seed,
                       "note": "OOD=held-out domain train split, source-only, no dataset-ID, model never trains on it; "
                               "source TEST used ONLY for retrospective Spearman"},
             "arms": {}}
    for arm in arms:
        ck = []
        test_final, curve, meta = _run_confirm_arm(seed, arm, TR_opt, TE_val, TE_test, src, refs,
                                                   max_steps, val_every, p_drop, ckpt_sink=ck)
        steps = [c["step"] for c in ck]
        iid = {ds: [] for ds in src}
        tst = {ds: [] for ds in src}
        oodc = {od: [] for od in ood_datasets}
        for c in ck:
            rv = _score_ckpt(arm, c["state"], TE_val, src)
            rt = _score_ckpt(arm, c["state"], TE_test, src)
            for ds in src:
                iid[ds].append(rv[ds]); tst[ds].append(rt[ds])
            for od in ood_datasets:
                ro = _score_ckpt(arm, c["state"], ood[od], [od])
                oodc[od].append(ro[od])
        ood_min = [float(np.nanmin([oodc[od][i] for od in ood_datasets])) for i in range(len(steps))]
        ood_mean = [float(np.nanmean([oodc[od][i] for od in ood_datasets])) for i in range(len(steps))]
        # retrospective correlations: does each validation signal rank-track real source TEST across checkpoints?
        sig = {"iid_" + ds: iid[ds] for ds in src}
        sig.update({"ood_" + od: oodc[od] for od in ood_datasets})
        sig["ood_min"] = ood_min; sig["ood_mean"] = ood_mean
        spear = {s: {("test_" + ds): _spearman(sig[s], tst[ds]) for ds in src} for s in sig}
        out["arms"][arm] = {"steps": steps, "iid_val": iid, "source_test": tst, "ood": oodc,
                            "ood_min": ood_min, "ood_mean": ood_mean,
                            "spearman_signal_vs_sourcetest": spear,
                            "best_step": meta["best_step"], "test_final_selected": test_final, "meta": meta}
        log.info("[%s] iid ρ vs test: %s | ood_min ρ vs test: %s", arm,
                 {ds: spear["iid_" + ds]["test_" + ds] for ds in src},
                 {ds: spear["ood_min"]["test_" + ds] for ds in src})
    json.dump(out, open(out_path, "w"), indent=2)
    log.info("-> %s", out_path)
    return out


# ── Q1-Q10 regime-balanced training ablation (R0-R3) ─────────────────────────────
import itertools as _it, math as _math
_DENSE, _OFFSET, _SPLADE, _RELATION, _PATH = 0, 1, 2, 3, 4
_SEM = [_DENSE, _SPLADE]; _REL = [_OFFSET, _RELATION, _PATH]
_REGIMES = ["hard", "missing_structure", "conflict", "single_rescue", "redundant", "mixed_coop", "semantic", "relational"]
_SUBSETS = [frozenset(c) for k in range(NE + 1) for c in _it.combinations(range(NE), k)]
_FACT = [_math.factorial(i) for i in range(NE + 1)]


def _q_phi_inter(pb):
    """Compact exact per-query Shapley phi(U_R5) + pairwise interaction over the 5 experts. gold-labelled (train only)."""
    T = cand_tensors(pb); rp = T["rank_pct"].cpu().numpy(); y = pb.y.cpu().numpy() > 0.5
    n = rp.shape[0]
    if y.sum() == 0:
        return None
    ng = max(float(pb.ng), 1.0)
    U = {}
    for S in _SUBSETS:
        if not S:
            U[S] = 0.0; continue
        fused = rp[:, list(S)].max(1)
        top5 = np.argsort(-fused, kind="stable")[:5]
        U[S] = float(y[top5].sum()) / ng
    phi = np.zeros(NE)
    for e in range(NE):
        for S in _SUBSETS:
            if e in S:
                continue
            w = _FACT[len(S)] * _FACT[NE - 1 - len(S)] / _FACT[NE]
            phi[e] += w * (U[S | {e}] - U[S])
    inter = {}
    for e, f in _it.combinations(range(NE), 2):
        v = 0.0
        for S in _SUBSETS:
            if e in S or f in S:
                continue
            w = _FACT[len(S)] * _FACT[NE - 2 - len(S)] / _FACT[NE - 1]
            v += w * (U[S | {e, f}] - U[S | {e}] - U[S | {f}] + U[S])
        inter[(e, f)] = v
    avail = (T["valid"].sum(0) > 0).cpu().numpy().astype(bool)
    top1 = [int(np.argmax(rp[:, e])) if avail[e] and rp[:, e].max() > 0 else -1 for e in range(NE)]
    return {"phi": phi, "inter": inter, "avail": avail, "top1": top1}


def _regime_of(rec, thr, neg_int):
    phi = rec["phi"]; av = rec["avail"]; inter = rec["inter"]; top1 = rec["top1"]
    useful = np.array([av[e] and phi[e] >= thr[e] for e in range(NE)])
    nu = int(useful.sum())
    sem_use = any(useful[e] for e in _SEM); rel_use = any(useful[e] for e in _REL)
    if nu == 0:
        return "hard"
    if (not av[_RELATION]) and (not av[_PATH]) and sem_use and not useful[_OFFSET]:
        return "missing_structure"
    present = [top1[e] for e in range(NE) if top1[e] >= 0]
    dominant = phi.max() > 0 and phi.max() >= 2.0 * (np.sort(phi)[-2] if (phi > 0).sum() >= 2 else 0.0)
    if len(set(present)) >= 2 and nu >= 2 and not dominant:
        if len(set(top1[e] for e in range(NE) if useful[e] and top1[e] >= 0)) >= 2:
            return "conflict"
    if nu == 1:
        return "single_rescue"
    uidx = [e for e in range(NE) if useful[e]]
    for e, f in _it.combinations(uidx, 2):
        if inter.get((e, f), inter.get((f, e), 0.0)) <= neg_int:
            return "redundant"
    if sem_use and rel_use:
        return "mixed_coop"
    if sem_use:
        return "semantic"
    return "relational"


def _label_train(TR):
    """Per-query regime labels + TRAIN-ONLY thresholds (Q75 phi per expert, Q25 interaction). Returns (labels, thr, neg_int)."""
    recs = [_q_phi_inter(pb) for pb in TR]
    phis = np.array([r["phi"] for r in recs if r is not None])
    thr = np.percentile(phis, 75, 0)
    allint = np.array([v for r in recs if r is not None for v in r["inter"].values()])
    neg_int = float(np.percentile(allint, 25))
    labels = [(_regime_of(r, thr, neg_int) if r is not None else "hard") for r in recs]
    return labels, thr, neg_int


def _sampler_weights(TR, labels, mode, cap=5.0):
    """Per-query sampling weights (sum=1). mode in {R0 natural, R1 dataset, R2 regime, R3 dataset_regime}."""
    n = len(TR)
    if mode == "R0":
        w = np.ones(n)
    elif mode == "R1":
        from collections import Counter
        c = Counter(pb.ds for pb in TR); w = np.array([1.0 / c[pb.ds] for pb in TR])
    elif mode == "R2":
        from collections import Counter
        c = Counter(labels); w = np.array([1.0 / c[labels[i]] for i in range(n)])
    else:  # R3 dataset x regime with oversample cap
        from collections import Counter
        key = [(TR[i].ds, labels[i]) for i in range(n)]
        c = Counter(key); nat = np.array([c[key[i]] / n for i in range(n)])
        uni = np.array([1.0 / c[key[i]] for i in range(n)]); uni = uni / uni.sum()
        w = np.minimum(uni, nat * cap)
    return w / w.sum()


def _rich_agg(recs):
    out = {}
    for ds, rs in recs.items():
        if not rs:
            continue
        r1 = r5 = mrr = n5 = 0.0
        for ry, ng in rs:
            ry = np.asarray(ry, float)
            r1 += ry[:1].sum() / ng; r5 += ry[:5].sum() / ng
            nz = np.nonzero(ry)[0]; mrr += 1.0 / (nz[0] + 1) if len(nz) else 0.0
            dcg = sum(ry[i] / _math.log2(i + 2) for i in range(min(5, len(ry))))
            idcg = sum(1.0 / _math.log2(i + 2) for i in range(min(5, int(ng))))
            n5 += dcg / idcg if idcg > 0 else 0.0
        m = len(rs)
        out[ds] = {"R@1": round(100 * r1 / m, 2), "R@5": round(100 * r5 / m, 2),
                   "MRR": round(mrr / m, 4), "NDCG@5": round(n5 / m, 4)}
    return out


def _train_I_sampled(seed, TR, weights, HO, ho_ds, max_steps, val_every, p_drop, ho_labels=None):
    """Train arm I (identical config) with a per-query weighted sampler; select on held-out-domain MRR (rank-sensitive)."""
    import time
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    cfg = ARMS["I"]; model = GateModel(cfg["use_int"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rnames = list(DROP.keys()); idx = np.arange(len(TR))
    curve = []; best = {"mrr": -1.0, "step": 0, "state": None}
    t0 = time.time(); step = 0
    while step < max_steps:
        j = int(rng.choice(idx, p=weights))
        pb = TR[j]
        pbc = CF.apply_drop(pb, DROP[rnames[rng.randint(len(rnames))]]) if (p_drop > 0 and rng.rand() < p_drop) else pb
        if float(pbc.qavail[EXPERTS].sum()) == 0:
            continue
        T = cand_tensors(pbc); s_out, hg, cg = model(T)
        S = coop_maxrank(T) + RES_SCALE * torch.tanh(s_out)
        loss = _listnet(S, pb.y) + 0.05 * (torch.tanh(s_out) ** 2).mean() + 0.3 * _corro_loss(S, T)
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % val_every == 0 or step >= max_steps:
            rich = _rich_agg(eval_gate(model, HO, [ho_ds], "I")).get(ho_ds, {})
            rec = {"step": step, "ho": rich}
            if ho_labels is not None:
                rec["gates_by_regime"] = _gates_by_regime(model, HO, ho_labels)
            curve.append(rec)
            if rich.get("MRR", -1) > best["mrr"]:
                best = {"mrr": rich["MRR"], "step": step, "state": _clone_state(model)}
    model.load_state_dict(best["state"])
    return model, curve, {"best_step": best["step"], "runtime_s": round(time.time() - t0, 1)}


def _gates_by_regime(model, vset, labels):
    """Mean offset/splade candidate-gate per regime on a labelled set (inference-time gate behaviour by query regime)."""
    from collections import defaultdict
    acc = defaultdict(lambda: {"off": [], "spl": [], "n": 0})
    model.eval()
    with torch.no_grad():
        for pb, lb in zip(vset, labels):
            T = cand_tensors(pb); _, _, cg = model(T)
            m = T["valid"] > 0.5
            for e, k in ((OFFSET, "off"), (SPLADE, "spl")):
                col = cg[:, e][m[:, e]]
                if col.numel():
                    acc[lb][k].append(float(col.mean()))
            acc[lb]["n"] += 1
    return {lb: {"offset": round(float(np.mean(v["off"])), 3) if v["off"] else None,
                 "splade": round(float(np.mean(v["spl"])), 3) if v["spl"] else None, "n": v["n"]}
            for lb, v in acc.items()}


def regime_ablation(out_path, max_steps=16000, val_every=500, p_drop=0.2, seed=0):
    """R0-R3 controlled sampler ablation across 3 held-out-domain episodes. Arm I only; only the SAMPLER changes."""
    ALLDS = ["webqsp", "metaqa", "squad_clean"]
    data = _load(ALLDS, DEV)
    episodes = {"A": (["metaqa", "squad_clean"], "webqsp"),
                "B": (["webqsp", "squad_clean"], "metaqa"),
                "C": (["webqsp", "metaqa"], "squad_clean")}
    out = {"_config": {"kind": "regime_ablation_R0R3", "episodes": {k: {"train": v[0], "val": v[1]} for k, v in episodes.items()},
                       "samplers": {"R0": "natural", "R1": "dataset-balanced", "R2": "regime-balanced", "R3": "dataset x regime (cap5x)"},
                       "max_steps": max_steps, "select_on": "held-out-domain MRR", "note": "identical arch/loss/opt; TRAIN-only regime thresholds; no dataset-ID; no final-test"},
           "episodes": {}}
    for ep, (trds, hods) in episodes.items():
        mu, sd = _pooled_musd(data, trds, DEV, "train")           # TRAIN-domain-only stats
        TR = [CF.prepbatch(r, mu, sd, DEV, ds) for ds in trds for r in data[ds]["train"]]
        HO = [CF.prepbatch(r, mu, sd, DEV, hods) for r in data[hods]["test"]]
        labels, thr, neg_int = _label_train(TR)
        ho_labels, _, _ = _label_train(HO)                        # labels for gate-by-regime readout (diagnostic only)
        from collections import Counter
        cell_counts = dict(Counter((TR[i].ds, labels[i]) for i in range(len(TR))))
        log.info("[ep %s] train %s (%d q) -> HO %s (%d q)", ep, trds, len(TR), hods, len(HO))
        ep_out = {"train": trds, "held_out": hods, "n_train": len(TR), "n_ho": len(HO),
                  "cell_counts": {f"{d}|{r}": n for (d, r), n in cell_counts.items()}, "samplers": {}}
        for mode in ["R0", "R1", "R2", "R3"]:
            w = _sampler_weights(TR, labels, mode)
            model, curve, meta = _train_I_sampled(seed, TR, w, HO, hods, max_steps, val_every, p_drop, ho_labels)
            best_rich = max(curve, key=lambda c: c["ho"].get("MRR", -1))
            gbr = _gates_by_regime(model, HO, ho_labels)
            ep_out["samplers"][mode] = {"best_step": meta["best_step"], "held_out_metrics": best_rich["ho"],
                                        "gates_by_regime": gbr, "runtime_s": meta["runtime_s"]}
            log.info("  [%s/%s] HO %s: %s", ep, mode, hods, best_rich["ho"])
        out["episodes"][ep] = ep_out
        json.dump(out, open(out_path, "w"), indent=2)             # incremental
    log.info("-> %s", out_path)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["F0", "F1", "G1", "G2", "H", "I"])
    p.add_argument("--datasets", nargs="+", default=["webqsp", "metaqa"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--p-drop", type=float, default=0.2)
    p.add_argument("--out", default="results/L2/crag_gates.json")
    p.add_argument("--batched", action="store_true", help="minibatch segment-softmax trainer (GPU-efficient)")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--confirm", action="store_true",
                   help="3-arm A/H/I confirmation: TRAIN-only val split, step budget, robust checkpoint, one test eval")
    p.add_argument("--valproxy", action="store_true",
                   help="validation-proxy audit: frozen I checkpoint trajectory scored on source-only proxies + test(once)")
    p.add_argument("--pseudood", action="store_true",
                   help="TRACK-2 pseudo-OOD validation: A/H/I checkpoints scored on held-out third-domain sets")
    p.add_argument("--regime-ablation", action="store_true",
                   help="R0-R3 regime-balanced sampler ablation across 3 held-out-domain episodes (arm I, sampler-only)")
    p.add_argument("--ood-datasets", nargs="+", default=["2wiki_clean", "musique_clean", "hotpotqa_clean"],
                   help="held-out third-domain validation sets (source-only, model never trains on them)")
    p.add_argument("--max-steps", type=int, default=16000, help="confirm: per-query optimizer-step budget")
    p.add_argument("--val-every", type=int, default=500, help="confirm: validate every N optimizer steps")
    p.add_argument("--frac-val", type=float, default=0.15, help="confirm: fraction of TRAIN held out for validation")
    p.add_argument("--split-seed", type=int, default=1234, help="confirm: RNG seed for the opt/val split (shared)")
    a = p.parse_args(argv)
    if a.regime_ablation:
        regime_ablation(a.out, a.max_steps, a.val_every, a.p_drop)
    elif a.pseudood:
        pseudood_run(a.datasets, a.ood_datasets, a.max_steps, a.val_every, a.p_drop,
                     a.frac_val, a.split_seed, a.out)
    elif a.valproxy:
        valproxy_run(a.datasets, a.max_steps, a.val_every, a.p_drop, a.frac_val, a.split_seed, a.out)
    elif a.confirm:
        arms = [x for x in a.arms if x in ("A", "H", "I")] or ["A", "H", "I"]
        confirm(a.datasets, arms, a.seeds, a.max_steps, a.val_every, a.p_drop,
                a.frac_val, a.split_seed, a.out)
    else:
        run(a.datasets, a.arms, a.epochs, a.seeds, a.p_drop, a.out, a.batched, a.bs)


if __name__ == "__main__":
    main()
