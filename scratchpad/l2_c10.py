"""C10 — SET-AWARE MULTI-EVIDENCE TOP-5 SELECTION. Tests whether DYNAMIC set-aware complementarity (marginal value
of a candidate given what is already selected) can push top-5 beyond the static per-candidate C8c/C9 reranker.

Uses ONLY existing canonical Qwen node embeddings (data/ukb_storage/{ds}/gte_qwen/nodes.npy, L2-normalized so
cosine = dot) and the existing 5-dim expert RRF-contribution vectors. No re-encoding, no new experts, no pool
enlargement, no graph traversal, no TEST, no L1/L3 change. Standing top-5 policy is C8c; C10 must BEAT it on
C9_DEV (select) + one DEV_INNER confirm to matter.

Primitives here; drivers do the phases. Bundles/base head reproduced exactly from the C9 machinery (l2_c9)."""
import os, sys, json
sys.path.insert(0, "scratchpad")
import numpy as np
import l2_c8 as C8, l2_c8c as C8C, l2_c9 as C9
import l2_c6 as C6
OUT = C8.OUT; DS = C8.DS; EPS = 1e-9
EXPN = ["dense", "splade", "offset", "mixture", "relation"]
_NODES = {}; _CAND = {}; _OFF = {}


def nodes(ds):
    if ds not in _NODES:
        _NODES[ds] = np.load(f"data/ukb_storage/{ds}/gte_qwen/nodes.npy", mmap_mode="r")
    return _NODES[ds]


def cand_ids(ds):
    if ds not in _CAND:
        _CAND[ds] = np.load(f"data/l2_corpus/{ds}/train/cand_ids.npy", mmap_mode="r")
        _OFF[ds] = np.load(f"data/l2_corpus/{ds}/train/query_offsets.npy")
    return _CAND[ds], _OFF[ds]


def pool_embs(ds, qi, pool_local):
    """Qwen embeddings (P,1536) float32 for the pool candidates of query qi (train split)."""
    c, off = cand_ids(ds); s = int(off[qi])
    gids = np.asarray(c[s + pool_local], np.int64)
    return np.asarray(nodes(ds)[gids], np.float32)


def _cos_to_set(V, sel_mask):
    """V=(P,d) row-normalized-ish (Qwen already unit). Return per-row (max,mean,min) cosine to the selected set,
    EXCLUDING self. sel_mask=(P,) bool of selected rows. Rows not needing it still computed cheaply."""
    S = V[sel_mask]                       # (k,d)
    if len(S) == 0:
        z = np.zeros(len(V), np.float32); return z, z, z.copy()
    sims = V @ S.T                        # (P,k) cosine (unit vectors)
    sel_idx = np.where(sel_mask)[0]
    out_max = np.empty(len(V), np.float32); out_mean = np.empty(len(V), np.float32); out_min = np.empty(len(V), np.float32)
    for j in range(len(V)):
        row = sims[j]
        if sel_mask[j]:                   # exclude self
            keep = sel_idx != j
            row = row[keep]
        if len(row) == 0:
            out_max[j] = out_mean[j] = out_min[j] = 0.0
        else:
            out_max[j] = row.max(); out_mean[j] = row.mean(); out_min[j] = row.min()
    return out_max, out_mean, out_min


def _supp_cos_to_set(Cvec, sel_mask):
    """Cvec=(P,5) expert RRF-contribution vectors. Cosine (L2-normalized) max/mean to selected (excl self)."""
    nrm = Cvec / (np.linalg.norm(Cvec, axis=1, keepdims=True) + EPS)
    S = nrm[sel_mask]
    if len(S) == 0:
        z = np.zeros(len(Cvec), np.float32); return z, z
    sims = nrm @ S.T; sel_idx = np.where(sel_mask)[0]
    mx = np.empty(len(Cvec), np.float32); mn = np.empty(len(Cvec), np.float32)
    for j in range(len(Cvec)):
        row = sims[j]
        if sel_mask[j]:
            row = row[sel_idx != j]
        if len(row) == 0: mx[j] = mn[j] = 0.0
        else: mx[j] = row.max(); mn[j] = row.mean()
    return mx, mn


def cohend(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2: return None
    na, nb = len(a), len(b); va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / (sp + EPS))


def base_head(c9tr):
    return C8.fit_soft_head("ndcg50", c9tr)


def dist(a):
    a = np.asarray(a, float)
    if len(a) == 0: return None
    return {"n": int(len(a)), "mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
            "p25": round(float(np.percentile(a, 25)), 4), "p75": round(float(np.percentile(a, 75)), 4),
            "std": round(float(a.std()), 4)}


# ---------------------------------------------------------------- per-query cache for set-aware selection
_RAW = {}
def _raw(ds):
    if ds not in _RAW: _RAW[ds] = C6._load_raw(ds, "train")
    return _RAW[ds]


def qc_one(m, sc):
    """Build one per-query cache (Qwen embs + 5-dim contrib vectors) on demand. sc = C8c scores for pool rows."""
    ds = m["ds"]; pool = m["pool"]; g = len(pool)
    V = pool_embs(ds, m["qi"], pool)
    R = _raw(ds); s0, e0 = int(R["query_offsets"][m["qi"]]), int(R["query_offsets"][m["qi"] + 1])
    Cpool = C8.contribs_of(R, s0, e0)[:, pool].T.astype(np.float32)
    Cn = Cpool / (np.linalg.norm(Cpool, axis=1, keepdims=True) + EPS)
    goldset = set(m["gold"].tolist()); isg = np.array([int(pool[j]) in goldset for j in range(g)])
    return {"ds": ds, "qi": m["qi"], "pool": pool, "brank": m["brank"], "n": m["n"], "gold": m["gold"],
            "c8c": np.asarray(sc, np.float64), "V": V, "Cn": Cn, "isg": isg, "g": g}


def precompute(B, c8c_scores):
    """Cache qc_one per query (stores Qwen embs — only use for the smaller dev/di splits)."""
    QC = []; pos = 0
    for m, g in zip(B["meta"], B["groups"]):
        QC.append(qc_one(m, c8c_scores[pos:pos + g])); pos += g
    return QC


def _relnorm(sc, window):
    """Min-max normalize C8c scores to [0,1] within the top-`window` candidates (by C8c). Others -> their value."""
    if len(sc) == 0: return sc
    lo, hi = sc.min(), sc.max()
    return (sc - lo) / (hi - lo + EPS)


def greedy_order(qc, lam_sem=0.0, lam_exp=0.0, cap=20, npick=5):
    """MMR-style greedy set selection over the top-`cap` C8c window. Returns a permutation of range(g) giving the
    final within-pool order: first `npick` are the greedy picks; remaining window candidates follow in C8c order;
    candidates beyond the window keep C8c order. lam_sem=lam_exp=0 reproduces C8c ordering exactly."""
    g = qc["g"]; sc = qc["c8c"]; order = np.argsort(-sc, kind="stable")
    win = order[:min(cap, g)]; rest = order[min(cap, g):]
    rel = _relnorm(sc[win], len(win))                       # normalized relevance within window
    V = qc["V"][win]; Cn = qc["Cn"][win]
    chosen = []; avail = list(range(len(win)))
    for t in range(min(npick, len(win))):
        if t == 0:
            pick = int(np.argmax(rel[avail])); pick = avail[pick]
        else:
            S = np.array(chosen)
            qsim = (V[avail] @ V[S].T).max(axis=1) if lam_sem else np.zeros(len(avail))
            esim = (Cn[avail] @ Cn[S].T).max(axis=1) if lam_exp else np.zeros(len(avail))
            mmr = rel[avail] - lam_sem * qsim - lam_exp * esim
            pick = avail[int(np.argmax(mmr))]
        chosen.append(pick); avail.remove(pick)
    win_order = [win[i] for i in chosen] + [win[i] for i in avail]        # picks first, then leftover window (C8c order preserved among avail)
    return np.array(win_order + list(rest), np.int64)


def order_to_scores(QC, orderings):
    """Convert per-query pool orderings into a flat cand_scores array (pool-row aligned) = -within-pool-position,
    so C9.eval_ranking reproduces the ordering. Returns array aligned to concatenated pool rows."""
    out = []
    for qc, order in zip(QC, orderings):
        g = qc["g"]; place = np.empty(g, np.int64); place[order] = np.arange(g); out.append(-place.astype(np.float64))
    return np.concatenate(out)


# ---------------------------------------------------------------- C10b learned greedy selector primitives
DYNN = ["dyn_qmax", "dyn_qmean", "dyn_qmin", "dyn_smax", "dyn_smean", "dyn_same_argmax_frac",
        "dyn_same_argmax_count", "dyn_novelty_rank", "dyn_step", "dyn_remaining"]


def dyn_block(qc, S, cand, t, npick=5):
    """Dynamic set features (len(cand), 10) for remaining `cand` window-indices given selected set S at step t.
    Qwen/support similarity to S (max/mean/min), same-argmax overlap, novelty rank among cand, step, remaining."""
    V = qc["V"]; Cn = qc["Cn"]; am = np.argmax(Cn, axis=1)
    m = len(cand); D = np.zeros((m, 10), np.float32)
    if len(S):
        qs = V[cand] @ V[S].T; ss = Cn[cand] @ Cn[S].T
        D[:, 0] = qs.max(1); D[:, 1] = qs.mean(1); D[:, 2] = qs.min(1)
        D[:, 3] = ss.max(1); D[:, 4] = ss.mean(1)
        sel_am = am[S]; cnt = np.array([(sel_am == am[c]).sum() for c in cand], np.float32)
        D[:, 5] = cnt / len(S); D[:, 6] = cnt
        nov = D[:, 0]; nr = np.empty(m, np.float32); nr[np.argsort(nov, kind="stable")] = np.arange(m); D[:, 7] = nr
    D[:, 8] = t; D[:, 9] = npick - t
    return D


def rollout(qc, Xq, predict, cap=20, npick=5):
    """Greedy 5-step selection over top-`cap` C8c window using marginal-utility `predict(static+dynamic)`.
    Xq = (g, F) static C9 features for this query's pool rows (row j <-> pool[j]). Returns pool ordering (perm)."""
    g = qc["g"]; sc = qc["c8c"]; order = np.argsort(-sc, kind="stable")
    win = order[:min(cap, g)]; rest = order[min(cap, g):]
    chosen = []; avail = list(win)
    for t in range(min(npick, len(win))):
        cand = np.array(avail, np.int64)
        D = dyn_block(qc, np.array(chosen, np.int64), cand, t, npick)
        feats = np.concatenate([Xq[cand], D], axis=1)
        p = predict(feats); pick = int(cand[int(np.argmax(p))])
        chosen.append(pick); avail.remove(pick)
    return np.array(chosen + avail + list(rest), np.int64)
