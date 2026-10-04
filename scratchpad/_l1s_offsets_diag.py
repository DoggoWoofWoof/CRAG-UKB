"""Graph-derived embedding offsets -- query-free diagnostics (metaqa first; any typed graph).

Per relation r (relation IDs only; labels never read):  delta_uv = e_v - e_u over served structural edges
(u, r, v);  Delta_r = mean delta;  coherence C_r = |mean delta| / mean |delta|.
Offset hit rate on held-out edges: e_hat = norm(e_u + Delta_r) -> rank of e_v among all nodes (vs e_u alone).
Global structural operator G = sum d d^T / sum |d|^2 over undirected STRUCT edges: spectrum.
One-point ceiling (uses gold labels, diagnostic only): e_hat = mean(e_gold) -> can the top-K nodes / top-50
blocks around a single point cover ALL gold blocks?"""
import json
import os
import sys

import numpy as np

import _l1s_core as S

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = S.Data(name)
E = D.E
N, dim = E.shape
s, d, r, ent = D.struct_edges()
vocab = D.relation_vocab()
nrel = int(r.max()) + 1
rng = np.random.RandomState(0)
out = {"cache": name, "N": int(N), "n_edges": int(len(s)), "nrel": nrel, "relations": {}}
S.log("%s: %d structural edges, %d relations" % (name, len(s), nrel))

# ---------------------------------------------------------------- per-relation coherence + offset hit rate
def topk_rank(Ehat, targets, batch=400):
    """rank of targets[i] among all nodes by cosine to Ehat[i] (0 = best)."""
    ranks = np.empty(len(targets), np.int64)
    for a in range(0, len(targets), batch):
        b = min(len(targets), a + batch)
        sims = Ehat[a:b] @ E.T
        tv = sims[np.arange(b - a), targets[a:b]]
        ranks[a:b] = (sims > tv[:, None]).sum(axis=1)
    return ranks


def block_cover(Ehat, targets, k=100, P=50):
    """block(target) inside the top-P blocks voted by the top-k nodes around Ehat (legacy membership, rank votes)."""
    hit = np.zeros(len(targets), bool)
    for a in range(0, len(targets), 400):
        b = min(len(targets), a + 400)
        sims = Ehat[a:b] @ E.T
        ids = np.argpartition(-sims, k, axis=1)[:, :k]
        o = np.take_along_axis(sims, ids, axis=1)
        ids = np.take_along_axis(ids, np.argsort(-o, axis=1), axis=1)
        Sb = S.hits_to_blocks(ids, np.tile(S.rankvec(k), (b - a, 1)), D.mem, D.npart, "sum")
        top = np.argsort(-Sb, axis=1)[:, :P]
        tb = D.hard[targets[a:b]]
        hit[a:b] = (top == tb[:, None]).any(axis=1)
    return hit


Delta = np.zeros((nrel, dim), np.float32)
for rel in range(nrel):
    m = np.nonzero(r == rel)[0]
    if len(m) == 0:
        continue
    delta = E[d[m]] - E[s[m]]
    mean = delta.mean(axis=0)
    C = float(np.linalg.norm(mean) / np.linalg.norm(delta, axis=1).mean())
    Delta[rel] = mean
    # held-out check: Delta from a random half of the edges, hit rate on the other half
    perm = rng.permutation(len(m))
    half = len(m) // 2
    fit, test = m[perm[:half]], m[perm[half:half + 1000]]
    Dr = (E[d[fit]] - E[s[fit]]).mean(axis=0)
    Eh = E[s[test]] + Dr[None, :]
    Eh /= np.linalg.norm(Eh, axis=1, keepdims=True) + 1e-9
    rk_off = topk_rank(Eh, d[test])
    rk_raw = topk_rank(E[s[test]], d[test])
    cov_off = block_cover(Eh, d[test])
    cov_raw = block_cover(E[s[test]], d[test])
    lab = vocab[rel] if rel < len(vocab) else str(rel)
    e = {"label_for_report_only": lab, "n_edges": int(len(m)), "coherence": round(C, 4), "norm_Delta": round(float(np.linalg.norm(mean)), 4),
         "mean_norm_delta": round(float(np.linalg.norm(delta, axis=1).mean()), 4),
         "heldout": {"n": int(len(test)), "hit@1_offset": float((rk_off < 1).mean()), "hit@10_offset": float((rk_off < 10).mean()), "hit@100_offset": float((rk_off < 100).mean()),
                     "hit@1_raw": float((rk_raw < 1).mean()), "hit@10_raw": float((rk_raw < 10).mean()), "hit@100_raw": float((rk_raw < 100).mean()),
                     "median_rank_offset": float(np.median(rk_off)), "median_rank_raw": float(np.median(rk_raw)),
                     "block_in_top50_offset": float(cov_off.mean()), "block_in_top50_raw": float(cov_raw.mean())}}
    out["relations"][str(rel)] = e
    S.log("  rel %d %-18s n=%7d C=%.3f |D|=%.3f | heldout hit@10 off %.3f raw %.3f  hit@100 off %.3f raw %.3f  medrank off %5.0f raw %5.0f | block@50 off %.3f raw %.3f" % (
        rel, lab[:18], len(m), C, e["norm_Delta"], e["heldout"]["hit@10_offset"], e["heldout"]["hit@10_raw"], e["heldout"]["hit@100_offset"], e["heldout"]["hit@100_raw"],
        e["heldout"]["median_rank_offset"], e["heldout"]["median_rank_raw"], cov_off.mean(), cov_raw.mean()))

# pairwise cosine between relation offsets (are the Delta_r distinct directions?)
Dn = Delta / (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-9)
out["Delta_cosine_matrix"] = np.round(Dn @ Dn.T, 3).tolist()
S.log("Delta_r pairwise cosines:\n%s" % np.round(Dn @ Dn.T, 2))

# ---------------------------------------------------------------- global operator G (undirected keys, each edge once)
Nn, STRUCT, KNN, NERX = D.cd.keysets()
u = (STRUCT // np.int64(N)).astype(np.int64)
v = (STRUCT % np.int64(N)).astype(np.int64)
G = np.zeros((dim, dim), np.float64)
den = 0.0
for a in range(0, len(u), 20000):
    dd = (E[v[a:a + 20000]] - E[u[a:a + 20000]]).astype(np.float64)
    G += dd.T @ dd
    den += float((dd * dd).sum())
G /= den
w, V = np.linalg.eigh(G)
w = w[::-1]
V = V[:, ::-1]
cum = np.cumsum(w) / w.sum()
out["G"] = {"n_edges_undirected": int(len(u)), "trace": float(w.sum()), "top_eigenvalues": np.round(w[:10], 5).tolist(),
            "energy_top1": float(cum[0]), "energy_top10": float(cum[9]), "energy_top50": float(cum[49]), "energy_top200": float(cum[199]),
            "effective_rank_exp_entropy": float(np.exp(-(w[w > 0] / w.sum() * np.log(w[w > 0] / w.sum())).sum()))}
S.log("G: trace %.3f, top eig %s, energy top1 %.3f top10 %.3f top50 %.3f top200 %.3f, eff rank %.1f" % (
    w.sum(), np.round(w[:5], 4), cum[0], cum[9], cum[49], cum[199], out["G"]["effective_rank_exp_entropy"]))
# how much of each Delta_r lies in the top-k eigen-subspace of G
proj = {}
for rel in range(nrel):
    z = Dn[rel].astype(np.float64) @ V
    proj[str(rel)] = {"top10": float((z[:10] ** 2).sum()), "top50": float((z[:50] ** 2).sum())}
out["Delta_energy_in_G_subspace"] = proj
np.save(os.path.join(S.OUT, "G_eigvecs_%s.npy" % name), V[:, :200].astype(np.float32))
np.save(os.path.join(S.OUT, "Delta_r_%s.npy" % name), Delta)

# ---------------------------------------------------------------- one-point ceiling on DEV_A queries (gold labels; diagnostic only)
A = np.nonzero(D.A)[0]
gc = np.zeros((len(A), dim), np.float32)
for j, i in enumerate(A):
    gc[j] = E[D.C.gold_nodes[i]].mean(axis=0)
gc /= np.linalg.norm(gc, axis=1, keepdims=True) + 1e-9
covs = {}
for k in (50, 100, 200):
    hit_all = np.zeros(len(A), bool)
    for a in range(0, len(A), 400):
        b = min(len(A), a + 400)
        sims = gc[a:b] @ E.T
        ids = np.argpartition(-sims, k, axis=1)[:, :k]
        o = np.take_along_axis(sims, ids, axis=1)
        ids = np.take_along_axis(ids, np.argsort(-o, axis=1), axis=1)
        Sb = S.hits_to_blocks(ids, np.tile(S.rankvec(k), (b - a, 1)), D.mem, D.npart, "sum")
        top = np.argsort(-Sb, axis=1)[:, :50]
        for j in range(a, b):
            hit_all[j] = all((top[j - a] == p).any() for p in D.C.gb[A[j]])
    covs["topk_nodes=%d" % k] = float(hit_all.mean())
    S.log("one-point ceiling (e_hat = mean gold embedding, %d nearest nodes -> legacy membership -> P50): DEV_A ALL %.4f" % (k, hit_all.mean()))
# the same with the gold set's own blocks: how many nearest-node votes land in gold blocks (spread)
hops = D.C.hops[A]
per_hop = {}
for h in sorted(set(int(x) for x in hops)):
    m = hops == h
    per_hop["hop%d" % h] = {}
out["one_point_ceiling_DEV_A"] = covs
S.wj(os.path.join(S.OUT, "offsets_diag_%s.json" % name), out)
S.log("wrote offsets_diag_%s.json" % name)
