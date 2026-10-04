"""L1-A1 (node -> block aggregation) and L1-A2 (direct query <-> static block signature) on DEV_A.

Every arm: existing dense + SPLADE node retrieval (K = K_LOCK = 100 hits per channel unless stated), a static
node -> blocks membership (BASE's legacy own + directed-out-neighbour blocks, or hard-only), an aggregation of
the hits into block scores, the frozen per-channel rank transform, and the frozen partition-level RRF.
No traversal, no training, no thresholds; DEV_A numbers only."""
import itertools
import os
import sys

import numpy as np

import _l1s_core as S

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = S.Data(name)
npart, nq = D.npart, D.nq
res = {"cache": name, "BASE_A": D.r_base["ALL_split"]["A"], "arms": {}}
S.log("%s DEV_A n=%d BASE ALL %.4f" % (name, int(D.A.sum()), res["BASE_A"]["ALL"]))


def weights(kind, sc, k):
    if kind == "rank":                       # frozen 1/(K0 + r)
        return np.tile(S.rankvec(k), (nq, 1))
    if kind == "steep":                      # 1/(1 + r)
        return np.tile(1.0 / (1.0 + np.arange(k)), (nq, 1))
    if kind == "score":                      # the served similarity itself
        return np.maximum(sc[:, :k], 0.0).astype(np.float64)
    if kind == "score_rel":                  # similarity relative to the channel's own top-1 (per query)
        s = np.maximum(sc[:, :k], 0.0).astype(np.float64)
        return s / np.maximum(s[:, :1], 1e-9)
    raise ValueError(kind)


def channel_rank(ids, sc, k, mem, wkind, how, m=3):
    """block ranking of one channel: aggregate, then the frozen rank transform (rr(S) + rr(M) for 'canon')."""
    if wkind == "rank" and how == "canon":   # the frozen numerics themselves (bit-exact BASE)
        return S.canon_channel_rank([ids[i, :k] for i in range(nq)], mem, npart)
    w = weights(wkind, sc, k)
    if how == "canon":
        Ssum = S.hits_to_blocks(ids[:, :k], w, mem, npart, "sum")
        Smax = S.hits_to_blocks(ids[:, :k], w, mem, npart, "max")
        votes = S.rank_votes(Ssum) + S.rank_votes(Smax)
        return np.argsort(-votes, axis=1).astype(np.int64)
    Sb = S.hits_to_blocks(ids[:, :k], w, mem, npart, how, sizes=D.sizes, m=m)
    return np.argsort(-Sb, axis=1, kind="stable").astype(np.int64)


def fused(k, mem, wkind, how, m=3):
    rd = channel_rank(D.d_ids, D.d_sc, k, mem, wkind, how, m)
    rs = channel_rank(D.s_ids, D.s_sc, k, mem, wkind, how, m)
    return S.rrf_ranks([rd, rs]), rd, rs


def run(tag, rank):
    out, allv = D.eval_rank(rank, tag)
    res["arms"][tag] = out["A"]
    return out["A"]["ALL"]


# ---- A0: reproduce BASE through the same code path
r0, rd0, rs0 = fused(S.K_LOCK, D.mem, "rank", "canon")
assert (r0[:, :50] == D.base_rank[:, :50]).all()
run("A0 BASE (legacy mem, rank votes, rr(sum)+rr(max), K=100)", r0)
run("A0 dense channel only", rd0)
run("A0 SPLADE channel only", rs0)

# ---- membership: hard-only vs legacy (own + out-neighbour blocks)
run("A1 hard-only membership (else BASE)", fused(S.K_LOCK, D.mem_hard, "rank", "canon")[0])

# ---- depth K (contract K=100; diagnostics)
for k in (20, 50, 200, 500, 1000):
    run("A1 BASE with K=%d" % k, fused(k, D.mem, "rank", "canon")[0])

# ---- aggregation grid at K=100, legacy membership
S.log("--- aggregation grid (legacy membership, K=100)")
for wkind, how in itertools.product(("rank", "steep", "score", "score_rel"), ("canon", "sum", "max", "topm", "count", "mean", "sum_norm", "sum_sqrt")):
    if wkind != "rank" and how == "count":
        continue
    run("A1 w=%s agg=%s" % (wkind, how), fused(S.K_LOCK, D.mem, wkind, how)[0])
S.log("--- aggregation grid (hard membership, K=100)")
for wkind, how in itertools.product(("rank", "score"), ("canon", "sum", "max", "topm", "sum_norm")):
    run("A1 hard w=%s agg=%s" % (wkind, how), fused(S.K_LOCK, D.mem_hard, wkind, how)[0])

# ---- A2: direct query <-> static block signatures
S.log("--- A2 direct block signatures")
mu = D.centroids()
Sd = (D.Q @ mu.T).astype(np.float64)                       # query . block centroid
r_cent = np.argsort(-Sd, axis=1, kind="stable")
run("A2 direct dense centroid only", r_cent)
Pm = D.splade_pool("mean")
Ss_mean = np.asarray((D.Qs @ Pm.T).todense(), np.float64)
r_sp_mean = np.argsort(-Ss_mean, axis=1, kind="stable")
run("A2 direct SPLADE mean-pool only", r_sp_mean)
Px = D.splade_pool("max")
Ss_max = np.asarray((D.Qs @ Px.T).todense(), np.float64)
r_sp_max = np.argsort(-Ss_max, axis=1, kind="stable")
run("A2 direct SPLADE max-pool only", r_sp_max)
run("A2 direct dense centroid + SPLADE mean (RRF)", S.rrf_ranks([r_cent, r_sp_mean]))
run("A2 BASE + direct dense centroid (RRF 3)", S.rrf_ranks([rd0, rs0, r_cent]))
run("A2 BASE + direct SPLADE mean (RRF 3)", S.rrf_ranks([rd0, rs0, r_sp_mean]))
run("A2 BASE + direct dense + direct SPLADE mean (RRF 4)", S.rrf_ranks([rd0, rs0, r_cent, r_sp_mean]))
run("A2 BASE + direct dense + direct SPLADE max (RRF 4)", S.rrf_ranks([rd0, rs0, r_cent, r_sp_max]))
run("A2 BASE-rank + direct (RRF of BASE fused rank and direct fused rank)", S.rrf_ranks([r0, S.rrf_ranks([r_cent, r_sp_mean])]))

# ---- where do the gold blocks sit under BASE? (diagnostic: rank positions of missed gold blocks)
pt = S.X.rank_pos(r0, npart)
miss_pos = []
for i in range(nq):
    if D.A[i] and not D.base_all[i]:
        miss_pos += [int(pt[i, p]) for p in D.C.gb[i] if pt[i, p] >= 50]
miss_pos = np.array(miss_pos)
res["diag_base_missed_gold_block_rank"] = {"n": int(len(miss_pos)), "p50": float(np.median(miss_pos)) if len(miss_pos) else None,
                                          "p90": float(np.percentile(miss_pos, 90)) if len(miss_pos) else None,
                                          "within_100": float((miss_pos < 100).mean()) if len(miss_pos) else None,
                                          "unranked": float((miss_pos >= 10 ** 6).mean()) if len(miss_pos) else None}
S.log("BASE missed gold blocks (DEV_A): n=%d median rank %s p90 %s within100 %s unranked %s" % tuple(res["diag_base_missed_gold_block_rank"].values()))
S.wj(os.path.join(S.OUT, "agg_A_%s.json" % name), res)
