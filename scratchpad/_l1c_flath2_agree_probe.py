"""Gold-free probe for the FLAT_BALANCED_H2 served-list sanity threshold (AGREE_MIN), run BEFORE the pre-registration is
written: on the first NPROBE split-A rows of a fresh dataset, the exhaustive fp32 dense / SPLADE top-100 over the FULL corpus
(the module's exact products: unit queries x fp16->fp32 node chunks; SPLADE shard x dense query columns) vs the served
retrieval caches (TF32 H100 top-200).  Gold COUNTS only (the population rule).  Writes nothing in the repository."""
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = "C:/Users/Swastik/Desktop/CRAG"
sys.path[:0] = [REPO, os.path.join(REPO, "scratchpad")]
os.chdir(REPO)
from src.l1_canonical import adapter as AD  # noqa: E402

BLOCK = 40000
KEEP = 400
ds = sys.argv[1]
NPROBE = int(sys.argv[2]) if len(sys.argv) > 2 else 40
t0 = time.time()
cd = AD.CanonicalDataset(ds)
N = int(cd.n_nodes)
zq = np.load(cd._query_index_path(), allow_pickle=False)
gold_ptr = zq["gold_ptr"]
ev = [int(x) for x in cd.eval_rows()]
np.random.seed(0)
np.random.shuffle(ev)
pop = sorted(ev[:AD.CONTRACT["EVAL_CAP"]])
ng = np.diff(gold_ptr)[np.asarray(pop, np.int64)]
pop = [r for r, n in zip(pop, ng) if n > 0]
qids = [cd.query_ids[r] for r in pop]
par = [int(hashlib.sha1(q.encode("utf-8")).hexdigest()[:8], 16) & 1 for q in qids]
rows_A = np.asarray([r for r, p in zip(pop, par) if p == 0], np.int64)
rows = rows_A[:NPROBE]
n = len(rows)
Q = cd.query_embeddings[rows].astype(np.float32)
Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
QsA = cd.embeddings("splade", "queries").read(rows).tocsr()
QdT = np.asarray(QsA.todense(), np.float32).T
Es = cd.embeddings("splade", "docs")
assert int(Es.shard_size) == BLOCK and int(Es.n_rows) == N
d200 = np.asarray(cd.dense_topk(200, rows), np.int64)
s200 = np.asarray(cd.splade_topk(200, rows), np.int64)
print("%s: N %d, population %d, split A %d, probing %d rows (%.0fs)" % (ds, N, len(pop), len(rows_A), n, time.time() - t0), flush=True)

cd_sc = [np.zeros(0, np.float32) for _ in range(n)]
cd_id = [np.zeros(0, np.int64) for _ in range(n)]
cs_sc = [np.zeros(0, np.float32) for _ in range(n)]
cs_id = [np.zeros(0, np.int64) for _ in range(n)]
srv_d = np.full((n, 200), np.nan, np.float32)      # fp32 score of each served node, the same products
srv_s = np.full((n, 200), np.nan, np.float32)
npos = np.zeros(n, np.int64)


def merge(sc_old, id_old, sc, ids):
    s_ = np.concatenate([sc_old, sc])
    i_ = np.concatenate([id_old, ids])
    o = np.lexsort((i_, -s_))[:KEEP]
    return s_[o], i_[o]


t1 = time.time()
n_ch = (N + BLOCK - 1) // BLOCK
for c_, (a, b_, blk) in enumerate(cd.node_embeddings.blocks(BLOCK)):
    SD = np.empty((n, b_ - a), np.float32)                 # sub-blocked conversion (low memory; a probe, not the identity)
    for s0 in range(0, b_ - a, 10000):
        SD[:, s0:s0 + 10000] = Q @ np.asarray(blk[s0:s0 + 10000], np.float32).T
    Msp = Es.shard(a // BLOCK)
    SS = np.ascontiguousarray(np.asarray(Msp @ QdT, np.float32).T)
    ids = np.arange(a, b_, dtype=np.int64)
    npos += (SS > 0).sum(1)
    for i in range(n):
        k = min(KEEP, b_ - a)
        p = np.argpartition(-SD[i], k - 1)[:k]
        cd_sc[i], cd_id[i] = merge(cd_sc[i], cd_id[i], SD[i][p], ids[p])
        pos = np.flatnonzero(SS[i] > 0)
        if pos.size:
            k = min(KEEP, pos.size)
            p = pos[np.argpartition(-SS[i][pos], k - 1)[:k]]
            cs_sc[i], cs_id[i] = merge(cs_sc[i], cs_id[i], SS[i][p], ids[p])
        m = (d200[i] >= a) & (d200[i] < b_)
        srv_d[i, m] = SD[i][d200[i][m] - a]
        m = (s200[i] >= a) & (s200[i] < b_)
        srv_s[i, m] = SS[i][s200[i][m] - a]
    del SD, SS, Msp
    if c_ % 25 == 0 or c_ == n_ch - 1:
        print("  chunk %d / %d (%.0fs)" % (c_ + 1, n_ch, time.time() - t1), flush=True)

ov_d, ov_s, rows_out = [], [], []
for i in range(n):
    od100 = set(cd_id[i][:100].tolist())
    sd100 = set(d200[i][:100].tolist())
    ov_d.append(len(od100 & sd100) / 100.0)
    ks = int(min(100, npos[i]))
    if ks:
        ov_s.append(len(set(cs_id[i][:ks].tolist()) & set(s200[i][:ks].tolist())) / float(ks))
    else:
        ov_s.append(1.0)
    s100 = float(cd_sc[i][99])
    gap = s100 - srv_d[i, :100]                    # >0: the served node scores below our fp32 100th (fp32)
    posn = {int(x): r for r, x in enumerate(cd_id[i].tolist())}
    disp = [posn.get(int(x), KEEP) for x in d200[i][:100]]
    rows_out.append({"row": int(rows[i]), "dense_ov100": ov_d[-1], "splade_ov100": ov_s[-1], "npos": int(npos[i]),
                     "dense_fp32_100th": round(s100, 6), "dense_score_spread_top100": round(float(cd_sc[i][0] - s100), 5),
                     "served_below_fp32_100th_max_gap": round(float(gap.max()), 6),
                     "served_top100_worst_fp32_rank": int(max(disp)),
                     "served_top100_rank1_equal": bool(cd_id[i][0] == d200[i][0])})
res = {"dataset": ds, "N": N, "rows_probed": n, "seconds": round(time.time() - t0, 1),
       "dense_ov100": {"mean": round(float(np.mean(ov_d)), 4), "min": round(float(np.min(ov_d)), 3),
                       "p10": round(float(np.percentile(ov_d, 10)), 3)},
       "splade_ov100": {"mean": round(float(np.mean(ov_s)), 4), "min": round(float(np.min(ov_s)), 3),
                        "p10": round(float(np.percentile(ov_s, 10)), 3)},
       "rank1_equal_dense": int(sum(r["served_top100_rank1_equal"] for r in rows_out)),
       "served_below_fp32_100th_max_gap": {"max": max(r["served_below_fp32_100th_max_gap"] for r in rows_out)},
       "served_top100_worst_fp32_rank": {"max": max(r["served_top100_worst_fp32_rank"] for r in rows_out),
                                         "median": float(np.median([r["served_top100_worst_fp32_rank"] for r in rows_out]))}}
print(json.dumps(res, indent=1), flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_agree_%s.json" % ds), "w", encoding="utf-8") as f:
    json.dump({"summary": res, "rows": rows_out}, f, indent=1)
