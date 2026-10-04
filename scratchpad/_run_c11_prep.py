"""C11 data prep: for C9_TRAIN / C9_DEV / DEV_INNER build TOP-20 (C8c) per-query tensors:
  qv (Nq,1536)  gte_qwen query embedding (via recovered qmap; frozen, gold-free)
  dv (Nrows,1536) gte_qwen candidate node embedding (frozen)
  E  (Nrows,18) compact proven retrieval evidence (from C9 features; C8c score/rank are OOF on TRAIN)
  c8c(Nrows,) OOF/full C8c backbone score   y (Nrows,) gold label   groups (Nq,) window size (<=20)
  meta: qi, ds, pool(top20 local idx), brank, n, gold(local idx)
Reuses _c10b_cache.joblib (bundles+X+scores). NO new encoder forward passes. Cache -> _ctrl/_c11_{split}.npz + meta.joblib"""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, joblib
import l2_c8 as C8, l2_c9 as C9
from _run_c9 import starts_of
OUT = C8.OUT; DS = C8.DS; CAP = 20; log = lambda *a: print(*a, flush=True); T0 = time.time()
EFEAT = ["c8c_oof_score", "c8c_oof_rank", "base_score", "c_dense", "c_splade", "c_offset", "c_mixture",
         "c_relation", "rk_dense", "rk_splade", "rk_offset", "rk_mixture", "rk_relation", "rel_mask",
         "max_contrib", "second_contrib", "n_top5", "votes_top10"]
EIDX = [C9.C9_FEATNAMES.index(e) for e in EFEAT]
_Q = {}; _N = {}; _C = {}; _OFF = {}


def qvec(ds, qi):
    if ds not in _Q:
        _Q[ds] = np.load(f"data/ukb_storage/{ds}/gte_qwen/queries_train.npy", mmap_mode="r")
        _QM = np.load(f"{OUT}/_c11_qmap_{ds}.npz")["qmap"]; _Q[ds + "_m"] = _QM
    return np.asarray(_Q[ds][int(_Q[ds + "_m"][qi])], np.float32)


def dvecs(ds, qi, pool_local):
    if ds not in _N:
        _N[ds] = np.load(f"data/ukb_storage/{ds}/gte_qwen/nodes.npy", mmap_mode="r")
        _C[ds] = np.load(f"data/l2_corpus/{ds}/train/cand_ids.npy", mmap_mode="r")
        _OFF[ds] = np.load(f"data/l2_corpus/{ds}/train/query_offsets.npy")
    s = int(_OFF[ds][qi]); gids = np.asarray(_C[ds][s + pool_local], np.int64)
    return np.asarray(_N[ds][gids], np.float32)


def build(split_name, B, X, scores):
    st = starts_of(B["groups"]); QV = []; DV = []; E = []; C8b = []; Y = []; G = []; meta = []
    for i, (m, g) in enumerate(zip(B["meta"], B["groups"])):
        a = st[i]; sc = scores[a:a + g]; ds = m["ds"]; pool = m["pool"]
        order = np.argsort(-sc, kind="stable"); win = order[:min(CAP, g)]         # top20 rows (pool-local positions)
        pool_win = pool[win]                                                       # candidate local indices in query
        QV.append(qvec(ds, m["qi"])); DV.append(dvecs(ds, m["qi"], pool_win))
        E.append(X[a + win][:, EIDX]); C8b.append(sc[win])
        goldset = set(m["gold"].tolist()); Y.append(np.array([1 if int(pl) in goldset else 0 for pl in pool_win], np.int8))
        G.append(len(win))
        meta.append({"qi": int(m["qi"]), "ds": ds, "pool": pool, "brank": m["brank"], "n": m["n"],
                     "gold": m["gold"], "win_pool_local": pool_win.astype(np.int64)})
        if i % 4000 == 0: log(f"[{time.time()-T0:.0f}s] {split_name} {i}/{len(B['meta'])}")
    qv = np.asarray(QV, np.float16); dv = np.concatenate(DV).astype(np.float16); E = np.concatenate(E).astype(np.float32)
    c8b = np.concatenate(C8b).astype(np.float32); y = np.concatenate(Y); G = np.asarray(G, np.int64)
    np.savez(f"{OUT}/_c11_{split_name}.npz", qv=qv, dv=dv, E=E, c8c=c8b, y=y, groups=G)
    joblib.dump(meta, f"{OUT}/_c11_{split_name}_meta.joblib")
    log(f"[{time.time()-T0:.0f}s] {split_name}: Nq={len(G)} rows={len(y)} pos={int(y.sum())} dv={dv.shape}")


def main():
    cache = joblib.load(f"{OUT}/_c10b_cache.joblib")
    log(f"[{time.time()-T0:.0f}s] cache loaded")
    build("train", cache["B_tr"], cache["X_tr"], cache["oof_tr"])
    build("dev", cache["B_dev"], cache["X_dev"], cache["s_dev"])
    build("di", cache["B_di"], cache["X_di"], cache["s_di"])
    json.dump({"EFEAT": EFEAT, "EIDX": EIDX, "CAP": CAP}, open(f"{OUT}/_c11_prep_meta.json", "w"), indent=1)
    log("C11_PREP_DONE")


if __name__ == "__main__":
    main()
