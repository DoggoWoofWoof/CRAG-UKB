"""C11 EMBEDDING ALIGNMENT AUDIT — establish REPRESENTATION ALIGNMENT (rank parity), not score equality.
No new encoder forward pass. Uses ONLY cached artifacts:
  cached query emb : gte_qwen/queries_train.npy  (row via recovered qmap)  AND independently queries_all.npy[row_all]
  cached node emb  : gte_qwen/nodes.npy
  L2/P50 scope     : l2_corpus/{ds}/train/{cand_ids,query_offsets}.npy
  frozen dense rank: l2_corpus/{ds}/train/dense_rank.npy (+ dense_score.npy)   <- canonical dense expert artifact
  frozen full-corpus dense retrieval: gte_qwen/dense_top200_all.npy (row via query_meta row_all)
Reports Spearman/Kendall/top-k overlap/top1-agree per-query + aggregate; Pearson-vs-Spearman transform check;
full-corpus top10/50/200 overlap. Emits QUERY_EMBED_ALIGNMENT / NODE_EMBED_ALIGNMENT / DENSE_RANK_PARITY."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
import l2_c8 as C8
OUT = C8.OUT; DS = C8.DS; T0 = time.time(); log = lambda *a: print(*a, flush=True)
N_PER = 150          # deterministic per-query sample (>=100)
N_FULL = 10          # full-corpus sanity sample
KS = [5, 10, 50]


def overlap(a_topk, b_topk):
    return len(set(a_topk.tolist()) & set(b_topk.tolist())) / max(len(a_topk), 1)


def audit_ds(ds):
    G = f"data/ukb_storage/{ds}/gte_qwen"; L = f"data/l2_corpus/{ds}/train"
    off = np.load(f"{L}/query_offsets.npy"); Nq = len(off) - 1
    cand = np.load(f"{L}/cand_ids.npy", mmap_mode="r")
    drank = np.load(f"{L}/dense_rank.npy", mmap_mode="r")
    dsc = np.load(f"{L}/dense_score.npy", mmap_mode="r")
    nodes = np.load(f"{G}/nodes.npy")                       # (Nn,1536) exact cached node embs
    qtr = np.load(f"{G}/queries_train.npy")                 # (Nq,1536)
    qall = np.load(f"{G}/queries_all.npy", mmap_mode="r")   # (Nall,1536) canonical, row_all-indexed
    qmap = np.load(f"{OUT}/_c11_qmap_{ds}.npz")["qmap"]     # corpus_qi -> queries_train row (C11's mapping)
    qmeta = json.load(open(f"{L}/query_meta.json"))         # per corpus_qi: row_all
    # node id range sanity
    nid_ok = int(np.asarray(cand[:1000000]).max()) < nodes.shape[0]

    idx = np.linspace(0, Nq - 1, N_PER).astype(int)        # deterministic even sample
    idx = np.unique(idx)
    sp = []; kt = []; ov = {k: [] for k in KS}; t1 = []; pear = []; spr_val = []
    qmap_rowall_cos = []; drank_matches_argsort = []
    for qi in idx:
        s, e = int(off[qi]), int(off[qi + 1]); n = e - s
        cids = np.asarray(cand[s:e], np.int64)
        dr = np.asarray(drank[s:e], np.int64)              # frozen dense rank within scope (0=best)
        ds_v = np.asarray(dsc[s:e], np.float32)            # frozen dense score
        q = qtr[int(qmap[qi])].astype(np.float32)          # cached query emb C11 uses
        D = nodes[cids].astype(np.float32)
        raw = D @ q                                        # raw_sim = q . node  (both L2-normalized)
        # qmap vs canonical row_all cross-check (independent of dense scores)
        ra = int(qmeta[qi]["row_all"]); q2 = qall[ra].astype(np.float32)
        qmap_rowall_cos.append(float(q @ q2 / ((np.linalg.norm(q) * np.linalg.norm(q2)) + 1e-9)))
        # frozen dense order from the rank artifact. dense_rank uses -1 SENTINEL for "outside dense top-N"
        # (truncated top-~199 within scope; rank 0 = best). Map -1 -> +inf so the tail sorts last.
        dr_fix = np.where(dr < 0, 1 << 30, dr)
        dense_order = np.argsort(dr_fix, kind="stable")               # scope-local positions, best-first
        # consistency in the RANKED region: does the frozen dense top-Nr equal argsort(-score) top-Nr?
        nr = int((dr >= 0).sum()); asc = np.argsort(-ds_v, kind="stable")[:nr]
        drank_matches_argsort.append(float(set(dense_order[:nr].tolist()) == set(asc.tolist())))
        raw_order = np.argsort(-raw, kind="stable")
        # rank correlation between raw_sim and frozen dense (use score; higher=better both)
        sp.append(spearmanr(raw, ds_v).statistic)
        kt.append(kendalltau(raw, ds_v).statistic)
        pear.append(pearsonr(raw, ds_v).statistic)
        spr_val.append(spearmanr(raw, ds_v).statistic)
        for k in KS:
            kk = min(k, n)
            ov[k].append(overlap(raw_order[:kk], dense_order[:kk]))
        t1.append(float(raw_order[0] == dense_order[0]))
    agg = dict(
        n_queries=int(len(idx)),
        spearman_mean=float(np.mean(sp)), spearman_min=float(np.min(sp)),
        kendall_mean=float(np.mean(kt)), kendall_min=float(np.min(kt)),
        top5_overlap=float(np.mean(ov[5])), top10_overlap=float(np.mean(ov[10])),
        top50_overlap=float(np.mean(ov[50])),
        top1_agree_frac=float(np.mean(t1)),
        pearson_mean=float(np.mean(pear)), spearman_val_mean=float(np.mean(spr_val)),
        qmap_vs_rowall_cos_mean=float(np.mean(qmap_rowall_cos)),
        qmap_vs_rowall_cos_min=float(np.min(qmap_rowall_cos)),
        dense_rank_eq_argsort_score_frac=float(np.mean(drank_matches_argsort)),
        node_id_in_range=bool(nid_ok),
    )
    # ---- full-corpus sanity: cached-vector top200 over ALL nodes vs frozen dense_top200_all ----
    d200 = np.load(f"{G}/dense_top200_all.npy", mmap_mode="r")
    fidx = np.linspace(0, Nq - 1, N_FULL).astype(int)
    fc = {10: [], 50: [], 200: []}
    for qi in fidx:
        q = qtr[int(qmap[qi])].astype(np.float32)
        scores = nodes @ q                                 # (Nn,) exact cached dot vs ALL canonical nodes
        my200 = np.argpartition(-scores, 200)[:200]
        my200 = my200[np.argsort(-scores[my200])]
        ra = int(qmeta[qi]["row_all"]); frozen200 = np.asarray(d200[ra], np.int64)
        for k in (10, 50, 200):
            fc[k].append(overlap(my200[:k], frozen200[:k]))
    agg["fullcorpus_top10_overlap"] = float(np.mean(fc[10]))
    agg["fullcorpus_top50_overlap"] = float(np.mean(fc[50]))
    agg["fullcorpus_top200_overlap"] = float(np.mean(fc[200]))
    log(f"[{time.time()-T0:.0f}s] {ds}: " + json.dumps(agg, indent=1))
    return agg


def main():
    res = {}
    for ds in DS:
        res[ds] = audit_ds(ds)
    # ---- pass conditions (near-parity, NOT score equality) ----
    def ok(f):  # aggregate condition across datasets
        return all(f(res[d]) for d in DS)
    query_align = ok(lambda a: a["qmap_vs_rowall_cos_min"] >= 0.999) and ok(lambda a: a["top1_agree_frac"] >= 0.90)
    node_align = ok(lambda a: a["fullcorpus_top200_overlap"] >= 0.90)
    rank_parity = (ok(lambda a: a["spearman_mean"] >= 0.98) and ok(lambda a: a["top10_overlap"] >= 0.90)
                   and ok(lambda a: a["top50_overlap"] >= 0.90) and ok(lambda a: a["top1_agree_frac"] >= 0.90))
    transformed = ok(lambda a: a["spearman_val_mean"] >= 0.98 and a["pearson_mean"] < 0.999)
    gates = {
        "QUERY_EMBED_ALIGNMENT": "PASS" if query_align else "FAIL",
        "NODE_EMBED_ALIGNMENT": "PASS" if node_align else "FAIL",
        "DENSE_RANK_PARITY": "PASS" if rank_parity else "FAIL",
        "DENSE_SCORE_IS_TRANSFORMED": "YES" if transformed else "NO",
        "NO_NEW_ENCODER_FORWARD": "PASS",
    }
    out = {"phase": "C11 embedding alignment audit", "N_PER": N_PER, "N_FULL": N_FULL,
           "per_dataset": res, "GATES": gates}
    json.dump(out, open(f"{OUT}/_c11_align_audit.json", "w"), indent=1)
    log("GATES " + json.dumps(gates))
    log("C11_ALIGN_DONE")


if __name__ == "__main__":
    main()
