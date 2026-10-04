"""DOWNSTREAM DIAGNOSTIC: feed the CORRECTED pre-partition Track-A P50 hard union into the FROZEN L2.

Frozen backbone = the G1 ZERO_SHOT bundle (C7b soft-archetype fusion -> C8c XGBRanker -> C11a interaction
MLP), loaded unmodified from results/GENERALIZATION/_g1_backbone. Nothing is trained, retrained or tuned.

Unlike Q2 (which only APPENDED structural candidates to a fixed P50), the corrected Track-A REPLACES
partitions, so the candidate set changes by ADDITION *and* EVICTION. build_scope() therefore takes a keep
mask over the stored corpus slice as well as an added list.

  kept    : candidate is in the stored corpus AND in the new hard union -> ALL stored expert scores reused
            bit-exactly (dense/splade/offset/mixture/relation + mask + label)
  evicted : in the stored corpus, not in the new union -> dropped
  added   : in the new union, not in the stored corpus -> deterministic scoring, no encoder pass:
              dense   = cos(q, node)                              [exact]
              offset  = node . universal_offset_head(q, dtop1)    [exact, frozen universal head]
              mixture = max_k node . universal_mixture_head_k     [exact, frozen universal head]
              splade  = per-query MIN - 1                         [CONSERVATIVE approximation]
              relation= ABSTAIN (mask False -> RRF contribution exactly 0)   [default]
            Only SPLADE is approximate and it UNDER-credits added candidates, so any downstream GAIN is a
            lower bound and any downstream LOSS is not an artefact of the approximation.

GATE: M=0 must reproduce the stored-corpus frozen baseline EXACTLY (no eviction, no addition).

  python scratchpad/_ta_down.py <ds> <cfg> <M>      cfg in {A1a,A1b,A2,A2dOnly,A2sOnly}
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np, torch, joblib
import l2_c8 as C8, l2_c8c as C8C, l2_c9 as C9, l2_c6 as C6
import l2_shapley as SH
import l2_lib as L
from l2_lib import OffsetHead, MixtureHead
import l2_c11 as M11
import _run_c11 as R11
import _ta_prepartition as TA

# ---------------------------------------------------------------------------------------------------
# EVALUATION-DOMAIN EXTENSION (not a scoring change).
#
# The corrected Track-A can EVICT a partition that held a query's only gold, leaving that query with ZERO
# golds in scope. The frozen scorer was never built for that case -- l2_c8.topk_metrics does gr.min() and
# g5/ng and raises on an empty gold list, because l2_c8.valid_queries guarantees >=1 stored gold. It is a
# domain gap, not a metric disagreement.
#
# Dropping such queries would be WRONG and self-serving: a config could raise its own average simply by
# evicting the golds of the queries it does worst on. The unbiased treatment is to score them 0 on every
# metric -- which is what the metric definitions give in the limit (no gold retrievable => nDCG 0,
# recall 0, all@k 0, MRR 0).
#
# This is done by WRAPPING C8.topk_metrics here, so l2_c8.py stays byte-identical on disk and the frozen
# pipeline is unmodified. The wrapper is a provable NO-OP whenever every query has >=1 in-scope gold --
# which is exactly the BASE case, so the M=0 parity gate is unaffected.
_TOPK_ORIG = C8.topk_metrics
_ZG = {"n": 0}
def _topk_metrics_zero_safe(score, gold):
    if len(gold):
        return _TOPK_ORIG(score, gold)
    _ZG["n"] += 1
    return {"any5": 0.0, "golds_in5": 0, "ng": 0, "recall5": 0.0, "ndcg5": 0.0, "ndcg50": 0.0,
            "mrr": 0.0, "all5": 0.0, "all10": 0.0, "all50": 0.0, "all5_feas": 0.0,
            "best": 1 << 30, "worst": 1 << 30, "gr": np.zeros(0, np.int64)}
C8.topk_metrics = _topk_metrics_zero_safe

DS = sys.argv[1] if len(sys.argv) > 1 else "2wiki_clean"
CFG = sys.argv[2] if len(sys.argv) > 2 else "A2"
MV = int(sys.argv[3]) if len(sys.argv) > 3 else 32
SPLIT = "val"; CORP = "data/l2_corpus"; HEADS = "results/L2/_heads"
BASE_EMB = f"data/ukb_storage/{DS}/gte_qwen/"
OUT = f"results/GENERALIZATION/_g2_tadown_{DS}_{CFG}_M{MV}.json"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def build_scope(qi_list, R, W, ra, di, keep_by_q, added_by_q, added_sc):
    """EXACT re-expression of the frozen 28-feature C8c contract over an ARBITRARY candidate set."""
    off = R["query_offsets"]
    X28 = []; Xex = []; Y = []; Gg = []; meta = []; cand_by_q = []
    for r, qi in enumerate(qi_list):
        s, e = int(off[qi]), int(off[qi + 1])
        k = keep_by_q[qi]
        dense = R["dense_score"][s:e][k].astype(np.float64)
        splade = R["splade_scope_score"][s:e][k].astype(np.float64)
        offs = R["offset_score"][s:e][k].astype(np.float64)
        mix = R["mixture_score"][s:e][k].astype(np.float64)
        rel = R["relation_qwen_score"][s:e][k].astype(np.float64)
        relm = (R["relation_mask"][s:e][k] > 0)
        lab = (R["labels"][s:e][k] == 1)
        add = added_by_q.get(qi, [])
        if len(add):
            a = added_sc[qi]
            dense = np.concatenate([dense, a["dense"]]); splade = np.concatenate([splade, a["splade"]])
            offs = np.concatenate([offs, a["offset"]]); mix = np.concatenate([mix, a["mixture"]])
            rel = np.concatenate([rel, np.zeros(len(add))]); relm = np.concatenate([relm, np.zeros(len(add), bool)])
            lab = np.concatenate([lab, a["label"].astype(bool)])
        n = len(dense)
        C = SH.contribs(dense, splade, offs, mix, rel, relm)
        w = W[r]
        base = (w[:, None] * C).sum(0); brank = C8.ranks_from_score(base)
        rk = np.stack([C8.ranks_from_score(C[i]) for i in range(5)]).astype(np.float64)
        gold = set(np.where(lab)[0].tolist())
        pool = np.argsort(-base, kind="stable")[:min(C8C.POOL, n)]
        min4 = rk[:4].min(0); mean4 = rk[:4].mean(0); maxc = C.max(0); votes10 = (rk < 10).sum(0)
        for pp, li in enumerate(pool):
            X28.append([brank[li], base[li], pp, rk[0, li], rk[1, li], rk[2, li], rk[3, li], rk[4, li],
                        C[0, li], C[1, li], C[2, li], C[3, li], C[4, li], (1.0 if relm[li] else 0.0), rel[li],
                        min4[li], mean4[li], maxc[li], rk[0, li] - rk[1, li], rk[2, li] - rk[3, li], votes10[li],
                        w[0], w[1], w[2], w[3], w[4], float(ra[r]), float(di[r])])
            Y.append(1 if int(li) in gold else 0)
        Xex.append(C9._extra_cols(C[:, pool], rk[:, pool], w))
        Gg.append(len(pool))
        meta.append({"qi": int(qi), "pool": pool.astype(np.int64), "brank": brank,
                     "gold": np.array(sorted(gold), np.int64), "n": n,
                     "n_added": int(len(add)), "n_evicted": int((~k).sum())})
        cand_by_q.append(np.concatenate([R["cand_ids"][s:e][k].astype(np.int64),
                                         np.asarray(add, np.int64)]) if len(add)
                         else R["cand_ids"][s:e][k].astype(np.int64))
    return {"X28": np.asarray(X28, np.float32), "Xex": np.concatenate(Xex).astype(np.float32),
            "y": np.asarray(Y, np.int8), "groups": np.asarray(Gg, np.int64), "meta": meta}, cand_by_q


def main():
    import _q2_eval as Q2                                   # frozen helpers: load_frozen / build_sp_aug
    bdir = "results/GENERALIZATION/_g1_backbone"
    log(f"=== TRACK-A DOWNSTREAM {DS} cfg={CFG} M={MV}  frozen backbone={bdir} ===")
    base_full, c8c, ma, em11, es11 = Q2.load_frozen(bdir); log("frozen backbone loaded (NOT modified)")

    R = C6._load_raw(DS, SPLIT)
    R["cand_ids"] = np.load(f"{CORP}/{DS}/{SPLIT}/cand_ids.npy")
    R["query_meta"] = json.load(open(f"{CORP}/{DS}/{SPLIT}/query_meta.json"))
    off = R["query_offsets"]; qm = R["query_meta"]
    log("stored corpus loaded nq=%d cands=%d" % (len(off) - 1, len(R["cand_ids"])))

    hard, mem, npart, (adjp, adji), deg, id2row = TA.load_topology(DS, log)
    docs_of_part = [np.where(hard == p)[0] for p in range(npart)]

    valid = sorted(set(int(x) for x in C8.precompute_arch(DS, SPLIT)["qi"].tolist()))
    CAPQ = int(os.environ.get("TAD_CAP", "0"))
    qi_list = valid[:CAPQ] if CAPQ > 0 else valid
    log(f"eval queries={len(qi_list)} (frozen valid universe={len(valid)})")
    rows = [int(qm[qi]["row_all"]) for qi in qi_list]

    dense_all = np.load(BASE_EMB + "dense_top200_all.npy", mmap_mode="r")
    splade_all = np.load(BASE_EMB + "splade_top200_all.npy", mmap_mode="r")
    qall = np.load(BASE_EMB + "queries_all.npy", mmap_mode="r")
    nodes = np.load(BASE_EMB + "nodes.npy", mmap_mode="r")
    dK = np.stack([np.asarray(dense_all[i][:TA.K_LOCK]) for i in rows]).astype(np.int64)
    sK = np.stack([np.asarray(splade_all[i][:TA.K_LOCK]) for i in rows]).astype(np.int64)

    Xn = np.array(nodes, np.float32)
    for a in range(0, len(Xn), 100000):
        b = min(a + 100000, len(Xn)); Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)

    PR_d = TA.partition_ranking(list(dK), mem, npart); PR_s = TA.partition_ranking(list(sK), mem, npart)
    base_rank = TA.rrf_partitions([PR_d, PR_s], npart)
    log("BASE partition ranking done")

    # ---- the corrected variant's partition ranking (identical algebra to _ta_run.py) -------------
    nq = len(qi_list)
    # NOTE: M<=0 short-circuits to BASE only for CFG=BASE. A1b at M=0 is NOT BASE -- it is node-level RRF
    # with no expansion, which is precisely the config that isolates the RRF-LEVEL effect from structure.
    if CFG == "BASE":
        new_rank = base_rank
    else:
        def expand(seed_fn):
            out = []
            for qi in range(nq):
                if MV <= 0:
                    out.append([]); continue
                sd = seed_fn(qi)
                r_q = TA.residual(np.asarray(qall[rows[qi]], np.float64), sd, Xn)
                a, _vm, _e = TA.expand_dir(sd, r_q, adjp, adji, deg, Xn, MV)
                out.append(a)
            return out
        dpresent = [set(int(x) for x in dK[qi]) for qi in range(nq)]
        spresent = [set(int(x) for x in sK[qi]) for qi in range(nq)]
        if CFG.startswith("A1"):
            fused = [TA.node_rrf(dK[qi], sK[qi]) for qi in range(nq)]
            add = expand(lambda qi: fused[qi][:TA.SEED_K])
            if CFG == "A1a":
                fp = [set(int(x) for x in fused[qi]) for qi in range(nq)]
                gl = [[v for v in add[qi] if v not in fp[qi]] for qi in range(nq)]
                PR_g = TA.partition_ranking(gl, mem, npart)
                msk = np.array([len(x) > 0 for x in gl], bool)
                new_rank = TA.rrf_partitions([PR_d, PR_s, PR_g], npart, masks=[None, None, msk])
            else:                                            # A1b: one fused node stream
                fl = [list(fused[qi]) + [v for v in add[qi] if v not in set(int(x) for x in fused[qi])]
                      for qi in range(nq)]
                new_rank = TA.rrf_partitions([TA.partition_ranking(fl, mem, npart)], npart)
        else:
            ad = expand(lambda qi: [int(x) for x in dK[qi][:TA.SEED_K]])
            asd = expand(lambda qi: [int(x) for x in sK[qi][:TA.SEED_K]])
            dl = [list(dK[qi]) + [v for v in ad[qi] if v not in dpresent[qi]] for qi in range(nq)]
            sl = [list(sK[qi]) + [v for v in asd[qi] if v not in spresent[qi]] for qi in range(nq)]
            PR_d2 = TA.partition_ranking(dl, mem, npart); PR_s2 = TA.partition_ranking(sl, mem, npart)
            new_rank = TA.rrf_partitions([PR_d2 if CFG in ("A2", "A2dOnly") else PR_d,
                                          PR_s2 if CFG in ("A2", "A2sOnly") else PR_s], npart)
    log(f"{CFG} partition ranking done")

    # ---- keep / evict / add against the stored corpus --------------------------------------------
    Dn, Qn, dtop = L.load_embeddings(DS)
    oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"{HEADS}/universal_offset_src_gteqwen.pt", map_location="cpu")); oh.eval()
    mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"{HEADS}/universal_mixture_src_gteqwen.pt", map_location="cpu")); mh.eval()
    labels_all = R["labels"]
    # TRUE gold universe, independent of the stored corpus. The stored corpus only contains golds that were
    # already inside the BASE scope, so deriving labels from it would silently mark a NEWLY ADMITTED gold as
    # a non-gold -- biasing every metric against the new scope. Gold is used ONLY to LABEL for evaluation,
    # never in scope construction (which ran above, before this block).
    _j = json.load(open(BASE_EMB + "query_ids_all.json")); _golds = _j["golds"]
    true_gold = [set(id2row[g] for g in _golds[rr] if g in id2row) for rr in rows]
    gold_by_q = {}; keep_by_q = {}; added_by_q = {}; added_sc = {}
    n_ev = n_ad = n_gold_ev = n_gold_ad = 0; n_lbl_chk = 0
    for r_, qi in enumerate(qi_list):
        s, e = int(off[qi]), int(off[qi + 1])
        cq = R["cand_ids"][s:e].astype(np.int64)
        sel = new_rank[r_][:TA.P_MAIN]
        newdocs = np.concatenate([docs_of_part[p] for p in sel]) if len(sel) else np.zeros(0, np.int64)
        # vectorised set algebra -- the Python-set version was O(15M) interpreter ops over the corpus
        k = np.isin(cq, newdocs, assume_unique=False)
        keep_by_q[qi] = k
        gold_by_q[qi] = true_gold[r_]
        gl = labels_all[s:e] == 1
        stored_g = set(int(x) for x in cq[gl])
        n_lbl_chk += int(stored_g <= gold_by_q[qi])
        n_ev += int((~k).sum())
        n_gold_ev += int((gl & ~k).sum())
        add = newdocs[~np.isin(newdocs, cq, assume_unique=False)]
        if not len(add):
            added_by_q[qi] = []; continue
        add = np.ascontiguousarray(add, np.int64); n_ad += len(add)
        rr = rows[r_]
        qn = np.asarray(qall[rr], np.float32); qn = qn / (np.linalg.norm(qn) + 1e-9)
        with torch.no_grad():
            qt = torch.from_numpy(qn).unsqueeze(0)
            si = int(dtop[rr, 0]); seedt = Dn[si:si + 1]
            po = oh(qt, seedt)[0]; pm = mh(qt, seedt)[0]
            Ca = Dn[torch.from_numpy(add)]
            dsc = (Ca @ torch.from_numpy(qn)).numpy().astype(np.float64)
            osc = (Ca @ po).numpy().astype(np.float64)
            msc = (Ca @ pm.T).max(dim=1).values.numpy().astype(np.float64)
        sp_min = float(R["splade_scope_score"][s:e].min()) - 1.0
        gset = gold_by_q[qi]
        lab = np.fromiter((1 if int(v) in gset else 0 for v in add), np.int8, len(add))
        n_gold_ad += int(lab.sum())
        added_by_q[qi] = add
        added_sc[qi] = {"dense": dsc, "offset": osc, "mixture": msc,
                        "splade": np.full(len(add), sp_min, np.float64), "label": lab}
    log(f"scope delta: evicted={n_ev} (gold {n_gold_ev})  added={n_ad} (gold {n_gold_ad})")
    log(f"label consistency: stored-corpus golds are a subset of the true gold set in "
        f"{n_lbl_chk}/{len(qi_list)} queries")

    qi_arr = np.array(qi_list, np.int64)
    W = C8.soft_weights(*base_full[:3], DS, SPLIT, qi_arr)
    ra, di = C9._qmeta(DS, SPLIT, qi_arr)
    B, cand_by_q = build_scope(qi_list, R, W, ra, di, keep_by_q, added_by_q, added_sc)
    sv = c8c.predict(B["X28"]).astype(np.float64)
    spv = Q2.build_sp_aug(B, cand_by_q, sv, nodes, qall, qm)
    spv["_t"] = R11.prep_tensors(spv, em11, es11)
    res = R11.evaluate(ma, spv, B, sv)
    log(f"queries left with ZERO in-scope golds by this config: {_ZG['n']} (scored 0 on every metric, "
        f"not dropped)")
    agg = {"ndcg5": round(res["ndcg5"], 4), "recall5_macro": round(res["recall5_macro"], 4),
           "all5": round(res["all5"], 4), "all50": round(res["all50"], 4),
           "mrr": round(res["mrr"], 4), "n": res["n"]}
    out = {"DS": DS, "CFG": CFG, "M": MV, "n_queries": len(qi_list), "AGG": agg,
           "SCOPE_DELTA": {"evicted_cands": n_ev, "evicted_golds": n_gold_ev,
                           "added_cands": n_ad, "added_golds": n_gold_ad},
           "FROZEN_L2": "C7b->C8c->C11a ZERO_SHOT bundle, unmodified; no training, no tuning",
           "APPROX": "splade=per-query MIN-1 and relation=ABSTAIN for ADDED candidates only (conservative)",
           "ZERO_GOLD_QUERIES": _ZG["n"]}
    # per-query metrics for a PAIRED test against BASE (qi_list is the frozen valid universe, so the
    # arrays align across configs by index)
    pq = res.get("_perq", {})
    if pq:
        # NOTE: _perq itself contains a "qi" key, so namespace ours to avoid a kwarg collision
        np.savez(OUT.replace(".json", "_perq.npz"), eval_qi=np.array(qi_list, np.int64),
                 **{("pq_" + k): np.asarray(v, np.float64) for k, v in pq.items()})
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
    log(f"AGG {agg}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
