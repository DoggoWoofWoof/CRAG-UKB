"""E0-E3 orchestration on the canonical C/P50 pilot. Stage-driven, checkpoints to JSON.

Stages: labels | e0e1 | smoke | train | e3b | report | all
All ranking metrics use the shared harness (3 denominators). Heads retrained on C/P50 (reference-only olds ignored).
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scratchpad"))
import l2_lib as L

DATASETS = ["2wiki_clean", "musique_clean"]
RES_JSON = "results/L2/L2_E0_E3_RESULTS.json"
REGIME_DIR = "results/L2/_regime_raw"


def log(*a): print(*a, flush=True)


def load_res():
    if os.path.exists(RES_JSON):
        return json.load(open(RES_JSON))
    return {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "config": {
        "topology": "C", "router": "Dense+SPLADE", "RRF_K0": L.RRF_K0, "K": 100, "P": 50,
        "TAU": L.TAU, "fusion_note": "RRF is rank-based (scale-free); znorm is per-query z-score sum"},
        "datasets": {d: {} for d in DATASETS}}


def save_res(r):
    r["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("results/L2", exist_ok=True)
    json.dump(r, open(RES_JSON, "w"), indent=2)


# ------------------------------------------------------------------ labels
def stage_labels():
    r = load_res()
    for ds in DATASETS:
        r["datasets"].setdefault(ds, {})
        lab = {}
        for split in ("train", "val", "test"):
            S = L.load_split(ds, split)
            meta = S["meta"]; Nq = len(meta)
            exp = np.array([m["N_GOLD_EXPECTED_INCORP"] for m in meta])
            insc = np.array([m["N_GOLD_IN_SCOPE"] for m in meta])
            raw = np.array([m["N_GOLD_EXPECTED_RAW"] for m in meta])
            stat = [m["L1_STATUS"] for m in meta]
            # positive-label definition = label==1 iff a doc that maps a gold node id is in the P50 scope
            lab[split] = {
                "Nq": Nq,
                "positive_label_def": "labels[cand]==1 iff cand is a gold doc (gold node-id mapped into corpus) present in the P50 scope",
                "N_GOLD_EXPECTED_RAW_mean": float(raw.mean()), "N_GOLD_EXPECTED_INCORP_mean": float(exp.mean()),
                "N_GOLD_IN_SCOPE_mean": float(insc.mean()),
                "multi_gold_frac": float((exp > 1).mean()),
                "L1_ANY_FAIL": int(sum(s == "L1_ANY_FAIL" for s in stat)),
                "L1_PARTIAL": int(sum(s == "L1_PARTIAL" for s in stat)),
                "L1_ALL_SUCCESS": int(sum(s == "L1_ALL_SUCCESS" for s in stat)),
                "any_gold_in_scope": int((insc >= 1).sum()), "all_gold_in_scope": int((insc >= exp).sum() and int(sum((insc >= exp) & (exp > 0)))),
            }
            log(f"[{ds}/{split}] Nq={Nq} expINCORP={exp.mean():.2f} inScope={insc.mean():.2f} "
                f"multi={100*(exp>1).mean():.1f}% L1_FAIL={lab[split]['L1_ANY_FAIL']} "
                f"PARTIAL={lab[split]['L1_PARTIAL']} SUCCESS={lab[split]['L1_ALL_SUCCESS']}")
        r["datasets"][ds]["label_semantics"] = lab
    save_res(r)
    log("L1_ANY_FAIL queries have N_GOLD_IN_SCOPE==0 -> excluded from COND_ANY/COND_ALL ranking denominators "
        "(cannot be L2 ranking failures); they appear only in ALL_EVAL_QUERIES as end-to-end misses.")


# ------------------------------------------------------------------ E0 / E1
def _scorer_dense(S): return lambda qi: L.dense_scores(S, qi)
def _scorer_splade(S): return lambda qi: L.splade_scores(S, qi)
def _scorer_rrf(S, experts): return lambda qi: L.rrf_fuse([e(S)(qi) for e in experts])
def _scorer_znorm(S, experts): return lambda qi: L.znorm_fuse([e(S)(qi) for e in experts])


def stage_e0e1(splits=("val",)):
    r = load_res()
    for ds in DATASETS:
        for split in splits:
            S = L.load_split(ds, split)
            configs = {
                "E0_dense": _scorer_dense(S),
                "E0s_splade": _scorer_splade(S),
                "E1a_dense_splade_RRF": _scorer_rrf(S, [_scorer_dense, _scorer_splade]),
                "E1b_dense_splade_ZNORM": _scorer_znorm(S, [_scorer_dense, _scorer_splade]),
            }
            r["datasets"].setdefault(ds, {}).setdefault("results", {})
            for name, fn in configs.items():
                t0 = time.time()
                m = L.eval_ranking(S, fn)
                r["datasets"][ds]["results"].setdefault(name, {})[split] = m
                cond = m["COND_ANY_GOLD_IN_SCOPE"]
                log(f"[{ds}/{split}] {name:26s} MRR={cond['MRR']:.4f} R@5={cond['R@5']:.4f} "
                    f"R@10={cond['R@10']:.4f} ANY@10={cond['ANY@10']:.4f} ALL@10={cond['ALL@10']:.4f} ({time.time()-t0:.1f}s)")
            save_res(r)


# ------------------------------------------------------------------ smoke
def stage_smoke():
    for ds in DATASETS:
        for kind in ("offset", "mixture"):
            log(f"--- SMOKE {ds}/{kind} ---")
            head, tlog = L.train_head(ds, kind, epochs=3, smoke_n=400, log=log)
            lc = tlog["loss_curve"]
            assert all(np.isfinite(x) for x in lc), "NaN in loss"
            log(f"  loss_curve={['%.4f'%x for x in lc]} decreasing={lc[-1] < lc[0]} "
                f"negs/q={tlog['mean_negs_per_query']:.1f} pos_collisions={tlog['positive_collisions']} (MUST be 0)")
            Sv = L.load_split(ds, "val")
            Dn, Qn, dtop = L.load_embeddings(ds)
            fn = lambda qi, h=head, k=kind: L.head_scores(Sv, qi, h, Dn, Qn, dtop, k)
            m = L.eval_ranking(Sv, fn)
            log(f"  VAL head-only MRR(COND_ANY)={m['COND_ANY_GOLD_IN_SCOPE']['MRR']:.4f} (evaluator OK)")


# ------------------------------------------------------------------ train (val checkpoint selection) + E2/E3
def _fused_val_mrr(S, base_experts, head, kind, Dn, Qn, dtop):
    hs = lambda qi: L.head_scores(S, qi, head, Dn, Qn, dtop, kind)
    fn = lambda qi: L.rrf_fuse([e(S)(qi) for e in base_experts] + [hs(qi)])
    m = L.eval_ranking(S, fn)
    return m["COND_ANY_GOLD_IN_SCOPE"]["MRR"], fn, m


def stage_train():
    r = load_res()
    os.makedirs("results/L2/_heads", exist_ok=True)
    for ds in DATASETS:
        Dn, Qn, dtop = L.load_embeddings(ds)
        base = [_scorer_dense, _scorer_splade]           # E1 base = dense+splade
        r["datasets"].setdefault(ds, {}).setdefault("results", {})
        r["datasets"][ds].setdefault("train_logs", {})
        offset_head = None
        for kind, ename in (("offset", "E2_dense_splade_offset_RRF"), ("mixture", "E3_dense_splade_offset_mixture_RRF")):
            log(f"=== TRAIN {ds}/{kind} (val checkpoint selection) ===")
            head, sel = L.train_head(ds, kind, epochs=10, val_select=True, base_experts=base, patience=3, log=log)
            log(f"    sampler: queries={sel['queries']} negs/q={sel['mean_negs_per_query']:.1f} "
                f"pos_collisions={sel['positive_collisions']} (MUST be 0)")
            torch.save(head.state_dict(), f"results/L2/_heads/{ds}_{kind}_cp50.pt")
            r["datasets"][ds]["train_logs"][kind] = sel
            if kind == "offset":
                offset_head = head
                experts = [_scorer_dense, _scorer_splade, _mk_head_scorer(head, kind, Dn, Qn, dtop)]
            else:
                experts = [_scorer_dense, _scorer_splade,
                           _mk_head_scorer(offset_head, "offset", Dn, Qn, dtop),
                           _mk_head_scorer(head, "mixture", Dn, Qn, dtop)]
            for split in ("val", "test"):
                S = L.load_split(ds, split)
                m = L.eval_ranking(S, _combine(experts, S))
                r["datasets"][ds]["results"].setdefault(ename, {})[split] = m
                c = m["COND_ANY_GOLD_IN_SCOPE"]
                log(f"  [{ds}/{split}] {ename} MRR={c['MRR']:.4f} R@5={c['R@5']:.4f} R@10={c['R@10']:.4f} ANY@10={c['ANY@10']:.4f}")
            save_res(r)
    save_res(r)


_GRAPH_CACHE = {}
def _load_C(ds):
    if ds in _GRAPH_CACHE:
        return _GRAPH_CACHE[ds]
    from ac_graph_diagnostic import load_graph_families
    fams, N = load_graph_families(ds)
    _GRAPH_CACHE[ds] = (fams["C"], fams["STRUCT"], fams["NER"], N)
    return _GRAPH_CACHE[ds]


def graph_seedprox_scores(S, qi, Cg, n_seed=5, max_hop=3):
    """Cheap graph feature: negative geodesic hop over the C graph (restricted to the P50 scope via a
    node mask) from the top-n_seed dense-ranked in-scope candidates. Closer-to-dense-anchor = higher.
    Uses FULL-N masked sparse matvecs (no per-query CSR submatrix slicing -> fast)."""
    s, e = L.scope(S, qi)
    cand = S["cand"][s:e]
    dsc = S["dense_score"][s:e]
    N = Cg.shape[0]
    cmask = np.zeros(N, bool); cmask[cand] = True
    seeds_local = np.argsort(-dsc, kind="stable")[:n_seed]
    seed_nodes = cand[seeds_local]
    hopN = np.full(N, max_hop + 1, dtype=np.float32)
    reached = np.zeros(N, bool)
    fv = np.zeros(N, dtype=np.float32); fv[seed_nodes] = 1.0
    reached[seed_nodes] = True; hopN[seed_nodes] = 0
    for h in range(1, max_hop + 1):
        nxt = (Cg.dot(fv) > 0) & cmask & (~reached)
        if not nxt.any():
            break
        hopN[nxt] = h; reached |= nxt; fv = nxt.astype(np.float32)
    return -hopN[cand]                                       # higher = closer to a dense anchor


def stage_e3b():
    r = load_res()
    for ds in DATASETS:
        Dn, Qn, dtop = L.load_embeddings(ds)
        Cg, STRUCT, NER, N = _load_C(ds)
        # rebuild trained heads
        from l2_lib import OffsetHead, MixtureHead
        oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_offset_cp50.pt")); oh.eval()
        mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_mixture_cp50.pt")); mh.eval()
        # best base config on VAL among E1a/E2/E3 (RRF family)
        res = r["datasets"][ds]["results"]
        cands = {k: res[k]["val"]["COND_ANY_GOLD_IN_SCOPE"]["MRR"] for k in
                 ("E1a_dense_splade_RRF", "E2_dense_splade_offset_RRF", "E3_dense_splade_offset_mixture_RRF") if k in res}
        best_name = max(cands, key=cands.get)
        base_experts = [_scorer_dense, _scorer_splade]
        if "offset" in best_name or "E2" in best_name or "E3" in best_name:
            base_experts.append(_mk_head_scorer(oh, "offset", Dn, Qn, dtop))
        if "mixture" in best_name or "E3" in best_name:
            base_experts.append(_mk_head_scorer(mh, "mixture", Dn, Qn, dtop))
        log(f"[{ds}] E3b base = {best_name} (VAL MRR {cands[best_name]:.4f}) + graph seed-proximity")
        for split in ("val", "test"):
            S = L.load_split(ds, split)
            gsc = _mk_graph_scorer(Cg, S)
            experts = base_experts + [gsc]
            m = L.eval_ranking(S, _combine(experts, S))
            r["datasets"][ds]["results"].setdefault("E3b_bestE0E3_plus_graph_RRF", {})[split] = m
            m["_base_config"] = best_name
            c = m["COND_ANY_GOLD_IN_SCOPE"]
            log(f"  [{ds}/{split}] E3b MRR={c['MRR']:.4f} R@5={c['R@5']:.4f} R@10={c['R@10']:.4f} ANY@10={c['ANY@10']:.4f}")
        save_res(r)


def _mk_graph_scorer(Cg, S):
    def f(SS, qi):
        return graph_seedprox_scores(SS, qi, Cg)
    f.is_head = True
    return f


def _mk_head_scorer(head, kind, Dn, Qn, dtop):
    def f(S, qi):
        return L.head_scores(S, qi, head, Dn, Qn, dtop, kind)
    f.is_head = True
    return f


def _combine(experts, S):
    def fn(qi):
        arrs = []
        for e in experts:
            if hasattr(e, "is_head"):
                arrs.append(e(S, qi))
            else:
                arrs.append(e(S)(qi))
        return L.rrf_fuse(arrs)
    return fn


def stage_regime(only_ds=None, only_split=None):
    """Collect (do not label) per-query best-gold-rank per expert on TRAIN+VAL for later regime work.
    Optional only_ds / only_split let a single split be run per foreground call (resume-safe)."""
    os.makedirs(REGIME_DIR, exist_ok=True)
    splits_want = (only_split,) if only_split else ("train", "val")
    for ds in DATASETS:
        if only_ds and ds != only_ds:
            continue
        # resume: skip if all wanted splits for this ds are already saved
        if all(os.path.exists(f"{REGIME_DIR}/{ds}_{sp}.npz") for sp in splits_want):
            log(f"[regime] {ds} already done, skipping"); continue
        Dn, Qn, dtop = L.load_embeddings(ds)
        from l2_lib import OffsetHead, MixtureHead
        oh = OffsetHead(1536); oh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_offset_cp50.pt")); oh.eval()
        mh = MixtureHead(1536, 8); mh.load_state_dict(torch.load(f"results/L2/_heads/{ds}_mixture_cp50.pt")); mh.eval()
        for split in splits_want:
            if os.path.exists(f"{REGIME_DIR}/{ds}_{split}.npz"):
                log(f"[regime] {ds}/{split} already saved, skipping"); continue
            S = L.load_split(ds, split)
            # batched forward once, then blocked doc-matmul → flat per-cand scores (O(1) slice/query)
            oh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, oh, Dn, Qn, dtop), Dn)
            mh_sc = L.precompute_head_scope_scores(S, L.precompute_head_preds(S, mh, Dn, Qn, dtop), Dn)
            _scorer_flat = lambda arr: (lambda qi: arr[S["off"][qi]:S["off"][qi + 1]])
            experts = {
                "dense": _scorer_dense(S), "splade": _scorer_splade(S),
                "E1a_rrf": _scorer_rrf(S, [_scorer_dense, _scorer_splade]),
                "offset": _scorer_flat(oh_sc),
                "mixture": _scorer_flat(mh_sc),
            }
            cols = {}
            for name, fn in experts.items():
                _, per_best, _ = L.eval_ranking(S, fn, collect_regime=True)
                cols[f"best_gold_rank_{name}"] = per_best
            # rescue: expert brings best gold into top-K that dense missed; disagreement dense vs splade
            d = cols["best_gold_rank_dense"]; sp = cols["best_gold_rank_splade"]
            valid = d >= 0
            resc = {}
            for name in ("splade", "offset", "mixture", "E1a_rrf"):
                b = cols[f"best_gold_rank_{name}"]
                for K in (10, 20, 50):
                    resc[f"rescue_{name}_into_top{K}"] = int(np.sum(valid & (d >= K) & (b < K)))
            disagree = int(np.sum(valid & (np.abs(d - sp) >= 20)))
            np.savez(f"{REGIME_DIR}/{ds}_{split}.npz", **cols)
            log(f"[regime] {ds}/{split} saved; dense-splade disagreement(|d-sp bestrank|>=20)={disagree} "
                f"rescue_splade@10={resc['rescue_splade_into_top10']} rescue_offset@10={resc['rescue_offset_into_top10']}")
    log("regime raw material saved to " + REGIME_DIR)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("e3b",): stage_e3b()
    if stage in ("regime",):
        stage_regime(only_ds=(sys.argv[2] if len(sys.argv) > 2 else None),
                     only_split=(sys.argv[3] if len(sys.argv) > 3 else None))
    if stage in ("labels", "all"): stage_labels()
    if stage in ("e0e1", "all"): stage_e0e1(splits=("val", "test"))
    if stage in ("smoke",): stage_smoke()
    if stage in ("train", "all"): stage_train()
    if stage in ("e3b", "all"): stage_e3b()
    if stage in ("regime", "all"): stage_regime()
    log(f"DONE_STAGE_{stage}")
