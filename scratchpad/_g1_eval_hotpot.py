"""G1 TEXT target-2 (HotpotQA) transfer evaluation. Same FROZEN backbone + IDENTICAL 4-system scoring as
_g1_eval_target (C7b_FUSION | C8c_ZERO_SHOT | ZERO_SHOT_C11A | REFIT_25K_C11A), but with the richer Hotpot
reporting the methodology requires:

  - L1_P50_CEILING (TRAIN+VAL): ANY / ALL / GOLD_RECALL_IN_SCOPE  (kept SEPARATE from L2 ranking quality)
  - Full metric suite: NDCG@5, GOLD_RECALL@5, ANY@5, ALL@5, ALL@5_FEASIBLE, MRR, NDCG@50, GOLD_RECALL@50,
    ANY@50, ALL@50   (core numbers come straight from the frozen C9.eval_ranking, unchanged)
  - CONDITIONAL_GOLD_RECALL@5 / CONDITIONAL_ALL@5 over the subset where ALL evidence survived P50
  - decomposition chain deltas: L1 ceiling -> C7b -> C8c -> ZERO_SHOT_C11A -> REFIT_25K_C11A
  - multi-gold breakdown by TOTAL in-corpus golds (1 / 2 / 3+): NDCG@5 / GOLD_RECALL@5 / ALL@5
  - relation-expert diagnostics (measured, not assumed)
  - zero-shot feature-compat (source stats) + expert-contribution distributions + relation frequency
  - paired query-level bootstraps (NDCG@5, GOLD_RECALL@5, ALL@5, MRR)

NOTHING about the architecture/scoring changes; this is reporting only. Env: G1_TGT (default hotpotqa_clean),
G1_BACKBONE_DIR (load frozen backbone). TEST never touched.
"""
import sys, os, json, time
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np, torch, joblib
import l2_c8 as C8, l2_c9 as C9, l2_c11 as M11
import _run_c11 as R
from _run_c9 import combined, starts_of, fit_c8c, paired_boot
from l2_c8 import ranks_from_score
import _g1_eval_target as EV   # reuse build_sp, prep_target, feature_compat, src_ref_from_bundle, EIDX/EFEAT

OUT = C8.OUT; SRC = list(C8.DS); TGT = os.environ.get("G1_TGT", "hotpotqa_clean"); CAP = 20
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
torch.manual_seed(0); np.random.seed(0)


# --------------------------------------------------------------- L1 P50 coverage ceiling (from query_meta)
def l1_ceiling(ds, split):
    qm = json.load(open(f"data/l2_corpus/{ds}/{split}/query_meta.json"))
    anyp = []; allp = []; gr = []  # gold_recall_in_scope = N_GOLD_IN_SCOPE / N_GOLD_EXPECTED_INCORP
    for m in qm:
        nt = int(m["N_GOLD_EXPECTED_INCORP"])
        if nt <= 0:
            continue
        anyp.append(1.0 if m["ANY_GOLD_PRESENT"] else 0.0)
        allp.append(1.0 if m["ALL_GOLD_PRESENT"] else 0.0)
        gr.append(int(m["N_GOLD_IN_SCOPE"]) / nt)
    return {"n_valid": len(anyp), "ANY": round(float(np.mean(anyp)), 4), "ALL": round(float(np.mean(allp)), 4),
            "GOLD_RECALL_IN_SCOPE": round(float(np.mean(gr)), 4)}


# --------------------------------------------------------------- full-pool C11 scores (replicate R.evaluate)
def c11_full_scores(model, sp, Bv, s_full):
    """Return the exact per-candidate cand-score array R.evaluate feeds to eval_ranking (top20 window = 1e6+C11
    final ordered; tail keeps C8c). Reporting-only replication of _run_c11.evaluate lines 74-83."""
    sp_t = R.rows_qv(sp, sp["_t"]); delta = R.c11_scores(model, sp_t); beta = float(model.beta)
    finalw = sp["_t"]["c8cz"].numpy() + beta * delta
    st = sp["st"]; groupsB = Bv["groups"]; stB = starts_of(groupsB); cand = np.empty(len(Bv["y"]), np.float64)
    for i, gB in enumerate(groupsB):
        aB = stB[i]; scq = s_full[aB:aB + gB]; order = np.argsort(-scq, kind="stable"); win = order[:min(CAP, gB)]
        cs = scq.astype(np.float64).copy(); a = st[i]; cs[win] = 1e6 + finalw[a:a + len(win)]
        cand[aB:aB + gB] = cs
    return cand


# --------------------------------------------------------------- rich per-query metrics from a full-pool score
def rich_metrics(Bv, cand, qmeta):
    """Per-query gold ranks under `cand`, joined with query_meta totals. Returns aggregates over ALL valid
    queries and over the ALL_GOLD_PRESENT (evidence-survived-P50) subset, plus per-query arrays for bootstrap
    and multi-gold groups. In-scope gold ranks use the frozen ranks_from_score; denominators noted per metric."""
    groups = Bv["groups"]; st = starts_of(groups); y = Bv["y"]; meta = Bv["meta"]
    def ndcg(gr, ng, k):
        dcg = np.sum([1.0 / np.log2(x + 2) for x in gr if x < k]); idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(ng, k))])
        return (dcg / idcg) if idcg > 0 else 0.0
    rows = {k: [] for k in ("ndcg5", "ndcg50", "mrr", "any5", "any50", "rec5_in", "rec50_in", "all5_in", "all50_in",
                            "rec5_tot", "ng_scope", "ng_total", "all_present", "hop")}
    for i, gB in enumerate(groups):
        aB = st[i]; sc = cand[aB:aB + gB]; yy = y[aB:aB + gB]
        rk = ranks_from_score(sc); gr = rk[yy == 1]; ng_scope = int((yy == 1).sum())
        qi = meta[i]["qi"]; qmi = qmeta[qi]; ng_total = int(qmi["N_GOLD_EXPECTED_INCORP"])
        if ng_total <= 0:
            continue
        allp = bool(qmi["ALL_GOLD_PRESENT"]); hop = qmi.get("hop")
        if ng_scope > 0:
            best = int(gr.min()); g5 = int((gr < 5).sum()); g50 = int((gr < 50).sum()); worst = int(gr.max())
        else:
            best = 10**9; g5 = g50 = 0; worst = 10**9
        rows["ndcg5"].append(ndcg(gr, ng_scope, 5)); rows["ndcg50"].append(ndcg(gr, ng_scope, 50))
        rows["mrr"].append(1.0 / (best + 1) if ng_scope else 0.0)
        rows["any5"].append(1.0 if best < 5 else 0.0); rows["any50"].append(1.0 if best < 50 else 0.0)
        rows["rec5_in"].append(g5 / ng_scope if ng_scope else 0.0); rows["rec50_in"].append(g50 / ng_scope if ng_scope else 0.0)
        rows["all5_in"].append(1.0 if (ng_scope and worst < 5) else 0.0); rows["all50_in"].append(1.0 if (ng_scope and worst < 50) else 0.0)
        rows["rec5_tot"].append(g5 / ng_total)
        rows["ng_scope"].append(ng_scope); rows["ng_total"].append(ng_total)
        rows["all_present"].append(allp); rows["hop"].append(hop)
    A = {k: np.asarray(v) for k, v in rows.items()}
    ap = A["all_present"].astype(bool); n = len(ap)
    feas = A["ng_total"] <= 5
    def m(x, mask=None): x = A[x] if isinstance(x, str) else x; return round(float(x[mask].mean()) if mask is not None else float(x.mean()), 4)
    agg = {
        "n": n,
        "NDCG@5": m("ndcg5"), "NDCG@50": m("ndcg50"), "MRR": m("mrr"),
        "ANY@5": m("any5"), "ANY@50": m("any50"),
        "GOLD_RECALL@5": m("rec5_in"), "GOLD_RECALL@50": m("rec50_in"),      # in-scope denominator (matches frozen)
        "GOLD_RECALL@5_vs_total": m("rec5_tot"),                              # unconditional (vs total in-corpus golds)
        "ALL@5": m("all5_in"), "ALL@50": m("all50_in"),
        "ALL@5_FEASIBLE": round(float(A["all5_in"][feas].mean()), 4) if feas.any() else None,
        "CONDITIONAL_GOLD_RECALL@5": round(float(A["rec5_in"][ap].mean()), 4) if ap.any() else None,
        "CONDITIONAL_ALL@5": round(float(A["all5_in"][ap].mean()), 4) if ap.any() else None,
        "n_all_gold_present": int(ap.sum()),
    }
    # multi-gold by TOTAL in-corpus golds
    mg = {}
    for label, sel in (("1", A["ng_total"] == 1), ("2", A["ng_total"] == 2), ("3+", A["ng_total"] >= 3)):
        if sel.any():
            mg[label] = {"n": int(sel.sum()), "NDCG@5": round(float(A["ndcg5"][sel].mean()), 4),
                         "GOLD_RECALL@5": round(float(A["rec5_in"][sel].mean()), 4),
                         "ALL@5": round(float(A["all5_in"][sel].mean()), 4)}
        else:
            mg[label] = None
    agg["MULTIGOLD_by_total"] = mg
    return agg, A


# --------------------------------------------------------------- relation-expert diagnostics (measured)
def relation_diag(ds, split):
    d = f"data/l2_corpus/{ds}/{split}"
    off = np.load(f"{d}/query_offsets.npy"); lab = np.load(f"{d}/labels.npy"); mask = np.load(f"{d}/relation_mask.npy")
    rscore = np.load(f"{d}/relation_qwen_score.npy")
    Z = np.load(f"{d}/expert_meta.npz")
    rel_any = Z["relation_any_signal_present"].astype(bool); rel_goldp = Z["relation_gold_present"].astype(bool)
    rel_nc = Z["relation_num_candidates"].astype(np.int64)
    nq = len(off) - 1
    gold_elig = 0; gold_scope = 0; gold_ranks = []       # rank of relation-eligible golds by relation score within scope
    for qi in range(nq):
        s, e = int(off[qi]), int(off[qi + 1]); m = mask[s:e]; g = lab[s:e] == 1
        gold_scope += int(g.sum()); gold_elig += int((m & g).sum())
        if m.any():
            rs = rscore[s:e].astype(np.float64).copy(); rs[~m] = -1e30    # rank only relation-eligible cands
            rk = ranks_from_score(rs)
            for idx in np.where(m & g)[0]:
                gold_ranks.append(int(rk[idx]))
    gr = np.asarray(gold_ranks)
    return {
        "n_queries": int(nq),
        "frac_queries_with_relation_signal": round(float(rel_any.mean()), 5),
        "mean_eligible_relation_candidates_per_query": round(float(rel_nc.mean()), 3),
        "mean_eligible_relation_candidates_per_query_when_signal": round(float(rel_nc[rel_any].mean()), 3) if rel_any.any() else 0.0,
        "frac_queries_relation_touches_gold": round(float(rel_goldp.mean()), 5),
        "frac_gold_candidates_relation_eligible": round(gold_elig / max(gold_scope, 1), 5),
        "n_relation_eligible_golds": int(gr.size),
        "relation_eligible_gold_rank_distribution": None if gr.size == 0 else {
            "median": float(np.median(gr)), "mean": round(float(gr.mean()), 2),
            "p25": float(np.percentile(gr, 25)), "p75": float(np.percentile(gr, 75)),
            "frac_rank0": round(float((gr == 0).mean()), 4), "frac_top5": round(float((gr < 5).mean()), 4),
            "frac_top50": round(float((gr < 50).mean()), 4)},
    }


def main():
    log(f"=== G1 HOTPOT transfer eval TGT={TGT} (source backbone={SRC}) ===")
    EV.prep_target(TGT)
    BACKBONE_DIR = os.environ.get("G1_BACKBONE_DIR")
    assert BACKBONE_DIR, "set G1_BACKBONE_DIR (frozen backbone bundle)"
    base_full = joblib.load(f"{BACKBONE_DIR}/base_full.joblib")
    c8c = joblib.load(f"{BACKBONE_DIR}/c8c.joblib")
    sd11 = joblib.load(f"{BACKBONE_DIR}/C11_models.joblib")
    src_ref = json.load(open(f"{BACKBONE_DIR}/source_val_stats.json"))
    ma = M11.C11a(); ma.load_state_dict(sd11["C11a"]); em11, es11 = sd11["emean"], sd11["estd"]
    log("frozen source backbone LOADED (C7b base_full + C8c + source C11a)")

    # ---- L1 P50 coverage ceiling (separate from ranking) ----
    L1 = {"train": l1_ceiling(TGT, "train"), "val": l1_ceiling(TGT, "val")}
    log(f"L1_P50_CEILING val={L1['val']}")

    # ---- target VAL + TRAIN bundles under the frozen backbone ----
    qmeta_val = json.load(open(f"data/l2_corpus/{TGT}/val/query_meta.json"))
    vqi = C8.precompute_arch(TGT, "val")["qi"]; Bv = C9.build_bundle(TGT, "val", vqi, base_full)
    for mm in Bv["meta"]: mm["ds"] = TGT
    sv = c8c.predict(Bv["X28"]).astype(np.float64)
    spv = EV.build_sp(TGT, "val", Bv, sv)
    tqi = C8.precompute_arch(TGT, "train")["qi"]; Bt = C9.build_bundle(TGT, "train", tqi, base_full)
    st_full = c8c.predict(Bt["X28"]).astype(np.float64); spt = EV.build_sp(TGT, "train", Bt, st_full)
    log(f"bundles: val_q={len(Bv['meta'])} train_q={len(Bt['meta'])}")

    # ---- 4 systems: full-pool cand scores (IDENTICAL scoring to _g1_eval_target) ----
    cand_fuse = Bv["X28"][:, 1].astype(np.float64)                                # C7b_FUSION
    cand_c8c = sv                                                                 # C8c_ZERO_SHOT
    spv_zs = dict(spv); spv_zs["_t"] = R.prep_tensors(spv, em11, es11)
    cand_zs = c11_full_scores(ma, spv_zs, Bv, sv)                                 # ZERO_SHOT_C11A
    # REFIT_25K_C11A: fresh C11a, target TRAIN only, target-train E-standardization, LOCKED hparams
    em_t = spt["E"].mean(0); es_t = spt["E"].std(0) + 1e-6
    spt["_t"] = R.prep_tensors(spt, em_t, es_t)
    mr = M11.C11a(); log("REFIT_25K_C11A: training on target TRAIN (locked 18ep bq512 lr1.3e-3)")
    R.train(mr, spt["_t"], epochs=18, bq=512, lr=1.3e-3)
    spv_rf = dict(spv); spv_rf["_t"] = R.prep_tensors(spv, em_t, es_t)
    cand_rf = c11_full_scores(mr, spv_rf, Bv, sv)                                 # REFIT_25K_C11A

    SYS = {"C7b_FUSION": cand_fuse, "C8c_ZERO_SHOT": cand_c8c, "ZERO_SHOT_C11A": cand_zs, "REFIT_25K_C11A": cand_rf}
    AGG = {}; PERQ = {}
    for nm, cand in SYS.items():
        AGG[nm], PERQ[nm] = rich_metrics(Bv, cand, qmeta_val)
        r = AGG[nm]
        log(f"  {nm}: NDCG@5={r['NDCG@5']:.4f} GR@5={r['GOLD_RECALL@5']:.4f} ALL@5={r['ALL@5']:.4f} "
            f"MRR={r['MRR']:.4f} NDCG@50={r['NDCG@50']:.4f} ALL@50={r['ALL@50']:.4f} condGR@5={r['CONDITIONAL_GOLD_RECALL@5']}")

    # ---- deep-recall invariant (ALL@50 identical across the 4 systems) ----
    a50 = {nm: AGG[nm]["ALL@50"] for nm in SYS}
    deep_ok = len(set(round(v, 6) for v in a50.values())) == 1
    log(f"DEEP_RECALL_INVARIANT all50={a50} preserved={deep_ok}")

    # ---- decomposition chain: L1 ceiling -> C7b -> C8c -> ZERO_SHOT -> REFIT_25K ----
    chain = ["C7b_FUSION", "C8c_ZERO_SHOT", "ZERO_SHOT_C11A", "REFIT_25K_C11A"]
    keymetrics = ["NDCG@5", "GOLD_RECALL@5", "ALL@5", "MRR", "NDCG@50", "GOLD_RECALL@50", "ANY@5", "ANY@50", "ALL@50"]
    decomposition = {"L1_P50_CEILING_val": L1["val"]}
    for a, b in zip(chain[:-1], chain[1:]):
        decomposition[f"{b}_minus_{a}"] = {k: round(AGG[b][k] - AGG[a][k], 4) for k in keymetrics}

    # ---- paired bootstraps (NDCG@5, GOLD_RECALL@5, ALL@5, MRR) ----
    pair_keys = {"ndcg5": "ndcg5", "recall5": "rec5_in", "all5": "all5_in", "mrr": "mrr"}
    boot = {}
    for aa, bb in (("C8c_ZERO_SHOT", "C7b_FUSION"), ("ZERO_SHOT_C11A", "C8c_ZERO_SHOT"),
                   ("REFIT_25K_C11A", "C8c_ZERO_SHOT"), ("REFIT_25K_C11A", "ZERO_SHOT_C11A")):
        boot[f"{aa}_vs_{bb}"] = {label: paired_boot(PERQ[aa][pk], PERQ[bb][pk]) for label, pk in pair_keys.items()}

    # ---- relation diagnostics (measured) ----
    rel = {"val": relation_diag(TGT, "val"), "train": relation_diag(TGT, "train")}
    log(f"RELATION val: signal={rel['val']['frac_queries_with_relation_signal']} "
        f"touches_gold={rel['val']['frac_queries_relation_touches_gold']} "
        f"gold_eligible={rel['val']['frac_gold_candidates_relation_eligible']}")

    # ---- feature-compat (source stats) + expert-contribution distributions + relation frequency ----
    fcompat = EV.feature_compat(spv, src_ref, em11, es11)
    EF = EV.EFEAT
    expert_contrib = {}
    for feat in ("c_dense", "c_splade", "c_offset", "c_mixture", "c_relation", "rel_mask", "max_contrib", "second_contrib"):
        if feat in EF:
            i = EF.index(feat)
            expert_contrib[feat] = {"target_mean": round(float(spv["E"][:, i].mean()), 4),
                                    "source_train_mean": round(float(em11[i]), 4),
                                    "z_vs_source_std": round(float((spv["E"][:, i].mean() - em11[i]) / es11[i]), 3)}
    fcompat["EXPERT_CONTRIB_and_RELATION_FREQ"] = expert_contrib

    # ---- gates + interpretation ----
    def sig(cmp, mk):
        b = boot[cmp][mk]; return bool(b["significant"] and b["delta"] > 0)
    def hurts(cmp, mk):
        b = boot[cmp][mk]; return bool(b["significant"] and b["delta"] < 0)
    gates = {
        "DEEP_RECALL_PRESERVED": "YES" if deep_ok else "NO",
        "C8C_BEATS_FUSION": "YES" if sig("C8c_ZERO_SHOT_vs_C7b_FUSION", "ndcg5") else "NO",
        "ZERO_SHOT_C11A_BEATS_C8C": "YES" if sig("ZERO_SHOT_C11A_vs_C8c_ZERO_SHOT", "ndcg5") else "NO",
        "ZERO_SHOT_C11A_HURTS_C8C": "YES" if hurts("ZERO_SHOT_C11A_vs_C8c_ZERO_SHOT", "ndcg5") else "NO",
        "REFIT_25K_C11A_BEATS_C8C": "YES" if sig("REFIT_25K_C11A_vs_C8c_ZERO_SHOT", "ndcg5") else "NO",
        "REFIT_BEATS_ZERO_SHOT": "YES" if sig("REFIT_25K_C11A_vs_ZERO_SHOT_C11A", "ndcg5") else "NO",
    }
    # relation helpful ONLY if diagnostics support it
    rv = rel["val"]
    rel_helpful = (rv["frac_queries_relation_touches_gold"] >= 0.10 and rv["frac_gold_candidates_relation_eligible"] >= 0.10
                   and rv["relation_eligible_gold_rank_distribution"] is not None
                   and rv["relation_eligible_gold_rank_distribution"]["frac_top50"] >= 0.5)
    gates["RELATIONAL_STRUCTURE_HELPFUL"] = "YES" if rel_helpful else "NO"
    coverage_limited = (L1["val"]["ALL"] < 0.85)   # substantial evidence lost at P50
    if gates["ZERO_SHOT_C11A_BEATS_C8C"] == "YES":
        interp = "PARAMETERS_GENERALIZE"
    elif gates["REFIT_25K_C11A_BEATS_C8C"] == "YES":
        interp = "ARCHITECTURE_GENERALIZES"
    elif coverage_limited:
        interp = "UPSTREAM_COVERAGE_LIMIT"
    elif fcompat["E_DRIFT_vs_source_train_std"]["max_abs_z"] > 3.0:
        interp = "REPRESENTATION_SHIFT"
    else:
        interp = "NEUTRAL_OR_UPSTREAM_LIMIT"
    gates["INTERPRETATION"] = interp

    out = {
        "phase": f"G1 HOTPOT transfer eval — target={TGT}; source backbone={SRC}; VAL only; TEST untouched",
        "target": TGT, "source": SRC, "n_val": AGG["C7b_FUSION"]["n"], "n_train_refit": len(Bt["meta"]),
        "L1_P50_CEILING": L1,
        "MAIN_TABLE": {nm: {k: AGG[nm][k] for k in
                           ["NDCG@5", "GOLD_RECALL@5", "ANY@5", "ALL@5", "ALL@5_FEASIBLE", "MRR",
                            "NDCG@50", "GOLD_RECALL@50", "ANY@50", "ALL@50",
                            "GOLD_RECALL@5_vs_total", "CONDITIONAL_GOLD_RECALL@5", "CONDITIONAL_ALL@5",
                            "n", "n_all_gold_present"]} for nm in SYS},
        "DECOMPOSITION_CHAIN": decomposition,
        "DEEP_RECALL": {"all50_by_system": a50, "preserved": deep_ok},
        "MULTIGOLD_by_total": {nm: AGG[nm]["MULTIGOLD_by_total"] for nm in SYS},
        "RELATION_DIAGNOSTICS": rel,
        "FEATURE_COMPAT": fcompat,
        "PAIRED_BOOTSTRAPS": boot,
        "GATES": gates, "INTERPRETATION": interp,
        "NEW_ENCODER_FORWARD_PASSES": 0, "NEW_LLM_COMPONENTS": 0, "TARGET_TEST_TOUCHED": "NO",
    }
    os.makedirs("results/GENERALIZATION", exist_ok=True)
    json.dump(out, open(f"results/GENERALIZATION/_g1_eval_{TGT}.json", "w"), indent=1, default=str)
    log(f"G1_HOTPOT_EVAL_DONE {TGT} interpretation={interp}")
    log("GATES " + json.dumps(gates))
    log("MAIN " + json.dumps({nm: {k: AGG[nm][k] for k in ("NDCG@5", "GOLD_RECALL@5", "ALL@5", "MRR")} for nm in SYS}))


if __name__ == "__main__":
    main()
