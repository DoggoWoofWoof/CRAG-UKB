"""G1 transfer evaluation for ONE target dataset (runs locally, CPU).

Backbone is the FROZEN locked text-pilot: source C7b soft-archetype fusion + C8c XGBRanker reranker,
fit on 2wiki_clean+musique_clean TRAIN only, dataset-agnostic, applied UNCHANGED to the target. Two
orthogonal transfer questions, both on the target's own VAL (TEST never touched):

  A. PARAMETER generalization  = ZERO_SHOT_C11A : exact source-trained C11a weights + ORIGINAL source-train
     E-standardization (emean/estd from C11_models.joblib) + original top20; NO target fitting.
  B. ARCHITECTURE generalization = REFIT_C11A   : same arch/hparams/loss/opt, trained on TARGET TRAIN only
     (target E-standardization computed on target train), evaluated on target VAL.

Four systems per target on VAL: C7b_fusion (frozen baseline) | C8c (source reranker, zero-shot) |
ZERO_SHOT_C11A | REFIT_C11A. Deep-recall invariant (ALL@50 identical across all four) is asserted.
Plus a zero-shot feature-compatibility audit (E-drift under source standardization; qv/dv norms; C8c
score dist; per-expert gold reachability) target vs source VAL.

Prereq: target substrate at data/l2_corpus/<TGT>/{train,val}/ (all 5 experts + expert_meta.npz).
Env: G1_TGT (default squad_clean).
"""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch, joblib
import l2_c8 as C8, l2_c9 as C9, l2_c11 as M11
import l2_c6_predict as C6P
import _run_c11 as R
from _run_c9 import combined, starts_of, fit_c8c, multigold, paired_boot
from _run_c11_prep import EIDX, EFEAT
OUT = C8.OUT; SRC = list(C8.DS); TGT = os.environ.get("G1_TGT", "squad_clean"); CAP = 20
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
torch.manual_seed(0); np.random.seed(0)
MET = ["ndcg5", "recall5_macro", "any5", "all5", "all5_feas", "mrr", "ndcg50", "all10", "all50"]


def slim(p): return {k: (round(p[k], 4) if isinstance(p.get(k), float) else p.get(k)) for k in MET + ["n"]}


def build_sp(ds, split, B, s):
    """Top-20 window bundle for C11a (qv/dv/E/c8c/y/g/st), identical construction to the VAL milestone."""
    voff = np.load(f"data/l2_corpus/{ds}/{split}/query_offsets.npy")
    cand = np.load(f"data/l2_corpus/{ds}/{split}/cand_ids.npy", mmap_mode="r")
    nodes = np.load(f"data/ukb_storage/{ds}/gte_qwen/nodes.npy", mmap_mode="r")
    qall = np.load(f"data/ukb_storage/{ds}/gte_qwen/queries_all.npy", mmap_mode="r")
    qm = json.load(open(f"data/l2_corpus/{ds}/{split}/query_meta.json"))
    X = C9.assemble(B, C9.resid_feats(s, B["groups"]))
    st = starts_of(B["groups"]); QV = []; DV = []; E = []; C8b = []; Y = []; G = []
    for i, (m, g) in enumerate(zip(B["meta"], B["groups"])):
        a = st[i]; sc = s[a:a + g]; pool = m["pool"]
        win = np.argsort(-sc, kind="stable")[:min(CAP, g)]; pool_win = pool[win]
        QV.append(np.asarray(qall[int(qm[m["qi"]]["row_all"])], np.float32))
        gids = np.asarray(cand[int(voff[m["qi"]]) + pool_win], np.int64); DV.append(np.asarray(nodes[gids], np.float32))
        E.append(X[a + win][:, EIDX]); C8b.append(sc[win])
        gs = set(m["gold"].tolist()); Y.append(np.array([1 if int(pl) in gs else 0 for pl in pool_win], np.int8)); G.append(len(win))
    G = np.asarray(G, np.int64)
    return {"qv": np.asarray(QV, np.float32), "dv": np.concatenate(DV).astype(np.float32),
            "E": np.concatenate(E).astype(np.float32), "c8c": np.concatenate(C8b).astype(np.float32),
            "y": np.concatenate(Y).astype(np.int64), "g": G, "st": starts_of(G)}


def prep_target(ds):
    """Build the C6 routing features + C8 archetype caches for a target split-pair (idempotent)."""
    for split in ("train", "val"):
        if not os.path.exists(f"{OUT}/_c6_feat_{ds}_{split}.npz"):
            log(f"prep: build_features {ds}/{split}"); C6P.build_features(ds, split, log=log)
        C8.precompute_arch(ds, split, log=log)


def _norms(v): n = np.linalg.norm(v, axis=1); return {"mean": round(float(n.mean()), 4), "std": round(float(n.std()), 4)}


def src_ref_from_bundle(sp_src):
    """Compress a source VAL bundle to the small reference dict used by feature_compat (so the source side can be
    fit LOCALLY once and shipped as tiny JSON, while the target side runs where the target substrate lives)."""
    return {"qv_norm": _norms(sp_src["qv"]), "dv_norm": _norms(sp_src["dv"]),
            "c8c_window": {"mean": round(float(sp_src["c8c"].mean()), 4), "std": round(float(sp_src["c8c"].std()), 4)}}


def feature_compat(spt_val, src_ref, em11, es11):
    """Zero-shot feature-compatibility: target VAL feature distributions vs source VAL reference, and E-drift under
    the SOURCE-train standardization (the exact stats ZERO_SHOT_C11A applies). Large |z| = representation shift."""
    Et = spt_val["E"]
    et_mean = Et.mean(0); z = (et_mean - em11) / es11                         # per-feature drift in source-std units
    order = np.argsort(-np.abs(z))
    top_drift = [{"feat": EFEAT[i], "target_mean": round(float(et_mean[i]), 4),
                  "source_train_mean": round(float(em11[i]), 4), "z_vs_source_std": round(float(z[i]), 3)} for i in order[:8]]
    return {"E_DRIFT_vs_source_train_std": {"max_abs_z": round(float(np.abs(z).max()), 3),
                                            "mean_abs_z": round(float(np.abs(z).mean()), 3), "top8": top_drift},
            "qv_norm": {"target": _norms(spt_val["qv"]), "source": src_ref["qv_norm"]},
            "dv_norm": {"target": _norms(spt_val["dv"]), "source": src_ref["dv_norm"]},
            "c8c_window": {"target": {"mean": round(float(spt_val["c8c"].mean()), 4), "std": round(float(spt_val["c8c"].std()), 4)},
                           "source": src_ref["c8c_window"]},
            "E_feat_names": EFEAT}


def main():
    log(f"=== G1 transfer eval TGT={TGT} (source backbone = {SRC}) ===")
    # ---- frozen source backbone (locked, dataset-agnostic) ----
    prep_target(TGT)
    BACKBONE_DIR = os.environ.get("G1_BACKBONE_DIR")
    if BACKBONE_DIR:                                                           # load pre-fit backbone (run where the
        base_full = joblib.load(f"{BACKBONE_DIR}/base_full.joblib")            # target substrate lives; no dev substrate)
        c8c = joblib.load(f"{BACKBONE_DIR}/c8c.joblib")
        sd11 = joblib.load(f"{BACKBONE_DIR}/C11_models.joblib")
        src_ref = json.load(open(f"{BACKBONE_DIR}/source_val_stats.json"))
        log(f"source backbone LOADED from {BACKBONE_DIR}")
    else:                                                                     # fit backbone locally from dev substrate
        full = {ds: C8.precompute_arch(ds, "train")["qi"] for ds in SRC}
        devinner = {ds: C8.load_split(ds)[1] for ds in SRC}
        base_full = C8.fit_soft_head("ndcg50", full)
        B_tr = combined("train", full, base_full); B_di = combined("train", devinner, base_full)
        c8c = fit_c8c(B_tr["X28"], B_tr["y"], B_tr["groups"], B_di["X28"], B_di["y"], B_di["groups"])
        log(f"source backbone frozen (C7b full + C8c best_iter={c8c.best_iteration})")
        sd11 = joblib.load(f"{OUT}/C11_models.joblib")
        src_ref = None
    ma = M11.C11a(); ma.load_state_dict(sd11["C11a"]); em11, es11 = sd11["emean"], sd11["estd"]

    # ---- target VAL + TRAIN bundles under the source backbone ----
    vqi = C8.precompute_arch(TGT, "val")["qi"]; Bv = C9.build_bundle(TGT, "val", vqi, base_full)
    for m in Bv["meta"]: m["ds"] = TGT
    sv = c8c.predict(Bv["X28"]).astype(np.float64)
    spv = build_sp(TGT, "val", Bv, sv)
    tqi = C8.precompute_arch(TGT, "train")["qi"]; Bt = C9.build_bundle(TGT, "train", tqi, base_full)
    st_full = c8c.predict(Bt["X28"]).astype(np.float64); spt = build_sp(TGT, "train", Bt, st_full)
    log(f"target bundles: val_q={len(Bv['meta'])} train_q={len(Bt['meta'])}")

    # ---- 4 systems on target VAL ----
    r_fuse = C9.eval_ranking(Bv, Bv["X28"][:, 1].astype(np.float64))             # C7b fusion (frozen baseline)
    r_c8c = C9.eval_ranking(Bv, sv)                                              # C8c source reranker, zero-shot
    spv11 = dict(spv); spv11["_t"] = R.prep_tensors(spv, em11, es11)
    r_zs = R.evaluate(ma, spv11, Bv, sv)                                         # ZERO_SHOT_C11A
    # REFIT_C11A: fresh C11a on target train (target E-standardization), locked hparams
    em_t = spt["E"].mean(0); es_t = spt["E"].std(0) + 1e-6
    spt["_t"] = R.prep_tensors(spt, em_t, es_t)
    mr = M11.C11a(); log("REFIT_C11A: training on target TRAIN (locked hparams: 18ep bq512 lr1.3e-3)")
    R.train(mr, spt["_t"], epochs=18, bq=512, lr=1.3e-3)
    spv_rf = dict(spv); spv_rf["_t"] = R.prep_tensors(spv, em_t, es_t)
    r_rf = R.evaluate(mr, spv_rf, Bv, sv)                                        # REFIT_C11A
    sysres = {"C7b_fusion": r_fuse, "C8c": r_c8c, "ZERO_SHOT_C11A": r_zs, "REFIT_C11A": r_rf}
    for nm, r in sysres.items():
        log(f"  {nm}: ndcg5={r['ndcg5']:.4f} recall5={r['recall5_macro']:.4f} mrr={r['mrr']:.4f} all50={r['all50']:.4f}")

    # ---- deep-recall invariant (pool identical -> ALL@50 identical across all four) ----
    a50 = {nm: round(r["all50"], 6) for nm, r in sysres.items()}
    deep_ok = len(set(a50.values())) == 1
    log(f"DEEP_RECALL_INVARIANT all50={a50} preserved={deep_ok}")

    # ---- paired bootstraps ----
    boot = {}
    for aa, bb in (("ZERO_SHOT_C11A", "C8c"), ("REFIT_C11A", "C8c"), ("REFIT_C11A", "ZERO_SHOT_C11A"),
                   ("C8c", "C7b_fusion")):
        boot[f"{aa}_vs_{bb}"] = {m: paired_boot(sysres[aa]["_perq"][mk], sysres[bb]["_perq"][mk]) for m, mk in
                                 (("ndcg5", "ndcg5"), ("recall5_macro", "recall5"), ("all5", "all5"), ("mrr", "mrr"))}

    # ---- feature-compat (source VAL reference distributions) ----
    if src_ref is None:                                                       # fit mode: build source VAL bundle now
        src_ds = SRC[0]
        svqi = C8.precompute_arch(src_ds, "val")["qi"]; Bsv = C9.build_bundle(src_ds, "val", svqi, base_full)
        ssv = c8c.predict(Bsv["X28"]).astype(np.float64); sp_src = build_sp(src_ds, "val", Bsv, ssv)
        src_ref = src_ref_from_bundle(sp_src)
    fcompat = feature_compat(spv, src_ref, em11, es11)

    # ---- multigold ----
    mg = {nm: multigold(r) for nm, r in sysres.items()}

    def sig(cmp, mk): b = boot[cmp][mk]; return bool(b["significant"] and b["delta"] > 0)
    gates = {
        "DEEP_RECALL_PRESERVED": "YES" if deep_ok else "NO",
        "C8C_BEATS_FUSION": "YES" if sig("C8c_vs_C7b_fusion", "ndcg5") else "NO",
        "ZERO_SHOT_C11A_BEATS_C8C": "YES" if sig("ZERO_SHOT_C11A_vs_C8c", "ndcg5") else "NO",
        "ZERO_SHOT_C11A_HURTS_C8C": "YES" if (boot["ZERO_SHOT_C11A_vs_C8c"]["ndcg5"]["significant"] and boot["ZERO_SHOT_C11A_vs_C8c"]["ndcg5"]["delta"] < 0) else "NO",
        "REFIT_C11A_BEATS_C8C": "YES" if sig("REFIT_C11A_vs_C8c", "ndcg5") else "NO",
        "REFIT_BEATS_ZERO_SHOT": "YES" if sig("REFIT_C11A_vs_ZERO_SHOT_C11A", "ndcg5") else "NO",
    }
    # interpretation category (text targets)
    if gates["ZERO_SHOT_C11A_BEATS_C8C"] == "YES":
        interp = "PARAMETERS_GENERALIZE"
    elif gates["REFIT_C11A_BEATS_C8C"] == "YES":
        interp = "ARCHITECTURE_GENERALIZES"
    elif r_c8c["ndcg5"] < r_fuse["ndcg5"] + 1e-6 and fcompat["E_DRIFT_vs_source_train_std"]["max_abs_z"] > 3.0:
        interp = "REPRESENTATION_SHIFT"
    else:
        interp = "NEUTRAL_OR_UPSTREAM_LIMIT"
    gates["INTERPRETATION"] = interp

    # ---- FAILURE DECOMPOSITION (per system): where in-scope golds land + P50 scope coverage ----
    def _decompose(res):
        gr = res["_goldranks"]                     # {(qi, gold_local): final_rank} over the FULL scope
        b5 = b520 = b2050 = b50 = 0
        for rk in gr.values():
            if rk < 5: b5 += 1
            elif rk < 20: b520 += 1
            elif rk < 50: b2050 += 1
            else: b50 += 1
        tot = max(len(gr), 1)
        return {"in_scope_golds": len(gr), "top5": b5, "top5_20": b520, "top20_50": b2050, "below50": b50,
                # user buckets: present-below-top20 = (top20_50 + below50); top20-below-top5 = top5_20
                "FAIL_below_top20": b2050 + b50, "FAIL_top20_below_top5": b520,
                "COND_ON_P50_gold_recall5_micro": round(b5 / tot, 4),
                "COND_ON_P50_gold_recall20_micro": round((b5 + b520) / tot, 4)}
    decomp = {nm: _decompose(r) for nm, r in sysres.items()}

    # ---- PER-HOP breakdown (metaqa 1/2/3-hop; empty for non-hop datasets) ----
    def _per_hop():
        try:
            qm = json.load(open(f"data/l2_corpus/{TGT}/val/query_meta.json"))
        except Exception as e:
            return {"error": repr(e)}
        hop_of = {i: qm[i].get("hop") for i in range(len(qm))}
        hops = sorted({h for h in hop_of.values() if h is not None})
        if not hops:
            return {}                                            # non-hop dataset -> no per-hop block
        MK = [("ndcg5", "ndcg5"), ("ndcg50", "ndcg50"), ("recall5_macro", "recall5"),
              ("any5", "any5"), ("all5", "all5"), ("all10", "all10"), ("all50", "all50"), ("mrr", "mrr")]
        blk = {}
        for h in hops:
            qset = set(i for i in range(len(qm)) if hop_of[i] == h)
            n_all = len(qset)
            n_feas = sum(1 for i in qset if qm[i].get("ANY_GOLD_PRESENT"))
            n_allg = sum(1 for i in qset if qm[i].get("ALL_GOLD_PRESENT"))
            incorp = sum(int(qm[i].get("N_GOLD_EXPECTED_INCORP", 0)) for i in qset)
            inscope = sum(int(qm[i].get("N_GOLD_IN_SCOPE", 0)) for i in qset)
            hp = {"hop": h, "n_val_queries": n_all,
                  "P50_ANY_coverage": round(n_feas / max(n_all, 1), 4),         # >=1 gold reached the ~5k P50 scope
                  "P50_ALL_coverage": round(n_allg / max(n_all, 1), 4),          # all golds reached scope
                  "golds_expected_in_corpus": incorp, "golds_in_P50_scope": inscope,
                  "L1_MISSING_golds": incorp - inscope, "systems": {}}
            for nm, res in sysres.items():
                pq = res["_perq"]; qiarr = pq["qi"]
                mask = np.array([int(q) in qset for q in qiarr])
                uncond, cond = {}, {}
                for label, key in MK:
                    vals = pq[key][mask].astype(np.float64)
                    cond[label] = round(float(vals.mean()), 4) if vals.size else None   # over feasible hop queries
                    uncond[label] = round(float(vals.sum()) / max(n_all, 1), 4)         # over ALL hop queries (L1-fails=0)
                b5 = b520 = b2050 = b50 = 0
                for (q, gl), rk in res["_goldranks"].items():
                    if q in qset:
                        if rk < 5: b5 += 1
                        elif rk < 20: b520 += 1
                        elif rk < 50: b2050 += 1
                        else: b50 += 1
                hp["systems"][nm] = {
                    "UNCONDITIONAL": uncond, "CONDITIONAL_ON_P50": cond,
                    "FAILURE_DECOMP": {"L1_MISSING": incorp - inscope,
                                       "in_scope_below_top20": b2050 + b50,
                                       "in_scope_top20_below_top5": b520,
                                       "in_scope_top5": b5}}
            blk[f"{h}"] = hp
        return blk
    per_hop = _per_hop()

    scope_cov = {}
    try:
        qm = json.load(open(f"data/l2_corpus/{TGT}/val/query_meta.json"))
        raw = int(sum(x.get("N_GOLD_EXPECTED_RAW", x.get("N_GOLD_EXPECTED_INCORP", 0)) for x in qm))
        mapped = int(sum(x.get("N_GOLD_EXPECTED_INCORP", 0) for x in qm))
        inscope = int(sum(len(m["gold"]) for m in Bv["meta"]))
        scope_cov = {"raw_expected_golds": raw, "mapped_in_corpus": mapped, "in_P50_scope": inscope,
                     "P50_ABSENT_of_mapped": mapped - inscope,
                     "ENTITY_UNMAPPABLE_raw_minus_mapped": raw - mapped,
                     "note": "per-gold; P50_ABSENT = mapped-in-corpus gold not in the ~5k P50 scope (routing/scope failure); ENTITY_UNMAPPABLE = raw gold id not resolvable to a corpus doc (alias/linking failure)"}
    except Exception as e:
        scope_cov = {"error": repr(e)}

    out = {"phase": f"G1 transfer eval — target={TGT}; source backbone={SRC}; VAL only; TEST untouched",
           "target": TGT, "source": SRC, "n_val": len(Bv["meta"]), "n_train_refit": len(Bt["meta"]),
           "MAIN_TABLE": {nm: slim(r) for nm, r in sysres.items()},
           "FAILURE_DECOMPOSITION": decomp, "P50_SCOPE_COVERAGE": scope_cov, "PER_HOP": per_hop,
           "DELTAS": {"ZERO_SHOT_minus_C8c": {k: round(r_zs[k] - r_c8c[k], 4) for k in MET if isinstance(r_zs.get(k), float)},
                      "REFIT_minus_C8c": {k: round(r_rf[k] - r_c8c[k], 4) for k in MET if isinstance(r_rf.get(k), float)},
                      "C8c_minus_fusion": {k: round(r_c8c[k] - r_fuse[k], 4) for k in MET if isinstance(r_c8c.get(k), float)}},
           "DEEP_RECALL": {"all50_by_system": a50, "preserved": deep_ok},
           "PAIRED_BOOTSTRAPS": boot, "MULTIGOLD": mg, "FEATURE_COMPAT": fcompat,
           "GATES": gates, "INTERPRETATION": interp,
           "NEW_ENCODER_FORWARD_PASSES": 0, "NEW_LLM_COMPONENTS": 0, "TARGET_TEST_TOUCHED": "NO"}
    os.makedirs("results/GENERALIZATION", exist_ok=True)
    json.dump(out, open(f"results/GENERALIZATION/_g1_eval_{TGT}.json", "w"), indent=1, default=str)
    log(f"G1_EVAL_DONE {TGT} interpretation={interp}")
    log("GATES " + json.dumps(gates))
    log("MAIN " + json.dumps({nm: {"ndcg5": round(r["ndcg5"], 4), "recall5": round(r["recall5_macro"], 4)} for nm, r in sysres.items()}))


if __name__ == "__main__":
    main()
