"""C12 OFFICIAL-VAL MILESTONE — ONE evaluation of the three PREDECLARED policies C8c / C11a / C12a on official
VAL (2wiki_clean + musique_clean), per-dataset + pooled. Step 1 writes results/L2/L2_PREVAL_FREEZE_MANIFEST.json
BEFORE any VAL access. Backbone (C7b soft-archetype fusion + C8c XGBRanker residual reranker) refit on FULL TRAIN
(milestone convention; early-stop on DEV_INNER), frozen; C11a/C12a are the frozen dev-selected models applied
UNCHANGED (trained on the C9_TRAIN backbone; z-scored C8c input makes the residual scale-invariant to the
backbone refit). NO refit on VAL, NO retune, NO TEST. Reorders within top20 -> ALL@50 preserved by construction.
NEW_ENCODER_FORWARD=0, NEW_LLM_COMPONENT=0."""
import sys, os, json, time, hashlib
sys.path.insert(0, "scratchpad")
import numpy as np, torch, joblib
import l2_c8 as C8, l2_c9 as C9, l2_c11 as M11, l2_c12 as M12
import _run_c11 as R
from _run_c9 import combined, starts_of, fit_c8c, multigold, rescue, paired_boot
from _run_c11_prep import EIDX, EFEAT
from _run_c12 import group_all, evaluate as eval_c12, delta_full as c12_delta, c11a_finalw
OUT = C8.OUT; DS = C8.DS; CAP = 20; log = lambda *a: print(*a, flush=True); T0 = time.time()
torch.manual_seed(0); np.random.seed(0)
REF_C8C = {"ndcg5": 0.8824, "recall5_macro": 0.9030, "mrr": 0.9569, "ndcg50": 0.9114, "all50": 0.9406}
MET = ["ndcg5", "recall5_macro", "any5", "all5", "all5_feas", "mrr", "ndcg50", "all10", "all50"]


def sha(path, cap=None):
    if not os.path.exists(path): return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b: break
            h.update(b)
            if cap and f.tell() >= cap: break
    return h.hexdigest()[:32]


def slim(p):
    return {k: (round(p[k], 4) if isinstance(p.get(k), float) else p.get(k)) for k in MET}


def poolw(perds, key, wkey="n"):
    num = sum(perds[ds][key] * perds[ds][wkey] for ds in DS if isinstance(perds[ds].get(key), (int, float)))
    den = sum(perds[ds][wkey] for ds in DS)
    return num / max(den, 1)


def build_val_sp(ds, B, s):
    voff = np.load(f"data/l2_corpus/{ds}/val/query_offsets.npy")
    cand = np.load(f"data/l2_corpus/{ds}/val/cand_ids.npy", mmap_mode="r")
    nodes = np.load(f"data/ukb_storage/{ds}/gte_qwen/nodes.npy", mmap_mode="r")
    qall = np.load(f"data/ukb_storage/{ds}/gte_qwen/queries_all.npy", mmap_mode="r")
    qm = json.load(open(f"data/l2_corpus/{ds}/val/query_meta.json"))
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


def context_effect_val(mc, ma, sp12, sp11):
    """Task: delta_context = C12a finalw - C11a finalw for rescued gold / removed FP / promoted non-gold (C12a top5
    vs C11a top5). Verifies the set context (not just C11a) drives the gain. C11a path uses its own standardization."""
    dC = c12_delta(mc, sp12); fC = sp12["_t"]["c8cz"].numpy() + float(mc.beta) * dC
    fA = c11a_finalw(ma, sp11); sp = sp12; st = sp["st"]; g = sp["g"]; y = sp["y"]
    rg = []; rf = []; pf = []
    for i in range(len(g)):
        a, b = st[i], st[i + 1]; n = b - a
        tA = set(np.argsort(-fA[a:b], kind="stable")[:min(5, n)].tolist())
        tC = set(np.argsort(-fC[a:b], kind="stable")[:min(5, n)].tolist())
        yb = y[a:b]; dctx = fC[a:b] - fA[a:b]
        for j in tC - tA:
            (rg if yb[j] == 1 else pf).append(float(dctx[j]))
        for j in tA - tC:
            if yb[j] == 0: rf.append(float(dctx[j]))
    mm = lambda L: [round(float(np.mean(L)), 4), round(float(np.median(L)), 4)] if L else None
    return {"rescued_gold": {"n": len(rg), "mean_median_delta_context": mm(rg)},
            "removed_fp": {"n": len(rf), "mean_median_delta_context": mm(rf)},
            "promoted_nongold": {"n": len(pf), "mean_median_delta_context": mm(pf)}}


def gold_rescue_expert(sp, B, s, base_gr, new_gr):
    """Rescued-into-top5 golds by source rank bin + strongest original expert (from E contrib cols 3:8)."""
    from collections import Counter
    exps = ["dense", "splade", "offset", "mixture", "relation"]; st = starts_of(B["groups"])
    bins = {"5-9": 0, "10-19": 0, "20-49": 0}; hurt = 0; expc = Counter(); Esp = sp["E"]; sst = sp["st"]
    for i, (m, g) in enumerate(zip(B["meta"], B["groups"])):
        a = st[i]; sc = s[a:a + g]; win = np.argsort(-sc, kind="stable")[:min(CAP, g)]; pool_win = m["pool"][win]
        gs = set(m["gold"].tolist()); aS = sst[i]
        for t, pl in enumerate(pool_win):
            key = (m["qi"], int(pl))
            if int(pl) in gs and new_gr.get(key, 99) < 5 and base_gr.get(key, 99) >= 5:
                ra = base_gr[key]; b = "5-9" if ra < 10 else "10-19" if ra < 20 else "20-49"
                bins[b] = bins.get(b, 0) + 1; expc[exps[int(np.argmax(Esp[aS + t, 3:8]))]] += 1
    for key, ra in base_gr.items():
        if ra < 5 and new_gr.get(key, 99) >= 5: hurt += 1
    return {"rescued_by_source_bin": bins, "TOP5_RESCUE": sum(bins.values()), "TOP5_HURT": hurt,
            "NET_TOP5_GOLD_GAIN": sum(bins.values()) - hurt, "rescued_gold_strongest_expert": dict(expc)}


def multigold_pool(res_by_ds):
    out = {}
    for lab in ("1", "2", "3+"):
        num = {k: 0.0 for k in ("recall5", "all5", "ndcg5")}; n = 0
        for ds in DS:
            b = res_by_ds[ds].get(lab)
            if b:
                n += b["n"]
                for k in num: num[k] += b[k] * b["n"]
        out[lab] = {"n": n, **{k: round(num[k] / max(n, 1), 4) for k in num}} if n else None
    return out


def main():
    devinner = {ds: C8.load_split(ds)[1] for ds in DS}
    full = {ds: C8.precompute_arch(ds, "train")["qi"] for ds in DS}
    sd11 = joblib.load(f"{OUT}/C11_models.joblib"); sd12 = joblib.load(f"{OUT}/C12_models.joblib")

    # ---------- STEP 1: PRE-VAL FREEZE MANIFEST (written BEFORE any VAL access) ----------
    manifest = {
        "milestone": "C12 official-VAL — three predeclared policies C8c/C11a/C12a; ONE evaluation; TEST untouched",
        "PREDECLARED_POLICIES": ["C8c", "C11a", "C12a"], "FINAL_PRE_VAL_POLICY": "C12a",
        "C12_VARIANT": "MEAN_ONLY", "C12_LATENT_DIM": 128, "C12_MAX_CONTEXT": "NO",
        "DATASET_ID_FEATURE": "NO", "NEW_ENCODER_FORWARD": 0, "NEW_LLM_COMPONENT": 0,
        "OFFICIAL_VAL_ACCESSED_FOR_C11_C12_SELECTION": "NO",
        "backbone": {"fusion": "C7b soft-archetype (fit_soft_head ndcg50)", "reranker": "C8c XGBRanker rank:ndcg top-50 pool",
                     "fit_data": "FULL TRAIN (both datasets), early-stop on DEV_INNER", "note": "C11a/C12a were trained on the C9_TRAIN backbone; applied unchanged here (z-scored C8c input => scale-invariant residual)."},
        "top20_construction": "per query: C7b-fused pool -> C8c score s -> take top-20 by s; C11a/C12a reorder within top-20 only; rest keep C8c order -> ALL@50 preserved",
        "c11a": {"arch": "Pq,Pd 1536->256; interaction [hq,hd,hq*hd,|hq-hd|,E(18)]; 512->128->1 zero-init residual; beta",
                 "params": 1387778, "loss": "pairwise RankNet multi-positive top5-weighted", "seed": 0},
        "c12a": {"arch": "C11a-style local z(Z=128) + leave-one-out MEAN set context + hq + expert-set(15) -> ctx MLP 256->128->1 zero-init residual; beta",
                 "params": 1321602, "use_max": False, "loss": "pairwise RankNet multi-positive top5-weighted", "seed": 0},
        "E_features": EFEAT, "E_standardization": "reuse TRAIN (C9_TRAIN top20) emean/estd saved in model joblibs (NOT recomputed on VAL)",
        "representation": "frozen gte_qwen queries_all[row_all] (audited == exact dense query rep, cosine 1.0) + nodes.npy; unit-norm",
        "artifact_sha256_32": {
            "C11_models.joblib": sha(f"{OUT}/C11_models.joblib"), "C12_models.joblib": sha(f"{OUT}/C12_models.joblib"),
            "C8c_reranker_fulltrain.joblib": sha(f"{OUT}/C8c_reranker_fulltrain.joblib"),
            "2wiki_nodes.npy": sha("data/ukb_storage/2wiki_clean/gte_qwen/nodes.npy"),
            "musique_nodes.npy": sha("data/ukb_storage/musique_clean/gte_qwen/nodes.npy"),
            "2wiki_queries_all.npy": sha("data/ukb_storage/2wiki_clean/gte_qwen/queries_all.npy"),
            "musique_queries_all.npy": sha("data/ukb_storage/musique_clean/gte_qwen/queries_all.npy"),
            "l2_c11.py": sha("scratchpad/l2_c11.py"), "l2_c12.py": sha("scratchpad/l2_c12.py"),
            "_c11_qmap_2wiki": sha(f"{OUT}/_c11_qmap_2wiki_clean.npz"), "_c11_qmap_musique": sha(f"{OUT}/_c11_qmap_musique_clean.npz")},
        "E_std_hash": {"c11_emean": float(np.asarray(sd11["emean"]).sum()), "c12_emean": float(np.asarray(sd12["emean"]).sum()),
                       "emean_identical": bool(np.allclose(np.asarray(sd11["emean"]), np.asarray(sd12["emean"])))},
        "seeds": {"torch": 0, "numpy": 0}, "written_before_val_access": True,
    }
    json.dump(manifest, open("results/L2/L2_PREVAL_FREEZE_MANIFEST.json", "w"), indent=1, default=str)
    log(f"[{time.time()-T0:.0f}s] MANIFEST_WRITTEN (before VAL access)")

    # ---------- freeze backbone on FULL TRAIN ----------
    base_full = C8.fit_soft_head("ndcg50", full)
    B_tr = combined("train", full, base_full); B_di = combined("train", devinner, base_full)
    c8c = fit_c8c(B_tr["X28"], B_tr["y"], B_tr["groups"], B_di["X28"], B_di["y"], B_di["groups"])
    log(f"[{time.time()-T0:.0f}s] backbone frozen (C7b full + C8c best_iter={c8c.best_iteration})")

    ma = M11.C11a(); ma.load_state_dict(sd11["C11a"]); em11, es11 = sd11["emean"], sd11["estd"]
    mc = M12.C12(use_max=False); mc.load_state_dict(sd12["C12a"]); em12, es12 = sd12["emean"], sd12["estd"]

    # ---------- OFFICIAL VAL: build per-ds bundles, evaluate the three policies ----------
    perds = {"C8c": {}, "C11a": {}, "C12a": {}}; perq = {"C8c": {}, "C11a": {}, "C12a": {}}
    mgds = {"C8c": {}, "C11a": {}, "C12a": {}}; rescue_out = {"C12a_vs_C11a": {}, "C12a_vs_C8c": {}}; ceff = {}
    for ds in DS:
        vqi = C8.precompute_arch(ds, "val")["qi"]
        B = C9.build_bundle(ds, "val", vqi, base_full)
        for m in B["meta"]: m["ds"] = ds
        s = c8c.predict(B["X28"]).astype(np.float64)
        sp = build_val_sp(ds, B, s)
        sp["_t11"] = R.prep_tensors({**sp, "E": sp["E"]}, em11, es11)
        sp["_t"] = R.prep_tensors({**sp, "E": sp["E"]}, em12, es12); sp["_grp"] = group_all(sp)
        r8 = C9.eval_ranking(B, s)
        spc11 = dict(sp); spc11["_t"] = sp["_t11"]
        r11 = R.evaluate(ma, spc11, B, s)
        r12 = eval_c12(mc, sp, B, s)
        for nm, r in (("C8c", r8), ("C11a", r11), ("C12a", r12)):
            perds[nm][ds] = {**r, "n": r["n"]}; perq[nm][ds] = r["_perq"]; mgds[nm][ds] = multigold(r)
        rescue_out["C12a_vs_C11a"][ds] = gold_rescue_expert(sp, B, s, r11["_goldranks"], r12["_goldranks"])
        rescue_out["C12a_vs_C8c"][ds] = gold_rescue_expert(sp, B, s, r8["_goldranks"], r12["_goldranks"])
        ceff[ds] = context_effect_val(mc, ma, sp, spc11)
        log(f"[{time.time()-T0:.0f}s] {ds}: C8c ndcg5={r8['ndcg5']:.4f} C11a={r11['ndcg5']:.4f} C12a={r12['ndcg5']:.4f} all50 8/11/12={r8['all50']:.4f}/{r11['all50']:.4f}/{r12['all50']:.4f}")

    # pooled (n-weighted per-query means; all5_feas weighted by its own support)
    def pooled(nm):
        o = {}
        for k in MET:
            wk = "n_all5_feas" if k == "all5_feas" else "n"
            vals = [(perds[nm][ds].get(k), perds[nm][ds].get(wk, 0)) for ds in DS]
            num = sum(v * w for v, w in vals if isinstance(v, (int, float))); den = sum(w for v, w in vals if isinstance(v, (int, float)))
            o[k] = round(num / max(den, 1), 4)
        o["n"] = sum(perds[nm][ds]["n"] for ds in DS)
        return o
    pool = {nm: pooled(nm) for nm in ("C8c", "C11a", "C12a")}

    # bootstrap (pooled per-query arrays)
    def cat(nm, key): return np.concatenate([perq[nm][ds][key] for ds in DS])
    boot = {}
    for aa, bb in (("C11a", "C8c"), ("C12a", "C11a"), ("C12a", "C8c")):
        boot[f"{aa}_vs_{bb}"] = {m: paired_boot(cat(aa, mk), cat(bb, mk)) for m, mk in
                                 (("ndcg5", "ndcg5"), ("recall5_macro", "recall5"), ("all5_feasible", "all5"), ("mrr", "mrr"))}

    mg_pool = {nm: multigold_pool(mgds[nm]) for nm in ("C8c", "C11a", "C12a")}
    parity = {k: [round(pool["C8c"][k], 4), REF_C8C[k], round(pool["C8c"][k] - REF_C8C[k], 4)] for k in REF_C8C}
    c8c_parity = all(abs(pool["C8c"][k] - REF_C8C[k]) <= 0.010 for k in ("ndcg5", "recall5_macro"))
    all50_eq = abs(pool["C8c"]["all50"] - pool["C11a"]["all50"]) < 1e-9 and abs(pool["C8c"]["all50"] - pool["C12a"]["all50"]) < 1e-9

    def sig(cmp, mk): b = boot[cmp][mk]; return b["significant"] and b["delta"] > 0
    c11_beats_c8c = sig("C11a_vs_C8c", "ndcg5"); c12_beats_c11 = sig("C12a_vs_C11a", "ndcg5"); c12_beats_c8c = sig("C12a_vs_C8c", "ndcg5")
    tpg = mg_pool["C12a"]["3+"]["recall5"] > mg_pool["C11a"]["3+"]["recall5"]
    if c12_beats_c11 and tpg: final = "C12a"
    elif c12_beats_c8c and not c12_beats_c11: final = "C12a (ties C11a; parsimony/support -> keep C12a already selected)" if pool["C12a"]["ndcg5"] >= pool["C11a"]["ndcg5"] else "C11a"
    elif c11_beats_c8c: final = "C11a"
    else: final = "C8c"

    gates = {
        "C8C_VAL_PARITY": "PASS" if c8c_parity else "FAIL",
        "C11A_BEATS_C8C_ON_VAL": "YES" if c11_beats_c8c else "NO",
        "C12A_BEATS_C11A_ON_VAL": "YES" if c12_beats_c11 else "NO",
        "C12A_BEATS_C8C_ON_VAL": "YES" if c12_beats_c8c else "NO",
        "THREE_PLUS_GOLD_GAIN_REPLICATES": "YES" if tpg else "NO",
        "LEARNED_LIST_CONTEXT_GENERALIZES_TO_VAL": "YES" if (c12_beats_c11 and tpg) else "NO",
        "DEEP_RECALL_PRESERVED": "YES" if all50_eq else "NO",
        "FINAL_TEXT_L2_POLICY": final.split(" ")[0],
        "C12_OFFICIAL_VAL_REPLICATES": "YES" if (c12_beats_c11 and c12_beats_c8c and tpg) else "NO",
        "SAFE_TO_BEGIN_CROSS_DATASET_GENERALIZATION": "YES" if c12_beats_c8c else "NO",
        "SAFE_TO_RUN_LOCKED_TEST": "NO",
        "NEW_ENCODER_FORWARD": "PASS", "NEW_LLM_COMPONENT": "PASS",
    }
    out = {"phase": "C12 OFFICIAL-VAL milestone — C8c/C11a/C12a; per-ds + pooled; one evaluation; TEST untouched",
           "OFFICIAL_VAL_MAIN_TABLE": {
               "pooled": {nm: pool[nm] for nm in ("C8c", "C11a", "C12a")},
               "per_ds": {ds: {nm: slim(perds[nm][ds]) for nm in ("C8c", "C11a", "C12a")} for ds in DS},
               "deltas_pooled": {"C11a_minus_C8c": {k: round(pool["C11a"][k] - pool["C8c"][k], 4) for k in MET},
                                 "C12a_minus_C11a": {k: round(pool["C12a"][k] - pool["C11a"][k], 4) for k in MET},
                                 "C12a_minus_C8c": {k: round(pool["C12a"][k] - pool["C8c"][k], 4) for k in MET}}},
           "C8C_VAL_PARITY_TABLE": parity, "PAIRED_BOOTSTRAPS": boot,
           "MULTIGOLD_BREAKDOWN": {"pooled": mg_pool, "per_ds": mgds},
           "GOLD_RESCUE": rescue_out, "CONTEXT_EFFECT_ANALYSIS": ceff,
           "GATES": gates, "FINAL_TEXT_L2_POLICY": final,
           "NEW_ENCODER_FORWARD_PASSES": 0, "NEW_LLM_COMPONENTS": 0}
    json.dump(out, open("results/L2/L2_C12_OFFICIAL_VAL.json", "w"), indent=1, default=str)
    log("PARITY " + json.dumps(parity))
    log("GATES " + json.dumps(gates))
    log(f"[{time.time()-T0:.0f}s] C12_VAL_DONE final_policy={final}")


if __name__ == "__main__":
    main()
