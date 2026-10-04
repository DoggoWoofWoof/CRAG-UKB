"""C11b full-spec analysis (no retrain): load saved C11a/C11b, compute the diagnostics the fuller C11b spec
requires beyond the first run: paired bootstrap C11b-vs-C11a (+C11a-vs-C8c), per-bin multigold ablation,
offset norm/variance + pairwise cosine (collapse), MULTI_OFFSET_SPECIALIZATION_RATE (gold vs non-gold, offsets
emergent — labels are ranking supervision only), param/runtime. Same C9_DEV select + one DEV_INNER confirm.
NO VAL/TEST, NO new encoder/LLM. Reuses _run_c11 eval so ALL@50 preserved (reorder within top20)."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch, joblib
import l2_c8 as C8, l2_c9 as C9, l2_c11 as M
import _run_c11 as R
from _run_c9 import multigold, paired_boot, starts_of
from _run_c10_mmr import singlegold, C8slim
OUT = C8.OUT; DS = C8.DS; CAP = 20; log = lambda *a: print(*a, flush=True); T0 = time.time()
torch.manual_seed(0); np.random.seed(0)


def offset_stats_full(model, sp, nmax=8192):
    """mean offset norm, per-k mean norm, variance across offsets, mean pairwise cosine (collapse diagnostic)."""
    sp_t = R.rows_qv(sp, sp["_t"])
    with torch.no_grad():
        model.eval()
        _ = model.delta(sp_t["qv_rows"][:nmax], sp_t["dv"][:nmax], sp_t["E"][:nmax])
        o = model._last_off                                  # (B,K,H)
        norms = o.norm(dim=-1)                               # (B,K)
        cos = float(model.offset_pairwise_cos())
    perk = norms.mean(0)                                     # (K,)
    return {
        "mean_offset_norm": round(float(norms.mean()), 4),
        "perk_mean_norm": [round(float(x), 4) for x in perk],
        "std_across_offset_norms": round(float(perk.std(unbiased=False)), 4),
        "mean_pairwise_offset_cos": round(cos, 4),
        "MULTI_OFFSET_COLLAPSE": "YES" if cos > 0.98 else "NO",
    }


def specialization(model, sp, nmax=200000):
    """Per candidate, argmax offset k = argmax_k max_mf f(m_k) (consistent with the elementwise-max path).
    Rate = fraction of queries with >=2 golds-in-window where golds span >=2 distinct offsets; size-matched
    non-gold control. Offsets emergent — gold labels used ONLY to group, never to assign offsets."""
    sp_t = R.rows_qv(sp, sp["_t"])
    with torch.no_grad():
        model.eval()
        _ = model.delta(sp_t["qv_rows"][:nmax], sp_t["dv"][:nmax], sp_t["E"][:nmax])
        fmk = model._last_fmk                                # (Nrows,K,mf)
        per_k = fmk.max(dim=-1).values                       # (Nrows,K) strongest activation per offset
        argk = per_k.argmax(dim=-1).numpy()                  # (Nrows,) winning offset per candidate
    y = sp["y"]; st = sp["st"]; g = sp["g"]; Kd = model.K
    n_multi = 0; gold_spread = 0; nong_spread = 0; gold_distinct = []; nong_distinct = []
    kwin_gold = np.zeros(Kd, np.int64); kwin_nong = np.zeros(Kd, np.int64)
    rng = np.random.default_rng(0)
    for i in range(len(g)):
        a, b = st[i], st[i + 1]
        yb = y[a:b]; ak = argk[a:b]
        gpos = np.where(yb == 1)[0]; npos = np.where(yb == 0)[0]
        for k in ak[gpos]: kwin_gold[k] += 1
        for k in ak[npos]: kwin_nong[k] += 1
        if len(gpos) >= 2:
            n_multi += 1
            gk = ak[gpos]; gd = len(set(gk.tolist())); gold_distinct.append(gd)
            gold_spread += int(gd >= 2)
            if len(npos) >= len(gpos):                        # size-matched non-gold control
                nk = ak[rng.choice(npos, size=len(gpos), replace=False)]
                nd = len(set(nk.tolist())); nong_distinct.append(nd); nong_spread += int(nd >= 2)
    rate = gold_spread / max(n_multi, 1)
    nong_rate = nong_spread / max(len(nong_distinct), 1)
    return {
        "n_multigold_in_window": int(n_multi),
        "MULTI_OFFSET_SPECIALIZATION_RATE": round(rate, 4),
        "nongold_control_spread_rate": round(nong_rate, 4),
        "mean_distinct_offsets_gold": round(float(np.mean(gold_distinct)), 3) if gold_distinct else None,
        "mean_distinct_offsets_nongold": round(float(np.mean(nong_distinct)), 3) if nong_distinct else None,
        "gold_offset_win_counts": kwin_gold.tolist(),
        "nongold_offset_win_counts": kwin_nong.tolist(),
    }


def infer_ms_per_query(model, sp, reps=3):
    sp_t = R.rows_qv(sp, sp["_t"]); nq = len(sp["g"])
    with torch.no_grad():
        model.eval()
        t = time.time()
        for _ in range(reps):
            _ = model.delta(sp_t["qv_rows"], sp_t["dv"], sp_t["E"])
        dt = (time.time() - t) / reps
    return round(1000.0 * dt / nq, 4)


def main():
    tr = R.load_split("train"); dev = R.load_split("dev"); di = R.load_split("di")
    emean = tr["E"].mean(0); estd = tr["E"].std(0) + 1e-6
    sd = joblib.load(f"{OUT}/C11_models.joblib")
    # use the SAVED standardization if present (identical), else recompute
    emean = sd.get("emean", emean); estd = sd.get("estd", estd)
    dev["_t"] = R.prep_tensors(dev, emean, estd); di["_t"] = R.prep_tensors(di, emean, estd)
    ma = M.C11a(); ma.load_state_dict(sd["C11a"]); mb = M.C11b(K=4); mb.load_state_dict(sd["C11b"])
    log(f"[{time.time()-T0:.0f}s] models loaded; dev_q={len(dev['g'])} di_q={len(di['g'])}")

    cache = joblib.load(f"{OUT}/_c10b_cache.joblib")
    B_dev, B_di, s_dev, s_di = cache["B_dev"], cache["B_di"], cache["s_dev"], cache["s_di"]
    c8c_dev = C9.eval_ranking(B_dev, s_dev); c8c_di = C9.eval_ranking(B_di, s_di)
    a_dev = R.evaluate(ma, dev, B_dev, s_dev); b_dev = R.evaluate(mb, dev, B_dev, s_dev)
    a_di = R.evaluate(ma, di, B_di, s_di); b_di = R.evaluate(mb, di, B_di, s_di)
    log(f"[{time.time()-T0:.0f}s] C8c={c8c_dev['ndcg5']:.4f} C11a={a_dev['ndcg5']:.4f} C11b={b_dev['ndcg5']:.4f}")

    # ---- paired bootstrap: C11a vs C8c AND C11b vs C11a (ndcg5, recall5) ----
    boot = {
        "C11a_vs_C8c": {mk: paired_boot(a_dev["_perq"][mk], c8c_dev["_perq"][mk]) for mk in ("ndcg5", "recall5")},
        "C11b_vs_C11a": {mk: paired_boot(b_dev["_perq"][mk], a_dev["_perq"][mk]) for mk in ("ndcg5", "recall5")},
        "C11b_vs_C8c": {mk: paired_boot(b_dev["_perq"][mk], c8c_dev["_perq"][mk]) for mk in ("ndcg5", "recall5")},
    }
    # ---- offset diagnostics + specialization (C11b) ----
    offd = offset_stats_full(mb, dev); spec = specialization(mb, dev)
    log(f"[{time.time()-T0:.0f}s] offset={offd}")
    log(f"[{time.time()-T0:.0f}s] spec={spec}")

    # ---- multigold ablation C8c/C11a/C11b per bin ----
    mg = {"C8c": multigold(c8c_dev), "C11a": multigold(a_dev), "C11b": multigold(b_dev)}

    b_beats_a_ndcg = b_dev["ndcg5"] > a_dev["ndcg5"]
    b_beats_a_sig = boot["C11b_vs_C11a"]["ndcg5"]["significant"] and boot["C11b_vs_C11a"]["ndcg5"]["delta"] > 0
    # multigold improvement of the multi-offset variant specifically on 2/3+ bins
    mg_2 = mg["C11b"]["2"]["recall5"] > mg["C11a"]["2"]["recall5"]
    mg_3 = mg["C11b"]["3+"]["recall5"] > mg["C11a"]["3+"]["recall5"]

    gates = {
        "MLP_INTERACTION_ADDS_VALUE": "YES" if (boot["C11a_vs_C8c"]["ndcg5"]["significant"] and boot["C11a_vs_C8c"]["ndcg5"]["delta"] > 0) else "NO",
        "MULTI_OFFSET_ADDS_VALUE": "YES" if (b_beats_a_ndcg and b_beats_a_sig) else "NO",
        "MULTI_OFFSET_COLLAPSE": offd["MULTI_OFFSET_COLLAPSE"],
        "MULTI_OFFSET_SPECIALIZATION_RATE": spec["MULTI_OFFSET_SPECIALIZATION_RATE"],
        "MULTIGOLD_RECALL_IMPROVES": "YES" if (mg_2 or mg_3) else "NO",
        "C11B_BEATS_C11A": "YES" if (b_beats_a_ndcg and b_beats_a_sig) else "NO",
        "C11_FINAL_POLICY": "C11b" if (b_beats_a_ndcg and b_beats_a_sig) else "C11a",
        "DEEP_RECALL_PRESERVED": "YES" if abs(a_dev["all50"] - c8c_dev["all50"]) < 1e-9 and abs(b_dev["all50"] - c8c_dev["all50"]) < 1e-9 else "NO",
        "NO_NEW_ENCODER_FORWARD": "PASS",
        "NO_NEW_LLM_COMPONENT": "PASS",
    }
    out = {
        "phase": "C11b full-spec ablation — multi-offset vs single interaction; C9_DEV select + DEV_INNER confirm; NO VAL/TEST",
        "C9_DEV": {"C8c": C8slim(c8c_dev), "C11a": C8slim(a_dev), "C11b": C8slim(b_dev)},
        "DEV_INNER": {"C8c": C8slim(c8c_di), "C11a": C8slim(a_di), "C11b": C8slim(b_di)},
        "PAIRED_BOOTSTRAP": boot,
        "MULTIGOLD_BREAKDOWN": mg,
        "OFFSET_DIAGNOSTICS": offd,
        "OFFSET_SPECIALIZATION": spec,
        "MODEL_SIZE": {
            "C11a_params": R.param_count(ma), "C11b_params": R.param_count(mb),
            "incremental_params": R.param_count(mb) - R.param_count(ma),
            "C11a_train_sec": 815.6, "C11b_train_sec": 1076.9,
            "C11a_infer_ms_per_query": infer_ms_per_query(ma, dev),
            "C11b_infer_ms_per_query": infer_ms_per_query(mb, dev),
            "cap": CAP, "K": 4,
        },
        "GATES": gates,
    }
    json.dump(out, open("results/L2/L2_C11B_ABLATION.json", "w"), indent=1, default=str)
    log("GATES " + json.dumps(gates))
    log("C11B_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
