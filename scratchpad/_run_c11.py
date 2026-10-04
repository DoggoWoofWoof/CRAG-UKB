"""C11 driver: train C11a (single interaction MLP) and C11b (K=4 multi-offset mixture MLP), residual on C8c.
Train on C9_TRAIN top20, select on C9_DEV, confirm once on DEV_INNER. Pairwise top5-focused ranking loss,
multi-positive, OOF C8c backbone/evidence at train. Reorders within top50 -> ALL@50 preserved. NO VAL/TEST.
NEW_ENCODER_FORWARD_PASSES=0, NEW_LLM_COMPONENTS=0 (only frozen cached gte_qwen embeddings + a small MLP)."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch, joblib
import l2_c8 as C8, l2_c9 as C9, l2_c11 as M
from _run_c9 import starts_of, multigold, rescue, paired_boot
from _run_c10_mmr import singlegold, C8slim
from _run_c10b import rescue_by_multigold
OUT = C8.OUT; DS = C8.DS; CAP = 20; log = lambda *a: print(*a, flush=True); T0 = time.time()
torch.manual_seed(0); np.random.seed(0)


def load_split(name):
    z = np.load(f"{OUT}/_c11_{name}.npz"); meta = joblib.load(f"{OUT}/_c11_{name}_meta.joblib")
    g = z["groups"].astype(np.int64); st = starts_of(g)
    return {"qv": z["qv"].astype(np.float32), "dv": z["dv"].astype(np.float32), "E": z["E"].astype(np.float32),
            "c8c": z["c8c"].astype(np.float32), "y": z["y"].astype(np.int64), "g": g, "st": st, "meta": meta}


def zc8c(c8c, g, st):
    """z-score C8c within each query window (order-preserving, scale-stabilizing backbone)."""
    out = np.empty_like(c8c)
    for i in range(len(g)):
        a, b = st[i], st[i + 1]; s = c8c[a:b]; out[a:b] = (s - s.mean()) / (s.std() + 1e-6)
    return out


def prep_tensors(sp, emean, estd):
    E = (sp["E"] - emean) / estd
    return {"qv": torch.tensor(sp["qv"]), "dv": torch.tensor(sp["dv"]), "E": torch.tensor(E, dtype=torch.float32),
            "c8cz": torch.tensor(zc8c(sp["c8c"], sp["g"], sp["st"])), "y": torch.tensor(sp["y"]),
            "g": sp["g"], "st": sp["st"]}


def train(model, tr, epochs=25, bq=256, lr=1e-3, wd=1e-5, loff=0.0, log_every=5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Q = len(tr["g"]); qv, dv, E, c8cz, y = tr["qv"], tr["dv"], tr["E"], tr["c8cz"], tr["y"]; st = tr["st"]; g = tr["g"]
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(Q); tot = 0.0; nb = 0
        for bs in range(0, Q, bq):
            qb = perm[bs:bs + bq]
            rows = np.concatenate([np.arange(st[i], st[i + 1]) for i in qb])
            ridx = np.concatenate([np.full(g[i], k) for k, i in enumerate(qb)])  # row -> batch-local query index
            qv_u = qv[qb]; d = dv[rows]; Eb = E[rows]; cb = c8cz[rows]; yb = y[rows]
            delta = model.delta_grouped(qv_u, torch.tensor(ridx), d, Eb); s = cb + model.beta * delta
            gi = []; off = 0
            for i in qb:
                gi.append((off, off + g[i])); off += g[i]
            br = torch.tensor(np.concatenate([np.arange(g[i]) for i in qb]), dtype=torch.float32)
            loss = M.pair_loss(s, yb, br, gi)
            if loff > 0 and hasattr(model, "offset_reg"): loss = loss + loff * model.offset_reg()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if ep % log_every == 0 or ep == epochs - 1:
            log(f"[{time.time()-T0:.0f}s]   ep{ep} loss={tot/nb:.4f} beta={float(model.beta):.3f}")
    return model


@torch.no_grad()
def c11_scores(model, sp_t):
    model.eval(); out = model.delta(sp_t["qv_rows"], sp_t["dv"], sp_t["E"]).numpy()
    return out  # Delta per row; final within-window score = c8cz + beta*Delta


def rows_qv(sp, sp_t):
    qrows = np.concatenate([np.full(sp["g"][i], i) for i in range(len(sp["g"]))])
    sp_t["qv_rows"] = sp_t["qv"][qrows]; return sp_t


def evaluate(model, sp, cacheB, s_full):
    """Build cand_scores for C9.eval_ranking: window candidates (top20) get 1e6+final; other pool keep C8c score."""
    sp_t = rows_qv(sp, sp["_t"]); delta = c11_scores(model, sp_t); beta = float(model.beta)
    finalw = sp["_t"]["c8cz"].numpy() + beta * delta
    st = sp["st"]; groupsB = cacheB["groups"]; stB = starts_of(groupsB); cand = np.empty(len(cacheB["y"]), np.float64)
    pos = 0
    for i, (m, gB) in enumerate(zip(cacheB["meta"], groupsB)):
        aB = stB[i]; scq = s_full[aB:aB + gB]; order = np.argsort(-scq, kind="stable"); win = order[:min(CAP, gB)]
        cs = scq.astype(np.float64).copy()                 # non-window pool keep C8c score
        a = st[i]; fw = finalw[a:a + len(win)]
        cs[win] = 1e6 + fw                                 # window on top, ordered by C11 final
        cand[aB:aB + gB] = cs; pos += 1
    return C9.eval_ranking(cacheB, cand)


def build_offset_stats(model, sp):
    if not hasattr(model, "offset_pairwise_cos"): return None
    sp_t = rows_qv(sp, sp["_t"])
    with torch.no_grad():
        model.eval(); _ = model.delta(sp_t["qv_rows"][:4096], sp_t["dv"][:4096], sp_t["E"][:4096])
        cos = float(model.offset_pairwise_cos())
    return {"mean_pairwise_offset_cos": round(cos, 4), "MULTI_OFFSET_COLLAPSE": "YES" if cos > 0.98 else "NO"}


def param_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def main():
    tr = load_split("train"); dev = load_split("dev"); di = load_split("di")
    emean = tr["E"].mean(0); estd = tr["E"].std(0) + 1e-6
    tr["_t"] = prep_tensors(tr, emean, estd); dev["_t"] = prep_tensors(dev, emean, estd); di["_t"] = prep_tensors(di, emean, estd)
    log(f"[{time.time()-T0:.0f}s] splits loaded tr_rows={len(tr['y'])} dev_q={len(dev['g'])} di_q={len(di['g'])}")

    cache = joblib.load(f"{OUT}/_c10b_cache.joblib")
    B_dev, B_di, s_dev, s_di = cache["B_dev"], cache["B_di"], cache["s_dev"], cache["s_di"]
    c8c_dev = C9.eval_ranking(B_dev, s_dev); c8c_di = C9.eval_ranking(B_di, s_di)
    log(f"[{time.time()-T0:.0f}s] C8c baseline dev ndcg5={c8c_dev['ndcg5']:.4f}")

    # ---- unit test: multi-positive pair loss touches all golds ----
    ut = unit_test_multipos()
    log(f"[{time.time()-T0:.0f}s] multipos unit test: {ut}")

    # ---- C11a ----
    t0 = time.time(); ma = M.C11a(); train(ma, tr["_t"], epochs=18, bq=512, lr=1.3e-3)
    ta = time.time() - t0
    a_dev = evaluate(ma, dev, B_dev, s_dev)
    log(f"[{time.time()-T0:.0f}s] C11a dev ndcg5={a_dev['ndcg5']:.4f} recall5={a_dev['recall5_macro']:.4f} all50={a_dev['all50']:.4f} beta={float(ma.beta):.3f}")

    # ---- C11b (K=4 multi-offset) ----
    t0 = time.time(); mb = M.C11b(K=4); train(mb, tr["_t"], epochs=18, bq=512, lr=1.3e-3, loff=1e-3)
    tb = time.time() - t0
    b_dev = evaluate(mb, dev, B_dev, s_dev)
    offstats = build_offset_stats(mb, dev)
    log(f"[{time.time()-T0:.0f}s] C11b dev ndcg5={b_dev['ndcg5']:.4f} recall5={b_dev['recall5_macro']:.4f} offset={offstats}")

    # ---- select on C9_DEV ----
    cands = {"C8c": c8c_dev, "C11a": a_dev, "C11b": b_dev}
    sel = max(cands, key=lambda k: (cands[k]["ndcg5"], cands[k]["recall5_macro"]))
    selmodel = {"C11a": ma, "C11b": mb}.get(sel)
    sel_dev = cands[sel]
    log(f"[{time.time()-T0:.0f}s] SELECTED on C9_DEV = {sel}")

    # ---- DEV_INNER confirm (once) ----
    conf = {"C8c": c8c_di, "C11a": evaluate(ma, di, B_di, s_di), "C11b": evaluate(mb, di, B_di, s_di)}

    # ---- diagnostics vs C8c on C9_DEV (for selected + both) ----
    diag = {}
    for nm, r in (("C11a", a_dev), ("C11b", b_dev)):
        mdl = {"C11a": ma, "C11b": mb}[nm]
        resc = rescue(c8c_dev["_goldranks"], r["_goldranks"]); rmg = rescue_by_multigold(B_dev, c8c_dev["_goldranks"], r["_goldranks"])
        boot = {mk: paired_boot(r["_perq"][mk], c8c_dev["_perq"][mk]) for mk in ("ndcg5", "recall5", "ndcg50", "all50")}
        diag[nm] = {"gold_rescue": resc, "gold_rescue_by_multigold": rmg, "bootstrap_vs_C8c": boot,
                    "multigold": multigold(r), "single_gold": singlegold(r),
                    "specialist_and_fp": specialist_fp(mdl, dev, B_dev, s_dev, c8c_dev, r)}

    res = {"phase": "C11 — learned query-candidate interaction MLP (residual on C8c); C9_DEV select, DEV_INNER confirm; NO VAL/TEST",
           "NEW_ENCODER_FORWARD_PASSES": 0, "NEW_LLM_COMPONENTS": 0,
           "representation": "frozen gte_qwen query+node embeddings (exact dense encoder; qmap-recovered), unit-norm",
           "C11A_SIMPLE_MLP": {"C9_DEV": C8slim(a_dev), "params": param_count(ma), "train_sec": round(ta, 1)},
           "C11B_MULTI_OFFSET_MLP": {"C9_DEV": C8slim(b_dev), "params": param_count(mb), "train_sec": round(tb, 1), "K": 4},
           "C8c_C9_DEV": C8slim(c8c_dev),
           "OFFSET_ANALYSIS": offstats,
           "SELECTED_ON_C9DEV": sel,
           "DEV_INNER_CONFIRM": {k: C8slim(v) for k, v in conf.items()},
           "DIAGNOSTICS": diag,
           "multipos_unit_test": ut,
           "RUNTIME_COST": {"C11a_params": param_count(ma), "C11b_params": param_count(mb), "cap": CAP,
                            "cand_evals_per_query": CAP, "device": "cpu"}}
    res["GATES"] = gates(a_dev, b_dev, c8c_dev, sel, cands[sel], offstats, diag)
    json.dump(res, open("results/L2/L2_C11_MLP_INTERACTION.json", "w"), indent=1, default=str)
    joblib.dump({"C11a": ma.state_dict(), "C11b": mb.state_dict(), "emean": emean, "estd": estd}, f"{OUT}/C11_models.joblib")
    log(f"[{time.time()-T0:.0f}s] C11_DONE selected={sel}")
    print(json.dumps({"C11a": C8slim(a_dev), "C11b": C8slim(b_dev), "C8c": C8slim(c8c_dev), "selected": sel,
                      "offset": offstats, "dev_inner": {k: C8slim(v) for k, v in conf.items()},
                      "gates": res["GATES"], "diag_boot": {k: diag[k]["bootstrap_vs_C8c"] for k in diag},
                      "rescue_mg": {k: diag[k]["gold_rescue_by_multigold"] for k in diag}}, indent=1, default=str))


def unit_test_multipos():
    """Confirm pair_loss produces gradient pressure for EVERY positive (multi-gold), not just the top gold."""
    torch.manual_seed(1); s = torch.zeros(6, requires_grad=True)
    y = torch.tensor([1, 0, 1, 0, 1, 0]); br = torch.tensor([5., 0., 6., 1., 7., 2.])
    loss = M.pair_loss(s, y, br, [(0, 6)]); loss.backward()
    gpos = s.grad[y == 1]
    return {"all_positives_have_grad": bool((gpos.abs() > 0).all()), "n_pos": int((y == 1).sum()),
            "grad_pos_mean": round(float(gpos.mean()), 4)}


def specialist_fp(model, sp, cacheB, s_full, c8c_res, c11_res):
    """Task 16/17: for golds rescued into top5 and FPs removed, record strongest expert + mean Delta sign."""
    import numpy as _np
    sp_t = rows_qv(sp, sp["_t"]); delta = c11_scores(model, sp_t)
    st = sp["st"]; exps = ["dense", "splade", "offset", "mixture", "relation"]
    ci = [C9.C9_FEATNAMES.index(f"c_{e}") for e in exps]
    from collections import Counter
    resc_exp = Counter(); d_gold = []; d_fp = []
    base_gr = c8c_res["_goldranks"]; new_gr = c11_res["_goldranks"]
    stB = starts_of(cacheB["groups"])
    for i, (m, gB) in enumerate(zip(cacheB["meta"], cacheB["groups"])):
        aB = stB[i]; scq = s_full[aB:aB + gB]; order = _np.argsort(-scq, kind="stable"); win = order[:min(CAP, gB)]
        a = st[i]; dl = delta[a:a + len(win)]; goldset = set(m["gold"].tolist())
        for t, prow in enumerate(win):
            li = int(m["pool"][prow]); key = (m["qi"], li); isg = li in goldset
            if isg and new_gr.get(key, 99) < 5 and base_gr.get(key, 99) >= 5:
                d_gold.append(float(dl[t]))
                # strongest expert from contrib features in E: E cols order = EFEAT; c_* at indices 3..7
                cvec = sp["E"][a + t, 3:8]; resc_exp[exps[int(_np.argmax(cvec))]] += 1
            if (not isg) and base_gr.get((m["qi"], li), 99) < 5 and new_gr.get(key, 99) >= 5:
                d_fp.append(float(dl[t]))
    return {"rescued_gold_strongest_expert": dict(resc_exp),
            "mean_delta_rescued_gold": round(float(_np.mean(d_gold)), 4) if d_gold else None,
            "mean_delta_removed_fp": round(float(_np.mean(d_fp)), 4) if d_fp else None,
            "n_rescued": len(d_gold), "n_removed_fp": len(d_fp)}


def gates(a_dev, b_dev, c8c, sel, seld, offstats, diag):
    def sig(nm):
        b = diag.get(nm, {}).get("bootstrap_vs_C8c", {}).get("ndcg5", {});
        return b.get("significant") and b.get("delta", 0) > 0
    a_help = sig("C11a"); b_help = sig("C11b")
    b_beats_a = b_dev["ndcg5"] > a_dev["ndcg5"]
    seld_mg = diag.get(sel, {}).get("gold_rescue_by_multigold", {})
    mg_imp = bool(seld_mg.get("3+", {}).get("net", 0) > 0) if seld_mg else False
    return {"MLP_INTERACTION_ADDS_VALUE": "YES" if a_help else "NO",
            "MULTI_OFFSET_ADDS_VALUE": "YES" if (b_help and b_beats_a) else "NO",
            "MULTI_OFFSET_COLLAPSE": offstats["MULTI_OFFSET_COLLAPSE"] if offstats else "NA",
            "C11_BEATS_C8C": "YES" if seld["ndcg5"] > c8c["ndcg5"] and sig(sel) else "NO",
            "MULTIGOLD_RECALL_IMPROVES": "YES" if mg_imp else "NO",
            "SPECIALIST_RESCUE_VALIDATION_LEARNED": _spec_gate(diag, sel),
            "DEEP_RECALL_PRESERVED": "YES" if abs(seld["all50"] - c8c["all50"]) < 1e-9 else "NO",
            "NO_NEW_LLM_COMPONENT": "PASS",
            "SAFE_TO_RUN_OFFICIAL_VAL": "YES" if (seld["ndcg5"] > c8c["ndcg5"] and sig(sel)) else "NO",
            "SAFE_TO_FREEZE_L2": "NO"}


def _spec_gate(diag, sel):
    sf = diag.get(sel, {}).get("specialist_and_fp", {})
    dg = sf.get("mean_delta_rescued_gold"); dfp = sf.get("mean_delta_removed_fp")
    if dg is None or dfp is None: return "NO"
    return "YES" if dg > 0 and dfp < 0 else "NO"


if __name__ == "__main__":
    main()
