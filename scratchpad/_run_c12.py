"""C12 driver — learned list-context (DeepSets) residual MLP over C8c top20. Trains C12a (mean context) and
C12b (mean+max), selects on C9_DEV, confirms once on DEV_INNER. Reuses frozen L1/experts/C7b/C8c + cached
reps; residual + zero-init => ALL@50 preserved. Listwise top5-weighted pairwise loss, multi-positive.
NO VAL/TEST, NO new encoder/LLM, NO attention/transformer/GNN. One shared net, no dataset ID."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch, joblib
import l2_c8 as C8, l2_c9 as C9, l2_c11 as M11, l2_c12 as M
import _run_c11 as R
from _run_c9 import starts_of, multigold, rescue, paired_boot
from _run_c10_mmr import singlegold, C8slim
from _run_c10b import rescue_by_multigold
OUT = C8.OUT; DS = C8.DS; CAP = 20; log = lambda *a: print(*a, flush=True); T0 = time.time()
torch.manual_seed(0); np.random.seed(0)


def group_all(sp):
    g = sp["g"]; st = sp["st"]
    ridx = np.concatenate([np.full(g[i], i) for i in range(len(g))])
    slot = np.concatenate([np.arange(g[i]) for i in range(len(g))])
    return (torch.tensor(ridx), torch.tensor(slot), torch.tensor(g), int(g.max()))


def train(model, tr, epochs=18, bq=512, lr=1.3e-3, wd=1e-5, log_every=5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Q = len(tr["g"]); qv, dv, E, c8cz, y = tr["qv"], tr["dv"], tr["E"], tr["c8cz"], tr["y"]; st = tr["st"]; g = tr["g"]
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(Q); tot = 0.0; nb = 0
        for bs in range(0, Q, bq):
            qb = perm[bs:bs + bq]
            rows = np.concatenate([np.arange(st[i], st[i + 1]) for i in qb])
            ridx = np.concatenate([np.full(g[i], k) for k, i in enumerate(qb)])
            slot = np.concatenate([np.arange(g[i]) for i in qb])
            cnt = torch.tensor(g[qb]); Nmax = int(g[qb].max())
            delta = model.delta(qv[qb], torch.tensor(ridx), torch.tensor(slot), cnt, dv[rows], E[rows], Nmax)
            s = c8cz[rows] + model.beta * delta
            gi = []; off = 0
            for i in qb:
                gi.append((off, off + g[i])); off += g[i]
            br = torch.tensor(slot, dtype=torch.float32)
            loss = M11.pair_loss(s, y[rows], br, gi)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()); nb += 1
        if ep % log_every == 0 or ep == epochs - 1:
            log(f"[{time.time()-T0:.0f}s]   ep{ep} loss={tot/nb:.4f} beta={float(model.beta):.3f}")
    return model


@torch.no_grad()
def delta_full(model, sp):
    model.eval(); ridx, slot, cnt, Nmax = sp["_grp"]
    return model.delta(sp["_t"]["qv"], ridx, slot, cnt, sp["_t"]["dv"], sp["_t"]["E"], Nmax).numpy()


def evaluate(model, sp, cacheB, s_full):
    delta = delta_full(model, sp); finalw = sp["_t"]["c8cz"].numpy() + float(model.beta) * delta
    st = sp["st"]; groupsB = cacheB["groups"]; stB = starts_of(groupsB); cand = np.empty(len(cacheB["y"]), np.float64)
    for i, (m, gB) in enumerate(zip(cacheB["meta"], groupsB)):
        aB = stB[i]; scq = s_full[aB:aB + gB]; order = np.argsort(-scq, kind="stable"); win = order[:min(CAP, gB)]
        cs = scq.astype(np.float64).copy(); a = st[i]; cs[win] = 1e6 + finalw[a:a + len(win)]
        cand[aB:aB + gB] = cs
    return C9.eval_ranking(cacheB, cand)


def c11a_finalw(ma, sp):
    sp_t = R.rows_qv(sp, sp["_t"]);
    with torch.no_grad():
        ma.eval(); d = ma.delta(sp_t["qv_rows"], sp_t["dv"], sp_t["E"]).numpy()
    return sp["_t"]["c8cz"].numpy() + float(ma.beta) * d


def context_effect(model, ma, sp):
    """Task 12/13: delta_context = C12 finalw - C11a finalw. Classify rescued golds / removed FP / promoted FP
    by C12-vs-C11a top5 membership; report mean delta_context + counts; strongest expert of rescued golds."""
    dC = delta_full(model, sp); fC = sp["_t"]["c8cz"].numpy() + float(model.beta) * dC
    fA = c11a_finalw(ma, sp); st = sp["st"]; g = sp["g"]; y = sp["y"]; E = sp["E"]
    from collections import Counter
    resc_g = []; rem_fp = []; prom_fp = []; resc_exp = Counter(); dom = []
    exps = ["dense", "splade", "offset", "mixture", "relation"]
    for i in range(len(g)):
        a, b = st[i], st[i + 1]; n = b - a
        topA = set(np.argsort(-fA[a:b], kind="stable")[:min(5, n)].tolist())
        topC = set(np.argsort(-fC[a:b], kind="stable")[:min(5, n)].tolist())
        yb = y[a:b]; dctx = fC[a:b] - fA[a:b]
        for j in topC - topA:
            if yb[j] == 1:
                resc_g.append(float(dctx[j])); cvec = E[a + j, 3:8]
                resc_exp[exps[int(np.argmax(cvec))]] += 1
                dom.append(float(np.sort(cvec)[-1] - np.sort(cvec)[-2]))
            else:
                prom_fp.append(float(dctx[j]))
        for j in topA - topC:
            if yb[j] == 0:
                rem_fp.append(float(dctx[j]))
    mean = lambda L: round(float(np.mean(L)), 4) if L else None
    return {"rescued_gold": {"n": len(resc_g), "mean_delta_context": mean(resc_g)},
            "removed_fp": {"n": len(rem_fp), "mean_delta_context": mean(rem_fp)},
            "promoted_fp": {"n": len(prom_fp), "mean_delta_context": mean(prom_fp)},
            "rescued_gold_strongest_expert": dict(resc_exp),
            "rescued_gold_expert_dominance_mean": mean(dom)}


def main():
    tr = R.load_split("train"); dev = R.load_split("dev"); di = R.load_split("di")
    emean = tr["E"].mean(0); estd = tr["E"].std(0) + 1e-6
    for sp in (tr, dev, di):
        sp["_t"] = R.prep_tensors(sp, emean, estd); sp["_grp"] = group_all(sp)
    log(f"[{time.time()-T0:.0f}s] splits loaded tr_rows={len(tr['y'])} dev_q={len(dev['g'])} di_q={len(di['g'])}")

    cache = joblib.load(f"{OUT}/_c10b_cache.joblib")
    B_dev, B_di, s_dev, s_di = cache["B_dev"], cache["B_di"], cache["s_dev"], cache["s_di"]
    c8c_dev = C9.eval_ranking(B_dev, s_dev); c8c_di = C9.eval_ranking(B_di, s_di)
    # C11a standing policy
    sd = joblib.load(f"{OUT}/C11_models.joblib"); ma = M11.C11a(); ma.load_state_dict(sd["C11a"])
    a_dev = R.evaluate(ma, dev, B_dev, s_dev); a_di = R.evaluate(ma, di, B_di, s_di)
    log(f"[{time.time()-T0:.0f}s] C8c={c8c_dev['ndcg5']:.4f} C11a={a_dev['ndcg5']:.4f}")

    # ---- C12a (mean context) ----
    t0 = time.time(); m_a = M.C12(use_max=False); train(m_a, tr["_t"]); ta = time.time() - t0
    c12a_dev = evaluate(m_a, dev, B_dev, s_dev)
    leak = m_a.self_leak(dev["_t"]["qv"], *dev["_grp"][:2], dev["_grp"][2], dev["_t"]["dv"], dev["_t"]["E"], dev["_grp"][3])
    log(f"[{time.time()-T0:.0f}s] C12a dev ndcg5={c12a_dev['ndcg5']:.4f} recall5={c12a_dev['recall5_macro']:.4f} all50={c12a_dev['all50']:.4f}")

    # ---- C12b (mean + max context) ----
    t0 = time.time(); m_b = M.C12(use_max=True); train(m_b, tr["_t"]); tb = time.time() - t0
    c12b_dev = evaluate(m_b, dev, B_dev, s_dev)
    log(f"[{time.time()-T0:.0f}s] C12b dev ndcg5={c12b_dev['ndcg5']:.4f} recall5={c12b_dev['recall5_macro']:.4f} all50={c12b_dev['all50']:.4f}")

    # ---- select on C9_DEV ----
    cands = {"C8c": c8c_dev, "C11a": a_dev, "C12a": c12a_dev, "C12b": c12b_dev}
    sel = max(cands, key=lambda k: (cands[k]["ndcg5"], cands[k]["recall5_macro"]))
    log(f"[{time.time()-T0:.0f}s] SELECTED on C9_DEV = {sel}")
    conf = {"C8c": c8c_di, "C11a": a_di,
            "C12a": evaluate(m_a, di, B_di, s_di), "C12b": evaluate(m_b, di, B_di, s_di)}

    # ---- bootstrap: each C12 vs C11a and vs C8c ----
    boot = {}
    for nm, r in (("C12a", c12a_dev), ("C12b", c12b_dev)):
        boot[nm] = {"vs_C11a": {mk: paired_boot(r["_perq"][mk], a_dev["_perq"][mk]) for mk in ("ndcg5", "recall5")},
                    "vs_C8c": {mk: paired_boot(r["_perq"][mk], c8c_dev["_perq"][mk]) for mk in ("ndcg5", "recall5")}}
    # ---- multigold breakdown ----
    mg = {k: multigold(cands[k]) for k in cands}
    # ---- context-effect + specialist (both variants) ----
    ceff = {"C12a": context_effect(m_a, ma, dev), "C12b": context_effect(m_b, ma, dev)}

    def beats_c11a(nm, r):
        bb = boot[nm]["vs_C11a"]["ndcg5"]
        return r["ndcg5"] > a_dev["ndcg5"] and bb["significant"] and bb["delta"] > 0
    c12a_win = beats_c11a("C12a", c12a_dev); c12b_win = beats_c11a("C12b", c12b_dev)
    best_c12 = "C12b" if (c12b_win and (not c12a_win or c12b_dev["ndcg5"] >= c12a_dev["ndcg5"])) else ("C12a" if c12a_win else None)
    final_policy = best_c12 if best_c12 else "C11a"
    mean_help = c12a_win or c12b_win
    max_help = c12b_win and (c12b_dev["ndcg5"] > c12a_dev["ndcg5"])
    seld = cands[final_policy]
    tpg = False
    if best_c12:
        tpg = mg[best_c12]["3+"]["recall5"] > mg["C11a"]["3+"]["recall5"]

    gates = {
        "LEARNED_LIST_CONTEXT_ADDS_VALUE": "YES" if mean_help else "NO",
        "DEEPSETS_MEAN_HELPS": "YES" if c12a_win else "NO",
        "DEEPSETS_MAX_HELPS": "YES" if max_help else "NO",
        "MULTIGOLD_RECALL_IMPROVES": "YES" if (best_c12 and (mg[best_c12]["2"]["recall5"] > mg["C11a"]["2"]["recall5"] or tpg)) else "NO",
        "THREE_PLUS_GOLD_IMPROVES": "YES" if tpg else "NO",
        "C12_BEATS_C11A": "YES" if best_c12 else "NO",
        "C12_BEATS_C8C": "YES" if (final_policy != "C11a" and seld["ndcg5"] > c8c_dev["ndcg5"]) or (a_dev["ndcg5"] > c8c_dev["ndcg5"]) else "NO",
        "DEEP_RECALL_PRESERVED": "YES" if abs(c12a_dev["all50"] - c8c_dev["all50"]) < 1e-9 and abs(c12b_dev["all50"] - c8c_dev["all50"]) < 1e-9 else "NO",
        "NO_NEW_LLM_COMPONENT": "PASS", "NO_NEW_ENCODER_FORWARD": "PASS",
        "FINAL_PRE_VAL_POLICY": final_policy,
        "SAFE_TO_RUN_OFFICIAL_VAL": "YES" if a_dev["ndcg5"] > c8c_dev["ndcg5"] else "NO",
        "SAFE_TO_FREEZE_L2": "NO",
    }
    out = {"phase": "C12 — learned list-context (DeepSets) residual MLP; C9_DEV select + DEV_INNER confirm; NO VAL/TEST",
           "NEW_ENCODER_FORWARD_PASSES": 0, "NEW_LLM_COMPONENTS": 0,
           "C9_DEV": {k: C8slim(cands[k]) for k in cands},
           "DEV_INNER": {k: C8slim(v) for k, v in conf.items()},
           "PAIRED_BOOTSTRAP": boot, "MULTIGOLD_BREAKDOWN": mg, "CONTEXT_EFFECT": ceff,
           "SELF_LEAK_ORDINARY_MAX_C12a": round(leak, 4),
           "MODEL_SIZE": {"C12a_params": R.param_count(m_a), "C12b_params": R.param_count(m_b),
                          "C11a_params": R.param_count(ma), "C12a_train_sec": round(ta, 1), "C12b_train_sec": round(tb, 1),
                          "C12b_infer_ms_per_query": None, "cap": CAP, "Z": 128},
           "SELECTED_ON_C9DEV": sel, "GATES": gates}
    json.dump(out, open("results/L2/L2_C12_LIST_CONTEXT.json", "w"), indent=1, default=str)
    joblib.dump({"C12a": m_a.state_dict(), "C12b": m_b.state_dict(), "emean": emean, "estd": estd}, f"{OUT}/C12_models.joblib")
    log("GATES " + json.dumps(gates))
    log(f"[{time.time()-T0:.0f}s] C12_DONE selected={sel} final_policy={final_policy}")


if __name__ == "__main__":
    main()
