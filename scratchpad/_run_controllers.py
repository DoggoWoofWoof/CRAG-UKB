"""Run the C0..C4 controlled progression, VAL only. Saves models, metrics, gate diagnostics, per-query
arrays (for bootstrap + failure analysis). Writes results/L2/_ctrl/<Cx>.json incrementally."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch
import l2_controller as CT

OUT = "results/L2/_ctrl"; os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)


def save_perq(tag, res):
    perq = res["_perq"]; d = {}
    for ds in CT.DS:
        for k, v in perq[ds].items():
            d[f"{ds}__{k}"] = v
    np.savez(f"{OUT}/_perq_{tag}.npz", **d)


def summarize(tag, res, meta=None):
    slim = {ds: {k: round(res[ds][k], 4) for k in res[ds] if k != "n"} | {"n": res[ds]["n"]} for ds in CT.DS}
    slim["POOLED"] = {k: round(res["POOLED"][k], 4) for k in res["POOLED"] if k != "n"} | {"n": res["POOLED"]["n"]}
    if "_gate" in res: slim["gate"] = res["_gate"]
    if meta: slim["train_meta"] = {k: meta[k] for k in ("kind", "epochs", "lr", "lam", "best_sel", "pos_collisions")}
    if meta: slim["history"] = meta["history"]
    json.dump(slim, open(f"{OUT}/{tag}.json", "w"), indent=1, default=str)
    p = res["POOLED"]
    log(f"== {tag}: NDCG50={p['NDCG@50']:.4f} MRR={p['MRR']:.4f} R@5={p['R@5']:.4f} R@10={p['R@10']:.4f} ALL@10={p['ALL@10']:.4f} ALL@50={p['ALL@50']:.4f}")
    return slim


def main():
    va = {ds: CT.load_cache(ds, "val") for ds in CT.DS}
    T0 = time.time()

    # ---- C0: fixed equal RRF over 5 experts (untrained GlobalW = equal weights) ----
    c0 = CT.GlobalW()
    r0 = CT.evaluate(c0, "C1", va, want_gate=True, collect=True); save_perq("C0", r0); summarize("C0", r0)

    # ---- C0_norel: best previous non-Relation stack (equal RRF over 4 experts, relation weight~0) ----
    cnr = CT.GlobalW()
    with torch.no_grad(): cnr.z.copy_(torch.tensor([0., 0., 0., 0., -30.]))
    rnr = CT.evaluate(cnr, "C1", va, want_gate=True); summarize("C0_norel", rnr)

    # ---- C1: learned global weights ----
    m1, meta1 = CT.train("C1", epochs=8, lr=3e-4, log=log)
    r1 = CT.evaluate(m1, "C1", va, want_gate=True, collect=True); save_perq("C1", r1)
    torch.save(m1.state_dict(), f"{OUT}/C1.pt"); summarize("C1", r1, meta1)
    with torch.no_grad():
        w = torch.nn.functional.softplus(m1.z); w = (w / w.sum()).tolist()
    log("C1 global weights " + json.dumps({e: round(x, 4) for e, x in zip(CT.EXPERTS, w)}))

    # ---- C2: query-only gate ----
    m2, meta2 = CT.train("C2", epochs=10, lr=3e-4, log=log)
    r2 = CT.evaluate(m2, "C2", va, want_gate=True, collect=True); save_perq("C2", r2)
    torch.save(m2.state_dict(), f"{OUT}/C2.pt"); summarize("C2", r2, meta2)

    # ---- C3: query+candidate gate ----
    m3, meta3 = CT.train("C3", epochs=10, lr=3e-4, log=log)
    r3 = CT.evaluate(m3, "C3", va, want_gate=True, collect=True); save_perq("C3", r3)
    torch.save(m3.state_dict(), f"{OUT}/C3.pt"); summarize("C3", r3, meta3)

    # ---- C4: C3 arch + positive-KL Shapley auxiliary (lam=0.05) ----
    m4, meta4 = CT.train("C4", epochs=10, lr=3e-4, lam=0.05, log=log)
    r4 = CT.evaluate(m4, "C4", va, want_gate=True, collect=True); save_perq("C4", r4)
    torch.save(m4.state_dict(), f"{OUT}/C4.pt"); summarize("C4", r4, meta4)

    log(f"ALL_CONTROLLERS_DONE total={time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
