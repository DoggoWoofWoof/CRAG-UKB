"""STEP 4 -- delayed pruning, oracle-free, with the loss decomposed into its three stages.

Delaying the prune can only help if the gold node survives ALL THREE cuts that follow it:

    DISCOVERY   the node is visited at all                      -> gold in `scope`
    ADDED       it survives the final static-score cut to 256    -> gold in `added`
    READ        S4 only reads the first M_struct = 64 of those   -> gold in `added[:64]`

Reporting only the last number would credit or blame the wrong stage, so all three are measured for
every policy here.  No gold label is visible to any policy; it is read by the measurement only.

  python scratchpad/_l1bm_step4.py ds [pol,pol,...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1sr_eval as EV
import _l1sr_diag as DG
import _l1bm_core as BM
import _l1bm_run as RUN

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
M_STRUCT = EV.M_STRUCT


def main(ds="metaqa", policies=("M0_BASELINE", "D0_DELAYED_PRUNE", "L1_MAX_FUTURE")):
    S = EV.substrate(ds); z = S["z"]; nq = S["nq"]; hops = S["hops"]
    gr, _ = DG.gold_rows_of(ds, z)
    Gs = [set(int(x) for x in g) for g in gr]
    adjp, adji, deg = RUN.topology(ds)
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    SD = [[int(s) for s in z["seeds"][qi] if s >= 0] for qi in range(nq)]
    RQ = [TA.residual(Qm[qi].astype(np.float64), SD[qi], Xn) for qi in range(nq)]
    hs = sorted(set(int(x) for x in hops)) if hops is not None else []
    blocks = [("ALL", np.ones(nq, bool))]
    if hs and min(hs) >= 0:
        blocks += [(f"hop{h}", np.asarray(hops) == h) for h in hs]
    OUT = {"ds": ds, "M_struct": M_STRUCT, "M_MAX": TA.M_MAX, "STEP4": {}}
    for pol in policies:
        disc = np.zeros(nq, np.int32); adds = np.zeros(nq, np.int32)
        read = np.zeros(nq, np.int32); ng = np.zeros(nq, np.int32)
        vis = np.zeros(nq, np.int64); edg = np.zeros(nq, np.int64)
        t = time.time()
        for qi in range(nq):
            a, vm, st = BM.expand_beam(SD[qi], RQ[qi], adjp, adji, deg, Xn, BM.BEAM, pol,
                                       keep_scope=True)
            G = Gs[qi]
            disc[qi] = sum(1 for v in st["scope_set"] if v in G)
            adds[qi] = sum(1 for v in a if int(v) in G)
            read[qi] = sum(1 for v in a[:M_STRUCT] if int(v) in G)
            ng[qi] = len(G); vis[qi] = st["scope"]; edg[qi] = st["edges"] + st["look_edges"]
            if (qi + 1) % 500 == 0:
                log(f"   {pol} {qi+1}/{nq} {time.time()-t:.0f}s")
        lat = round(1000.0 * (time.time() - t) / nq, 2)
        OUT["STEP4"][pol] = {"latency_ms_per_q": lat,
                             "unique_nodes_evaluated_per_q": round(float(vis.mean()), 1),
                             "graph_edges_inspected_per_q": round(float(edg.mean()), 1)}
        for nm, m in blocks:
            OUT["STEP4"][pol][nm] = {
                "n_gold": int(ng[m].sum()),
                "gold_DISCOVERED": int(disc[m].sum()), "gold_ADDED": int(adds[m].sum()),
                "gold_READ_by_S4": int(read[m].sum()),
                "frac_discovered": round(float(disc[m].sum() / max(1, ng[m].sum())), 4),
                "frac_added": round(float(adds[m].sum() / max(1, ng[m].sum())), 4),
                "frac_read": round(float(read[m].sum() / max(1, ng[m].sum())), 4)}
        a = OUT["STEP4"][pol]["ALL"]
        log(f"  {pol:24s} visited {OUT['STEP4'][pol]['unique_nodes_evaluated_per_q']:8.1f}  "
            f"gold discovered {a['frac_discovered']:.4f} -> added {a['frac_added']:.4f} -> "
            f"read {a['frac_read']:.4f}   {lat:.1f} ms/q")
    fp = f"{BM.BMD}/diag/step4_{ds}.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")
    return OUT


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "metaqa",
         sys.argv[2].split(",") if len(sys.argv) > 2 else
         ("M0_BASELINE", "D0_DELAYED_PRUNE", "L1_MAX_FUTURE"))
