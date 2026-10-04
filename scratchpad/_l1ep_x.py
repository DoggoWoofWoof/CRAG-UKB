"""FINAL L1 EDGE-SUBSTRATE + PARTITION UTILITY PROGRAM -- C3: the JOINT cell of the 2x2.

Phase A varied the edge substrate at the SHIPPED partitioning.  Phase C varied the partitioning at
the FROZEN edge substrate.  Neither produces the fourth cell, so INTERACTION was left PENDING.

Phase A licensed an edge arm (E6_TOPOLOGY_C never significantly regresses and significantly gains
on 2wiki), so the 2x2 is run rather than invented:

                          |  shipped partitioning  |  candidate partitioning
    ----------------------+------------------------+-------------------------
    E0_STRUCT   (frozen)  |          a             |          b
    E6_TOPOLOGY_C         |          c             |          d

    a  Phase C  PM_CURRENT_EXACT          b  Phase C  <candidate>
    c  Phase A  E6_TOPOLOGY_C             d  THIS SCRIPT
    EDGE_EFFECT = c-a   PARTITION_EFFECT = b-a   INTERACTION = d - b - c + a

Composition, nothing new invented: the partition-dependent arrays come from `_l1ep_c.rebuild`
(mem_idx, base_rank, gold labels, part_sizes) and the traversal comes from `_l1ep_sub.substrate`,
exactly as the A2 pass runs it.  The selector is untouched: P = 50, B = 6, M_struct = 64,
M_ret = 32, S4, F6.  Nothing is retuned and no gold touches either build.

PARITY: run with tag PM_CURRENT_EXACT and substrate E0_STRUCT this must reproduce the frozen
scoreboard exactly; that is asserted as cell `a` before any other cell is scored.

  python scratchpad/_l1ep_x.py <ds> <partition_tag> [substrate ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1ps_router as RT
import _l1kb_core as KB
import _l1ss_core as SS
import _l1bm_core as BM
import _l1bc_ledger as LG
import _l1kn_run as KR
import _l1ep_sub as EP
import _l1ep_pu as PU
import _l1ep_c as EC

OUT, ROOT, CACHE = EP.OUT, EC.ROOT, "scratchpad/_l1ep"
CFG = dict(KB.BASE_CFG)
P = 50
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def cell(ds, ptag, subs, stride=1):
    """F6 ALL@P50 for every (substrate x this partitioning) cell, one traversal pass per query."""
    z0 = np.load(f"{ROOT}/runs/cache_{ds}.npz", allow_pickle=True)
    meta = json.loads(str(z0["meta_json"]))
    z = {k: z0[k] for k in z0.files}

    if ptag == "PM_CURRENT_EXACT":
        hard, npart = PU.load_assignment(ds, "CURRENT")
    else:
        hard = np.load(f"{CACHE}/parts/{ds}__{ptag}.npy")
        npart = int(hard.max()) + 1
    hard = np.asarray(hard, np.int64)

    z.update(EC.rebuild(ds, hard, npart, z, meta, log))
    C = RT.build_cache(ds, z, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
    nq = meta["n_dev_queries"]
    goldp = [set(int(x) for x in z["gold_part"][z["gold_ptr"][qi]:z["gold_ptr"][qi + 1]])
             for qi in range(nq)]
    ind_base = np.array([int(goldp[qi] <= C["base50"][qi]) for qi in range(nq)], np.int8)
    ctxs = KB.contexts(z, meta, C, CFG["B"], CFG)

    import _l1bm_run as RUN
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    ndocs = (Xn.A.shape[0] if hasattr(Xn, "A") else Xn.shape[0])
    W = SS.workspace(ndocs)
    TOPO = {t: EP.substrate(ds, t, log)[:3] for t in subs}
    hops = np.asarray(z["hops"]) if "hops" in z else None

    qs = list(range(0, nq, stride))
    out = {t: np.zeros(len(qs), np.int8) for t in subs}
    ms = {t: 0.0 for t in subs}
    for j, qi in enumerate(qs):
        c = ctxs[qi]
        sd0 = [int(s) for s in z["seeds"][qi] if s >= 0]
        rq = TA.residual(Qm[qi].astype(np.float64), sd0, Xn)
        REQ = set(int(p) for p in goldp[qi])
        RF = [int(p) for p in C["rfull"][CFG["M_ret"]][qi]]
        b50, bnd, cpos, rpos = c["base50"], c["bnd"], c["cpos"], c["rpos"]
        for t in subs:
            ap, ai, dg = TOPO[t]
            t0 = time.perf_counter()
            added, vmeta, st, FE = SS.expand_feat(sd0, rq, ap, ai, dg, Xn, W,
                                                  want_future=False, want_visited=False)
            ms[t] += time.perf_counter() - t0
            sn, sh, sdd, scn = BM.arrays_of(added, vmeta)
            SF = KR._sf(sn, sh, sdd, scn, hard)
            spos = {p: r for r, p in enumerate(SF)}
            chal = [p for p in SF if p not in b50] + [p for p in RF if p not in b50]
            pick, _ = KR._f6(bnd, chal, spos, rpos, cpos)
            fs = c["prot_set"] | set(pick)
            assert len(fs) == P
            out[t][j] = int(REQ <= fs)

    R = {"ds": ds, "partition_tag": ptag, "npart": int(npart), "nq": len(qs),
         "stride": stride, "BASE_ALL_P50": round(float(ind_base[qs].mean()), 4), "CELLS": {}}
    for t in subs:
        R["CELLS"][t] = {"F6_ALL_P50": round(float(out[t].mean()), 4),
                         "traversal_ms_per_query": round(1000 * ms[t] / max(len(qs), 1), 2),
                         "_ind_F6": out[t].tolist()}
        if ds == "metaqa" and hops is not None:
            h = np.asarray(hops)[qs]
            R["CELLS"][t]["by_hop"] = {f"hop{k}": round(float(out[t][h == k].mean()), 4)
                                       for k in (1, 2, 3) if (h == k).any()}
    return R


if __name__ == "__main__":
    ds, ptag = sys.argv[1], sys.argv[2]
    subs = sys.argv[3:] or ["E0_STRUCT", "E6_TOPOLOGY_C"]
    os.makedirs(f"{OUT}/INTERACTION", exist_ok=True)
    fp = f"{OUT}/INTERACTION/X_{ds}.json"
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    R = cell(ds, ptag, subs)

    if ptag == "PM_CURRENT_EXACT":
        ref = json.load(open(f"{OUT}/INTERACTION/C_{ds}.json"))["PM_CURRENT_EXACT"]["F6_ALL_P50"]
        got = R["CELLS"].get("E0_STRUCT", {}).get("F6_ALL_P50")
        R["C3_PARITY"] = ("EXACT" if got is not None and abs(got - ref) < 1e-9
                          else f"MISMATCH ref={ref} got={got}")
        log(f"  C3_PARITY = {R['C3_PARITY']}")
    rec[ptag] = R
    json.dump(rec, open(fp, "w"), indent=1)
    for t, v in R["CELLS"].items():
        log(f"  {ds:16s} {ptag:22s} {t:16s} F6_ALL {v['F6_ALL_P50']:.4f}  "
            f"{v['traversal_ms_per_query']:.1f} ms/q"
            + (f"  by_hop {v['by_hop']}" if "by_hop" in v else ""))
    log("wrote", fp)
