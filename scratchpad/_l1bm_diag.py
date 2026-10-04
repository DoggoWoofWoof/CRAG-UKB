"""STEP 7 (binding-hop diagnostic) + STEP 8 arrays + STEP 9 internal-work, in ONE pass per policy.

The diagnostic is instrumentation only -- gold labels are read by the MEASUREMENT, never by any
policy.  Every policy sees exactly the same inputs as the frozen beam.

At path position 2 (hop index 1, the binding prune) we record, per query:

    n_frontier          frontier nodes entering the prune
    n_cand              distinct fresh candidates produced by them
    gold_cand           candidates that ARE gold nodes
    gold_parent         candidates with >= 1 legal child that is a gold node not already in scope
                        (exactly what a one-step lookahead could in principle detect)
    *_pct               percentile position of those candidates under the policy ordering, 0 = best,
                        computed on the FULL pre-prune candidate list
    *_surv              how many of them survive into the beam

and after the whole expansion: gold nodes present in the final `added` list, split by the hop at
which they were first reached.  Position-2 survival is the primary number; final coverage is STEP 8.

  python scratchpad/_l1bm_diag.py ds pol,pol,...
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1sr_eval as EV
import _l1sr_seq as SQ
import _l1sr_diag as DG
import _l1bm_core as BM
import _l1bm_run as RUN

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
FIELDS = ["n_frontier", "n_cand", "gold_cand", "gold_cand_surv", "gold_parent",
          "gold_parent_surv", "gold_added", "gold_added_h1", "gold_added_h2", "gold_added_h3",
          "n_gold"]


def one_query(seeds, r_q, adjp, adji, deg, Xn, G, policy):
    a, vm, st, tr = BM.expand_beam(seeds, r_q, adjp, adji, deg, Xn, BM.BEAM, policy, trace_hop=BM.BIND_HOP)
    row = {k: 0 for k in FIELDS}
    row["n_gold"] = len(G)
    pc_c, pc_p = [], []
    if tr is not None and len(tr["cand"]):
        cand, ordr, M = tr["cand"], tr["order"], tr["M"]
        n = len(cand)
        pos = np.empty(n, np.int64); pos[ordr] = np.arange(n)      # rank of each candidate
        kept = set(int(v) for v in tr["kept"])
        isg = np.fromiter((int(v) in G for v in cand), bool, n)
        # a candidate is a "gold parent" if it can still deliver a gold node one hop later
        own, ch, _ = BM._children(cand, adjp, adji, deg)
        gp = np.zeros(n, bool)
        if len(ch):
            hit = np.fromiter((int(c) in G for c in ch), bool, len(ch))
            fresh = ~np.isin(ch, tr["scope"])
            u = np.unique(own[hit & fresh])
            gp[u] = True
        gp &= ~isg
        den = max(1, n - 1)
        row["n_frontier"] = tr["n_frontier"] if "n_frontier" in tr else len(tr["frontier"])
        row["n_cand"] = n
        row["gold_cand"] = int(isg.sum())
        row["gold_parent"] = int(gp.sum())
        row["gold_cand_surv"] = int(sum(1 for v in cand[isg] if int(v) in kept))
        row["gold_parent_surv"] = int(sum(1 for v in cand[gp] if int(v) in kept))
        pc_c = [float(pos[i]) / den for i in np.where(isg)[0]]
        pc_p = [float(pos[i]) / den for i in np.where(gp)[0]]
    inA = [v for v in a if int(v) in G]
    row["gold_added"] = len(inA)
    for v in inA:
        h = int(vm[int(v)][0])
        if 1 <= h <= 3:
            row[f"gold_added_h{h}"] += 1
    return a, vm, st, row, pc_c, pc_p


def run(ds, policies, cache=True, log=log):
    S = EV.substrate(ds); z = S["z"]; nq = S["nq"]
    gr, _ = DG.gold_rows_of(ds, z)
    Gs = [set(int(x) for x in g) for g in gr]
    adjp, adji, deg = RUN.topology(ds)
    Xn, Qm, big = RUN.load_emb_frozen(ds, z)
    log(f"[emb] {ds} fp16_path={big}")
    RQ = [TA.residual(Qm[qi].astype(np.float64), [int(s) for s in z["seeds"][qi] if s >= 0], Xn)
          for qi in range(nq)]
    SD = [[int(s) for s in z["seeds"][qi] if s >= 0] for qi in range(nq)]
    OUT = {}
    for pol in policies:
        fp = f"{BM.BMD}/cache/{ds}__{pol}.npz"
        if cache and os.path.exists(fp):
            zz = np.load(fp, allow_pickle=True)
            OUT[pol] = {k: zz[k] for k in ("s_node", "s_hop", "s_sdir", "s_cnt")}
            OUT[pol]["stats"] = json.loads(str(zz["stats"]))
            OUT[pol]["rows"] = zz["rows"]
            OUT[pol]["pct_cand"] = zz["pct_cand"]; OUT[pol]["pct_parent"] = zz["pct_parent"]
            OUT[pol]["pc_ptr"] = zz["pc_ptr"]; OUT[pol]["pp_ptr"] = zz["pp_ptr"]
            log(f"[cached] {ds} {pol}")
            continue
        sn = np.full((nq, BM.SMAX), -1, np.int32); sh = np.zeros((nq, BM.SMAX), np.int8)
        sd = np.zeros((nq, BM.SMAX), np.float32); sc = np.zeros((nq, BM.SMAX), np.int32)
        rows = np.zeros((nq, len(FIELDS)), np.int32)
        PC, PP_ = [], []
        agg = {"edges": 0, "look_edges": 0, "cand_total": 0, "scope": 0,
               "frontier": np.zeros(TA.MAX_HOPS), "cands": np.zeros(TA.MAX_HOPS),
               "kept": np.zeros(TA.MAX_HOPS), "seen": np.zeros(TA.MAX_HOPS)}
        t = time.time()
        for qi in range(nq):
            a, vm, st, row, pc, pp = one_query(SD[qi], RQ[qi], adjp, adji, deg, Xn, Gs[qi], pol)
            sn[qi], sh[qi], sd[qi], sc[qi] = BM.arrays_of(a, vm)
            rows[qi] = [row[k] for k in FIELDS]
            PC.append(np.asarray(pc, np.float32)); PP_.append(np.asarray(pp, np.float32))
            for k in ("edges", "look_edges", "cand_total", "scope"):
                agg[k] += st[k]
            for h in range(len(st["cands"])):
                agg["frontier"][h] += st["frontier"][h]; agg["cands"][h] += st["cands"][h]
                agg["kept"][h] += st["kept"][h]; agg["seen"][h] += 1
            if (qi + 1) % 500 == 0:
                log(f"   {pol} {ds} {qi+1}/{nq}  {time.time()-t:.0f}s")
        agg["latency_ms"] = round(1000.0 * (time.time() - t) / nq, 3)
        for k in ("frontier", "cands", "kept"):
            agg[k] = [round(float(agg[k][h] / max(1, agg["seen"][h])), 1) for h in range(TA.MAX_HOPS)]
        agg["seen"] = [int(x) for x in agg["seen"]]
        for k in ("edges", "look_edges", "cand_total", "scope"):
            agg[k + "_per_q"] = round(agg[k] / nq, 1)
        pc_flat = np.concatenate(PC) if PC else np.zeros(0, np.float32)
        pp_flat = np.concatenate(PP_) if PP_ else np.zeros(0, np.float32)
        pc_ptr = np.cumsum([0] + [len(x) for x in PC]).astype(np.int32)
        pp_ptr = np.cumsum([0] + [len(x) for x in PP_]).astype(np.int32)
        OUT[pol] = {"s_node": sn, "s_hop": sh, "s_sdir": sd, "s_cnt": sc, "stats": agg,
                    "rows": rows, "pct_cand": pc_flat, "pct_parent": pp_flat,
                    "pc_ptr": pc_ptr, "pp_ptr": pp_ptr}
        if cache:
            np.savez_compressed(fp, stats=json.dumps(agg), rows=rows, pct_cand=pc_flat,
                                pct_parent=pp_flat, pc_ptr=pc_ptr, pp_ptr=pp_ptr,
                                s_node=sn, s_hop=sh, s_sdir=sd, s_cnt=sc)
        st = agg
        log(f"[built] {ds} {pol}  edges/q {st['edges_per_q']:.0f} (+look {st['look_edges_per_q']:.0f})"
            f"  cand/hop {st['cands']}  kept/hop {st['kept']}  {st['latency_ms']:.2f} ms/q")
    return OUT, S, Gs


def summarise(OUT, S, policies, ds):
    """position-2 survival + reach, overall and per metaqa hop block."""
    nq = S["nq"]; hops = S["hops"]
    hs = sorted(set(int(x) for x in hops)) if hops is not None else []
    blocks = [("ALL", np.ones(nq, bool))]
    if hs and min(hs) >= 0:
        blocks += [(f"hop{h}", np.asarray(hops) == h) for h in hs]
    F = {k: i for i, k in enumerate(FIELDS)}
    T = {}
    for pol in policies:
        R = OUT[pol]["rows"]; pc = OUT[pol]["pct_cand"]
        pcp = OUT[pol].get("pc_ptr"); ppp = OUT[pol].get("pp_ptr")
        pp = OUT[pol]["pct_parent"]
        T[pol] = {}
        for nm, m in blocks:
            r = R[m]
            sub = {"queries": int(m.sum()),
                   "n_frontier_p2": round(float(r[:, F["n_frontier"]].mean()), 1),
                   "n_cand_p2": round(float(r[:, F["n_cand"]].mean()), 1),
                   "gold_cand_p2": int(r[:, F["gold_cand"]].sum()),
                   "gold_cand_surv_p2": int(r[:, F["gold_cand_surv"]].sum()),
                   "gold_parent_p2": int(r[:, F["gold_parent"]].sum()),
                   "gold_parent_surv_p2": int(r[:, F["gold_parent_surv"]].sum()),
                   "gold_added": int(r[:, F["gold_added"]].sum()),
                   "gold_added_h3": int(r[:, F["gold_added_h3"]].sum()),
                   "n_gold": int(r[:, F["n_gold"]].sum())}
            sub["gold_cand_survival"] = round(sub["gold_cand_surv_p2"] /
                                              max(1, sub["gold_cand_p2"]), 4)
            sub["gold_parent_survival"] = round(sub["gold_parent_surv_p2"] /
                                                max(1, sub["gold_parent_p2"]), 4)
            sub["gold_reach_frac"] = round(sub["gold_added"] / max(1, sub["n_gold"]), 4)
            if pcp is not None:
                idx = np.where(m)[0]
                v = np.concatenate([pc[pcp[i]:pcp[i + 1]] for i in idx]) if len(idx) else pc[:0]
                w = np.concatenate([pp[ppp[i]:ppp[i + 1]] for i in idx]) if len(idx) else pp[:0]
            else:
                v, w = pc, pp
            for key, arr in (("gold_cand_pct", v), ("gold_parent_pct", w)):
                sub[key] = (round(float(np.median(arr)), 4) if len(arr) else None)
                sub[key + "_mean"] = (round(float(arr.mean()), 4) if len(arr) else None)
                sub[key + "_n"] = int(len(arr))
            T[pol][nm] = sub
    return T


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
    pols = sys.argv[2].split(",") if len(sys.argv) > 2 else BM.POLICIES
    OUT, S, _ = run(ds, pols)
    T = summarise(OUT, S, pols, ds)
    json.dump({"ds": ds, "policies": pols, "STEP7": T,
               "STEP9": {p: OUT[p]["stats"] for p in pols}},
              open(f"{BM.BMD}/diag/step7_{ds}.json", "w"), indent=1)
    for p in pols:
        a = T[p]["ALL"]
        log(f"{p:26s} p2 cand {a['n_cand_p2']:7.1f}  goldcand surv {a['gold_cand_survival']:.3f} "
            f"({a['gold_cand_surv_p2']}/{a['gold_cand_p2']})  goldparent surv "
            f"{a['gold_parent_survival']:.3f} ({a['gold_parent_surv_p2']}/{a['gold_parent_p2']})  "
            f"reach {a['gold_reach_frac']:.4f}  pct {a['gold_cand_pct']}")
