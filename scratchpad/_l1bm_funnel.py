"""The transition table: where in the pipeline a required gold node is actually lost.

Candidate survival at one hop is not the deliverable.  What matters is the funnel, stage by stage,
for every REQUIRED gold node:

    REACHABLE            inside the unbounded 3-hop legal closure of the seeds at all
    ALIVE after p1       some position-1 survivor is still within 2 legal hops of it
    ALIVE after p2       some position-2 survivor is still within 1 legal hop of it
    ALIVE after p3       it is in the beam scope -- the structural search actually reached it
    NODE in added(256)   it survived the final static-score cut
    NODE read by S4(64)  it is inside the depth S4 actually aggregates
    PARTITION in S4      its partition appears in the structural partition ranking
    PARTITION in P50     its partition is in the final exactly-50 output

"alive after pK" is WON OR STILL WINNABLE: the gold node has already been found by position k, or
some survivor of that prune can still reach it within the hops that remain.  Both halves are needed --
a node found early is excluded from every later candidate set, so a purely forward-looking test is not
monotone and would report a funnel that rises.  Reachability, not membership of some pre-chosen
shortest path, is what attributes loss to a specific compression point.

The graph is undirected (verified), so the distance-k ball around a gold node is computed by forward
BFS on the legal induced subgraph (both endpoints deg <= DEG_CAP -- exactly the beam's own rule).

The last two rows are NOT nested inside the structural ones: a partition can also be covered by the
canonical and retrieval channels.  They are the end-to-end outcome, not a structural stage.

  python scratchpad/_l1bm_funnel.py ds [pol,pol,...]
"""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1ps_router as RT
import _l1sr_eval as EV
import _l1sr_diag as DG
import _l1bm_core as BM
import _l1bm_run as RUN

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
CAP = TA.DEG_CAP
ROWS = ["REACHABLE", "ALIVE_p1", "ALIVE_p2", "ALIVE_p3", "NODE_added256", "NODE_read_S4",
        "PART_in_S4", "PART_in_P50"]


def ball(seed_nodes, k, adjp, adji, deg):
    """all nodes within k legal hops of `seed_nodes` (legal = both endpoints deg <= DEG_CAP)."""
    cur = np.unique(np.asarray([int(x) for x in seed_nodes], np.int64))
    cur = cur[deg[cur] <= CAP]
    seen = set(int(x) for x in cur)
    for _ in range(k):
        if not len(cur):
            break
        _, ch, _ = BM._children(cur, adjp, adji, deg, CAP)
        if not len(ch):
            break
        ch = np.unique(ch)
        new = np.fromiter((int(v) for v in ch if int(v) not in seen), np.int64)
        if not len(new):
            break
        seen.update(int(x) for x in new)
        cur = new
    return seen


def geometry(ds, S, adjp, adji, deg, Gs, cache=True):
    """per query: the 3-hop seed closure and, per gold node, its distance-1 and distance-2 balls."""
    fp = f"{BM.BMD}/cache/{ds}__geom.pkl"
    if cache and os.path.exists(fp):
        with open(fp, "rb") as fh:
            return pickle.load(fh)
    z = S["z"]; nq = S["nq"]
    OUT = []
    t = time.time()
    for qi in range(nq):
        sd = [int(s) for s in z["seeds"][qi] if s >= 0]
        clo = ball(sd, TA.MAX_HOPS, adjp, adji, deg)
        B1, B2 = {}, {}
        for g in Gs[qi]:
            B1[g] = ball([g], 1, adjp, adji, deg)
            B2[g] = ball([g], 2, adjp, adji, deg)
        OUT.append({"closure": clo, "B1": B1, "B2": B2})
        if (qi + 1) % 250 == 0:
            log(f"   geom {ds} {qi+1}/{nq}  {time.time()-t:.0f}s")
    if cache:
        with open(fp, "wb") as fh:
            pickle.dump(OUT, fh, protocol=5)
    return OUT


def main(ds="metaqa", policies=None):
    policies = policies or BM.POLICIES
    S = EV.substrate(ds); z = S["z"]; nq = S["nq"]; hard = S["hard"]; hops = S["hops"]
    gr, _ = DG.gold_rows_of(ds, z)
    Gs = [set(int(x) for x in g) for g in gr]
    adjp, adji, deg = RUN.topology(ds)
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    SD = [[int(s) for s in z["seeds"][qi] if s >= 0] for qi in range(nq)]
    RQ = [TA.residual(Qm[qi].astype(np.float64), SD[qi], Xn) for qi in range(nq)]
    GEO = geometry(ds, S, adjp, adji, deg, Gs)
    log("geometry ready")

    SAFE = [RT.order_struct(c["sagg"], "S4") for c in S["ctxs"]]
    i_safe, _ = EV.evaluate(S, SAFE, "A_F6", (EV.P,))
    hs = sorted(set(int(x) for x in hops)) if hops is not None else []
    blocks = [("ALL", np.ones(nq, bool))]
    if hs and min(hs) >= 0:
        blocks += [(f"hop{h}", np.asarray(hops) == h) for h in hs]

    OUT = {"ds": ds, "nq": nq, "rows": ROWS, "FUNNEL": {}, "SUMMARY": {}}
    for pol in policies:
        node_ok = {r: np.zeros(nq, np.int32) for r in ROWS}
        qry_ok = {r: np.zeros(nq, np.int8) for r in ROWS}
        ngold = np.zeros(nq, np.int32)
        t = time.time()
        SFs = []
        for qi in range(nq):
            a, vm, st = BM.expand_beam(SD[qi], RQ[qi], adjp, adji, deg, Xn, BM.BEAM, pol,
                                       keep_scope=True, keep_hops=True)
            kept = st.get("kept_nodes", [])
            K = [set(int(x) for x in kept[h]) if h < len(kept) else set() for h in range(3)]
            scope = st["scope_set"]
            # a gold node already FOUND at an earlier position is excluded from later candidate
            # sets, so "still winnable" alone is not monotone -- it must be "won or still winnable"
            found0 = set(SD[qi]) | K[0]
            found1 = found0 | K[1]
            A256 = set(int(v) for v in a); A64 = set(int(v) for v in a[:EV.M_STRUCT])
            g0 = GEO[qi]
            sn, sh, sd_, sc = BM.arrays_of(a, vm)
            SF = EV.sf_from_cache({"s_node": sn[None], "s_hop": sh[None], "s_sdir": sd_[None],
                                   "s_cnt": sc[None]}, 0, hard, "P0_S4")
            SFs.append(SF)
            sfset = set(int(p) for p in SF)
            G = Gs[qi]; ngold[qi] = len(G)
            hit = {r: 0 for r in ROWS}
            for g in G:
                b1, b2 = g0["B1"].get(g, set()), g0["B2"].get(g, set())
                r = {}
                r["REACHABLE"] = g in g0["closure"]
                r["ALIVE_p1"] = (g in found0) or bool(K[0] & b2)
                r["ALIVE_p2"] = (g in found1) or bool(K[1] & b1)
                r["ALIVE_p3"] = g in scope
                r["NODE_added256"] = g in A256
                r["NODE_read_S4"] = g in A64
                r["PART_in_S4"] = int(hard[g]) in sfset
                r["PART_in_P50"] = False        # filled per query below (needs the swap)
                for k, v in r.items():
                    hit[k] += int(v)
            for k in ROWS:
                node_ok[k][qi] = hit[k]
                qry_ok[k][qi] = int(hit[k] == len(G) and len(G) > 0)
            if (qi + 1) % 500 == 0:
                log(f"   {pol} {ds} {qi+1}/{nq}  {time.time()-t:.0f}s")
        lat = round(1000.0 * (time.time() - t) / nq, 2)
        ind, _ = EV.evaluate(S, SFs, "A_F6", (EV.P,))
        node_ok["PART_in_P50"] = ind[EV.P].astype(np.int32) * ngold   # all-or-nothing by definition
        qry_ok["PART_in_P50"] = ind[EV.P].astype(np.int8)
        m = PP.mcnemar(ind[EV.P], i_safe[EV.P])
        F = {}
        for nm, mk in blocks:
            F[nm] = {"queries": int(mk.sum()), "gold_nodes": int(ngold[mk].sum())}
            for r in ROWS:
                F[nm][r] = {"nodes": int(node_ok[r][mk].sum()),
                            "node_frac": round(float(node_ok[r][mk].sum() /
                                                     max(1, ngold[mk].sum())), 4),
                            "queries_all": int(qry_ok[r][mk].sum()),
                            "query_frac": round(float(qry_ok[r][mk].mean()), 4)}
        OUT["FUNNEL"][pol] = F
        OUT["SUMMARY"][pol] = {"P50": EV.by_hop(ind[EV.P], hops), "latency_ms_per_q": lat,
                               **{f"vs_SAFE_{k}": v for k, v in m.items()}}
        s = OUT["SUMMARY"][pol]
        log(f"  {pol:26s} p2 {F['ALL']['ALIVE_p2']['node_frac']:.4f} "
            f"p3 {F['ALL']['ALIVE_p3']['node_frac']:.4f} "
            f"S4 {F['ALL']['NODE_read_S4']['node_frac']:.4f}  "
            f"ALL {s['P50']['ALL']:.4f} net {m['net']:+d} {lat:.0f} ms/q")
        json.dump(OUT, open(f"{BM.BMD}/diag/funnel_{ds}.json", "w"), indent=1)
    log(f"wrote {BM.BMD}/diag/funnel_{ds}.json")
    return OUT


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "metaqa",
         sys.argv[2].split(",") if len(sys.argv) > 2 else None)
