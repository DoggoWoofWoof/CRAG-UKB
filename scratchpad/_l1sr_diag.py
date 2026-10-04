"""STEP 1 -- rank-transfer diagnostic, and STEP 8 -- node->partition displacement.

No new algorithm.  For every MetaQA hop2/hop3 gold partition that SAFE misses, follow the same
partition through every stage of the existing pipeline and record where it is:

    structural NODE rank      position in s_node (the frozen directional expansion output)
    min hop                   s_hop of that node -- the only provenance the frozen substrate keeps
    S4 partition rank         position in order_struct(struct_aggregate_full, "S4")
    F6 boundary rank          position in the f6_select score order over the candidate set
    canonical partition rank  cpos, i.e. rank in base_rank
    selected                  yes/no

CLASSIFICATION uses the frozen operational depths, not gold-tuned cuts:
    M_struct = 64   the depth at which struct_aggregate_full stops reading s_node
    B        = 6    the number of swap slots f6_select fills

    A NODE_DISCOVERY_LIMITED   no gold node of the partition inside the first M_struct entries
    B AGGREGATION_LIMITED      a gold node is inside M_struct, but the S4 partition rank is >= B
    C SWAP_LIMITED             S4 partition rank < B, yet the boundary selection rejects it

  python scratchpad/_l1sr_diag.py [ds]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1ps_router as RT
import _l1kb_core as KB
import _l1cv_core as CV

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
SRD = f"{PP.ROOT}/L1_STRUCT_RECOVERY"
M_STRUCT = KB.BASE_CFG["M_struct"]
B = CV.B
NOPE = 10 ** 6


def gold_rows_of(ds, z):
    """reconstruct gold NODE rows exactly as _l1ps_cache.py did (the cache keeps only partitions)."""
    hard, _, _, _, _, id2row = TA.load_topology(ds, lambda *a: None)
    j = json.load(open(f"data/ukb_storage/{ds}/gte_qwen/query_ids_all.json"))
    golds = j["golds"]
    rows = np.asarray(z["rows"])
    out = [[id2row[x] for x in golds[int(r)] if x in id2row] for r in rows]
    return out, np.asarray(hard)


def pct(v):
    v = np.asarray([x for x in v if x < NOPE], np.float64)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
            "p90": float(np.percentile(v, 90))}


def main(ds="metaqa"):
    S = CV.substrate(ds)
    z = {k: S["z"][k] for k in S["z"].files} if hasattr(S["z"], "files") else S["z"]
    gr, hard = gold_rows_of(ds, z)
    goldp, ctxs, hops = S["goldp"], S["ctxs"], S["hops"]
    # parity: reconstructed gold nodes must reproduce the cached gold partitions exactly
    bad = sum(1 for qi in range(S["nq"])
              if {int(hard[g]) for g in gr[qi]} != set(goldp[qi]))
    log(f"gold-node reconstruction parity: {S['nq'] - bad}/{S['nq']} queries exact")
    assert bad == 0, "reconstructed gold nodes do not match cached gold partitions"

    recs, disp, disp_gold = [], [], []
    top8_in = {50: 0, 64: 0, 80: 0, 100: 0}
    top8_tot = 0
    for qi, c in enumerate(ctxs):
        X, sc = KB.f6_select(c["bnd"], list(c["chal"]), c["spos"], c["rpos"], c["cpos"], B)
        final = c["prot_set"] | set(X)
        frank = {p: i for i, (_, _, p) in enumerate(sc)}
        s4 = RT.order_struct(c["sagg"], "S4")
        s4r = {p: i for i, p in enumerate(s4)}
        sn = z["s_node"][qi]; sh = z["s_hop"][qi]
        nrank, nhop = {}, {}
        for jj in range(len(sn)):
            v = int(sn[jj])
            if v < 0:
                break
            p = int(hard[v])
            if p >= 0 and p not in nrank:
                nrank[p] = jj; nhop[p] = int(sh[jj])
            if jj < 8:                                     # STEP 8: displacement of the top-8
                d = s4r.get(p, NOPE)
                disp.append((jj, d))
        gset = set(goldp[qi])
        gnrank = {}
        for g in gr[qi]:
            p = int(hard[g])
            w = np.where(sn == g)[0]
            r = int(w[0]) if len(w) else NOPE
            if p not in gnrank or r < gnrank[p]:
                gnrank[p] = r
        for p, r in gnrank.items():
            if r < NOPE:
                disp_gold.append((r, s4r.get(p, NOPE)))
                if r < 8:
                    top8_tot += 1
                    for t in top8_in:
                        top8_in[t] += int(s4r.get(p, NOPE) < t)
        if hops is None or hops[qi] not in (2, 3):
            continue
        for p in sorted(gset - final):
            gn = gnrank.get(p, NOPE)
            an = nrank.get(p, NOPE)
            s4rk = s4r.get(p, NOPE)
            cls = ("A_NODE_DISCOVERY_LIMITED" if gn >= M_STRUCT else
                   "B_AGGREGATION_LIMITED" if s4rk >= B else "C_SWAP_LIMITED")
            recs.append({"qi": qi, "hop": int(hops[qi]), "part": int(p),
                         "gold_node_struct_rank": gn, "any_node_struct_rank": an,
                         "min_hop": nhop.get(p, -1), "s4_partition_rank": s4rk,
                         "f6_boundary_rank": frank.get(p, NOPE),
                         "canonical_partition_rank": c["cpos"].get(p, NOPE),
                         "selected": False, "class": cls})

    out = {"ds": ds, "M_struct": M_STRUCT, "B": B, "n_missed_gold_partitions": len(recs),
           "DIST": {}, "CLASS": {}, "STEP8": {}}
    for h in (2, 3):
        R = [r for r in recs if r["hop"] == h]
        out["DIST"][f"hop{h}"] = {
            k: pct([r[k] for r in R]) for k in
            ["gold_node_struct_rank", "any_node_struct_rank", "s4_partition_rank",
             "f6_boundary_rank", "canonical_partition_rank"]}
        out["DIST"][f"hop{h}"]["n_missed"] = len(R)
        cc = {}
        for r in R:
            cc[r["class"]] = cc.get(r["class"], 0) + 1
        out["CLASS"][f"hop{h}"] = {k: {"n": v, "frac": round(v / max(1, len(R)), 4)}
                                   for k, v in sorted(cc.items())}
    d = np.array([x[1] for x in disp_gold if x[1] < NOPE], np.float64)
    lost = sum(1 for x in disp_gold if x[1] >= NOPE)
    out["STEP8"] = {
        "n_gold_nodes_in_struct_list": len(disp_gold),
        "gold_node_to_S4_partition_rank": {
            "median": float(np.median(d)) if len(d) else None,
            "p90": float(np.percentile(d, 90)) if len(d) else None,
            "partition_absent_from_S4": lost},
        "top8_gold_nodes": {"n": top8_tot,
                            **{f"partition_in_S4_top{t}": {
                                "n": v, "frac": round(v / max(1, top8_tot), 4)}
                               for t, v in top8_in.items()}}}
    dd = {}
    for k, v in disp_gold:
        dd.setdefault(min(k, 63), []).append(v if v < NOPE else NOPE)
    out["STEP8"]["by_node_rank"] = {
        str(k): {"n": len(v),
                 "median_partition_rank": (float(np.median([x for x in v if x < NOPE]))
                                           if any(x < NOPE for x in v) else None)}
        for k in sorted(dd) if k < 16 for v in [dd[k]]}
    os.makedirs(f"{SRD}/diag", exist_ok=True)
    fp = f"{SRD}/diag/step1_{ds}.json"
    json.dump({"summary": out, "records": recs[:4000]}, open(fp, "w"), indent=1)

    for h in (2, 3):
        log(f"  hop{h}: {out['DIST'][f'hop{h}']['n_missed']} missed gold partitions   "
            + "  ".join(f"{k.split('_')[0]}={v['n']}({v['frac']:.1%})"
                        for k, v in out["CLASS"][f"hop{h}"].items()))
        for k, v in out["DIST"][f"hop{h}"].items():
            if isinstance(v, dict) and v.get("n"):
                log(f"      {k:26s} n={v['n']:5d} p25 {v['p25']:8.1f} med {v['median']:8.1f} "
                    f"p75 {v['p75']:8.1f} p90 {v['p90']:8.1f}")
    s8 = out["STEP8"]
    log(f"  STEP8 gold nodes in struct list={s8['n_gold_nodes_in_struct_list']}  "
        f"median partition rank {s8['gold_node_to_S4_partition_rank']['median']}  "
        f"p90 {s8['gold_node_to_S4_partition_rank']['p90']}  "
        f"absent {s8['gold_node_to_S4_partition_rank']['partition_absent_from_S4']}")
    log(f"  STEP8 top-8 gold nodes n={s8['top8_gold_nodes']['n']}  "
        + "  ".join(f"top{t}={s8['top8_gold_nodes'][f'partition_in_S4_top{t}']['frac']:.1%}"
                    for t in (50, 64, 80, 100)))
    log(f"wrote {fp}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
