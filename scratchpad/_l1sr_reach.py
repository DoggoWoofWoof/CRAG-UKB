"""STEP 1b -- split NODE_DISCOVERY_LIMITED into unreachable vs beam-pruned, and STEP 8 corrected.

STEP 1 says ~87-92% of missed MetaQA gold partitions have no gold node inside the structural list.
That is only actionable if the node was REACHABLE and the beam dropped it.  So compute the closure
of the frozen traversal with the beam removed but every other frozen constant kept:

    seeds = fused200[:SEED_K]      identical
    MAX_HOPS = 3, DEG_CAP = 300    identical
    beam top-M per hop             REMOVED (this is the whole point)

    A1 UNREACHABLE   gold node outside the 3-hop DEG_CAP closure -> no scoring change can find it
    A2 BEAM_PRUNED   gold node inside the closure but absent from s_node -> addressable by scoring

STEP 8 correction: struct_aggregate_full reads only the first M_struct = 64 structural nodes, so the
S4 list is at most 64 long and "partition in S4 top-64/80/100" is trivially true.  The displacement
is therefore re-measured against S4 aggregated over the FULL 256-node list, and against the final
P50 itself, with the S4 list length reported so the numbers are interpretable.

  python scratchpad/_l1sr_reach.py [ds]
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
import _l1sr_diag as D1

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
SRD = D1.SRD
NOPE = D1.NOPE


def closure(adjp, adji, deg, seeds, hops=TA.MAX_HOPS, cap=TA.DEG_CAP):
    """the frozen traversal with the beam removed; every other frozen constant kept."""
    seen = set(int(x) for x in seeds)
    fr = np.array([int(x) for x in seeds], np.int64)
    per_hop = []
    for _ in range(hops):
        fr = fr[deg[fr] <= cap]
        if not len(fr):
            break
        sl = [np.arange(adjp[e], adjp[e + 1]) for e in fr]
        if not sl:
            break
        nb = np.unique(adji[np.concatenate(sl)]) if sl else np.array([], np.int64)
        nb = nb[deg[nb] <= cap]
        nb = np.array([v for v in nb if int(v) not in seen], np.int64)
        if not len(nb):
            break
        seen.update(int(v) for v in nb)
        per_hop.append(len(nb))
        fr = nb
    return seen, per_hop


def agg_at(z, qi, hard, depth):
    agg = {}
    sn = z["s_node"][qi]; sh = z["s_hop"][qi]; sd = z["s_sdir"][qi]; sc = z["s_cnt"][qi]
    for jj in range(min(len(sn), depth)):
        v = int(sn[jj])
        if v < 0:
            break
        p = int(hard[v])
        if p < 0:
            continue
        a = agg.get(p)
        if a is None:
            agg[p] = [jj, 1, 1.0 / (RT.K0 + jj), int(sh[jj]), float(sd[jj]), int(sc[jj])]
        else:
            a[1] += 1; a[2] += 1.0 / (RT.K0 + jj)
            a[3] = min(a[3], int(sh[jj])); a[4] = max(a[4], float(sd[jj])); a[5] += int(sc[jj])
    return agg


def main(ds="metaqa"):
    S = CV.substrate(ds)
    z = {k: S["z"][k] for k in S["z"].files} if hasattr(S["z"], "files") else S["z"]
    gr, hard = D1.gold_rows_of(ds, z)
    hard_t, _, npart, (adjp, adji), deg, _ = TA.load_topology(ds, lambda *a: None)
    adjp = np.asarray(adjp); adji = np.asarray(adji); deg = np.asarray(deg)
    ctxs, goldp, hops = S["ctxs"], S["goldp"], S["hops"]

    cnt = {"A1_UNREACHABLE": 0, "A2_BEAM_PRUNED": 0}
    by_hop = {2: dict(cnt), 3: dict(cnt)}
    reach_sz, hop_sizes = [], []
    s4len64, s4len256 = [], []
    d64, d256, dfin = [], [], []
    top8 = {"n": 0, "S4_256_top50": 0, "S4_256_top64": 0, "S4_256_top80": 0,
            "S4_256_top100": 0, "in_final_P50": 0}
    for qi, c in enumerate(ctxs):
        X, _ = KB.f6_select(c["bnd"], list(c["chal"]), c["spos"], c["rpos"], c["cpos"], CV.B)
        final = c["prot_set"] | set(X)
        a64 = c["sagg"]; a256 = agg_at(z, qi, hard, 256)
        r64 = {p: i for i, p in enumerate(RT.order_struct(a64, "S4"))}
        r256 = {p: i for i, p in enumerate(RT.order_struct(a256, "S4"))}
        s4len64.append(len(r64)); s4len256.append(len(r256))
        sn = z["s_node"][qi]
        instruct = set(int(v) for v in sn if v >= 0)
        need = hops is not None and hops[qi] in (2, 3)
        cl = None
        for g in gr[qi]:
            p = int(hard[g])
            w = np.where(sn == g)[0]
            k = int(w[0]) if len(w) else NOPE
            if k < NOPE:
                d64.append((k, r64.get(p, NOPE)))
                d256.append((k, r256.get(p, NOPE)))
                dfin.append((k, int(p in final)))
                if k < 8:
                    top8["n"] += 1
                    for t in (50, 64, 80, 100):
                        top8[f"S4_256_top{t}"] += int(r256.get(p, NOPE) < t)
                    top8["in_final_P50"] += int(p in final)
            elif need and p not in final:
                if cl is None:
                    cl, ph = closure(adjp, adji, deg, z["seeds"][qi][z["seeds"][qi] >= 0])
                    reach_sz.append(len(cl)); hop_sizes.append(ph)
                key = "A2_BEAM_PRUNED" if int(g) in cl else "A1_UNREACHABLE"
                cnt[key] += 1; by_hop[int(hops[qi])][key] += 1
        if (qi + 1) % 500 == 0:
            log(f"   {qi+1}/{S['nq']}  closure mean {np.mean(reach_sz) if reach_sz else 0:.0f}")

    tot = sum(cnt.values())
    out = {"ds": ds, "n_nodes": int(len(hard)),
           "A_SPLIT": {k: {"n": v, "frac": round(v / max(1, tot), 4)} for k, v in cnt.items()},
           "A_SPLIT_by_hop": {f"hop{h}": {k: {"n": v, "frac": round(v / max(1, sum(d.values())), 4)}
                                          for k, v in d.items()} for h, d in by_hop.items()},
           "closure": {"queries_measured": len(reach_sz),
                       "mean_nodes_reached": round(float(np.mean(reach_sz)), 1) if reach_sz else 0,
                       "frac_of_corpus": (round(float(np.mean(reach_sz)) / len(hard), 4)
                                          if reach_sz else 0),
                       "mean_new_per_hop": [round(float(np.mean([h[i] for h in hop_sizes
                                                                 if len(h) > i])), 1)
                                            for i in range(3)
                                            if any(len(h) > i for h in hop_sizes)]},
           "S4_list_length": {"at_M_struct_64": round(float(np.mean(s4len64)), 1),
                              "at_depth_256": round(float(np.mean(s4len256)), 1)},
           "STEP8": {}}
    for nm, arr in (("S4_at_64", d64), ("S4_at_256", d256)):
        v = np.array([x[1] for x in arr if x[1] < NOPE], np.float64)
        out["STEP8"][nm] = {"n": len(arr), "absent": int(sum(1 for x in arr if x[1] >= NOPE)),
                            "median_partition_rank": float(np.median(v)) if len(v) else None,
                            "p90_partition_rank": float(np.percentile(v, 90)) if len(v) else None}
    out["STEP8"]["top8_gold_nodes"] = {
        "n": top8["n"],
        **{k: {"n": top8[k], "frac": round(top8[k] / max(1, top8["n"]), 4)}
           for k in top8 if k != "n"}}
    band = {}
    for k, r in d256:
        b = "0-7" if k < 8 else "8-15" if k < 16 else "16-31" if k < 32 else \
            "32-63" if k < 64 else "64-127" if k < 128 else "128-255"
        band.setdefault(b, []).append(r)
    out["STEP8"]["by_node_rank_band"] = {
        b: {"n": len(v), "median_S4_256_partition_rank":
            (float(np.median([x for x in v if x < NOPE])) if any(x < NOPE for x in v) else None),
            "absent": int(sum(1 for x in v if x >= NOPE))}
        for b, v in sorted(band.items(), key=lambda kv: int(kv[0].split("-")[0]))}
    fp = f"{SRD}/diag/step1b_{ds}.json"
    json.dump(out, open(fp, "w"), indent=1)
    log(f"  A SPLIT total={tot}  " + "  ".join(
        f"{k}={v['n']}({v['frac']:.1%})" for k, v in out["A_SPLIT"].items()))
    for h in (2, 3):
        d = out["A_SPLIT_by_hop"][f"hop{h}"]
        log(f"    hop{h}: " + "  ".join(f"{k}={v['n']}({v['frac']:.1%})" for k, v in d.items()))
    log(f"  closure: {out['closure']['mean_nodes_reached']} nodes "
        f"({out['closure']['frac_of_corpus']:.1%} of corpus), new/hop "
        f"{out['closure']['mean_new_per_hop']}")
    log(f"  S4 list length: at M_struct=64 -> {out['S4_list_length']['at_M_struct_64']}, "
        f"at depth 256 -> {out['S4_list_length']['at_depth_256']}")
    log(f"  STEP8 S4@256: median partition rank "
        f"{out['STEP8']['S4_at_256']['median_partition_rank']}, "
        f"p90 {out['STEP8']['S4_at_256']['p90_partition_rank']}, "
        f"absent {out['STEP8']['S4_at_256']['absent']}/{out['STEP8']['S4_at_256']['n']}")
    t8 = out["STEP8"]["top8_gold_nodes"]
    log(f"  STEP8 top-8 gold nodes n={t8['n']}  " + "  ".join(
        f"{k}={t8[k]['frac']:.1%}" for k in ["S4_256_top50", "S4_256_top64", "S4_256_top80",
                                             "S4_256_top100", "in_final_P50"]))
    log(f"wrote {fp}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
