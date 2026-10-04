"""STEPS 3-7 -- UNIVERSAL PARAMETER-FREE FIXED-P50 BOUNDARY ROUTER SEARCH.

    base_rank[:50-B]   protected core, never changes
    base_rank[50-B:50] boundary, the ONLY B slots a challenger may take
    output             EXACTLY 50 canonical C partitions, no stray nodes

Structural aggregations (node evidence -> partition evidence), all deterministic:
  S1 BEST_NODE_RANK   best (lowest) structural residual rank inside the partition
  S2 SUM_NODE_RRF     sum of 1/(K0+j) over the partition's structural nodes
  S3 SUPPORT_FIRST    lexicographic: #distinct structural nodes desc, best node rank asc,
                      total structural support count desc, partition id asc
                      (per-SEED provenance is not produced by the frozen mechanism, so the
                       support term is node-count + expand_dir support count, not seed count)
  S4 MULTI_SIGNAL_RRF fixed RRF over four separate partition rankings: best node rank,
                      node-count support, min hop, best s_dir

Boundary fusion families, all parameter-free (fixed canonical K0, no weights):
  F1 BOUNDARY_RRF            cands = boundary u STRUCT;      channels = canonical + structural
  F2 COMBINED_BOUNDARY_RRF   cands = boundary u STRUCT u RET; channels = canonical + structural + ret
  F3 STRUCT_PROMOTION_ONLY   top-k structural challengers displace the last k boundary slots
  F4 SUPPORT_GATED_PROMOTION F3 restricted to challengers with >= T distinct structural nodes
  F5 RET_BOUNDARY_RRF        CONTROL: cands = boundary u RET; channels = canonical + ret only
                             (isolates whether structure adds anything over retrieval continuation)

A channel that has no evidence for a candidate contributes 0 (canonical per-query masking).
No learned weights, no fitted thresholds, no dataset branch, no gold at inference.

  python scratchpad/_l1ps_router.py [round1|round2]
"""
import os, sys, json, time, hashlib, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np

ROOT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH"
DSETS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
K0 = 60
P = 50
ROUND = sys.argv[1] if len(sys.argv) > 1 else "round1"


def mcnemar(cur, base):
    from math import comb
    win = int(((cur == 1) & (base == 0)).sum()); los = int(((cur == 0) & (base == 1)).sum())
    n2 = win + los
    p = 1.0 if n2 == 0 else min(1.0, 2.0 * sum(comb(n2, i) for i in range(min(win, los) + 1)) / (2.0 ** n2))
    return {"gained": win, "lost": los, "net": win - los, "mcnemar_p": round(p, 5),
            "sig": bool(p < 0.05)}


def struct_aggregate(z, qi, hard, base50, M_struct):
    """partition -> (best_node_rank, n_nodes, sum_rrf, min_hop, best_sdir, total_support)
    over the first M_struct out-of-P50 structural residual NODES (the validated STRUCT-M set)."""
    agg = {}
    got = 0
    sn = z["s_node"][qi]; sh = z["s_hop"][qi]; sd = z["s_sdir"][qi]; sc = z["s_cnt"][qi]
    for jj in range(len(sn)):
        v = int(sn[jj])
        if v < 0:
            break
        p = int(hard[v])
        if p < 0 or p in base50:
            continue
        got += 1
        a = agg.get(p)
        if a is None:
            agg[p] = [jj, 1, 1.0 / (K0 + jj), int(sh[jj]), float(sd[jj]), int(sc[jj])]
        else:
            a[1] += 1; a[2] += 1.0 / (K0 + jj)
            a[3] = min(a[3], int(sh[jj])); a[4] = max(a[4], float(sd[jj])); a[5] += int(sc[jj])
        if got >= M_struct:
            break
    return agg


def order_struct(agg, mode):
    """deterministic challenger ordering; ties always broken by canonical partition id."""
    if mode == "S1":
        return [p for p, _ in sorted(agg.items(), key=lambda kv: (kv[1][0], kv[0]))]
    if mode == "S2":
        return [p for p, _ in sorted(agg.items(), key=lambda kv: (-kv[1][2], kv[0]))]
    if mode == "S3":
        return [p for p, _ in sorted(agg.items(),
                                     key=lambda kv: (-kv[1][1], kv[1][0], -kv[1][5], kv[0]))]
    if mode == "S4":
        ps = list(agg)
        rk = {}
        for key, rev in ((lambda p: agg[p][0], False), (lambda p: -agg[p][1], False),
                         (lambda p: agg[p][3], False), (lambda p: -agg[p][4], False)):
            for r, p in enumerate(sorted(ps, key=lambda p: (key(p), p))):
                rk[p] = rk.get(p, 0.0) + 1.0 / (K0 + r)
        return [p for p in sorted(ps, key=lambda p: (-rk[p], p))]
    raise ValueError(mode)


def struct_aggregate_full(z, qi, hard, M_struct):
    """SYMMETRIC evidence: identical aggregation over the first M_struct structural nodes but
    WITHOUT dropping nodes whose partition is already inside P50, so an incumbent boundary
    partition that structure also supports keeps its structural vote and can defend its slot.
    (Round-1 diagnosis: challengers got 2-3 channels and incumbents got 1, so churn was ~100%
    of B and the family-independent eviction cost was paid on every query.)"""
    agg = {}
    sn = z["s_node"][qi]; sh = z["s_hop"][qi]; sd = z["s_sdir"][qi]; sc = z["s_cnt"][qi]
    for jj in range(min(len(sn), M_struct)):
        v = int(sn[jj])
        if v < 0:
            break
        p = int(hard[v])
        if p < 0:
            continue
        a = agg.get(p)
        if a is None:
            agg[p] = [jj, 1, 1.0 / (K0 + jj), int(sh[jj]), float(sd[jj]), int(sc[jj])]
        else:
            a[1] += 1; a[2] += 1.0 / (K0 + jj)
            a[3] = min(a[3], int(sh[jj])); a[4] = max(a[4], float(sd[jj])); a[5] += int(sc[jj])
    return agg


def ret_aggregate_full(z, qi, hard, M_ret):
    """symmetric counterpart for the retrieval continuation channel."""
    agg = {}
    rr = z["ret_rrf"][qi]
    for jj in range(min(len(rr), M_ret)):
        v = int(rr[jj])
        if v < 0:
            break
        p = int(hard[v])
        if p < 0:
            continue
        a = agg.get(p)
        if a is None:
            agg[p] = [jj, 1, 1.0 / (K0 + jj), 0, 0.0, 1]
        else:
            a[1] += 1; a[2] += 1.0 / (K0 + jj)
    return agg


def ret_challengers(z, qi, hard, base50, M_ret):
    """ordered out-of-P50 partitions from the canonical rrf200 node continuation."""
    out = []; got = 0
    for v in z["ret_rrf"][qi]:
        v = int(v)
        if v < 0:
            break
        p = int(hard[v])
        if p < 0 or p in base50:
            continue
        got += 1
        if p not in out:
            out.append(p)
        if got >= M_ret:
            break
    return out


def evaluate(ds, z, meta, cfg, cache):
    hard = z["hard"]; base_rank = z["base_rank"]; part_sizes = z["part_sizes"]
    gp, gptr, gc = z["gold_part"], z["gold_ptr"], z["gold_cnt"]
    hops = z["hops"]; nq = meta["n_dev_queries"]
    B = cfg["B"]; fam = cfg["fusion"]; agg_mode = cfg["agg"]; T = cfg.get("T", 1)
    S = cache["struct"][(cfg["M_struct"], agg_mode)]
    A = cache["agg"][cfg["M_struct"]]
    Rc = cache["ret"][cfg["M_ret"]]
    base50 = cache["base50"]; goldp = cache["goldp"]; ind_base = cache["ind_base"]

    ind = np.zeros(nq, np.int8); churn = np.zeros(nq, np.int32)
    scope = np.zeros(nq, np.int64)
    ind_any = np.zeros(nq, np.int8); any_base = np.zeros(nq, np.int8)
    src = {"struct_only": 0, "ret_only": 0, "both": 0, "boundary_reorder": 0}
    adm = ev = 0
    for qi in range(nq):
        br = base_rank[qi]
        prot = [int(x) for x in br[:P - B]]
        bnd = [int(x) for x in br[P - B:P]]
        sch, rch = S[qi], Rc[qi]
        if fam in ("F6", "F7", "F8", "F9", "FA"):
            # SYMMETRIC boundary competition: incumbents and challengers are scored on exactly the
            # same channels, so a slot only turns over when the evidence actually prefers the
            # challenger. B becomes a CAP on churn, not a forced churn.
            SF = cache["sfull"][(cfg["M_struct"], agg_mode)][qi]
            RF = cache["rfull"][cfg["M_ret"]][qi]
            use_s = fam in ("F6", "F7", "F9", "FA")
            use_r = fam in ("F6", "F8", "F9", "FA")
            chal = [p for p in (SF if use_s else []) if p not in base50[qi]] + \
                   [p for p in (RF if use_r else []) if p not in base50[qi]]
            if fam in ("F9", "FA"):
                # refinement H: one structural node is weak, unstable evidence for a whole
                # partition.  Require deterministic multi-node support before a challenger
                # may compete at all.  Incumbents are NEVER gated -- they hold their slot by
                # default, so the gate can only reduce churn, never force it.
                # F9 gates both channels at T; FA gates only the structural channel, because
                # retrieval continuation is already a dense/SPLADE consensus signal.
                SA = cache["saggf"][cfg["M_struct"]][qi]
                RA = cache["raggf"][cfg["M_ret"]][qi]
                keep = []
                for p in chal:
                    ns = SA[p][1] if p in SA else 0
                    nr = RA[p][1] if p in RA else 0
                    if (max(ns, nr) >= T) if fam == "F9" else (nr >= 1 or ns >= T):
                        keep.append(p)
                chal = keep
            cands = bnd + [p for p in dict.fromkeys(chal) if p not in bnd]
            spos = {p: r for r, p in enumerate(SF)}
            rpos = {p: r for r, p in enumerate(RF)}
            cpos = cache["cpos"][qi]
            sc = []
            for p in cands:
                s = 1.0 / (K0 + cpos[p]) if p in cpos else 0.0
                if use_s and p in spos:
                    s += 1.0 / (K0 + spos[p])
                if use_r and p in rpos:
                    s += 1.0 / (K0 + rpos[p])
                sc.append((-s, cpos.get(p, 10 ** 6), p))
            sc.sort()
            final = prot + [p for _, _, p in sc[:B]]
        elif fam in ("F3", "F4"):
            pool = sch if fam == "F3" else [p for p in sch if A[qi][p][1] >= T]
            k = min(B, len(pool))
            final = [int(x) for x in br[:P - k]] + pool[:k]
        else:
            if fam == "F1":
                cands = bnd + [p for p in sch if p not in bnd]
                chans = (True, True, False)
            elif fam == "F5":
                cands = bnd + [p for p in rch if p not in bnd]
                chans = (True, False, True)
            else:
                cands = bnd + [p for p in sch if p not in bnd] + \
                        [p for p in rch if p not in bnd and p not in sch]
                chans = (True, True, True)
            spos = {p: r for r, p in enumerate(sch)}
            rpos = {p: r for r, p in enumerate(rch)}
            cpos = cache["cpos"][qi]
            sc = []
            for p in cands:
                s = 0.0
                if chans[0] and p in cpos:
                    s += 1.0 / (K0 + cpos[p])
                if chans[1] and p in spos:
                    s += 1.0 / (K0 + spos[p])
                if chans[2] and p in rpos:
                    s += 1.0 / (K0 + rpos[p])
                sc.append((-s, cpos.get(p, 10 ** 6), p))
            sc.sort()
            final = prot + [p for _, _, p in sc[:B]]
        assert len(set(final)) == P, f"{ds} {cfg}: produced {len(set(final))} partitions, not {P}"
        fs = set(final)
        ind[qi] = int(goldp[qi] <= fs)
        ind_any[qi] = int(bool(goldp[qi] & fs))
        any_base[qi] = int(bool(goldp[qi] & base50[qi]))
        newp = fs - base50[qi]
        churn[qi] = len(newp)
        scope[qi] = int(part_sizes[sorted(fs)].sum())
        adm += len(goldp[qi] & newp)
        ev += len(goldp[qi] & (base50[qi] - fs))
        if ind[qi] and not ind_base[qi]:
            # which channel supplied the partition(s) that actually flipped this query?
            dec = goldp[qi] & newp
            sset = set(cache["sfull"][(cfg["M_struct"], agg_mode)][qi])
            rset = set(cache["rfull"][cfg["M_ret"]][qi])
            a = any(p in sset for p in dec); b = any(p in rset for p in dec)
            src["both" if (a and b) else "struct_only" if a
                else "ret_only" if b else "boundary_reorder"] += 1
    m = mcnemar(ind, ind_base)
    out = {"ALL": round(float(ind.mean()), 4),
           "dALL": round(float(ind.mean()) - cache["BASE_ALL"], 4),
           "newly_covered": m["gained"], "newly_uncovered": m["lost"], "net": m["net"],
           "mcnemar_p": m["mcnemar_p"], "sig": m["sig"],
           "gold_parts_admitted": int(adm), "gold_parts_evicted": int(ev),
           "net_gold_parts": int(adm - ev),
           "churn_per_query": round(float(churn.mean()), 3),
           "churn_max": int(churn.max()),
           "queries_with_zero_churn": int((churn == 0).sum()),
           "ANY": round(float(ind_any.mean()), 4),
           "dANY": round(float(ind_any.mean() - any_base.mean()), 4),
           "decisive_channel_for_newly_covered": src,
           "final_scope_nodes_mean": round(float(scope.mean()), 1)}
    if (hops >= 0).any():
        out["per_hop"] = {str(h): {"n": int((hops == h).sum()),
                                   "BASE_ALL": round(float(ind_base[hops == h].mean()), 4),
                                   "ALL": round(float(ind[hops == h].mean()), 4)}
                          for h in sorted(set(int(x) for x in hops if x >= 0))}
    out["_ind"] = ind
    return out


def build_cache(ds, z, meta, m_structs, m_rets, aggs):
    hard = z["hard"]; base_rank = z["base_rank"]; gp, gptr = z["gold_part"], z["gold_ptr"]
    nq = meta["n_dev_queries"]
    base50 = [set(int(x) for x in base_rank[qi][:P]) for qi in range(nq)]
    goldp = [set(int(x) for x in gp[gptr[qi]:gptr[qi + 1]]) for qi in range(nq)]
    ind_base = np.array([int(goldp[qi] <= base50[qi]) for qi in range(nq)], np.int8)
    cpos = [{int(p): r for r, p in enumerate(base_rank[qi])} for qi in range(nq)]
    C = {"base50": base50, "goldp": goldp, "ind_base": ind_base, "cpos": cpos,
         "BASE_ALL": float(ind_base.mean()), "agg": {}, "struct": {}, "ret": {},
         "sfull": {}, "rfull": {}, "saggf": {}, "raggf": {}}
    for M in m_structs:
        C["agg"][M] = [struct_aggregate(z, qi, hard, base50[qi], M) for qi in range(nq)]
        full = [struct_aggregate_full(z, qi, hard, M) for qi in range(nq)]
        C["saggf"][M] = full
        for a in aggs:
            C["struct"][(M, a)] = [order_struct(C["agg"][M][qi], a) for qi in range(nq)]
            C["sfull"][(M, a)] = [order_struct(full[qi], a) for qi in range(nq)]
    for M in m_rets:
        C["ret"][M] = [ret_challengers(z, qi, hard, base50[qi], M) for qi in range(nq)]
        C["raggf"][M] = [ret_aggregate_full(z, qi, hard, M) for qi in range(nq)]
        C["rfull"][M] = [[p for p, _ in sorted(C["raggf"][M][qi].items(),
                                               key=lambda kv: (kv[1][0], kv[0]))]
                         for qi in range(nq)]
    return C


def main():
    GRIDS = {
        # round 1: the four aggregations x the asymmetric families of STEP 6
        "round1": dict(B=[1, 2, 4, 8, 12], M_struct=[32], M_ret=[32], agg=["S1", "S2", "S3", "S4"],
                       fusion=["F1", "F2", "F3", "F4T1", "F4T2", "F4T3", "F5"]),
        # round 2 (refinement A+F, mechanism-guided): SYMMETRIC competition, so B caps churn
        # instead of forcing it. F6 = struct+ret, F7 = struct only, F8 = ret only.
        "round2": dict(B=[1, 2, 4, 8, 12, 16, 24], M_struct=[32], M_ret=[32],
                       agg=["S1", "S2", "S3", "S4"], fusion=["F6", "F7", "F8"]),
        # round 3 (refinement E): widen the challenger SOURCE at the winning B, never the slots
        "round3": dict(B=[2, 4, 6, 8], M_struct=[32, 64, 128], M_ret=[32, 64, 128],
                       agg=["S1", "S4"], fusion=["F6"]),
        # round 4 (refinement H): require deterministic MULTI-NODE support before a challenger may
        # compete at all.  F6 is carried as the in-round ungated control at identical cells.
        # F9 (gate BOTH channels) was refuted on a musique smoke test -- it zeroes the retrieval
        # channel (ret_only decisive 10 -> 0), so only FA (gate the STRUCTURAL channel) is carried.
        "round4": dict(B=[4, 6, 8], M_struct=[64, 128], M_ret=[32, 64], agg=["S1", "S4"],
                       fusion=["F6", "FAT2", "FAT3"]),
    }
    grid = GRIDS[ROUND]
    t0 = time.time()
    SB = {}
    for ds in DSETS:
        p = f"{ROOT}/runs/cache_{ds}.npz"
        z = np.load(p, allow_pickle=True); meta = json.loads(str(z["meta_json"]))
        cfgs = [dict(B=b, M_struct=ms, M_ret=mr, agg=a,
                     fusion=f[:2],
                     T=(int(f[3:]) if len(f) > 2 and f[2] == "T" else 1), tag=f)
                for b, ms, mr, a, f in itertools.product(
                    grid["B"], grid["M_struct"], grid["M_ret"], grid["agg"], grid["fusion"])]
        ms = sorted({c["M_struct"] for c in cfgs}); mr = sorted({c["M_ret"] for c in cfgs})
        ag = sorted({c["agg"] for c in cfgs})
        C = build_cache(ds, z, meta, ms, mr, ag)
        assert abs(round(C["BASE_ALL"], 4) - meta["BASE_ALL_P50"]) < 1e-9
        res = {}
        for cfg in cfgs:
            name = f"B{cfg['B']}_{cfg['agg']}_{cfg.get('tag', cfg['fusion'])}" \
                   f"_Ms{cfg['M_struct']}_Mr{cfg['M_ret']}"
            r = evaluate(ds, z, meta, cfg, C)
            r.pop("_ind")
            res[name] = r
        SB[ds] = {"meta": {k: meta[k] for k in ("n_dev_queries", "BASE_ALL_P50", "BASE_ANY_P50",
                                                "BASE_SCOPE_NODES", "sample_rule")},
                  "configs": res}
        best = max(res, key=lambda k: res[k]["dALL"])
        print(f"[{ds}] BASE {meta['BASE_ALL_P50']:.4f}  best {best} -> {res[best]['ALL']:.4f} "
              f"({res[best]['dALL']:+.4f}, net {res[best]['net']:+})", flush=True)

    # ---- universal selection (STEP 9): lexicographic, one global config
    names = set.intersection(*[set(SB[d]["configs"]) for d in SB])
    rows = []
    for n in sorted(names):
        ds_d = {d: SB[d]["configs"][n] for d in SB}
        worst = min(v["dALL"] for v in ds_d.values())
        macro = float(np.mean([v["dALL"] for v in ds_d.values()]))
        nreg = sum(1 for v in ds_d.values() if v["sig"] and v["dALL"] < 0)
        rows.append({"config": n, "worst_dALL": round(worst, 4), "macro_dALL": round(macro, 4),
                     "n_sig_regressions": nreg,
                     "churn": round(float(np.mean([v["churn_per_query"] for v in ds_d.values()])), 3),
                     "per_ds": {d: ds_d[d]["dALL"] for d in ds_d},
                     "per_ds_net": {d: ds_d[d]["net"] for d in ds_d}})
    rows.sort(key=lambda r: (r["n_sig_regressions"], -r["worst_dALL"], -r["macro_dALL"], r["churn"]))
    out = {"round": ROUND, "K0": K0, "P": P, "runtime_sec": round(time.time() - t0, 1),
           "SCOREBOARD": SB, "RANKING": rows,
           "FINALISTS": [r["config"] for r in rows[:8]]}
    fp = f"{ROOT}/scoreboard_{ROUND}.json"
    json.dump(out, open(fp, "w"), indent=1)
    print(f"\n--- universal ranking (lexicographic: fewest sig regressions, then worst-dataset) ---")
    print(f"{'config':34s} {'worst':>8s} {'macro':>8s} {'reg':>4s} {'churn':>6s}  per-dataset dALL")
    for r in rows[:14]:
        print(f"{r['config']:34s} {r['worst_dALL']:+8.4f} {r['macro_dALL']:+8.4f} "
              f"{r['n_sig_regressions']:4} {r['churn']:6.2f}  " +
              " ".join(f"{d[:4]}:{v:+.4f}" for d, v in r["per_ds"].items()))
    print(f"\nwrote {fp}  ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
