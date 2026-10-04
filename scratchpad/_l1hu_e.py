"""PART E -- halo CANDIDATE-POOL family leave-one-out + ranking-rule controls.

Stays inside the shipped PER-CORE mechanism (each of a query's 50 selected cores gets its own
independently-capped budget, floor(beta*|C_j|), beta=0.5) -- only WHICH EDGES populate the
candidate pool, or HOW candidates within it are ranked, changes between variants.  The per-core
cap count is pinned to the SAME beta=0.5xcore-size formula in every variant, so any delta is
entirely pool/ranking, never a bigger budget.

  E0_FULL_C            shipped pool: union(STRUCT,NERX,KNN), boundary-mass ranked  (cached, reused)
  E1_MINUS_STRUCT      pool = union(NERX,KNN),    boundary-mass ranked
  E2_MINUS_NERX        pool = union(STRUCT,KNN),  boundary-mass ranked
  E3_MINUS_KNN         pool = union(STRUCT,NERX), boundary-mass ranked
  E4_RANDOM_SAME_POOL  E0's SAME pool, SAME per-core cap, seed=0 random pick instead of top-mass
  E5_BOUNDARY_MASS      alias of E0 -- the "does the RANKING RULE matter" contrast partner for E4

  python scratchpad/_l1hu_e.py run <ds>
  python scratchpad/_l1hu_e.py report
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ov_eval as OVE
import _l1ep_pu as PU
import _l1kn_sub as KS
import _l1hu_hard as HH
import _l1hu_global as GL

OUT = HH.OUT
TAG = GL.TAG
BETA_REF = GL.BETA_REF
BMCACHE = "scratchpad/_l1hu/attr_cache"
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def boundary_mass_keys(ds, hard, N, npart, keys, label, log=log, tag=None):
    """OV.boundary_mass's own formula, generalized to an arbitrary undirected key array. Cached.

    `tag` defaults to the module-level shipped TAG -- pass an explicit tag when `hard` is NOT the
    shipped C1_HYPER_UNIVERSAL partition, or this cache (keyed only by ds+label otherwise) will
    silently return mass computed under a DIFFERENT partition assignment.
    """
    tag = tag or TAG
    os.makedirs(BMCACHE, exist_ok=True)
    fp = "%s/bmass_%s__%s__%s.npz" % (BMCACHE, ds, tag, label)
    if os.path.exists(fp):
        z = np.load(fp)
        return z["pairs"], z["mass"]
    u = (keys // np.int64(N)).astype(np.int64)
    v = (keys % np.int64(N)).astype(np.int64)
    deg = np.bincount(np.concatenate([u, v]), minlength=N).astype(np.float64)
    deg[deg == 0] = 1.0
    hu, hv = hard[u], hard[v]
    cross = hu != hv
    u, v, hu, hv = u[cross], v[cross], hu[cross], hv[cross]
    pk = np.concatenate([hu * np.int64(N) + v, hv * np.int64(N) + u])
    w = np.concatenate([1.0 / deg[u], 1.0 / deg[v]])
    o = np.argsort(pk, kind="stable")
    pk, w = pk[o], w[o]
    uniq, start = np.unique(pk, return_index=True)
    mass = np.add.reduceat(w, start)
    np.savez_compressed(fp, pairs=uniq, mass=mass)
    log("  %s %s: boundary mass for %d pairs" % (ds, label, len(uniq)))
    return uniq, mass


def bounded_pairs_random(pairs, hard, npart, N, beta, seed=0):
    """OV.bounded_pairs with a deterministic random priority instead of -mass -- same cap formula."""
    j = (np.asarray(pairs) // np.int64(N)).astype(np.int64)
    core = np.bincount(np.asarray(hard, np.int64), minlength=npart).astype(np.int64)
    cap = np.floor(beta * core).astype(np.int64)
    rnd = np.random.RandomState(seed).rand(len(j))
    order = np.lexsort((rnd, j))
    js = j[order]
    start = np.searchsorted(js, np.arange(npart), side="left")
    rank = np.arange(len(js), dtype=np.int64) - start[js]
    keep = rank < cap[js]
    return np.sort(np.asarray(pairs)[order][keep])


def eval_pairs(hard, npart, N, pairs, base50, f650, need, hops=None, lanes=("F6", "BASE")):
    nptr, nidx = OV.to_node_csr(pairs, npart, N)
    bptr, bidx = OV.to_block_csr(pairs, npart, N)
    core_sizes = np.bincount(hard, minlength=npart).astype(np.int64)
    SELS = {"BASE": base50, "F6": f650}
    out = {}
    for lane in lanes:
        SEL = SELS[lane]
        nq = len(SEL)
        allf = np.zeros(nq, np.int8); newh = np.zeros(nq, np.int64)
        expo = np.zeros(nq, np.int64); expo_core = np.zeros(nq, np.int64)
        selmask = np.zeros(npart, bool)
        for qi in range(nq):
            nd = need[qi]
            if not nd:
                continue
            S = SEL[qi]; Ss = set(S)
            selmask[:] = False; selmask[np.asarray(S, np.int64)] = True
            got = {x for x in nd if int(hard[x]) in Ss}
            incore_n = len(got)
            for x in nd:
                if x in got:
                    continue
                bs = nidx[nptr[x]:nptr[x + 1]]
                if len(bs) and Ss.intersection(bs.tolist()):
                    got.add(x)
            allf[qi] = int(len(got) == len(nd))
            newh[qi] = len(got) - incore_n
            expo_core[qi] = int(core_sizes[S].sum())
            if len(bidx):
                u = np.unique(np.concatenate([bidx[bptr[j]:bptr[j + 1]] for j in S]))
                expo[qi] = expo_core[qi] + int(np.count_nonzero(~selmask[hard[u.astype(np.int64)]]))
            else:
                expo[qi] = expo_core[qi]
        ok = np.array([bool(x) for x in need])
        extra = float(expo[ok].mean() - expo_core[ok].mean())
        rec = {"ALL_REQUIRED_FETCHED": round(float(allf[ok].mean()), 4),
              "NEW_REQUIRED_FROM_HALO": int(newh.sum()),
              "MEAN_EXTRA_EXPOSURE": round(extra, 1),
              "required_per_1k_extra_exposed": (round(float(newh.sum()) / nq / max(extra, 1e-9)
                                                       * 1000, 4) if extra > 0 else None),
              "_ind_ALL": allf.tolist()}
        if hops is not None:
            h = np.asarray(hops)[:nq]
            rec["hop3_ALL_REQUIRED_FETCHED"] = (round(float(allf[(h == 3) & ok].mean()), 4)
                                                if ((h == 3) & ok).any() else None)
        out[lane] = rec
    return out


def run(ds, log=log):
    hard, npart, N = GL.load_core(ds)
    z, meta, C, ctxs, base50, f650, ind_base, PAR = OVE.selected_blocks(ds, hard, npart, log)
    g, gptr, rows, hops = PU.gold_rows(ds)
    nq = meta["n_dev_queries"]
    need = [sorted({int(x) for x in g[gptr[qi]:gptr[qi + 1]]}) for qi in range(nq)]
    hops_arg = hops if ds == "metaqa" else None

    _, S, K, X = KS.keysets(ds, log)

    variants = {}
    cov = json.load(open("%s/halo/COVERAGE_%s.json" % (OUT, ds)))["C1_HYPER_UNIVERSAL"]["O4_FULL_C_b0.5"]
    e0 = {lane: {"ALL_REQUIRED_FETCHED": cov[lane]["ALL_REQUIRED_FETCHED"],
                "NEW_REQUIRED_FROM_HALO": cov[lane]["NEW_REQUIRED_FROM_HALO"],
                "required_per_1k_extra_exposed": cov[lane]["required_per_1k_extra_exposed"],
                "_ind_ALL": cov[lane]["_ind_ALL"]}
         for lane in ("F6", "BASE")}
    if ds == "metaqa":
        for lane in e0:
            e0[lane]["hop3_ALL_REQUIRED_FETCHED"] = cov[lane].get("BY_HOP", {}).get("hop3", {}).get(
                "ALL_REQUIRED_FETCHED")
    variants["E0_FULL_C"] = e0
    variants["E5_BOUNDARY_MASS"] = e0

    full_pairs, full_mass = OV.boundary_mass(ds, hard, TAG, "O4_FULL_C", N, npart, log)

    for label, keys in (("E1_MINUS_STRUCT", np.union1d(X, K)),
                        ("E2_MINUS_NERX", np.union1d(S, K)),
                        ("E3_MINUS_KNN", np.union1d(S, X))):
        uniq, mass = boundary_mass_keys(ds, hard, N, npart, keys, label, log)
        pairs = OV.bounded_pairs(uniq, mass, hard, npart, N, BETA_REF)
        t0 = time.time()
        variants[label] = eval_pairs(hard, npart, N, pairs, base50, f650, need, hops_arg)
        log("  %s %s evaluated in %.1fs" % (ds, label, time.time() - t0))

    rnd_pairs = bounded_pairs_random(full_pairs, hard, npart, N, BETA_REF, seed=0)
    t0 = time.time()
    variants["E4_RANDOM_SAME_POOL"] = eval_pairs(hard, npart, N, rnd_pairs, base50, f650, need, hops_arg)
    log("  %s E4_RANDOM_SAME_POOL evaluated in %.1fs" % (ds, time.time() - t0))

    os.makedirs("%s/halo_family" % OUT, exist_ok=True)
    fp = "%s/halo_family/E_%s.json" % (OUT, ds)
    json.dump({"ds": ds, "nq": nq, "variants": variants}, open(fp, "w"), indent=1)
    log("wrote", fp)
    return variants


def report():
    out_rows = []
    for ds in HH.DS:
        fp = "%s/halo_family/E_%s.json" % (OUT, ds)
        if not os.path.exists(fp):
            continue
        rec = json.load(open(fp))["variants"]
        ref = rec["E0_FULL_C"]["F6"]
        row = {"ds": ds}
        for label in ("E1_MINUS_STRUCT", "E2_MINUS_NERX", "E3_MINUS_KNN", "E4_RANDOM_SAME_POOL"):
            v = rec[label]["F6"]
            g_, l_, p_ = HH.mcnemar(ref["_ind_ALL"], v["_ind_ALL"])
            d = round(v["ALL_REQUIRED_FETCHED"] - ref["ALL_REQUIRED_FETCHED"], 4)
            row[label] = {"ALL_REQUIRED": v["ALL_REQUIRED_FETCHED"], "delta_vs_E0": d,
                         "gained": g_, "lost": l_, "p": p_, "sig": bool(p_ < 0.05)}
        row["E0_ALL_REQUIRED"] = ref["ALL_REQUIRED_FETCHED"]
        out_rows.append(row)
    json.dump(out_rows, open("%s/halo_family/E_REPORT.json" % OUT, "w"), indent=1)
    for r in out_rows:
        print(json.dumps(r, indent=1))
    return out_rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
