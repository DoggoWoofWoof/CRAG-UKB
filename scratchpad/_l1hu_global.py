"""PART F -- global dedup-aware halo budget allocation (the PRIMARY optimization experiment).

F0 (shipped today): each of the 50 selected cores gets its OWN beta=0.5 boundary-mass budget,
capped independently, then a query's fetch is the union of 50 independently-capped halos.  A node
supported by several selected cores gets no credit for that; a core with a locally weak boundary
can still burn its whole budget on nodes no other core would have picked.

F1/F2 keep the core RANKING bit-identical (nothing here touches base50/f650) and change only what
a query's *fetch* pays for: gather the UNCAPPED FULL_C candidate halo nodes from every one of the
query's 50 selected cores, score each candidate ONCE across all of them, and cut at the SAME
per-query node budget F0 would have spent (K_q = F0's own new-halo-node count for that query, not
a global constant) -- so any gain is entirely reallocation, not more exposure.

  F0_CURRENT_PER_CORE_BETA05   shipped mechanism (recomputed here, must reproduce cached COVERAGE)
  F1_MAX_BOUNDARY              score(v) = max boundary_mass(v, core_j) over supporting cores j
  F2_SUM_NORMALIZED            score(v) = sum boundary_mass(v,core_j)/max_mass(core_j) over j

No learned weight anywhere; both are deterministic functions of the same boundary-mass primitive
Phase 13 already validated.  Promote only if zero significant regressions AND improves the worst
corpus or the macro score; otherwise F0 stands and the halo-scorer search stops there.

  python scratchpad/_l1hu_global.py run <ds> [nsample]
  python scratchpad/_l1hu_global.py run_all [nsample]
  python scratchpad/_l1hu_global.py report
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ov_eval as OVE
import _l1ep_pu as PU
import _l1hu_hard as HH

OUT = HH.OUT
EPARTS = "scratchpad/_l1ep/parts"
TAG = "C1_HYPER_UNIVERSAL"
BETA_REF = 0.5
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def load_core(ds):
    hard = np.load("%s/%s__%s.npy" % (EPARTS, ds, TAG))
    hard = np.asarray(hard, np.int64)
    npart = int(hard.max()) + 1
    return hard, npart, len(hard)


def block_mass_csr(pairs, mass, npart, N):
    """block -> (node ids, mass), aligned.  pairs are sorted by j*N+v so groups are contiguous."""
    pairs = np.asarray(pairs)
    j = (pairs // np.int64(N)).astype(np.int64)
    v = (pairs % np.int64(N)).astype(np.int32)
    cnt = np.bincount(j, minlength=npart).astype(np.int64)
    ptr = np.zeros(npart + 1, np.int64)
    ptr[1:] = np.cumsum(cnt)
    return ptr, v, np.asarray(mass)


def block_max_mass(ptr, mass, npart):
    """per-block max candidate mass -- the normalizer F2 uses so no single core dominates by scale."""
    out = np.full(npart, 1e-12, np.float64)
    for j in np.where(np.diff(ptr) > 0)[0]:
        out[j] = float(mass[ptr[j]:ptr[j + 1]].max())
    return out


def run(ds, sample=None, lanes=("F6", "BASE"), log=log):
    hard, npart, N = load_core(ds)
    z, meta, C, ctxs, base50, f650, ind_base, PAR = OVE.selected_blocks(ds, hard, npart, log)
    g, gptr, rows, hops = PU.gold_rows(ds)
    nq = meta["n_dev_queries"]
    need = [sorted({int(x) for x in g[gptr[qi]:gptr[qi + 1]]}) for qi in range(nq)]

    pairs, mass = OV.boundary_mass(ds, hard, TAG, "O4_FULL_C", N, npart, log)
    bptr, bidx, bmass = block_mass_csr(pairs, mass, npart, N)
    bmax = block_max_mass(bptr, bmass, npart)

    f0_pairs = OV.bounded_pairs(pairs, mass, hard, npart, N, BETA_REF)
    f0_nptr, f0_nidx = OV.to_node_csr(f0_pairs, npart, N)
    f0_bptr, f0_bidx = OV.to_block_csr(f0_pairs, npart, N)

    qids = list(range(nq)) if sample is None else list(sample)
    SELS = {"BASE": base50, "F6": f650}
    out = {}
    for lane in lanes:
        SEL = SELS[lane]
        rows_ = {k: {"allf": np.zeros(len(qids), np.int8),
                     "newh": np.zeros(len(qids), np.int64),
                     "expo": np.zeros(len(qids), np.int64)}
                 for k in ("F0", "F1", "F2")}
        selmask = np.zeros(npart, bool)
        t_lane = time.time()
        for ii, qi in enumerate(qids):
            nd = need[qi]
            if not nd:
                continue
            nd_set = set(nd)
            S = SEL[qi]
            Ss = set(S)
            selmask[:] = False
            selmask[np.asarray(S, np.int64)] = True
            got_core = {x for x in nd if int(hard[x]) in Ss}

            # ---- F0: existing per-core beta=0.5 mechanism (recomputed for an exact per-query budget)
            got0 = set(got_core)
            for x in nd:
                if x in got0:
                    continue
                bs = f0_nidx[f0_nptr[x]:f0_nptr[x + 1]]
                if len(bs) and Ss.intersection(bs.tolist()):
                    got0.add(x)
            if len(f0_bidx):
                u0 = np.unique(np.concatenate([f0_bidx[f0_bptr[j]:f0_bptr[j + 1]] for j in S]))
                new0 = u0[~selmask[hard[u0.astype(np.int64)]]]
            else:
                new0 = np.zeros(0, np.int64)
            K_q = len(new0)
            rows_["F0"]["allf"][ii] = int(len(got0) == len(nd))
            rows_["F0"]["newh"][ii] = len(got0) - len(got_core)
            rows_["F0"]["expo"][ii] = K_q

            if K_q == 0:
                for k in ("F1", "F2"):
                    rows_[k]["allf"][ii] = int(len(got_core) == len(nd))
                continue

            # ---- gather UNCAPPED candidates from every selected core, score once, cut at K_q ----
            cv, cj, cm = [], [], []
            for j in S:
                s, e = bptr[j], bptr[j + 1]
                if e > s:
                    cv.append(bidx[s:e]); cj.append(np.full(e - s, j, np.int64)); cm.append(bmass[s:e])
            if cv:
                V = np.concatenate(cv); J = np.concatenate(cj); Mm = np.concatenate(cm)
                keep = ~selmask[hard[V.astype(np.int64)]]
                V, J, Mm = V[keep], J[keep], Mm[keep]
            else:
                V = np.zeros(0, np.int64)

            if len(V):
                order = np.argsort(V, kind="stable")
                Vs, Ms, Js = V[order], Mm[order], J[order]
                uniq_v, start = np.unique(Vs, return_index=True)
                f1_score = np.maximum.reduceat(Ms, start)
                f2_score = np.add.reduceat(Ms / bmax[Js], start)
            else:
                uniq_v = np.zeros(0, np.int64); f1_score = np.zeros(0); f2_score = np.zeros(0)

            for label, score in (("F1", f1_score), ("F2", f2_score)):
                if len(uniq_v):
                    k = min(K_q, len(uniq_v))
                    top = uniq_v[np.argpartition(-score, k - 1)[:k]] if k < len(uniq_v) else uniq_v
                    gotX = got_core | ({int(t) for t in top} & nd_set)
                else:
                    gotX = set(got_core)
                rows_[label]["allf"][ii] = int(len(gotX) == len(nd))
                rows_[label]["newh"][ii] = len(gotX) - len(got_core)
                rows_[label]["expo"][ii] = min(K_q, len(uniq_v))
        log("  %s lane=%s  done %d queries in %.1fs" % (ds, lane, len(qids), time.time() - t_lane))
        out[lane] = {k: {"ALL_REQUIRED_FETCHED": round(float(v["allf"].mean()), 4),
                         "NEW_REQUIRED_FROM_HALO": int(v["newh"].sum()),
                         "MEAN_EXPOSURE_USED": round(float(v["expo"].mean()), 1),
                         "_ind_ALL": v["allf"].tolist()}
                     for k, v in rows_.items()}

    os.makedirs("%s/global_halo" % OUT, exist_ok=True)
    fp = "%s/global_halo/F_%s.json" % (OUT, ds)
    rec = {"ds": ds, "N": N, "npart": npart, "nq": nq, "n_sampled": len(qids),
          "sampled": sample is not None, "CELLS": out}
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


def report():
    rows_out = []
    for ds in HH.DS:
        fp = "%s/global_halo/F_%s.json" % (OUT, ds)
        if not os.path.exists(fp):
            continue
        rec = json.load(open(fp))
        c = rec["CELLS"].get("F6", {})
        if not c:
            continue
        f0, f1, f2 = c["F0"], c["F1"], c["F2"]
        g1, l1, p1 = HH.mcnemar(f0["_ind_ALL"], f1["_ind_ALL"])
        g2, l2, p2 = HH.mcnemar(f0["_ind_ALL"], f2["_ind_ALL"])
        floor = HH.FLOOR.get(ds, 0.0)
        d1, d2 = round(f1["ALL_REQUIRED_FETCHED"] - f0["ALL_REQUIRED_FETCHED"], 4), \
                 round(f2["ALL_REQUIRED_FETCHED"] - f0["ALL_REQUIRED_FETCHED"], 4)
        rows_out.append({
            "ds": ds, "sampled": rec.get("sampled"), "n": rec.get("n_sampled"),
            "F0_ALL_REQUIRED": f0["ALL_REQUIRED_FETCHED"], "F1_ALL_REQUIRED": f1["ALL_REQUIRED_FETCHED"],
            "F2_ALL_REQUIRED": f2["ALL_REQUIRED_FETCHED"],
            "F0_expo": f0["MEAN_EXPOSURE_USED"], "F1_expo": f1["MEAN_EXPOSURE_USED"],
            "F2_expo": f2["MEAN_EXPOSURE_USED"],
            "F1_vs_F0": {"delta": d1, "gained": g1, "lost": l1, "p": p1,
                        "sig": bool(p1 < 0.05 and abs(d1) > floor)},
            "F2_vs_F0": {"delta": d2, "gained": g2, "lost": l2, "p": p2,
                        "sig": bool(p2 < 0.05 and abs(d2) > floor)}})
    json.dump(rows_out, open("%s/global_halo/F_REPORT.json" % OUT, "w"), indent=1)
    for r in rows_out:
        print(json.dumps(r, indent=1))
    return rows_out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        n = int(a[2]) if len(a) > 2 else None
        run(a[1], sample=list(range(n)) if n else None)
    elif a and a[0] == "run_all":
        n = int(a[1]) if len(a) > 1 else None
        for d in HH.DS:
            run(d, sample=list(range(n)) if n else None)
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
