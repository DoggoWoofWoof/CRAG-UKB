"""PART H -- residual headroom: classify every node still missing under the BEST halo config.

BEST halo config from Part F = F2_SUM_NORMALIZED (significant on metaqa/webqsp/hotpotqa_clean,
zero significant regressions -- see global_halo/F_REPORT.json).  For every query that still fails
ALL_REQUIRED_FETCHED under F2 (F6/SAFE lane), every one of its missing required nodes x is
classified into exactly one bucket (interpretive, documented -- the spec named buckets, not a
formal predicate; provably exhaustive + mutually exclusive):

  H0  sanity: x's own partition IS a selected core (should be ~0; anything else is a bug)
  H1  1-hop from a selected core, appears as an F2 candidate, cut by budget, but WOULD be kept
      if the same per-query budget K_q were doubled -- "just outside the budget line"
  H2  1-hop from a selected core, appears as an F2 candidate, but STILL cut even at 2x budget --
      "low-ranked", a modestly bigger budget alone would not fix it
  H3  not 1-hop from any selected core, but some corpus-graph neighbour of x is (in-core or in
      that core's 1-hop halo) -- reachable by a 2nd hop, not visible to the current mechanism
  H4  not 1-hop or 2-hop from any selected core, but has nonzero degree in the union graph --
      connected to the corpus somewhere, just not near the selected cores
  H5  zero degree in STRUCT u NERX u KNN -- genuinely absent from the current graph relations;
      no graph-based mechanism (halo, 2-hop, PPR) could ever reach it

  python scratchpad/_l1hu_h.py run <ds>
  python scratchpad/_l1hu_h.py report
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
BUCKETS = ["H0", "H1", "H2", "H3", "H4", "H5"]
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def run(ds, log=log):
    hard, npart, N = GL.load_core(ds)
    z, meta, C, ctxs, base50, f650, ind_base, PAR = OVE.selected_blocks(ds, hard, npart, log)
    g, gptr, rows, hops = PU.gold_rows(ds)
    nq = meta["n_dev_queries"]
    need = [sorted({int(x) for x in g[gptr[qi]:gptr[qi + 1]]}) for qi in range(nq)]

    pairs, mass = OV.boundary_mass(ds, hard, TAG, "O4_FULL_C", N, npart, log)
    bptr, bidx, bmass = GL.block_mass_csr(pairs, mass, npart, N)
    bmax = GL.block_max_mass(bptr, bmass, npart)
    nptr, nidx = OV.to_node_csr(pairs, npart, N)          # node -> supporting core ids (1-hop halo)

    # F0's own per-query budget K_q -- corpus-level, computed ONCE (matches Part F's pattern exactly;
    # these do NOT depend on the query, only on the fixed beta=0.5 per-core cap)
    f0_pairs = OV.bounded_pairs(pairs, mass, hard, npart, N, BETA_REF)
    f0_nptr_, f0_nidx_ = OV.to_node_csr(f0_pairs, npart, N)
    f0_bptr_, f0_bidx_ = OV.to_block_csr(f0_pairs, npart, N)

    _, S_, K_, X_ = KS.keysets(ds, log)
    uptr, uidx, udeg = KS._csr(np.union1d(np.union1d(S_, K_), X_), N)   # union-graph CSR, all edges
    log("  %s: setup done, N=%d npart=%d nq=%d, entering per-query loop" % (ds, N, npart, nq))

    SEL = f650
    counts = {b: 0 for b in BUCKETS}
    hop_counts = {}
    n_missing_queries = 0
    selmask = np.zeros(npart, bool)
    # PART I: 2-hop oracle -- would UNBOUNDED 2-hop-from-S coverage (on top of F2's actual fetch)
    # close this query's gap?  True iff every currently-missing required node is bucket H3 (the
    # only bucket a 2-hop mechanism could reach that F2 doesn't already reach at 1-hop).
    ind_oracle = np.zeros(nq, np.int8)
    t0 = time.time()
    for qi in range(nq):
        if qi and qi % 200 == 0:
            log("  %s: %d/%d queries, %.1fs elapsed" % (ds, qi, nq, time.time() - t0))
        nd = need[qi]
        if not nd:
            continue
        nd_set = set(nd)
        S = SEL[qi]
        Ss = set(S)
        selmask[:] = False
        selmask[np.asarray(S, np.int64)] = True
        got = {x for x in nd if int(hard[x]) in Ss}

        # ---- gather uncapped F2 candidates from S, score, cut at F0's own K_q (identical to Part F)
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
        cand_score = {}
        if len(V):
            order = np.argsort(V, kind="stable")
            Vs, Ms, Js = V[order], Mm[order], J[order]
            uniq_v, start = np.unique(Vs, return_index=True)
            f2_score = np.add.reduceat(Ms / bmax[Js], start)
            cand_score = dict(zip(uniq_v.tolist(), f2_score.tolist()))

        K_new = np.zeros(0, np.int64)
        if len(f0_bidx_):
            u0 = np.unique(np.concatenate([f0_bidx_[f0_bptr_[j]:f0_bptr_[j + 1]] for j in S]))
            K_new = u0[~selmask[hard[u0.astype(np.int64)]]]
        K_q = len(K_new)

        if K_q and cand_score:
            uniq_v = np.array(list(cand_score.keys()))
            score = np.array(list(cand_score.values()))
            k = min(K_q, len(uniq_v))
            top = uniq_v[np.argpartition(-score, k - 1)[:k]] if k < len(uniq_v) else uniq_v
            got |= ({int(t) for t in top} & nd_set)

        missing = [x for x in nd if x not in got]
        if not missing:
            ind_oracle[qi] = 1
            continue
        n_missing_queries += 1
        h_this = int(np.asarray(hops)[qi]) if ds == "metaqa" and hops is not None else None
        sorted_scores = np.sort(np.fromiter(cand_score.values(), np.float64)) if cand_score else None
        all_h3 = True

        for x in missing:
            if int(hard[x]) in Ss:
                bucket = "H0"
            elif x in cand_score:
                sc = cand_score[x]
                better = len(sorted_scores) - np.searchsorted(sorted_scores, sc, side="right")
                bucket = "H1" if better < 2 * K_q else "H2"
            else:
                nbrs = uidx[uptr[x]:uptr[x + 1]]
                two_hop = False
                for y in nbrs:
                    yi = int(y)
                    if hard[yi] in Ss:
                        two_hop = True; break
                    bs = nptr[yi + 1] - nptr[yi]
                    if bs and Ss.intersection(nidx[nptr[yi]:nptr[yi + 1]].tolist()):
                        two_hop = True; break
                if two_hop:
                    bucket = "H3"
                elif udeg[x] > 0:
                    bucket = "H4"
                else:
                    bucket = "H5"
            if bucket != "H3":
                all_h3 = False
            counts[bucket] += 1
            if h_this is not None:
                hop_counts.setdefault(h_this, {b: 0 for b in BUCKETS})
                hop_counts[h_this][bucket] += 1
        ind_oracle[qi] = int(all_h3)
    log("  %s: %d/%d queries with residual misses, classified in %.1fs" % (
        ds, n_missing_queries, nq, time.time() - t0))

    ok = np.array([bool(x) for x in need])
    oracle_rate = round(float(ind_oracle[ok].mean()), 4)

    os.makedirs("%s/residual" % OUT, exist_ok=True)
    rec = {"ds": ds, "nq": nq, "n_missing_queries": n_missing_queries,
          "total_missing_nodes": sum(counts.values()), "counts": counts,
          "TWO_HOP_ORACLE_ALL_REQUIRED_FETCHED": oracle_rate,
          "_ind_2hop_oracle": ind_oracle.tolist(),
          "by_hop": hop_counts if hop_counts else None}
    json.dump(rec, open("%s/residual/H_%s.json" % (OUT, ds), "w"), indent=1)
    log("wrote %s/residual/H_%s.json" % (OUT, ds))
    return rec


def report():
    f2 = {}
    fp_f = "%s/global_halo/F_REPORT.json" % OUT
    if os.path.exists(fp_f):
        f2 = {r["ds"]: r["F2_ALL_REQUIRED"] for r in json.load(open(fp_f))}
    rows = []
    for ds in HH.DS:
        fp = "%s/residual/H_%s.json" % (OUT, ds)
        if not os.path.exists(fp):
            continue
        rec = json.load(open(fp))
        tot = max(rec["total_missing_nodes"], 1)
        f2_rate = f2.get(ds)
        oracle = rec.get("TWO_HOP_ORACLE_ALL_REQUIRED_FETCHED")
        row = {"ds": ds, "n_missing_queries": rec["n_missing_queries"],
              "total_missing_nodes": rec["total_missing_nodes"],
              "frac": {b: round(rec["counts"][b] / tot, 4) for b in BUCKETS},
              "counts": rec["counts"], "F2_ALL_REQUIRED": f2_rate,
              "TWO_HOP_ORACLE_ALL_REQUIRED_FETCHED": oracle,
              "two_hop_upside": round(oracle - f2_rate, 4) if (oracle is not None and
                                                                f2_rate is not None) else None}
        rows.append(row)
    json.dump(rows, open("%s/residual/H_REPORT.json" % OUT, "w"), indent=1)
    for r in rows:
        print("%-16s missing_nodes=%-6d missing_queries=%-5d  %s" % (
            r["ds"], r["total_missing_nodes"], r["n_missing_queries"],
            "  ".join("%s=%d(%.0f%%)" % (b, r["counts"][b], 100 * r["frac"][b]) for b in BUCKETS)))
        print("  F2=%s  2hop_oracle=%s  upside=%s" % (
            r["F2_ALL_REQUIRED"], r["TWO_HOP_ORACLE_ALL_REQUIRED_FETCHED"], r["two_hop_upside"]))
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
