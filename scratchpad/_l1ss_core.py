"""STRUCTURAL SCORE SEPARABILITY -- the exact read-stage dataset.

`expand_feat` is `_l1bm_core.expand_beam(policy="M0_BASELINE")` line for line, with per-node
provenance accumulators bolted on.  It changes nothing about the search: same per-frontier-node
matvec, same `np.unique` dedup, same `np.maximum.at`, same scope/budget breaks, same `added` cut.
`_l1ss_build.step1` asserts `(added, vmeta)` equal the plain beam on every query before any feature
is trusted, so the dataset describes the frozen read stage and not a lookalike.

Everything recorded is inference-safe: it is a function of the retrieval seeds, the graph, the frozen
embeddings and the cached partition priors.  No gold anywhere.

One structural fact worth knowing before reading the tables: a node enters the frontier at exactly one
hop (`new_mask` excludes anything already in scope), so distinct parent STATES and distinct parent
NODES coincide, and both coincide with the arrival count `cnt`.  `PATH_SUPPORT` and
`DISTINCT_PARENTS` are therefore the same feature on this substrate, not two.

Seeds have no structural ancestor.  For `PARENT_SCORE` they are given 1.0 -- the maximum a cosine can
be -- which says "a retrieval seed is a maximally trusted structural ancestor".  That choice only
touches hop-1 nodes and is reported separately by node hop.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1bm_core as BM

SSD = f"{PP.ROOT}/L1_SCORE_SEPARABILITY"
NEG = BM.NEG
BIG = 1 << 30

# per-node feature columns, in dataset order
FEATS = ["CS_ADMIT", "STATIC_SDIR", "HOP", "PATH_SUPPORT", "DISTINCT_SEED_SUPPORT", "SEED_RANK",
         "PARENT_SCORE", "RR_PARENT_SUPPORT", "PATH_SUM", "PATH_MIN", "FUTURE_MAX", "QSIM", "RSIM",
         "PARTITION_CANONICAL_PRIOR", "PARTITION_RETRIEVAL_PRIOR", "ADMIT_RANK"]
# +1 = larger is better, -1 = smaller is better.  Fixed a priori, not chosen from the results.
SIGN = {"CS_ADMIT": +1, "STATIC_SDIR": +1, "HOP": -1, "PATH_SUPPORT": +1,
        "DISTINCT_SEED_SUPPORT": +1, "SEED_RANK": -1, "PARENT_SCORE": +1, "RR_PARENT_SUPPORT": +1,
        "PATH_SUM": +1, "PATH_MIN": +1, "FUTURE_MAX": +1, "QSIM": +1, "RSIM": +1,
        "PARTITION_CANONICAL_PRIOR": -1, "PARTITION_RETRIEVAL_PRIOR": -1, "ADMIT_RANK": -1}


def workspace(ndocs):
    """preallocated per-node accumulators, reused across queries (reset only where touched)."""
    return {"cs": np.zeros(ndocs, np.float64), "sd": np.zeros(ndocs, np.float64),
            "hop": np.zeros(ndocs, np.int16), "cnt": np.zeros(ndocs, np.int32),
            "psc": np.zeros(ndocs, np.float64), "rr": np.zeros(ndocs, np.float64),
            "psum": np.zeros(ndocs, np.float64), "pmin": np.zeros(ndocs, np.float64),
            "fut": np.zeros(ndocs, np.float64), "ark": np.zeros(ndocs, np.int32),
            "smask": np.zeros(ndocs, np.int64)}


def _rrf_key(cs, cnt, srank, K0=60):
    """STEP 7 beam key: the SAME equal-weight RRF the read stage uses, over the SAME three channels.

    Applied inside the beam it selects which candidates survive the hop; applied over the admitted set
    it orders `added` and therefore the M_struct read.  One semantics at every stage, which is what
    STEP 7 asks for -- unlike the lookahead, which admitted under one key and rejected under another.
    """
    n = len(cs)
    ties = np.arange(n, dtype=np.int64)
    s = np.zeros(n)
    for vals, sign in ((cs, +1.0), (cnt.astype(np.float64), +1.0), (srank.astype(np.float64), -1.0)):
        r = np.argsort(np.lexsort((ties, (-sign) * vals))).astype(np.float64)
        s += 1.0 / (K0 + r)
    return s


def expand_feat(seed_rows, r_q, adjp, adji, deg, Xn, W, M=BM.BEAM, budget=TA.MAX_EDGES_SCORED,
                want_future=True, beam_rank=None, want_visited=False, want_edges=False):
    """(added, vmeta, stats, FE).  FE has one row per admitted non-seed node, in `added` order.

    `beam_rank=None` is a line-for-line replay of the frozen M0 beam.  `beam_rank="R1"` replaces the
    survival key with the STEP 7 RRF above; nothing else changes -- same M, same budget, same hops."""
    scope = set(int(x) for x in seed_rows)
    seeds = np.array([int(x) for x in seed_rows], np.int64)
    F = seeds.copy()
    F_seed = np.arange(len(seeds), dtype=np.int64)
    F_edge = np.ones(len(seeds), np.float64)          # a seed is a maximally trusted ancestor
    F_rank = np.arange(len(seeds), dtype=np.float64)  # seed rank = its RRF position
    F_psum = np.zeros(len(seeds), np.float64)
    F_pmin = np.ones(len(seeds), np.float64)
    # STRUCTURAL-OFFSET TRIPLET PHASE: chain provenance only.  F_tid is written but NEVER read by
    # the search, so parity is preserved by construction.  -1 marks a retrieval seed (no ancestor).
    F_tid = np.full(len(seeds), -1, np.int64)
    EDG, e_base = [], 0
    rq = r_q.astype(np.float32)
    order, vmeta = [], {}
    touched = []
    st = {"edges": 0, "look_edges": 0, "cand_total": 0, "frontier": [], "cands": [], "kept": []}
    for hop in range(TA.MAX_HOPS):
        Ps, Vs, Ss = [], [], []
        for i, e in enumerate(F):                     # per-node loop == frozen numerics
            e = int(e)
            if deg[e] > TA.DEG_CAP:
                continue
            nb = adji[adjp[e]:adjp[e + 1]]
            if len(nb) == 0:
                continue
            st["edges"] += len(nb)
            nb = nb[deg[nb] <= TA.DEG_CAP]
            if len(nb) == 0:
                continue
            delta = Xn[nb] - Xn[e]; nd = np.linalg.norm(delta, axis=1)
            ok = nd > 1e-9
            if not ok.any():
                continue
            Ps.append(np.full(int(ok.sum()), i, np.int64))
            Vs.append(nb[ok]); Ss.append((delta[ok] / nd[ok, None]) @ rq)
        if not Vs:
            break
        PAR = np.concatenate(Ps)
        V = np.concatenate(Vs); S = np.concatenate(Ss).astype(np.float64)

        if want_edges:
            # every ACTUAL directed edge u->v this hop scored, with the offset cosine the search
            # already computed for it (S) and the provenance it already had.  Nothing recomputed.
            EDG.append({"u": F[PAR].astype(np.int64), "v": V.astype(np.int64), "T0_OFFSET": S,
                        "T1_SOURCE": F_edge[PAR].copy(), "T3_HOP": np.full(len(V), hop + 1, np.int16),
                        "seed": F_seed[PAR].astype(np.int64), "parent_rank": F_rank[PAR].copy(),
                        "path_sum": (F_psum[PAR] + S), "path_min": np.minimum(F_pmin[PAR], S),
                        "tid": e_base + np.arange(len(V), dtype=np.int64),
                        "parent_tid": F_tid[PAR].astype(np.int64)})
        uv, inv = np.unique(V, return_inverse=True)
        cnt = np.bincount(inv)
        best_s = np.full(len(uv), NEG); np.maximum.at(best_s, inv, S)

        # ---- provenance, reduced over the arrivals of this hop --------------------------------
        n = len(uv)
        a_psc = np.full(n, NEG); np.maximum.at(a_psc, inv, F_edge[PAR])
        a_rr = np.zeros(n); np.add.at(a_rr, inv, 1.0 / (1.0 + F_rank[PAR]))
        a_sum = np.full(n, NEG); np.maximum.at(a_sum, inv, F_psum[PAR] + S)
        a_min = np.full(n, NEG); np.maximum.at(a_min, inv, np.minimum(F_pmin[PAR], S))
        a_msk = np.zeros(n, np.int64)
        np.bitwise_or.at(a_msk, inv, (np.int64(1) << np.minimum(F_seed[PAR], 62)).astype(np.int64))

        for k in range(n):                            # frozen vmeta semantics, verbatim
            v = int(uv[k]); m = vmeta.get(v)
            if m is None:
                vmeta[v] = [hop + 1, float(best_s[k]), int(cnt[k])]
            else:
                m[2] += int(cnt[k])
                if best_s[k] > m[1]:
                    m[1] = float(best_s[k])
        # accumulators run over EVERY arrival, including re-reaches after admission
        fresh = W["cnt"][uv] == 0
        if fresh.any():
            nu = uv[fresh]
            touched.append(nu)
            W["hop"][nu] = hop + 1
            W["sd"][nu] = NEG; W["psc"][nu] = NEG; W["psum"][nu] = NEG; W["pmin"][nu] = NEG
            W["fut"][nu] = NEG; W["cs"][nu] = NEG; W["ark"][nu] = BIG
        W["cnt"][uv] += cnt.astype(np.int32)
        np.maximum.at(W["sd"], uv, best_s)
        np.maximum.at(W["psc"], uv, a_psc)
        np.maximum.at(W["psum"], uv, a_sum)
        np.maximum.at(W["pmin"], uv, a_min)
        W["rr"][uv] += a_rr
        W["smask"][uv] |= a_msk

        new_mask = np.fromiter((int(v) not in scope for v in uv), bool, len(uv))
        if not new_mask.any():
            break
        cv, cs, cp = uv[new_mask], best_s[new_mask], None
        st["frontier"].append(len(F)); st["cands"].append(len(cv)); st["cand_total"] += len(cv)
        # smallest parent index attaining the max, exactly as the frozen beam records it
        bpar = np.full(len(uv), 1 << 40, np.int64)
        top = S >= best_s[inv]
        np.minimum.at(bpar, inv[top], PAR[top])
        cp = bpar[new_mask]

        if want_future and hop + 1 < TA.MAX_HOPS:
            sc = np.fromiter(scope, np.int64, len(scope))
            fut, raw = BM.future_scores(cv, rq, adjp, adji, deg, Xn, sc)
            st["look_edges"] += raw
            W["fut"][cv] = fut                        # value at the admission decision

        if beam_rank == "R1":
            bk = _rrf_key(cs, W["cnt"][cv], np.array(
                [(int(x) & -int(x)).bit_length() - 1 if x else -1 for x in W["smask"][cv]], np.int64))
            ordr = np.lexsort((np.arange(len(cs), dtype=np.int64), -bk))
        else:
            ordr = np.argsort(-cs, kind="stable")
        keep = ordr[:M]
        st["kept"].append(len(keep))
        W["ark"][cv[keep]] = np.arange(len(keep), dtype=np.int32)
        W["cs"][cv[keep]] = cs[keep]
        for idx in keep:
            v = int(cv[idx])
            if v not in scope:
                scope.add(v); order.append((float(cs[idx]), v))
        F_seed = F_seed[cp[keep]]
        F_edge = cs[keep].astype(np.float64)
        F_rank = np.arange(len(keep), dtype=np.float64)
        F_psum = a_sum[new_mask][keep]
        F_pmin = a_min[new_mask][keep]
        if want_edges:
            # the triplet that CREATED this frontier node: smallest edge index attaining the max.
            # PAR is non-decreasing over the concatenated arrivals, so this is the same arrival the
            # frozen beam already picks for `bpar` -- consistent provenance, no new choice.
            bed = np.full(len(uv), 1 << 40, np.int64)
            np.minimum.at(bed, inv[top], np.flatnonzero(top))
            F_tid = e_base + bed[new_mask][keep]
            e_base += len(V)
        F = cv[keep]
        if len(scope) >= TA.M_MAX * 4:
            break
        # the lookahead is INSTRUMENTATION here, not part of the search, so it must not be able to
        # trip the edge budget and cut the beam short -- that would break parity with the frozen run
        if st["edges"] >= budget:
            break

    added = [v for _, v in sorted(order, key=lambda x: -x[0])][:TA.M_MAX]
    st["scope"] = len(scope)
    # ---- the read-stage rows: every ADMITTED non-seed node, in frozen `added` order ------------
    adm = np.array([v for _, v in sorted(order, key=lambda x: -x[0])], np.int64)
    FE = {"node": adm,
          "CS_ADMIT": W["cs"][adm].copy(),
          "STATIC_SDIR": W["sd"][adm].copy(),
          "HOP": W["hop"][adm].astype(np.float64),
          "PATH_SUPPORT": W["cnt"][adm].astype(np.float64),
          "DISTINCT_SEED_SUPPORT": np.array([bin(int(x)).count("1") for x in W["smask"][adm]],
                                            np.float64),
          "SEED_RANK": np.array([(int(x) & -int(x)).bit_length() - 1 if x else -1
                                 for x in W["smask"][adm]], np.float64),
          "PARENT_SCORE": W["psc"][adm].copy(),
          "RR_PARENT_SUPPORT": W["rr"][adm].copy(),
          "PATH_SUM": W["psum"][adm].copy(),
          "PATH_MIN": W["pmin"][adm].copy(),
          "FUTURE_MAX": W["fut"][adm].copy(),
          "ADMIT_RANK": W["ark"][adm].astype(np.float64)}
    if want_visited:
        # every node the bounded search ever SCORED, with the S4 statistics it already accumulated.
        # The beam throws the pruned ones away; nothing here re-searches, it just keeps them.
        t = np.concatenate(touched) if len(touched) else np.zeros(0, np.int64)
        adm_mask = W["ark"][t] < BIG
        pr = t[~adm_mask]
        po = np.lexsort((pr, -W["sd"][pr]))          # static structural score desc, node id asc
        pr = pr[po]
        FE["VIS_PRUNED"] = pr
        FE["VIS_PRUNED_HOP"] = W["hop"][pr].astype(np.int64).copy()
        FE["VIS_PRUNED_SDIR"] = W["sd"][pr].copy()
        FE["VIS_PRUNED_CNT"] = W["cnt"][pr].astype(np.int64).copy()
        FE["VIS_TOTAL"] = int(len(t))
        # extra provenance for the PARTITION CALIBRATION audit.  Every one of these accumulators is
        # updated over ALL arrivals (above, before the prune), so the pruned rows are as valid as
        # the admitted ones.  W["cs"]/W["ark"] are kept-only and are deliberately NOT exported here.
        FE["VIS_PRUNED_SMASK"] = W["smask"][pr].copy()
        FE["VIS_PRUNED_PSC"] = W["psc"][pr].copy()
        FE["VIS_PRUNED_RR"] = W["rr"][pr].copy()
        FE["VIS_PRUNED_PSUM"] = W["psum"][pr].copy()
        FE["VIS_PRUNED_PMIN"] = W["pmin"][pr].copy()
        FE["ADM_SMASK"] = W["smask"][adm].copy()
    if want_edges:
        FE["EDGES"] = ({k: np.concatenate([e[k] for e in EDG]) for k in EDG[0]} if EDG else
                       {k: np.zeros(0, np.int64) for k in
                        ("u", "v", "T0_OFFSET", "T1_SOURCE", "T3_HOP", "seed", "parent_rank",
                         "path_sum", "path_min", "tid", "parent_tid")})
    if len(touched):
        t = np.concatenate(touched)
        for k in W:
            W[k][t] = 0
    return added, vmeta, st, FE
