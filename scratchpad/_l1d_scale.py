"""L1 DEVELOPMENT -- SCALE: does IR_L1 + ES.KNEE_SDIV scale as the number of partitions grows?  (user 2026-09-29, a forwarded
direction after REPORT section 34: "I would NOT go to the final held-out evaluation yet ... Does this architecture actually scale
as the number of partitions grows? ... construct multiple partition granularities, for example: K in {100,250,500,1000,2000,5000}
where feasible. Keep the algorithm identical: IR_L1 + ES.KNEE_SDIV. Then measure three things. 1. Retrieval quality R_ALL(B_N,K)
... 2. Absolute fan-out B_P(q,K) ... 3. Relative fan-out rho(q,K)=B_P(q,K)/K ... At each K, compare at least: balanced random
partition versus PHG/H4_SK structural partition.")  Development numbers; any p-value is descriptive.

Fixed, exactly as route v1 / v2 (asserted per query on the served cells): the L1a node score IR_L1, the served order O (IR_L1
FLAT+LOC, RRF K0 = 60), the router's own order O' = RRF(H_q, LOC) (K0 = 60; ties -> seeds by FLAT rank, then LOC rank), the
population, and the serving rule (the first B_N nodes of O inside the B_P contacted partitions).  The L1b rule is the carried-forward
development candidate, unchanged:
  ES         partitions by first appearance in O' (partitions without evidence last, in static id order)
  KNEE_SDIV  B_P(q) = #{SDIV(P) >= the mean of the positive SDIV}, SDIV(P) = the sum of 1/rank(s) over the seeds s in H_q =
             FLAT[:200] with IR_L1 1-hop evidence in P; a query without evidence contacts every partition.
Reference arm: S.KNEE_SDIV (first appearance in O, which needs the global order; the same count).
Only the shard map varies.  A cell = (partitioner, K), K in {100, 250, 500, 1000, 2000, 5000} plus the native K = N // 100:
  PHG    the H4_SK hypergraph at K (the frozen rule, cap = round(N / K); scratchpad/_l1d_scale_parts.py HG) partitioned by the
         validated Zoltan-PHG substitute (scratchpad/_l1c_phg.py unchanged: NP 4, IMBALANCE_TOL 1.03, CONNECTIVITY, then
         PHG_NONEMPTY_REPAIR_V1); at the native K the served PHG cell (the rebuild must equal it)
  MTK    the same hypergraph partitioned by the frozen Mt-KaHyPar contract (DETERMINISTIC_QUALITY, KM1, eps 0.03, seed 0) where
         it ran inside the memory cap; a FAILED cell is absent, never replaced; at the native K the served H4_SK cell; MetaQA
         and SQuAD only (MuSiQue's native Mt-KaHyPar run is FAILED_MEMORY_CAP)
  RAND<s> balanced random, s in {0, 1, 2}: perm = numpy default_rng(s).permutation(N), P(perm[i]) = i mod K (sizes floor or
         ceil of N / K)
Measured per cell:
  R_ALL(B_N) of ES.KNEE_SDIV and S.KNEE_SDIV (ALL gold served, B_N in M_CURVE): vs unrouted, paired, lost to reach / crowding
  B_P(q); rho(q) = B_P(q) / K; CMASS(q) = the nodes inside the contacted partitions (the local scan)
  the fixed-fan-out surfaces ALL(b, B_N) of ES and S (best fixed b; the smallest b within 0.01 of / at or above unrouted)
  lossless references: NGP = the distinct gold partitions (gold-dependent: the fewest shards that hold the evidence); LO = the
  ES / S position of the last gold partition; PSTAR_M = the distinct partitions of O[:M] (S at PSTAR_M == unrouted, asserted);
  ESTAR_M = #{partitions whose first O' position is < M} (route v3's router-side estimate: every partition once |O'| < M);
  ES|ESTAR_M is a B_N-coupled diagnostic arm
  communication proxies: NSEEDP = the distinct partitions of H_q (the shards owning the seeds' rows, if node rows are sharded
  with their owners); XC = the seeds' 1-hop entries whose target lies outside the seed's partition (of ENT entries); NEV = the
  partitions holding evidence; the router's reads for the 200 seeds (node rows RDN vs the union partition sketch RDS)
  static: route v3's router_state (per family: node-row entries, sketch entries, co-partitioned share) and the cross-partition
  share of all entries of E
  load: the per-partition contacts of ES.KNEE_SDIV, S.KNEE_SDIV, ES|ESTAR@1000 and S|PSTAR@1000 (route v3's load_stats)
  structural vs random at the same K: paired ALL of ES.KNEE_SDIV (descriptive McNemar)
Identities asserted: FLAT == loc v1 (runner); IR_L1 LOC / FLAT+LOC positions == node1h v1 on every gold node; on the served
cells: the ES LO / HI / PRG, the KNEE_SDIV, NSEEDP and NEV counts == route v2, the S LO / HI / PRG and PSTAR == route v1, ESTAR ==
route v3 (per query), and on the full population the route v3 load of the four load arms; on every cell: every partition
contacted == unrouted (per row, per ranking), S at PSTAR_M == unrouted (every M), PSTAR_M == #(first O rank < M) (first rows),
the per-seed masses sum to SUM (first rows), every ranking a permutation (first rows); every scale file's sha256 == its build
record; the native-K PHG / MTK rebuilds == the served cells.

Usage: python scratchpad/_l1d_scale.py RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L1_DEV/scale_<dataset>__<tag>.{json,npz} (write-once)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_arms as R
import _l1d_edgediag as E
import _l1d_node1h as NH
import _l1d_adaptbp as AB
import _l1d_route as RT
import _l1d_route3 as RT3
import _l1d_scale_parts as SP

log = D.log
K0, ACT = D.K0, D.ACT
FAMS = NH.FAMS
NODE = "IR_L1"
KGRID = SP.KGRID
RAND_SEEDS = (0, 1, 2)
STRUCTURAL = ("PHG", "MTK")
SERVED = {"metaqa": {"metaqa": "MTK", "metaqa_phg": "PHG"}, "squad": {"squad": "MTK", "squad_phg": "PHG"},
          "musique": {"musique": "PHG"}}
RANKS = ("ES", "S")
COUNT = "KNEE_SDIV"
M_LOAD = 1000
MI_LOAD = D.M_CURVE.index(M_LOAD)
LOAD_ARMS = ("ES|KNEE_SDIV", "S|KNEE_SDIV", "ES|ESTAR_BN@%d" % M_LOAD, "S|PSTAR_BN@%d" % M_LOAD)
RT3_LOAD = {"ES|KNEE_SDIV": "ES|KNEE_SDIV", "S|KNEE_SDIV": "S|KNEE_SDIV", "ES|ESTAR_BN@%d" % M_LOAD: "ES|ESTAR_BN@%d" % M_LOAD,
            "S|PSTAR_BN@%d" % M_LOAD: "S|PSTAR_BN@%d" % M_LOAD}
BP_SHOW = tuple(AB.BP_SHOW) + (700, 1000, 1500, 2000, 3000, 5000)
MSA = np.asarray(D.M_CURVE, np.int64)
CHECK_ROWS = RT.CHECK_ROWS
PATHS = {"route": os.path.join(D.HERE, "_l1d_route.py"), "route2": os.path.join(D.HERE, "_l1d_route2.py"),
         "route3": os.path.join(D.HERE, "_l1d_route3.py"), "scale_parts": os.path.join(D.HERE, "_l1d_scale_parts.py")}
assert FAMS == RT.FAMS == RT3.FAMS and set(RT3_LOAD.values()) <= set("%s|%s" % x for x in RT3.LOAD_ARMS)
assert COUNT in RT3.REF_COUNTS and set(RANKS) <= set(RT3.REF_RANKS)


def sha_bytes(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rand_hard(N, K, seed):
    """balanced random single-owner partition: P(perm[i]) = i mod K, perm = default_rng(seed).permutation(N)."""
    perm = np.random.default_rng(seed).permutation(N)
    hard = np.empty(N, np.int64)
    hard[perm] = np.arange(N, dtype=np.int64) % K
    return hard


class HardPart(object):
    """a cell's hard single-owner partition from a vector (the fields of _l1d_lib.Part, same arithmetic)."""

    def __init__(self, hard, tag, path=None):
        self.tag, self.path = tag, path
        self.hard = np.asarray(hard, np.int64)
        assert self.hard.min() >= 0
        self.npart = int(self.hard.max()) + 1
        self.sizes = np.bincount(self.hard, minlength=self.npart).astype(np.int64)
        assert (self.sizes >= 1).all(), "%s: empty partitions" % tag
        self.order_nodes = np.argsort(self.hard, kind="stable")
        self.ptr = np.zeros(self.npart + 1, np.int64)
        self.ptr[1:] = np.cumsum(self.sizes)


def cell_name(pz, K):
    return "%s_k%d" % (pz, K)


def build_cells(cd, served_parts):
    """(ordered {cell: HardPart}, {cell: record}, {cell: absent record}, native K)."""
    ds, N = cd.name, int(cd.n_nodes)
    kn = SP.HG.frozen_k(N)
    Ks = sorted(set(KGRID) | {kn})
    by_pz = {pz: c for c, pz in SERVED[ds].items()}
    parts, recs, absent = {}, {}, {}
    for K in Ks:
        for pz in STRUCTURAL:
            if pz == "MTK" and ds not in SP.MTK_SETS:
                continue
            name = cell_name(pz, K)
            fp = SP.phg_npy(ds, K) if pz == "PHG" else SP.mtk_npy(ds, K)
            side = fp[:-4] + (".SCALE.json" if pz == "PHG" else ".RUN.json")
            failp = fp[:-4] + ".FAILED.json"
            hgj = SP.hg_npz(ds, K)[:-4] + ".json"
            hgm = json.load(open(hgj, encoding="utf-8"))
            assert D.sha_file(SP.hg_npz(ds, K)) == hgm["file_sha256"], "hypergraph %s changed since its record" % hgj
            rec = {"partitioner": pz, "K": K, "native": K == kn,
                   "hypergraph": {"file": D.rel(SP.hg_npz(ds, K)), "sha256": hgm["file_sha256"], "content_digest": hgm["content_digest"],
                                  "cap": hgm["cap"], "hyperedges": hgm["hyperedges"], "pins": hgm["pins"],
                                  "content_equals_frozen_H4_SK": hgm["scale"]["content_equals_frozen_H4_SK"]}}
            if K == kn:
                P = served_parts[by_pz[pz]]
                rec.update({"source": "served cell %s (partition %s)" % (by_pz[pz], P.tag), "file": D.rel(P.path), "sha256": D.sha_file(P.path)})
                if os.path.exists(side):
                    rb = np.load(fp).astype(np.int64)
                    assert np.array_equal(rb, P.hard), "%s: the native-K rebuild differs from the served cell" % name
                    rec["scale_rebuild"] = {"file": D.rel(fp), "sha256": D.sha_file(fp), "record": D.rel(side), "equals_served": True}
                elif os.path.exists(failp):
                    rec["scale_rebuild"] = {"record": D.rel(failp), "status": json.load(open(failp, encoding="utf-8"))["STATUS"]}
                else:
                    rec["scale_rebuild"] = None
            else:
                if os.path.exists(failp) and not os.path.exists(side):
                    fr = json.load(open(failp, encoding="utf-8"))
                    absent[name] = {"partitioner": pz, "K": K, "status": fr["STATUS"], "record": D.rel(failp), "sha256": D.sha_file(failp),
                                    "note": "absent from the factorial (never replaced by another partitioner)"}
                    if pz == "MTK":
                        absent[name].update({"rss_cap_gb": fr["guard"].get("rss_cap_gb"), "peak_rss_kb": fr["guard"].get("peak_rss_kb"),
                                             "previous_attempts": fr.get("previous_attempts", [])})
                    continue
                if not os.path.exists(side) and R.argv_opts()[1] is not None:
                    absent[name] = {"partitioner": pz, "K": K, "status": "NOT_BUILT (smoke run only; a full run requires the cell)"}
                    continue
                assert os.path.exists(side), "%s %s: not built (run scratchpad/_l1d_scale_parts.py %s %s %d)" % (ds, name, pz, ds, K)
                sr = json.load(open(side, encoding="utf-8"))
                if pz == "MTK":
                    assert sr["STATUS"] == "OK" and sr["post_checks"]["length_ok"], side
                    if not sr["post_checks"]["every_block_used"]:
                        absent[name] = {"partitioner": pz, "K": K, "status": "EMPTY_BLOCKS (%d of %d used)" % (sr["post_checks"]["blocks_used"], K),
                                        "record": D.rel(side), "sha256": D.sha_file(side),
                                        "note": "not K shards; absent from the factorial (the PHG lane's non-empty repair is not part of the "
                                                "Mt-KaHyPar contract, so it is not applied)"}
                        continue
                assert D.sha_file(fp) == sr["output"]["sha256"], "%s changed since its record" % fp
                P = HardPart(np.load(fp), name, fp)
                rec.update({"source": "scale build", "file": D.rel(fp), "sha256": sr["output"]["sha256"], "record": D.rel(side),
                            "record_sha256": D.sha_file(side)})
                if pz == "PHG":
                    rec["repair"] = sr.get("repair")
                    rec["balance"] = sr["balance"]
                else:
                    rec["worker_stats_km1"] = sr["worker_stats"].get("objective_km1")
                    rec["post_checks"] = sr["post_checks"]
                    rec["guard"] = {k: sr["guard"].get(k) for k in ("rss_cap_gb", "peak_rss_kb", "wall_seconds")}
                    rec["previous_attempts"] = sr.get("previous_attempts", [])
            assert P.npart == K and len(P.hard) == N, (name, P.npart, K)
            parts[name], recs[name] = P, rec
        for s in RAND_SEEDS:
            name = cell_name("RAND%d" % s, K)
            h = rand_hard(N, K, s)
            P = HardPart(h, name)
            assert P.npart == K and P.sizes.min() == N // K and P.sizes.max() == -(-N // K)
            parts[name] = P
            recs[name] = {"partitioner": "RAND", "seed": s, "K": K, "native": K == kn, "source": "built in memory",
                          "rule": "perm = numpy.random.default_rng(%d).permutation(N); P(perm[i]) = i mod K" % s,
                          "sha256_int64_vector": sha_bytes(h), "numpy": np.__version__}
    return parts, recs, absent, kn


class ScaleSpec(object):
    def __init__(self, cd, pop, parts_served):
        self.ds = ds = cd.name
        N = self.N = int(cd.n_nodes)
        nq, ng = pop.nq, pop.ng_tot
        self.nq_full = None
        # ---- the route v1 / v2 / v3 records whose arithmetic this harness reuses (code and records pinned)
        self.REC = {}
        for key in ("route", "route2", "route3"):
            fp = os.path.join(D.OUT, "%s_%s__v1.json" % (key, ds))
            rec = json.load(open(fp, encoding="utf-8"))
            npz = os.path.join(D.OUT, rec["npz"]["path"])
            assert D.sha_file(npz) == rec["npz"]["sha256"], "%s npz changed" % key
            assert D.sha_file(PATHS[key]) == rec["code"]["harness"]["sha256"], "%s differs from the code of its v1 record" % key
            assert rec["code"]["arms"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_arms.py"))
            assert rec["code"]["lib"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))
            self.REC[key] = {"json": fp, "json_sha256": D.sha_file(fp), "npz": npz, "npz_sha256": rec["npz"]["sha256"], "n_rows": rec["n_rows"]}
            if key == "route3":
                for k_, p_ in rec["structures"]["imports"].items():
                    assert D.sha_file(os.path.join(D.REPO, k_)) == p_, "%s differs from route v3's import" % k_
                self.imports = dict(rec["structures"]["imports"], **{"scratchpad/_l1d_route3.py": rec["code"]["harness"]["sha256"]})
                fam_ref = rec["structures"]["families"]
            if key == "route":
                n1 = rec["structures"]["reproduces"]["node1h"]
                assert D.sha_file(os.path.join(D.REPO, n1["path"])) == n1["sha256"]
                self.node1h_npz = os.path.join(D.OUT, json.load(open(os.path.join(D.REPO, n1["path"]), encoding="utf-8"))["npz"]["path"])
                assert D.sha_file(self.node1h_npz) == n1["npz_sha256"]
        self.sha_code = {k: D.sha_file(p) for k, p in PATHS.items()}
        t0 = time.time()
        self.F, famrec = NH.build_families(cd, N)
        t_fam = time.time() - t0
        assert famrec == fam_ref, "families differ from route v3"
        # ---- the cells
        t0 = time.time()
        self.parts, self.cellrec, self.absent, self.kn = build_cells(cd, parts_served)
        self.cells = list(self.parts)
        self.served_map = {c: cell_name(pz, self.kn) for c, pz in SERVED[ds].items()}
        t_cells = time.time() - t0
        # ---- static router state and cross-partition share per cell (route v3's router_state)
        t0 = time.time()
        self.UNILEN, self.ROWLEN = {}, None
        for c, P in self.parts.items():
            rs, rowlen, _, unilen = RT3.router_state(self.F, P.hard, P.npart, N)
            self.ROWLEN = rowlen if self.ROWLEN is None else self.ROWLEN
            assert (rowlen == self.ROWLEN).all()
            self.UNILEN[c] = unilen
            ent = sum(rs[f]["node_row_entries"] for f in FAMS)
            co = sum(rs[f]["co_partitioned_entries (inside the source's own partition)"] for f in FAMS)
            self.cellrec[c]["router_state"] = rs
            self.cellrec[c]["cross_partition_share_of_E_entries"] = D.q4(1.0 - co / float(ent))
            self.cellrec[c]["partition_size"] = D.stats(P.sizes)
        t_rs = time.time() - t0
        self.names = [NODE]
        self.hard_ref = {NODE: None}
        self.v1_hard = {}
        self.lat = {NODE: []}
        self.FUSE_LAT = []
        self.TL = {c: [] for c in self.cells}
        self.FRK = np.zeros(ng, np.int64)
        nr, nm = len(RANKS), len(D.M_CURVE)
        C_ = self.cells
        self.LO = {c: np.zeros((nq, nr), np.int32) for c in C_}
        self.HI = {c: np.zeros((nq, nr, nm), np.int32) for c in C_}
        self.PRG = {c: np.zeros((ng, nr), np.int32) for c in C_}
        self.CNT = {c: np.zeros(nq, np.int32) for c in C_}
        self.CMASS = {c: np.zeros((nq, nr), np.int64) for c in C_}
        self.PSTAR = {c: np.zeros((nq, nm), np.int32) for c in C_}
        self.ESTAR = {c: np.zeros((nq, nm), np.int32) for c in C_}
        self.NGP = {c: np.zeros(nq, np.int32) for c in C_}
        self.NSEEDP = {c: np.zeros(nq, np.int32) for c in C_}
        self.NEV = {c: np.zeros(nq, np.int32) for c in C_}
        self.XC = {c: np.zeros(nq, np.int64) for c in C_}
        self.RDS = {c: np.zeros(nq, np.int64) for c in C_}
        self.MSUM = {c: np.zeros((nr, P.npart)) for c, P in self.parts.items()}
        self.LOAD = {c: np.zeros((len(LOAD_ARMS), P.npart), np.int64) for c, P in self.parts.items()}
        self.NOEV = {c: 0 for c in C_}
        self.ENT = np.zeros(nq, np.int64)
        self.RDN = np.zeros(nq, np.int64)
        self.OQLEN = np.zeros(nq, np.int64)
        self.checks = {c: 0 for c in C_}
        self.record = {
            "node_score": "IR_L1 (sections 32-34, recomputed; asserted identical through node1h v1 and route v1 / v2 / v3)",
            "served_order": "IR_L1 FLAT+LOC (RRF K0 = 60); the first B_N nodes of it inside the contacted partitions",
            "L1b_rule": "ES ranking + KNEE_SDIV count (route v2 arithmetic, unchanged)", "reference_arm": "S ranking + KNEE_SDIV count",
            "rankings": list(RANKS), "count": COUNT, "load_arms": list(LOAD_ARMS),
            "K_grid": list(KGRID), "native_K": self.kn, "random_seeds": list(RAND_SEEDS),
            "served_cells": self.served_map, "cells": self.cellrec, "absent_cells": self.absent,
            "families": famrec, "seconds_families": round(t_fam, 1), "seconds_cells": round(t_cells, 1),
            "seconds_router_state": round(t_rs, 1),
            "reproduces": {k: {"path": D.rel(v["json"]), "sha256": v["json_sha256"], "npz_sha256": v["npz_sha256"]} for k, v in self.REC.items()},
            "imports": dict(self.imports, **{"scratchpad/_l1d_scale_parts.py": self.sha_code["scale_parts"]})}
        log("SCALE %s: %d cells (%s), absent %s, native K %d" % (ds, len(self.cells), ", ".join(self.cells), sorted(self.absent), self.kn))

    def row(self, j, of, od, os_, npos, fv, frank, g, sl):
        N = self.N
        top = of[:ACT]
        nh = len(top)
        # ---- L1a: the IR_L1 node score (the arithmetic of _l1d_route / _l1d_route2 / _l1d_route3 for this rung)
        t0 = time.perf_counter()
        U, H, GW = [], [], []
        for f in FAMS:
            Fm = self.F[f]
            st = Fm["xadj"][top]
            hit, pos = E.entries(st, Fm["xadj"][top + 1] - st)
            U.append(Fm["adj"][pos].astype(np.int64))
            H.append(hit)
            GW.append(Fm["g"][top][hit])
        u, hit, gw = np.concatenate(U), np.concatenate(H), np.concatenate(GW)
        x = (1.0 / np.arange(1.0, nh + 1.0))[hit]
        x = x * gw
        L = np.bincount(u, weights=x, minlength=N)
        Lord = D.order_from_score(L, frank)
        self.lat[NODE].append(time.perf_counter() - t0)
        # ---- the served order O = IR_L1 FLAT+LOC
        t0 = time.perf_counter()
        f = 1.0 / (K0 + frank.astype(np.float64))
        f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        FO = np.lexsort((frank, -f))
        frk = np.empty(N, np.int64)
        frk[FO] = np.arange(N)
        self.FUSE_LAT.append(time.perf_counter() - t0)
        pg = frk[g]
        self.FRK[sl] = pg
        lim = int(pg.max()) + 1
        # ---- the router's own fused order O' = RRF(H_q, LOC) over the evidence set V_q = H_q u {L > 0}
        vq = np.union1d(top, Lord)
        fq = np.zeros(len(vq))
        tq = np.zeros(len(vq))
        fq[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
        il = np.searchsorted(vq, Lord)
        fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        tq[il] = ACT + np.arange(len(Lord))
        tq[np.searchsorted(vq, top)] = np.arange(nh)
        oq = vq[np.lexsort((tq, -fq))]
        self.OQLEN[j] = len(oq)
        self.ENT[j] = len(u)
        self.RDN[j] = int(self.ROWLEN[top].sum())
        wseed = 1.0 / np.arange(1.0, nh + 1.0)
        chk = j < CHECK_ROWS
        if chk:
            nz = np.flatnonzero(L)
            Lv = L[nz]
        for c in self.cells:
            t1 = time.perf_counter()
            P = self.parts[c]
            npart, hard = P.npart, P.hard
            fa = np.minimum.reduceat(frk[P.order_nodes], P.ptr[:-1])
            hs = hard[top]
            hu = hard[u]
            Cm = np.bincount(hit * npart + hu, weights=x, minlength=nh * npart).reshape(nh, npart)
            if chk:
                assert np.allclose(Cm.sum(0), np.bincount(hard[nz], weights=Lv, minlength=npart), rtol=1e-9, atol=1e-15), \
                    "per-seed masses do not sum to SUM (%s)" % c
            sdiv = wseed @ (Cm > 0)
            hq = hard[oq]
            up, first = np.unique(hq, return_index=True)
            fe = np.full(npart, len(oq), np.int64)
            fe[up] = first
            RO = {"ES": np.lexsort((np.arange(npart), fe)), "S": np.argsort(fa, kind="stable")}
            kn_, npz_ = RT.knee(sdiv)
            b = kn_ if npz_ else npart
            if not npz_:
                self.NOEV[c] += 1
            assert 1 <= b <= npart
            self.CNT[c][j] = b
            self.NSEEDP[c][j] = len(np.unique(hs))
            self.NEV[c][j] = len(up)
            self.XC[c][j] = int((hu != hs[hit]).sum())
            self.RDS[c][j] = int(self.UNILEN[c][top].sum())
            hg = hard[g]
            self.NGP[c][j] = len(np.unique(hg))
            for mi, M in enumerate(D.M_CURVE):
                ps_ = int((fa < M).sum())
                if chk:
                    assert ps_ == len(np.unique(hard[FO[:M]])), "PSTAR != #(first O rank < M) (%s %d)" % (c, M)
                self.PSTAR[c][j, mi] = ps_
                self.ESTAR[c][j, mi] = int((fe < M).sum())
            for ai, (rk, bb) in enumerate(((RO["ES"], b), (RO["S"], b), (RO["ES"], int(self.ESTAR[c][j, MI_LOAD])),
                                           (RO["S"], int(self.PSTAR[c][j, MI_LOAD])))):
                self.LOAD[c][ai, rk[:bb]] += 1
            # ---- evaluation (gold-dependent; not part of any rule): every fan-out of both rankings (route v2 arithmetic)
            hO = hard[FO[:lim]]
            for ri, rk in enumerate(RANKS):
                ro = RO[rk]
                if chk:
                    assert len(ro) == npart and (np.sort(ro) == np.arange(npart)).all(), "not a permutation (%s %s)" % (c, rk)
                pr = RT.inv(ro)
                prg = pr[hg]
                C = AB.cumcounts(pr[hO], pg, npart)
                assert (C[:, npart - 1] - 1 == pg).all(), "every partition contacted != unrouted (%s %s)" % (c, rk)
                ms = np.cumsum(P.sizes[ro])
                lo = int(prg.max()) + 1
                qmax = (C[:, lo - 1:] - 1).max(axis=0)
                assert (np.diff(qmax) >= 0).all()
                self.LO[c][j, ri] = lo
                self.HI[c][j, ri] = lo - 1 + np.searchsorted(qmax, MSA, side="left")
                self.PRG[c][sl, ri] = prg
                self.MSUM[c][ri] += ms
                self.CMASS[c][j, ri] = ms[b - 1]
            self.TL[c].append(time.perf_counter() - t1)
            if chk:
                self.checks[c] += 1
        return {NODE: Lord}

    def arm(self, c, ri, b, unr, SFr, hop_m):
        LO, HI = self.LO[c].astype(np.int64), self.HI[c].astype(np.int64)
        out = {}
        for mi, M in enumerate(D.M_CURVE):
            a_ = AB.served_at(LO[:, ri], HI[:, ri, mi], b)
            row_ = SFr[mi]
            bm = float(b.mean())
            bl, bh = int(np.floor(bm)), int(np.ceil(bm))
            e = {"ALL": D.q4(a_.mean()), "n_ALL": int(a_.sum()), "delta_vs_unrouted": D.q4(a_.mean() - unr[M].mean()),
                 "paired_vs_unrouted": D.paired(unr[M], a_),
                 "same_ranking_fixed_at_mean_fan_out_interpolated": D.q4(row_[bl - 1] + (bm - bl) * (row_[bh - 1] - row_[bl - 1])),
                 "lost_reach (B_P(q) < LO)": D.q4((b < LO[:, ri]).mean()),
                 "lost_crowding (LO <= HI < B_P(q))": D.q4(((LO[:, ri] <= HI[:, ri, mi]) & (HI[:, ri, mi] < b)).mean()),
                 "unservable_at_any_B_P (LO > HI)": D.q4((LO[:, ri] > HI[:, ri, mi]).mean())}
            if hop_m:
                e["per_hop"] = {hk: {"n": int(qm.sum()), "ALL": D.q4(a_[qm].mean()), "unrouted_ALL": D.q4(unr[M][qm].mean())}
                                for hk, qm in hop_m.items()}
            out[str(M)] = e
        return out

    def finish(self, ctx):
        POS, gptr, ngold, pop, ST = ctx["POS"], ctx["gptr"], ctx["ngold"], ctx["pop"], ctx["ST"]
        nq, ng, N = pop.nq, pop.ng_tot, self.N
        assert {k: D.sha_file(p) for k, p in PATHS.items()} == self.sha_code, "imported code changed during the run"
        allv = lambda pos, M: D.per_query(pos, M, gptr, ngold)[0]
        diag, arrays = {}, {}
        # (0) identities against node1h v1 and route v1 / v2 / v3 on the served cells
        z = np.load(self.node1h_npz)
        assert (z["rows"][:nq] == pop.rows).all() and (z["gptr"][:nq + 1] == gptr).all()
        assert (z["pos_LOC__" + NODE][:ng] == POS[NODE]["LOC"]).all(), "IR_L1 LOC positions differ from node1h v1"
        assert (z["pos_FLATLOC__" + NODE][:ng] == POS[NODE]["FLAT+LOC"]).all(), "IR_L1 FLAT+LOC positions differ from node1h v1"
        assert (self.FRK == POS[NODE]["FLAT+LOC"]).all()
        Z = {k: np.load(v["npz"]) for k, v in self.REC.items()}
        for k, zz in Z.items():
            assert (zz["rows"][:nq] == pop.rows).all() and (zz["gptr"][:nq + 1] == gptr).all(), k
        r1, r2, r3 = ([str(s) for s in Z[k]["ranks"]] for k in ("route", "route2", "route3"))
        c2 = [str(s) for s in Z["route2"]["counts"]]
        la3 = [str(s) for s in Z["route3"]["load_arms"]]
        full = nq == self.REC["route3"]["n_rows"]
        ies, iss = RANKS.index("ES"), RANKS.index("S")
        for sc_, c in self.served_map.items():
            z1, z2, z3 = Z["route"], Z["route2"], Z["route3"]
            for nm_, a_ in (("LO", self.LO[c][:, ies]), ("HI", self.HI[c][:, ies])):
                assert (z2["%s__%s" % (nm_, sc_)][:nq, r2.index("ES")] == a_).all(), "ES %s differs from route v2 (%s)" % (nm_, c)
            assert (z2["PRG__" + sc_][:ng, r2.index("ES")] == self.PRG[c][:, ies]).all(), "ES PRG differs from route v2 (%s)" % c
            for cn, a_ in ((COUNT, self.CNT[c]), ("NSEEDP", self.NSEEDP[c]), ("NEV", self.NEV[c])):
                assert (z2["CNT__" + sc_][:nq, c2.index(cn)] == a_).all(), "%s differs from route v2 (%s)" % (cn, c)
            for nm_, a_ in (("LO", self.LO[c][:, iss]), ("HI", self.HI[c][:, iss])):
                assert (z1["%s__%s" % (nm_, sc_)][:nq, r1.index("S")] == a_).all(), "S %s differs from route v1 (%s)" % (nm_, c)
            assert (z1["PRG__" + sc_][:ng, r1.index("S")] == self.PRG[c][:, iss]).all(), "S PRG differs from route v1 (%s)" % c
            assert (z1["PSTAR__" + sc_][:nq] == self.PSTAR[c]).all(), "PSTAR differs from route v1 (%s)" % c
            assert (z3["ESTAR__" + sc_][:nq] == self.ESTAR[c]).all(), "ESTAR differs from route v3 (%s)" % c
            assert (z3["LO__" + sc_][:nq, r3.index("ES")] == self.LO[c][:, ies]).all()
            if full:
                for ai, a in enumerate(LOAD_ARMS):
                    assert (z3["LOAD__" + sc_][la3.index(RT3_LOAD[a])] == self.LOAD[c][ai]).all(), "load differs from route v3 (%s %s)" % (c, a)
        unr = {M: allv(POS[NODE]["FLAT+LOC"], M) for M in D.M_CURVE}
        for c in self.cells:
            for mi, M in enumerate(D.M_CURVE):
                assert (AB.served_at(self.LO[c][:, iss], self.HI[c][:, iss, mi], self.PSTAR[c][:, mi]) == unr[M]).all(), \
                    "S at PSTAR != unrouted (%s %d)" % (c, M)
        diag["identity"] = ("IR_L1 LOC and FLAT+LOC positions == node1h v1 on all %d gold nodes; on the served cells %s: ES LO / HI / "
                            "PRG and the counts %s / NSEEDP / NEV == route v2, S LO / HI / PRG and PSTAR == route v1, ESTAR == route v3%s; "
                            "every cell: every partition contacted == unrouted (per row, per ranking), S at PSTAR_M == unrouted at every "
                            "M; first %d rows of every cell: PSTAR_M == #(first O rank < M), per-seed masses sum to SUM, every ranking a "
                            "permutation" % (ng, self.served_map, COUNT, ", and the load of %s (full population)" % list(LOAD_ARMS) if full else
                                             " (load not compared: first rows only)", CHECK_ROWS))
        diag["checked_rows"] = self.checks
        hop_m = ST.get("per_hop", {})
        diag["unrouted_ALL"] = {str(M): {"ALL": D.q4(unr[M].mean()), "n_ALL": int(unr[M].sum())} for M in D.M_CURVE}
        diag["query_side (cell-independent)"] = {"seed_1hop_entries ENT": AB.dist(self.ENT), "router_node_row_reads RDN": AB.dist(self.RDN),
                                                 "router_order_length |O'|": AB.dist(self.OQLEN)}
        cells_out = {}
        SERV = {}
        for c in self.cells:
            P = self.parts[c]
            npart = P.npart
            LO, HI = self.LO[c].astype(np.int64), self.HI[c].astype(np.int64)
            b = self.CNT[c].astype(np.int64)
            e_c = {"partitioner": self.cellrec[c]["partitioner"], "K": npart, "native": self.cellrec[c]["native"]}
            SF = {rk: AB.surface(LO[:, ri], HI[:, ri], npart) for ri, rk in enumerate(RANKS)}
            e_c["B_P(q) = KNEE_SDIV"] = AB.dist(b)
            e_c["rho(q) = B_P / K"] = AB.dist(b / float(npart))
            e_c["queries_without_localised_evidence (contact every partition)"] = self.NOEV[c]
            arms = {}
            for ri, rk in enumerate(RANKS):
                arms["%s|%s" % (rk, COUNT)] = {"contacted_mass CMASS": AB.dist(self.CMASS[c][:, ri]),
                                               "contacted_mass_fraction_of_N": AB.dist(self.CMASS[c][:, ri] / float(N)),
                                               "B_N": self.arm(c, ri, b, unr, SF[rk], hop_m)}
                SERV["%s|%s" % (rk, c)] = {M: AB.served_at(LO[:, ri], HI[:, ri, mi], b) for mi, M in enumerate(D.M_CURVE)}
            e_c["arms"] = arms
            sfo = {}
            for ri, rk in enumerate(RANKS):
                e = {"contacted_mass_mean": {str(bb): round(float(self.MSUM[c][ri, bb - 1] / nq), 1) for bb in BP_SHOW if bb <= npart},
                     "gold_partition_contacted_share": {str(bb): D.q4((self.PRG[c][:, ri] < bb).mean()) for bb in BP_SHOW if bb <= npart},
                     "LO": AB.dist(LO[:, ri])}
                for mi, M in enumerate(D.M_CURVE):
                    row_ = SF[rk][mi]
                    assert abs(row_[-1] - unr[M].mean()) < 1e-12, "surface at every partition != unrouted (%s %s %d)" % (c, rk, M)
                    e[str(M)] = {"ALL_at_B_P": {str(bb): D.q4(row_[bb - 1]) for bb in BP_SHOW if bb <= npart},
                                 "best_fixed": {"B_P": int(np.argmax(row_)) + 1, "ALL": D.q4(row_.max())},
                                 "smallest_B_P_within_0.01_of_unrouted": int(np.flatnonzero(row_ >= row_[-1] - 0.01)[0]) + 1,
                                 "smallest_B_P_at_or_above_unrouted": int(np.flatnonzero(row_ >= row_[-1] - 1e-12)[0]) + 1}
                sfo[rk] = e
            e_c["fixed_fan_out_surfaces"] = sfo
            ies_ = RANKS.index("ES")
            bn = {}
            for mi, M in enumerate(D.M_CURVE):
                be = self.ESTAR[c][:, mi].astype(np.int64)
                a_ = AB.served_at(LO[:, ies_], HI[:, ies_, mi], be)
                bn[str(M)] = {"ALL": D.q4(a_.mean()), "delta_vs_unrouted": D.q4(a_.mean() - unr[M].mean()), "paired_vs_unrouted": D.paired(unr[M], a_),
                              "ESTAR_M": AB.dist(be), "ESTAR_fraction_of_partitions": AB.dist(be / float(npart)),
                              "PSTAR_M": AB.dist(self.PSTAR[c][:, mi]), "PSTAR_fraction_of_partitions": AB.dist(self.PSTAR[c][:, mi] / float(npart)),
                              "rows_with_|O'| < M": D.q4((self.OQLEN < M).mean())}
            e_c["lossless_references"] = {"NGP (distinct gold partitions; gold-dependent)": AB.dist(self.NGP[c]),
                                          "NGP_fraction_of_partitions": AB.dist(self.NGP[c] / float(npart)),
                                          "B_N_coupled ES|ESTAR_M and PSTAR_M": bn}
            e_c["communication"] = {"NSEEDP (distinct partitions of H_q)": AB.dist(self.NSEEDP[c]),
                                    "NEV (partitions holding evidence)": AB.dist(self.NEV[c]),
                                    "XC (seeds' 1-hop entries crossing partitions)": AB.dist(self.XC[c]),
                                    "XC_share_of_ENT (per query)": AB.dist(self.XC[c] / np.maximum(self.ENT, 1).astype(np.float64)),
                                    "XC_share_pooled": D.q4(self.XC[c].sum() / float(max(self.ENT.sum(), 1))),
                                    "router_sketch_reads RDS (union sketch entries of the 200 seeds)": AB.dist(self.RDS[c]),
                                    "RDS_over_RDN_pooled": D.q4(self.RDS[c].sum() / float(max(self.RDN.sum(), 1))),
                                    "cross_partition_share_of_E_entries (static)": self.cellrec[c]["cross_partition_share_of_E_entries"]}
            e_c["load (per-partition contacts over the population)"] = {a: RT3.load_stats(self.LOAD[c][ai], nq) for ai, a in enumerate(LOAD_ARMS)}
            e_c["latency_ms_per_row (cell: scores, ES / S, counts, evaluation)"] = D.ms_stats(self.TL[c])
            cells_out[c] = e_c
            for nm_, A_ in (("LO", self.LO), ("HI", self.HI), ("PRG", self.PRG), ("CNT", self.CNT), ("CMASS", self.CMASS), ("PSTAR", self.PSTAR),
                            ("ESTAR", self.ESTAR), ("NGP", self.NGP), ("NSEEDP", self.NSEEDP), ("NEV", self.NEV), ("XC", self.XC), ("RDS", self.RDS),
                            ("MSUM", self.MSUM), ("LOAD", self.LOAD)):
                arrays["%s__%s" % (nm_, c)] = A_[c]
            arrays["SIZES__" + c] = P.sizes
        diag["cells"] = cells_out
        # structural vs balanced random at the same K (paired ALL of ES.KNEE_SDIV; descriptive)
        svr = {}
        for c in self.cells:
            pz, K = self.cellrec[c]["partitioner"], self.cellrec[c]["K"]
            if pz not in STRUCTURAL:
                continue
            o = {}
            for s in RAND_SEEDS:
                rc = cell_name("RAND%d" % s, K)
                o["RAND%d" % s] = {str(M): dict(D.paired(SERV["ES|" + rc][M], SERV["ES|" + c][M]), structural_ALL=D.q4(SERV["ES|" + c][M].mean()),
                                                random_ALL=D.q4(SERV["ES|" + rc][M].mean())) for M in D.M_CURVE}
            svr[c] = o
        diag["structural_vs_random (gained = the structural cell serves ALL gold, the random cell does not)"] = svr
        diag["latency_ms"] = {"node_score_and_LOC_order": D.ms_stats(self.lat[NODE]), "fused_order": D.ms_stats(self.FUSE_LAT)}
        arrays["HOPS"] = np.asarray(pop.hops, np.int64)
        arrays["ENT"] = self.ENT
        arrays["RDN"] = self.RDN
        arrays["OQLEN"] = self.OQLEN
        arrays["cells"] = np.array(self.cells)
        arrays["ranks"] = np.array(RANKS)
        arrays["load_arms"] = np.array(LOAD_ARMS)
        arrays["m_curve"] = MSA
        return diag, arrays


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    assert mode == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
    R.run(sys.argv[2], sys.argv[3], "scale", "L1_DEVELOPMENT_SCALE", __file__, ScaleSpec, __doc__)


if __name__ == "__main__":
    main()
