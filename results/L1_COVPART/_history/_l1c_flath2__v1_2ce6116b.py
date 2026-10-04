"""FLAT_BALANCED_H2 -- are fixed graph partitions necessary once flat semantic retrieval receives the same bounded real-edge
expansion privilege?  One isolated architectural ablation (the user's ruling of 2026-09-25T22:00:40Z, verbatim in
results/L1_COVPART/PREREGISTRATION_FLAT_BALANCED_H2.json).  No partition anywhere in the arm: no PHG, no H4_SK, no block
ownership, no node->block voting, no SAFE, no QMAX block aggregation and no new scoring mechanism.

Rankings and arms (C = 100 = one block-equivalent: npart = N // 100 blocks of mean size 100; cap = round(N / npart) = 100)
    FLAT_RRF             the frozen node RRF extended to full depth (REPORT section 29): exhaustive fp32 dense products (unit
                         queries x fp16->fp32 node chunks) and fp32 SPLADE products; 1/(K0 + p_dense) + 1/(K0 + p_splade) with
                         0-based positions over all N nodes, K0 = 60, float64; a node without a positive SPLADE score takes the
                         SPLADE list depth (the positive-score count); descending, ties -> dense order
    FLAT@M               FLAT_RRF[:M]
    FLAT_BALANCED_H2@M   FLAT_RRF[:M - C] (semantic base) + the first C BALANCED-H2 nodes not in the base (bal1 then bal2, in
                         first-visit order; the frozen section-21 balanced beam: seeds = node_rrf(served dense top-200, served
                         SPLADE top-200)[:5], depth 2 over directed STRUCT, out-edges always and in-edges only at a non-hub
                         (hub: deg_u + 1 > cap), the frozen dst/dir/mid transition geometry fused by RRF, per-parent round-robin
                         compression to BEAM = 100 per hop) + fill (FLAT_RRF[M - C:] minus the patch, in order) until exactly M
The beam functions are the pinned source text of _l1c_microl3.py (admissible, transition_scores, rr_rank, fused_order, rows,
expand_real) and _l1c_transfer_composite.py (rr_compress, beam_balanced), exec'd over partition-free STRUCT arrays.
Bounded exact FLAT_RRF[:M] (the fresh corpora, N up to 6.0M, cannot be ranked in full per query on this host): every node of
FLAT_RRF[:M] is in the dense or the SPLADE top K_TOP when K_TOP >= K0 + 2M - 2 (a node below both has
fv <= 2/(K0 + K_TOP) <= 1/(K0 + M - 1) < fv of each of the M dense-top nodes).  Sweep 1 streams (BLOCK-node chunk, CQ-row
batch) products and keeps each row's exact dense and SPLADE top K_TOP (uint64 keys = score descending, position ascending)
with the other channel's score; bounds [lo, hi] on every unknown position prune the union to the contenders (upper bound >=
the M-th largest lower bound); sweep 2 re-streams the identical products and counts the exact position of every unknown
contender (plus self-check targets of known rank).  The top M of the contenders by (fv desc, dense position asc) == FLAT_RRF[:M].

Modes
    DEV_A caches (metaqa, metaqa_phg, squad, squad_phg, musique), split A, NON-SELECTING:
        python -u scratchpad/_l1c_flath2.py <cache> [--dry]  -> results/L1_COVPART/flath2_A_<cache>.json
            harness identity (partition-free STRUCT arrays, queries, seeds and balanced beam == the pinned uL3 head + the
            section-21 slice on every split-A row); the section-29 flat pass reproduced (deepest gold position per row ==
            flat_A_<cache>.json); the bounded algorithm == a batchwise full materialisation with the same products (every row);
            D2 = PRIMARY vs FLAT_BALANCED_H2 at PRIMARY's per-query exposure B_q; D3 = FLAT vs FLAT_BALANCED_H2 at B_q;
            fixed M descriptive
    fresh datasets (webqsp train_holdout, hotpotqa validation, 2wiki dev), split A, one shot:
        python -u scratchpad/_l1c_flath2.py <dataset> G [--dry]  gold-free (gold COUNTS only, for the population rule):
            queries, seeds, STRUCT, balanced beam, bounded exact FLAT_RRF[:M_MAIN] -> flath2_G_<dataset>.npz / .json
        python -u scratchpad/_l1c_flath2.py <dataset> E [--dry]  gold: the arms at M in M_CURVE; D1 = FLAT@M_MAIN vs
            FLAT_BALANCED_H2@M_MAIN (pre-registered) -> flath2_E_<dataset>.json
    --dry   DEV_A: the real-gold identities only (known reference numbers, none printed), every H2 path on STAND-IN gold, no
            record.  G: a smoke on the first DRY_ROWS split-A rows and the first DRY_CHUNKS node chunks, nothing written unless
            --dry-out=<dir outside the repository>.  E: reads that smoke (--dry-in=<dir>), STAND-IN gold, nothing written.
    --check-prereg   verify every pin against the pre-registration (code, function sources, constants, records, cache / dataset /
            derived files, population, write-once), then exit before any data is read; nothing written.
"""
import ast
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

import _l1g_core as G
import _l1c_transfer_blockmax as TB
import _ta_prepartition as TA
from src.l1_canonical import adapter as AD

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = G.X.REPO
OUT = os.path.join(REPO, "results", "L1_COVPART")
log = G.log
PINNED = {"_l1c_microl3.py": "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5",
          "_l1c_transfer_composite.py": "1241ec58e759ab99e77ff17936b6edc6b072344ea1d47a76a79f940a1047cc5b",
          "_l1c_transfer_blockmax.py": "f9232ef95b5f317bef85cb1f7304569ea4b00bf6d283a99fc8abe153ad239ee9",
          "_l1c_flat.py": "27aafbfc9d24e39e92154cd6d3ebe8364c07e32af5a8ced51a88e766bc6b3c95",
          "_l1c_h2patch.py": "a1ed9caa906fc50a1cee573e346bec1fde70e1947badd68c9b86cd2ab455fa9d",
          "_l1c_h2balanced.py": "7aed6b1d4399a1b432d0e6be48013cf73c92d4e47d2e5d2dd7b669b508d84bbe",
          "_l1c_qmaxbal.py": "cae88fa64a0516a4f55c99a4a3b464afd39e467dca42e4380a546ebbd545b1c8",
          "_l1g_candidate.py": "b01c27a6e64b18756db1ca1d8910e8db4bb54fa019d9809698cb1369206169e0",
          "_l1g_core.py": "05923d66d5754f1573414fff0d119b32d93e877e8014b26e970e7ff74dc61327",
          "_l1s_core.py": "ae3f087d0a785dc4d5d1c10367fc4cc21db0a4bb353efbe0265172b199492861",
          "_l1x90_core.py": "6eba63051165588c53900d98ec5644782ad107206f05a32ba6500d16cda76068",
          "_ta_prepartition.py": "271f8985f39c090a2bb996a8deeb056cc085a1010e28ba32231bb4f413bbfa07"}
PINNED_REPO = {"src/l1_canonical/adapter.py": "aaa3304dc8d7266fdbc59d3f58a77d6956b740efb842f8e90364dc39750f4c01",
               "src/l1_canonical/replay_cache.py": "1fc5411c9f12f25d81586aabdb1af0da371752f20be5d4d09d8d17529a89f0d1",
               "data/final_canonical/canonical.py": "c657830b304afe861893b516726af419d35142334f866517b83221427a9016f5"}
FLAT_RECORD = {"metaqa": "10769e0ca8c2171155e5b885f67403d40741ff1c0792f1701c60eaa9b5727ce8",
               "metaqa_phg": "f4fc7a2e6469097bf9e6357160e76ec56a5995a773eb4e1724041005af6a4ad9",
               "squad": "2552d3aff4d4847a19244352a685fbb9f52d5d5b5bd961efe52986700a10689d",
               "squad_phg": "7a459c4df4ab984d743f5ffe16b3d0378e8cedca547c1a64f417b3ab83ef7098",
               "musique": "bc22960ddcd787b1b23f78d5a250d4e37116c3ad80bed50810f308f119879507"}
DEV_A_CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
FRESH = {"webqsp": "train_holdout", "hotpotqa": "validation", "2wiki": "dev"}
POP = {"hotpotqa": {"row_query_ids_sha256": "e68f50610d35992806ae13359634770c98fe9b2c99fd1073156f70787e7d8e33",
                    "rows_sha256": "8f4bea60cd622414f7cdd85d7e5780c87ffbd834ff7f134159876f1f7b0c4175", "n": 2000, "n_A": 1031, "n_B": 969},
       "2wiki": {"row_query_ids_sha256": "1781c90d0c2cdd3ee40988081791575f3426cc3b2541cff8470e6dcc0eb7282c",
                 "rows_sha256": "9c29d7f6c978e005f6c823284b0fb28d2fb8301bc25dfe59cc27fe5b0db8d6f5", "n": 2000, "n_A": 1004, "n_B": 996},
       "webqsp": {"row_query_ids_sha256": "faa1d372294cc62f116549bdce087bb0d3f8ce533dce8bfd814c56e1c42e7a2e",
                  "rows_sha256": None, "n": 1503, "n_A": 786, "n_B": 717}}
K0 = TA.K0                                  # 60, the frozen RRF constant
C_PATCH = TA.K_LOCK                         # 100 = one block-equivalent (= cap = the router width)
BEAM, DEPTH = TA.K_LOCK, 2                  # as the pinned uL3 head (BEAM = K_LOCK, DEPTH = 2)
M_MAIN = 5000                               # the pre-registered total node budget of the fresh comparison (= P_MAIN x C)
M_CURVE = (500, 1000, 2000, 5000)
K_TOP = 12000                               # > K0 + 2 * M_MAIN - 2 = 10058
CQ, BLOCK = TB.CQ, TB.BLOCK                 # 200 query rows per product batch, 40000 nodes per chunk (= the shard size)
ALPHA = 0.01
EVAL_CAP = AD.CONTRACT["EVAL_CAP"]          # 2000
N_AGREE = 100
AGREE_MIN = 0.90                            # served-list sanity (wrong vectors / row mapping), not a float-noise test
DRY_ROWS, DRY_CHUNKS = 200, 3
NG_BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5-10", 5, 10), ("11+", 11, 1 << 40))
MICRO_FNS = ("admissible", "transition_scores", "rr_rank", "fused_order", "rows", "expand_real")
COMP_FNS = ("rr_compress", "beam_balanced")
MARK_REPAIR = "# ---------------------------------------------------------------- repair + evaluation"
MARK_BEAM = "# ---------------------------------------------------------------- the pinned beam re-run WITH candidate recording"
MARK_SAFE = "# ---------------------------------------------------------------- canonical SAFE machinery"
WORDS_PATCH = {"GAIN": "GRAPH_PATCH_HELPS", "LOSS": "GRAPH_PATCH_HURTS", "NEUTRAL": "NO_SIGNIFICANT_DIFFERENCE"}
WORDS_PART = {"GAIN": "FLAT_H2_BEATS_PARTITIONS", "LOSS": "PARTITIONS_BEAT_FLAT_H2", "NEUTRAL": "NO_SIGNIFICANT_DIFFERENCE"}
U64MAX = np.uint64(0xFFFFFFFFFFFFFFFF)
LO32 = np.uint64(0xFFFFFFFF)
CONSTANTS = {"K0": K0, "C_PATCH": C_PATCH, "BEAM": BEAM, "DEPTH": DEPTH, "M_MAIN": M_MAIN, "M_CURVE": list(M_CURVE), "K_TOP": K_TOP,
             "CQ": CQ, "BLOCK": BLOCK, "ALPHA": ALPHA, "EVAL_CAP": EVAL_CAP, "SEED_K": TA.SEED_K, "TOP200": TA.TOP200,
             "N_AGREE": N_AGREE, "AGREE_MIN": AGREE_MIN, "NG_BUCKETS": [list(b) for b in NG_BUCKETS]}
assert K_TOP >= K0 + 2 * M_MAIN - 2 and max(M_CURVE) == M_MAIN and C_PATCH == 100


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b_ in iter(lambda: f.read(1 << 20), b""):
            h.update(b_)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def label(o, alpha=ALPHA):
    if o["gained"] > o["lost"] and o["p"] < alpha:
        return "GAIN"
    if o["lost"] > o["gained"] and o["p"] < alpha:
        return "LOSS"
    return "NEUTRAL"


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def host_state():
    o = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import psutil
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        o.update({"host_total_gib": round(vm.total / 2 ** 30, 2), "host_available_gib": round(vm.available / 2 ** 30, 2),
                  "swap_used_gib": round(sw.used / 2 ** 30, 2)})
        me, fo = os.getpid(), []
        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                if p.info["pid"] != me and (p.info["name"] or "").lower().startswith("python") and p.info["memory_info"].rss > 500 * 2 ** 20:
                    fo.append({"pid": p.info["pid"], "rss_mb": int(p.info["memory_info"].rss / 2 ** 20)})
            except Exception:
                pass
        o["other_python_processes_over_500_MB (untouched)"] = fo
    except Exception as e:
        o["error"] = repr(e)
    return o


def stats(x):
    x = np.asarray(x, np.float64)
    if not x.size:
        return None
    return {"mean": round(float(x.mean()), 2), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


def q4(x):
    return round(float(x), 4)


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def fn_sources(fn, names):
    """verbatim source text of the named top-level functions of a pinned module (each defined exactly once)."""
    src = open(os.path.join(HERE, fn), "rb").read().decode("utf-8")
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            assert node.name not in out, "%s defines %s twice" % (fn, node.name)
            out[node.name] = ast.get_source_segment(src, node)
    assert set(out) == set(names), (fn, sorted(set(names) - set(out)))
    return out


# ---------------------------------------------------------------- keys, bounded top-K, flat ranking
def fkeys(sc, ids):
    """uint64 keys over float32 scores: ascending key == score descending, then position ascending (-0.0 == +0.0)."""
    b = (np.asarray(sc, np.float32) + np.float32(0.0)).view(np.uint32)
    o = np.where(b >= np.uint32(0x80000000), ~b, b | np.uint32(0x80000000))
    return ((~o).astype(np.uint64) << np.uint64(32)) | ids


def score_of_key(k):
    o = 0xFFFFFFFF - (int(k) >> 32)
    b = (o & 0x7FFFFFFF) if o >= 0x80000000 else (0xFFFFFFFF - o)
    return np.array([b], np.uint32).view(np.float32)[0]


class TopK(object):
    """per-row exact top-K by key over streamed chunks, with the other channel's score of every kept node."""

    def __init__(self, n, K, positive_only):
        self.K, self.pos = K, positive_only
        self.keys = [np.zeros(0, np.uint64) for _ in range(n)]
        self.oth = [np.zeros(0, np.float32) for _ in range(n)]
        self.thr = np.full(n, U64MAX, np.uint64)
        self.thr_sc = np.full(n, -np.inf, np.float32)

    def offer(self, j, sc, ids, oth):
        m = sc >= self.thr_sc[j]                              # superset of key < threshold key
        if self.pos:
            m &= sc > 0
        ix = np.flatnonzero(m)
        if not ix.size:
            return
        k = fkeys(sc[ix], ids[ix])
        keep = k < self.thr[j]
        if not keep.any():
            return
        kk = np.concatenate([self.keys[j], k[keep]])
        oo = np.concatenate([self.oth[j], oth[ix[keep]]])
        if len(kk) > self.K:
            p = np.argpartition(kk, self.K - 1)[:self.K]
            kk, oo = kk[p], oo[p]
        if len(kk) == self.K:
            t = kk.max()
            self.thr[j], self.thr_sc[j] = t, score_of_key(t)
        self.keys[j], self.oth[j] = kk, oo


def chunk_iter(E16, n_chunks=None):
    for c_, (a, b_, blk) in enumerate(E16.blocks(BLOCK)):
        if n_chunks is not None and c_ >= n_chunks:
            break
        yield a, b_, blk


def dense_products(Qu, Ec, j0, j1):
    return Qu[j0:j1] @ Ec.T                                   # (m, c) fp32, the section-29 product shape


def splade_products(Msp, QsA, j0, j1):
    Qd = np.asarray(QsA[j0:j1].todense(), np.float32)
    return np.ascontiguousarray(np.asarray(Msp @ Qd.T, np.float32).T)


def rrf_full(sd, ss):
    """the section-29 FLAT_RRF of one query over all nodes (full materialisation)."""
    n_ = len(sd)
    od = np.argsort(-sd, kind="stable")
    pd = np.empty(n_, np.int64)
    pd[od] = np.arange(n_)
    npos = int((ss > 0).sum())
    os_ = np.argsort(-ss, kind="stable")[:npos]
    ps = np.full(n_, npos, np.int64)
    ps[os_] = np.arange(npos)
    fv = 1.0 / (K0 + pd) + 1.0 / (K0 + ps)
    return od[np.argsort(-fv[od], kind="stable")], od, os_, npos


def flat_bounded(Qu, QsA, E16, Es, N, M, K, n_chunks=None, tag=""):
    """exact FLAT_RRF[:M] for every row of Qu (see the module docstring).  Returns top (nA, M) int64, dense / SPLADE top-200
    ids (-1 padded), npos and diagnostics."""
    nA = Qu.shape[0]
    assert int(Es.shard_size) == BLOCK and int(Es.n_rows) == N
    batches = [(j0, min(nA, j0 + CQ)) for j0 in range(0, nA, CQ)]
    n_ch = (N + BLOCK - 1) // BLOCK if n_chunks is None else min(n_chunks, (N + BLOCK - 1) // BLOCK)
    Nc = min(N, n_ch * BLOCK)                                 # nodes covered (N unless a dry smoke)
    TDk, TSk = TopK(nA, K, False), TopK(nA, K, True)
    npos = np.zeros(nA, np.int64)
    t0_, secs_ = time.time(), {}
    for c_, (a, b_, blk) in enumerate(chunk_iter(E16, n_ch)):
        Ec = np.ascontiguousarray(np.asarray(blk, np.float32))
        Msp = Es.shard(a // BLOCK)
        assert Msp.shape[0] == b_ - a and int(Msp.shape[1]) == int(QsA.shape[1])
        ids = np.arange(a, b_, dtype=np.uint64)
        for j0, j1 in batches:
            SDc = dense_products(Qu, Ec, j0, j1)
            SSc = splade_products(Msp, QsA, j0, j1)
            assert np.isfinite(SDc).all() and np.isfinite(SSc).all() and (SSc >= 0).all()
            npos[j0:j1] += (SSc > 0).sum(1)
            for i in range(j1 - j0):
                TDk.offer(j0 + i, SDc[i], ids, SSc[i])
                TSk.offer(j0 + i, SSc[i], ids, SDc[i])
        del Ec, Msp
        if c_ % 10 == 0 or c_ == n_ch - 1:
            el = time.time() - t0_
            log("  %s sweep 1 chunk %d / %d (%.0fs, ETA %.0fs, RSS %.0f MB)" % (tag, c_ + 1, n_ch, el, el / (c_ + 1) * (n_ch - c_ - 1), _rss_mb()))
    secs_["sweep1"] = round(time.time() - t0_, 1)
    # ---- resolve: exact where known, bounds elsewhere, contenders, counting targets
    top200_d = np.full((nA, TA.TOP200), -1, np.int64)
    top200_s = np.full((nA, TA.TOP200), -1, np.int64)
    R = []
    info = {"union": np.zeros(nA, np.int64), "contenders": np.zeros(nA, np.int64), "dense_targets": np.zeros(nA, np.int64),
            "splade_targets": np.zeros(nA, np.int64)}
    for j in range(nA):
        o = np.argsort(TDk.keys[j])
        kd, s_of_d = TDk.keys[j][o], TDk.oth[j][o]
        o = np.argsort(TSk.keys[j])
        ks, d_of_s = TSk.keys[j][o], TSk.oth[j][o]
        TDk.keys[j] = TDk.oth[j] = TSk.keys[j] = TSk.oth[j] = None
        nD, nS, npj = len(kd), len(ks), int(npos[j])
        assert nD == min(K, Nc) and nS == min(K, npj), (j, nD, nS, npj)
        idd = (kd & LO32).astype(np.int64)
        ids_ = (ks & LO32).astype(np.int64)
        top200_d[j, :min(TA.TOP200, nD)] = idd[:TA.TOP200]
        top200_s[j, :min(TA.TOP200, nS)] = ids_[:TA.TOP200]
        u = np.union1d(idd, ids_)
        ia, ib = np.searchsorted(u, idd), np.searchsorted(u, ids_)
        nU = len(u)
        pd_lo = np.full(nU, nD, np.int64)
        pd_hi = np.full(nU, Nc - 1, np.int64)
        pd_lo[ia] = np.arange(nD)
        pd_hi[ia] = np.arange(nD)
        s_sc = np.zeros(nU, np.float32)
        s_sc[ia] = s_of_d
        d_sc = np.zeros(nU, np.float32)
        d_sc[ib] = d_of_s
        pos_ = s_sc > 0
        ps_lo = np.where(pos_, nS, npj).astype(np.int64)
        ps_hi = np.where(pos_, npj - 1, npj).astype(np.int64)
        ps_lo[ib] = np.arange(nS)
        ps_hi[ib] = np.arange(nS)
        kn_d, kn_s = pd_lo == pd_hi, ps_lo == ps_hi
        assert kn_s.all() or nS == K, "a positive-score dense top-K node outside a SPLADE list that is not full"
        lb = 1.0 / (K0 + pd_hi) + 1.0 / (K0 + ps_hi)
        ub = 1.0 / (K0 + pd_lo) + 1.0 / (K0 + ps_lo)
        assert nU >= M
        tau = np.partition(lb, nU - M)[nU - M]                # the M-th largest lower bound
        cont = ub >= tau
        cid = u[cont]
        pd_c, ps_c = pd_lo[cont].copy(), ps_lo[cont].copy()
        td = np.flatnonzero(~kn_d[cont])
        ts = np.flatnonzero(~kn_s[cont])
        tdk = fkeys(d_sc[cont][td], cid[td].astype(np.uint64))
        tsk = fkeys(s_sc[cont][ts], cid[ts].astype(np.uint64))
        chk_d = [(kd[r], r) for r in sorted({nD - 1, min(M, nD) - 1})]           # self-check targets of known rank
        chk_s = [(ks[r], r) for r in sorted({nS - 1, min(M, nS) - 1})] if nS else []
        kd_t = np.concatenate([tdk, np.array([c[0] for c in chk_d], np.uint64)])
        ks_t = np.concatenate([tsk, np.array([c[0] for c in chk_s], np.uint64)])
        # a node counted for a target (key < target key) scores >= the lowest target score: sweep 2 only keys those nodes
        smin_d = score_of_key(kd_t.max())
        smin_s = score_of_key(ks_t.max()) if len(ks_t) else np.float32(np.inf)
        assert smin_s > 0
        R.append({"cid": cid, "pd": pd_c, "ps": ps_c, "td": td, "ts": ts, "kd": kd_t, "ks": ks_t, "smin_d": smin_d, "smin_s": smin_s,
                  "chk_d": [c[1] for c in chk_d], "chk_s": [c[1] for c in chk_s], "nD": nD, "nS": nS, "npos": npj})
        R[-1]["cnt_d"] = np.zeros(len(R[-1]["kd"]), np.int64)
        R[-1]["cnt_s"] = np.zeros(len(R[-1]["ks"]), np.int64)
        info["union"][j], info["contenders"][j], info["dense_targets"][j], info["splade_targets"][j] = nU, len(cid), len(td), len(ts)
    del TDk, TSk
    # ---- sweep 2: the identical products, exact counts for every target
    t1_ = time.time()
    for c_, (a, b_, blk) in enumerate(chunk_iter(E16, n_ch)):
        Ec = np.ascontiguousarray(np.asarray(blk, np.float32))
        Msp = Es.shard(a // BLOCK)
        ids = np.arange(a, b_, dtype=np.uint64)
        for j0, j1 in batches:
            SDc = dense_products(Qu, Ec, j0, j1)
            SSc = splade_products(Msp, QsA, j0, j1)
            for i in range(j1 - j0):
                r_ = R[j0 + i]
                px = np.flatnonzero(SDc[i] >= r_["smin_d"])
                if px.size:
                    k_ = fkeys(SDc[i][px], ids[px])
                    k_.sort()
                    r_["cnt_d"] += np.searchsorted(k_, r_["kd"], "left")
                px = np.flatnonzero(SSc[i] >= r_["smin_s"])         # smin_s > 0: positive-score nodes only
                if px.size:
                    k_ = fkeys(SSc[i][px], ids[px])
                    k_.sort()
                    r_["cnt_s"] += np.searchsorted(k_, r_["ks"], "left")
        del Ec, Msp
        if c_ % 10 == 0 or c_ == n_ch - 1:
            el = time.time() - t1_
            log("  %s sweep 2 chunk %d / %d (%.0fs, ETA %.0fs, RSS %.0f MB)" % (tag, c_ + 1, n_ch, el, el / (c_ + 1) * (n_ch - c_ - 1), _rss_mb()))
    secs_["sweep2"] = round(time.time() - t1_, 1)
    top = np.empty((nA, M), np.int64)
    for j, r_ in enumerate(R):
        nt_d, nt_s = len(r_["td"]), len(r_["ts"])
        cd_, cs_ = r_["cnt_d"], r_["cnt_s"]
        assert list(cd_[nt_d:]) == r_["chk_d"] and list(cs_[nt_s:]) == r_["chk_s"], "self-check counts != known ranks (row %d)" % j
        assert (cd_[:nt_d] >= r_["nD"]).all() and (cd_[:nt_d] <= Nc - 1).all()
        assert (cs_[:nt_s] >= r_["nS"]).all() and (cs_[:nt_s] <= r_["npos"] - 1).all()
        pd_c, ps_c = r_["pd"], r_["ps"]
        pd_c[r_["td"]] = cd_[:nt_d]
        ps_c[r_["ts"]] = cs_[:nt_s]
        fv = 1.0 / (K0 + pd_c) + 1.0 / (K0 + ps_c)
        o = np.lexsort((pd_c, -fv))                            # fv descending, ties -> dense position
        top[j] = r_["cid"][o[:M]]
        R[j] = None
    info = {k: stats(v) for k, v in info.items()}
    info["seconds"] = secs_
    info["nodes_covered"] = int(Nc)
    return top, top200_d, top200_s, npos, info


def flat_full_top(Qu, QsA, E16, Es, N, M):
    """validation: the same (batch, chunk) products materialised in full per batch, ranked by rrf_full -> top M."""
    nA = Qu.shape[0]
    top = np.empty((nA, M), np.int64)
    for j0 in range(0, nA, CQ):
        j1 = min(nA, j0 + CQ)
        SD = np.empty((j1 - j0, N), np.float32)
        SS = np.empty((j1 - j0, N), np.float32)
        for a, b_, blk in chunk_iter(E16):
            Ec = np.ascontiguousarray(np.asarray(blk, np.float32))
            SD[:, a:b_] = dense_products(Qu, Ec, j0, j1)
            SS[:, a:b_] = splade_products(Es.shard(a // BLOCK), QsA, j0, j1)
        for i in range(j1 - j0):
            top[j0 + i] = rrf_full(SD[i], SS[i])[0][:M]
        del SD, SS
    return top


# ---------------------------------------------------------------- partition-free STRUCT, queries, seeds, the balanced beam
def struct_arrays(cd, N):
    """directed STRUCT out-CSR (== adapter struct_csr(directed=True)) and its in-lists (== the pinned head's scipy transpose);
    int32 neighbour ids."""
    s, d, _, _ = cd.family("structural")
    s, d = np.asarray(s), np.asarray(d)
    o = np.argsort(s, kind="stable")
    xo = np.zeros(N + 1, np.int64)
    xo[1:] = np.cumsum(np.bincount(s, minlength=N))
    ao = d[o].astype(np.int32)
    del o
    oi = np.argsort(ao, kind="stable")
    xi = np.zeros(N + 1, np.int64)
    xi[1:] = np.cumsum(np.bincount(ao, minlength=N))
    ai = np.empty(len(ao), np.int32)
    for a in range(0, len(oi), 1 << 22):
        ai[a:a + (1 << 22)] = np.searchsorted(xo, oi[a:a + (1 << 22)], side="right") - 1
    del oi
    return xo, ao, xi, ai


def undirected_degree(cd, N):
    """deg_u of the undirected, deduped, loop-free STRUCT (== np.diff(struct_csr(directed=False)[0])) from the stamped keys.npz,
    read directly: the adapter's keysets() would rebuild and write keys.npz on a stamp mismatch."""
    kp = cd._keys_path()
    z = np.load(kp, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    z.close()
    assert meta.get("DATASET_json_RECORD_SHA256") == cd.record_sha, "keys.npz stamp != the DATASET.json record"
    ST = AD.NpzView(kp)["STRUCT"]
    assert len(ST) == int(meta["STRUCT"])
    deg = np.zeros(N, np.int64)
    for a in range(0, len(ST), 1 << 22):
        k = np.asarray(ST[a:a + (1 << 22)], np.int64)
        deg += np.bincount(k // N, minlength=N)
        deg += np.bincount(k % N, minlength=N)
    return deg


def unit_queries(cd, rows_g):
    Q = cd.query_embeddings[np.asarray(rows_g, np.int64)].astype(np.float32)
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)     # as _l1s_core.Data
    return Q


def served_seeds(cd, rows_g):
    d200 = np.asarray(cd.dense_topk(TA.TOP200, rows_g), np.int64)
    s200 = np.asarray(cd.splade_topk(TA.TOP200, rows_g), np.int64)
    out = np.full((len(rows_g), TA.SEED_K), -1, np.int64)
    for j in range(len(rows_g)):
        sd = TA.node_rrf(d200[j], s200[j])[:TA.SEED_K]         # as replay_cache.build
        out[j, :len(sd)] = sd
    return out, d200, s200


class _NS(object):
    pass


def beam_namespace(FSRC, Q, E16, seeds, N, xo, ao, xi, ai, hub):
    D_ = _NS()
    D_.Q, D_._E16 = Q, E16
    NS = {"__builtins__": __builtins__, "np": np, "time": time, "D": D_, "N": int(N), "seeds": seeds, "DEPTH": DEPTH, "BEAM": BEAM,
          "K0": K0, "xo": xo, "ao": ao, "xi": xi, "ai": ai, "hub": hub}
    for nm in MICRO_FNS + COMP_FNS:
        exec(FSRC[nm], NS)
    return NS


def run_beams(NS, nA, tag):
    bal1, bal2 = [], []
    ne = np.zeros((nA, DEPTH), np.int64)
    sz = np.zeros((nA, DEPTH), np.int64)
    tr = np.zeros((nA, DEPTH), bool)
    secs_ = np.zeros(nA)
    t = time.time()
    for j in range(nA):
        beams, _c, _p, st, n_edges, dt = NS["beam_balanced"](j)
        bal1.append([int(x) for x in beams[0]])
        bal2.append([int(x) for x in beams[1]])
        ne[j], secs_[j] = n_edges, dt
        sz[j] = [len(beams[0]), len(beams[1])]
        tr[j] = [bool(s_ and s_["truncated"]) for s_ in st]
        if j % 200 == 0:
            log("  %s beam %d / %d (%.0fs) sizes %d/%d (RSS %.0f MB)" % (tag, j, nA, time.time() - t, sz[j, 0], sz[j, 1], _rss_mb()))
    st_ = {"transitions_scored_per_query_mean (hop1 / hop2)": [round(float(x), 1) for x in ne.mean(0)],
           "beam_size_mean (hop1 / hop2)": [round(float(x), 1) for x in sz.mean(0)],
           "queries_with_an_empty_hop (hop1 / hop2)": [int((sz[:, h] == 0).sum()) for h in range(DEPTH)],
           "queries_truncated_at_hop (hop1 / hop2)": [int(tr[:, h].sum()) for h in range(DEPTH)],
           "wall_clock_ms_per_query": {"mean": round(1000 * float(secs_.mean()), 1), "p95": round(1000 * float(np.percentile(secs_, 95)), 1)},
           "seconds": round(time.time() - t, 1)}
    return bal1, bal2, st_


# ---------------------------------------------------------------- the arms
def arm_row(order, bal, M):
    """FLAT_BALANCED_H2@M of one row: (base set, FLAT tail set order[M-C:M], novel list, fill list); |served| == M asserted."""
    base = order[:M - C_PATCH].tolist()
    bset = set(base)
    novel = []
    for v in bal:
        if v not in bset:
            novel.append(v)
            if len(novel) == C_PATCH:
                break
    nset = set(novel)
    fill, k = [], M - C_PATCH
    while len(fill) < C_PATCH - len(novel):
        v = int(order[k])
        k += 1
        if v not in nset:
            fill.append(v)
    assert k <= M
    tail = set(order[M - C_PATCH:M].tolist())
    assert len(bset) == M - C_PATCH and not (nset & bset) and not (set(fill) & (bset | nset)) and len(bset) + len(nset) + len(fill) == M
    return bset, tail, novel, fill


def evaluate(orders, bals, bal1_sets, Mvec, golds=None):
    """per-row arrays of FLAT@M and FLAT_BALANCED_H2@M at the per-row budget Mvec (gold-free patch statistics always)."""
    nA = len(orders)
    r = {k: np.zeros(nA, np.int64) for k in ("n_novel", "n_novel_hop1", "n_fill", "n_bal", "n_bal_in_base")}
    if golds is not None:
        for k in ("flat_all", "flat_any", "h2_all", "h2_any"):
            r[k] = np.zeros(nA, bool)
        for k in ("g_rescued (gold in the patch, not in FLAT@M)", "g_displaced (gold in FLAT@M tail, not served)", "g_patch_already_in_FLAT_tail"):
            r[k] = np.zeros(nA, np.int64)
    for j in range(nA):
        M = int(Mvec[j])
        bset, tail, novel, fill = arm_row(orders[j], bals[j], M)
        r["n_novel"][j], r["n_fill"][j], r["n_bal"][j] = len(novel), len(fill), len(bals[j])
        r["n_novel_hop1"][j] = sum(1 for v in novel if v in bal1_sets[j])
        r["n_bal_in_base"][j] = sum(1 for v in bals[j] if v in bset)
        if golds is None:
            continue
        g = [int(x) for x in golds[j]]
        if not g:
            continue
        nset, fset = set(novel), set(fill)
        inf = [(x in bset) or (x in tail) for x in g]
        inh = [(x in bset) or (x in nset) or (x in fset) for x in g]
        r["flat_all"][j], r["flat_any"][j] = all(inf), any(inf)
        r["h2_all"][j], r["h2_any"][j] = all(inh), any(inh)
        r["g_rescued (gold in the patch, not in FLAT@M)"][j] = sum(1 for x in g if x in nset and x not in bset and x not in tail)
        r["g_displaced (gold in FLAT@M tail, not served)"][j] = sum(1 for x in g if x in tail and not ((x in nset) or (x in fset)))
        r["g_patch_already_in_FLAT_tail"][j] = sum(1 for x in g if x in nset and x in tail)
    return r


def patch_summary(r):
    return {"patch_nodes (novel beam nodes served)": stats(r["n_novel"]), "patch_nodes_from_hop1": stats(r["n_novel_hop1"]),
            "fill_nodes": stats(r["n_fill"]), "beam_nodes (bal1 + bal2)": stats(r["n_bal"]),
            "beam_nodes_already_in_the_semantic_base": stats(r["n_bal_in_base"]),
            "rows_with_a_full_patch (C novel nodes)": int((r["n_novel"] == C_PATCH).sum()), "rows_with_an_empty_patch": int((r["n_novel"] == 0).sum())}


def paired(ref, arm, s=None):
    s = np.ones(len(ref), bool) if s is None else s
    g_, l_, p_ = G.X.mcnemar(ref[s], arm[s])
    return {"gained": g_, "lost": l_, "p": p_}


def strata_masks(hops_, ngold):
    out = {}
    if hops_ is not None and (np.asarray(hops_) >= 0).any():
        out["per_hop"] = {"hop%d" % h: np.asarray(hops_) == h for h in sorted(set(int(x) for x in hops_ if x >= 0))}
    out["per_n_gold"] = {nm: (ngold >= lo) & (ngold <= hi) for nm, lo, hi in NG_BUCKETS}
    return out


def compare(ref_nm, ref, arm_nm, arm, words, strata):
    o = paired(ref, arm)
    lab = label(o)
    out = {"reference": ref_nm, "arm": arm_nm, "ALL_reference": q4(ref.mean()), "ALL_arm": q4(arm.mean()),
           "points_arm_minus_reference": round(100.0 * (float(arm.mean()) - float(ref.mean())), 2),
           "paired (gained = arm covers ALL gold, reference does not)": o, "label": lab, "verdict": words[lab], "strata": {}}
    for sn, masks in strata.items():
        out["strata"][sn] = {k: {"n": int(m.sum()), "ALL_reference": q4(ref[m].mean()), "ALL_arm": q4(arm[m].mean()), "paired": paired(ref, arm, m)}
                             for k, m in masks.items() if m.any()}
    return out


def gold_accounting(r):
    return {k: int(r[k].sum()) for k in r if k.startswith("g_")}


# ---------------------------------------------------------------- arguments, pins, pre-registration
CODE_SHA = sha_file(os.path.abspath(__file__))
name = sys.argv[1]
DRY = "--dry" in sys.argv
CHECK = "--check-prereg" in sys.argv                         # verify every pin against the pre-registration, read no data, write nothing
assert not (CHECK and DRY)
DRY_OUT = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--dry-out=")), None)
DRY_IN = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--dry-in=")), None)
for p_ in (DRY_OUT, DRY_IN):
    assert p_ is None or (DRY and not os.path.abspath(p_).lower().startswith(os.path.abspath(REPO).lower())), "--dry-out / --dry-in: dry runs only, outside the repository"
if name in DEV_A_CACHES:
    MODE = "DEV_A"
else:
    assert name in FRESH and len(sys.argv) > 2 and sys.argv[2] in ("G", "E"), "usage: <DEV_A cache> [--dry] | <webqsp|hotpotqa|2wiki> <G|E> [--dry]"
    MODE = sys.argv[2]
for fn, sha in PINNED.items():
    assert sha_file(os.path.join(HERE, fn)) == sha, "pinned module %s changed" % fn
for fn, sha in PINNED_REPO.items():
    assert sha_file(os.path.join(REPO, fn)) == sha, "pinned file %s changed" % fn
FSRC = dict(fn_sources("_l1c_microl3.py", MICRO_FNS), **fn_sources("_l1c_transfer_composite.py", COMP_FNS))
FSRC_SHA = {k: sha_text(v) for k, v in sorted(FSRC.items())}
PRE = os.path.join(OUT, "PREREGISTRATION_FLAT_BALANCED_H2.json")
pre, pre_sha = None, None
if not DRY:
    pre = json.load(open(PRE, encoding="utf-8"))
    pre_sha = sha_file(PRE)
    for grp in ("new_modules", "imports_unchanged"):
        for mod, pn in pre["code"][grp].items():
            assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    assert pre["code"]["new_modules"]["_l1c_flath2.py"]["sha256"] == CODE_SHA
    assert pre["code"]["function_sources_sha256"] == FSRC_SHA, "a pinned beam function changed"
    assert pre["constants"] == CONSTANTS, "constants differ from the pre-registration"
    for nm, pn in pre["records_read_only"].items():
        assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "record %s changed" % nm
HOST0 = host_state()
log("%s %s%s: host at start %s" % (name, MODE, " (DRY)" if DRY else "", json.dumps(HOST0)))
T_ALL = time.time()
TIMES = {}


def finish(res, fp_out):
    res["seconds_by_stage"] = TIMES
    res["seconds"] = round(time.time() - T_ALL, 1)
    res["process_rss_mb_at_end"] = round(_rss_mb(), 1)
    assert sha_file(os.path.abspath(__file__)) == CODE_SHA, "the module file changed during the run"
    if DRY:
        blob = json.dumps(res)
        if DRY_OUT:
            with open(os.path.join(DRY_OUT, os.path.basename(fp_out)), "w", encoding="utf-8") as f_:
                f_.write(blob)
        log("DRY RUN (%.0fs, RSS %.0f MB): record assembled (%d bytes) and NOT written to the repository.  times %s" % (
            time.time() - T_ALL, _rss_mb(), len(blob), json.dumps(TIMES)))
        sys.exit(0)
    G.S.wj(fp_out, res)
    log("done (%.0fs) -> %s sha256 %s" % (res["seconds"], rel(fp_out), sha_file(fp_out)[:12]))


# ======================================================================== DEV_A (split A, non-selecting)
if MODE == "DEV_A":
    fp_out = os.path.join(OUT, "flath2_A_%s.json" % name)
    if not DRY:
        assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
        assert sha_file(G.X.CACHES[name]) == pre["caches_read_only"][name]["sha256"], "cache %s changed" % name
    fp_flat = os.path.join(OUT, "flat_A_%s.json" % name)
    assert sha_file(fp_flat) == FLAT_RECORD[name], "the section-29 record changed"
    if CHECK:
        log("pre-registration check passed (%s): code, function sources, constants, records and cache pinned; no data read, nothing written" % name)
        sys.exit(0)
    frec = json.load(open(fp_flat, encoding="utf-8"))
    # ---- the pinned uL3 head (data, STRUCT, the pinned beam on split A) and the section-21 slice (bal1, bal2), in their own namespace
    src = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read().decode("utf-8")
    head = src[src.index("import json"):src.index(MARK_REPAIR)]
    assert head.count("name = sys.argv[1]") == 1 and head.count('LATENT = "--latent" in sys.argv') == 1
    head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
    csrc = open(os.path.join(HERE, "_l1c_transfer_composite.py"), "rb").read().decode("utf-8")
    cslice = csrc[csrc.index(MARK_BEAM):csrc.index(MARK_SAFE)]
    for nm in MICRO_FNS:
        assert FSRC[nm] in head
    for nm in COMP_FNS:
        assert FSRC[nm] in cslice
    t_ = time.time()
    H = {"__name__": "_l1c_flath2_pinned_head", "__builtins__": __builtins__}
    exec(head, H)
    H["secs_pinned"], H["edges_pinned"], H["REPRO"], H["rec21"] = H["secs"].copy(), H["edges_h"].copy(), False, None
    exec(cslice, H)
    H.pop("CAND_B", None)
    TIMES["pinned_head_and_balanced_beam"] = round(time.time() - t_, 1)
    D, C = H["D"], H["C"]
    cd, N, nq = D.cd, int(H["N"]), int(H["nq"])
    rowsA = np.asarray(H["rowsA"], np.int64)
    nA = len(rowsA)
    assert nA == int(frec["n_split_A"]) and N == int(frec["N"]) and int(H["cap"]) == C_PATCH and H["BEAM"] == BEAM and H["DEPTH"] == DEPTH
    rows_g = np.asarray(D.rows, np.int64)[rowsA]
    # ---- harness identity: the partition-free fresh-path machinery reproduces the pinned beam on every split-A row
    t_ = time.time()
    xo, ao, xi, ai = struct_arrays(cd, N)
    deg_u = undirected_degree(cd, N)
    hub = (deg_u + 1) > C_PATCH
    assert (xo == H["xo"]).all() and (ao == H["ao"]).all() and (xi == H["xi"]).all() and (ai == H["ai"]).all(), "STRUCT arrays differ from the pinned head"
    assert (deg_u == H["deg_u"]).all() and (hub == H["hub"]).all(), "hub rule differs from the pinned head"
    Qu = unit_queries(cd, rows_g)
    assert Qu.tobytes() == np.ascontiguousarray(D.Q[rowsA]).tobytes(), "unit queries differ from the pinned Data"
    seedsA, d200, s200 = served_seeds(cd, rows_g)
    assert (seedsA == np.asarray(C.seeds, np.int64)[rowsA]).all(), "seeds differ from the replay cache"
    QsA = cd.embeddings("splade", "queries").read(rows_g).tocsr()
    QsD = D.Qs[rowsA]
    assert (QsA.indptr == QsD.indptr).all() and (QsA.indices == QsD.indices).all() and (QsA.data == QsD.data).all(), "SPLADE queries differ"
    NS = beam_namespace(FSRC, Qu, D._E16, seedsA, N, xo, ao, xi, ai, hub)
    bal1, bal2, BEAM_STATS = run_beams(NS, nA, name)
    for j, qi in enumerate(rowsA):
        assert bal1[j] == [int(x) for x in H["bal1"][qi]] and bal2[j] == [int(x) for x in H["bal2"][qi]], "harness beam != pinned balanced beam (row %d)" % qi
    NS.clear()                                               # the exec'd functions hold NS as their globals (a cycle): free STRUCT now
    del NS, xo, ao, xi, ai
    TIMES["harness_identity"] = round(time.time() - t_, 1)
    IDENT = ["partition-free STRUCT out-CSR / in-lists / undirected degree / hub == the pinned uL3 head",
             "unit queries (bitwise), SPLADE queries and served-list seeds == the pinned Data / replay cache",
             "the pinned beam functions exec'd over the partition-free arrays == the pinned section-21 balanced beam (bal1, bal2) on all %d split-A rows" % nA]
    log("harness identity: %s (%.0fs)" % ("; ".join(IDENT), TIMES["harness_identity"]))
    # ---- the section-29 flat pass (F1: batches of TB.CQ cache rows, split-B rows only as product companions), real gold
    t_ = time.time()
    B_q = np.asarray(frec["_budget_PRIMARY"], np.int64)
    LMAX = int(max(B_q.max(), M_MAIN))
    gold_real = [np.asarray(C.gold_nodes[qi], np.int64) for qi in rowsA]
    Q = np.ascontiguousarray(D.Q, dtype=np.float32)
    Es = cd.embeddings("splade", "docs")
    SPL = [Es.shard(k) for k in range(Es.n_shards)]
    isA = np.zeros(nq, bool)
    isA[rowsA] = True
    jA = {int(q): j for j, q in enumerate(rowsA)}
    F1 = np.empty((nA, LMAX), np.int64)
    GPOS = [None] * nA
    for qa in range(0, nq, TB.CQ):
        qz = min(nq, qa + TB.CQ)
        SD = np.empty((qz - qa, N), np.float32)
        for a, b_, blk in D._E16.blocks(TB.BLOCK):
            Ec = np.ascontiguousarray(np.asarray(blk, np.float32))
            SD[:, a:b_] = Q[qa:qz] @ Ec.T
        rows_here = [qi for qi in range(qa, qz) if isA[qi]]
        if not rows_here:
            continue
        Qd = np.asarray(D.Qs[rows_here].todense(), np.float32)
        SS = np.empty((len(rows_here), N), np.float32)
        a = 0
        for Msp in SPL:
            SS[:, a:a + Msp.shape[0]] = np.asarray(Msp @ Qd.T, np.float32).T
            a += Msp.shape[0]
        for i, qi in enumerate(rows_here):
            j = jA[qi]
            of = rrf_full(SD[qi - qa], SS[i])[0]
            pf = np.empty(N, np.int64)
            pf[of] = np.arange(N)
            F1[j] = of[:LMAX]
            GPOS[j] = pf[gold_real[j]]
        del SD, SS, Qd
    del SPL
    TIMES["flat_pass_F1"] = round(time.time() - t_, 1)
    gmax = np.array([int(g.max()) if len(g) else -1 for g in GPOS], np.int64)
    assert gmax.tolist() == frec["_gmax_FLAT_RRF (0-based deepest gold position per split-A row)"], "F1 != the section-29 FLAT_RRF (deepest gold positions)"
    ind_prim_real = np.asarray(frec["_ind_PRIMARY"], bool)
    pr = frec["pairs (pre-registered; route = reference, flat = arm at the route's per-query budget)"]["PRIMARY vs FLAT_RRF"]
    assert q4((gmax < B_q).mean()) == pr["ALL_flat"] and q4(ind_prim_real.mean()) == pr["ALL_route"], "FLAT@B_q / PRIMARY != the section-29 record"
    log("section-29 flat pass reproduced (deepest gold position of every split-A row == flat_A_%s.json; FLAT@B_q and PRIMARY ALL == its PRIMARY pair; no number printed) in %.0fs" % (name, TIMES["flat_pass_F1"]))
    # ---- the bounded algorithm (F2: batches of TB.CQ split-A rows, the fresh-path products) == a full materialisation of the same products
    t_ = time.time()
    F2, t2d, t2s, npos2, F2INFO = flat_bounded(Qu, QsA, D._E16, Es, N, M_MAIN, K_TOP, tag=name)
    TIMES["bounded_F2"] = round(time.time() - t_, 1)
    t_ = time.time()
    FULL = flat_full_top(Qu, QsA, D._E16, Es, N, M_MAIN)
    assert (F2 == FULL).all(), "bounded exact top-M != the full materialisation of the same products"
    TIMES["full_materialisation_check"] = round(time.time() - t_, 1)
    del FULL
    same_rows = int(sum(1 for j in range(nA) if (F1[j, :M_MAIN] == F2[j]).all()))
    ovl = np.array([len(set(F1[j, :M_MAIN].tolist()) & set(F2[j].tolist())) / float(M_MAIN) for j in range(nA)])
    VALID = {"bounded_top_M == full materialisation of the same (batch, chunk) products": "every split-A row (asserted)",
             "M": M_MAIN, "K_TOP": K_TOP, "bounded_diagnostics": F2INFO,
             "F1 (section-29 batches of cache rows) vs F2 (fresh-path batches of split-A rows), descriptive":
                 {"rows_with_identical_top_M_order": same_rows, "top_M_set_overlap_mean": round(float(ovl.mean()), 6), "top_M_set_overlap_min": round(float(ovl.min()), 6)},
             "served_top100_overlap_mean (dense / splade)": [round(float(np.mean([len(set(t2d[j, :N_AGREE].tolist()) & set(D.d_ids[qi, :N_AGREE].tolist())) / float(N_AGREE) for j, qi in enumerate(rowsA)])), 5),
                                                            round(float(np.mean([len(set(t2s[j, :min(N_AGREE, npos2[j])].tolist()) & set(D.s_ids[qi, :min(N_AGREE, npos2[j])].tolist())) / float(max(1, min(N_AGREE, npos2[j])))
                                                                                 for j, qi in enumerate(rowsA)])), 5)]}
    log("bounded algorithm validated: %s" % json.dumps(VALID))
    # ---- gold (real, or STAND-IN in a dry run) and the arms
    REPRODUCED = "; ".join(IDENT) + "; section-29 FLAT_RRF (F1) reproduced on the real gold; bounded exact top-M == full materialisation"
    golds, ind_prim = gold_real, ind_prim_real
    if DRY:
        rng_dry = np.random.default_rng(12345)
        golds = [np.sort(rng_dry.choice(N, size=len(g), replace=False)).astype(np.int64) for g in gold_real]
        ind_prim = rng_dry.random(nA) < 0.5
        REPRODUCED += "; DRY RUN: every arm below ran on STAND-IN gold nodes and a STAND-IN PRIMARY indicator"
    bals = [bal1[j] + bal2[j] for j in range(nA)]
    b1s = [set(b) for b in bal1]
    hopsA = np.asarray(H["hops"])[rowsA]
    ngold = np.array([len(g) for g in golds], np.int64)
    ST_ = strata_masks(hopsA, ngold)
    t_ = time.time()
    rq = evaluate(F1, bals, b1s, B_q, golds)
    D2 = compare("PRIMARY (QMAX_BALANCED_H2_PATCH1, section-29 record)", ind_prim, "FLAT_BALANCED_H2@B_q", rq["h2_all"], WORDS_PART, ST_)
    D3 = compare("FLAT@B_q", rq["flat_all"], "FLAT_BALANCED_H2@B_q", rq["h2_all"], WORDS_PATCH, ST_)
    assert DRY or q4(rq["flat_all"].mean()) == pr["ALL_flat"]
    FIX, FIXR = {}, {}
    for M in M_CURVE:
        r_ = evaluate(F1, bals, b1s, np.full(nA, M), golds)
        FIXR[M] = r_
        FIX[str(M)] = {"FLAT_ALL": q4(r_["flat_all"].mean()), "FLAT_H2_ALL": q4(r_["h2_all"].mean()),
                       "FLAT_ANY": q4(r_["flat_any"].mean()), "FLAT_H2_ANY": q4(r_["h2_any"].mean()),
                       "FLAT vs FLAT_H2 (descriptive)": compare("FLAT@%d" % M, r_["flat_all"], "FLAT_BALANCED_H2@%d" % M, r_["h2_all"], WORDS_PATCH, {}),
                       "patch": patch_summary(r_), "gold_accounting": gold_accounting(r_)}
    r5 = evaluate(F2, bals, b1s, np.full(nA, M_MAIN), golds)
    TIMES["arms"] = round(time.time() - t_, 1)
    if not DRY:
        for k_, v_ in (("D2", D2), ("D3", D3)):
            o = v_["paired (gained = arm covers ALL gold, reference does not)"]
            log("%s %-58s ref %.4f arm %.4f (%+.2f pts) +%d/-%d p=%.3g %s" % (k_, v_["reference"] + " vs " + v_["arm"], v_["ALL_reference"], v_["ALL_arm"],
                                                                             v_["points_arm_minus_reference"], o["gained"], o["lost"], o["p"], v_["verdict"]))
        log("fixed M (FLAT / FLAT_H2 ALL): %s" % json.dumps({m: [FIX[m]["FLAT_ALL"], FIX[m]["FLAT_H2_ALL"]] for m in FIX}))
    res = {"cache": name, "dataset": G.X.DS_OF[name], "partition": G.PARTITION_OF.get(name), "n_split_A": nA, "N": N,
           "mode": "PREREGISTERED_FLAT_BALANCED_H2_DEV_A_SPLIT_A", "status": "DEV_A_NON_SELECTING (diagnostic; no selection, no adoption)",
           "preregistration": {"path": rel(PRE), "sha256": pre_sha},
           "cache_file": {"path": rel(G.X.CACHES[name]), "sha256": sha_file(G.X.CACHES[name])},
           "section29_record": {"path": rel(fp_flat), "sha256": FLAT_RECORD[name]},
           "code": {"path": rel(os.path.abspath(__file__)), "sha256": CODE_SHA}, "pinned": PINNED, "pinned_repo": PINNED_REPO,
           "function_sources_sha256": FSRC_SHA, "constants": CONSTANTS,
           "platform": {"python": platform.python_version(), "numpy": np.__version__}, "host_at_start": HOST0,
           "reference_reproduced": REPRODUCED, "bounded_validation": VALID, "balanced_beam (split A)": BEAM_STATS,
           "budget_B_q (PRIMARY per-query exposure)": stats(B_q),
           "D2 (PRIMARY vs FLAT_BALANCED_H2 at B_q; gained = FLAT_H2 covers, PRIMARY does not)": D2,
           "D3 (FLAT vs FLAT_BALANCED_H2 at B_q; gained = the patch helps)": D3,
           "patch_at_B_q": patch_summary(rq), "gold_accounting_at_B_q": gold_accounting(rq),
           "ANY_at_B_q": {"FLAT": q4(rq["flat_any"].mean()), "FLAT_H2": q4(rq["h2_any"].mean())},
           "fixed_M (descriptive)": FIX,
           "bounded_F2_at_M_MAIN (descriptive: the fresh-path ranking)": {"FLAT_ALL": q4(r5["flat_all"].mean()), "FLAT_H2_ALL": q4(r5["h2_all"].mean())},
           "hops_available": sorted(set(int(x) for x in hopsA if x >= 0)),
           "_ind_FLAT_Bq": rq["flat_all"].astype(int).tolist(), "_ind_FLAT_H2_Bq": rq["h2_all"].astype(int).tolist(),
           "_ind_PRIMARY (section-29 record)": ind_prim.astype(int).tolist(),
           "_ind_FLAT_%d" % M_MAIN: FIXR[M_MAIN]["flat_all"].astype(int).tolist(), "_ind_FLAT_H2_%d" % M_MAIN: FIXR[M_MAIN]["h2_all"].astype(int).tolist(),
           "_n_novel_Bq": rq["n_novel"].tolist()}
    finish(res, fp_out)

# ======================================================================== fresh datasets
ds = name
cd = AD.CanonicalDataset(ds)
N = int(cd.n_nodes)
assert cd.eval_split == FRESH[ds]
npart_eq = N // 100
cap = int(round(N / npart_eq))
assert cap == C_PATCH, (ds, cap)
if not DRY:
    dp = pre["population"][ds]["dataset_pins"]
    assert cd.record_sha == dp["DATASET_json_RECORD_SHA256"], "DATASET.json record changed"
    for nm_, pth in (("keys.npz", cd._keys_path()), ("query_index.npz", cd._query_index_path())):
        assert sha_file(pth) == pre["derived_read_only"][ds][nm_]["sha256"], "%s of %s changed" % (nm_, ds)
qp = cd._query_index_path()
zq = np.load(qp, allow_pickle=False)
assert json.loads(str(zq["meta_json"])).get("DATASET_json_RECORD_SHA256") == cd.record_sha, "query_index.npz stamp != the DATASET.json record"
gold_ptr = zq["gold_ptr"]                                    # gold COUNTS (the population rule drops rows without gold)
ev = [int(x) for x in cd.eval_rows()]
np.random.seed(0)                                            # replay_cache.build: the first RNG use, shuffle, first EVAL_CAP, sorted
np.random.shuffle(ev)
pop = sorted(ev[:EVAL_CAP])
ng_all = np.diff(gold_ptr)[np.asarray(pop, np.int64)]
pop = [r for r, n in zip(pop, ng_all) if n > 0]
qids = [cd.query_ids[r] for r in pop]
POPV = {"row_query_ids_sha256": sha_text(",".join(qids)), "rows_sha256": sha_text(",".join(str(r) for r in pop)), "n": len(pop)}
par = np.array([int(hashlib.sha1(q.encode("utf-8")).hexdigest()[:8], 16) & 1 for q in qids], np.int8)
rows_A = np.asarray([r for r, p in zip(pop, par) if p == 0], np.int64)
POPV["n_A"], POPV["n_B"] = int(len(rows_A)), int((par == 1).sum())
want = dict(POP[ds])
if want["rows_sha256"] is None and not DRY:
    want["rows_sha256"] = pre["population"][ds]["rows_sha256"]
for k_ in ("row_query_ids_sha256", "rows_sha256", "n", "n_A", "n_B"):
    assert want[k_] is None or POPV[k_] == want[k_], "population %s differs: %s" % (k_, POPV[k_])
POPV["rows_A_sha256"] = sha_text(",".join(str(int(r)) for r in rows_A))
log("%s population reproduced: %s (split B sealed)" % (ds, json.dumps(POPV)))
fpG_json = os.path.join(OUT, "flath2_G_%s.json" % ds)
fpG_npz = os.path.join(OUT, "flath2_G_%s.npz" % ds)

# ---------------------------------------------------------------- stage G (gold-free)
if MODE == "G":
    if not DRY:
        assert not os.path.exists(fpG_json) and not os.path.exists(fpG_npz), "write-once: stage G of %s exists" % ds
    if CHECK:
        log("pre-registration check passed (%s G): code, function sources, constants, records, dataset and derived pins, population; nothing written" % ds)
        sys.exit(0)
    rows_g = rows_A[:DRY_ROWS] if DRY else rows_A
    nA = len(rows_g)
    t_ = time.time()
    Qu = unit_queries(cd, rows_g)
    seedsA, d200, s200 = served_seeds(cd, rows_g)
    xo, ao, xi, ai = struct_arrays(cd, N)
    cd._csr.clear()
    deg_u = undirected_degree(cd, N)
    hub = (deg_u + 1) > cap
    STRUCT_INFO = {"directed_edges": int(len(ao)), "undirected_edges": int(deg_u.sum() // 2), "hubs (deg_u + 1 > cap)": int(hub.sum()), "cap": cap,
                   "seeds_all_hubs (queries)": int(sum(1 for j in range(nA) if hub[seedsA[j][seedsA[j] >= 0]].all()))}
    TIMES["queries_seeds_struct"] = round(time.time() - t_, 1)
    log("%s: N %d, split A %d rows%s; STRUCT %s (%.0fs, RSS %.0f MB)" % (ds, N, nA, " (DRY smoke)" if DRY else "", json.dumps(STRUCT_INFO), TIMES["queries_seeds_struct"], _rss_mb()))
    NS = beam_namespace(FSRC, Qu, cd.node_embeddings, seedsA, N, xo, ao, xi, ai, hub)
    bal1, bal2, BEAM_STATS = run_beams(NS, nA, ds)
    TIMES["balanced_beam"] = BEAM_STATS["seconds"]
    NS.clear()                                               # the exec'd functions hold NS as their globals (a cycle): free STRUCT now
    del NS, xo, ao, xi, ai, deg_u, hub
    log("%s balanced beam: %s (RSS %.0f MB)" % (ds, json.dumps(BEAM_STATS), _rss_mb()))
    t_ = time.time()
    QsA = cd.embeddings("splade", "queries").read(rows_g).tocsr()
    Es = cd.embeddings("splade", "docs")
    top, t2d, t2s, npos, FINFO = flat_bounded(Qu, QsA, cd.node_embeddings, Es, N, M_MAIN, K_TOP, n_chunks=DRY_CHUNKS if DRY else None, tag=ds)
    TIMES["bounded_flat"] = round(time.time() - t_, 1)
    agd = np.array([len(set(t2d[j, :N_AGREE].tolist()) & set(d200[j, :N_AGREE].tolist())) / float(N_AGREE) for j in range(nA)])
    kas = np.minimum(N_AGREE, npos)
    ags = np.array([(len(set(t2s[j, :kas[j]].tolist()) & set(s200[j, :kas[j]].tolist())) / float(kas[j])) if kas[j] else 1.0 for j in range(nA)])
    ex_d = np.array([(t2d[j] == d200[j]).all() for j in range(nA)])
    ex_s = np.array([npos[j] >= TA.TOP200 and (t2s[j] == s200[j]).all() for j in range(nA)])
    for j in np.flatnonzero(ex_d & ex_s):
        assert TA.node_rrf(t2d[j], t2s[j])[:TA.SEED_K] == [int(x) for x in seedsA[j] if x >= 0]
    AGREE = {"dense_top100_set_overlap_with_served_mean": round(float(agd.mean()), 5), "dense_top100_set_overlap_min": round(float(agd.min()), 3),
             "splade_top100_set_overlap_with_served_mean": round(float(ags.mean()), 5), "splade_top100_set_overlap_min": round(float(ags.min()), 3),
             "rows_dense_top200_order_identical_to_served": int(ex_d.sum()), "rows_splade_top200_order_identical_to_served": int(ex_s.sum()),
             "rows_where_both_are_identical (seeds from the exhaustive lists == served seeds, asserted)": int((ex_d & ex_s).sum()),
             "served_lists": "the frozen H100 retrieval caches (TF32 products); the exhaustive lists here are fp32",
             "splade_positive_nodes_per_query": stats(npos)}
    log("%s served-list agreement: %s" % (ds, json.dumps(AGREE)))
    if not DRY:
        assert AGREE["dense_top100_set_overlap_with_served_mean"] >= AGREE_MIN and AGREE["splade_top100_set_overlap_with_served_mean"] >= AGREE_MIN, \
            "the exhaustive rankings disagree with the served retrieval caches (wrong vectors or row mapping)"
    bals = [bal1[j] + bal2[j] for j in range(nA)]
    b1s = [set(b) for b in bal1]
    PATCH = {str(M): patch_summary(evaluate(top, bals, b1s, np.full(nA, M))) for M in M_CURVE}
    log("%s gold-free patch statistics: %s" % (ds, json.dumps({m: [PATCH[m]["patch_nodes (novel beam nodes served)"]["mean"], PATCH[m]["beam_nodes_already_in_the_semantic_base"]["mean"]] for m in PATCH})))
    bal_arr = np.full((nA, 2 * BEAM), -1, np.int32)
    bal_len = np.zeros((nA, 2), np.int32)
    for j in range(nA):
        bal_arr[j, :len(bals[j])] = bals[j]
        bal_len[j] = [len(bal1[j]), len(bal2[j])]
    arrays = {"rows": np.asarray(rows_g, np.int64), "flat_top": top.astype(np.int32), "bal": bal_arr, "bal_len": bal_len,
              "seeds": seedsA.astype(np.int32), "npos": npos, "top200_dense": t2d.astype(np.int32), "top200_splade": t2s.astype(np.int32)}
    res = {"dataset": ds, "stage": "G (gold-free)", "dry": DRY, "eval_split": cd.eval_split, "N": N, "npart_equivalent (N // 100)": npart_eq,
           "mode": "PREREGISTERED_FLAT_BALANCED_H2_FRESH_SPLIT_A", "population": POPV, "n_split_A_rows_run": nA,
           "DATASET_json_RECORD_SHA256": cd.record_sha, "preregistration": {"path": rel(PRE), "sha256": pre_sha},
           "code": {"path": rel(os.path.abspath(__file__)), "sha256": CODE_SHA}, "pinned": PINNED, "pinned_repo": PINNED_REPO,
           "function_sources_sha256": FSRC_SHA, "constants": CONSTANTS,
           "platform": {"python": platform.python_version(), "numpy": np.__version__}, "host_at_start": HOST0,
           "struct": STRUCT_INFO, "balanced_beam (split A)": BEAM_STATS, "bounded_flat": FINFO, "served_list_agreement": AGREE,
           "patch_gold_free (per M)": PATCH, "npz": {"path": rel(fpG_npz), "arrays": sorted(arrays)}}
    if DRY:
        if DRY_OUT:
            np.savez(os.path.join(DRY_OUT, os.path.basename(fpG_npz)), **arrays)
            res["npz"]["sha256"] = sha_file(os.path.join(DRY_OUT, os.path.basename(fpG_npz)))
        finish(res, fpG_json)
    np.savez(fpG_npz, **arrays)
    res["npz"]["sha256"] = sha_file(fpG_npz)
    finish(res, fpG_json)

# ---------------------------------------------------------------- stage E (gold)
fp_out = os.path.join(OUT, "flath2_E_%s.json" % ds)
gdir = DRY_IN if DRY else OUT
gj, gz = os.path.join(gdir, os.path.basename(fpG_json)), os.path.join(gdir, os.path.basename(fpG_npz))
if not DRY:
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
if CHECK:
    log("pre-registration check passed (%s E): code, function sources, constants, records, dataset and derived pins, population; nothing written" % ds)
    sys.exit(0)
grec = json.load(open(gj, encoding="utf-8"))
assert grec["dry"] == DRY and grec["code"]["sha256"] == CODE_SHA and grec["population"] == POPV and grec["constants"] == CONSTANTS
assert grec["npz"]["sha256"] == sha_file(gz), "the stage-G arrays changed"
Z = np.load(gz, allow_pickle=False)
rows_g = Z["rows"].astype(np.int64)
nA = len(rows_g)
assert (rows_g == (rows_A[:DRY_ROWS] if DRY else rows_A)).all()
top = Z["flat_top"].astype(np.int64)
bal_arr, bal_len = Z["bal"], Z["bal_len"]
bal1 = [[int(x) for x in bal_arr[j, :bal_len[j, 0]]] for j in range(nA)]
bals = [[int(x) for x in bal_arr[j, :bal_len[j, 0] + bal_len[j, 1]]] for j in range(nA)]
b1s = [set(b) for b in bal1]
gp, gs, hop_all = zq["gold_ptr"], zq["gold_pos"], zq["hop"]
golds = [np.asarray(gs[gp[r]:gp[r + 1]], np.int64) for r in rows_g]
assert all(len(g) for g in golds)
hopsA = np.asarray(hop_all)[rows_g].astype(np.int64)
if DRY:
    rng_dry = np.random.default_rng(12345)
    golds = [np.sort(rng_dry.choice(N, size=len(g), replace=False)).astype(np.int64) for g in golds]
ngold = np.array([len(g) for g in golds], np.int64)
ST_ = strata_masks(hopsA, ngold)
t_ = time.time()
CURVE, RR = {}, {}
for M in M_CURVE:
    r_ = evaluate(top, bals, b1s, np.full(nA, M), golds)
    RR[M] = r_
    assert patch_summary(r_) == grec["patch_gold_free (per M)"][str(M)], "patch statistics differ from stage G"
    CURVE[str(M)] = {"FLAT_ALL": q4(r_["flat_all"].mean()), "FLAT_H2_ALL": q4(r_["h2_all"].mean()),
                     "FLAT_ANY": q4(r_["flat_any"].mean()), "FLAT_H2_ANY": q4(r_["h2_any"].mean()),
                     "FLAT vs FLAT_H2": compare("FLAT@%d" % M, r_["flat_all"], "FLAT_BALANCED_H2@%d" % M, r_["h2_all"], WORDS_PATCH, {} if M != M_MAIN else ST_),
                     "gold_accounting": gold_accounting(r_)}
D1 = CURVE[str(M_MAIN)]["FLAT vs FLAT_H2"]
TIMES["arms"] = round(time.time() - t_, 1)
if not DRY:
    o = D1["paired (gained = arm covers ALL gold, reference does not)"]
    log("D1 %s: FLAT@%d %.4f vs FLAT_BALANCED_H2@%d %.4f (%+.2f pts) +%d/-%d p=%.3g %s" % (ds, M_MAIN, D1["ALL_reference"], M_MAIN, D1["ALL_arm"], D1["points_arm_minus_reference"],
                                                                                        o["gained"], o["lost"], o["p"], D1["verdict"]))
    log("curve (FLAT / FLAT_H2 ALL): %s" % json.dumps({m: [CURVE[m]["FLAT_ALL"], CURVE[m]["FLAT_H2_ALL"]] for m in CURVE}))
res = {"dataset": ds, "stage": "E", "dry": DRY, "eval_split": cd.eval_split, "N": N, "n_split_A": nA,
       "mode": "PREREGISTERED_FLAT_BALANCED_H2_FRESH_SPLIT_A", "status": "PRE-REGISTERED ONE-SHOT TRANSFER (split A; split B sealed)",
       "population": POPV, "stage_G": {"json": {"path": rel(fpG_json), "sha256": sha_file(gj)}, "npz": {"path": rel(fpG_npz), "sha256": grec["npz"]["sha256"]}},
       "preregistration": {"path": rel(PRE), "sha256": pre_sha}, "code": {"path": rel(os.path.abspath(__file__)), "sha256": CODE_SHA},
       "platform": {"python": platform.python_version(), "numpy": np.__version__}, "host_at_start": HOST0,
       "gold": "DRY RUN: STAND-IN gold nodes" if DRY else "query_index.npz gold positions (stamp-checked), split-A rows only",
       "gold_nodes_per_query": stats(ngold), "hops_available": sorted(set(int(x) for x in hopsA if x >= 0)),
       "D1 (pre-registered: FLAT@%d vs FLAT_BALANCED_H2@%d; gained = the patch helps)" % (M_MAIN, M_MAIN): D1,
       "curve (fixed M; M_MAIN is D1, the others descriptive)": CURVE,
       "patch_at_M_MAIN": patch_summary(RR[M_MAIN]),
       "_ind_FLAT_%d" % M_MAIN: RR[M_MAIN]["flat_all"].astype(int).tolist(), "_ind_FLAT_H2_%d" % M_MAIN: RR[M_MAIN]["h2_all"].astype(int).tolist(),
       "_n_novel_%d" % M_MAIN: RR[M_MAIN]["n_novel"].tolist()}
finish(res, fp_out)
