"""Shared library of the L1 DEVELOPMENT harnesses (_l1d_*.py; user ruling 2026-09-26: development -> mechanism selection -> one
final held-out evaluation; no H2 / PPR / traversal / learned weight in any L1 comparison).

The helpers are those of scratchpad/_l1d_loc.py (the v1 hard-LOC harness, whose records pin its own sha), moved here unchanged
so later harnesses share one copy; the pinned section-29/30 functions are exec'd verbatim from _l1c_flath2.py, exactly as there.
Importing this module runs no experiment: it checks the pins and defines functions."""
import ast
import hashlib
import json
import os
import platform
import re
import time

import numpy as np

import _l1g_core as G
import _l1c_transfer_blockmax as TB
import _ta_prepartition as TA
from src.l1_canonical import adapter as AD

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = G.X.REPO
OUT = os.path.join(REPO, "results", "L1_DEV")
POP_FILE = os.path.join(OUT, "loc_population.json")
POP_SHA = "c7f7080639d6"                    # prefix of the population record written by _l1d_loc.py POP (2026-09-26)
log = G.log
PINNED = {"_l1c_flath2.py": "5a54b2f906336d5ce97ba5c0c92353afc7b76dc04272ae2dd2851bb572e0758b",
          "_l1c_transfer_blockmax.py": "f9232ef95b5f317bef85cb1f7304569ea4b00bf6d283a99fc8abe153ad239ee9",
          "_l1g_core.py": "05923d66d5754f1573414fff0d119b32d93e877e8014b26e970e7ff74dc61327",
          "_l1s_core.py": "ae3f087d0a785dc4d5d1c10367fc4cc21db0a4bb353efbe0265172b199492861",
          "_l1x90_core.py": "6eba63051165588c53900d98ec5644782ad107206f05a32ba6500d16cda76068",
          "_ta_prepartition.py": "271f8985f39c090a2bb996a8deeb056cc085a1010e28ba32231bb4f413bbfa07"}
PINNED_REPO = {"src/l1_canonical/adapter.py": "aaa3304dc8d7266fdbc59d3f58a77d6956b740efb842f8e90364dc39750f4c01",
               "data/final_canonical/canonical.py": "c657830b304afe861893b516726af419d35142334f866517b83221427a9016f5"}
TAG_OF = {"metaqa": "H4_SK", "metaqa_phg": "LOWMEM__PHG_REPAIR1_con", "squad": "H4_SK", "squad_phg": "LOWMEM__PHG_con",
          "musique": "LOWMEM__PHG_C1_con"}
CELLS = {"metaqa": ["metaqa", "metaqa_phg"], "squad": ["squad", "squad_phg"], "musique": ["musique"]}
K0 = TA.K0                                  # 60, the frozen RRF constant
ACT = TA.TOP200                             # 200, the activation depth
DEG_CAP = TA.DEG_CAP                        # 300, the frozen hub degree cap of the expansion contract
CQ, BLOCK = TB.CQ, TB.BLOCK                 # 200 query rows per product batch, 40000 nodes per chunk
M_CURVE = (100, 250, 500, 1000, 2000, 5000)
MMAX = max(M_CURVE)
N_AGREE, AGREE_MIN = 100, 0.90
NG_BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5-10", 5, 10), ("11+", 11, 1 << 40))
FLATH2_FNS = ("sha_file", "sha_text", "_rss_mb", "host_state", "stats", "q4", "rel", "rrf_full", "chunk_iter",
              "dense_products", "splade_products", "unit_queries", "paired", "strata_masks")
assert K0 == 60 and ACT == 200 and DEG_CAP == 300 and MMAX == 5000


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b_ in iter(lambda: f.read(1 << 20), b""):
            h.update(b_)
    return h.hexdigest()


def pinned_sources(fn, names):
    src = open(os.path.join(HERE, fn), "rb").read().decode("utf-8")
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            assert node.name not in out, "%s defines %s twice" % (fn, node.name)
            out[node.name] = ast.get_source_segment(src, node)
    assert set(out) == set(names), (fn, sorted(set(names) - set(out)))
    return out


for fn_, sha_ in PINNED.items():
    assert _sha_file(os.path.join(HERE, fn_)) == sha_, "pinned module %s changed" % fn_
for fn_, sha_ in PINNED_REPO.items():
    assert _sha_file(os.path.join(REPO, fn_)) == sha_, "pinned file %s changed" % fn_
FLATH2_SRC = pinned_sources("_l1c_flath2.py", FLATH2_FNS)
for nm_ in FLATH2_FNS:
    exec(FLATH2_SRC[nm_], globals())
SRC_SHA = {k: sha_text(v) for k, v in sorted(FLATH2_SRC.items())}
CONSTANTS = {"K0": K0, "ACT": ACT, "DEG_CAP": DEG_CAP, "CQ": CQ, "BLOCK": BLOCK, "M_CURVE": list(M_CURVE), "N_AGREE": N_AGREE,
             "AGREE_MIN": AGREE_MIN, "NG_BUCKETS": [list(b) for b in NG_BUCKETS]}


class Part(object):
    """one cell's frozen hard single-owner partition: P(v) = hard[v]; nodes of block b in ascending id order."""

    def __init__(self, cd, tag):
        self.tag, self.path = tag, cd.partition_path(tag)
        self.hard = cd.partition(tag)
        assert self.hard is not None and len(self.hard) == cd.n_nodes and self.hard.min() >= 0
        self.npart = int(self.hard.max()) + 1
        self.sizes = np.bincount(self.hard, minlength=self.npart).astype(np.int64)
        assert (self.sizes >= 1).all()
        self.order_nodes = np.argsort(self.hard, kind="stable")
        self.ptr = np.zeros(self.npart + 1, np.int64)
        self.ptr[1:] = np.cumsum(self.sizes)

    def nodes_of(self, b):
        return self.order_nodes[self.ptr[b]:self.ptr[b + 1]]


def activation(top_act, fv, hard):
    """A over the activated blocks: (block ids ascending, A float64); fv summed in FLAT order."""
    ub, inv = np.unique(hard[top_act], return_inverse=True)
    A = np.zeros(len(ub), np.float64)
    np.add.at(A, inv.ravel(), fv[top_act])
    return ub, A


def loc_order(ub, A, frank, P):
    """every node of an activated block, sorted by (-A(P(u)), FLAT rank of u)  (== _l1d_loc.loc_order)."""
    cand = np.concatenate([P.nodes_of(b) for b in ub])
    Ac = np.repeat(A, P.sizes[ub])
    return cand[np.lexsort((frank[cand], -Ac))]


def order_from_score(L, frank):
    """the LOC order of a node score L >= 0: every node with L > 0, sorted by (-L, FLAT rank)."""
    cand = np.flatnonzero(L > 0)
    return cand[np.lexsort((frank[cand], -L[cand]))]


def flat_row(sd, ss):
    """FLAT_RRF of one row: order (the pinned rrf_full), fv (the same formula) and the rank of every node."""
    of, od, os_, npos = rrf_full(sd, ss)
    n_ = len(sd)
    pd = np.empty(n_, np.int64)
    pd[od] = np.arange(n_)
    ps = np.full(n_, npos, np.int64)
    ps[os_] = np.arange(npos)
    fv = 1.0 / (K0 + pd) + 1.0 / (K0 + ps)
    assert (np.diff(fv[of]) <= 0).all()
    frank = np.empty(n_, np.int64)
    frank[of] = np.arange(n_)
    return of, od, os_, npos, fv, frank


def arm_positions(Lord, frank, g, N):
    """per-gold 0-based positions under (a) the LOC arm (LOC order, then FLAT fill of the unserved nodes in FLAT order) and
    (b) FLAT+LOC (f(u) = 1/(K0 + FLAT rank) + [u in LOC] / (K0 + LOC rank), descending, ties -> FLAT rank); plus the
    LOC-order position of each gold (-1: not in the LOC order).  (== the v1 harness's per-row arithmetic.)"""
    nL = len(Lord)
    lrank = np.full(N, -1, np.int64)
    lrank[Lord] = np.arange(nL)
    lp = lrank[g]
    sf = np.sort(frank[Lord])
    pos_loc = np.where(lp >= 0, lp, nL + frank[g] - np.searchsorted(sf, frank[g]))
    f = 1.0 / (K0 + frank.astype(np.float64))
    f[Lord] += 1.0 / (K0 + np.arange(nL, dtype=np.float64))
    FO = np.lexsort((frank, -f))
    frk = np.empty(N, np.int64)
    frk[FO] = np.arange(N)
    return pos_loc, frk[g], lp


def batches(cd, rows_g, Qu, N):
    """(j0, j1, SD, SS, seconds, splade_nnz): full dense / SPLADE products of CQ population rows over all N nodes."""
    QsA = cd.embeddings("splade", "queries").read(rows_g).tocsr()
    Es = cd.embeddings("splade", "docs")
    assert int(Es.shard_size) == BLOCK and int(Es.n_rows) == N
    n = len(rows_g)
    for j0 in range(0, n, CQ):
        j1 = min(n, j0 + CQ)
        t0 = time.perf_counter()
        SD = np.empty((j1 - j0, N), np.float32)
        SS = np.empty((j1 - j0, N), np.float32)
        nnz = 0
        for a, b_, blk in chunk_iter(cd.node_embeddings):
            Ec = np.ascontiguousarray(np.asarray(blk, np.float32))
            Msp = Es.shard(a // BLOCK)
            assert Msp.shape[0] == b_ - a and int(Msp.shape[1]) == int(QsA.shape[1])
            nnz += int(Msp.nnz)
            SD[:, a:b_] = dense_products(Qu, Ec, j0, j1)
            SS[:, a:b_] = splade_products(Msp, QsA, j0, j1)
            del Ec, Msp
        assert np.isfinite(SD).all() and np.isfinite(SS).all() and (SS >= 0).all()
        yield j0, j1, SD, SS, time.perf_counter() - t0, nnz
        del SD, SS


def qindex(cd):
    """the stored query index, read-only; its stamp must match the dataset record (the adapter would otherwise rebuild it
    under data/, which these harnesses never do)."""
    zq = np.load(cd._query_index_path(), allow_pickle=False)
    assert json.loads(str(zq["meta_json"])).get("DATASET_json_RECORD_SHA256") == cd.record_sha, "stale query index: %s" % cd.name
    return zq


def sk_csr(cd):
    """the undirected, deduped, loop-free STRUCT u KNN adjacency (the families of the H4_SK hypergraph), read-only: the stored
    key sets must match the dataset record (the adapter would otherwise rebuild them under data/)."""
    z = np.load(cd._keys_path())
    assert json.loads(str(z["meta_json"])).get("DATASET_json_RECORD_SHA256") == cd.record_sha, "stale key sets: %s" % cd.name
    N, STRUCT, KNN, _ = cd.keysets()
    assert N == cd.n_nodes and len(np.intersect1d(STRUCT, KNN, assume_unique=True)) == 0
    xadj, adj = cd.csr_from_keys(np.concatenate([STRUCT, KNN]), N)
    return xadj, adj.astype(np.int32), {"STRUCT_edges": int(len(STRUCT)), "KNN_edges": int(len(KNN)), "keys_file": rel(cd._keys_path()),
                                        "keys_sha256": sha_file(cd._keys_path())}


def hops_of(cd, rows_g, qids):
    if cd.name == "musique":                                 # the canonical id prefix (2hop__, 3hop1__, 4hop2__, ...)
        return np.array([int(re.match(r"^(\d)hop", q).group(1)) for q in qids], np.int64)
    return qindex(cd)["hop"][np.asarray(rows_g, np.int64)].astype(np.int64)


def ms_stats(x):
    x = np.asarray(x, np.float64) * 1000.0
    if not x.size:
        return None
    return {"mean_ms": round(float(x.mean()), 3), "median_ms": round(float(np.median(x)), 3), "p95_ms": round(float(np.percentile(x, 95)), 3)}


def per_query(pos, M, gptr, ngold):
    sv = pos < M
    cnt = np.add.reduceat(sv.astype(np.int64), gptr[:-1])
    return cnt == ngold, cnt > 0, cnt / ngold.astype(np.float64), sv


def flat_budget_matching(maxrank_flat, target):
    n = len(maxrank_flat)
    need = int(np.ceil(target * n - 1e-9))
    if need <= 0:
        return 0
    return int(np.sort(maxrank_flat)[need - 1]) + 1


def peak_rss_mb():
    try:
        import psutil
        mi = psutil.Process(os.getpid()).memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)
    except Exception:
        return None


PAIRED_KEY = "paired_vs_FLAT (gained = arm serves ALL gold, FLAT does not; McNemar p descriptive)"


def score_block(all_, any_, frac, sv, ref=None):
    o = {"ALL": q4(all_.mean()), "ANY": q4(any_.mean()), "FRAC": q4(frac.mean())}
    if ref is not None:
        rall, rsv = ref
        o[PAIRED_KEY] = paired(rall, all_)
        o["gold_nodes_gained_vs_FLAT"] = int((sv & ~rsv).sum())
        o["gold_nodes_lost_vs_FLAT"] = int((~sv & rsv).sum())
    return o


def strata_block(ST, all_, any_, frac, ref_all=None):
    out = {}
    for sn, masks in ST.items():
        out[sn] = {}
        for k, m in masks.items():
            if not m.any():
                continue
            e = {"n": int(m.sum()), "ALL": q4(all_[m].mean()), "ANY": q4(any_[m].mean()), "FRAC": q4(frac[m].mean())}
            if ref_all is not None:
                e["paired_vs_FLAT"] = paired(ref_all, all_, m)
            out[sn][k] = e
    return out


class Population(object):
    """the development population of one dataset (results/L1_DEV/loc_population.json), verified against the dataset pins;
    rows, gold nodes (flattened with pointers), hops and strata."""

    def __init__(self, cd, rows_limit=None):
        prec = json.load(open(POP_FILE, encoding="utf-8"))
        assert sha_file(POP_FILE).startswith(POP_SHA), "the population record changed"
        pp = prec["populations"][cd.name]
        rows = np.asarray(pp["rows"], np.int64)
        assert [cd.query_ids[int(r)] for r in rows] == pp["query_ids"] and sha_text(",".join(pp["query_ids"])) == pp["query_ids_sha256"]
        assert pp["dataset_pins"] == {"DATASET_json_RECORD_SHA256": cd.record_sha, "query_index.npz": sha_file(cd._query_index_path())}
        if rows_limit is not None:
            rows = rows[:rows_limit]
        self.rows = rows
        self.nq = len(rows)
        self.qids = [cd.query_ids[int(r)] for r in rows]
        qindex(cd)
        self.golds = [np.asarray(g, np.int64) for g in cd.gold(rows)]
        self.ngold = np.array([len(g) for g in self.golds], np.int64)
        assert (self.ngold >= 1).all()
        self.gptr = np.zeros(self.nq + 1, np.int64)
        self.gptr[1:] = np.cumsum(self.ngold)
        self.ng_tot = int(self.gptr[-1])
        self.row_of_gold = np.repeat(np.arange(self.nq), self.ngold)
        self.hops = hops_of(cd, rows, self.qids)
        self.ST = strata_masks(self.hops, self.ngold)
        if cd.name == "musique":
            dv0, dv1 = cd.split_ranges["dev"]
            self.ST["per_source"] = {"dev": (rows >= dv0) & (rows < dv1), "train": (rows < dv0) | (rows >= dv1)}
        self.record = {"file": {"path": rel(POP_FILE), "sha256": sha_file(POP_FILE)}, "n": self.nq, "first_rows_only": rows_limit,
                       "query_ids_sha256": sha_text(",".join(self.qids)), "label": prec["label"]}


def platform_record():
    return {"python": platform.python_version(), "numpy": np.__version__}
