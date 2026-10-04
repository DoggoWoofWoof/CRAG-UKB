"""L1 DEVELOPMENT harness -- static structural localization (LOC) vs flat semantic retrieval, node level, budget curves.

Workflow (the user's ruling of 2026-09-26): development -> mechanism selection -> ONE final held-out evaluation of the frozen
architecture.  MetaQA, MuSiQue and SQuAD are DEVELOPMENT datasets (historically exposed, freely reusable); nothing computed
here is a confirmatory test and no p-value below is a verdict.  L1 = static localization: no H2, no PPR, no traversal and no
learned weight anywhere in these comparisons.

FLAT_RRF     the section-29 node RRF over all N corpus nodes (the pinned rrf_full of _l1c_flath2.py): exhaustive fp32 dense
             products (unit queries x fp16->fp32 node chunks) and fp32 SPLADE products; fv(v) = 1/(K0 + p_dense) +
             1/(K0 + p_splade), 0-based positions, K0 = 60; descending, ties -> dense order
A(B)         the sum of fv(v) over v in FLAT_RRF[:ACT] with P(v) = B, ACT = 200; P = a frozen hard single-owner partition
LOC order    every node u of an activated block (A(P(u)) > 0), sorted by (-A(P(u)), FLAT rank of u)
Arms (each serves exactly M nodes, so every comparison is at equal node exposure; M in M_CURVE)
    FLAT       FLAT_RRF[:M]
    LOC        the LOC order[:M]; when the LOC order is shorter than M, FLAT_RRF fills (unserved nodes, FLAT order)
    FLAT+LOC   node-level rank fusion f(u) = 1/(K0 + FLAT rank(u)) + [u in LOC] / (K0 + LOC rank(u)); descending, ties ->
               FLAT rank; [:M]  (parameter-free: the frozen K0, no weight)
    SLOT_C     M = 5000 only: the HARD_LOC_A interface of ruling 1 -- FLAT_RRF[:M - C] + the first C LOC nodes not in it +
               FLAT fill, C in C_CURVE (the pinned section-30 arm_row)
Scores per arm and M: ALL-gold recall (primary), ANY-gold, mean fractional gold recall; per hop, per gold-count bucket
(answer cardinality), MuSiQue per source; paired vs FLAT@M (queries gained / lost, exact McNemar p: descriptive only); gold
nodes gained / lost; the FLAT budget that matches the arm's ALL; the A/B/C audit of every FLAT@M-missed gold node against the
LOC order at the same M; the FLAT ranks of the gold nodes LOC recovers; per-query latency of this CPU implementation; index and
structure bytes.  Exact per-gold positions under every arm go to the .npz, so any budget can be re-read without recomputing.

Modes
    POP                  the development populations -> results/L1_DEV/loc_population.json (write-once)
    RUN <dataset> <tag>  every cell of the dataset -> results/L1_DEV/loc_<dataset>__<tag>.json + .npz (write-once per tag)
    --rows=K             RUN on the first K population rows only (smoke); needs --out=<dir outside the repository>
"""
import ast
import collections
import hashlib
import json
import os
import platform
import re
import sys
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
log = G.log
PINNED = {"_l1c_flath2.py": "5a54b2f906336d5ce97ba5c0c92353afc7b76dc04272ae2dd2851bb572e0758b",
          "_l1c_transfer_blockmax.py": "f9232ef95b5f317bef85cb1f7304569ea4b00bf6d283a99fc8abe153ad239ee9",
          "_l1g_core.py": "05923d66d5754f1573414fff0d119b32d93e877e8014b26e970e7ff74dc61327",
          "_l1s_core.py": "ae3f087d0a785dc4d5d1c10367fc4cc21db0a4bb353efbe0265172b199492861",
          "_l1x90_core.py": "6eba63051165588c53900d98ec5644782ad107206f05a32ba6500d16cda76068",
          "_ta_prepartition.py": "271f8985f39c090a2bb996a8deeb056cc085a1010e28ba32231bb4f413bbfa07"}
PINNED_REPO = {"src/l1_canonical/adapter.py": "aaa3304dc8d7266fdbc59d3f58a77d6956b740efb842f8e90364dc39750f4c01",
               "data/final_canonical/canonical.py": "c657830b304afe861893b516726af419d35142334f866517b83221427a9016f5"}
DEV_CACHES = {"metaqa": ["metaqa", "metaqa_phg"], "squad": ["squad", "squad_phg"], "musique": ["musique"]}
TAG_OF = {"metaqa": "H4_SK", "metaqa_phg": "LOWMEM__PHG_REPAIR1_con", "squad": "H4_SK", "squad_phg": "LOWMEM__PHG_con",
          "musique": "LOWMEM__PHG_C1_con"}
CELLS = DEV_CACHES
K0 = TA.K0                                  # 60, the frozen RRF constant
ACT = TA.TOP200                             # 200, the activation depth (the frozen served-list depth)
CQ, BLOCK = TB.CQ, TB.BLOCK                 # 200 query rows per product batch, 40000 nodes per chunk (= the SPLADE shard size)
M_CURVE = (100, 250, 500, 1000, 2000, 5000)
MMAX = max(M_CURVE)
C_CURVE = (25, 50, 100, 200, 500)           # SLOT_C at M = MMAX (ruling 1: C = 100 at M = 5000)
N_POP = AD.CONTRACT["EVAL_CAP"]             # 2000
PER_HOP = N_POP // 3                        # 666 (MetaQA, hop-balanced)
SEED = 20260926                             # the one fixed sampling seed
N_AGREE = 100
AGREE_MIN = 0.90                            # served-list sanity (wrong vectors / row mapping), not a float-noise test
BF_FIRST, BF_EVERY = 100, 10                # the brute-force LOC cross-check: the first 100 rows, then every 10th
NG_BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5-10", 5, 10), ("11+", 11, 1 << 40))
FLATH2_FNS = ("sha_file", "sha_text", "_rss_mb", "host_state", "stats", "q4", "rel", "rrf_full", "chunk_iter",
              "dense_products", "splade_products", "unit_queries", "arm_row", "paired", "strata_masks")
ARMS = ("FLAT", "LOC", "FLAT+LOC")
CATS = ("LOC_SERVES", "TYPE_A", "TYPE_B", "TYPE_C")
CONSTANTS = {"K0": K0, "ACT": ACT, "CQ": CQ, "BLOCK": BLOCK, "M_CURVE": list(M_CURVE), "C_CURVE": list(C_CURVE), "N_POP": N_POP,
             "PER_HOP": PER_HOP, "SEED": SEED, "N_AGREE": N_AGREE, "AGREE_MIN": AGREE_MIN, "NG_BUCKETS": [list(b) for b in NG_BUCKETS]}
assert K0 == 60 and ACT == 200 and MMAX == 5000 and max(C_CURVE) < MMAX - ACT


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b_ in iter(lambda: f.read(1 << 20), b""):
            h.update(b_)
    return h.hexdigest()


def pinned_sources(fn, names):
    """verbatim source text of the named top-level functions of a pinned module (each defined exactly once)."""
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
for nm_ in FLATH2_FNS:                      # the section-29/30 harness functions, verbatim, in this module's namespace
    exec(FLATH2_SRC[nm_], globals())
SRC_SHA = {k: sha_text(v) for k, v in sorted(FLATH2_SRC.items())}


def make_arm(C):
    """the pinned arm_row with C structural slots (its global C_PATCH), in its own namespace."""
    ns = {"np": np, "C_PATCH": int(C), "__builtins__": __builtins__}
    exec(FLATH2_SRC["arm_row"], ns)
    return ns["arm_row"]


ARM = {C: make_arm(C) for C in C_CURVE}


# ---------------------------------------------------------------- the partition, FLAT_RRF and the LOC order
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
    """every node of an activated block, sorted by (-A(P(u)), FLAT rank of u)."""
    cand = np.concatenate([P.nodes_of(b) for b in ub])
    Ac = np.repeat(A, P.sizes[ub])
    return cand[np.lexsort((frank[cand], -Ac))]


def loc_bruteforce(of, fv, frank, P):
    """an independent pure-Python implementation of the same definition (the periodic cross-check)."""
    Ab = {}
    for v in of[:ACT].tolist():
        b = int(P.hard[v])
        Ab[b] = Ab.get(b, 0.0) + float(fv[v])
    nodes = [u for b in Ab for u in P.nodes_of(b).tolist()]
    hl, fl = P.hard, frank
    nodes.sort(key=lambda u: (-Ab[int(hl[u])], int(fl[u])))
    return nodes, Ab


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


def picks(L, frank, C, M):
    """the first C LOC nodes outside FLAT_RRF[:M - C] (the novel slots) and the LOC index of the C-th (-1: fewer exist)."""
    idx = np.flatnonzero(frank[L] >= M - C)
    return L[idx[:C]], (int(idx[C - 1]) if len(idx) >= C else -1)


def batches(cd, rows_g, Qu, N):
    """(j0, j1, SD, SS, seconds, splade_nnz): the full dense / SPLADE products of CQ population rows over all N nodes (the
    section-30 fresh-path product shapes: pinned dense_products / splade_products per TB.BLOCK chunk == SPLADE shard)."""
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
    """the stored query index, read-only: its stamp must match the dataset record (else the adapter would rebuild and write
    it under data/, which this harness never does)."""
    zq = np.load(cd._query_index_path(), allow_pickle=False)
    assert json.loads(str(zq["meta_json"])).get("DATASET_json_RECORD_SHA256") == cd.record_sha, "stale query index: %s" % cd.name
    return zq


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
    """ALL / ANY / fraction of gold served per query at budget M, from exact 0-based per-gold positions."""
    sv = pos < M
    cnt = np.add.reduceat(sv.astype(np.int64), gptr[:-1])
    return cnt == ngold, cnt > 0, cnt / ngold.astype(np.float64), sv


def flat_budget_matching(maxrank_flat, target):
    """the smallest FLAT budget M' with ALL_FLAT(M') >= target (maxrank = 0-based deepest gold FLAT rank per query)."""
    n = len(maxrank_flat)
    need = int(np.ceil(target * n - 1e-9))
    if need <= 0:
        return 0
    return int(np.sort(maxrank_flat)[need - 1]) + 1


# ---------------------------------------------------------------- arguments
CODE_SHA = sha_file(os.path.abspath(__file__))
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
assert MODE in ("POP", "RUN"), "usage: POP | RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
ROWS = next((int(a_.split("=", 1)[1]) for a_ in sys.argv if a_.startswith("--rows=")), None)
SMOKE_OUT = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--out=")), None)
assert (ROWS is None) == (SMOKE_OUT is None), "--rows and --out go together (smoke runs write outside the repository only)"
assert SMOKE_OUT is None or not os.path.abspath(SMOKE_OUT).lower().startswith(os.path.abspath(REPO).lower())
HOST0 = host_state()
T_ALL = time.time()
TIMES = {}
os.makedirs(OUT, exist_ok=True)


def common_record():
    return {"code": {"path": rel(os.path.abspath(__file__)), "sha256": CODE_SHA}, "pinned": PINNED, "pinned_repo": PINNED_REPO,
            "function_sources_sha256": SRC_SHA, "constants": CONSTANTS, "platform": {"python": platform.python_version(), "numpy": np.__version__},
            "host_at_start": HOST0, "workflow": "L1 DEVELOPMENT (user ruling 2026-09-26): development -> mechanism selection -> one final held-out "
                                                "evaluation; MetaQA / MuSiQue / SQuAD are development datasets; no p-value here is a verdict"}


def peak_rss_mb():
    try:
        import psutil
        mi = psutil.Process(os.getpid()).memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)
    except Exception:
        return None


def write(res, fp_out, arrays=None):
    res["seconds_by_stage"] = TIMES
    res["seconds"] = round(time.time() - T_ALL, 1)
    res["process_peak_rss_mb"] = peak_rss_mb()
    assert sha_file(os.path.abspath(__file__)) == CODE_SHA, "the module file changed during the run"
    if SMOKE_OUT is not None:
        fp_out = os.path.join(SMOKE_OUT, os.path.basename(fp_out))
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    if arrays is not None:
        fz = fp_out.replace(".json", ".npz")
        assert not os.path.exists(fz), "write-once: %s exists" % fz
        np.savez_compressed(fz, **arrays)
        res["npz"] = {"path": os.path.basename(fz), "sha256": sha_file(fz), "arrays": sorted(arrays)}
    G.S.wj(fp_out, res)
    log("done (%.0fs, peak RSS %s MB) -> %s sha256 %s" % (res["seconds"], res["process_peak_rss_mb"], fp_out, sha_file(fp_out)[:12]))


# ======================================================================== POP
if MODE == "POP":
    assert ROWS is None
    assert not os.path.exists(POP_FILE), "write-once: %s exists" % POP_FILE
    RULE = ("Development population (user ruling 2026-09-26: MetaQA, MuSiQue and SQuAD are development datasets -- historically exposed, "
            "freely reusable, never a confirmatory population).  Obvious in-tree exclusions only: the rows of the DEV_A/DEV_B replay caches "
            "(the rows sections 14-30 iterated on) and of the LEGACY_CONTINUITY lanes; TEST is never read.  Eligible = rows of the evaluation "
            "split (dev) with >= 1 resolved gold (MetaQA: hop 1-3); MuSiQue tops up with train rows under the same rule.  One fixed seed "
            "20260926: a fresh numpy RandomState(seed) permutation of each sorted eligible list; MetaQA the first 666 of each hop, SQuAD "
            "the first 2000, MuSiQue every eligible dev row + the first (2000 - n_dev) eligible train rows.  Rows sorted ascending.")
    res = {"mode": "L1_DEVELOPMENT_POPULATION", "label": "development population (historically exposed; not held-out; freely reusable)",
           "rule": RULE, "seed": SEED, "populations": {}, "exclusions": {}}
    for ds in ("metaqa", "squad", "musique"):
        cd = AD.CanonicalDataset(ds)
        assert cd.eval_split == "dev", (ds, cd.eval_split)
        zq = qindex(cd)
        gp = zq["gold_ptr"].astype(np.int64)
        ng = gp[1:] - gp[:-1]
        row_of = {q: i for i, q in enumerate(cd.query_ids)}
        excl, src = set(), {}
        for c in DEV_CACHES[ds]:
            r_ = np.load(G.X.CACHES[c], allow_pickle=True)["rows"].astype(np.int64)
            src["replay_cache " + c] = {"path": rel(G.X.CACHES[c]), "sha256": sha_file(G.X.CACHES[c]), "rows": int(len(r_))}
            excl |= set(r_.tolist())
        lane = os.path.join(cd.dir, "queries", "lanes", "LEGACY_CONTINUITY.jsonl")
        with open(lane, encoding="utf-8") as f_:
            lr = [row_of[json.loads(line)["query_id"]] for line in f_ if line.strip()]
        src["LEGACY_CONTINUITY lane"] = {"path": rel(lane), "sha256": sha_file(lane), "rows": len(lr)}
        excl |= set(lr)
        a, b = cd.split_ranges["dev"]
        dev = [r for r in range(a, b) if ng[r] >= 1 and r not in excl]
        comp = {"dev_rows": b - a, "dev_rows_with_gold": int((ng[a:b] >= 1).sum()), "dev_rows_excluded": len(set(range(a, b)) & excl),
                "dev_eligible": len(dev)}
        if ds == "metaqa":
            hop = zq["hop"].astype(np.int64)
            rows = []
            for h in (1, 2, 3):
                el = np.array(sorted(r for r in dev if hop[r] == h), np.int64)
                assert len(el) >= PER_HOP, (h, len(el))
                perm = np.random.RandomState(SEED).permutation(len(el))
                rows += el[perm[:PER_HOP]].tolist()
                comp["hop%d_eligible" % h] = int(len(el))
            comp["rows_other_hops_or_no_hop"] = int(sum(1 for r in dev if hop[r] not in (1, 2, 3)))
        elif ds == "squad":
            el = np.array(sorted(dev), np.int64)
            assert len(el) >= N_POP
            perm = np.random.RandomState(SEED).permutation(len(el))
            rows = el[perm[:N_POP]].tolist()
        else:
            a2, b2 = cd.split_ranges["train"]
            tr = np.array(sorted(r for r in range(a2, b2) if ng[r] >= 1 and r not in excl), np.int64)
            k = N_POP - len(dev)
            assert 0 < k <= len(tr)
            perm = np.random.RandomState(SEED).permutation(len(tr))
            rows = list(dev) + tr[perm[:k]].tolist()
            comp.update({"train_rows": b2 - a2, "train_rows_excluded": len(set(range(a2, b2)) & excl), "train_eligible": int(len(tr)),
                         "train_sampled": int(k), "dev_sampled": len(dev)})
        rows = sorted(int(r) for r in rows)
        assert len(set(rows)) == len(rows) and not (set(rows) & excl) and all(ng[r] >= 1 for r in rows)
        qids = [cd.query_ids[r] for r in rows]
        res["populations"][ds] = {"n": len(rows), "rows": rows, "query_ids": qids, "query_ids_sha256": sha_text(",".join(qids)),
                                  "rows_sha256": sha_text(",".join(str(r) for r in rows)), "composition": comp,
                                  "dataset_pins": {"DATASET_json_RECORD_SHA256": cd.record_sha, "query_index.npz": sha_file(cd._query_index_path())}}
        res["exclusions"][ds] = src
        log("POP %s: %d rows %s" % (ds, len(rows), json.dumps(comp)))
        del row_of, zq
    res.update(common_record())
    write(res, POP_FILE)
    sys.exit(0)


# ======================================================================== RUN
ds, tag = sys.argv[2], sys.argv[3]
assert ds in CELLS and re.match(r"^[A-Za-z0-9_.-]+$", tag)
fp_out = os.path.join(OUT, "loc_%s__%s.json" % (ds, tag))
if SMOKE_OUT is None:
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
cd = AD.CanonicalDataset(ds)
N = int(cd.n_nodes)
cells = CELLS[ds]
prec = json.load(open(POP_FILE, encoding="utf-8"))
pp = prec["populations"][ds]
rows_g = np.asarray(pp["rows"], np.int64)
assert [cd.query_ids[int(r)] for r in rows_g] == pp["query_ids"] and sha_text(",".join(pp["query_ids"])) == pp["query_ids_sha256"]
assert pp["dataset_pins"] == {"DATASET_json_RECORD_SHA256": cd.record_sha, "query_index.npz": sha_file(cd._query_index_path())}
if ROWS is not None:
    rows_g = rows_g[:ROWS]
nq = len(rows_g)
qids = [cd.query_ids[int(r)] for r in rows_g]
POPV = {"file": {"path": rel(POP_FILE), "sha256": sha_file(POP_FILE)}, "n": nq, "smoke_first_rows": ROWS,
        "query_ids_sha256": sha_text(",".join(qids)), "label": prec["label"]}
parts = {c: Part(cd, TAG_OF[c]) for c in cells}
qindex(cd)
golds = [np.asarray(g, np.int64) for g in cd.gold(rows_g)]
ngold = np.array([len(g) for g in golds], np.int64)
assert (ngold >= 1).all()
gptr = np.zeros(nq + 1, np.int64)
gptr[1:] = np.cumsum(ngold)
ng_tot = int(gptr[-1])
gold_flat = np.concatenate(golds)
row_of_gold = np.repeat(np.arange(nq), ngold)
hops = hops_of(cd, rows_g, qids)
ST = strata_masks(hops, ngold)
if ds == "musique":
    dv0, dv1 = cd.split_ranges["dev"]
    ST["per_source"] = {"dev": (rows_g >= dv0) & (rows_g < dv1), "train": (rows_g < dv0) | (rows_g >= dv1)}
log("RUN %s %s: N %d, %d rows, %d gold nodes, cells %s%s" % (ds, tag, N, nq, ng_tot, cells, " (SMOKE)" if ROWS else ""))
# ---- per-gold exact positions (0-based) under every arm, and the LOC-order facts the audit needs
POS_FLAT = np.zeros(ng_tot, np.int64)
POS = {c: {"LOC": np.zeros(ng_tot, np.int64), "FLAT+LOC": np.zeros(ng_tot, np.int64)} for c in cells}
LPOS = {c: np.full(ng_tot, -1, np.int64) for c in cells}          # position in the LOC order (-1: block not activated)
BFIRST = {c: np.full(ng_tot, -1, np.int64) for c in cells}        # LOC position of the gold block's first node (-1: not activated)
BRANK = {c: np.zeros(ng_tot, np.int64) for c in cells}            # block rank by A, 1-based (0: not activated)
AOF = {c: np.zeros(ng_tot, np.float64) for c in cells}            # A(P(g))
SLOT = {c: np.zeros((ng_tot, len(C_CURVE)), bool) for c in cells}
ROWF = {c: {k: np.zeros(nq, np.int64) for k in ("n_blocks_activated", "loc_len", "top_block_nodes_in_flat_top200")} for c in cells}
TOPSHARE = {c: np.zeros(nq) for c in cells}
LAT = {"products_amortized": [], "flat_rrf": [], **{"loc_" + c: [] for c in cells}, **{"fusion_" + c: [] for c in cells}}
agd, ags, npos_all = np.zeros(nq), np.zeros(nq), np.zeros(nq, np.int64)
d200 = np.asarray(cd.dense_topk(ACT, rows_g), np.int64)
s200 = np.asarray(cd.splade_topk(ACT, rows_g), np.int64)
n_bf = 0
splade_nnz = None
t_ = time.time()
Qu = unit_queries(cd, rows_g)
for j0, j1, SD, SS, sec, nnz in batches(cd, rows_g, Qu, N):
    splade_nnz = nnz
    LAT["products_amortized"] += [sec / float(j1 - j0)] * (j1 - j0)
    for i in range(j1 - j0):
        j = j0 + i
        t0 = time.perf_counter()
        of, od, os_, npos_j, fv, frank = flat_row(SD[i], SS[i])
        LAT["flat_rrf"].append(time.perf_counter() - t0)
        npos_all[j] = npos_j
        k_ = min(N_AGREE, npos_j)
        agd[j] = len(set(od[:N_AGREE].tolist()) & set(d200[j, :N_AGREE].tolist())) / float(N_AGREE)
        ags[j] = (len(set(os_[:k_].tolist()) & set(s200[j, :k_].tolist())) / float(k_)) if k_ else 1.0
        g = golds[j]
        sl = slice(gptr[j], gptr[j + 1])
        POS_FLAT[sl] = frank[g]
        for c in cells:
            P = parts[c]
            t0 = time.perf_counter()
            ub, A = activation(of[:ACT], fv, P.hard)
            L = loc_order(ub, A, frank, P)
            LAT["loc_" + c].append(time.perf_counter() - t0)
            if j < BF_FIRST or j % BF_EVERY == 0:
                bf, Ab = loc_bruteforce(of, fv, frank, P)
                assert bf == L.tolist() and sorted(Ab) == ub.tolist() and [Ab[int(b)] for b in ub] == A.tolist(), "LOC order != brute force (%s row %d)" % (c, j)
                n_bf += 1
            nL = len(L)
            assert nL == int(P.sizes[ub].sum()) and len(np.unique(L)) == nL
            lrank = np.full(N, -1, np.int64)
            lrank[L] = np.arange(nL)
            lp = lrank[g]
            LPOS[c][sl] = lp
            A_of = dict(zip(ub.tolist(), A.tolist()))
            for t, (x, l_) in enumerate(zip(g.tolist(), lp.tolist())):
                if l_ >= 0:
                    b = int(P.hard[x])
                    BFIRST[c][gptr[j] + t] = int(lrank[P.nodes_of(b)].min())
                    AOF[c][gptr[j] + t] = A_of[b]
                    BRANK[c][gptr[j] + t] = 1 + int((A > A_of[b]).sum())
            sf = np.sort(frank[L])                               # LOC arm: the LOC order, then FLAT fill of the rest
            POS[c]["LOC"][sl] = np.where(lp >= 0, lp, nL + frank[g] - np.searchsorted(sf, frank[g]))
            t0 = time.perf_counter()
            f = 1.0 / (K0 + frank.astype(np.float64))
            f[L] += 1.0 / (K0 + np.arange(nL, dtype=np.float64))
            FO = np.lexsort((frank, -f))
            LAT["fusion_" + c].append(time.perf_counter() - t0)
            frk = np.empty(N, np.int64)
            frk[FO] = np.arange(N)
            POS[c]["FLAT+LOC"][sl] = frk[g]
            ol = of[:MMAX]
            for ci, C in enumerate(C_CURVE):
                nov, cut = picks(L, frank, C, MMAX)
                bset, tail, novel, fill = ARM[C](ol, L[:cut + 1].tolist() if cut >= 0 else L.tolist(), MMAX)
                assert [int(x) for x in novel] == nov.tolist()
                served = bset | set(int(x) for x in novel) | set(fill)
                SLOT[c][sl, ci] = [int(x) in served for x in g]
            ROWF[c]["n_blocks_activated"][j], ROWF[c]["loc_len"][j] = len(ub), nL
            ROWF[c]["top_block_nodes_in_flat_top200"][j] = int((P.hard[of[:ACT]] == ub[int(np.argmax(A))]).sum())
            TOPSHARE[c][j] = float(A.max() / A.sum())
            del lrank, f, FO, frk
    log("  %s rows %d / %d (%.0fs, RSS %.0f MB)" % (ds, j1, nq, time.time() - t_, _rss_mb()))
TIMES["flat_loc_fusion"] = round(time.time() - t_, 1)
AGREE = {"dense_top100_set_overlap_with_served_mean": round(float(agd.mean()), 5), "dense_top100_set_overlap_min": round(float(agd.min()), 3),
         "splade_top100_set_overlap_with_served_mean": round(float(ags.mean()), 5), "splade_top100_set_overlap_min": round(float(ags.min()), 3),
         "served_lists": "the frozen retrieval caches (TF32 products); the exhaustive lists here are fp32",
         "splade_positive_nodes_per_query": stats(npos_all)}
log("%s served-list agreement: %s" % (ds, json.dumps(AGREE)))
assert AGREE["dense_top100_set_overlap_with_served_mean"] >= AGREE_MIN and AGREE["splade_top100_set_overlap_with_served_mean"] >= AGREE_MIN, \
    "the exhaustive rankings disagree with the served retrieval caches (wrong vectors or row mapping)"


# ---------------------------------------------------------------- scores
def block(all_, any_, frac, sv, ref=None):
    o = {"ALL": q4(all_.mean()), "ANY": q4(any_.mean()), "FRAC": q4(frac.mean())}
    if ref is not None:
        rall, rsv = ref
        o["paired_vs_FLAT (gained = arm serves ALL gold, FLAT does not; McNemar p descriptive)"] = paired(rall, all_)
        o["gold_nodes_gained_vs_FLAT"] = int((sv & ~rsv).sum())
        o["gold_nodes_lost_vs_FLAT"] = int((~sv & rsv).sum())
    return o


def strata_block(all_, any_, frac, ref_all=None):
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


maxrank_flat = np.maximum.reduceat(POS_FLAT, gptr[:-1])
FL = {}
FLAT_RES = {}
for M in M_CURVE:
    a_, y_, f_, sv = per_query(POS_FLAT, M, gptr, ngold)
    FL[M] = (a_, y_, f_, sv)
    FLAT_RES[str(M)] = dict(block(a_, y_, f_, sv), nodes_exposed=M, strata=strata_block(a_, y_, f_))
CELLRES = {}
for c in cells:
    P = parts[c]
    arms_out = {}
    for arm in ("LOC", "FLAT+LOC"):
        arms_out[arm] = {}
        for M in M_CURVE:
            a_, y_, f_, sv = per_query(POS[c][arm], M, gptr, ngold)
            e = block(a_, y_, f_, sv, ref=(FL[M][0], FL[M][3]))
            e["nodes_exposed"] = M
            e["FLAT_budget_matching_this_ALL"] = flat_budget_matching(maxrank_flat, float(a_.mean()))
            e["strata"] = strata_block(a_, y_, f_, FL[M][0])
            arms_out[arm][str(M)] = e
    slot = {}
    a5, y5, f5, sv5 = FL[MMAX]
    for ci, C in enumerate(C_CURVE):
        svc = SLOT[c][:, ci]
        cnt = np.add.reduceat(svc.astype(np.int64), gptr[:-1])
        a_, y_, f_ = cnt == ngold, cnt > 0, cnt / ngold.astype(np.float64)
        slot[str(C)] = dict(block(a_, y_, f_, svc, ref=(a5, sv5)), nodes_exposed=MMAX, strata=strata_block(a_, y_, f_, a5))
    # ---- the A/B/C audit of every FLAT@M-missed gold node against the LOC order at the same M
    audit = {}
    for M in M_CURVE:
        miss = POS_FLAT >= M
        lp, bf_ = LPOS[c], BFIRST[c]
        cat = {"LOC_SERVES": miss & (lp >= 0) & (lp < M), "TYPE_A": miss & (lp < 0),
               "TYPE_B": miss & (lp >= M) & (bf_ >= M), "TYPE_C": miss & (lp >= M) & (bf_ >= 0) & (bf_ < M)}
        tot = sum(int(v.sum()) for v in cat.values())
        assert tot == int(miss.sum()), "audit categories do not partition the FLAT@M misses"
        fused = miss & (POS[c]["FLAT+LOC"] < M)
        nm = int(miss.sum())
        e = {"n_FLAT@M_missed_gold_nodes": nm, "counts": {k: int(v.sum()) for k, v in cat.items()},
             "shares": {k: (round(int(v.sum()) / float(nm), 4) if nm else None) for k, v in cat.items()},
             "FLAT+LOC@M serves (of the FLAT@M misses)": int(fused.sum()),
             "FLAT_rank_1based_of_golds_LOC@M_serves": stats(POS_FLAT[cat["LOC_SERVES"]] + 1),
             "FLAT_rank_1based_of_golds_FLAT+LOC@M_serves": stats(POS_FLAT[fused] + 1),
             "FLAT_rank_bins_of_golds_LOC@M_serves": {"(M,2M]": int((cat["LOC_SERVES"] & (POS_FLAT < 2 * M)).sum()),
                                                      "(2M,10M]": int((cat["LOC_SERVES"] & (POS_FLAT >= 2 * M) & (POS_FLAT < 10 * M)).sum()),
                                                      ">10M": int((cat["LOC_SERVES"] & (POS_FLAT >= 10 * M)).sum())},
             "TYPE_B_block_rank_by_A": stats(BRANK[c][cat["TYPE_B"]]),
             "TYPE_B_block_first_LOC_position_minus_M": stats(bf_[cat["TYPE_B"]] - M),
             "TYPE_C_gold_LOC_position_minus_block_first": stats(lp[cat["TYPE_C"]] - bf_[cat["TYPE_C"]]),
             "by_stratum": {}}
        for sn, masks in ST.items():
            e["by_stratum"][sn] = {}
            for k, m in masks.items():
                mg = m[row_of_gold]
                if (mg & miss).any():
                    e["by_stratum"][sn][k] = {"n_missed_gold_nodes": int((mg & miss).sum()), **{kk: int((vv & mg).sum()) for kk, vv in cat.items()},
                                              "FLAT+LOC_serves": int((fused & mg).sum())}
        audit[str(M)] = e
    LOCG = {"blocks_activated_per_query": stats(ROWF[c]["n_blocks_activated"]), "loc_order_length": stats(ROWF[c]["loc_len"]),
            "strongest_block_share_of_total_A": stats(TOPSHARE[c]), "flat_top200_nodes_in_the_strongest_block": stats(ROWF[c]["top_block_nodes_in_flat_top200"]),
            "gold_nodes_in_an_activated_block": q4((LPOS[c] >= 0).mean())}
    CELLRES[c] = {"partition": {"tag": TAG_OF[c], "path": rel(P.path), "sha256": sha_file(P.path), "npart": P.npart, "block_size": stats(P.sizes)},
                  "loc_gold_free": LOCG, "arms": arms_out, "SLOT_C_at_M5000 (ruling-1 interface)": slot, "audit_ABC (FLAT@M misses vs the LOC order at M)": audit,
                  "latency_ms": {"loc (activation + LOC order)": ms_stats(LAT["loc_" + c]), "fusion (f + lexsort over N)": ms_stats(LAT["fusion_" + c])},
                  "structure_bytes": {"hard_int32": 4 * N, "order_int32": 4 * N, "ptr_int64": 8 * (P.npart + 1), "partition_file": os.path.getsize(P.path)}}
    lines = ["%5d  FLAT %.4f  LOC %.4f (+%d/-%d)  FLAT+LOC %.4f (+%d/-%d)" % (
        M, FLAT_RES[str(M)]["ALL"], arms_out["LOC"][str(M)]["ALL"],
        arms_out["LOC"][str(M)]["paired_vs_FLAT (gained = arm serves ALL gold, FLAT does not; McNemar p descriptive)"]["gained"],
        arms_out["LOC"][str(M)]["paired_vs_FLAT (gained = arm serves ALL gold, FLAT does not; McNemar p descriptive)"]["lost"],
        arms_out["FLAT+LOC"][str(M)]["ALL"],
        arms_out["FLAT+LOC"][str(M)]["paired_vs_FLAT (gained = arm serves ALL gold, FLAT does not; McNemar p descriptive)"]["gained"],
        arms_out["FLAT+LOC"][str(M)]["paired_vs_FLAT (gained = arm serves ALL gold, FLAT does not; McNemar p descriptive)"]["lost"]) for M in M_CURVE]
    log("%s ALL-gold by budget:\n    %s" % (c, "\n    ".join(lines)))
    log("%s audit @5000 %s ; SLOT_100 ALL %.4f" % (c, json.dumps(audit[str(MMAX)]["counts"]), slot["100"]["ALL"]))
dim = int(Qu.shape[1])
res = {"dataset": ds, "tag": tag, "mode": "L1_DEVELOPMENT_LOC", "status": "DEVELOPMENT (not confirmatory; p-values descriptive)",
       "eval_split": cd.eval_split, "N": N, "n_rows": nq, "n_gold_nodes": ng_tot, "population": POPV, "cells": cells,
       "gold_nodes_per_query": stats(ngold), "hops_available": sorted(set(int(x) for x in hops if x >= 0)),
       "FLAT": FLAT_RES, "cells_result": CELLRES, "served_list_agreement": AGREE,
       "latency_ms (this exhaustive CPU implementation, per query)": {"flat_products_amortized (dense + SPLADE over all N)": ms_stats(LAT["products_amortized"]),
                                                                     "flat_rrf (full sort)": ms_stats(LAT["flat_rrf"])},
       "index_bytes": {"dense_fp16 (N x dim x 2)": N * dim * 2, "splade_csr (nnz x 8 + (N + 1) x 8)": int(splade_nnz) * 8 + (N + 1) * 8, "dim": dim,
                       "splade_nnz": int(splade_nnz)},
       "loc_cross_check": "vectorised LOC order == an independent pure-Python implementation on %d row-cell pairs (rows < %d and every %dth)" % (n_bf, BF_FIRST, BF_EVERY),
       "definitions": {"ALL": "every gold node of the query served", "ANY": ">= 1 gold node served", "FRAC": "mean over queries of the served fraction of gold nodes",
                       "LOC_SERVES": "FLAT@M misses the gold but its LOC-order position < M", "TYPE_A": "A(P(g)) = 0: no FLAT top-200 node in the gold's block",
                       "TYPE_B": "the block is activated but its first node sits at LOC position >= M (the region ranks too weakly for budget M)",
                       "TYPE_C": "the block has >= 1 node inside LOC[:M] but the gold is deeper in the block's FLAT order",
                       "FLAT_budget_matching_this_ALL": "the smallest M' with ALL(FLAT@M') >= the arm's ALL at M"},
       "_row_query_ids": qids}
res.update(common_record())
arrays = {"rows": rows_g, "gptr": gptr, "gold_nodes": gold_flat, "pos_FLAT": POS_FLAT, "npos": npos_all,
          **{"pos_LOC__" + c: POS[c]["LOC"] for c in cells}, **{"pos_FLATLOC__" + c: POS[c]["FLAT+LOC"] for c in cells},
          **{"lpos__" + c: LPOS[c] for c in cells}, **{"bfirst__" + c: BFIRST[c] for c in cells}, **{"brank__" + c: BRANK[c] for c in cells},
          **{"A__" + c: AOF[c] for c in cells}, **{"slot_served__" + c: SLOT[c] for c in cells},
          **{"loc_len__" + c: ROWF[c]["loc_len"] for c in cells}, **{"n_blocks__" + c: ROWF[c]["n_blocks_activated"] for c in cells}}
write(res, fp_out, arrays)
