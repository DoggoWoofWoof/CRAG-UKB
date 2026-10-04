"""HARD_LOC_A -- static, traversal-free structural localization vs flat semantic retrieval at a MATCHED node budget.

The one L1 mechanism test authorised by the user's rulings of 2026-09-26 (verbatim, sha256-pinned, in
results/L1_COVPART/PREREGISTRATION_HARD_LOC_A.json).  Architecture (ruling 1): L1 = static, traversal-free structural
localization; L2 = learned query-conditioned reranking; L3 = H2, PPR, bounded traversal and other dynamic graph reasoning.
The L1 baseline is FLAT; no traversal contributes to the L1 verdict.

Rankings and arms (M = M_MAIN = 5000 nodes, C = C_MAIN = 100 structural slots, activation depth ACT = TA.TOP200 = 200)
    FLAT_RRF     the section-29/30 node RRF over all N corpus nodes (the pinned rrf_full of _l1c_flath2.py): exhaustive fp32
                 dense products (unit queries x fp16->fp32 node chunks of TB.BLOCK nodes) and fp32 SPLADE products;
                 fv(v) = 1/(K0 + p_dense) + 1/(K0 + p_splade), 0-based positions, K0 = 60, float64; a node without a
                 positive SPLADE score takes p_splade = the positive-score count; descending, ties -> dense order
    FLAT@M       FLAT_RRF[:M]
    A(B)         the sum of fv(v) over v in FLAT_RRF[:ACT] with P(v) = B; P = the cell's frozen hard single-owner partition
    LOC order    every node u with A(P(u)) > 0, sorted by (-A(P(u)), FLAT rank of u)  (no learned weight, no graph walk, no
                 relation logic, no new embedding, no soft membership, no query-conditioned partition construction)
    FLAT+LOC@M   the pinned section-30 interface arm_row(FLAT_RRF, LOC order, M) with C slots: FLAT_RRF[:M - C] + the first
                 C LOC nodes not in it + FLAT_RRF[M - C:] fill (skipping those) until exactly M
    FLAT+H2@M    L3 REFERENCE ONLY (never a competing L1 arm): the same interface with the frozen section-21 balanced H2
                 beam (bal1 + bal2) at C = 100, exactly as section 30
    PRIMARY      secondary historical reference: _l1c_hardloc_primary.py, never here
Decisive comparison per cell: FLAT@M vs FLAT+LOC@M, ALL-gold, exact McNemar, alpha 0.01 -> LOC_HELPS / LOC_HURTS /
NO_SIGNIFICANT_DIFFERENCE.  Explanatory only: the G/L decomposition (a lost gold must sit at FLAT rank M-C+1..M), the
A/B/C audit of every FLAT@M-missed gold node, the C curve (C in C_CURVE at M), the H2 cross-tab, strata.

Modes (every record write-once; --dry writes nothing in the repository)
    POP                  the HARD_LOC_A populations (eligibility rule of ruling 3, sampling of ruling 2) -> hardloc_population.json
    IDENT <DEV_A cache>  identity gate on DEV_A split A (real gold, asserted against the frozen records, no number printed): the
                         section-29 FLAT_RRF (deepest gold position of every row == flat_A_<cache>.json) and FLAT@5000 /
                         FLAT+H2@5000 (== flath2_A_<cache>.json) through this module's code paths -> hardloc_ident_<cache>.json
    G <dataset>          gold-free: FLAT_RRF, per cell the activation and the LOC order (every row cross-checked against an
                         independent brute-force implementation), the H2 beam, gold-free statistics, per-row sha256 digests
                         -> hardloc_G_<dataset>.json / .npz
    E <dataset>          recomputes FLAT_RRF and every LOC order (per-row digests == stage G, asserted; the H2 beam is read
                         from stage G), then reads gold once -> hardloc_E_<dataset>.json
    SUMMARY              the pre-registered five-cell pattern -> hardloc_SUMMARY.json
    --check-prereg       verify every pin against the pre-registration, read no data, write nothing
    --dry                G / E on the first DRY_ROWS rows of the dataset's DEV_A cache population (E: STAND-IN gold of the true
                         per-row count; no gold identity is read); IDENT without a record; POP: the exclusion sources only
                         (no eligible universe, no sample); --dry-out / --dry-in = a directory outside the repository
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
OUT = os.path.join(REPO, "results", "L1_COVPART")
PRE = os.path.join(OUT, "PREREGISTRATION_HARD_LOC_A.json")
POP_FILE = os.path.join(OUT, "hardloc_population.json")
log = G.log
PINNED = {"_l1c_flath2.py": "5a54b2f906336d5ce97ba5c0c92353afc7b76dc04272ae2dd2851bb572e0758b",
          "_l1c_microl3.py": "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5",
          "_l1c_transfer_composite.py": "1241ec58e759ab99e77ff17936b6edc6b072344ea1d47a76a79f940a1047cc5b",
          "_l1c_transfer_blockmax.py": "f9232ef95b5f317bef85cb1f7304569ea4b00bf6d283a99fc8abe153ad239ee9",
          "_l1c_flat.py": "27aafbfc9d24e39e92154cd6d3ebe8364c07e32af5a8ced51a88e766bc6b3c95",
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
FLATH2_RECORD = {"metaqa": "dbc681193a3c17d13a557785ab5b425cbc0228bd0c842e2e0aef350468cb6bbc",
                 "metaqa_phg": "cffe1a1f24d5ac57156221f82091e8539f6ea459c646214a71c73c54002dec10",
                 "squad": "da0f472be2d5402b27d10774745cee509499ba7fcfc9eff44cf9d81bee99c09f",
                 "squad_phg": "3578461788bc8977c9c454b13d21530d871298b077c9ed13bc2ce6a16048e1a8",
                 "musique": "9f416ee5b2424e7ad7d0ff670401886b495d51631d2bab4cc03c1d704786a7bd"}
DEV_A_CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
DS_OF = {"metaqa": "metaqa", "metaqa_phg": "metaqa", "squad": "squad", "squad_phg": "squad", "musique": "musique"}
TAG_OF = {"metaqa": "H4_SK", "metaqa_phg": "LOWMEM__PHG_REPAIR1_con", "squad": "H4_SK", "squad_phg": "LOWMEM__PHG_con",
          "musique": "LOWMEM__PHG_C1_con"}
CELLS = {"metaqa": ["metaqa", "metaqa_phg"], "squad": ["squad", "squad_phg"], "musique": ["musique"]}
K0 = TA.K0                                  # 60, the frozen RRF constant
C_MAIN = TA.K_LOCK                          # 100 structural slots (= one block-equivalent)
C_PATCH = C_MAIN                            # the global of the pinned arm_row / evaluate: the H2 reference at C = 100 (section 30)
M_MAIN = 5000                               # the total node budget (ruling 1)
C_CURVE = (25, 50, 100, 200, 500)           # descriptive only; C_MAIN is the decisive one
ACT = TA.TOP200                             # 200: the activation depth = the frozen served-list depth; pinned, never swept
BEAM, DEPTH = TA.K_LOCK, 2                  # the frozen section-21 balanced beam (the H2 reference)
CQ, BLOCK = TB.CQ, TB.BLOCK                 # 200 query rows per product batch, 40000 nodes per chunk (= the SPLADE shard size)
ALPHA = 0.01
N_POP = AD.CONTRACT["EVAL_CAP"]             # 2000
PER_HOP = N_POP // 3                        # 666 (metaqa, hop-balanced)
SEED = 20260926                             # the one new fixed sampling seed (ruling 2 step 3)
N_AGREE = 100
AGREE_MIN = 0.90                            # served-list sanity (wrong vectors / row mapping), not a float-noise test
DRY_ROWS = 200
NG_BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5-10", 5, 10), ("11+", 11, 1 << 40))
FLATH2_FNS = ("sha_file", "sha_text", "label", "_rss_mb", "host_state", "stats", "q4", "rel", "rrf_full", "chunk_iter",
              "dense_products", "splade_products", "struct_arrays", "undirected_degree", "unit_queries", "served_seeds", "_NS",
              "beam_namespace", "run_beams", "arm_row", "evaluate", "paired", "strata_masks", "compare")
MICRO_FNS = ("admissible", "transition_scores", "rr_rank", "fused_order", "rows", "expand_real")
COMP_FNS = ("rr_compress", "beam_balanced")
WORDS_LOC = {"GAIN": "LOC_HELPS", "LOSS": "LOC_HURTS", "NEUTRAL": "NO_SIGNIFICANT_DIFFERENCE"}
WORDS_H2 = {"GAIN": "H2_REFERENCE_ADDS_OVER_FLAT", "LOSS": "H2_REFERENCE_LOSES_TO_FLAT", "NEUTRAL": "NO_SIGNIFICANT_DIFFERENCE"}
CATS = ("RESCUED", "TYPE_A", "TYPE_B", "TYPE_C")
CONSTANTS = {"K0": K0, "C_MAIN": C_MAIN, "M_MAIN": M_MAIN, "C_CURVE": list(C_CURVE), "ACT": ACT, "BEAM": BEAM, "DEPTH": DEPTH,
             "CQ": CQ, "BLOCK": BLOCK, "ALPHA": ALPHA, "N_POP": N_POP, "PER_HOP": PER_HOP, "SEED": SEED, "N_AGREE": N_AGREE,
             "AGREE_MIN": AGREE_MIN, "SEED_K": TA.SEED_K, "NG_BUCKETS": [list(b) for b in NG_BUCKETS]}
assert C_MAIN == 100 and ACT == 200 and K0 == 60 and C_MAIN in C_CURVE and max(C_CURVE) < M_MAIN - ACT and N_POP == 2000


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b_ in iter(lambda: f.read(1 << 20), b""):
            h.update(b_)
    return h.hexdigest()


def pinned_sources(fn, names):
    """verbatim source text of the named top-level functions / classes of a pinned module (each defined exactly once)."""
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
for nm_ in FLATH2_FNS:                      # the section-30 harness functions, verbatim, in this module's namespace
    exec(FLATH2_SRC[nm_], globals())
FSRC = dict(pinned_sources("_l1c_microl3.py", MICRO_FNS), **pinned_sources("_l1c_transfer_composite.py", COMP_FNS))
SRC_SHA = {k: sha_text(v) for k, v in sorted(dict(FLATH2_SRC, **FSRC).items())}


def make_arm(C):
    """the pinned arm_row with C structural slots (its global C_PATCH), in its own namespace."""
    ns = {"np": np, "C_PATCH": int(C), "__builtins__": __builtins__}
    exec(FLATH2_SRC["arm_row"], ns)
    return ns["arm_row"]


ARM = {C: make_arm(C) for C in C_CURVE}


# ---------------------------------------------------------------- the partition and the LOC order
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
    """an independent pure-Python implementation of the same definition (the per-row cross-check of stage G)."""
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


def picks(L, frank, C):
    """the first C LOC nodes outside FLAT_RRF[:M - C] (the novel slots) and the LOC index of the C-th (-1: fewer exist)."""
    idx = np.flatnonzero(frank[L] >= M_MAIN - C)
    return L[idx[:C]], (int(idx[C - 1]) if len(idx) >= C else -1)


def dig(*arrs):
    h = hashlib.sha256()
    for a_ in arrs:
        h.update(np.ascontiguousarray(a_).tobytes())
    return np.frombuffer(h.digest(), np.uint8)


# ---------------------------------------------------------------- arguments, pins, pre-registration
CODE_SHA = sha_file(os.path.abspath(__file__))
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
DRY = "--dry" in sys.argv
CHECK = "--check-prereg" in sys.argv
assert not (CHECK and DRY)
DRY_OUT = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--dry-out=")), None)
DRY_IN = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--dry-in=")), None)
for p_ in (DRY_OUT, DRY_IN):
    assert p_ is None or (DRY and not os.path.abspath(p_).lower().startswith(os.path.abspath(REPO).lower())), "--dry-out / --dry-in: dry runs only, outside the repository"
assert MODE in ("POP", "IDENT", "G", "E", "SUMMARY"), "usage: POP | IDENT <DEV_A cache> | G <dataset> | E <dataset> | SUMMARY  [--dry | --check-prereg]"
ARG = sys.argv[2] if MODE in ("IDENT", "G", "E") else None
assert MODE != "IDENT" or ARG in DEV_A_CACHES
assert MODE not in ("G", "E") or ARG in CELLS
pre, pre_sha = None, None
if not DRY:
    pre = json.load(open(PRE, encoding="utf-8"))
    pre_sha = sha_file(PRE)
    assert pre["code"]["new_modules"]["_l1c_hardloc.py"]["sha256"] == CODE_SHA, "this module changed since the pre-registration"
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    assert pre["code"]["function_sources_sha256"] == SRC_SHA, "a pinned function changed"
    assert pre["constants"] == CONSTANTS, "constants differ from the pre-registration"
    for nm, pn in pre["records_read_only"].items():
        assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "record %s changed" % nm
    for cell, pn in pre["partitions_read_only"].items():
        assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "partition of %s changed" % cell
for c_, s_ in FLAT_RECORD.items():
    assert sha_file(os.path.join(OUT, "flat_A_%s.json" % c_)) == s_, "the section-29 record %s changed" % c_
for c_, s_ in FLATH2_RECORD.items():
    assert sha_file(os.path.join(OUT, "flath2_A_%s.json" % c_)) == s_, "the section-30 record %s changed" % c_
HOST0 = host_state()
log("%s %s%s: host at start %s" % (MODE, ARG or "", " (DRY)" if DRY else "", json.dumps(HOST0)))
T_ALL = time.time()
TIMES = {}


def dataset_pins(cd):
    return {"DATASET_json_RECORD_SHA256": cd.record_sha, "query_index.npz": sha_file(cd._query_index_path()), "keys.npz": sha_file(cd._keys_path())}


def check_dataset(cd):
    if not DRY:
        want = pre["datasets"][cd.name]
        got = dataset_pins(cd)
        assert got == want, "dataset pins of %s changed: %s" % (cd.name, got)


def finish(res, fp_out, arrays=None):
    res["seconds_by_stage"] = TIMES
    res["seconds"] = round(time.time() - T_ALL, 1)
    res["process_rss_mb_at_end"] = round(_rss_mb(), 1)
    assert sha_file(os.path.abspath(__file__)) == CODE_SHA, "the module file changed during the run"
    if DRY:
        blob = json.dumps(res)
        if DRY_OUT:
            if arrays is not None:
                np.savez(os.path.join(DRY_OUT, os.path.basename(fp_out).replace(".json", ".npz")), **arrays)
            with open(os.path.join(DRY_OUT, os.path.basename(fp_out)), "w", encoding="utf-8") as f_:
                f_.write(blob)
        log("DRY RUN (%.0fs, RSS %.0f MB): record assembled (%d bytes) and NOT written to the repository.  times %s" % (
            time.time() - T_ALL, _rss_mb(), len(blob), json.dumps(TIMES)))
        sys.exit(0)
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    if arrays is not None:
        fz = fp_out.replace(".json", ".npz")
        assert not os.path.exists(fz), "write-once: %s exists" % fz
        np.savez(fz, **arrays)
        res["npz"] = {"path": rel(fz), "sha256": sha_file(fz), "arrays": sorted(arrays)}
    G.S.wj(fp_out, res)
    log("done (%.0fs) -> %s sha256 %s" % (res["seconds"], rel(fp_out), sha_file(fp_out)[:12]))
    sys.exit(0)


def common_record():
    return {"preregistration": {"path": rel(PRE), "sha256": pre_sha}, "code": {"path": rel(os.path.abspath(__file__)), "sha256": CODE_SHA},
            "pinned": PINNED, "pinned_repo": PINNED_REPO, "function_sources_sha256": SRC_SHA, "constants": CONSTANTS,
            "platform": {"python": platform.python_version(), "numpy": np.__version__}, "host_at_start": HOST0, "dry": DRY}


# ======================================================================== POP
if MODE == "POP":
    import _l1c_hardloc_pop as HP                            # the population rule's own module (pinned by the pre-registration)
    fp_out = POP_FILE
    if not DRY:
        assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
        assert sha_file(os.path.join(HERE, "_l1c_hardloc_pop.py")) == pre["code"]["new_modules"]["_l1c_hardloc_pop.py"]["sha256"]
    if CHECK:
        log("pre-registration check passed (POP); no data read, nothing written")
        sys.exit(0)
    res = HP.build(dry=DRY, seed=SEED, n_pop=N_POP, per_hop=PER_HOP, log=log)
    res.update(common_record())
    finish(res, fp_out)


# ======================================================================== IDENT (DEV_A split A; real gold, identities only)
def split_A_mask(qids):
    return np.array([int(hashlib.sha1(q.encode("utf-8")).hexdigest()[:8], 16) & 1 for q in qids], np.int8) == 0


def beams_for(cd, rows_g, N, Qu, tag):
    seeds_, d200, s200 = served_seeds(cd, rows_g)
    xo, ao, xi, ai = struct_arrays(cd, N)
    cd._csr.clear()
    deg_u = undirected_degree(cd, N)
    hub = (deg_u + 1) > C_MAIN
    NS = beam_namespace(FSRC, Qu, cd.node_embeddings, seeds_, N, xo, ao, xi, ai, hub)
    bal1, bal2, BS = run_beams(NS, len(rows_g), tag)
    NS.clear()                                               # the exec'd functions hold NS as their globals (a cycle): free STRUCT now
    del NS, xo, ao, xi, ai, deg_u
    return bal1, bal2, BS, seeds_, d200, s200, hub


if MODE == "IDENT":
    cache = ARG
    ds, tag = DS_OF[cache], TAG_OF[cache]
    fp_out = os.path.join(OUT, "hardloc_ident_%s.json" % cache)
    if not DRY:
        assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
        assert sha_file(G.X.CACHES[cache]) == pre["caches_read_only"][cache]["sha256"], "cache %s changed" % cache
    if CHECK:
        log("pre-registration check passed (IDENT %s); no data read, nothing written" % cache)
        sys.exit(0)
    frec = json.load(open(os.path.join(OUT, "flat_A_%s.json" % cache), encoding="utf-8"))
    hrec = json.load(open(os.path.join(OUT, "flath2_A_%s.json" % cache), encoding="utf-8"))
    cd = AD.CanonicalDataset(ds)
    check_dataset(cd)
    N = int(cd.n_nodes)
    z = np.load(G.X.CACHES[cache], allow_pickle=True)
    zmeta = json.loads(str(z["meta_json"]))
    rows_all = z["rows"].astype(np.int64)
    nq = len(rows_all)
    qids = [cd.query_ids[int(r)] for r in rows_all]
    assert sha_text(",".join(qids)) == zmeta["row_query_ids_sha256"], "cache rows != its recorded query ids"
    isA = split_A_mask(qids)
    rowsA = np.flatnonzero(isA)
    nA = len(rowsA)
    assert nA == int(frec["n_split_A"]) == int(hrec["n_split_A"]) and N == int(frec["N"]) == int(hrec["N"])
    rows_g = rows_all[rowsA]
    P = Part(cd, tag)
    assert (P.hard == z["hard"].astype(np.int64)).all() and (P.sizes == z["part_sizes"]).all(), "partition file != the cache's hard array"
    gold = [np.asarray(g, np.int64) for g in cd.gold(rows_g)]
    for j, qi in enumerate(rowsA):                           # the cache's gold blocks == the blocks of the query-index gold nodes
        gb = z["gold_part"][z["gold_ptr"][qi]:z["gold_ptr"][qi + 1]]
        assert sorted(set(P.hard[gold[j]].tolist())) == sorted(int(x) for x in gb), "gold blocks differ (row %d)" % qi
    t_ = time.time()
    Qu = unit_queries(cd, rows_g)
    bal1, bal2, BS, seedsA, _d200, _s200, _hub = beams_for(cd, rows_g, N, Qu, cache)
    assert (seedsA == z["seeds"][rowsA].astype(np.int64)).all(), "served seeds != the replay cache seeds"
    TIMES["beams"] = round(time.time() - t_, 1)
    # the section-29 F1 pass: batches of CQ CACHE rows (split-B rows only as dense product companions), per-chunk products
    t_ = time.time()
    Qall = unit_queries(cd, rows_all)
    assert Qall[rowsA].tobytes() == Qu.tobytes()
    QsAll = cd.embeddings("splade", "queries").read(rows_all).tocsr()
    Es = cd.embeddings("splade", "docs")
    assert int(Es.shard_size) == BLOCK
    E16 = cd.node_embeddings
    jA = {int(q): j for j, q in enumerate(rowsA)}
    tops = np.empty((nA, M_MAIN), np.int32)
    gmax = np.full(nA, -1, np.int64)
    for qa in range(0, nq, CQ):
        qz = min(nq, qa + CQ)
        here = [qi for qi in range(qa, qz) if isA[qi]]
        if not here:
            continue
        SD = np.empty((qz - qa, N), np.float32)
        SS = np.empty((len(here), N), np.float32)
        Qd = np.asarray(QsAll[here].todense(), np.float32)
        for a, b_, blk in E16.blocks(BLOCK):
            Ec = np.ascontiguousarray(np.asarray(blk, np.float32))
            SD[:, a:b_] = Qall[qa:qz] @ Ec.T
            Msp = Es.shard(a // BLOCK)
            assert Msp.shape[0] == b_ - a
            SS[:, a:b_] = np.asarray(Msp @ Qd.T, np.float32).T
            del Ec, Msp
        for i, qi in enumerate(here):
            j = jA[qi]
            of, od, os_, npos, fv, frank = flat_row(SD[qi - qa], SS[i])
            tops[j] = of[:M_MAIN]
            gmax[j] = int(frank[gold[j]].max())
        del SD, SS, Qd
        log("  %s F1 batch %d / %d (RSS %.0f MB)" % (cache, qa // CQ + 1, (nq + CQ - 1) // CQ, _rss_mb()))
    TIMES["flat_F1"] = round(time.time() - t_, 1)
    assert gmax.tolist() == frec["_gmax_FLAT_RRF (0-based deepest gold position per split-A row)"], "FLAT_RRF != the section-29 record (deepest gold positions)"
    bals = [bal1[j] + bal2[j] for j in range(nA)]
    r_ = evaluate(tops, bals, [set(b) for b in bal1], np.full(nA, M_MAIN), gold)
    assert r_["flat_all"].astype(int).tolist() == hrec["_ind_FLAT_%d" % M_MAIN], "FLAT@5000 != the section-30 record"
    assert r_["h2_all"].astype(int).tolist() == hrec["_ind_FLAT_H2_%d" % M_MAIN], "FLAT+H2@5000 != the section-30 record"
    for C in C_CURVE:                                        # the per-C arm function == the pinned global arm_row at C = 100 (and runs at every C)
        for j in range(min(nA, 50)):
            out_c = ARM[C](tops[j], bals[j], M_MAIN)
            if C == C_MAIN:
                assert out_c == arm_row(tops[j], bals[j], M_MAIN)
    IDENT = ["cache rows == recorded query ids; split A by sha1 parity (%d rows)" % nA,
             "partition file == the cache hard array; query-index gold blocks == the cache gold blocks",
             "served seeds == the cache seeds; the pinned beam functions over the partition-free STRUCT arrays (fresh path)",
             "section-29 FLAT_RRF (F1 batching): the deepest gold position of every split-A row == flat_A_%s.json" % cache,
             "FLAT@%d and FLAT+H2@%d indicator vectors == flath2_A_%s.json (every split-A row)" % (M_MAIN, M_MAIN, cache)]
    log("IDENTITY GATE PASSED (%s): %s" % (cache, "; ".join(IDENT)))
    res = {"cache": cache, "dataset": ds, "partition_tag": tag, "mode": "HARD_LOC_A_IDENTITY_GATE_DEV_A_SPLIT_A", "n_split_A": nA, "N": N,
           "status": "PASS (identities only; no coverage number computed beyond the asserted equalities; nothing selected)",
           "identities": IDENT, "cache_file": {"path": rel(G.X.CACHES[cache]), "sha256": sha_file(G.X.CACHES[cache])},
           "records_reproduced": {"flat_A": {"path": "results/L1_COVPART/flat_A_%s.json" % cache, "sha256": FLAT_RECORD[cache]},
                                  "flath2_A": {"path": "results/L1_COVPART/flath2_A_%s.json" % cache, "sha256": FLATH2_RECORD[cache]}},
           "balanced_beam (split A)": BS}
    res.update(common_record())
    finish(res, fp_out)


# ======================================================================== the fresh populations: G (gold-free) and E (gold)
ds = ARG if MODE in ("G", "E") else None
if MODE in ("G", "E"):
    cd = AD.CanonicalDataset(ds)
    check_dataset(cd)
    N = int(cd.n_nodes)
    cells = CELLS[ds]
    if DRY:                                                  # smoke population: the first DRY_ROWS rows of the dataset's DEV_A cache (sorted)
        rows_g = np.sort(np.load(G.X.CACHES[cells[0]], allow_pickle=True)["rows"].astype(np.int64))[:DRY_ROWS]
        pop_sha = None
    else:
        pop_sha = sha_file(POP_FILE)
        prec = json.load(open(POP_FILE, encoding="utf-8"))
        assert prec["preregistration"]["sha256"] == pre_sha, "the population was not produced under this pre-registration"
        pp = prec["populations"][ds]
        rows_g = np.asarray(pp["rows"], np.int64)
        assert [cd.query_ids[int(r)] for r in rows_g] == pp["query_ids"] and sha_text(",".join(pp["query_ids"])) == pp["query_ids_sha256"]
        assert (np.diff(rows_g) > 0).all() and len(rows_g) == (3 * PER_HOP if ds == "metaqa" else N_POP)
    nA = len(rows_g)
    qids = [cd.query_ids[int(r)] for r in rows_g]
    POPV = {"n": nA, "rows_sha256": sha_text(",".join(str(int(r)) for r in rows_g)), "query_ids_sha256": sha_text(",".join(qids)),
            "population_file": None if DRY else {"path": rel(POP_FILE), "sha256": pop_sha}}
    parts = {c: Part(cd, TAG_OF[c]) for c in cells}
    if not DRY:
        for c in cells:
            assert sha_file(parts[c].path) == pre["partitions_read_only"][c]["sha256"]
    fpG = os.path.join(OUT, "hardloc_G_%s.json" % ds)
    fpE = os.path.join(OUT, "hardloc_E_%s.json" % ds)
    if CHECK:
        if MODE == "G":
            assert not os.path.exists(fpG)
        else:
            assert os.path.exists(fpG) and not os.path.exists(fpE)
        log("pre-registration check passed (%s %s): code, sources, constants, records, partitions, dataset pins, population; nothing written" % (MODE, ds))
        sys.exit(0)
    log("%s %s: N %d, %d population rows, cells %s (%s)" % (MODE, ds, N, nA, cells, json.dumps(POPV)))


def batches(cd, rows_g, Qu):
    """(j0, j1, SD, SS): the full dense / SPLADE products of CQ population rows over all N nodes (the section-30 fresh-path
    product shapes: pinned dense_products / splade_products per TB.BLOCK chunk == SPLADE shard)."""
    QsA = cd.embeddings("splade", "queries").read(rows_g).tocsr()
    Es = cd.embeddings("splade", "docs")
    assert int(Es.shard_size) == BLOCK and int(Es.n_rows) == N
    n = len(rows_g)
    for j0 in range(0, n, CQ):
        j1 = min(n, j0 + CQ)
        SD = np.empty((j1 - j0, N), np.float32)
        SS = np.empty((j1 - j0, N), np.float32)
        for a, b_, blk in chunk_iter(cd.node_embeddings):
            Ec = np.ascontiguousarray(np.asarray(blk, np.float32))
            Msp = Es.shard(a // BLOCK)
            assert Msp.shape[0] == b_ - a and int(Msp.shape[1]) == int(QsA.shape[1])
            SD[:, a:b_] = dense_products(Qu, Ec, j0, j1)
            SS[:, a:b_] = splade_products(Msp, QsA, j0, j1)
            del Ec, Msp
        assert np.isfinite(SD).all() and np.isfinite(SS).all() and (SS >= 0).all()
        yield j0, j1, SD, SS
        del SD, SS


def hops_of(cd, rows_g, qids):
    if cd.name == "musique":                                 # the canonical id prefix (2hop__, 3hop1__, 4hop2__, ...)
        return np.array([int(re.match(r"^(\d)hop", q).group(1)) for q in qids], np.int64)
    zq = np.load(cd._query_index_path(), allow_pickle=False)
    return zq["hop"][np.asarray(rows_g, np.int64)].astype(np.int64)


def gold_counts(cd, rows_g):
    zq = np.load(cd._query_index_path(), allow_pickle=False)
    gp = zq["gold_ptr"]
    r = np.asarray(rows_g, np.int64)
    return (gp[r + 1] - gp[r]).astype(np.int64)


# ---------------------------------------------------------------- stage G (gold-free)
if MODE == "G":
    if not DRY:
        assert not os.path.exists(fpG) and not os.path.exists(fpG.replace(".json", ".npz")), "write-once: stage G of %s exists" % ds
    t_ = time.time()
    Qu = unit_queries(cd, rows_g)
    bal1, bal2, BEAM_STATS, seeds_, d200, s200, hub = beams_for(cd, rows_g, N, Qu, ds)
    TIMES["h2_beam"] = round(time.time() - t_, 1)
    t_ = time.time()
    dgF = np.zeros((nA, 32), np.uint8)
    dgL = {c: np.zeros((nA, 32), np.uint8) for c in cells}
    npos = np.zeros(nA, np.int64)
    agd, ags, ex_both = np.zeros(nA), np.zeros(nA), np.zeros(nA, bool)
    GF = {c: {k: np.zeros(nA, np.int64) for k in ["n_blocks_activated", "loc_len", "top_block_nodes_in_flat_top200"]} for c in cells}
    GFC = {c: {C: {k: np.full(nA, -1, np.int64) for k in ("n_novel", "cut_index", "picks_in_flat_tail", "picks_beyond_M", "blocks_with_picks",
                                                              "picks_flat_rank_median")} for C in C_CURVE} for c in cells}
    top_share = {c: np.zeros(nA) for c in cells}
    for j0, j1, SD, SS in batches(cd, rows_g, Qu):
        for i in range(j1 - j0):
            j = j0 + i
            of, od, os_, npos_j, fv, frank = flat_row(SD[i], SS[i])
            npos[j] = npos_j
            dgF[j] = dig(of[:M_MAIN].astype(np.int32))
            k_ = min(N_AGREE, npos_j)
            agd[j] = len(set(od[:N_AGREE].tolist()) & set(d200[j, :N_AGREE].tolist())) / float(N_AGREE)
            ags[j] = (len(set(os_[:k_].tolist()) & set(s200[j, :k_].tolist())) / float(k_)) if k_ else 1.0
            if (od[:ACT] == d200[j]).all() and npos_j >= ACT and (os_[:ACT] == s200[j]).all():
                ex_both[j] = True
                assert TA.node_rrf(od[:ACT], os_[:ACT])[:TA.SEED_K] == [int(x) for x in seeds_[j] if x >= 0], "seeds from the exhaustive lists != served seeds"
            for c in cells:
                P = parts[c]
                ub, A = activation(of[:ACT], fv, P.hard)
                L = loc_order(ub, A, frank, P)
                bf, Ab = loc_bruteforce(of, fv, frank, P)       # the independent implementation, every row
                assert bf == L.tolist() and sorted(Ab) == ub.tolist() and [Ab[int(b)] for b in ub] == A.tolist(), "LOC order != brute force (%s row %d)" % (c, j)
                assert len(L) == int(P.sizes[ub].sum()) and len(set(L.tolist())) == len(L)
                dgL[c][j] = dig(ub.astype(np.int32), A, L.astype(np.int32))
                GF[c]["n_blocks_activated"][j], GF[c]["loc_len"][j] = len(ub), len(L)
                GF[c]["top_block_nodes_in_flat_top200"][j] = int((P.hard[of[:ACT]] == ub[int(np.argmax(A))]).sum())
                top_share[c][j] = float(A.max() / A.sum())
                ol = of[:M_MAIN]
                for C in C_CURVE:
                    nov, cut = picks(L, frank, C)
                    bset, tail, novel, fill = ARM[C](ol, L[:cut + 1].tolist() if cut >= 0 else L.tolist(), M_MAIN)
                    assert [int(x) for x in novel] == nov.tolist(), "arm_row novel != picks (%s row %d C %d)" % (c, j, C)
                    g_ = GFC[c][C]
                    g_["n_novel"][j], g_["cut_index"][j] = len(nov), cut
                    fr = frank[nov]
                    g_["picks_in_flat_tail"][j] = int((fr < M_MAIN).sum())
                    g_["picks_beyond_M"][j] = int((fr >= M_MAIN).sum())
                    g_["blocks_with_picks"][j] = len(set(P.hard[nov].tolist()))
                    g_["picks_flat_rank_median"][j] = int(np.median(fr)) + 1 if len(fr) else -1
        log("  %s G rows %d / %d (%.0fs, RSS %.0f MB)" % (ds, j1, nA, time.time() - t_, _rss_mb()))
    TIMES["flat_and_loc"] = round(time.time() - t_, 1)
    AGREE = {"dense_top100_set_overlap_with_served_mean": round(float(agd.mean()), 5), "dense_top100_set_overlap_min": round(float(agd.min()), 3),
             "splade_top100_set_overlap_with_served_mean": round(float(ags.mean()), 5), "splade_top100_set_overlap_min": round(float(ags.min()), 3),
             "rows_where_both_exhaustive_top200_lists_equal_the_served (seeds re-derived and asserted equal)": int(ex_both.sum()),
             "served_lists": "the frozen retrieval caches (TF32 products); the exhaustive lists here are fp32",
             "splade_positive_nodes_per_query": stats(npos)}
    log("%s served-list agreement: %s" % (ds, json.dumps(AGREE)))
    if not DRY:
        assert AGREE["dense_top100_set_overlap_with_served_mean"] >= AGREE_MIN and AGREE["splade_top100_set_overlap_with_served_mean"] >= AGREE_MIN, \
            "the exhaustive rankings disagree with the served retrieval caches (wrong vectors or row mapping)"
    LOCG = {}
    for c in cells:
        e = {"partition": {"tag": TAG_OF[c], "path": rel(parts[c].path), "sha256": sha_file(parts[c].path), "npart": parts[c].npart,
                           "block_size": stats(parts[c].sizes)},
             "blocks_activated_per_query": stats(GF[c]["n_blocks_activated"]), "loc_order_length": stats(GF[c]["loc_len"]),
             "strongest_block_share_of_total_A": stats(top_share[c]),
             "flat_top200_nodes_in_the_strongest_block": stats(GF[c]["top_block_nodes_in_flat_top200"]), "per_C": {}}
        for C in C_CURVE:
            g_ = GFC[c][C]
            full = g_["n_novel"] == C
            e["per_C"][str(C)] = {"novel_slots_filled": stats(g_["n_novel"]), "rows_with_all_C_slots_filled": int(full.sum()),
                                  "loc_index_of_the_C_th_pick (rows with all slots)": stats(g_["cut_index"][full]),
                                  "picks_already_in_FLAT@M_tail (served anyway)": stats(g_["picks_in_flat_tail"]),
                                  "picks_beyond_FLAT@M (new exposure)": stats(g_["picks_beyond_M"]),
                                  "blocks_contributing_picks": stats(g_["blocks_with_picks"]),
                                  "median_FLAT_rank_of_picks (1-based)": stats(g_["picks_flat_rank_median"][g_["picks_flat_rank_median"] >= 0])}
        LOCG[c] = e
        log("%s gold-free LOC (%s): blocks activated %s, LOC length %s, C=%d picks beyond FLAT@M %s" % (
            ds, c, e["blocks_activated_per_query"], e["loc_order_length"], C_MAIN, e["per_C"][str(C_MAIN)]["picks_beyond_FLAT@M (new exposure)"]))
    bal_arr = np.full((nA, 2 * BEAM), -1, np.int32)
    bal_len = np.zeros((nA, 2), np.int32)
    for j in range(nA):
        b_ = bal1[j] + bal2[j]
        bal_arr[j, :len(b_)] = b_
        bal_len[j] = [len(bal1[j]), len(bal2[j])]
    arrays = {"rows": np.asarray(rows_g, np.int64), "bal": bal_arr, "bal_len": bal_len, "seeds": seeds_.astype(np.int32), "npos": npos,
              "digest_flat_top_M": dgF, **{"digest_loc_%s" % c: dgL[c] for c in cells}}
    res = {"dataset": ds, "stage": "G (gold-free)", "mode": "PREREGISTERED_HARD_LOC_A", "eval_split": cd.eval_split, "N": N, "cells": cells,
           "population": POPV, "n_rows": nA, "hubs (deg_u + 1 > 100)": int(hub.sum()),
           "h2_reference_beam": BEAM_STATS, "served_list_agreement": AGREE, "loc_gold_free": LOCG,
           "loc_cross_check": "every row, every cell: vectorised LOC order == an independent pure-Python implementation (A bitwise, order exact); arm_row novel slots == picks at every C",
           "digests": {"flat_top_M": sha_text(dgF.tobytes().hex()), **{"loc_%s" % c: sha_text(dgL[c].tobytes().hex()) for c in cells}}}
    res.update(common_record())
    finish(res, fpG, arrays)


# ---------------------------------------------------------------- stage E (gold, once)
if MODE == "E":
    gdir = DRY_IN if DRY else OUT
    gj, gz = os.path.join(gdir, os.path.basename(fpG)), os.path.join(gdir, os.path.basename(fpG).replace(".json", ".npz"))
    if not DRY:
        assert not os.path.exists(fpE), "write-once: %s exists" % fpE
    grec = json.load(open(gj, encoding="utf-8"))
    assert grec["dry"] == DRY and grec["code"]["sha256"] == CODE_SHA and grec["population"] == POPV and grec["constants"] == CONSTANTS
    Z = np.load(gz, allow_pickle=False)
    if not DRY:
        assert grec["npz"]["sha256"] == sha_file(gz), "the stage-G arrays changed"
    assert (Z["rows"] == rows_g).all()
    bal_arr, bal_len = Z["bal"], Z["bal_len"]
    bal1 = [[int(x) for x in bal_arr[j, :bal_len[j, 0]]] for j in range(nA)]
    bals = [[int(x) for x in bal_arr[j, :bal_len[j, 0] + bal_len[j, 1]]] for j in range(nA)]
    # ---- gold: real (once), or STAND-IN of the true per-row count in a dry run (no gold identity read)
    if DRY:
        rng_dry = np.random.default_rng(12345)
        golds = [np.sort(rng_dry.choice(N, size=int(k), replace=False)).astype(np.int64) for k in gold_counts(cd, rows_g)]
    else:
        golds = [np.asarray(g, np.int64) for g in cd.gold(rows_g)]
    assert all(len(g) for g in golds)
    ngold = np.array([len(g) for g in golds], np.int64)
    hopsA = hops_of(cd, rows_g, qids)
    t_ = time.time()
    Qu = unit_queries(cd, rows_g)
    tops = np.empty((nA, M_MAIN), np.int32)
    flat_all, flat_any = np.zeros(nA, bool), np.zeros(nA, bool)
    IND = {c: {C: np.zeros(nA, bool) for C in C_CURVE} for c in cells}
    INDANY = {c: {C: np.zeros(nA, bool) for C in C_CURVE} for c in cells}
    RESC = {c: {C: np.zeros(nA, np.int64) for C in C_CURVE} for c in cells}
    DISP = {c: {C: np.zeros(nA, np.int64) for C in C_CURVE} for c in cells}
    DISP_RANKS = {c: [] for c in cells}                      # 1-based FLAT ranks of displaced gold nodes at C_MAIN
    RESC_RANKS = {c: [] for c in cells}                      # 1-based FLAT ranks of rescued gold nodes at C_MAIN
    AUD = {c: [] for c in cells}                             # one dict per FLAT@M-missed gold node (C_MAIN)
    QTYPES = {c: [None] * nA for c in cells}
    h2_gold_served = [None] * nA
    blocks_spanned = {c: np.zeros(nA, np.int64) for c in cells}
    blocks_spanned_missed = {c: np.zeros(nA, np.int64) for c in cells}
    for j0, j1, SD, SS in batches(cd, rows_g, Qu):
        for i in range(j1 - j0):
            j = j0 + i
            of, od, os_, npos_j, fv, frank = flat_row(SD[i], SS[i])
            ol = of[:M_MAIN]
            assert (dig(ol.astype(np.int32)) == Z["digest_flat_top_M"][j]).all(), "FLAT_RRF != stage G (row %d)" % j
            tops[j] = ol
            g = golds[j]
            fpos = frank[g]
            fin = fpos < M_MAIN
            flat_all[j], flat_any[j] = bool(fin.all()), bool(fin.any())
            bset_h, tail_h, novel_h, fill_h = arm_row(ol, bals[j], M_MAIN)
            served_h = bset_h | set(novel_h) | set(fill_h)
            h2_gold_served[j] = [int(x) in served_h for x in g]
            for c in cells:
                P = parts[c]
                ub, A = activation(of[:ACT], fv, P.hard)
                L = loc_order(ub, A, frank, P)
                assert (dig(ub.astype(np.int32), A, L.astype(np.int32)) == Z["digest_loc_%s" % c][j]).all(), "LOC != stage G (%s row %d)" % (c, j)
                blocks_spanned[c][j] = len(set(P.hard[g].tolist()))
                blocks_spanned_missed[c][j] = len(set(P.hard[g[~fin]].tolist()))
                served_main = novel_main = None
                for C in C_CURVE:
                    nov, cut = picks(L, frank, C)
                    bset, tail, novel, fill = ARM[C](ol, L[:cut + 1].tolist() if cut >= 0 else L.tolist(), M_MAIN)
                    assert [int(x) for x in novel] == nov.tolist()
                    served = bset | set(int(x) for x in novel) | set(fill)
                    sv = np.array([int(x) in served for x in g])
                    IND[c][C][j], INDANY[c][C][j] = bool(sv.all()), bool(sv.any())
                    nset = set(nov.tolist())
                    resc = [int(x) for x, fp in zip(g, fpos) if fp >= M_MAIN and int(x) in nset]
                    disp = [int(x) for x, fp, s_ in zip(g, fpos, sv) if fp < M_MAIN and not s_]
                    for x in disp:                           # the interface can only displace the FLAT tail
                        assert M_MAIN - C <= int(frank[x]) <= M_MAIN - 1, "a displaced gold outside the FLAT tail"
                    assert all(bool(s_) for x, s_ in zip(g, sv) if int(frank[x]) < M_MAIN - C), "a base gold not served"
                    RESC[c][C][j], DISP[c][C][j] = len(resc), len(disp)
                    if C == C_MAIN:
                        served_main, novel_main = served, nset
                        DISP_RANKS[c] += [int(frank[x]) + 1 for x in disp]
                        RESC_RANKS[c] += [int(frank[x]) + 1 for x in resc]
                # ---- the A/B/C audit of every FLAT@M-missed gold node (C_MAIN)
                A_of = dict(zip(ub.tolist(), A.tolist()))
                nov_main = [v for v in novel_main]
                pick_blocks = collections.Counter(int(P.hard[v]) for v in nov_main)
                nov_idx, cut_main = picks(L, frank, C_MAIN)
                qt = []
                for x, fp, hs in zip(g.tolist(), fpos.tolist(), h2_gold_served[j]):
                    if fp < M_MAIN:
                        continue
                    b = int(P.hard[x])
                    a_ = A_of.get(b, 0.0)
                    if x in novel_main:
                        cat = "RESCUED"
                    elif a_ == 0.0:
                        cat = "TYPE_A"
                    elif pick_blocks.get(b, 0) == 0:
                        cat = "TYPE_B"
                    else:
                        cat = "TYPE_C"
                    d = {"row": j, "cat": cat, "flat_rank": fp + 1, "h2_serves": bool(hs), "block_size": int(P.sizes[b])}
                    if a_ > 0:
                        bn = P.nodes_of(b)
                        fb = frank[bn]
                        d["A"] = a_
                        d["block_rank_by_A"] = 1 + int(sum(1 for v in A_of.values() if v > a_))
                        d["n_blocks_activated"] = len(A_of)
                        d["loc_index"] = int(np.flatnonzero(L == x)[0])
                        d["cut_index"] = cut_main
                        d["block_picks"] = int(pick_blocks.get(b, 0))
                        d["block_nonbase_nodes"] = int((fb >= M_MAIN - C_MAIN).sum())
                        d["nonbase_block_nodes_ahead_of_gold"] = int(((fb >= M_MAIN - C_MAIN) & (fb < fp)).sum())
                        d["weakest_picked_block_rank_by_A"] = (1 + max(int(sum(1 for v in A_of.values() if v > A_of[bb])) for bb in pick_blocks)) if pick_blocks else None
                    AUD[c].append(d)
                    qt.append(cat)
                QTYPES[c][j] = qt
        log("  %s E rows %d / %d (%.0fs, RSS %.0f MB)" % (ds, j1, nA, time.time() - t_, _rss_mb()))
    TIMES["flat_loc_gold"] = round(time.time() - t_, 1)
    # ---- the H2 reference through the pinned evaluate (exactly the section-30 path)
    rH = evaluate(tops, bals, [set(b) for b in bal1], np.full(nA, M_MAIN), golds)
    assert (rH["flat_all"] == flat_all).all() and (rH["flat_any"] == flat_any).all()
    assert (rH["h2_all"] == np.array([all(f) for f in h2_gold_served])).all()
    ST_ = strata_masks(hopsA, ngold)
    if ds == "musique":
        dv0, dv1 = cd.split_ranges["dev"]
        ST_["per_source"] = {"dev": (rows_g >= dv0) & (rows_g < dv1), "train": (rows_g < dv0) | (rows_g >= dv1)}
    H2REF = compare("FLAT@%d" % M_MAIN, flat_all, "FLAT+H2@%d (L3 reference)" % M_MAIN, rH["h2_all"], WORDS_H2, ST_)
    H2REF["status"] = "L3 REFERENCE ONLY: never an L1 arm, never part of the verdict (ruling 1)"
    CELLRES = {}
    for c in cells:
        D = compare("FLAT@%d" % M_MAIN, flat_all, "FLAT+LOC@%d" % M_MAIN, IND[c][C_MAIN], WORDS_LOC, ST_)
        o = D["paired (gained = arm covers ALL gold, reference does not)"]
        lost_q = np.flatnonzero(flat_all & ~IND[c][C_MAIN])
        gain_q = np.flatnonzero(~flat_all & IND[c][C_MAIN])
        assert len(lost_q) == o["lost"] and len(gain_q) == o["gained"]
        assert all(DISP[c][C_MAIN][j] > 0 for j in lost_q), "a lost query without a displaced gold"
        dr = np.asarray(DISP_RANKS[c], np.int64)
        assert (dr >= M_MAIN - C_MAIN + 1).all() and (dr <= M_MAIN).all()
        GL = {"G (queries LOC fixes)": int(len(gain_q)), "L (queries LOC breaks)": int(len(lost_q)),
              "gold_nodes_rescued (FLAT rank > M, served by a LOC slot)": int(RESC[c][C_MAIN].sum()),
              "gold_nodes_displaced (FLAT rank in [M-C+1, M], not served)": int(DISP[c][C_MAIN].sum()),
              "lost_gold_flat_ranks_all_in_[M-C+1,M] (asserted)": True,
              "lost_gold_flat_rank_1based": stats(dr) if len(dr) else None,
              "rescued_gold_flat_rank_1based": stats(RESC_RANKS[c]) if RESC_RANKS[c] else None,
              "queries_with_a_rescued_gold": int((RESC[c][C_MAIN] > 0).sum()),
              "queries_with_a_rescued_gold_but_still_not_ALL": int(((RESC[c][C_MAIN] > 0) & ~IND[c][C_MAIN]).sum()),
              "queries_with_a_displaced_gold": int((DISP[c][C_MAIN] > 0).sum()),
              "queries_with_a_displaced_gold_but_still_ALL (never: asserted by construction)": int(((DISP[c][C_MAIN] > 0) & IND[c][C_MAIN]).sum())}
        assert GL["queries_with_a_displaced_gold_but_still_ALL (never: asserted by construction)"] == 0
        au = AUD[c]
        cnt = collections.Counter(d["cat"] for d in au)
        nm_ = len(au)
        AUDIT = {"n_FLAT@M_missed_gold_nodes": nm_, "counts": {k: int(cnt.get(k, 0)) for k in CATS},
                 "shares": {k: (round(cnt.get(k, 0) / float(nm_), 4) if nm_ else None) for k in CATS},
                 "definitions": {"RESCUED": "served by one of the C LOC slots", "TYPE_A": "A(P(g)) = 0: no FLAT top-200 node in the gold's block",
                                 "TYPE_B": "A(P(g)) > 0 but the block received no LOC slot (region ranks too weakly for the C-slot budget)",
                                 "TYPE_C": "the block received >= 1 LOC slot but the gold is deeper in the block's FLAT order than its slots reach"},
                 "H2_crosstab (does the L3 reference serve the gold?)": {k: {"H2_serves": int(sum(1 for d in au if d["cat"] == k and d["h2_serves"])),
                                                                            "H2_misses": int(sum(1 for d in au if d["cat"] == k and not d["h2_serves"]))} for k in CATS},
                 "flat_rank_1based_by_type": {k: stats([d["flat_rank"] for d in au if d["cat"] == k]) for k in CATS},
                 "TYPE_B_detail": {"block_rank_by_A": stats([d["block_rank_by_A"] for d in au if d["cat"] == "TYPE_B"]),
                                   "weakest_picked_block_rank_by_A": stats([d["weakest_picked_block_rank_by_A"] for d in au if d["cat"] == "TYPE_B" and d.get("weakest_picked_block_rank_by_A") is not None]),
                                   "loc_index_minus_cut_index": stats([d["loc_index"] - d["cut_index"] for d in au if d["cat"] == "TYPE_B" and d["cut_index"] >= 0])},
                 "TYPE_C_detail": {"block_picks": stats([d["block_picks"] for d in au if d["cat"] == "TYPE_C"]),
                                   "nonbase_block_nodes_ahead_of_gold": stats([d["nonbase_block_nodes_ahead_of_gold"] for d in au if d["cat"] == "TYPE_C"]),
                                   "block_nonbase_nodes": stats([d["block_nonbase_nodes"] for d in au if d["cat"] == "TYPE_C"])},
                 "by_stratum": {}}
        for sn, masks in ST_.items():
            AUDIT["by_stratum"][sn] = {}
            for k_, m_ in masks.items():
                sub = [d for d in au if m_[d["row"]]]
                cc = collections.Counter(d["cat"] for d in sub)
                AUDIT["by_stratum"][sn][k_] = {"n_missed_gold_nodes": len(sub), **{k: int(cc.get(k, 0)) for k in CATS}}
        qsets = collections.Counter("+".join(sorted(set(QTYPES[c][j]))) for j in range(nA) if QTYPES[c][j])
        AUDIT["query_level (FLAT@M-ALL-missed queries by the set of types among their missed golds)"] = dict(sorted(qsets.items()))
        AUDIT["blocks_spanned (distinct blocks of a query's gold nodes)"] = {"all_golds": stats(blocks_spanned[c]),
                                                                            "missed_golds (queries with >= 1 FLAT@M-missed gold)": stats(blocks_spanned_missed[c][~flat_all])}
        REM = {"remaining_misses (gold nodes FLAT+LOC@M does not serve)": nm_ - int(cnt.get("RESCUED", 0)) + int(DISP[c][C_MAIN].sum()),
               "of_which_FLAT_missed_not_rescued (TYPE_A + TYPE_B + TYPE_C)": nm_ - int(cnt.get("RESCUED", 0)),
               "of_which_displaced_FLAT_tail": int(DISP[c][C_MAIN].sum()),
               "TYPE_A_share_of_FLAT_missed_not_rescued": (round(cnt.get("TYPE_A", 0) / float(nm_ - cnt.get("RESCUED", 0)), 4) if nm_ - cnt.get("RESCUED", 0) else None)}
        CURVE = {}
        for C in C_CURVE:
            CURVE[str(C)] = {"ALL_FLAT+LOC": q4(IND[c][C].mean()), "ANY_FLAT+LOC": q4(INDANY[c][C].mean()),
                             "paired_vs_FLAT@M": paired(flat_all, IND[c][C]), "gold_nodes_rescued": int(RESC[c][C].sum()),
                             "gold_nodes_displaced": int(DISP[c][C].sum())}
        CELLRES[c] = {"partition": {"tag": TAG_OF[c], "path": rel(parts[c].path), "sha256": sha_file(parts[c].path)},
                      "D_LOC (pre-registered decisive: FLAT@5000 vs FLAT+LOC@5000, C = 100; gained = LOC covers ALL gold, FLAT does not)": D,
                      "GL_decomposition": GL, "audit_ABC (explanatory only; cannot modify A)": AUDIT, "remaining_misses": REM,
                      "C_curve (descriptive; M = 5000)": CURVE, "ANY": {"FLAT": q4(flat_any.mean()), "FLAT+LOC": q4(INDANY[c][C_MAIN].mean())},
                      "_ind_FLAT+LOC_%d" % M_MAIN: IND[c][C_MAIN].astype(int).tolist()}
        if not DRY:
            log("D_LOC %s: FLAT@%d %.4f vs FLAT+LOC@%d %.4f (%+.2f pts) +%d/-%d p=%.3g -> %s" % (
                c, M_MAIN, D["ALL_reference"], M_MAIN, D["ALL_arm"], D["points_arm_minus_reference"], o["gained"], o["lost"], o["p"], D["verdict"]))
            log("   audit %s: %s ; remaining %s" % (c, json.dumps(AUDIT["counts"]), json.dumps(REM)))
    if not DRY:
        oh = H2REF["paired (gained = arm covers ALL gold, reference does not)"]
        log("L3 reference %s: FLAT@%d %.4f vs FLAT+H2@%d %.4f (+%d/-%d)" % (ds, M_MAIN, H2REF["ALL_reference"], M_MAIN, H2REF["ALL_arm"], oh["gained"], oh["lost"]))
    res = {"dataset": ds, "stage": "E", "mode": "PREREGISTERED_HARD_LOC_A", "status": "PRE-REGISTERED ONE-SHOT (the one mechanism-selection read for LOC)",
           "population_label": "HARD_LOC_A selection-fresh (previously unused for L1 mechanism evaluation/selection); not globally untouched",
           "eval_split": cd.eval_split, "N": N, "n_rows": nA, "population": POPV, "cells": cells,
           "stage_G": {"json": {"path": rel(fpG), "sha256": sha_file(gj)}, "npz": {"path": rel(fpG.replace(".json", ".npz")), "sha256": sha_file(gz)}},
           "gold": "DRY RUN: STAND-IN gold nodes of the true per-row count" if DRY else "query_index.npz gold positions (stamp-checked), read once",
           "gold_nodes_per_query": stats(ngold), "hops_available": sorted(set(int(x) for x in hopsA if x >= 0)),
           "FLAT@%d" % M_MAIN: {"ALL": q4(flat_all.mean()), "ANY": q4(flat_any.mean())},
           "cells_result": CELLRES, "H2_L3_reference": H2REF,
           "_ind_FLAT_%d" % M_MAIN: flat_all.astype(int).tolist(), "_ind_FLAT+H2_%d" % M_MAIN: rH["h2_all"].astype(int).tolist(),
           "_row_query_ids": qids}
    res.update(common_record())
    finish(res, fpE)


# ======================================================================== SUMMARY (the pre-registered five-cell pattern)
if MODE == "SUMMARY":
    fp_out = os.path.join(OUT, "hardloc_SUMMARY.json")
    if not DRY:
        assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    if CHECK:
        log("pre-registration check passed (SUMMARY); nothing written")
        sys.exit(0)
    per_cell, recs = {}, {}
    for d_ in CELLS:
        fp_ = os.path.join(DRY_IN if DRY else OUT, "hardloc_E_%s.json" % d_)
        r_ = json.load(open(fp_, encoding="utf-8"))
        assert r_["code"]["sha256"] == CODE_SHA and r_["dry"] == DRY
        recs[d_] = {"path": rel(fp_) if not DRY else os.path.basename(fp_), "sha256": sha_file(fp_)}
        for c in CELLS[d_]:
            cr = r_["cells_result"][c]
            Dk = [k for k in cr if k.startswith("D_LOC")][0]
            o = cr[Dk]["paired (gained = arm covers ALL gold, reference does not)"]
            per_cell[c] = {"label": cr[Dk]["label"], "verdict": cr[Dk]["verdict"], "ALL_FLAT": cr[Dk]["ALL_reference"], "ALL_FLAT+LOC": cr[Dk]["ALL_arm"],
                           "points": cr[Dk]["points_arm_minus_reference"], "gained": o["gained"], "lost": o["lost"], "p": o["p"],
                           "audit_shares": cr["audit_ABC (explanatory only; cannot modify A)"]["shares"],
                           "TYPE_A_share_of_FLAT_missed_not_rescued": cr["remaining_misses"]["TYPE_A_share_of_FLAT_missed_not_rescued"]}
    labs = [v["label"] for v in per_cell.values()]
    if all(x == "GAIN" for x in labs):
        pattern = "LOC_HELPS_IN_EVERY_CELL"
    elif "GAIN" in labs and "LOSS" not in labs:
        pattern = "LOC_HELPS_REGIME_DEPENDENT"
    elif "GAIN" in labs and "LOSS" in labs:
        pattern = "LOC_MIXED"
    elif "LOSS" in labs:
        pattern = "LOC_HURTS"
    else:
        pattern = "LOC_NO_SIGNIFICANT_EFFECT"
    res = {"mode": "PREREGISTERED_HARD_LOC_A_SUMMARY", "records": recs, "per_cell": per_cell, "pattern": pattern,
           "pattern_rule": "LOC_HELPS_IN_EVERY_CELL iff all five cells GAIN; LOC_HELPS_REGIME_DEPENDENT iff >= 1 GAIN and no LOSS; LOC_MIXED iff >= 1 GAIN and >= 1 LOSS; LOC_HURTS iff >= 1 LOSS and no GAIN; LOC_NO_SIGNIFICANT_EFFECT otherwise (GAIN / LOSS = exact McNemar alpha 0.01 with gained > lost / lost > gained)",
           "positive_structural_signal (>= 1 cell LOC_HELPS; the operational reading of ruling 1 section 9)": bool("GAIN" in labs),
           "reopening": "soft membership / hierarchy stay CLOSED; any reopening is the user's ruling on this record (ruling 1 sections 8-10, verbatim in the pre-registration)"}
    res.update(common_record())
    log("SUMMARY: %s %s" % (pattern, json.dumps({c: v["verdict"] for c, v in per_cell.items()})))
    finish(res, fp_out)
