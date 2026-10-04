"""HARD_LOC_A secondary reference arm: the frozen composite PRIMARY (QMAX_BALANCED_H2_PATCH1) on the HARD_LOC_A populations.

Ruling 1 (verbatim in results/L1_COVPART/PREREGISTRATION_HARD_LOC_A.json), section 4: include PRIMARY "but secondary only";
"No architectural decision should be based on PRIMARY during this experiment."  The question it answers (same section):
"Does the old partition composite's MetaQA advantage reproduce on these L1-fresh rows?"

PRIMARY      the frozen section-24 composite (FREEZE f81ebb84) computed by the pinned transfer runner's own code: the constants and
             helpers of _l1c_transfer_composite.py and its body from the pinned uL3 head to the served-set indicators are exec'd
             verbatim (read-only); only its argv parsing / pre-registration block is replaced by this module's (a new cache name
             registered at run time, exactly as its transfer mode does).  The replay cache of the population is built by the
             pinned builder (src/l1_canonical/replay_cache.py build(), rows = the population) OUTSIDE data/.  Evaluation scope =
             split A of the population (sha1 parity of the query id, the composite's frozen rule).
B_q          PRIMARY's per-query exposure: kept QMAX_SAFE blocks + the patch nodes (== the QMAX_SAFE exposure, asserted)
Comparisons  split A of the population, ALL-gold, exact McNemar, alpha 0.01; SECONDARY -- never part of the HARD_LOC_A verdict
    P1  PRIMARY vs FLAT@B_q          the section-29 matched-exposure question on the fresh rows
    P2  PRIMARY vs FLAT+H2@B_q       the section-30 D2 question (H2 = the stage-G L3 reference beam, asserted == the composite's)
    P3  PRIMARY vs FLAT+LOC@B_q      descriptive
Modes (records write-once)
    IDENT <DEV_A cache> --scratch=<dir>   harness identity on the frozen DEV_A cache (read-only; split A; the DEV_A gold only in
                          asserted equalities): B_q and the PRIMARY indicator == flat_A_<cache>.json; the builder, given the DEV_A
                          rows, reproduces every array of the frozen cache (output in <dir>, outside the repository)
                          -> results/L1_COVPART/hardloc_P_ident_<cache>.json
    RUN <dataset>         after stage E of the dataset: per cell, the population cache (results/L1_COVPART/hardloc_caches/) and
                          the composite -> PRIMARY; one FLAT_RRF pass (population batches; every top-M and LOC digest == stage G);
                          gold -> results/L1_COVPART/hardloc_P_<dataset>.json
    --dry                 IDENT: no record.  RUN: the first DRY_ROWS rows of the dataset's DEV_A cache, caches in --dry-out, stage G
                          from --dry-in, STAND-IN gold (the true per-row count) for every number this module reports; the
                          composite's own internal checks read the DEV_A gold as they always do; no record in the repository
Memory guard (pre-registered): a cache build starts only if the host's available memory >= the builder's predicted peak
(N x 1536 x 4 bytes + adjacency) + GUARD_GIB; otherwise that cell is NOT_RUN (host memory): no record, no foreign process
touched; it may run later under the same pinned code.
"""
import ast
import gc
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
from src.l1_canonical import replay_cache as RC

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = G.X.REPO
OUT = os.path.join(REPO, "results", "L1_COVPART")
PRE = os.path.join(OUT, "PREREGISTRATION_HARD_LOC_A.json")
POP_FILE = os.path.join(OUT, "hardloc_population.json")
CACHE_DIR = os.path.join(OUT, "hardloc_caches")
COMP = os.path.join(HERE, "_l1c_transfer_composite.py")
HLOC = os.path.join(HERE, "_l1c_hardloc.py")
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
DEV_A_CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
DS_OF = {"metaqa": "metaqa", "metaqa_phg": "metaqa", "squad": "squad", "squad_phg": "squad", "musique": "musique"}
TAG_OF = {"metaqa": "H4_SK", "metaqa_phg": "LOWMEM__PHG_REPAIR1_con", "squad": "H4_SK", "squad_phg": "LOWMEM__PHG_con",
          "musique": "LOWMEM__PHG_C1_con"}
CELLS = {"metaqa": ["metaqa", "metaqa_phg"], "squad": ["squad", "squad_phg"], "musique": ["musique"]}
K0 = TA.K0
C_MAIN = TA.K_LOCK
C_PATCH = C_MAIN                                             # the global of the pinned arm_row: the H2 reference at C = 100
M_MAIN = 5000
C_CURVE = (25, 50, 100, 200, 500)
ACT = TA.TOP200
BEAM, DEPTH = TA.K_LOCK, 2
CQ, BLOCK = TB.CQ, TB.BLOCK
ALPHA = 0.01
N_POP = AD.CONTRACT["EVAL_CAP"]
PER_HOP = N_POP // 3
SEED = 20260926
N_AGREE = 100
AGREE_MIN = 0.90
DRY_ROWS = 200
NG_BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5-10", 5, 10), ("11+", 11, 1 << 40))
GUARD_GIB = 0.5                                              # memory guard: available >= the builder's predicted peak + 0.5 GiB
CONSTANTS = {"K0": K0, "C_MAIN": C_MAIN, "M_MAIN": M_MAIN, "C_CURVE": list(C_CURVE), "ACT": ACT, "BEAM": BEAM, "DEPTH": DEPTH,
             "CQ": CQ, "BLOCK": BLOCK, "ALPHA": ALPHA, "N_POP": N_POP, "PER_HOP": PER_HOP, "SEED": SEED, "N_AGREE": N_AGREE,
             "AGREE_MIN": AGREE_MIN, "SEED_K": TA.SEED_K, "NG_BUCKETS": [list(b) for b in NG_BUCKETS]}
FLATH2_FNS = ("sha_file", "sha_text", "label", "_rss_mb", "host_state", "stats", "q4", "rel", "rrf_full", "chunk_iter",
              "dense_products", "splade_products", "arm_row", "paired", "strata_masks", "compare")
HLOC_FNS = ("pinned_sources", "make_arm", "Part", "activation", "loc_order", "flat_row", "dig", "batches", "hops_of", "gold_counts")
WORDS_P1 = {"GAIN": "FLAT_BEATS_PRIMARY", "LOSS": "PRIMARY_BEATS_FLAT", "NEUTRAL": "NO_SIGNIFICANT_DIFFERENCE"}
WORDS_P2 = {"GAIN": "FLAT_H2_BEATS_PRIMARY", "LOSS": "PRIMARY_BEATS_FLAT_H2", "NEUTRAL": "NO_SIGNIFICANT_DIFFERENCE"}
WORDS_P3 = {"GAIN": "FLAT_LOC_BEATS_PRIMARY", "LOSS": "PRIMARY_BEATS_FLAT_LOC", "NEUTRAL": "NO_SIGNIFICANT_DIFFERENCE"}
COMP_PRE_END = "name = sys.argv[1]\n"
COMP_MARK_HEAD = "# ---------------------------------------------------------------- the pinned uL3 head"
COMP_MARK_END = "\n\ndef mc(a, b, s):"
assert C_MAIN == 100 and ACT == 200 and K0 == 60


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b_ in iter(lambda: f.read(1 << 20), b""):
            h.update(b_)
    return h.hexdigest()


def _sources(path, names):
    src = open(path, "rb").read().decode("utf-8")
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            assert node.name not in out, "%s defines %s twice" % (path, node.name)
            out[node.name] = ast.get_source_segment(src, node)
    assert set(out) == set(names), (path, sorted(set(names) - set(out)))
    return out


for fn_, sha_ in PINNED.items():
    assert _sha_file(os.path.join(HERE, fn_)) == sha_, "pinned module %s changed" % fn_
for fn_, sha_ in PINNED_REPO.items():
    assert _sha_file(os.path.join(REPO, fn_)) == sha_, "pinned file %s changed" % fn_
FLATH2_SRC = _sources(os.path.join(HERE, "_l1c_flath2.py"), FLATH2_FNS)
HLOC_SRC = _sources(HLOC, HLOC_FNS)
for nm_ in FLATH2_FNS:                                       # the section-30 harness functions, verbatim
    exec(FLATH2_SRC[nm_], globals())
for nm_ in HLOC_FNS:                                         # the HARD_LOC_A functions (pinned with _l1c_hardloc.py), verbatim
    exec(HLOC_SRC[nm_], globals())
SRC_SHA = {k: sha_text(v) for k, v in sorted(dict(FLATH2_SRC, **{"hardloc." + k: v for k, v in HLOC_SRC.items()}).items())}
CSRC = open(COMP, "rb").read().decode("utf-8")
assert CSRC.count(COMP_PRE_END) == 1 and CSRC.count(COMP_MARK_HEAD) == 1 and CSRC.count(COMP_MARK_END) == 1
COMP_PRE = CSRC[:CSRC.index(COMP_PRE_END)]                     # docstring, imports, constants, helpers of the pinned runner
COMP_BODY = CSRC[CSRC.index(COMP_MARK_HEAD):CSRC.index(COMP_MARK_END)]   # uL3 head .. served-set indicators, verbatim
COMP_SLICES_SHA = {"pre (to 'name = sys.argv[1]')": sha_text(COMP_PRE), "body (uL3 head .. before 'def mc')": sha_text(COMP_BODY)}
FLATH2_SRC_ARM = FLATH2_SRC                                  # make_arm reads FLATH2_SRC["arm_row"]
ARM100 = make_arm(C_MAIN)


def run_composite(nm, cache_path):
    """the pinned runner's constants + helpers and its body, exec'd verbatim on `cache_path` under the new name `nm`;
    returns the gold-free served sets of PRIMARY on split A plus the composite's own PRIMARY indicator."""
    ns = {"__name__": "_l1c_transfer_composite__exec", "__file__": COMP, "__builtins__": __builtins__}
    exec(COMP_PRE, ns)
    for fn, sha in ns["PINNED"].items():                     # the runner's own pin check (its argv block, replaced here)
        assert _sha_file(os.path.join(HERE, fn)) == sha, "pinned module %s changed" % fn
    cache_path = os.path.abspath(cache_path)
    assert os.path.exists(cache_path), cache_path
    z_meta = json.loads(str(np.load(cache_path, allow_pickle=True)["meta_json"]))
    assert nm not in ns["DEV_A_CACHES"]
    G.X.CACHES[nm] = cache_path                              # registration at run time, exactly as the runner's transfer mode
    G.X.DS_OF[nm] = z_meta["dataset"]
    ns.update({"name": nm, "REPRO": False, "DRY": False, "CACHE_PATH": cache_path, "SAFE_REC_PATH": None, "fp_out": None,
               "z_meta": z_meta, "rec16": None, "rec21": None, "rec24": None})
    exec(COMP_BODY, ns)
    PRIM, ch = ns["PRIMARY"], "QMAX"
    rowsA = np.asarray(ns["rowsA"], np.int64)
    C = ns["C"]
    pt = ns["patches"][PRIM]
    out = {"rowsA": rowsA, "qids": list(C.qids), "rows": np.asarray(C.rows, np.int64), "hard": np.asarray(ns["hard"], np.int64),
           "sizes": np.asarray(ns["sizes"], np.int64),
           "kept": [sorted(int(b) for b in pt[qi]["kept"]) for qi in rowsA], "nodes": [sorted(int(v) for v in pt[qi]["nodes"]) for qi in rowsA],
           "B_q": np.asarray(ns["exposure"][ns["SAFE_OF"][ch]], np.int64), "ind_internal": np.asarray(ns["ind"][PRIM], bool)[rowsA],
           "bal1": [[int(x) for x in ns["bal1"][qi]] for qi in rowsA], "bal2": [[int(x) for x in ns["bal2"][qi]] for qi in rowsA],
           "n_fill": np.array([int(pt[qi]["n_fill"]) for qi in rowsA], np.int64), "nb": np.array([int(pt[qi]["nb"]) for qi in rowsA], np.int64),
           "meta": z_meta}
    for j in range(len(rowsA)):
        assert int(out["sizes"][out["kept"][j]].sum()) + len(out["nodes"][j]) == int(out["B_q"][j])
    ns.clear()                                               # break the exec'd functions' globals cycle: free D, C, STRUCT now
    del ns
    G.X.CACHES.pop(nm, None)
    G.X.DS_OF.pop(nm, None)
    gc.collect()
    return out


def primary_served(res, j, g):
    """ALL / ANY of the gold nodes g under PRIMARY's served set of split-A row j (kept blocks or patch nodes)."""
    kept, nodes, hard = set(res["kept"][j]), set(res["nodes"][j]), res["hard"]
    f = [int(hard[x]) in kept or int(x) in nodes for x in g]
    return all(f), any(f)


def predicted_peak_gib(ds, tag):
    cd_ = AD.CanonicalDataset(ds)
    N_ = int(cd_.n_nodes)
    s, d, _, _ = cd_.family("structural")
    n_e = int(len(s))
    return (N_ * 1536 * 4 + 2 * n_e * 4 + N_ * 24) / 2.0 ** 30


def guard(ds, tag):
    hs = host_state()
    need = predicted_peak_gib(ds, tag) + GUARD_GIB
    ok = hs.get("host_available_gib", 0.0) >= need
    return ok, {"host": hs, "need_gib": round(need, 3), "ok": bool(ok)}


def build_cache(ds, tag, rows_g, out, extra):
    I = RC.CanonicalInputs(ds, tag, log=log)
    I.rows = [int(r) for r in rows_g]
    I.sample_rule = "HARD_LOC_A population (rows given)"
    meta = RC.build(I, out, log=log, extra_meta=extra)
    del I
    gc.collect()
    z = np.load(out, allow_pickle=True)
    assert (z["rows"].astype(np.int64) == np.asarray(sorted(int(r) for r in rows_g), np.int64)).all(), "the builder did not keep exactly the given rows"
    z.close()
    return meta


# ---------------------------------------------------------------- arguments, pins, pre-registration
CODE_SHA = sha_file(os.path.abspath(__file__))
HLOC_SHA = sha_file(HLOC)
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
DRY = "--dry" in sys.argv
CHECK = "--check-prereg" in sys.argv
assert not (CHECK and DRY)
DRY_OUT = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--dry-out=")), None)
DRY_IN = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--dry-in=")), None)
SCRATCH = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--scratch=")), None)
for p_ in (DRY_OUT, DRY_IN, SCRATCH):
    assert p_ is None or not os.path.abspath(p_).lower().startswith(os.path.abspath(REPO).lower()), "scratch / dry directories live outside the repository"
assert DRY_OUT is None or DRY and DRY_IN is None or DRY
assert MODE in ("IDENT", "RUN"), "usage: IDENT <DEV_A cache> --scratch=<dir> | RUN <dataset>  [--dry | --check-prereg]"
ARG = sys.argv[2]
assert (MODE == "IDENT" and ARG in DEV_A_CACHES) or (MODE == "RUN" and ARG in CELLS)
pre, pre_sha = None, None
if not DRY:
    pre = json.load(open(PRE, encoding="utf-8"))
    pre_sha = sha_file(PRE)
    assert pre["code"]["new_modules"]["_l1c_hardloc_primary.py"]["sha256"] == CODE_SHA, "this module changed since the pre-registration"
    assert pre["code"]["new_modules"]["_l1c_hardloc.py"]["sha256"] == HLOC_SHA, "_l1c_hardloc.py changed since the pre-registration"
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    assert pre["constants"] == CONSTANTS, "constants differ from the pre-registration"
    assert pre["primary"]["composite_slices_sha256"] == COMP_SLICES_SHA and pre["primary"]["GUARD_GIB"] == GUARD_GIB
    for nm, pn in pre["records_read_only"].items():
        assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "record %s changed" % nm
    for cell, pn in pre["partitions_read_only"].items():
        assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "partition of %s changed" % cell
for c_, s_ in FLAT_RECORD.items():
    assert sha_file(os.path.join(OUT, "flat_A_%s.json" % c_)) == s_, "the section-29 record %s changed" % c_
HOST0 = host_state()
log("P %s %s%s: host at start %s" % (MODE, ARG, " (DRY)" if DRY else "", json.dumps(HOST0)))
T_ALL = time.time()
TIMES = {}


def finish(res, fp_out):
    res.update({"preregistration": {"path": rel(PRE), "sha256": pre_sha}, "code": {"path": rel(os.path.abspath(__file__)), "sha256": CODE_SHA},
                "hardloc_module": {"path": rel(HLOC), "sha256": HLOC_SHA}, "pinned": PINNED, "pinned_repo": PINNED_REPO,
                "function_sources_sha256": SRC_SHA, "composite_slices_sha256": COMP_SLICES_SHA, "constants": CONSTANTS, "GUARD_GIB": GUARD_GIB,
                "platform": {"python": platform.python_version(), "numpy": np.__version__}, "host_at_start": HOST0, "dry": DRY,
                "seconds_by_stage": TIMES, "seconds": round(time.time() - T_ALL, 1), "process_rss_mb_at_end": round(_rss_mb(), 1)})
    assert sha_file(os.path.abspath(__file__)) == CODE_SHA, "the module file changed during the run"
    if DRY:
        blob = json.dumps(res)
        if DRY_OUT:
            with open(os.path.join(DRY_OUT, os.path.basename(fp_out)), "w", encoding="utf-8") as f_:
                f_.write(blob)
        log("DRY RUN (%.0fs, RSS %.0f MB): record assembled (%d bytes) and NOT written to the repository.  times %s" % (
            time.time() - T_ALL, _rss_mb(), len(blob), json.dumps(TIMES)))
        sys.exit(0)
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    G.S.wj(fp_out, res)
    log("done (%.0fs) -> %s sha256 %s" % (res["seconds"], rel(fp_out), sha_file(fp_out)[:12]))
    sys.exit(0)


# ======================================================================== IDENT (frozen DEV_A cache; identities only)
if MODE == "IDENT":
    cache = ARG
    ds, tag = DS_OF[cache], TAG_OF[cache]
    fp_out = os.path.join(OUT, "hardloc_P_ident_%s.json" % cache)
    assert SCRATCH is not None and os.path.isdir(SCRATCH), "IDENT needs --scratch=<dir> outside the repository"
    if not DRY:
        assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
        assert sha_file(G.X.CACHES[cache]) == pre["caches_read_only"][cache]["sha256"], "cache %s changed" % cache
    if CHECK:
        log("pre-registration check passed (P IDENT %s); no data read, nothing written" % cache)
        sys.exit(0)
    frec = json.load(open(os.path.join(OUT, "flat_A_%s.json" % cache), encoding="utf-8"))
    t_ = time.time()
    res_c = run_composite("hlpid_%s" % cache, G.X.CACHES[cache])
    TIMES["composite"] = round(time.time() - t_, 1)
    nA = len(res_c["rowsA"])
    assert nA == int(frec["n_split_A"])
    assert res_c["B_q"].tolist() == frec["_budget_PRIMARY"], "PRIMARY exposure != the section-29 record"
    cd = AD.CanonicalDataset(ds)
    golds = [np.asarray(g, np.int64) for g in cd.gold(res_c["rows"][res_c["rowsA"]])]
    mine = np.array([primary_served(res_c, j, golds[j])[0] for j in range(nA)], bool)
    assert (mine == res_c["ind_internal"]).all(), "PRIMARY served-set evaluation != the composite's own indicator"
    assert mine.astype(int).tolist() == [int(x) for x in frec["_ind_PRIMARY"]], "PRIMARY indicator != the section-29 record"
    del golds, mine
    # the builder, given the DEV_A rows, reproduces every array of the frozen cache
    ok, gstate = guard(ds, tag)
    builder = {"guard": gstate}
    if ok:
        t_ = time.time()
        zf = np.load(G.X.CACHES[cache], allow_pickle=True)
        out_s = os.path.join(SCRATCH, "hlpid_rebuild_%s.npz" % cache)
        assert not os.path.exists(out_s)
        build_cache(ds, tag, zf["rows"], out_s, {"HARD_LOC_A": "P IDENT builder identity (scratch)"})
        zb = np.load(out_s, allow_pickle=True)
        diff = [k for k in RC.CACHE_KEYS if not (zb[k].shape == zf[k].shape and zb[k].dtype == zf[k].dtype and np.array_equal(zb[k], zf[k]))]
        mb, mf = json.loads(str(zb["meta_json"])), json.loads(str(zf["meta_json"]))
        assert not diff, "the rebuilt cache differs from the frozen DEV_A cache in %s" % diff
        assert mb["row_query_ids_sha256"] == mf["row_query_ids_sha256"] and mb["BASE_ALL_P50"] == mf["BASE_ALL_P50"] and mb["npart"] == mf["npart"]
        zb.close()
        zf.close()
        TIMES["builder_identity"] = round(time.time() - t_, 1)
        builder.update({"status": "PASS: every CACHE_KEYS array of the rebuilt cache == the frozen DEV_A cache (bitwise); row ids, BASE and npart meta equal",
                        "scratch_output": out_s, "arrays": list(RC.CACHE_KEYS)})
    else:
        builder["status"] = "NOT_RUN (host memory guard)"
    log("P IDENTITY GATE (%s): harness PASS (B_q and the PRIMARY indicator == flat_A_%s.json on %d split-A rows; served-set evaluation == the composite's indicator); builder %s" % (
        cache, cache, nA, builder["status"]))
    res = {"cache": cache, "dataset": ds, "partition_tag": tag, "mode": "HARD_LOC_A_PRIMARY_IDENTITY_GATE_DEV_A_SPLIT_A", "n_split_A": nA,
           "status": "harness PASS; builder %s (identities only; no number computed; nothing selected)" % builder["status"],
           "harness": "B_q == _budget_PRIMARY and the PRIMARY ALL indicator == _ind_PRIMARY of flat_A_%s.json (every split-A row); the served-set evaluation == the composite's own indicator" % cache,
           "builder_identity": builder, "cache_file": {"path": rel(G.X.CACHES[cache]), "sha256": sha_file(G.X.CACHES[cache])},
           "record_reproduced": {"path": "results/L1_COVPART/flat_A_%s.json" % cache, "sha256": FLAT_RECORD[cache]}}
    finish(res, fp_out)


# ======================================================================== RUN (population; after stage E)
ds = ARG
cd = AD.CanonicalDataset(ds)
N = int(cd.n_nodes)
cells = CELLS[ds]
fp_out = os.path.join(OUT, "hardloc_P_%s.json" % ds)
gdir = DRY_IN if DRY else OUT
fpG = os.path.join(gdir, "hardloc_G_%s.json" % ds)
fpGz = fpG.replace(".json", ".npz")
fpE = os.path.join(OUT, "hardloc_E_%s.json" % ds)
if DRY:
    rows_g = np.sort(np.load(G.X.CACHES[cells[0]], allow_pickle=True)["rows"].astype(np.int64))[:DRY_ROWS]
    cdir = DRY_OUT
    assert cdir is not None and DRY_IN is not None
else:
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    assert os.path.exists(fpE), "PRIMARY runs only after stage E of %s (the one gold read of the decisive comparison comes first)" % ds
    want = pre["datasets"][ds]
    assert {"DATASET_json_RECORD_SHA256": cd.record_sha, "query_index.npz": sha_file(cd._query_index_path()), "keys.npz": sha_file(cd._keys_path())} == want
    prec = json.load(open(POP_FILE, encoding="utf-8"))
    assert prec["preregistration"]["sha256"] == pre_sha
    pp = prec["populations"][ds]
    rows_g = np.asarray(pp["rows"], np.int64)
    assert [cd.query_ids[int(r)] for r in rows_g] == pp["query_ids"]
    cdir = CACHE_DIR
if CHECK:
    log("pre-registration check passed (P RUN %s); nothing written" % ds)
    sys.exit(0)
nA_all = len(rows_g)
qids = [cd.query_ids[int(r)] for r in rows_g]
POPV = {"n": nA_all, "rows_sha256": sha_text(",".join(str(int(r)) for r in rows_g)), "query_ids_sha256": sha_text(",".join(qids)),
        "population_file": None if DRY else {"path": rel(POP_FILE), "sha256": sha_file(POP_FILE)}}
grec = json.load(open(fpG, encoding="utf-8"))
assert grec["population"] == POPV and grec["dry"] == DRY and grec["code"]["sha256"] == HLOC_SHA
Z = np.load(fpGz, allow_pickle=False)
if not DRY:
    assert grec["npz"]["sha256"] == sha_file(fpGz)
assert (Z["rows"] == rows_g).all()
isA = np.array([int(hashlib.sha1(q.encode("utf-8")).hexdigest()[:8], 16) & 1 for q in qids], np.int8) == 0
jA = np.flatnonzero(isA)                                     # population positions of split A (sorted rows == cache row order)
nA = len(jA)
bal_arr, bal_len = Z["bal"], Z["bal_len"]
CELLRES, PRIM, NOT_RUN = {}, {}, {}
os.makedirs(cdir, exist_ok=True)
for c in cells:
    tag = TAG_OF[c]
    ok, gstate = guard(ds, tag)
    if not ok:
        NOT_RUN[c] = {"status": "NOT_RUN (host memory guard)", "guard": gstate}
        log("P %s: NOT_RUN (host memory guard: available %.2f GiB < need %.2f GiB)" % (c, gstate["host"].get("host_available_gib", -1), gstate["need_gib"]))
        continue
    t_ = time.time()
    outc = os.path.join(cdir, "hardloc_%s.npz" % c)
    extra = {"HARD_LOC_A": {"population": POPV, "cell": c, "partition_tag": tag, "dry": DRY}}
    if os.path.exists(outc):                                 # a derived cache from an interrupted run: reused only if it is this population's
        zm = json.loads(str(np.load(outc, allow_pickle=True)["meta_json"]))
        assert zm["row_query_ids_sha256"] == POPV["query_ids_sha256"] and zm["input_provider"] == tag and zm["HARD_LOC_A"]["population"] == POPV
        log("P %s: reusing the population cache %s (same population and partition)" % (c, outc))
    else:
        build_cache(ds, tag, rows_g, outc, extra)
    TIMES["cache_%s" % c] = round(time.time() - t_, 1)
    t_ = time.time()
    r_ = run_composite("hardloc_%s" % c, outc)
    TIMES["composite_%s" % c] = round(time.time() - t_, 1)
    assert r_["qids"] == qids and (r_["rows"] == rows_g).all() and (r_["rowsA"] == jA).all(), "cache rows / split A differ from the population"
    for k_, j in enumerate(jA):                              # the composite's balanced beam == the stage-G L3 reference beam
        assert r_["bal1"][k_] == [int(x) for x in bal_arr[j, :bal_len[j, 0]]] and r_["bal2"][k_] == [int(x) for x in bal_arr[j, bal_len[j, 0]:bal_len[j, 0] + bal_len[j, 1]]], \
            "composite balanced beam != stage-G beam (%s row %d)" % (c, j)
    PRIM[c] = r_
    PRIM[c]["cache_file"] = {"path": rel(outc) if not DRY else os.path.basename(outc), "sha256": sha_file(outc), "built_meta": {k: r_["meta"].get(k) for k in ("n_dev_queries", "npart", "BASE_SCOPE_NODES", "node_embedding_path", "built_utc")}}
    log("P %s: PRIMARY served sets on %d split-A rows; B_q %s (%.0fs)" % (c, nA, json.dumps(stats(r_["B_q"])), TIMES["composite_%s" % c]))
done = [c for c in cells if c in PRIM]
if not done:
    log("P %s: every cell NOT_RUN (host memory); no record" % ds)
    sys.exit(0)
LMAX = int(max(int(PRIM[c]["B_q"].max()) for c in done))
assert LMAX >= M_MAIN - C_MAIN
# ---- one FLAT_RRF pass over the population (the stage-G batches); every top-M / LOC digest == stage G; split-A orders kept to LMAX
t_ = time.time()
parts = {c: Part(cd, TAG_OF[c]) for c in done}
for c in done:
    assert (parts[c].hard == PRIM[c]["hard"]).all()
Qu = unit_queries(cd, rows_g) if "unit_queries" in globals() else None
if Qu is None:
    Qu = cd.query_embeddings[np.asarray(rows_g, np.int64)].astype(np.float32)
    Qu /= (np.linalg.norm(Qu, axis=1, keepdims=True) + 1e-9)     # as _l1s_core.Data / the pinned unit_queries
posA = {int(j): k for k, j in enumerate(jA)}
ordA = np.empty((nA, LMAX), np.int64)
locA = {c: [None] * nA for c in done}
for j0, j1, SD, SS in batches(cd, rows_g, Qu):
    for i in range(j1 - j0):
        j = j0 + i
        of, od, os_, npos_j, fv, frank = flat_row(SD[i], SS[i])
        assert (dig(of[:M_MAIN].astype(np.int32)) == Z["digest_flat_top_M"][j]).all(), "FLAT_RRF != stage G (row %d)" % j
        for c in done:
            P = parts[c]
            ub, A = activation(of[:ACT], fv, P.hard)
            L = loc_order(ub, A, frank, P)
            assert (dig(ub.astype(np.int32), A, L.astype(np.int32)) == Z["digest_loc_%s" % c][j]).all(), "LOC != stage G (%s row %d)" % (c, j)
            if j in posA:
                locA[c][posA[j]] = L.tolist()
        if j in posA:
            ordA[posA[j]] = of[:LMAX]
    log("  %s P flat rows %d / %d (%.0fs, RSS %.0f MB)" % (ds, j1, nA_all, time.time() - t_, _rss_mb()))
TIMES["flat_pass"] = round(time.time() - t_, 1)
# ---- gold (real, or STAND-IN of the true per-row count in a dry run -- the same stand-in as stage E's dry run)
if DRY:
    rng_dry = np.random.default_rng(12345)
    golds_all = [np.sort(rng_dry.choice(N, size=int(k), replace=False)).astype(np.int64) for k in gold_counts(cd, rows_g)]
else:
    golds_all = [np.asarray(g, np.int64) for g in cd.gold(rows_g)]
golds = [golds_all[j] for j in jA]
ngold = np.array([len(g) for g in golds], np.int64)
hopsA = hops_of(cd, rows_g, qids)[jA]
ST_ = strata_masks(hopsA, ngold)
if ds == "musique":
    dv0, dv1 = cd.split_ranges["dev"]
    rA = rows_g[jA]
    ST_["per_source"] = {"dev": (rA >= dv0) & (rA < dv1), "train": (rA < dv0) | (rA >= dv1)}
for c in done:
    r_ = PRIM[c]
    B_q = r_["B_q"]
    prim = np.zeros(nA, bool)
    prim_any = np.zeros(nA, bool)
    flat_b, h2_b, loc_b = np.zeros(nA, bool), np.zeros(nA, bool), np.zeros(nA, bool)
    for k in range(nA):
        g = golds[k]
        prim[k], prim_any[k] = primary_served(r_, k, g)
        Mq = int(B_q[k])
        o_ = ordA[k]
        fr = set(o_[:Mq].tolist())
        flat_b[k] = all(int(x) in fr for x in g)
        j = int(jA[k])
        bal = [int(x) for x in bal_arr[j, :bal_len[j, 0] + bal_len[j, 1]]]
        bset, tail, novel, fill = arm_row(o_, bal, Mq)
        sv = bset | set(novel) | set(fill)
        h2_b[k] = all(int(x) in sv for x in g)
        bset, tail, novel, fill = ARM100(o_, locA[c][k], Mq)
        sv = bset | set(int(x) for x in novel) | set(fill)
        loc_b[k] = all(int(x) in sv for x in g)
    if not DRY:
        assert (prim == r_["ind_internal"]).all(), "PRIMARY served-set evaluation != the composite's own indicator"
    P1 = compare("PRIMARY@B_q", prim, "FLAT@B_q", flat_b, WORDS_P1, ST_)
    P2 = compare("PRIMARY@B_q", prim, "FLAT+H2@B_q (L3 reference)", h2_b, WORDS_P2, ST_)
    P3 = compare("PRIMARY@B_q", prim, "FLAT+LOC@B_q", loc_b, WORDS_P3, ST_)
    CELLRES[c] = {"status": "SECONDARY REFERENCE ONLY: no architectural decision is based on PRIMARY (ruling 1 section 4); never part of the HARD_LOC_A verdict",
                  "partition": {"tag": TAG_OF[c], "path": rel(parts[c].path), "sha256": sha_file(parts[c].path)}, "cache": r_["cache_file"],
                  "n_split_A": nA, "B_q (PRIMARY per-query exposure)": stats(B_q),
                  "patch": {"patch_nodes (nb)": stats(r_["nb"]), "fill_nodes": stats(r_["n_fill"])},
                  "ALL": {"PRIMARY": q4(prim.mean()), "FLAT@B_q": q4(flat_b.mean()), "FLAT+H2@B_q": q4(h2_b.mean()), "FLAT+LOC@B_q": q4(loc_b.mean())},
                  "ANY": {"PRIMARY": q4(prim_any.mean())},
                  "P1 (PRIMARY vs FLAT@B_q; gained = FLAT covers ALL gold, PRIMARY does not)": P1,
                  "P2 (PRIMARY vs FLAT+H2@B_q; gained = FLAT+H2 covers, PRIMARY does not)": P2,
                  "P3 (descriptive: PRIMARY vs FLAT+LOC@B_q; gained = FLAT+LOC covers, PRIMARY does not)": P3,
                  "_ind_PRIMARY": prim.astype(int).tolist(), "_B_q": B_q.tolist()}
    if not DRY:
        for nm_, D_ in (("P1", P1), ("P2", P2), ("P3", P3)):
            o_ = D_["paired (gained = arm covers ALL gold, reference does not)"]
            log("%s %s: %s %.4f vs %s %.4f (+%d/-%d p=%.3g) -> %s" % (nm_, c, D_["reference"], D_["ALL_reference"], D_["arm"], D_["ALL_arm"], o_["gained"], o_["lost"], o_["p"], D_["verdict"]))
res = {"dataset": ds, "mode": "PREREGISTERED_HARD_LOC_A_PRIMARY_SECONDARY", "status": "SECONDARY REFERENCE (ruling 1 section 4); not a verdict",
       "population": POPV, "split": "A (sha1 parity of the query id; the composite's frozen evaluation scope)", "n_split_A": nA,
       "stage_G": {"path": rel(fpG) if not DRY else os.path.basename(fpG), "sha256": sha_file(fpG)},
       "gold": "DRY RUN: STAND-IN gold nodes of the true per-row count" if DRY else "query_index.npz gold positions (stamp-checked)",
       "cells_result": CELLRES, "cells_not_run": NOT_RUN, "_row_query_ids_split_A": [qids[j] for j in jA]}
finish(res, fp_out)
