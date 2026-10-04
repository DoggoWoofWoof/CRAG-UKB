"""Comparison-only canonical replay on the LEGACY cache rows (query subsetting only).

For a dataset whose legacy evaluation population lies outside the canonical eval split -- squad:
the legacy val carve (SQuAD 2.0 train -> train/val/test) sits inside the official train split, so
bridge.py compare reports NO_OVERLAP by construction -- the user's question "does canonical-layout
L1 reproduce legacy L1 on the same queries?" still has an answer: build a second canonical replay
cache on exactly the bridged legacy rows and pair it with the legacy H4_SK cell.

Rules kept: the production cache (replay_cache.npz, EVAL_SPLITS population) is never touched; TEST
rows are never read (legacy rows that would land in a canonical test split are dropped and counted);
no contract file changes -- this module subclasses replay_cache.CanonicalInputs and re-uses the frozen
numerics (_l1ps_router / _l1kb_core via l1_eval) unchanged.  Query subsetting is allowed; the corpus is
never subset.

    python src/l1_canonical/comparison.py <ds> [ds ...]
        -> data/l1_canonical/<ds>/replay_cache_legacyrows.{npz,json}
        -> results/L1_CANONICAL/L1_REPLAY_<ds>_legacyrows.json
        -> results/L1_CANONICAL/LEGACY_VS_CANONICAL_<ds>_legacyrows.json   (paired McNemar, bridge.compare)
"""
import io
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for p in (REPO, os.path.join(REPO, "scratchpad")):
    if p not in sys.path:
        sys.path.insert(0, p)
import _l1ps_router as RT  # noqa: E402
import _l1kb_core as KB  # noqa: E402
from src.l1_canonical.adapter import CanonicalDataset, contract_hash, sha_file  # noqa: E402
from src.l1_canonical import replay_cache as RC  # noqa: E402
from src.l1_canonical import l1_eval as LE  # noqa: E402
from src.l1_canonical import bridge as BR  # noqa: E402

SUFFIX = "_legacyrows"
OUT = LE.OUT
log = LE.log


class LegacyRowsInputs(RC.CanonicalInputs):
    """CanonicalInputs whose row population is the legacy replay-cache rows, bridged by question
    text into canonical query rows of any NON-TEST split, instead of EVAL_SPLITS."""

    def __init__(self, ds, tag="H4_SK"):
        super(LegacyRowsInputs, self).__init__(ds, tag)
        d = self.d
        legacy = BR.LEGACY[ds]
        bp = os.path.join(d.derived_dir, "legacy_bridge.json")
        br = json.load(io.open(bp, encoding="utf-8"))["map"] if os.path.exists(bp) else BR.build(ds)["map"]
        lcp = os.path.join(BR.LEGACY_RUNS, "cache_%s.npz" % legacy)
        lz = np.load(lcp, allow_pickle=True)
        lj = json.load(io.open(os.path.join(REPO, "data", "ukb_storage", legacy, "gte_qwen", "query_ids_all.json"), encoding="utf-8"))
        row_of = {qid: i for i, qid in enumerate(d.query_ids)}
        test = set()
        if "test" in d.split_ranges:
            a, b = d.split_ranges["test"]
            test = set(range(a, b))
        rows, where = [], {}
        for r in (int(x) for x in lz["rows"]):
            cid = br.get(lj["ids"][r])
            cr = row_of.get(cid) if cid is not None else None
            k = "unbridged" if cr is None else ("test_EXCLUDED" if cr in test else "kept")
            where[k] = where.get(k, 0) + 1
            if k == "kept":
                rows.append(int(cr))
        lz.close()
        self.rows = sorted(set(rows))
        self.sample_rule = ("COMPARISON_ONLY: rows = legacy runs/cache_%s.npz rows bridged by normalised question text into canonical "
                            "query rows (non-test splits only; TEST never read)" % legacy)
        self.val_set = set(self.rows)
        self.bridge_census = where
        self.legacy_cache = {"file": os.path.relpath(lcp, REPO).replace("\\", "/"), "sha256": sha_file(lcp), "rows": int(sum(where.values()))}
        self.bridge_file = os.path.relpath(bp, REPO).replace("\\", "/")


def cache_path(d):
    return d.cache_path()[:-4] + SUFFIX + ".npz"


def build_comparison(ds, tag="H4_SK"):
    c = RC.census([ds])
    if not c["ok"]:
        raise RuntimeError("free space %.1f GB would drop under the %.1f GB floor" % (c["disk_free_GB"], RC.FREE_SPACE_FLOOR_GB))
    I = LegacyRowsInputs(ds, tag)
    if not I.rows:
        raise RuntimeError("%s: no legacy cache row bridges into a non-test canonical row (%s)" % (ds, I.bridge_census))
    log("%s: comparison rows %d  bridge census %s" % (ds, len(I.rows), I.bridge_census))
    out = cache_path(I.d)
    ch = contract_hash()
    meta = RC.build(I, out, extra_meta={"COMPARISON_ONLY": True, "bridge_census": I.bridge_census, "legacy_cache": I.legacy_cache,
                                        "FROZEN_AGAINST": {
        "dataset_hash": I.d.record_sha, "nodes_jsonl_sha256(node_order)": I.d.manifest["nodes"]["sha256"],
        "query_hash": {"query_ids_sha256": I.d.manifest["queries"]["query_ids"]["sha256"],
                       "split_files_sha256": {sp: I.d.manifest["queries"]["by_split"][sp]["sha256"] for sp in I.d.splits}},
        "partition_hash": I.part_manifest["output"]["sha256"], "L1_CONTRACT_SHA256": ch["L1_CONTRACT_SHA256"],
        "retrieval_cache_sha256": I.d.pins()["retrieval_cache_sha256"]}})
    man = {"dataset": ds, "COMPARISON_ONLY": True, "production_cache_untouched": os.path.relpath(I.d.cache_path(), REPO).replace("\\", "/"),
           "cache": os.path.relpath(out, REPO).replace("\\", "/"), "bytes": os.path.getsize(out), "sha256": sha_file(out),
           "meta": {k: v for k, v in meta.items() if k != "row_query_ids"}, "contract_files": ch["files"],
           "bridge_file": I.bridge_file, "legacy_cache": I.legacy_cache}
    with io.open(out[:-4] + ".json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(man, indent=1))
    return man


def evaluate_comparison(ds, tag="H4_SK"):
    """l1_eval.evaluate's numerics, unchanged, on the comparison-only cache; writes L1_REPLAY_<ds>_legacyrows.json."""
    d = CanonicalDataset(ds)
    cp = cache_path(d)
    man = json.load(io.open(cp[:-4] + ".json", encoding="utf-8"))
    if man["sha256"] != sha_file(cp):
        raise RuntimeError("%s: comparison cache digest changed since its manifest" % ds)
    z0 = np.load(cp, allow_pickle=True)
    meta = json.loads(str(z0["meta_json"]))
    z = {k: z0[k] for k in z0.files if k != "meta_json"}
    fa = meta["FROZEN_AGAINST"]
    if fa["dataset_hash"] != d.record_sha:
        raise RuntimeError("%s: cache frozen against another DATASET.json record" % ds)
    if fa["partition_hash"] != sha_file(d.partition_path(tag)):
        raise RuntimeError("%s: cache frozen against another %s partition" % (ds, tag))
    CFG = LE.CFG
    hard = np.asarray(z["hard"], np.int64)
    npart = int(meta["npart"])
    nq = meta["n_dev_queries"]
    t = time.time()
    C = RT.build_cache(ds, z, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
    goldp = [set(int(x) for x in z["gold_part"][z["gold_ptr"][qi]:z["gold_ptr"][qi + 1]]) for qi in range(nq)]
    ind_base = np.array([int(goldp[qi] <= C["base50"][qi]) for qi in range(nq)], np.int8)
    ctxs = KB.contexts(z, meta, C, CFG["B"], CFG)
    r6 = KB.run_selector(ctxs, KB.sel_f6, goldp, ind_base)
    rb = KB.run_selector(ctxs, KB.sel_base, goldp, ind_base)
    sizes = np.asarray(z["part_sizes"])
    scope = np.array([int(sizes[sorted(C["base50"][qi])].sum()) for qi in range(nq)])
    scope6 = np.array([int(sizes[sorted(fs)].sum()) for fs in r6["finals"]])
    mom = round(float(sizes.max() / max(sizes.mean(), 1e-9)), 4)
    R = {"RECORD": "L1_CANONICAL_REPLAY_COMPARISON_ONLY", "dataset": ds, "partition_tag": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "COMPARISON_ONLY": True, "not_a_production_number": "population = bridged legacy cache rows (see population.sample_rule), not EVAL_SPLITS; "
                                                            "the production replay is results/L1_CANONICAL/L1_REPLAY_%s.json" % ds,
         "population": {"sample_rule": meta["sample_rule"], "nq": nq, "row_query_ids_sha256": meta["row_query_ids_sha256"],
                        "bridge_census": meta.get("bridge_census"), "legacy_cache": meta.get("legacy_cache")},
         "npart": npart, "n_docs": int(len(hard)),
         "BASE_ALL_P50": round(float(ind_base.mean()), 4),
         "BASE_ANY_P50": round(float(np.mean([int(bool(goldp[qi] & C["base50"][qi])) for qi in range(nq)])), 4),
         "BASE_SCOPE_NODES": round(float(scope.mean()), 1),
         "SAFE_ALL_P50": round(r6["ALL"], 4), "SAFE_ANY_P50": round(r6["ANY"], 4), "SAFE_SCOPE_NODES": round(float(scope6.mean()), 1),
         "BND_ALL_P50": round(rb["ALL"], 4),
         "CORR": {"dALL": round(r6["ALL"] - float(ind_base.mean()), 4), "gained": r6["gained"], "lost": r6["lost"], "net": r6["net"],
                  "mcnemar_p": r6["mcnemar_p"], "sig": r6["sig"]},
         "SELECTOR": {"cfg": {k: v for k, v in CFG.items()}, "code": "_l1ps_router.build_cache + _l1kb_core.contexts/run_selector/sel_f6 (unchanged; via l1_eval)"},
         "BALANCE": {"npart": npart, "blocks_used": int((sizes > 0).sum()), "max_over_mean": mom, "ELIGIBLE": bool(mom <= LE.MAX_OVER_MEAN and bool((sizes > 0).all()))},
         "FROZEN_AGAINST": fa, "cache": {"file": man["cache"], "sha256": man["sha256"], "bytes": man["bytes"]},
         "L1_CONTRACT_SHA256": contract_hash()["L1_CONTRACT_SHA256"],
         "seconds": round(time.time() - t, 1)}
    hops = np.asarray(z["hops"])
    if (hops >= 0).any():
        R["by_hop"] = {}
        for h in sorted(set(int(x) for x in hops if x >= 0)):
            m = hops == h
            R["by_hop"]["hop%d" % h] = {"n": int(m.sum()), "BASE_ALL": round(float(ind_base[m].mean()), 4), "SAFE_ALL": round(float(r6["ind"][m].mean()), 4)}
    R["_ind_SAFE"] = r6["ind"].tolist()
    R["_ind_BASE"] = ind_base.tolist()
    fp = os.path.join(OUT, "L1_REPLAY_%s%s.json" % (ds, SUFFIX))
    with io.open(fp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(R, indent=1))
    log("%-9s COMPARISON-ONLY nq %4d  BASE %.4f  SAFE %.4f  corr %+.4f (p=%s, +%d/-%d)  ANY %.4f/%.4f  scope %.0f -> %s"
        % (ds, nq, R["BASE_ALL_P50"], R["SAFE_ALL_P50"], R["CORR"]["dALL"], R["CORR"]["mcnemar_p"], R["CORR"]["gained"], R["CORR"]["lost"],
           R["BASE_ANY_P50"], R["SAFE_ANY_P50"], R["BASE_SCOPE_NODES"], os.path.relpath(fp, REPO)))
    return R


def run(ds, tag="H4_SK"):
    build_comparison(ds, tag)
    evaluate_comparison(ds, tag)
    return BR.compare(ds, suffix=SUFFIX)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    for ds in a:
        run(ds)
