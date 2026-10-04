"""Experiment 2 (algorithm substitution): the exact canonical L1 evaluation on the FREIGHT partition.

Same corpus, same H4_SK hypergraph, same k / imbalance / objective family, same query population (EVAL_SPLITS
sample, np.random.seed(0) -- the row sample does not depend on the partition), same frozen numerics
(_l1ps_router.build_cache + _l1kb_core selector via the l1_eval code path, unchanged), same BASE / SAFE
definitions; paired against the production replay results/L1_CANONICAL/L1_REPLAY_<ds>.json (Mt-KaHyPar H4_SK)
by exact two-sided McNemar on the per-query indicators.  Block labels are never compared -- only the downstream
mechanism (partition -> P50 ranking -> selector) on identical populations.  No tuning from the outcome.

    python -u src/l1_lowmem/l1_downstream.py <ds> [ds ...]
        -> data/l1_lowmem/<ds>/replay_cache__LOWMEM__FREIGHT_con.{npz,json}      (EXPERIMENTAL cache)
        -> results/L1_LOWMEM/L1_REPLAY_<ds>__LOWMEM__FREIGHT_con.json           (l1_eval numerics, FREIGHT arm)
        -> results/L1_LOWMEM/L1_DOWNSTREAM_<ds>.json                            (paired vs production, hop-wise)
The production cache / replay / partitions are never written.
"""
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, log, sha_file, rj, wj, ds_dir  # noqa: E402
import _l1ps_router as RT  # noqa: E402
import _l1kb_core as KB  # noqa: E402
from src.l1_canonical.adapter import CanonicalDataset, contract_hash  # noqa: E402
from src.l1_canonical import replay_cache as RC  # noqa: E402
from src.l1_canonical import l1_eval as LE  # noqa: E402
from src.l1_canonical import bridge as BR  # noqa: E402

TAG = "LOWMEM__FREIGHT_con"


def arm_name(tag):
    """label of the experimental partitioner behind a tag (FREIGHT for the frozen tags, PHG for the Zoltan-PHG lane)."""
    return "PHG" if "PHG" in tag else "FREIGHT"
PROD_OUT = LE.OUT


def cache_path(ds, tag=TAG):
    return os.path.join(ds_dir(ds), "replay_cache__%s.npz" % tag)


def build_cache(ds, tag=TAG):
    c = RC.census([ds])
    if not c["ok"]:
        raise RuntimeError("free space %.1f GB would drop under the %.1f GB floor" % (c["disk_free_GB"], RC.FREE_SPACE_FLOOR_GB))
    I = RC.CanonicalInputs(ds, tag)                       # reads parts/<tag>.{npy,json}; everything else identical
    if not I.part_manifest.get("EXPERIMENTAL"):
        raise RuntimeError("%s: parts/%s.json is not an experimental import" % (ds, tag))
    out = cache_path(ds, tag)
    ch = contract_hash()
    man = rj(out[:-4] + ".json")
    if man and os.path.exists(out) and man["sha256"] == sha_file(out):   # frozen replay-cache rule: reuse only if every pin still matches
        fa = man["meta"]["FROZEN_AGAINST"]
        if (fa["dataset_hash"] == I.d.record_sha and fa["partition_hash"] == I.part_manifest["output"]["sha256"]
                and fa["L1_CONTRACT_SHA256"] == ch["L1_CONTRACT_SHA256"] and fa["retrieval_cache_sha256"] == I.d.pins()["retrieval_cache_sha256"]):
            log("%s: experimental cache exists and matches every pin -- reused (%s)" % (ds, man["sha256"][:16]))
            return man
        log("%s: experimental cache pins changed -- rebuilding" % ds)
    meta = RC.build(I, out, extra_meta={"EXPERIMENTAL": True, "L1_LOWMEM_ARM": arm_name(tag), "FROZEN_AGAINST": {
        "dataset_hash": I.d.record_sha, "nodes_jsonl_sha256(node_order)": I.d.manifest["nodes"]["sha256"],
        "query_hash": {"query_ids_sha256": I.d.manifest["queries"]["query_ids"]["sha256"],
                       "split_files_sha256": {sp: I.d.manifest["queries"]["by_split"][sp]["sha256"] for sp in I.d.splits}},
        "partition_hash": I.part_manifest["output"]["sha256"], "partition_tag": tag, "L1_CONTRACT_SHA256": ch["L1_CONTRACT_SHA256"],
        "retrieval_cache_sha256": I.d.pins()["retrieval_cache_sha256"]}})
    man = {"dataset": ds, "EXPERIMENTAL": True, "partition_tag": tag, "production_cache_untouched": os.path.relpath(I.d.cache_path(), REPO).replace("\\", "/"),
           "cache": os.path.relpath(out, REPO).replace("\\", "/"), "bytes": os.path.getsize(out), "sha256": sha_file(out),
           "meta": {k: v for k, v in meta.items() if k != "row_query_ids"}, "contract_files": ch["files"]}
    wj(out[:-4] + ".json", man)
    return man


def evaluate_cache(ds, tag=TAG):
    """l1_eval.evaluate's numerics, unchanged, on the experimental cache (the production evaluate() hard-codes the production path)."""
    d = CanonicalDataset(ds)
    cp = cache_path(ds, tag)
    man = rj(cp[:-4] + ".json")
    if man is None or man["sha256"] != sha_file(cp):
        raise RuntimeError("%s: experimental cache missing or changed since its manifest" % ds)
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
    R = {"RECORD": "L1_LOWMEM_REPLAY", "dataset": ds, "partition_tag": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "EXPERIMENTAL": True, "not_a_production_number": "%s arm of the H4_SK low-memory validation; the production replay is "
                                                         "results/L1_CANONICAL/L1_REPLAY_%s.json (Mt-KaHyPar H4_SK)" % (arm_name(tag), ds),
         "population": {"eval_split": d.eval_split, "sample_rule": meta["sample_rule"], "nq": nq, "row_query_ids_sha256": meta["row_query_ids_sha256"]},
         "npart": npart, "n_docs": int(len(hard)),
         "BASE_ALL_P50": round(float(ind_base.mean()), 4),
         "BASE_ANY_P50": round(float(np.mean([int(bool(goldp[qi] & C["base50"][qi])) for qi in range(nq)])), 4),
         "BASE_SCOPE_NODES": round(float(scope.mean()), 1),
         "SAFE_ALL_P50": round(r6["ALL"], 4), "SAFE_ANY_P50": round(r6["ANY"], 4), "SAFE_SCOPE_NODES": round(float(scope6.mean()), 1),
         "BND_ALL_P50": round(rb["ALL"], 4),
         "CORR": {"dALL": round(r6["ALL"] - float(ind_base.mean()), 4), "gained": r6["gained"], "lost": r6["lost"], "net": r6["net"],
                  "mcnemar_p": r6["mcnemar_p"], "sig": r6["sig"], "gold_admitted": r6["gold_admitted"], "gold_evicted": r6["gold_evicted"],
                  "churn_per_query": round(float(r6["churn"].mean()), 3), "churn_max": int(r6["churn"].max()),
                  "queries_with_zero_churn": int((r6["churn"] == 0).sum())},
         "SELECTOR": {"cfg": {k: v for k, v in CFG.items()}, "code": "_l1ps_router.build_cache + _l1kb_core.contexts/run_selector/sel_f6 (unchanged; l1_eval numerics)"},
         "BALANCE": {"npart": npart, "blocks_used": int((sizes > 0).sum()), "size_min": int(sizes.min()), "size_max": int(sizes.max()),
                     "size_median": float(np.median(sizes[sizes > 0])), "size_mean": round(float(sizes.mean()), 2),
                     "size_p90": float(np.percentile(sizes[sizes > 0], 90)), "size_p99": float(np.percentile(sizes[sizes > 0], 99)),
                     "size_cv": round(float(sizes.std() / max(sizes.mean(), 1e-9)), 4), "max_over_mean": mom,
                     "ELIGIBLE": bool(mom <= LE.MAX_OVER_MEAN and bool((sizes > 0).all()))},
         "CUT": LE.cut_stats(d, hard),
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
    R["_hops"] = [int(x) for x in hops]
    fp = os.path.join(OUT, "L1_REPLAY_%s__%s.json" % (ds, tag))
    wj(fp, R)
    log("%-9s %s arm nq %4d  BASE %.4f  SAFE %.4f  corr %+.4f (p=%s, +%d/-%d)  ANY %.4f/%.4f  scope %.0f  max/mean %.3f  cut STRUCT %.4f KNN %.4f -> %s"
        % (ds, arm_name(tag), nq, R["BASE_ALL_P50"], R["SAFE_ALL_P50"], R["CORR"]["dALL"], R["CORR"]["mcnemar_p"], R["CORR"]["gained"], R["CORR"]["lost"],
           R["BASE_ANY_P50"], R["SAFE_ANY_P50"], R["BASE_SCOPE_NODES"], mom, R["CUT"]["STRUCT"]["edge_cut_fraction"], R["CUT"]["KNN"]["edge_cut_fraction"],
           os.path.relpath(fp, REPO)))
    return R


def paired(ds, F, tag=TAG):
    """FREIGHT arm vs the production Mt-KaHyPar replay on the identical query population (exact McNemar, hop-wise)."""
    P = rj(os.path.join(PROD_OUT, "L1_REPLAY_%s.json" % ds))
    D = {"RECORD": "L1_LOWMEM_DOWNSTREAM", "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "arms": {"A": "canonical Mt-KaHyPar H4_SK (results/L1_CANONICAL/L1_REPLAY_%s.json)" % ds, "B": "%s connectivity (%s)" % (arm_name(tag), tag)},
         "B": {k: F[k] for k in ("BASE_ALL_P50", "BASE_ANY_P50", "BASE_SCOPE_NODES", "SAFE_ALL_P50", "SAFE_ANY_P50", "SAFE_SCOPE_NODES", "BND_ALL_P50",
                                 "CORR", "BALANCE", "CUT", "by_hop") if k in F},
         "population_B": F["population"]}
    if P is None:
        D["status"] = "BASELINE_ABSENT"
        D["note"] = "no production replay for %s (the canonical Mt-KaHyPar H4_SK partition FAILED / is absent locally); %s-arm numbers only" % (ds, arm_name(tag))
        log("%-9s paired: baseline ABSENT" % ds)
        return D
    same_pop = P["population"]["row_query_ids_sha256"] == F["population"]["row_query_ids_sha256"] and P["population"]["nq"] == F["population"]["nq"]
    D["population_A"] = P["population"]
    D["identical_population"] = bool(same_pop)
    if not same_pop:
        D["status"] = "POPULATION_MISMATCH"
        log("%-9s paired: POPULATION MISMATCH (%s vs %s)" % (ds, P["population"]["row_query_ids_sha256"][:12], F["population"]["row_query_ids_sha256"][:12]))
        return D
    D["A"] = {k: P[k] for k in ("BASE_ALL_P50", "BASE_ANY_P50", "BASE_SCOPE_NODES", "SAFE_ALL_P50", "SAFE_ANY_P50", "SAFE_SCOPE_NODES", "BND_ALL_P50",
                                "CORR", "BALANCE", "CUT", "by_hop") if k in P}
    a_safe, b_safe = np.asarray(P["_ind_SAFE"], np.int8), np.asarray(F["_ind_SAFE"], np.int8)
    a_base, b_base = np.asarray(P["_ind_BASE"], np.int8), np.asarray(F["_ind_BASE"], np.int8)
    def mc(a, b):
        m = BR.mcnemar(a, b)
        return {"A_only_covered": m["legacy_only_covered"], "B_only_covered": m["canonical_only_covered"], "net_B_minus_A": m["net_canonical_minus_legacy"],
                "mcnemar_p": m["mcnemar_p"], "sig": m["sig"], "A": round(float(a.mean()), 4), "B": round(float(b.mean()), 4),
                "delta_B_minus_A": round(float(b.mean() - a.mean()), 4),
                "relative_change": round(float((b.mean() - a.mean()) / max(a.mean(), 1e-9)), 4)}
    D["paired"] = {"SAFE_ALL_P50": mc(a_safe, b_safe), "BASE_ALL_P50": mc(a_base, b_base), "nq": int(len(a_safe))}
    D["safe_additions"] = {"A": {"gained": P["CORR"]["gained"], "lost": P["CORR"]["lost"], "net": P["CORR"]["net"]},
                           "B": {"gained": F["CORR"]["gained"], "lost": F["CORR"]["lost"], "net": F["CORR"]["net"]}}
    D["coverage"] = {"BASE_ANY": {"A": P["BASE_ANY_P50"], "B": F["BASE_ANY_P50"]}, "SAFE_ANY": {"A": P["SAFE_ANY_P50"], "B": F["SAFE_ANY_P50"]},
                     "BASE_SCOPE_NODES": {"A": P["BASE_SCOPE_NODES"], "B": F["BASE_SCOPE_NODES"]},
                     "SAFE_SCOPE_NODES": {"A": P["SAFE_SCOPE_NODES"], "B": F["SAFE_SCOPE_NODES"]}}
    hops = np.asarray(F.get("_hops", []), np.int64)
    if hops.size and (hops >= 0).any():
        D["by_hop_paired"] = {}
        for h in sorted(set(int(x) for x in hops if x >= 0)):
            m = hops == h
            D["by_hop_paired"]["hop%d" % h] = {"n": int(m.sum()), "SAFE": mc(a_safe[m], b_safe[m]), "BASE": mc(a_base[m], b_base[m])}
    D["status"] = "PAIRED"
    s = D["paired"]["SAFE_ALL_P50"]
    log("%-9s paired SAFE A %.4f  B %.4f  d %+.4f (rel %+.1f%%, p=%s, sig %s, +%d/-%d)   BASE A %.4f B %.4f d %+.4f (p=%s)" % (
        ds, s["A"], s["B"], s["delta_B_minus_A"], 100 * s["relative_change"], s["mcnemar_p"], s["sig"], s["B_only_covered"], s["A_only_covered"],
        D["paired"]["BASE_ALL_P50"]["A"], D["paired"]["BASE_ALL_P50"]["B"], D["paired"]["BASE_ALL_P50"]["delta_B_minus_A"], D["paired"]["BASE_ALL_P50"]["mcnemar_p"]))
    return D


def run(ds, tag=TAG, suffix=""):
    """tag/suffix default to the frozen experiment's arm; a follow-up arm passes its own tag and a non-empty suffix so
    the frozen L1_DOWNSTREAM_<ds>.json is never rewritten."""
    if (tag != TAG) != bool(suffix):
        raise RuntimeError("a non-default tag needs a non-empty output suffix (and vice versa)")
    build_cache(ds, tag)
    F = evaluate_cache(ds, tag)
    D = paired(ds, F, tag)
    fp = os.path.join(OUT, "L1_DOWNSTREAM_%s%s.json" % (ds, suffix))
    wj(fp, D)
    log("wrote", os.path.relpath(fp, REPO))
    return D


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    for ds in a:
        run(ds)
