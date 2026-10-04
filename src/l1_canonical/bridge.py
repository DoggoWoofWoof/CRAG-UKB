"""Legacy <-> canonical QUERY bridge, for the one question the legacy L1 artefacts still answer:
does canonical-layout L1 reproduce legacy L1 where the evaluation populations overlap?

Legacy query ids (data/ukb_storage/<legacy>/gte_qwen/query_ids_all.json, e.g. musique_clean_q_train_1)
share nothing with canonical ids (2hop__482757_12019) except the question text, which the legacy
question nodes carry in data/processed/master_nodes_<legacy>.json ("type": "question", "content")
and the canonical split files carry as "question" (metaqa: question_plain).  The bridge is built
by normalised question text, streamed (no full json.load), and records every ambiguity.  Reads only.

    python src/l1_canonical/bridge.py build <ds> [ds ...]   -> data/l1_canonical/<ds>/legacy_bridge.json
    python src/l1_canonical/bridge.py compare <ds> [ds ...] -> results/L1_CANONICAL/LEGACY_VS_CANONICAL_<ds>.json
        paired BASE / SAFE ALL@P50 indicators on the overlap of the legacy replay cache rows and the
        canonical replay cache rows (each in its own universe), exact McNemar

TEST rows are never read on either side: the legacy cache rows are val(+train) rows, the canonical
cache rows are EVAL_SPLITS rows.  For webqsp the legacy universe was built from WebQSP TEST ids
while the canonical eval population is the train_holdout carve, so the overlap is empty by design
and no comparison is made there.
"""
import io
import json
import os
import re
import sys
import time
import unicodedata

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for p in (REPO, os.path.join(REPO, "scratchpad")):
    if p not in sys.path:
        sys.path.insert(0, p)
from src.l1_canonical.adapter import CanonicalDataset, sha_file  # noqa: E402

LEGACY = {"metaqa": "metaqa", "squad": "squad_clean", "musique": "musique_clean", "2wiki": "2wiki_clean", "hotpotqa": "hotpotqa_clean", "webqsp": "webqsp"}
LEGACY_RUNS = os.path.join(REPO, "results", "GENERALIZATION", "G2_L1_PARTITION_SEARCH", "runs")
OUT = os.path.join(REPO, "results", "L1_CANONICAL")
_Q_RE = re.compile(r'\{"node_id": "((?:[^"\\]|\\.)*_q_(?:[^"\\]|\\.)*)", "content": "((?:[^"\\]|\\.)*)"')
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def norm(q):
    q = unicodedata.normalize("NFKC", q).lower()
    q = re.sub(r"[\[\]]", "", q)                      # metaqa topic-entity brackets
    q = re.sub(r"[^\w\s]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def legacy_questions(legacy):
    """legacy query id -> question text, streamed from master_nodes_<legacy>.json."""
    p = os.path.join(REPO, "data", "processed", "master_nodes_%s.json" % legacy)
    out = {}
    with io.open(p, encoding="utf-8") as f:
        buf = ""
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            last = 0
            for m in _Q_RE.finditer(buf):
                out[json.loads('"' + m.group(1) + '"')] = json.loads('"' + m.group(2) + '"')
                last = m.end()
            buf = buf[max(last, len(buf) - 8192):]
    return out


def build(ds):
    legacy = LEGACY[ds]
    d = CanonicalDataset(ds)
    lj = json.load(io.open(os.path.join(REPO, "data", "ukb_storage", legacy, "gte_qwen", "query_ids_all.json"), encoding="utf-8"))
    lids = lj["ids"]
    si = lj["split_indices"]
    no_test = set(si.get("val", [])) | set(si.get("train", []))
    lq = legacy_questions(legacy)
    log("%s: %d legacy question nodes, %d legacy query ids" % (ds, len(lq), len(lids)))
    can = {}
    dup_c = set()
    field = "question_plain" if ds == "metaqa" else "question"
    for sp in d.splits:
        if sp == "test":
            continue
        for q in d.ds.queries(sp):
            k = norm(q.get(field) or q.get("question") or "")
            if k in can and can[k] != q["query_id"]:
                dup_c.add(k)
            can.setdefault(k, q["query_id"])
    bridge, missing, ambiguous, dup_l = {}, [], [], 0
    seen = {}
    for i in no_test:
        lid = lids[i]
        t = lq.get(lid)
        if t is None:
            missing.append(lid)
            continue
        k = norm(t)
        if k in seen:
            dup_l += 1
        seen[k] = lid
        if k in dup_c:
            ambiguous.append(lid)
        elif k in can:
            bridge[lid] = can[k]
        else:
            missing.append(lid)
    rec = {"dataset": ds, "legacy": legacy, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rule": "normalised question text (NFKC, lower, brackets dropped, punctuation -> space); legacy val+train rows only; canonical non-test splits only",
           "legacy_rows_considered": len(no_test), "bridged": len(bridge), "unbridged": len(missing), "ambiguous_canonical_text": len(ambiguous),
           "legacy_duplicate_texts": dup_l, "canonical_duplicate_texts": len(dup_c), "examples_unbridged": missing[:5],
           "DATASET_json_RECORD_SHA256": d.record_sha, "map": bridge}
    os.makedirs(d.derived_dir, exist_ok=True)
    p = os.path.join(d.derived_dir, "legacy_bridge.json")
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=0))
    log("%s: bridged %d / %d legacy (val+train) rows; unbridged %d; ambiguous %d -> %s" % (ds, len(bridge), len(no_test), len(missing), len(ambiguous), os.path.relpath(p, REPO)))
    return rec


def mcnemar(a, b):
    from math import comb
    a, b = np.asarray(a, np.int8), np.asarray(b, np.int8)
    g = int(((b == 1) & (a == 0)).sum())
    l = int(((b == 0) & (a == 1)).sum())
    n = g + l
    p = 1.0 if n == 0 else min(1.0, 2.0 * sum(comb(n, i) for i in range(min(g, l) + 1)) / (2.0 ** n))
    return {"legacy_only_covered": l, "canonical_only_covered": g, "net_canonical_minus_legacy": g - l, "mcnemar_p": round(p, 5), "sig": bool(p < 0.05)}


def compare(ds, suffix=""):
    """suffix '' = the production cache / replay (EVAL_SPLITS population); '_legacyrows' = the comparison-only
    cache built by comparison.py on the bridged legacy rows (writes LEGACY_VS_CANONICAL_<ds><suffix>.json)."""
    legacy = LEGACY[ds]
    d = CanonicalDataset(ds)
    ccp = d.cache_path()[:-4] + suffix + ".npz"
    bp = os.path.join(d.derived_dir, "legacy_bridge.json")
    br = json.load(io.open(bp, encoding="utf-8"))["map"] if os.path.exists(bp) else build(ds)["map"]
    rp = os.path.join(OUT, "L1_REPLAY_%s%s.json" % (ds, suffix))
    if not os.path.exists(rp):
        raise RuntimeError("%s: run src/l1_canonical/%s first" % (ds, "comparison.py" if suffix else "l1_eval.py"))
    R = json.load(io.open(rp, encoding="utf-8"))
    cz = np.load(ccp, allow_pickle=True)
    cmeta = json.loads(str(cz["meta_json"]))
    crow_of = {qid: i for i, qid in enumerate(cmeta["row_query_ids"])}
    lz = np.load(os.path.join(LEGACY_RUNS, "cache_%s.npz" % legacy), allow_pickle=True)
    lj = json.load(io.open(os.path.join(REPO, "data", "ukb_storage", legacy, "gte_qwen", "query_ids_all.json"), encoding="utf-8"))
    lrows = [int(x) for x in lz["rows"]]
    # legacy BASE / SAFE indicators for the legacy H4_SK partition (the like-for-like legacy cell)
    leg_rep = os.path.join(REPO, "results", "GENERALIZATION", "G2_L1_PARTITION_SEARCH", "L1_HYPERGRAPH_UNIVERSAL", "partitions", "REPLAY_%s.json" % legacy)
    LR = json.load(io.open(leg_rep, encoding="utf-8")) if os.path.exists(leg_rep) else {}
    tag = next((t for t in LR if "H4_SPLIT_PRESERVE" in t and t.endswith("SK")), None)
    if tag is None:
        tag = next((t for t in LR if "H4" in t and "SK" in t), None)
    if tag is None:
        raise RuntimeError("%s: no legacy H4 SK replay cell in %s (tags: %s)" % (ds, leg_rep, sorted(LR)[:8]))
    l_base = np.asarray(LR[tag]["_ind_BASE"], np.int8)
    l_safe = np.asarray(LR[tag]["_ind_F6"], np.int8)
    c_base = np.asarray(R["_ind_BASE"], np.int8)
    c_safe = np.asarray(R["_ind_SAFE"], np.int8)
    pairs = []
    for li, r in enumerate(lrows):
        cid = br.get(lj["ids"][r])
        if cid is not None and cid in crow_of:
            pairs.append((li, crow_of[cid]))
    if not pairs:
        split_of = {}
        for sp in d.splits:
            if sp != "test":
                for q in d.ds.queries(sp):
                    split_of[q["query_id"]] = sp
        where = {}
        for r in lrows:
            cid = br.get(lj["ids"][r])
            k = "unbridged" if cid is None else split_of.get(cid, "test_or_unknown")
            where[k] = where.get(k, 0) + 1
        rec = {"dataset": ds, "status": "NO_OVERLAP", "legacy_rows": len(lrows), "canonical_rows": len(crow_of),
               "legacy_cache_rows_land_in_canonical_split": where,
               "note": "empty overlap by construction: the legacy cache rows map into a canonical split that is not the canonical eval split "
                       "(webqsp: legacy universe = TEST ids vs train_holdout; squad: legacy val carve lies inside the official train split) -- "
                       "a paired check would need a comparison-only canonical cache on those (non-test) rows"}
    else:
        li = np.array([p[0] for p in pairs])
        ci = np.array([p[1] for p in pairs])
        rec = {"dataset": ds, "status": "COMPARED", "legacy_cell": tag, "legacy_rows": len(lrows), "canonical_rows": len(crow_of), "overlap": len(pairs),
               "legacy_universe": {"n_docs": int(LR[tag].get("nq", 0)) and None, "npart": LR[tag].get("npart")},
               "BASE": {"legacy_ALL_on_overlap": round(float(l_base[li].mean()), 4), "canonical_ALL_on_overlap": round(float(c_base[ci].mean()), 4), **mcnemar(l_base[li], c_base[ci])},
               "SAFE": {"legacy_ALL_on_overlap": round(float(l_safe[li].mean()), 4), "canonical_ALL_on_overlap": round(float(c_safe[ci].mean()), 4), **mcnemar(l_safe[li], c_safe[ci])},
               "agreement": {"BASE_same_indicator": round(float((l_base[li] == c_base[ci]).mean()), 4), "SAFE_same_indicator": round(float((l_safe[li] == c_safe[ci]).mean()), 4)},
               "full_population": {"legacy_BASE_ALL": LR[tag]["BASE_ALL_P50"], "legacy_SAFE_ALL": LR[tag]["F6_ALL_P50"], "canonical_BASE_ALL": R["BASE_ALL_P50"], "canonical_SAFE_ALL": R["SAFE_ALL_P50"]},
               "caveat": "different node universes (canonical is a strict superset), different partitions (both H4_SK, k = N // 100 of each universe) -- this is an outcome-level reproduction check, not a bit-parity claim"}
    rec.update({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "legacy_cache_sha256": sha_file(os.path.join(LEGACY_RUNS, "cache_%s.npz" % legacy)),
                "canonical_cache": os.path.relpath(ccp, REPO).replace("\\", "/"), "canonical_cache_sha256": sha_file(ccp),
                "canonical_population": ("COMPARISON_ONLY cache on the bridged legacy rows (comparison.py); not a production number"
                                         if suffix else "production cache (EVAL_SPLITS population)"),
                "bridge_file": os.path.relpath(bp, REPO).replace("\\", "/")})
    p = os.path.join(OUT, "LEGACY_VS_CANONICAL_%s%s.json" % (ds, suffix))
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    log("%s: %s %s" % (ds, rec["status"], {k: rec[k] for k in ("overlap", "BASE", "SAFE") if k in rec}))
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "build":
        for ds in a[1:]:
            build(ds)
    elif a and a[0] == "compare":
        for ds in a[1:]:
            compare(ds)
    else:
        print(__doc__)
