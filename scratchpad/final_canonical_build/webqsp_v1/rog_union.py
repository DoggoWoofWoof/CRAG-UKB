"""V1 §1 -- rebuild the deterministic RoG WebQSP union CWQ union and verify the hard consistency check.

    python scratchpad/final_canonical_build/webqsp_v1/rog_union.py

Required by the directive to reproduce EXACTLY:
    triples   = 8,309,195
    relations = 7,058
If it does not reproduce, STOP -- do not substitute the legacy 781k graph, the Phase-C 1.316M
graph, a third-party reconstruction, or a WebQSP-only union.

Sources (already on disk, per-shard sha256 verified against HF LFS by a prior audit):
    data/original/webqsp/rog_webqsp   5 shards   518,205,557 B   rev c0632533...
    data/original/cwq/rog_cwq        24 shards 3,504,045,643 B
data/original/ is READ-ONLY here; nothing is written to it.

Memory: endpoints and relations are interned to ints and each distinct triple is packed into a
single 58-bit int, so the 8.3M-triple set costs roughly 400 MB instead of ~1.3 GB of strings.
The packing is exact -- there is no hashing and therefore no collision risk on the acceptance count.
"""
import json, os, time

import pyarrow.parquet as pq

SRC = {"webqsp": "data/original/webqsp/rog_webqsp", "cwq": "data/original/cwq/rog_cwq"}
OUT = "data/final_canonical/webqsp/ROG_UNION_REBUILD.json"

TARGET_TRIPLES = 8309195
TARGET_RELATIONS = 7058
ENDPOINTS_REFERENCE = 2592894      # reported, NEVER an acceptance criterion (see pre-registration)

EP_BITS, REL_BITS = 22, 14         # 4,194,304 endpoints / 16,384 relations
EP_MAX, REL_MAX = (1 << EP_BITS) - 1, (1 << REL_BITS) - 1


def shards(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".parquet"))


def main():
    t0 = time.time()
    ep2id, rel2id = {}, {}
    triples = set()                                  # packed 58-bit codes, exact
    per = {}
    raw_triples = malformed = examples = 0
    shard_log = []

    for ds, d in SRC.items():
        seen_before = len(triples)
        ds_triples, ds_examples = set(), 0
        for sh in shards(d):
            p = f"{d}/{sh}"
            pf = pq.ParquetFile(p)
            for rg in range(pf.metadata.num_row_groups):
                # only the graph column is read: id/question/answer/entities are not needed for §1
                col = pf.read_row_group(rg, columns=["graph"]).column("graph").to_pylist()
                for g in col:
                    ds_examples += 1
                    for t in g:
                        raw_triples += 1
                        if t is None or len(t) != 3:
                            malformed += 1
                            continue
                        h, r, o = t
                        hi = ep2id.get(h)
                        if hi is None:
                            hi = ep2id[h] = len(ep2id)
                        oi = ep2id.get(o)
                        if oi is None:
                            oi = ep2id[o] = len(ep2id)
                        ri = rel2id.get(r)
                        if ri is None:
                            ri = rel2id[r] = len(rel2id)
                        if hi > EP_MAX or oi > EP_MAX or ri > REL_MAX:
                            raise SystemExit(
                                f"bit budget exceeded: endpoints={len(ep2id)} (max {EP_MAX + 1}), "
                                f"relations={len(rel2id)} (max {REL_MAX + 1}). Widen EP_BITS/REL_BITS.")
                        code = (hi << (REL_BITS + EP_BITS)) | (ri << EP_BITS) | oi
                        triples.add(code)
                        ds_triples.add(code)
            shard_log.append({"dataset": ds, "shard": sh, "bytes": os.path.getsize(p),
                              "cumulative_distinct_triples": len(triples)})
            print(f"[{ds}] {sh}  distinct={len(triples):,}  raw={raw_triples:,}  "
                  f"eps={len(ep2id):,}  rels={len(rel2id):,}  t={time.time() - t0:.0f}s", flush=True)
        per[ds] = {"examples": ds_examples, "distinct_triples": len(ds_triples),
                   "new_triples_contributed": len(triples) - seen_before}
        examples += ds_examples
        del ds_triples

    n_tri, n_rel, n_ep = len(triples), len(rel2id), len(ep2id)
    exact = (n_tri == TARGET_TRIPLES and n_rel == TARGET_RELATIONS)
    doc = {
        "schema": "ROG_UNION_REBUILD/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "V1 §1 hard consistency check on the released RoG WebQSP union CWQ graph",
        "sources": {ds: {"dir": d, "shards": len(shards(d)),
                         "bytes": sum(os.path.getsize(f"{d}/{s}") for s in shards(d))}
                    for ds, d in SRC.items()},
        "measured": {"triples": n_tri, "relations": n_rel, "endpoints": n_ep,
                     "examples": examples, "raw_triple_occurrences": raw_triples,
                     "malformed_triples": malformed,
                     "duplication_factor": round(raw_triples / n_tri, 3) if n_tri else None},
        "targets": {"triples": TARGET_TRIPLES, "relations": TARGET_RELATIONS},
        "deltas": {"triples": n_tri - TARGET_TRIPLES, "relations": n_rel - TARGET_RELATIONS,
                   "endpoints_vs_prior_audit": n_ep - ENDPOINTS_REFERENCE},
        "per_dataset": per,
        "endpoints_note": "Endpoints are REPORTED, never an acceptance criterion. The paper's 2,566,291 "
                          "and the prior audit's measured 2,592,894 differ by +1.04% and no counting "
                          "convention explains it (strip / whitespace / NFC / casefold / drop-literals / "
                          "head-only all tested and rejected).",
        "HARD_CHECK_EXACT": exact,
        "VERDICT": ("REPRODUCED -- proceed to V1 §2" if exact else
                    "NOT REPRODUCED -- STOP per directive §1; do not substitute any other graph"),
        "elapsed_s": round(time.time() - t0, 1),
        "shard_log": shard_log,
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k: doc[k] for k in ("measured", "targets", "deltas", "per_dataset",
                                          "HARD_CHECK_EXACT", "VERDICT", "elapsed_s")}, indent=1))


if __name__ == "__main__":
    main()
