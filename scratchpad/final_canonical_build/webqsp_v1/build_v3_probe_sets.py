"""Precompute everything checks 2-4 need from OUR side, before IDIR lands.

    python scratchpad/final_canonical_build/webqsp_v1/build_v3_probe_sets.py

When the package arrives the comparison should be a streaming set-membership test over its triple
files, not a re-derivation. This writes the probe sets to disk once.

Instrument choice matters here. V1's edges cannot serve as the MID-to-MID comparison target,
because RoG substituted names for MIDs and only 1,652,618 of 2,592,894 endpoints kept an
identifier at all. The NSM sibling preserves MIDs on both ends of every triple, so it -- not V1 --
is the structural oracle for check 4. V1 supplies the relation vocabulary and the frozen node_kind
labels for checks 3 and 7.

Writes to data/final_canonical/freebase_v3/probe/.
"""
import json, os, re, time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

V1 = "data/final_canonical/webqsp/v1"
NSM = "data/final_canonical/webqsp/_acquisition/nsm/extracted"
SPLITS = {"webqsp": f"{NSM}/webqsp/webqsp", "CWQ": f"{NSM}/CWQ/CWQ"}
OUTD = "data/final_canonical/freebase_v3/probe"
REPORT = "data/final_canonical/freebase_v3/V3_PROBE_SETS.json"

MID_RE = re.compile(r"^[mg]\.[0-9A-Za-z_]+$")


def main():
    t0 = time.time()
    os.makedirs(OUTD, exist_ok=True)

    # ---- V1 side: relation vocabulary (check 3) and node_kind by MID (check 7) ----
    rels = pq.read_table(f"{V1}/relations.parquet", columns=["relation_key"])
    rkeys = rels.column("relation_key").to_pylist()
    pq.write_table(pa.table({"relation_key": pa.array(rkeys, pa.string())}),
                   f"{OUTD}/v1_relation_keys.parquet", compression="zstd")

    nodes = pq.read_table(f"{V1}/nodes.parquet", columns=["source_mid", "node_kind"])
    smid = nodes.column("source_mid").to_pylist()
    kind = nodes.column("node_kind").to_pylist()
    pairs = [(m, k) for m, k in zip(smid, kind) if m]
    pq.write_table(pa.table({
        "mid": pa.array([m for m, _ in pairs], pa.string()),
        "node_kind": pa.array([k for _, k in pairs], pa.string()),
    }), f"{OUTD}/v1_mid_node_kind.parquet", compression="zstd")
    v1_kind_counts = Counter(k for _, k in pairs)
    print(f"[v1] relations={len(rkeys):,} mid_labelled={len(pairs):,} t={time.time()-t0:.0f}s",
          flush=True)

    # ---- NSM side: MID vocabulary, MID-to-MID triples, topic and answer MIDs (checks 2, 4) ----
    all_mids = set()
    topic_mids = set()
    answer_mids = set()
    tri = set()
    rel_used = set()
    per_ds = {}

    for ds, d in SPLITS.items():
        with open(f"{d}/entities.txt", encoding="utf-8") as fh:
            ents = [ln.rstrip("\n") for ln in fh]
        with open(f"{d}/relations.txt", encoding="utf-8") as fh:
            rnames = [ln.rstrip("\n") for ln in fh]
        for e in ents:
            if MID_RE.match(e):
                all_mids.add(e)

        n_q = n_tri = 0
        for split in ("train_simple.json", "dev_simple.json", "test_simple.json"):
            p = f"{d}/{split}"
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    n_q += 1
                    sg = r["subgraph"]
                    for t in r.get("entities", ()):
                        gi = t if isinstance(t, int) else None
                        if gi is not None and 0 <= gi < len(ents) and MID_RE.match(ents[gi]):
                            topic_mids.add(ents[gi])
                    for a in r.get("answers", ()):
                        kb = a.get("kb_id") if isinstance(a, dict) else a
                        if isinstance(kb, str) and MID_RE.match(kb):
                            answer_mids.add(kb)
                    # tuples index GLOBALLY into entities.txt / relations.txt. sg["entities"]
                    # is the subgraph's node SET, not a local->global map; indexing through it
                    # both resolves the wrong entity and silently drops every tuple whose index
                    # exceeds the set size.
                    for s_i, rel_i, o_i in sg["tuples"]:
                        if s_i >= len(ents) or o_i >= len(ents):
                            continue
                        s, o = ents[s_i], ents[o_i]
                        if MID_RE.match(s) and MID_RE.match(o):
                            rn = rnames[rel_i] if rel_i < len(rnames) else None
                            if rn:
                                rel_used.add(rn)
                                tri.add((s, rn, o))
                            n_tri += 1
        per_ds[ds] = {"questions": n_q, "mid_to_mid_triple_slots": n_tri}
        print(f"[{ds}] {json.dumps(per_ds[ds])} distinct_tri={len(tri):,} t={time.time()-t0:.0f}s",
              flush=True)

    tri = sorted(tri)
    pq.write_table(pa.table({
        "subject_mid": pa.array([a for a, _, _ in tri], pa.string()),
        "predicate": pa.array([b for _, b, _ in tri], pa.string()),
        "object_mid": pa.array([c for _, _, c in tri], pa.string()),
    }), f"{OUTD}/nsm_mid_triples.parquet", compression="zstd")

    for name, s in (("nsm_all_mids", all_mids), ("nsm_topic_mids", topic_mids),
                    ("nsm_answer_mids", answer_mids)):
        pq.write_table(pa.table({"mid": pa.array(sorted(s), pa.string())}),
                       f"{OUTD}/{name}.parquet", compression="zstd")
    pq.write_table(pa.table({"predicate": pa.array(sorted(rel_used), pa.string())}),
                   f"{OUTD}/nsm_predicates.parquet", compression="zstd")

    doc = {
        "schema": "V3_PROBE_SETS/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "our side of checks 2, 3, 4 and 7, precomputed so the IDIR comparison is a "
                   "streaming membership test rather than a re-derivation.",
        "INSTRUMENT_NOTE": "NSM, not V1, is the MID-to-MID structural oracle: RoG substituted names "
                           "for MIDs so only 1,652,618 of 2,592,894 V1 endpoints kept an "
                           "identifier. NSM preserves MIDs on both ends of every triple.",
        "SERIALISATION_NOTE": "we hold m.0abc; IDIR and the raw dump write /m/0abc. The join must "
                              "normalise one to the other; identity is the same.",
        "v1_relation_keys": len(rkeys),
        "v1_mid_labelled_nodes": len(pairs),
        "v1_node_kind_counts_over_labelled_mids": dict(v1_kind_counts.most_common()),
        "nsm_distinct_mids": len(all_mids),
        "nsm_topic_mids": len(topic_mids),
        "nsm_answer_mids": len(answer_mids),
        "nsm_distinct_mid_to_mid_triples": len(tri),
        "nsm_distinct_predicates_used": len(rel_used),
        "per_dataset": per_ds,
        "files": sorted(os.listdir(OUTD)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
