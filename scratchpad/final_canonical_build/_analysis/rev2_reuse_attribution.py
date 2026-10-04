"""Attribute a TEXTUALIZATION_REV 2 reuse recount to the rows the revision actually changed.

    python scratchpad/final_canonical_build/_analysis/rev2_reuse_attribution.py <ds>

reuse_map/index.json reports corpus-wide totals.  The question the gate asks is narrower and cannot be read off
those totals: *of the rows this revision rewrote, how many lost reuse, and how many distinct encoder forward
passes do the survivors actually require?*  Answering it needs the per-node map, so this streams
reuse_map/shard_*.jsonl once and partitions every row by whether its node_id is in REV2_CHANGED_NODE_IDS.txt.

DISTINCT FORWARD PASSES is the number that matters for cost and is NOT the row count: two rows whose frozen
token-ID sequences are equal are one forward pass.  It is computed here from the map's own
dense_input_hash / splade_input_hash, i.e. from token IDs, never from raw text.

No encoder, no tokenizer, no model is loaded.  Read-only over reuse_map/.
"""
import os, sys, json, glob, collections, time

ROOT = "data/final_canonical"


def main(ds):
    t0 = time.time()
    d = f"{ROOT}/{ds}"
    changed = {l.strip() for l in open(f"{d}/REV2_CHANGED_NODE_IDS.txt", encoding="utf-8") if l.strip()}
    shards = sorted(glob.glob(f"{d}/reuse_map/shard_*.jsonl"))
    assert shards, f"no reuse map shards under {d}/reuse_map/"

    seen = set()
    ch = {"n": 0, "dense_reusable": 0, "splade_reusable": 0,
          "dense_needs_encode": 0, "splade_needs_encode": 0,
          "status": collections.Counter(), "dense_reuse_source": collections.Counter(),
          "splade_reuse_source": collections.Counter()}
    un = {"n": 0, "dense_reusable": 0, "splade_reusable": 0,
          "dense_needs_encode": 0, "splade_needs_encode": 0}
    legacy_only = 0
    # distinct token-ID digests among the rows that still need an encode, split by side
    dense_todo_dig, splade_todo_dig = set(), set()
    ch_dense_dig, ch_splade_dig = set(), set()

    for p in shards:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                nid = r["canonical_node_id"]
                if nid is None:
                    legacy_only += 1
                    continue
                tgt = ch if nid in changed else un
                if tgt is ch:
                    seen.add(nid)
                    ch["status"][r["status"]] += 1
                    ch["dense_reuse_source"][str(r["dense_reuse_source"]).split(":")[0]] += 1
                    ch["splade_reuse_source"][str(r["splade_reuse_source"]).split(":")[0]] += 1
                    ch_dense_dig.add(r["dense_input_hash"]); ch_splade_dig.add(r["splade_input_hash"])
                tgt["n"] += 1
                if r["dense_reusable"]:
                    tgt["dense_reusable"] += 1
                else:
                    tgt["dense_needs_encode"] += 1
                    dense_todo_dig.add(r["dense_input_hash"])
                if r["splade_reusable"]:
                    tgt["splade_reusable"] += 1
                else:
                    tgt["splade_needs_encode"] += 1
                    splade_todo_dig.add(r["splade_input_hash"])

    idx = json.load(open(f"{d}/reuse_map/index.json", encoding="utf-8"))
    c = idx["counts"]
    out = {
        "dataset": ds,
        "reuse_map_index_counts": c,
        "n_changed_ids_expected": len(changed),
        "n_changed_ids_found_in_map": len(seen),
        "ALL_CHANGED_NODES_PRESENT_IN_MAP": len(seen) == len(changed),
        "CHANGED_by_TEXTUALIZATION_REV_2": {k: (dict(v) if isinstance(v, collections.Counter) else v)
                                            for k, v in ch.items()},
        "UNCHANGED_rows": un,
        "LEGACY_ONLY_rows": legacy_only,
        "corpus_totals": {
            "DENSE_ROWS_REUSED": c.get("dense_reusable", 0),
            "DENSE_ROWS_NEED_ENCODE": c.get("dense_needs_encode", 0),
            "SPLADE_ROWS_REUSED": c.get("splade_reusable", 0),
            "SPLADE_ROWS_NEED_ENCODE": c.get("splade_needs_encode", 0),
        },
        "DISTINCT_FORWARD_PASSES": {
            "dense": len(dense_todo_dig), "splade": len(splade_todo_dig),
            "rule": "distinct frozen-tokenizer token-ID digests among the rows that need an encode; two rows "
                    "with equal token IDs are ONE forward pass. Taken from the map's dense_input_hash / "
                    "splade_input_hash, i.e. from token IDs, never from raw text.",
        },
        "DISTINCT_TOKEN_ID_DIGESTS_over_all_changed_rows": {"dense": len(ch_dense_dig), "splade": len(ch_splade_dig)},
        "ATTRIBUTION_CHECK": {
            "unchanged_rows_needing_dense_encode": un["dense_needs_encode"],
            "unchanged_rows_needing_splade_encode": un["splade_needs_encode"],
            "ALL_ENCODE_DEMAND_COMES_FROM_THE_CHANGED_ROWS":
                un["dense_needs_encode"] == 0 and un["splade_needs_encode"] == 0,
        },
        "encoder_run": False,
        "seconds": round(time.time() - t0, 1),
    }
    # Upstream copy + tracked mirror: mirror_audit.py exiles a mirrored file with no original under
    # data/final_canonical/, so a results-only deliverable would be swept into _superseded_mirror/.
    for op in (f"{ROOT}/{ds}/REV2_REUSE_ATTRIBUTION.json",
               f"results/data_audit/final_canonical_v1/{ds}/REV2_REUSE_ATTRIBUTION.json"):
        os.makedirs(os.path.dirname(op), exist_ok=True)
        json.dump(out, open(op, "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: out[k] for k in ("n_changed_ids_expected", "n_changed_ids_found_in_map",
                                          "ALL_CHANGED_NODES_PRESENT_IN_MAP", "corpus_totals",
                                          "CHANGED_by_TEXTUALIZATION_REV_2", "UNCHANGED_rows",
                                          "DISTINCT_FORWARD_PASSES", "ATTRIBUTION_CHECK", "seconds")}, indent=1))
    print("wrote", op)


if __name__ == "__main__":
    main(sys.argv[1])
