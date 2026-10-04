"""TEXTUALIZATION_REV 2 diff audit: prove a rev2 rebuild changed ONLY the intended text fields.

    python scratchpad/final_canonical_build/rev2_diff.py <ds> [--new PATH_TO_NEW_nodes.jsonl]

Streams the installed (old) and staged (new) nodes.jsonl in lockstep -- both are strictly node_id-ordered --
and asserts membership / identity / order are byte-identical, reporting exactly which nodes and which fields
moved.  Dataset-general: this is hotpot_rev2_diff.py (2026-09-05, hotpotqa-only, paths and N hardcoded)
rewritten to take <ds> and to read the expected node count from the OLD build_info.json instead of a literal.
The hotpot-specific original is KEPT as the historical record of that run.

The changed-id list is the one output that can be large (hotpotqa 94, 2wiki 87,765).  results/ is tracked, so
the JSON keeps only the first 200 ids and the FULL list is written to
data/final_canonical/<ds>/REV2_CHANGED_NODE_IDS.txt (gitignored, same rule mirror_audit.py uses for bulky tables).
"""
import os, sys, json, hashlib, collections, argparse, time

ROOT = "data/final_canonical"
AUDR = "results/data_audit/final_canonical_v1"
ID_CAP = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--new", default=None, help="new nodes.jsonl (default data/final_canonical/_stage_<ds>_rev2/nodes.jsonl)")
    a = ap.parse_args()
    ds = a.ds
    old = f"{ROOT}/{ds}/nodes.jsonl"
    new = a.new or f"{ROOT}/_stage_{ds}_rev2/nodes.jsonl"
    out = f"{AUDR}/{ds}/REV2_DIFF_AUDIT.json"
    ids_out = f"{ROOT}/{ds}/REV2_CHANGED_NODE_IDS.txt"
    expected_n = json.load(open(f"{ROOT}/{ds}/build_info.json", encoding="utf-8"))["canonical_node_count"]

    t0 = time.time()
    n_old = n_new = 0
    order_ok = True
    first_order_mismatch = None
    changed_ids = []                 # every changed node id, in file order
    changed_detail = []              # rich record for the first ID_CAP of them
    field_delta = collections.Counter()
    old_empty = new_eq_title = new_nonempty = 0
    h_old_ids, h_new_ids = hashlib.sha256(), hashlib.sha256()

    with open(old, encoding="utf-8") as fo, open(new, encoding="utf-8") as fn:
        for lo, ln in zip(fo, fn):
            a_ = json.loads(lo); b_ = json.loads(ln)
            pre = b"" if n_old == 0 else b"\n"
            h_old_ids.update(pre + a_["node_id"].encode("utf-8"))
            h_new_ids.update(pre + b_["node_id"].encode("utf-8"))
            n_old += 1; n_new += 1
            if a_["node_id"] != b_["node_id"]:
                order_ok = False
                if first_order_mismatch is None:
                    first_order_mismatch = {"row": n_old, "old": a_["node_id"], "new": b_["node_id"]}
                continue
            if lo == ln:
                continue
            diff = sorted(k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k))
            for k in diff:
                field_delta[k] += 1
            changed_ids.append(a_["node_id"])
            if (a_.get("text") or "") == "":
                old_empty += 1
            if b_.get("text") == b_.get("title"):
                new_eq_title += 1
            if (b_.get("text") or "").strip():
                new_nonempty += 1
            if len(changed_detail) < ID_CAP:
                changed_detail.append({"node_id": a_["node_id"], "title": b_.get("title"),
                                       "fields_changed": diff,
                                       "old_text": a_.get("text"), "new_text": b_.get("text"),
                                       "old_text_len": len(a_.get("text") or ""), "new_text_len": len(b_.get("text") or ""),
                                       "n_sentences": b_.get("n_sentences"),
                                       "old_content_hash": a_.get("content_hash"), "new_content_hash": b_.get("content_hash")})
            if n_old % 1_000_000 == 0:
                print(f"   {n_old} rows, {len(changed_ids)} changed ({time.time()-t0:.0f}s)", flush=True)
        for _ in fo: n_old += 1          # drain: if either file is longer the counts diverge
        for _ in fn: n_new += 1

    os.makedirs(os.path.dirname(ids_out), exist_ok=True)
    with open(ids_out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(i + "\n" for i in changed_ids))

    res = {
        "dataset": ds, "OLD": old, "NEW": new,
        "N_BEFORE": n_old, "N_AFTER": n_new, "N_EXPECTED": expected_n,
        "N_UNCHANGED": n_old == n_new == expected_n,
        "NODE_ORDER_IDENTICAL": order_ok,
        "first_order_mismatch": first_order_mismatch,
        "ORDERED_NODE_ID_HASH_old": h_old_ids.hexdigest(),
        "ORDERED_NODE_ID_HASH_new": h_new_ids.hexdigest(),
        "ORDERED_NODE_ID_HASH_EQUAL": h_old_ids.hexdigest() == h_new_ids.hexdigest(),
        "NODE_ID_SET_IDENTICAL": h_old_ids.hexdigest() == h_new_ids.hexdigest() and n_old == n_new,
        "NODE_ID_SET_IDENTICAL_why": "the files are compared row-by-row in their common strict node_id order, so an "
                                     "identical ORDERED id hash at an identical row count implies identical SETS -- "
                                     "no addition, removal or reorder is representable without breaking it",
        "CHANGED_NODE_COUNT": len(changed_ids),
        "fields_that_changed": dict(field_delta),
        "ONLY_TEXT_AND_CONTENT_HASH_CHANGED": set(field_delta) <= {"text", "content_hash"},
        "CHANGED_NODE_IDS_first_%d" % ID_CAP: changed_ids[:ID_CAP],
        "CHANGED_NODE_IDS_full_file": ids_out,
        "changed_detail_first_%d" % ID_CAP: changed_detail,
        "all_old_texts_were_empty": old_empty == len(changed_ids),
        "all_new_texts_equal_title": new_eq_title == len(changed_ids),
        "all_new_texts_nonempty": new_nonempty == len(changed_ids),
        "seconds": round(time.time() - t0, 1),
    }
    # Write the UPSTREAM copy under data/final_canonical/<ds>/ as well as the tracked mirror.  mirror_audit.py
    # treats data/final_canonical as the source of truth and exiles any mirrored file with no upstream original
    # into _superseded_mirror/, so a results-only deliverable would be swept away on the next mirror run.
    for out in (f"{ROOT}/{ds}/REV2_DIFF_AUDIT.json", f"{AUDR}/{ds}/REV2_DIFF_AUDIT.json"):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    for k in ("N_BEFORE", "N_AFTER", "N_UNCHANGED", "NODE_ORDER_IDENTICAL", "ORDERED_NODE_ID_HASH_EQUAL",
              "NODE_ID_SET_IDENTICAL", "CHANGED_NODE_COUNT", "fields_that_changed",
              "ONLY_TEXT_AND_CONTENT_HASH_CHANGED", "all_old_texts_were_empty", "all_new_texts_equal_title",
              "all_new_texts_nonempty", "seconds"):
        print("%-38s %s" % (k, res[k]))
    print("\nfirst 5 changed:")
    for c in changed_detail[:5]:
        print("   %s  n_sentences=%s  %r -> %r" % (c["node_id"], c["n_sentences"], c["old_text"], c["new_text"]))


if __name__ == "__main__":
    main()
