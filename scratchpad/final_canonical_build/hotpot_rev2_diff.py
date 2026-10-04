"""SECTION 7 diff audit: prove the REV2 rebuild changed ONLY the intended text fields.

Streams the old and new nodes.jsonl in lockstep (both are strictly node_id-ordered) and asserts
membership/identity/order are byte-identical, reporting exactly which nodes and which fields moved.
"""
import os, sys, json, hashlib, collections

ROOT = r"C:\Users\Swastik\Desktop\CRAG"
OLD = os.path.join(ROOT, "data/final_canonical/hotpotqa/nodes.jsonl")
NEW = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "data/final_canonical/_work/hotpot_rev2/nodes.jsonl")
OUT = os.path.join(ROOT, "results/data_audit/final_canonical_v1/hotpotqa/REV2_DIFF_AUDIT.json")


def main():
    n_old = n_new = 0
    order_ok = True
    first_order_mismatch = None
    changed = []
    field_delta = collections.Counter()
    h_old_ids = hashlib.sha256()
    h_new_ids = hashlib.sha256()

    with open(OLD, encoding="utf-8") as fo, open(NEW, encoding="utf-8") as fn:
        for lo, ln in zip(fo, fn):
            a = json.loads(lo); b = json.loads(ln)
            pre = b"" if n_old == 0 else b"\n"
            h_old_ids.update(pre + a["node_id"].encode("utf-8"))
            h_new_ids.update(pre + b["node_id"].encode("utf-8"))
            n_old += 1; n_new += 1
            if a["node_id"] != b["node_id"]:
                order_ok = False
                if first_order_mismatch is None:
                    first_order_mismatch = {"row": n_old, "old": a["node_id"], "new": b["node_id"]}
                continue
            if lo == ln:
                continue
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            for k in diff:
                field_delta[k] += 1
            if len(changed) < 200:
                changed.append({"node_id": a["node_id"], "title": b.get("title"),
                                "fields_changed": diff,
                                "old_text": a.get("text"), "new_text": b.get("text"),
                                "old_text_len": len(a.get("text") or ""), "new_text_len": len(b.get("text") or ""),
                                "n_sentences": b.get("n_sentences"),
                                "old_content_hash": a.get("content_hash"), "new_content_hash": b.get("content_hash")})
            else:
                changed.append({"node_id": a["node_id"]})
        # drain: if either file is longer the counts diverge
        for _ in fo: n_old += 1
        for _ in fn: n_new += 1

    only_text_and_hash = set(field_delta) <= {"text", "content_hash"}
    res = {
        "OLD": OLD, "NEW": NEW,
        "N_BEFORE": n_old, "N_AFTER": n_new, "N_UNCHANGED": n_old == n_new == 5233329,
        "NODE_ORDER_IDENTICAL": order_ok,
        "first_order_mismatch": first_order_mismatch,
        "ORDERED_NODE_ID_HASH_old": h_old_ids.hexdigest(),
        "ORDERED_NODE_ID_HASH_new": h_new_ids.hexdigest(),
        "ORDERED_NODE_ID_HASH_EQUAL": h_old_ids.hexdigest() == h_new_ids.hexdigest(),
        "CHANGED_NODE_COUNT": len(changed),
        "fields_that_changed": dict(field_delta),
        "ONLY_TEXT_AND_CONTENT_HASH_CHANGED": only_text_and_hash,
        "CHANGED_NODE_IDS": [c["node_id"] for c in changed],
        "changed_detail_first_200": [c for c in changed if "fields_changed" in c],
        "all_old_texts_were_empty": all((c.get("old_text") or "") == "" for c in changed if "old_text" in c),
        "all_new_texts_equal_title": all(c.get("new_text") == c.get("title") for c in changed if "new_text" in c),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    for k in ("N_BEFORE", "N_AFTER", "N_UNCHANGED", "NODE_ORDER_IDENTICAL", "ORDERED_NODE_ID_HASH_EQUAL",
              "CHANGED_NODE_COUNT", "fields_that_changed", "ONLY_TEXT_AND_CONTENT_HASH_CHANGED",
              "all_old_texts_were_empty", "all_new_texts_equal_title"):
        print("%-38s %s" % (k, res[k]))
    print("\nfirst 5 changed:")
    for c in res["changed_detail_first_200"][:5]:
        print("   %s  n_sentences=%s  %r -> %r" % (c["node_id"], c["n_sentences"], c["old_text"], c["new_text"]))


if __name__ == "__main__":
    main()
