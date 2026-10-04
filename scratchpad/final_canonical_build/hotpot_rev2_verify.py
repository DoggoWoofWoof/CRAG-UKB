"""SECTION 11 -- independent verification of the installed HotpotQA rev2 node table.

Recomputes CORPUS_HASH, NODE_ORDER_HASH, every content_hash and the empty/non-empty census straight from
data/final_canonical/hotpotqa/nodes.jsonl, WITHOUT trusting any counter the builder wrote.  Also re-derives
the strict-ordering invariant and checks the 94 audited nodes individually.

    python scratchpad/final_canonical_build/hotpot_rev2_verify.py
"""
import os, sys, json, hashlib, time

ROOT = r"C:\Users\Swastik\Desktop\CRAG"
D = os.path.join(ROOT, "data/final_canonical/hotpotqa")
AUD = os.path.join(ROOT, "results/data_audit/final_canonical_v1/hotpotqa")
SEP = "\x1f"


def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    t0 = time.time()
    changed = set(json.load(open(f"{AUD}/REV2_DIFF_AUDIT.json", encoding="utf-8"))["CHANGED_NODE_IDS"])
    h_ids, h_corpus = hashlib.sha256(), hashlib.sha256()
    n = 0
    empty_text = ws_only_text = nonempty_text = empty_title = 0
    text_eq_title = 0
    content_hash_mismatch = []
    order_ok = True
    prev = None
    changed_seen = 0
    changed_bad = []
    n_sent_zero_among_changed = 0

    with open(f"{D}/nodes.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            nid = r["node_id"]
            if prev is not None and not nid > prev:
                order_ok = False
            prev = nid
            pre = b"" if n == 0 else b"\n"
            h_ids.update(pre + nid.encode("utf-8"))
            h_corpus.update(pre + f"{nid}\t{r['content_hash']}".encode("utf-8"))
            n += 1

            txt = r["text"]; ttl = r["title"]
            if txt == "":
                empty_text += 1
            elif not txt.strip():
                ws_only_text += 1
            else:
                nonempty_text += 1
            if not str(ttl).strip():
                empty_title += 1
            if txt == ttl:
                text_eq_title += 1

            # content_hash must be sha(curid + US + text) -- recomputed, not read
            cid = nid.split(":c", 1)[1]
            if sha(cid + SEP + txt) != r["content_hash"] and len(content_hash_mismatch) < 20:
                content_hash_mismatch.append(nid)

            if nid in changed:
                changed_seen += 1
                if r.get("n_sentences") == 0:
                    n_sent_zero_among_changed += 1
                if txt != ttl or not ttl.strip():
                    changed_bad.append(nid)

            if n % 1000000 == 0:
                print(f"   {n} nodes ({time.time()-t0:.0f}s)", flush=True)

    bi = json.load(open(f"{D}/build_info.json", encoding="utf-8"))
    ir = json.load(open(f"{D}/integrity_report.json", encoding="utf-8"))
    res = {
        "TOTAL_N": n,
        "EMPTY_TEXT_N": empty_text,
        "WHITESPACE_ONLY_TEXT_N": ws_only_text,
        "NONEMPTY_TEXT_N": nonempty_text,
        "EMPTY_TITLE_N": empty_title,
        "TEXT_EQUALS_TITLE_N": text_eq_title,
        "SEMANTIC_KNN_ELIGIBLE": nonempty_text,
        "SEMANTIC_KNN_ELIGIBLE_rule": "a node is eligible iff its canonical text is non-empty after strip(), i.e. it "
                                      "yields a non-degenerate token sequence under the frozen dense tokenizer",
        "accounting_identity_holds": empty_text + ws_only_text + nonempty_text == n,
        "STRICTLY_ORDERED": order_ok,
        "recomputed_NODE_ORDER_HASH": h_ids.hexdigest(),
        "recomputed_CORPUS_HASH": h_corpus.hexdigest(),
        "stored_NODE_ORDER_HASH": bi["NODE_ORDER_HASH"],
        "stored_CORPUS_HASH": bi["CORPUS_HASH"],
        "NODE_ORDER_HASH_MATCHES": h_ids.hexdigest() == bi["NODE_ORDER_HASH"],
        "CORPUS_HASH_MATCHES": h_corpus.hexdigest() == bi["CORPUS_HASH"],
        "content_hash_recomputed_for_every_node": True,
        "content_hash_mismatches": content_hash_mismatch,
        "CONTENT_HASH_ALL_CORRECT": not content_hash_mismatch,
        "integrity_report_empty_text_nodes": ir["empty_text_nodes"],
        "integrity_report_matches_measured": ir["empty_text_nodes"] == empty_text + ws_only_text,
        "changed_94": {
            "expected": len(changed), "seen_in_table": changed_seen,
            "all_present": changed_seen == len(changed),
            "text_equals_title_and_title_nonempty": len(changed) - len(changed_bad),
            "violations": changed_bad,
            "n_sentences_zero": n_sent_zero_among_changed,
            "n_sentences_zero_note": "n_sentences is left at 0 for the fallback rows -- it counts SOURCE abstract "
                                     "sentences, so it stays a faithful provenance marker of the empty abstract",
        },
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(AUD, exist_ok=True)
    json.dump(res, open(f"{AUD}/REV2_VERIFY.json", "w", encoding="utf-8"), indent=2)
    for k, v in res.items():
        if k != "changed_94":
            print("%-42s %s" % (k, v))
    print("changed_94:")
    for k, v in res["changed_94"].items():
        print("   %-40s %s" % (k, v))


if __name__ == "__main__":
    main()
