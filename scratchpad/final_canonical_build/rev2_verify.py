"""Independent verification of an INSTALLED TEXTUALIZATION_REV 2 node table.

    python scratchpad/final_canonical_build/rev2_verify.py <ds>

Recomputes CORPUS_HASH, NODE_ORDER_HASH, every content_hash and the empty/non-empty census straight from
data/final_canonical/<ds>/nodes.jsonl, WITHOUT trusting any counter the builder wrote.  Also re-derives the
strict-ordering invariant and re-checks every node the diff audit flagged as changed.

Dataset-general version of hotpot_rev2_verify.py (2026-09-05, hotpotqa-only, paths hardcoded, changed set read
from the JSON's inline id list).  The changed set is read from the FULL id file the diff audit writes, falling
back to the inline list for the hotpot record written before that file existed.  Both datasets share the
node_id shape "<ds>:c<curid>" and the content_hash formula sha256(curid + US + text), so one implementation
covers them; the split is asserted rather than assumed.
"""
import os, sys, json, hashlib, time, argparse

ROOT = "data/final_canonical"
AUDR = "results/data_audit/final_canonical_v1"
SEP = "\x1f"


def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()


def changed_set(ds, aud):
    f = f"{ROOT}/{ds}/REV2_CHANGED_NODE_IDS.txt"
    if os.path.exists(f):
        return {l.strip() for l in open(f, encoding="utf-8") if l.strip()}, f
    j = json.load(open(f"{aud}/REV2_DIFF_AUDIT.json", encoding="utf-8"))
    for k in j:
        if k.startswith("CHANGED_NODE_IDS") and isinstance(j[k], list):
            return set(j[k]), f"{aud}/REV2_DIFF_AUDIT.json:{k}"
    raise SystemExit("no changed-id list found")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ds"); ds = ap.parse_args().ds
    D, AUD = f"{ROOT}/{ds}", f"{AUDR}/{ds}"
    t0 = time.time()
    changed, changed_src = changed_set(ds, AUD)
    h_ids, h_corpus = hashlib.sha256(), hashlib.sha256()
    n = 0
    empty_text = ws_only_text = nonempty_text = empty_title = text_eq_title = 0
    content_hash_mismatch = []
    order_ok = True
    prev = None
    changed_seen = 0
    changed_bad = []
    n_sent_zero_among_changed = 0
    prefix = f"{ds}:c"

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
            assert nid.startswith(prefix), nid
            cid = nid[len(prefix):]
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
        "dataset": ds,
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
        "changed_nodes": {
            "source_of_id_list": changed_src,
            "expected": len(changed), "seen_in_table": changed_seen,
            "all_present": changed_seen == len(changed),
            "text_equals_title_and_title_nonempty": len(changed) - len(changed_bad),
            "violations": changed_bad[:50], "n_violations": len(changed_bad),
            "n_sentences_zero": n_sent_zero_among_changed,
            "n_sentences_zero_note": "n_sentences is left at 0 for the fallback rows -- it counts SOURCE sentences, "
                                     "so it stays a faithful provenance marker of the empty source body",
        },
        "seconds": round(time.time() - t0, 1),
    }
    # Upstream copy as well as the tracked mirror -- see the same note in rev2_diff.py: mirror_audit.py exiles a
    # mirrored file that has no original under data/final_canonical/.
    for _d in (D, AUD):
        os.makedirs(_d, exist_ok=True)
        json.dump(res, open(f"{_d}/REV2_VERIFY.json", "w", encoding="utf-8"), indent=2)
    for k, v in res.items():
        if k != "changed_nodes":
            print("%-42s %s" % (k, v))
    print("changed_nodes:")
    for k, v in res["changed_nodes"].items():
        print("   %-40s %s" % (k, v))


if __name__ == "__main__":
    main()
