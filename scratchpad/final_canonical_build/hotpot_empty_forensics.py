"""SECTION 1-4: derive the empty-canonical-text HotpotQA records FROM SOURCE and dump every
source-native field they carry.  Read-only: touches no canonical file.

Replicates build_kb.py:432-447 exactly (_flat_sentences + ' '.join, NO .strip()) so the set it
finds is the set the builder found -- not a manually supplied list.
"""
import os, sys, json, bz2, tarfile, time, collections, hashlib

ROOT = r"C:\Users\Swastik\Desktop\CRAG"
TB = os.path.join(ROOT, "data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2")
OUT = os.path.join(ROOT, "results/data_audit/final_canonical_v1/hotpotqa")
os.makedirs(OUT, exist_ok=True)


def flat_sentences(t):                      # build_kb.py:394-400, verbatim
    if isinstance(t, str):
        if t:
            yield t
    elif isinstance(t, list):
        for x in t:
            yield from flat_sentences(x)


def main():
    hits = []
    raw = 0
    nm = 0
    key_inventory = collections.Counter()
    title_seen = collections.Counter()       # only for the hit titles, to detect dup titles later
    t0 = time.time()

    tar = tarfile.open(TB, "r:bz2")
    for member in tar:
        if not member.isfile():
            continue
        nm += 1
        mname = member.name.split("/", 1)[-1] if "/" in member.name else member.name
        try:
            blob = bz2.decompress(tar.extractfile(member).read()).decode("utf-8")
        except Exception:
            continue
        for li, line in enumerate(blob.splitlines()):
            if not line.strip():
                continue
            raw += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            cid = r.get("id")
            if cid is None or str(cid) == "":
                continue
            cid = str(cid)
            sents = list(flat_sentences(r.get("text")))
            body = " ".join(sents)
            if body.strip():                                   # non-empty canonical text -> not our case
                continue

            # ---- this record's canonical text is empty (or whitespace-only) ----
            for k in r.keys():
                key_inventory[k] += 1
            twl = list(flat_sentences(r.get("text_with_links")))
            twl_body = " ".join(twl)
            title = r.get("title", "")
            title_seen[title] += 1
            hits.append({
                "canonical_node_id": "hotpotqa:c%s" % cid,
                "source_identity_curid": cid,
                "source_record_id": "%s:%d" % (mname, li),
                "canonical_title": title,
                "canonical_text_current": body,
                "canonical_text_current_repr": repr(body),
                "canonical_text_current_len": len(body),
                "canonical_text_is_exactly_empty_string": body == "",
                "canonical_text_is_whitespace_only": (body != "" and body.strip() == ""),
                "raw_text_field": r.get("text"),
                "raw_text_field_type": type(r.get("text")).__name__,
                "raw_text_n_leaf_sentences": len(sents),
                "raw_text_with_links_field_type": type(r.get("text_with_links")).__name__,
                "raw_text_with_links_n_leaf": len(twl),
                "raw_text_with_links_joined": twl_body,
                "raw_text_with_links_nonempty": bool(twl_body.strip()),
                "phase_c_encoder_input": (twl_body if twl_body else body),   # c1_hotpot_fullwiki.py:38-39 then .strip()
                "phase_c_encoder_input_stripped": (twl_body if twl_body else body).strip(),
                "url": r.get("url"),
                "all_source_keys": sorted(r.keys()),
                "other_source_text_fields": {k: v for k, v in r.items()
                                             if k not in ("id", "title", "text", "text_with_links", "url")
                                             and isinstance(v, (str, list))},
            })
        if nm % 4000 == 0:
            print("  %d members, %d records, %d hits, %.0fs" % (nm, raw, len(hits), time.time() - t0), flush=True)
    tar.close()

    print("DONE scan: %d members, %d raw records, %d empty-canonical-text records, %.0fs"
          % (nm, raw, len(hits), time.time() - t0), flush=True)

    hits.sort(key=lambda h: h["canonical_node_id"])
    with open(os.path.join(OUT, "EMPTY_TEXT_94_AUDIT.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False, sort_keys=True) + "\n")

    # ---------- Section 2 counts ----------
    c = collections.Counter()
    combos = collections.Counter()
    for h in hits:
        ab = bool(h["raw_text_n_leaf_sentences"])
        twl = h["raw_text_with_links_nonempty"]
        ti = bool((h["canonical_title"] or "").strip())
        other = any(str(v).strip() for v in h["other_source_text_fields"].values())
        c["abstract_nonempty" if ab else "abstract_empty"] += 1
        c["text_with_links_nonempty" if twl else "text_with_links_empty"] += 1
        c["title_nonempty" if ti else "title_empty"] += 1
        c["other_source_text_nonempty" if other else "other_source_text_empty"] += 1
        if not ab and not twl and not ti and not other:
            c["truly_no_readable_source_text"] += 1
        combos["abstract_%s + links_%s + title_%s + other_%s"
               % ("nonempty" if ab else "empty", "nonempty" if twl else "empty",
                  "present" if ti else "absent", "present" if other else "absent")] += 1
        c["exact_empty_string" if h["canonical_text_is_exactly_empty_string"] else "whitespace_only"] += 1

    summary = {
        "identified_empty_text_nodes": len(hits),
        "scan": {"tar_members_read": nm, "raw_source_records": raw, "seconds": round(time.time() - t0, 1),
                 "tarball": TB, "rule": "build_kb.py:432-439 -- body=' '.join(_flat_sentences(text)); empty iff not body.strip()"},
        "counts": dict(c),
        "distinct_combinations": dict(combos),
        "source_key_inventory_over_hits": dict(key_inventory),
        "titles": {"unique_titles": len(title_seen),
                   "empty_titles": sum(v for k, v in title_seen.items() if not (k or "").strip()),
                   "duplicate_title_groups": {k: v for k, v in title_seen.items() if v > 1}},
        "duplicate_curids": {k: v for k, v in collections.Counter(h["source_identity_curid"] for h in hits).items() if v > 1},
    }
    with open(os.path.join(OUT, "EMPTY_TEXT_94_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
