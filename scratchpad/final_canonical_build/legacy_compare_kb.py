"""Phase 6 for the two 5-6M-node corpora -- LEGACY substrate vs canonical_v1, streaming.

    python scratchpad/final_canonical_build/legacy_compare_kb.py <ds>        ds in {2wiki, hotpotqa}

Same outputs and schema as legacy_compare.py (legacy_comparison.json + node_id_map_legacy.json), but the
in-memory index is built from the SMALL side (the legacy node table: 2wiki_clean 398k, hotpotqa_clean 507k)
and the 6M-node canonical table is STREAMED.  legacy_compare.py's dict-of-all-canonical-titles approach would
need ~1 GB here.

Identity used for both substrates: the article/paragraph TITLE, exactly as legacy_compare.py already does for
2wiki.  A legacy title that now matches several canonical curids is reported as AMBIGUOUS: the lexicographically
first canonical node_id is recorded in the id map so downstream code has a total function, and every candidate is
listed under "_ambiguous".  Nothing is silently dropped.
"""
import sys, os, json, hashlib, unicodedata, collections, time

ROOT = "data/final_canonical"; WORK = f"{ROOT}/_work"
LEGACY = {"2wiki": "2wiki_clean", "hotpotqa": "hotpotqa_clean"}


def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
def ws(s): return " ".join(s.split())
def nfc(s): return unicodedata.normalize("NFC", s)


def iter_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def text_diff_class(d, canon_text):
    if sha(canon_text) == d["sha_exact"]: return "exact_equal"
    if sha(ws(canon_text)) == d["sha_ws"]: return "whitespace_only_diff"
    if sha(ws(nfc(canon_text))) == d["sha_nfc_ws"]: return "unicode_nfc_only_diff"
    return "real_text_diff"


def main(ds):
    t0 = time.time(); L = LEGACY[ds]
    canon_path = f"{ROOT}/{ds}/nodes.jsonl"; legacy_path = f"{WORK}/legacy/{L}/docs.jsonl"

    # ---- legacy side (small): title -> list of legacy docs
    by_title = collections.defaultdict(list); n_leg = 0
    for dct in iter_jsonl(legacy_path):
        n_leg += 1
        by_title[dct.get("title", "")].append(dct)
    legacy_titles_multi = {t: len(v) for t, v in by_title.items() if len(v) > 1}

    # ---- canonical side (6M): streamed once
    n_canon = 0
    cand = collections.defaultdict(list)        # title -> [(canonical_node_id, text)] for titles legacy knows
    extra_by_prov = collections.Counter()
    for n in iter_jsonl(canon_path):
        n_canon += 1
        t = n["title"]
        if t in by_title:
            cand[t].append((n["node_id"], n["text"]))
        else:
            extra_by_prov["+".join(n["split_provenance"])] += 1
        if n_canon % 1_000_000 == 0:
            print(f"  [{ds}] {n_canon} canonical nodes scanned, {len(cand)} legacy titles hit "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- resolve
    idmap = {}; ambiguous = {}; missing = []
    match_class = collections.Counter(); text_class = collections.Counter()
    text_examples = collections.defaultdict(list); matched_canon = set()
    for t, ldocs in by_title.items():
        cs = sorted(cand.get(t, []))
        for dct in ldocs:
            lid = dct["node_id"]
            if not cs:
                idmap[lid] = None; match_class["no_match"] += 1
                missing.append({"legacy_id": lid, "title": t, "class": "no_match"}); continue
            cid, ctext = cs[0]
            if len(cs) > 1:
                ambiguous[lid] = [c for c, _ in cs]; match_class["title_ambiguous_first_taken"] += 1
            else:
                match_class["title"] += 1
            idmap[lid] = cid; matched_canon.add(cid)
            tc = text_diff_class(dct, ctext); text_class[tc] += 1
            if tc != "exact_equal" and len(text_examples[tc]) < 5:
                text_examples[tc].append({"legacy": lid, "canonical": cid, "title": t})
    # canonical nodes whose title IS in the legacy table but which no legacy node was mapped onto
    for t, cs in cand.items():
        for cid, _ in cs:
            if cid not in matched_canon:
                extra_by_prov["title_present_but_not_the_mapped_curid"] += 1

    # ---- legacy eval-gold consistency on the frozen subset
    ev = None
    evf = [f for f in os.listdir(f"{ROOT}/{ds}") if f.startswith("eval_") and f.endswith(".jsonl") and "random" not in f]
    if evf:
        n_eq = n_sub = n_diff = n_unmapped = 0; ex = []
        for r in iter_jsonl(f"{ROOT}/{ds}/{evf[0]}"):
            lg = r.get("legacy_gold_ids") or []
            mapped = [idmap.get(g) for g in lg]
            if any(m is None for m in mapped): n_unmapped += 1
            ms = set(m for m in mapped if m); cs_ = set(r["gold_node_ids"])
            if ms == cs_: n_eq += 1
            elif ms and ms <= cs_: n_sub += 1
            else:
                n_diff += 1
                if len(ex) < 5: ex.append({"query_id": r["query_id"], "legacy_gold": lg,
                                           "mapped": sorted(ms), "canonical_gold": sorted(cs_)})
        ev = {"file": evf[0], "n": n_eq + n_sub + n_diff, "legacy_golds_equal_canonical": n_eq,
              "legacy_golds_strict_subset_of_canonical": n_sub, "differ": n_diff,
              "queries_with_unmapped_legacy_gold": n_unmapped, "examples": ex}

    n_missing = len(missing)
    result = {"dataset": ds, "legacy_substrate": L, "legacy_master": f"data/processed/master_nodes_{L}.json",
              "legacy_count": n_leg, "canonical_count": n_canon,
              "overlap_legacy_nodes_mapped": n_leg - n_missing, "missing_in_canonical": n_missing,
              "missing_examples": missing[:20],
              "ambiguous_legacy_nodes": len(ambiguous), "ambiguous_examples": dict(list(ambiguous.items())[:10]),
              "extra_canonical_nodes_not_in_legacy": n_canon - len(matched_canon),
              "extra_by_split_provenance": dict(extra_by_prov),
              "canonical_nodes_matched_by_legacy": len(matched_canon),
              "legacy_titles_with_several_legacy_nodes": len(legacy_titles_multi),
              "match_classes": dict(match_class), "changed_text": dict(text_class),
              "changed_text_examples": dict(text_examples),
              "identity_rule": "article/paragraph TITLE, exact string (same rule legacy_compare.py uses for 2wiki)",
              "ambiguity_policy": "a legacy title matching several canonical curids maps to the lexicographically first "
                                  "canonical node_id; every candidate is listed under _ambiguous in node_id_map_legacy.json",
              "id_mapping_coverage": round((n_leg - n_missing) / max(1, n_leg), 6),
              "eval_subset_gold_consistency": ev, "seconds": round(time.time() - t0, 1)}
    json.dump(result, open(f"{ROOT}/{ds}/legacy_comparison.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump({"legacy_to_canonical": idmap, "_ambiguous": ambiguous,
               "_note": "legacy node_id (master_nodes_<legacy>.json doc node) -> canonical_v1 node_id; null = no canonical "
                        "match; _ambiguous lists every canonical candidate where one legacy title now matches several curids"},
              open(f"{ROOT}/{ds}/node_id_map_legacy.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("missing_examples", "ambiguous_examples", "changed_text_examples")}, ensure_ascii=True))


if __name__ == "__main__":
    main(sys.argv[1])
