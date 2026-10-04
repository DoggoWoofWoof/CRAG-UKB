"""Phase 6 -- LEGACY substrate vs canonical_v1 comparison.

    python scratchpad/final_canonical_build/legacy_compare.py <ds>

Legacy = data/processed/master_nodes_<legacy>.json doc nodes (the exact node table the frozen G2 cache indexes;
dumped by legacy_dump.py to data/final_canonical/_work/legacy/<legacy>/docs.jsonl).
Canonical = data/final_canonical/<ds>/nodes.jsonl.
Writes data/final_canonical/<ds>/legacy_comparison.json and node_id_map_legacy.json.
"""
import sys, os, json, hashlib, unicodedata, collections, time

ROOT = "data/final_canonical"; WORK = f"{ROOT}/_work"
LEGACY = {"metaqa": "metaqa", "2wiki": "2wiki_clean", "musique": "musique_clean", "squad": "squad_clean"}
SEP = "\x1f"


def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
def ws(s): return " ".join(s.split())
def nfc(s): return unicodedata.normalize("NFC", s)


def iter_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def text_diff_class(legacy_sha_exact, legacy_sha_ws, legacy_sha_nfc_ws, canon_text):
    if sha(canon_text) == legacy_sha_exact: return "exact_equal"
    if sha(ws(canon_text)) == legacy_sha_ws: return "whitespace_only_diff"
    if sha(ws(nfc(canon_text))) == legacy_sha_nfc_ws: return "unicode_nfc_only_diff"
    return "real_text_diff"


def main(ds):
    t0 = time.time(); L = LEGACY[ds]
    canon_path = f"{ROOT}/{ds}/nodes.jsonl"; legacy_path = f"{WORK}/legacy/{L}/docs.jsonl"
    # ---- canonical identity maps
    n_canon = 0; by_title = {}; by_key = {}; by_text = collections.defaultdict(list); by_name = {}; by_lower = collections.defaultdict(list)
    prov = {}
    for n in iter_jsonl(canon_path):
        n_canon += 1; prov[n["node_id"]] = tuple(n["split_provenance"])
        if ds == "metaqa":
            by_name[n["source_id"]] = n["node_id"]; by_lower[n["source_id"].strip().lower().replace("_", " ")].append(n["node_id"])
        elif ds == "2wiki":
            by_title[n["title"]] = (n["node_id"], n["text"])
        else:
            by_key[n["title"] + SEP + n["text"]] = n["node_id"]; by_text[sha(n["text"])].append((n["title"], n["node_id"]))
    # ---- legacy pass
    n_leg = 0; idmap = {}; ambiguous = {}; missing = []; match_class = collections.Counter(); text_class = collections.Counter()
    text_examples = collections.defaultdict(list); matched_canon = set(); title_diff = 0; title_diff_examples = []
    for d in iter_jsonl(legacy_path):
        n_leg += 1; lid = d["node_id"]; cid = None; cls = None
        if ds == "metaqa":
            cid = by_name.get(d["title"])
            if cid: cls = "exact_name"
            else:
                key = lid[len("metaqa_ent_"):]          # legacy id = metaqa_ent_<lowercase name>
                grp = by_lower.get(key) or by_lower.get(d["title"].strip().lower().replace("_", " ")) or []
                if len(grp) == 1: cid = grp[0]; cls = "case_variant_single"
                elif len(grp) > 1: ambiguous[lid] = grp; cls = "case_variant_ambiguous(legacy merged several canonical entities)"
                else: cls = "no_match"
        elif ds == "2wiki":
            hit = by_title.get(d["title"])
            if hit:
                cid, ctext = hit; cls = "title"
                tc = text_diff_class(d["sha_exact"], d["sha_ws"], d["sha_nfc_ws"], ctext); text_class[tc] += 1
                if tc != "exact_equal" and len(text_examples[tc]) < 5: text_examples[tc].append({"legacy": lid, "canonical": cid, "title": d["title"]})
            else: cls = "no_match"
        else:
            cid = by_key.get(d.get("title", "") + SEP + d["content"])
            if cid: cls = "title+text"
            else:
                cands = by_text.get(d["sha_exact"], [])
                if len(cands) == 1:
                    cid = cands[0][1]; cls = "text_only(title differs or legacy title empty)"
                    if d.get("title", "") != cands[0][0]:
                        title_diff += 1
                        if len(title_diff_examples) < 5: title_diff_examples.append({"legacy": lid, "legacy_title": d.get("title", ""), "canonical_title": cands[0][0]})
                elif len(cands) > 1: ambiguous[lid] = [c[1] for c in cands]; cls = "text_only_ambiguous"
                else: cls = "no_match"
            if cid: text_class["exact_equal"] += 1
        match_class[cls] += 1
        idmap[lid] = cid
        if cid: matched_canon.add(cid)
        else: missing.append({"legacy_id": lid, "title": d.get("title", ""), "class": cls})
    # ---- extra canonical nodes (not matched by any legacy node), broken down by split provenance
    extra_ids = [c for c in prov if c not in matched_canon]
    extra_by_prov = collections.Counter("+".join(prov[c]) for c in extra_ids)
    # ---- legacy eval-gold consistency on the frozen subset: legacy golds (mapped) vs canonical gold_node_ids
    ev = None
    evf = [f for f in os.listdir(f"{ROOT}/{ds}") if f.startswith("eval_") and f.endswith(".jsonl") and "random" not in f]
    if evf:
        n_eq = n_sub = n_diff = n_unmapped = 0; ex = []
        for r in iter_jsonl(f"{ROOT}/{ds}/{evf[0]}"):
            lg = r.get("legacy_gold_ids") or []
            mapped = [idmap.get(g) for g in lg]
            if any(m is None for m in mapped): n_unmapped += 1
            ms = set(m for m in mapped if m); cs = set(r["gold_node_ids"])
            if ms == cs: n_eq += 1
            elif ms and ms <= cs: n_sub += 1
            else:
                n_diff += 1
                if len(ex) < 5: ex.append({"query_id": r["query_id"], "legacy_gold": lg, "mapped": sorted(ms), "canonical_gold": sorted(cs)})
        ev = {"file": evf[0], "n": n_eq + n_sub + n_diff, "legacy_golds_equal_canonical": n_eq, "legacy_golds_strict_subset_of_canonical": n_sub,
              "differ": n_diff, "queries_with_unmapped_legacy_gold": n_unmapped, "examples": ex}
    result = {"dataset": ds, "legacy_substrate": L, "legacy_master": f"data/processed/master_nodes_{L}.json",
              "legacy_count": n_leg, "canonical_count": n_canon,
              "overlap_legacy_nodes_mapped": n_leg - len(missing), "missing_in_canonical": len(missing), "missing_examples": missing[:20],
              "ambiguous_legacy_nodes": len(ambiguous), "ambiguous_examples": dict(list(ambiguous.items())[:10]),
              "extra_canonical_nodes_not_in_legacy": len(extra_ids), "extra_by_split_provenance": dict(extra_by_prov),
              "canonical_nodes_matched_by_legacy": len(matched_canon),
              "match_classes": dict(match_class), "changed_text": dict(text_class), "changed_text_examples": dict(text_examples),
              "title_differs_but_text_equal": title_diff, "title_differs_examples": title_diff_examples,
              "id_mapping_coverage": round((n_leg - len(missing)) / n_leg, 6),
              "eval_subset_gold_consistency": ev, "seconds": round(time.time() - t0, 1)}
    json.dump(result, open(f"{ROOT}/{ds}/legacy_comparison.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump({"legacy_to_canonical": idmap, "_ambiguous": ambiguous,
               "_note": "legacy node_id (master_nodes_<legacy>.json doc node) -> canonical_v1 node_id; null = no canonical match; "
                        "_ambiguous lists canonical candidates where the legacy identity merged several canonical entities"},
              open(f"{ROOT}/{ds}/node_id_map_legacy.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ("missing_examples", "ambiguous_examples", "changed_text_examples")}, ensure_ascii=True))


if __name__ == "__main__":
    main(sys.argv[1])
