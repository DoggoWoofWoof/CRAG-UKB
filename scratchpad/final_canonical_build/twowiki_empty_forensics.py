"""2WIKI empty-canonical-text forensics -- READ-ONLY over the official source.

Mirrors scratchpad/final_canonical_build/hotpot_empty_forensics.py, adapted to 2wiki's rule.

Replays build_kb.py::build_corpus_2wiki lines 269-271 VERBATIM

    sents = r.get("sentences") or []
    if isinstance(sents, str): sents = [sents]
    body = " ".join(sents)              # NO .strip()

over data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip and reports every record whose
canonical text is empty (`not body.strip()`), exactly as the builder counts `empty_text`.

It ALSO builds, in the same single pass, the byte-identity set of the ALREADY-ENCODED encoder inputs
(Phase-C 2wiki_universe input = strip(" ".join(sentences)), per scratchpad/build_2wiki_universe.py:42)
plus the 398,354 PHASE_C_VIEW398 stored inputs, so the feature-bill projection for a title fallback can
be answered WITHOUT running any encoder: byte-identical input => identical token IDs (the reuse rule of
scratchpad/final_canonical_build/reuse_map_kb.py, independently verified there k=20000 mismatches=0).

Writes ONLY to results/data_audit/final_canonical_v1/2wiki/.  Runs no builder, touches no node table.
"""
import os, sys, json, zipfile, io, time, collections, hashlib, unicodedata
import numpy as np

ROOT = r"C:\Users\Swastik\Desktop\CRAG"
ZP = os.path.join(ROOT, "data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip")
MEM = "para_with_hyperlink.jsonl"
VIEW398 = os.path.join(ROOT, "data/canonical/2wiki/encodings/_src/docs")
UNIV_SRC = os.path.join(ROOT, "data/canonical/2wiki_universe/encodings/_src/docs")
OUT = os.environ.get("TWOWIKI_FORENSICS_OUT") or os.path.join(ROOT, "results/data_audit/final_canonical_v1/2wiki")
os.makedirs(OUT, exist_ok=True)

CAP = 6_100_000          # >= 5,989,847 source records
EXPECTED_EMPTY = 87765
EXPECTED_N = 5989847
LIMIT = int(os.environ.get("TWOWIKI_FORENSICS_LIMIT", "0"))   # >0: smoke-test on a prefix of the member


def h64(s):
    """first 8 bytes of sha256(s) as an int -- a 64-bit set key (collision prob ~1e-6 at 6.4e6 keys)."""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def wsnorm(s):
    return " ".join(s.split())


def categorize(t):
    """Coarse page-kind bucket from the source title alone (no external lookup)."""
    tl = t.lower()
    if not t.strip():
        return "empty_title"
    for p in ("category:", "template:", "wikipedia:", "file:", "portal:", "help:", "module:",
              "draft:", "mediawiki:", "book:", "talk:", "user:"):
        if tl.startswith(p):
            return "namespace_" + p[:-1]
    if "(disambiguation)" in tl:
        return "disambiguation"
    if tl.startswith("list of ") or tl.startswith("lists of "):
        return "list_page"
    if tl.startswith("index of ") or tl.startswith("outline of ") or tl.startswith("timeline of "):
        return "index_outline_timeline"
    return "ordinary_article"


def main():
    t0 = time.time()
    enc_exact = np.zeros(CAP, np.uint64)      # sha64(strip(body))    -- gte/BPE byte-identity class
    enc_ws = np.zeros(CAP, np.uint64)         # sha64(wsnorm(body))   -- whitespace-insensitive class
    enc_wslow = np.zeros(CAP, np.uint64)      # sha64(wsnorm(body).lower()) -- WordPiece-uncased class
    title_all = np.zeros(CAP, np.uint64)      # sha64(title) for EVERY record (dup-title accounting)
    n = 0
    raw = 0
    empty_text = 0
    strip_delta = 0
    strip_examples = []
    n_sent_total = 0
    n_mentions_total = 0
    key_inventory = collections.Counter()
    hits = []

    z = zipfile.ZipFile(ZP)
    with z.open(MEM) as fh:
        for li, rawline in enumerate(io.TextIOWrapper(fh, encoding="utf-8", newline="")):
            if LIMIT and raw >= LIMIT:
                break
            if not rawline.strip():
                continue
            raw += 1
            try:
                r = json.loads(rawline)
            except Exception:
                continue
            cid = r.get("id")
            if cid is None or str(cid) == "":
                continue
            cid = str(cid)

            # ---- builder rule, verbatim (build_kb.py:269-271) ----
            sents = r.get("sentences") or []
            if isinstance(sents, str):
                sents = [sents]
            body = " ".join(sents)
            # ------------------------------------------------------

            title = r.get("title", "")
            ments = r.get("mentions") or []
            n_sent_total += len(sents)
            n_mentions_total += len(ments)
            if body != body.strip():
                strip_delta += 1
                if len(strip_examples) < 20:
                    strip_examples.append({"curid": cid, "title": title, "len": len(body),
                                           "len_stripped": len(body.strip())})
            sb = body.strip()
            enc_exact[n] = h64(sb)
            w = wsnorm(body)
            enc_ws[n] = h64(w)
            enc_wslow[n] = h64(w.lower())
            title_all[n] = h64(title)
            n += 1

            if sb:
                continue
            # ---------------- this record's canonical text is empty ----------------
            empty_text += 1
            for k in r.keys():
                key_inventory[k] += 1
            ref_urls = [m.get("ref_url") for m in ments if isinstance(m, dict) and m.get("ref_url")]
            other = {k: v for k, v in r.items() if k not in ("id", "title", "sentences", "mentions")}
            hits.append({
                "canonical_node_id": "2wiki:c%s" % cid,
                "source_identity_curid": cid,
                "source_record_id": "%s:%d" % (MEM, li),
                "canonical_title": title,
                "canonical_title_len": len(title),
                "canonical_text_current": body,
                "canonical_text_current_repr": repr(body),
                "canonical_text_current_len": len(body),
                "canonical_text_is_exactly_empty_string": body == "",
                "canonical_text_is_whitespace_only": (body != "" and body.strip() == ""),
                "raw_sentences_field": sents,
                "raw_sentences_field_repr": repr(r.get("sentences")),
                "raw_sentences_field_type": type(r.get("sentences")).__name__,
                "raw_sentences_n": len(sents),
                "raw_sentences_n_nonempty": sum(1 for s in sents if isinstance(s, str) and s.strip()),
                "raw_mentions_n": len(ments),
                "raw_mentions_ref_urls": ref_urls[:50],
                "raw_mentions_ref_url_residue_nonempty": bool(ref_urls),
                "other_source_fields": {k: (v if isinstance(v, (str, int, float, bool, type(None)))
                                            else json.dumps(v, ensure_ascii=False)[:400]) for k, v in other.items()},
                "other_source_text_nonempty": any(str(v).strip() for v in other.values()),
                "all_source_keys": sorted(r.keys()),
                "proposed_fallback_text": title,
                "category": categorize(title),
                "_ti_exact": h64(title),
                "_ti_ws": h64(wsnorm(title)),
                "_ti_wslow": h64(wsnorm(title).lower()),
                "_ti_nfkd_ascii": h64(unicodedata.normalize("NFKD", wsnorm(title).lower())
                                      .encode("ascii", "ignore").decode("ascii")),
            })
            if raw % 1_000_000 == 0:
                print("  %d records, %d empty, %.0fs" % (raw, empty_text, time.time() - t0), flush=True)

    print("DONE scan: %d raw lines-with-json, %d records, %d empty-canonical-text, %.0fs"
          % (raw, n, empty_text, time.time() - t0), flush=True)

    enc_exact = enc_exact[:n]; enc_ws = enc_ws[:n]; enc_wslow = enc_wslow[:n]; title_all = title_all[:n]

    # ---------- fold in the second already-encoded family (PHASE_C_VIEW398, 398,354 stored inputs) ----------
    v_exact, v_ws, v_wslow = [], [], []
    nv = 0
    import glob as _g
    for p in sorted(_g.glob(os.path.join(VIEW398, "shard_*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line).get("text") or ""
                v_exact.append(h64(t)); w = wsnorm(t)
                v_ws.append(h64(w)); v_wslow.append(h64(w.lower())); nv += 1
    print("  PHASE_C_VIEW398 stored encoder inputs hashed: %d" % nv, flush=True)

    # ---------- sanity: my derived universe inputs must equal the STORED universe encoder inputs ----------
    se = np.sort(enc_exact)
    checked = matched = 0
    for p in sorted(_g.glob(os.path.join(UNIV_SRC, "shard_*.jsonl")))[:1]:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line).get("text") or ""
                checked += 1
                i = np.searchsorted(se, np.uint64(h64(t)))
                matched += int(i < len(se) and se[i] == np.uint64(h64(t)))
    print("  derived-vs-stored universe encoder-input check: %d/%d present" % (matched, checked), flush=True)

    def mkset(a, extra):
        return np.unique(np.concatenate([a, np.array(extra, np.uint64)]) if extra else a)

    S_exact = mkset(enc_exact, v_exact)
    S_ws = mkset(enc_ws, v_ws)
    S_wslow = mkset(enc_wslow, v_wslow)
    st_all = np.sort(title_all)

    def inset(S, q):
        q = np.array(q, np.uint64)
        i = np.searchsorted(S, q)
        ic = np.clip(i, 0, max(len(S) - 1, 0))
        return (i < len(S)) & (S[ic] == q)

    ti_e = inset(S_exact, [h["_ti_exact"] for h in hits])
    ti_w = inset(S_ws, [h["_ti_ws"] for h in hits])
    ti_l = inset(S_wslow, [h["_ti_wslow"] for h in hits])
    # how many corpus records (of any kind) carry each hit title
    qt = np.array([h["_ti_exact"] for h in hits], np.uint64)
    lo = np.searchsorted(st_all, qt, "left"); hi = np.searchsorted(st_all, qt, "right")
    title_count_corpus = (hi - lo)

    hit_title_ct = collections.Counter(h["canonical_title"] for h in hits)
    for k, h in enumerate(hits):
        h["title_is_already_encoded_input_exact"] = bool(ti_e[k])
        h["title_is_already_encoded_input_wsnorm"] = bool(ti_w[k])
        h["title_is_already_encoded_input_wsnorm_lower"] = bool(ti_l[k])
        h["n_corpus_records_with_this_title"] = int(title_count_corpus[k])
        h["n_empty_records_with_this_title"] = hit_title_ct[h["canonical_title"]]
        for kk in ("_ti_exact", "_ti_ws", "_ti_wslow", "_ti_nfkd_ascii"):
            h.pop(kk)

    hits.sort(key=lambda x: int(x["source_identity_curid"]))
    with open(os.path.join(OUT, "EMPTY_TEXT_AUDIT.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False, sort_keys=True) + "\n")

    # ------------------------------- counts -------------------------------
    c = collections.Counter()
    combos = collections.Counter()
    sent_shape = collections.Counter()
    cats = collections.Counter()
    tl = []
    for h in hits:
        se_ = h["raw_sentences_n_nonempty"] > 0
        ti = bool((h["canonical_title"] or "").strip())
        me = h["raw_mentions_n"] > 0
        ot = h["other_source_text_nonempty"]
        c["sentences_nonempty" if se_ else "sentences_empty"] += 1
        c["title_nonempty" if ti else "title_empty"] += 1
        c["mentions_nonempty" if me else "mentions_empty"] += 1
        c["other_source_text_nonempty" if ot else "other_source_text_empty"] += 1
        if not se_ and not ti and not me and not ot:
            c["truly_no_readable_source_text"] += 1
        c["exact_empty_string" if h["canonical_text_is_exactly_empty_string"] else "whitespace_only"] += 1
        combos["sentences_%s + title_%s + mentions_%s + other_%s"
               % ("nonempty" if se_ else "empty", "present" if ti else "absent",
                  "nonempty" if me else "empty", "present" if ot else "absent")] += 1
        sent_shape["n_sentences=%d (all-empty-strings)" % h["raw_sentences_n"]] += 1
        cats[h["category"]] += 1
        tl.append(h["canonical_title_len"])
        c["title_reusable_dense_byte_identical" if h["title_is_already_encoded_input_exact"]
          else "title_needs_new_dense_encode"] += 1
        c["title_reusable_splade_wsnorm_lower" if h["title_is_already_encoded_input_wsnorm_lower"]
          else "title_needs_new_splade_encode"] += 1

    distinct_titles = len(hit_title_ct)
    distinct_titles_needing_dense = len({h["canonical_title"] for h in hits
                                         if not h["title_is_already_encoded_input_exact"]})
    distinct_wslow_needing_splade = len({wsnorm(h["canonical_title"]).lower() for h in hits
                                         if not h["title_is_already_encoded_input_wsnorm_lower"]})

    summary = {
        "dataset": "2wiki",
        "scan": {"zip": ZP, "member": MEM, "raw_json_lines": raw, "records_with_identity": n,
                 "seconds": round(time.time() - t0, 1),
                 "rule": "build_kb.py:269-271 -- sents=r.get('sentences') or []; body=' '.join(sents); "
                         "empty iff not body.strip()  (NO .strip() on body)"},
        "assertions": {
            "identified_empty_text_nodes": empty_text,
            "expected_empty_text_nodes": EXPECTED_EMPTY,
            "EMPTY_COUNT_MATCHES": empty_text == EXPECTED_EMPTY,
            "records_scanned": n, "expected_records": EXPECTED_N,
            "RECORD_COUNT_MATCHES": n == EXPECTED_N,
            "n_records_where_strip_changes_text": strip_delta,
            "strip_examples": strip_examples,
        },
        "corpus_totals": {"total_sentences": n_sent_total, "total_hyperlink_mentions": n_mentions_total},
        "counts": dict(c),
        "distinct_combinations": dict(combos),
        "sentences_field_shape_over_hits": dict(sent_shape),
        "source_key_inventory_over_hits": dict(key_inventory),
        "categories": dict(cats),
        "titles": {
            "unique_titles": distinct_titles,
            "empty_titles": sum(v for k, v in hit_title_ct.items() if not (k or "").strip()),
            "duplicate_title_groups": sum(1 for v in hit_title_ct.values() if v > 1),
            "records_in_duplicate_title_groups": sum(v for v in hit_title_ct.values() if v > 1),
            "top_duplicate_titles": hit_title_ct.most_common(15),
            "title_len_min": min(tl) if tl else None, "title_len_max": max(tl) if tl else None,
            "title_len_mean": round(sum(tl) / len(tl), 2) if tl else None,
            "hits_whose_title_also_titles_a_NONEMPTY_record":
                sum(1 for h in hits if h["n_corpus_records_with_this_title"] > h["n_empty_records_with_this_title"]),
        },
        "duplicate_curids": {k: v for k, v in
                             collections.Counter(h["source_identity_curid"] for h in hits).items() if v > 1},
        "feature_bill_projection": {
            "rule": "reuse_map_kb.py: a row is reusable IFF its frozen-tokenizer token-ID sequence equals an "
                    "already-encoded row's. Byte-identical input => identical token IDs (sufficient, verified "
                    "k=20000 mismatches=0). MEASURED here = byte-identity membership; no tokenizer/encoder was run.",
            "already_encoded_input_pool": {"PHASE_C_UNIVERSE": n, "PHASE_C_VIEW398": nv,
                                           "distinct_exact": int(len(S_exact)),
                                           "distinct_wsnorm": int(len(S_ws)),
                                           "distinct_wsnorm_lower": int(len(S_wslow))},
            "derived_vs_stored_universe_input_check": {"checked": checked, "present": matched},
            "dense_rows_currently_reusable_measured_by_reuse_map": 5989847,
            "dense_rows_newly_required_upper_bound": int(c["title_needs_new_dense_encode"]),
            "dense_distinct_new_forward_passes_upper_bound": distinct_titles_needing_dense,
            "splade_rows_newly_required_upper_bound": int(c["title_needs_new_splade_encode"]),
            "splade_distinct_new_forward_passes_upper_bound": distinct_wslow_needing_splade,
        },
    }
    with open(os.path.join(OUT, "EMPTY_TEXT_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:6000])


def pass2_inlinks():
    """SECOND read-only pass: how the rest of the corpus REFERS to the empty-text records.

    The only structural signal in the source is mentions[].ref_ids (curids of hyperlink targets).  A record
    with zero sentences also has zero mentions, so it emits no outgoing edge; this pass measures its INCOMING
    edges, which is what decides whether it is an isolated node in the SK/H4 topology.  Also records whether
    each empty node is referenced by the official queries (gold / context) -- impact assessment only, read from
    the ALREADY BUILT query files; nothing about the corpus depends on it.
    """
    t0 = time.time()
    hits = [json.loads(l) for l in open(os.path.join(OUT, "EMPTY_TEXT_AUDIT.jsonl"), encoding="utf-8")]
    cur = np.array(sorted(int(h["source_identity_curid"]) for h in hits), np.int64)
    inlink_mentions = np.zeros(len(cur), np.int64)
    inlink_articles = np.zeros(len(cur), np.int64)
    seen_from = np.zeros(len(cur), np.int64) - 1        # last source row that credited this target
    row = 0
    z = zipfile.ZipFile(ZP)
    with z.open(MEM) as fh:
        for rawline in io.TextIOWrapper(fh, encoding="utf-8", newline=""):
            if not rawline.strip():
                continue
            try:
                r = json.loads(rawline)
            except Exception:
                continue
            row += 1
            for m in (r.get("mentions") or []):
                for rid in (m.get("ref_ids") or []):
                    try:
                        v = int(rid)
                    except Exception:
                        continue
                    i = np.searchsorted(cur, v)
                    if i < len(cur) and cur[i] == v:
                        inlink_mentions[i] += 1
                        if seen_from[i] != row:
                            seen_from[i] = row
                            inlink_articles[i] += 1
            if row % 1_000_000 == 0:
                print("  [pass2] %d records, %.0fs" % (row, time.time() - t0), flush=True)
    print("  [pass2] done %d records, %.0fs" % (row, time.time() - t0), flush=True)

    # official query references (impact assessment; read from the already-built query layer)
    qref_gold = collections.Counter(); qref_ctx = collections.Counter()
    qdir = os.path.join(ROOT, "data/final_canonical/2wiki/queries")
    ids = set(h["canonical_node_id"] for h in hits)
    q_seen = {}
    for fn in sorted(os.listdir(qdir)) if os.path.isdir(qdir) else []:
        if not fn.endswith(".jsonl"):
            continue
        for line in open(os.path.join(qdir, fn), encoding="utf-8"):
            if not line.strip():
                continue
            q = json.loads(line)
            for nid in (q.get("gold_node_ids") or []):
                if nid in ids:
                    qref_gold[fn] += 1; q_seen.setdefault(nid, set()).add("gold:" + fn)
            for nid in (q.get("context_node_ids") or []):
                if nid in ids:
                    qref_ctx[fn] += 1; q_seen.setdefault(nid, set()).add("context:" + fn)

    pos = {int(c): i for i, c in enumerate(cur)}
    for h in hits:
        i = pos[int(h["source_identity_curid"])]
        h["n_incoming_hyperlink_mentions"] = int(inlink_mentions[i])
        h["n_distinct_articles_linking_here"] = int(inlink_articles[i])
        h["is_isolated_in_mention_graph"] = bool(inlink_mentions[i] == 0)   # outgoing is 0 by construction
        h["official_query_roles"] = sorted(q_seen.get(h["canonical_node_id"], []))
    with open(os.path.join(OUT, "EMPTY_TEXT_AUDIT.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for h in sorted(hits, key=lambda x: int(x["source_identity_curid"])):
            f.write(json.dumps(h, ensure_ascii=False, sort_keys=True) + "\n")

    nz = int((inlink_mentions > 0).sum())
    p = os.path.join(OUT, "EMPTY_TEXT_SUMMARY.json")
    j = json.load(open(p, encoding="utf-8"))
    j["inlink_structure_MEASURED"] = {
        "records_scanned": row,
        "empty_nodes_with_at_least_one_incoming_hyperlink": nz,
        "empty_nodes_with_zero_incoming_hyperlinks": int(len(cur) - nz),
        "total_incoming_mentions": int(inlink_mentions.sum()),
        "distinct_linking_articles_total": int(inlink_articles.sum()),
        "max_incoming_mentions": int(inlink_mentions.max()) if len(cur) else 0,
        "median_incoming_mentions": float(np.median(inlink_mentions)) if len(cur) else 0,
        "outgoing_edges": "0 by construction -- a record with no sentences carries no mentions (measured: "
                          "mentions_empty == EMPTY_BEFORE in counts)",
        "official_query_reference_counts": {"gold_by_split": dict(qref_gold), "context_by_split": dict(qref_ctx),
                                            "distinct_empty_nodes_referenced": len(q_seen)},
        "seconds": round(time.time() - t0, 1)}
    json.dump(j, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(j["inlink_structure_MEASURED"], indent=2, ensure_ascii=False))


def split_tables():
    """The full per-record table is 102 MB -- the largest TRACKED file anywhere under results/ is 2.85 MB, and
    GitHub refuses >100 MB, so it cannot live in the tracked mirror.  Apply the repo's own rule (mirror_audit.py:
    "bulky tables stay only under data/final_canonical/"): FULL table -> data/final_canonical/2wiki/ (gitignored),
    COMPACT per-record table (same 87,765 rows, redundant/constant columns dropped) -> results/.  Nothing is lost:
    every column dropped from the compact table is either constant across all rows (and recorded in
    EMPTY_TEXT_SUMMARY.json) or present in the full table.
    """
    src = os.path.join(OUT, "EMPTY_TEXT_AUDIT.jsonl")
    full_dir = os.path.join(ROOT, "data/final_canonical/2wiki")
    os.makedirs(full_dir, exist_ok=True)
    full = os.path.join(full_dir, "EMPTY_TEXT_AUDIT.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    with open(full, "w", encoding="utf-8", newline="\n") as f:          # FULL fidelity, gitignored tree
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    # compact schema -- only the columns that actually VARY per record and are needed to re-verify the report.
    # legend:  id = canonical_node_id | title = source-native title (= the proposed fallback text)
    #          cat = category | inlinks = n_incoming_hyperlink_mentions
    # everything else is either constant over all 87,765 rows (recorded in EMPTY_TEXT_SUMMARY.json) or in the
    # full table under data/final_canonical/2wiki/EMPTY_TEXT_AUDIT.jsonl.
    with open(src, "w", encoding="utf-8", newline="\n") as f:            # COMPACT, tracked tree
        for r in rows:
            f.write(json.dumps({"id": r["canonical_node_id"], "title": r["canonical_title"],
                                "cat": r["category"], "inlinks": r["n_incoming_hyperlink_mentions"]},
                               ensure_ascii=False) + "\n")
    print("full  -> %s (%.1f MB)" % (full, os.path.getsize(full) / 1e6))
    print("compact -> %s (%.1f MB)" % (src, os.path.getsize(src) / 1e6))


def tokenizer_probe():
    """MEASURED tokenizer evidence (no encoder, no forward pass): what the two FROZEN tokenizers do to the
    empty string, to whitespace, and to a title.  Patches EMPTY_TEXT_SUMMARY.json with the result."""
    import glob as _g
    from transformers import AutoTokenizer
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    def snap(rd, mh):
        return os.path.dirname(sorted(_g.glob(os.path.join(hub, rd, "snapshots", "*", mh)))[0])

    d = AutoTokenizer.from_pretrained(snap("models--Alibaba-NLP--gte-Qwen2-1.5B-instruct", "tokenizer.json"),
                                      local_files_only=True)
    s = AutoTokenizer.from_pretrained(snap("models--naver--splade-cocondenser-ensembledistil", "tokenizer.json"),
                                      local_files_only=True)
    tests = ["", " ", "  ", "Masaki Kobayashi", " Masaki Kobayashi ", "Masaki  Kobayashi",
             "masaki kobayashi", "Tó Neinilii", "To Neinilii"]
    ev = {"gte_qwen2_dense_bpe": {repr(t): d(t, add_special_tokens=True, truncation=True,
                                             max_length=32768)["input_ids"] for t in tests},
          "splade_wordpiece": {repr(t): s(t, add_special_tokens=True, truncation=True,
                                          max_length=256)["input_ids"] for t in tests},
          "splade_do_lower_case": s.init_kwargs.get("do_lower_case"),
          "reading": "gte BPE is whitespace- and case-preserving and maps '' to the ZERO-LENGTH sequence []; "
                     "SPLADE WordPiece maps '', ' ' and '  ' all to [CLS][SEP] and is whitespace-, case- and "
                     "accent-insensitive, so its equivalence classes are strictly coarser than gte's."}
    p = os.path.join(OUT, "EMPTY_TEXT_SUMMARY.json")
    j = json.load(open(p, encoding="utf-8"))
    j["tokenizer_evidence_MEASURED"] = ev
    json.dump(j, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(ev, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if os.environ.get("TWOWIKI_FORENSICS_TOKPROBE") == "1":
        tokenizer_probe()
    elif os.environ.get("TWOWIKI_FORENSICS_PASS2") == "1":
        pass2_inlinks()
    elif os.environ.get("TWOWIKI_FORENSICS_SPLIT") == "1":
        split_tables()
    else:
        main()
