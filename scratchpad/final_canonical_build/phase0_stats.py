"""Phase 0 forensic stats over OFFICIAL sources + legacy-eval identity recovery. Read-only over
data/original + data/final_canonical/_work/legacy. Writes data/final_canonical/_work/phase0/{ds}.json.
Streaming; peak RSS small. Usage: python phase0_stats.py <metaqa|musique|squad|2wiki>
"""
import sys, os, json, hashlib, unicodedata, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jsonstream import iter_json_array, iter_json_array_at, iter_jsonl

W = "data/final_canonical/_work"
os.makedirs(f"{W}/phase0", exist_ok=True)
SCR = "C:/Users/Swastik/AppData/Local/Temp/claude/C--Users-Swastik-Desktop-CRAG/70ead4d5-16f9-428c-af03-4cb084412966/scratchpad"


def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
def ws(s): return " ".join(s.split())
def nfc(s): return unicodedata.normalize("NFC", s)


def legacy_eval_ids(ds):
    return json.load(open(f"{SCR}/legacy_eval_{ds}.json"))


def legacy_questions(ds):
    q = {}
    for r in iter_jsonl(f"{W}/legacy/{ds}/questions.jsonl"):
        q[r["node_id"]] = r
    return q


# --------------------------------------------------------------------------------------- metaqa
def metaqa():
    out = {}
    dict_rows = []; names = collections.Counter(); blank = 0; bad = 0
    for line in open("data/original/metaqa/entity/kb_entity_dict.txt", encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            bad += 1; continue
        if "\t" not in line:
            bad += 1; continue
        idx, name = line.split("\t", 1)
        dict_rows.append((idx, name)); names[name] += 1
        if not name.strip(): blank += 1
    kb_ents = set(); triples = 0; malformed = 0
    for line in open("data/original/metaqa/kb.txt", encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) != 3:
            malformed += 1; continue
        triples += 1; kb_ents.add(p[0]); kb_ents.add(p[2])
    dict_names = set(names)
    # legacy normalization (loaders.load_metaqa): strip, lower, '_'->' '
    low = collections.defaultdict(list)
    for idx, name in dict_rows:
        low[name.strip().lower().replace("_", " ")].append(name)
    collided = {k: v for k, v in low.items() if len(v) > 1}
    # ids: are they contiguous ints?
    ints = [int(i) for i, _ in dict_rows]
    out.update({
        "dict_rows": len(dict_rows), "dict_blank_or_malformed_lines": bad, "dict_blank_names": blank,
        "dict_unique_names": len(dict_names), "dict_duplicate_names": {k: v for k, v in names.items() if v > 1},
        "dict_ids_min": min(ints), "dict_ids_max": max(ints), "dict_ids_contiguous_0_n": ints == list(range(len(ints))),
        "kb_triples": triples, "kb_malformed_lines": malformed, "kb_unique_entities": len(kb_ents),
        "kb_entities_not_in_dict": sorted(kb_ents - dict_names)[:20], "n_kb_entities_not_in_dict": len(kb_ents - dict_names),
        "dict_entities_not_in_kb": sorted(dict_names - kb_ents)[:20], "n_dict_entities_not_in_kb": len(dict_names - kb_ents),
        "legacy_lowercase_collapse": {"n_lowercase_keys": len(low), "n_collided_groups": len(collided),
                                      "n_names_absorbed": sum(len(v) - 1 for v in collided.values()),
                                      "examples": dict(list(collided.items())[:8])},
        "names_with_leading_trailing_ws": sum(1 for _, n in dict_rows if n != n.strip()),
        "names_nfc_changed": sum(1 for _, n in dict_rows if nfc(n) != n),
    })
    # legacy questions: ids metaqa_q_{hop}hop_{split}_{counter}; reproduce the counter from vanilla files
    lq = legacy_questions("metaqa")
    ev = legacy_eval_ids("metaqa")
    counter = 0; recovered = {}; mism = 0; want = set(ev["ids"])
    for hop in (1, 2, 3):
        for split in ("train", "dev", "test"):
            path = f"data/original/metaqa/{hop}-hop/vanilla/qa_{split}.txt"
            for li, line in enumerate(open(path, encoding="utf-8")):
                s = line.strip()
                if not s or "\t" not in s:
                    continue
                qtext, ans = s.split("\t", 1)
                nid = f"metaqa_q_{hop}hop_{split}_{counter}"; counter += 1
                if nid in want:
                    node = lq.get(nid)
                    ok = node is not None and node["content"] == qtext
                    if not ok: mism += 1
                    recovered[nid] = {"hop": hop, "split": split, "line_index": li, "question": qtext,
                                      "answers": [a.strip() for a in ans.split("|") if a.strip()],
                                      "legacy_text_match": ok, "legacy_golds": node["neighbors"] if node else None}
    out["legacy_eval"] = {"n": len(ev["ids"]), "recovered": len(recovered), "text_mismatch": mism,
                          "per_hop": collections.Counter(v["hop"] for v in recovered.values()),
                          "per_split": collections.Counter(v["split"] for v in recovered.values()),
                          "total_question_lines_counted": counter, "legacy_master_questions": len(lq)}
    json.dump(recovered, open(f"{W}/phase0/metaqa_legacy_eval_recovered.json", "w", encoding="utf-8"), ensure_ascii=False)
    return out


# --------------------------------------------------------------------------------------- musique
def musique():
    out = {}; F = {"train": "musique_ans_v1.0_train.jsonl", "dev": "musique_ans_v1.0_dev.jsonl", "test": "musique_ans_v1.0_test.jsonl"}
    n_para = collections.Counter(); text_h = {}; tt_h = set(); title_by_text = collections.defaultdict(set)
    ws_h = set(); nfcws_h = set(); empty_text = 0; n_q = collections.Counter(); sup_cnt = collections.Counter()
    title_variants_same_text = 0
    lq = legacy_questions("musique_clean"); ev = legacy_eval_ids("musique_clean")
    # legacy id musique_clean_q_train_N  -> FlashRAG train_N -> official train line N (verify by text)
    match = {"same_text_at_index": 0, "mismatch": 0, "examples": []}
    text_to_lines = collections.defaultdict(list)
    for split, fn in F.items():
        for i, m in enumerate(iter_jsonl(f"data/original/musique/v1.0/{fn}")):
            n_q[split] += 1
            sup_cnt[sum(1 for p in m["paragraphs"] if p.get("is_supporting"))] += 1
            for p in m["paragraphs"]:
                n_para[split] += 1
                t = p["paragraph_text"]
                if not t.strip(): empty_text += 1
                h = sha(t); text_h[h] = text_h.get(h, 0) + 1
                tt_h.add(sha(p["title"] + "\x00" + t)); title_by_text[h].add(p["title"])
                ws_h.add(sha(ws(t))); nfcws_h.add(sha(ws(nfc(t))))
            if split == "train":
                text_to_lines[m["question"]].append(i)
                nid = f"musique_clean_q_train_{i}"
                if nid in lq:
                    if lq[nid]["content"] == m["question"]: match["same_text_at_index"] += 1
                    else:
                        match["mismatch"] += 1
                        if len(match["examples"]) < 3: match["examples"].append((nid, lq[nid]["content"], m["question"]))
    multi = {h: sorted(s) for h, s in title_by_text.items() if len(s) > 1}
    out.update({"n_questions": dict(n_q), "n_paragraph_records": dict(n_para), "total_paragraph_records": sum(n_para.values()),
                "unique_text_sha": len(text_h), "unique_title_text_pairs": len(tt_h),
                "texts_under_multiple_titles": len(multi), "texts_under_multiple_titles_examples": dict(list(multi.items())[:5]),
                "unique_ws_normalized_text": len(ws_h), "unique_nfc_ws_text": len(nfcws_h), "empty_paragraph_texts": empty_text,
                "supporting_per_question_dist": dict(sup_cnt),
                "legacy_index_alignment": match, "legacy_master_questions": len(lq),
                "n_eval_ids_all_train_prefixed": sum(1 for x in ev["ids"] if "_q_train_" in x), "n_eval": len(ev["ids"])})
    return out


# --------------------------------------------------------------------------------------- squad
def squad():
    out = {}; F = {"train": "train-v2.0.json", "dev": "dev-v2.0.json"}
    n_art = collections.Counter(); n_par = collections.Counter(); n_q = collections.Counter()
    ctx_h = collections.Counter(); tc_h = set(); title_by_ctx = collections.defaultdict(set); ws_h = set()
    titles = collections.Counter(); titles_split = collections.defaultdict(set)
    lq = legacy_questions("squad_clean"); ev = legacy_eval_ids("squad_clean"); want = set(ev["ids"])
    counter = 0; recovered = {}; mism = 0
    for split, fn in F.items():
        for art in iter_json_array_at(f"data/original/squad/v2.0/{fn}", "data"):
            n_art[split] += 1; titles[art["title"]] += 1; titles_split[art["title"]].add(split)
            for p in art["paragraphs"]:
                n_par[split] += 1; c = p["context"]; h = sha(c); ctx_h[h] += 1
                tc_h.add(sha(art["title"] + "\x00" + c)); title_by_ctx[h].add(art["title"]); ws_h.add(sha(ws(c)))
                for qa in p["qas"]:
                    n_q[split] += 1
                    if split == "train":
                        nid = f"squad_clean_q_{counter}"; counter += 1
                        if nid in want:
                            node = lq.get(nid); ok = node is not None and node["content"] == qa["question"]
                            if not ok: mism += 1
                            recovered[nid] = {"official_id": qa["id"], "question": qa["question"], "legacy_text_match": ok,
                                              "is_impossible": qa.get("is_impossible", False), "legacy_golds": node["neighbors"] if node else None}
    out.update({"n_articles": dict(n_art), "n_paragraphs": dict(n_par), "n_questions": dict(n_q),
                "unique_context_sha": len(ctx_h), "unique_title_context_pairs": len(tc_h),
                "contexts_under_multiple_titles": sum(1 for s in title_by_ctx.values() if len(s) > 1),
                "duplicate_context_texts": sum(v - 1 for v in ctx_h.values() if v > 1),
                "unique_ws_normalized_context": len(ws_h),
                "titles_in_both_splits": sorted(t for t, s in titles_split.items() if len(s) > 1)[:10],
                "n_titles_in_both_splits": sum(1 for s in titles_split.values() if len(s) > 1),
                "duplicate_article_titles_within_all": {t: c for t, c in titles.items() if c > 1},
                "legacy_eval": {"n": len(ev["ids"]), "recovered": len(recovered), "text_mismatch": mism,
                                "train_question_lines_counted": counter, "legacy_master_questions": len(lq)}})
    json.dump(recovered, open(f"{W}/phase0/squad_legacy_eval_recovered.json", "w", encoding="utf-8"), ensure_ascii=False)
    return out


# --------------------------------------------------------------------------------------- 2wiki
def twowiki():
    out = {}; B = "data/original/2wiki/v1.0_ids_april2021"
    lq = legacy_questions("2wiki_clean"); ev = legacy_eval_ids("2wiki_clean"); want = set(ev["ids"])
    legacy_text = collections.defaultdict(list)
    for nid, r in lq.items(): legacy_text[r["content"]].append(nid)
    n_q = collections.Counter(); n_ctx = collections.Counter(); title_first_sha = {}; title_multi_text = collections.Counter()
    text_sha_set = set(); ws_sha_set = set(); empty_ctx = 0; sf_titles_not_in_ctx = 0; sf_total = 0
    matched = {}; multi_match_texts = 0; per_split_matches = collections.Counter(); dup_official_texts = collections.Counter()
    t0 = time.time()
    for split in ("train", "dev", "test"):
        for i, q in enumerate(iter_json_array(f"{B}/{split}.json")):
            n_q[split] += 1
            ctx_titles = set()
            for e in q["context"]:
                title, sents = e[0], e[1]; text = " ".join(sents); n_ctx[split] += 1; ctx_titles.add(title)
                if not text.strip(): empty_ctx += 1
                h = sha(text); text_sha_set.add(h); ws_sha_set.add(sha(ws(text)))
                prev = title_first_sha.get(title)
                if prev is None: title_first_sha[title] = h
                elif prev != h: title_multi_text[title] += 1
            for s in q.get("supporting_facts", []):
                sf_total += 1
                if s[0] not in ctx_titles: sf_titles_not_in_ctx += 1
            if q["question"] in legacy_text:
                dup_official_texts[q["question"]] += 1
                for nid in legacy_text[q["question"]]:
                    matched.setdefault(nid, []).append({"official_id": q["_id"], "split": split, "index": i,
                                                        "type": q.get("type"), "sf_titles": sorted({s[0] for s in q.get("supporting_facts", [])}),
                                                        "legacy_golds": lq[nid]["neighbors"]})
                per_split_matches[split] += 1
            if (i + 1) % 20000 == 0:
                print(f"  2wiki {split} {i+1} ({time.time()-t0:.0f}s)", flush=True)
    amb = {k: v for k, v in matched.items() if len(v) > 1}
    out.update({"n_questions": dict(n_q), "n_context_records": dict(n_ctx), "total_context_records": sum(n_ctx.values()),
                "unique_titles": len(title_first_sha), "titles_with_multiple_texts": len(title_multi_text),
                "titles_with_multiple_texts_examples": dict(list(title_multi_text.items())[:5]),
                "unique_text_sha": len(text_sha_set), "unique_ws_text_sha": len(ws_sha_set), "empty_context_texts": empty_ctx,
                "supporting_fact_titles_total": sf_total, "supporting_fact_titles_not_in_own_context": sf_titles_not_in_ctx,
                "legacy_questions": len(lq), "legacy_questions_matched_by_text": len(matched),
                "legacy_questions_matched_ambiguous(>1 official)": len(amb), "official_matches_per_split": dict(per_split_matches),
                "official_question_texts_seen_more_than_once_among_matched": sum(1 for v in dup_official_texts.values() if v > 1),
                "legacy_eval": {"n": len(want), "matched": sum(1 for x in want if x in matched),
                                "ambiguous": sum(1 for x in want if x in amb)}})
    json.dump({k: v for k, v in matched.items()}, open(f"{W}/phase0/2wiki_legacy_questions_matched.json", "w", encoding="utf-8"), ensure_ascii=False)
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    t0 = time.time()
    fn = {"metaqa": metaqa, "musique": musique, "squad": squad, "2wiki": twowiki}[ds]
    res = fn(); res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{W}/phase0/{ds}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=lambda o: o if not isinstance(o, set) else sorted(o))
    print(json.dumps(res, ensure_ascii=True, default=str)[:6000])
