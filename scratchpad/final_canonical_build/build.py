"""FINAL canonical dataset builder (Track B, canonical_v1).

    python scratchpad/final_canonical_build/build.py <ds> [--eval-subset legacy|none|random_dev:<seed>|PATH]
                                                          [--out DIR] [--tag NAME]

ds in {metaqa, 2wiki, musique, squad}.  Default --out = data/final_canonical/<ds>

INVARIANT (enforced structurally + tested by qi_test.py):
    QUERY SUBSETTING IS ALLOWED.  CORPUS SUBSETTING BY QUERY IS FORBIDDEN.
    build_corpus_<ds>() receives ONLY official corpus source paths; it never sees --eval-subset, never reads
    gold / supporting / answer fields, and its output is a pure function of the official source files.
    The query stage (build_queries_<ds>) runs AFTER the corpus is frozen and only *indexes* into it.

Outputs (only under --out):
    nodes.jsonl                 canonical node table, sorted by node_id, one JSON object per line (sort_keys)
    queries/{train,dev,test}.jsonl   official queries with gold refs resolved to canonical node_ids
    eval_<N>.jsonl              frozen eval subset (only if --eval-subset != none)
    integrity_report.json       Phase-5 report
    build_info.json             command, builder sha256, git commit, timestamps, hashes, peak RSS

Normalization policy: stored text is the EXACT official string (no NFC/NFKC, no casefold, no whitespace
collapse). Identity keys are exact strings too.  The only construction step is 2wiki's
text = " ".join(sentences) (official file stores sentence lists).  Diagnostics report how many extra
collapses NFC / whitespace / casefold WOULD cause, so the alternative is quantified, not applied.
"""
import sys, os, json, hashlib, unicodedata, time, argparse, random, subprocess, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jsonstream import iter_json_array, iter_json_array_at, iter_jsonl

VERSION = "canonical_v1"
ROOT = "data/final_canonical"
WORK = f"{ROOT}/_work"
SRC = {
    "metaqa": {"dict": "data/original/metaqa/entity/kb_entity_dict.txt", "kb": "data/original/metaqa/kb.txt",
               "qa_dir": "data/original/metaqa"},
    "2wiki": {"base": "data/original/2wiki/v1.0_ids_april2021", "splits": ["train", "dev", "test"]},
    "musique": {"base": "data/original/musique/v1.0",
                "files": {"train": "musique_ans_v1.0_train.jsonl", "dev": "musique_ans_v1.0_dev.jsonl", "test": "musique_ans_v1.0_test.jsonl"},
                "singlehop": "dev_test_singlehop_questions_v1.0.json"},
    "squad": {"base": "data/original/squad/v2.0", "files": {"train": "train-v2.0.json", "dev": "dev-v2.0.json"}},
}
LEGACY_NAME = {"metaqa": "metaqa", "2wiki": "2wiki_clean", "musique": "musique_clean", "squad": "squad_clean"}
SEP = "\x1f"  # unit separator between identity components inside hashes


def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
def ws(s): return " ".join(s.split())
def nfc(s): return unicodedata.normalize("NFC", s)
def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    os.makedirs(ROOT, exist_ok=True)
    with open(f"{ROOT}/_build.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def peak_rss_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)
    except Exception:
        return None


class Diag:
    """Counts how many extra collapses lossy normalizations WOULD cause (never applied)."""
    def __init__(self):
        self.exact = set(); self.nfc_ = set(); self.ws_ = set(); self.cf = set()
    def add(self, key):
        self.exact.add(key); self.nfc_.add(nfc(key)); self.ws_.add(ws(key)); self.cf.add(ws(nfc(key)).casefold())
    def report(self):
        n = len(self.exact)
        return {"exact_identities": n, "extra_collapses_if_NFC": n - len(self.nfc_),
                "extra_collapses_if_whitespace_collapse": n - len(self.ws_),
                "extra_collapses_if_NFC+ws+casefold": n - len(self.cf)}


# =====================================================================================  CORPUS STAGE
# Each build_corpus_* returns (nodes: list[dict], accounting: dict). NO gold / eval information is read.

def build_corpus_metaqa(S):
    nodes = []; raw = 0; dropped = []; names = {}; diag = Diag()
    for li, line in enumerate(open(S["dict"], encoding="utf-8")):
        line = line.rstrip("\n")
        if not line:
            dropped.append({"line": li, "reason": "blank_line"}); continue
        if "\t" not in line:
            dropped.append({"line": li, "reason": "no_tab"}); continue
        raw += 1
        idx, name = line.split("\t", 1)
        idx = int(idx)
        if name in names:
            dropped.append({"line": li, "reason": "duplicate_entity_name", "name": name}); continue
        names[name] = idx; diag.add(name)
        nodes.append({"node_id": f"metaqa:e{idx:05d}", "source_id": name, "source_record_id": f"kb_entity_dict.txt:{idx}",
                      "split_provenance": ["kb"], "title": name, "text": name, "source_file": S["dict"],
                      "content_hash": sha(name + SEP + name), "kb_entity_index": idx, "kb_degree": 0,
                      "n_source_records": 1})
    # kb.txt cross-check (official, query-independent): entity set + degree
    by_name = {n["source_id"]: n for n in nodes}
    kb_ents = set(); triples = 0; malformed = 0
    for line in open(S["kb"], encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) != 3:
            malformed += 1; continue
        triples += 1
        for e in (p[0], p[2]):
            kb_ents.add(e)
            if e in by_name: by_name[e]["kb_degree"] += 1
    missing = sorted(kb_ents - set(by_name)); extra = sorted(set(by_name) - kb_ents)
    acc = {"raw_source_records": raw, "dropped_records": dropped, "duplicates_collapsed": 0,
           "identity": "official kb_entity_dict.txt entity index (name exact); node_id = metaqa:e<idx zero-padded 5>",
           "kb_txt_cross_check": {"triples": triples, "malformed_lines": malformed, "kb_unique_entities": len(kb_ents),
                                  "kb_entities_missing_from_dict": missing[:20], "n_kb_entities_missing_from_dict": len(missing),
                                  "dict_entities_absent_from_kb": extra[:20], "n_dict_entities_absent_from_kb": len(extra)},
           "identity_normalization_diagnostics": diag.report(),
           "note": "text = official entity name (verbalization with kb.txt triples is a downstream encoding choice, not corpus identity)"}
    return nodes, acc


def build_corpus_2wiki(S):
    by_title = {}; raw = 0; dropped = []; conflicts = []; dup = 0; diag = Diag()
    for split in S["splits"]:
        path = f"{S['base']}/{split}.json"
        for qi, q in enumerate(iter_json_array(path)):
            for pi, entry in enumerate(q["context"]):          # ONLY the context field is read here
                title, sents = entry[0], entry[1]
                raw += 1
                text = " ".join(sents)
                if not isinstance(title, str) or title == "":
                    dropped.append({"where": f"{split}:{qi}:{pi}", "reason": "empty_or_nonstring_title"}); continue
                nd = by_title.get(title)
                if nd is None:
                    diag.add(title)
                    by_title[title] = {"node_id": f"2wiki:{sha(title)[:24]}", "source_id": title,
                                       "source_record_id": f"{split}.json:{qi}:{pi}", "split_provenance": {split},
                                       "title": title, "text": text, "source_file": path,
                                       "content_hash": sha(title + SEP + text), "n_sentences": len(sents),
                                       "n_source_records": 1}
                else:
                    dup += 1; nd["n_source_records"] += 1; nd["split_provenance"].add(split)
                    if nd["text"] != text:
                        conflicts.append({"title": title, "first": nd["source_record_id"], "other": f"{split}.json:{qi}:{pi}",
                                          "first_sha": sha(nd["text"])[:16], "other_sha": sha(text)[:16],
                                          "ws_equal": ws(nd["text"]) == ws(text)})
            if (qi + 1) % 50000 == 0: log(f"  [2wiki corpus] {split} {qi+1} records, {len(by_title)} titles")
    nodes = list(by_title.values())
    for nd in nodes: nd["split_provenance"] = sorted(nd["split_provenance"], key=["train", "dev", "test"].index)
    acc = {"raw_source_records": raw, "dropped_records": dropped, "duplicates_collapsed": dup,
           "identity": "context paragraph TITLE (exact string); node_id = 2wiki:<sha256(title)[:24]>",
           "identity_text_conflicts": {"n": len(conflicts), "examples": conflicts[:20]},
           "identity_normalization_diagnostics": diag.report(),
           "text_construction": "text = ' '.join(sentences) (official file stores a sentence list)",
           "note": "2wiki_universe (para_with_hyperlink, 5,989,847 articles) is a separate full-universe corpus, OUT OF SCOPE here"}
    return nodes, acc


def build_corpus_musique(S):
    by_key = {}; raw = 0; dropped = []; dup = 0; diag = Diag(); text_only = {}
    for split in ("train", "dev", "test"):
        path = f"{S['base']}/{S['files'][split]}"
        for qi, m in enumerate(iter_jsonl(path)):
            for pi, p in enumerate(m["paragraphs"]):          # ONLY title + paragraph_text are read here
                raw += 1
                title, text = p.get("title", ""), p.get("paragraph_text", "")
                if not isinstance(text, str) or not text.strip():
                    dropped.append({"where": f"{split}:{m['id']}:{pi}", "reason": "empty_paragraph_text"}); continue
                key = title + SEP + text
                nd = by_key.get(key)
                if nd is None:
                    diag.add(key)
                    th = sha(text); text_only.setdefault(th, set()).add(title)
                    by_key[key] = {"node_id": f"musique:{sha(key)[:24]}", "source_id": key.replace(SEP, " || "),
                                   "source_record_id": f"{split}:{m['id']}:{pi}", "split_provenance": {split},
                                   "title": title, "text": text, "source_file": path, "content_hash": sha(key),
                                   "n_source_records": 1}
                else:
                    dup += 1; nd["n_source_records"] += 1; nd["split_provenance"].add(split)
            if (qi + 1) % 5000 == 0: log(f"  [musique corpus] {split} {qi+1} records, {len(by_key)} paragraphs")
    nodes = list(by_key.values())
    for nd in nodes: nd["split_provenance"] = sorted(nd["split_provenance"], key=["train", "dev", "test"].index)
    multi = {h: sorted(t) for h, t in text_only.items() if len(t) > 1}
    acc = {"raw_source_records": raw, "dropped_records": dropped, "duplicates_collapsed": dup,
           "identity": "(title, paragraph_text) exact pair; node_id = musique:<sha256(title+US+text)[:24]>",
           "alternative_identity_text_hash_only": {"n_nodes": len(text_only), "delta_vs_chosen": len(text_only) - len(nodes),
                                                   "texts_under_multiple_titles": multi},
           "identity_normalization_diagnostics": diag.report(),
           "note": "MuSiQue paragraphs carry no global id; a paragraph is identified by its (article title, text). "
                   "Text-hash-only identity would merge paragraphs from different articles that share text."}
    return nodes, acc


def build_corpus_squad(S):
    by_key = {}; raw = 0; dropped = []; dup = 0; diag = Diag(); ctx_only = {}
    for split in ("train", "dev"):
        path = f"{S['base']}/{S['files'][split]}"
        for ai, art in enumerate(iter_json_array_at(path, "data")):
            title = art.get("title", "")
            for pi, p in enumerate(art["paragraphs"]):        # ONLY title + context are read here
                raw += 1
                ctx = p["context"]
                if not ctx.strip():
                    dropped.append({"where": f"{split}:{ai}:{pi}", "reason": "empty_context"}); continue
                key = title + SEP + ctx
                nd = by_key.get(key)
                if nd is None:
                    diag.add(key); ctx_only.setdefault(sha(ctx), set()).add(title)
                    by_key[key] = {"node_id": f"squad:{sha(key)[:24]}", "source_id": key.replace(SEP, " || "),
                                   "source_record_id": f"{split}:{ai}:{pi}", "split_provenance": {split},
                                   "title": title, "text": ctx, "source_file": path, "content_hash": sha(key),
                                   "n_source_records": 1}
                else:
                    dup += 1; nd["n_source_records"] += 1; nd["split_provenance"].add(split)
    nodes = list(by_key.values())
    for nd in nodes: nd["split_provenance"] = sorted(nd["split_provenance"], key=["train", "dev"].index)
    acc = {"raw_source_records": raw, "dropped_records": dropped, "duplicates_collapsed": dup,
           "identity": "(article title, context) exact pair; node_id = squad:<sha256(title+US+context)[:24]>",
           "alternative_identity_context_hash_only": {"n_nodes": len(ctx_only), "delta_vs_chosen": len(ctx_only) - len(nodes)},
           "identity_normalization_diagnostics": diag.report()}
    return nodes, acc


CORPUS = {"metaqa": build_corpus_metaqa, "2wiki": build_corpus_2wiki, "musique": build_corpus_musique, "squad": build_corpus_squad}


def write_nodes(nodes, out):
    nodes.sort(key=lambda n: n["node_id"])
    ids = [n["node_id"] for n in nodes]
    assert len(set(ids)) == len(ids), "node_id collision"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/nodes.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for n in nodes:
            f.write(json.dumps(n, ensure_ascii=False, sort_keys=True) + "\n")
    order_hash = sha("\n".join(ids))
    corpus_hash = sha("\n".join(f"{n['node_id']}\t{n['content_hash']}" for n in nodes))
    return ids, order_hash, corpus_hash


# =====================================================================================  QUERY STAGE

class GoldResolver:
    def __init__(self, nodes, ds):
        self.ds = ds
        if ds == "metaqa":
            self.by_name = {n["source_id"]: n["node_id"] for n in nodes}
            self.by_lower = collections.defaultdict(list)
            for n in nodes: self.by_lower[n["source_id"].strip().lower().replace("_", " ")].append(n["node_id"])
        elif ds == "2wiki":
            self.by_title = {n["title"]: n["node_id"] for n in nodes}
        else:
            self.by_key = {n["title"] + SEP + n["text"]: n["node_id"] for n in nodes}
        self.unresolved = collections.Counter(); self.unresolved_examples = []

    def _miss(self, split, ref):
        self.unresolved[split] += 1
        if len(self.unresolved_examples) < 50: self.unresolved_examples.append({"split": split, "ref": ref})

    def metaqa_answer(self, split, name):
        nid = self.by_name.get(name)
        if nid is None:
            self._miss(split, name)
        return nid

    def title(self, split, t):
        nid = self.by_title.get(t)
        if nid is None: self._miss(split, t)
        return nid

    def key(self, split, title, text):
        nid = self.by_key.get(title + SEP + text)
        if nid is None: self._miss(split, {"title": title, "text_sha": sha(text)[:16]})
        return nid


def qwrite(out, split, recs):
    os.makedirs(f"{out}/queries", exist_ok=True)
    with open(f"{out}/queries/{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def light(rec):
    """Memory-light index entry (full records are re-read from queries/<split>.jsonl when a subset needs them)."""
    return {"split": rec["split"], "question": rec["question"], "gold_refs": rec["gold_refs"], "hop": rec.get("hop")}


def fetch_full(out, qids):
    """Stream queries/*.jsonl and return {query_id: full record} for the requested ids."""
    want = set(qids); got = {}
    for fn in sorted(os.listdir(f"{out}/queries")):
        if not fn.endswith(".jsonl"): continue   # skip queries/lanes/ etc. (build_kb.py already did this)
        for r in iter_jsonl(f"{out}/queries/{fn}"):
            if r["query_id"] in want: got[r["query_id"]] = r
    return got


def build_queries_metaqa(S, nodes, out):
    R = GoldResolver(nodes, "metaqa"); stats = {}; index = {}
    brack = re.compile(r"\[(.+?)\]")
    lower_rescue = 0
    for split in ("train", "dev", "test"):
        recs = []
        for hop in (1, 2, 3):
            qa = f"{S['qa_dir']}/{hop}-hop/vanilla/qa_{split}.txt"; qt = f"{S['qa_dir']}/{hop}-hop/qa_{split}_qtype.txt"
            qtypes = [l.rstrip("\n") for l in open(qt, encoding="utf-8")]
            lines = [l.rstrip("\n") for l in open(qa, encoding="utf-8")]
            for li, line in enumerate(lines):
                if not line.strip() or "\t" not in line: continue
                q, ans = line.split("\t", 1)
                answers = [a for a in ans.split("|") if a != ""]
                gold, unres = [], []                       # PER-REFERENCE: track which refs did not resolve
                for a in answers:
                    nid = R.metaqa_answer(split, a)
                    if nid is None:
                        alt = R.by_lower.get(a.strip().lower().replace("_", " "))
                        if alt: lower_rescue += 1
                        unres.append(a)
                    else: gold.append(nid)
                gold = list(dict.fromkeys(gold))
                m = brack.search(q); topic = m.group(1) if m else None
                qid = f"metaqa:{hop}hop:{split}:{li}"
                recs.append({"query_id": qid, "dataset": "metaqa", "split": split, "hop": hop, "question": q,
                             "question_plain": q.replace("[", "").replace("]", "") , "topic_entity": topic,
                             "topic_entity_node_id": R.by_name.get(topic) if topic else None,
                             "qtype": qtypes[li] if li < len(qtypes) else None,
                             "answers": answers, "gold_node_ids": gold, "gold_refs": answers,
                             "gold_refs_unresolved": unres, "gold_title_ambiguous": [],
                             "source_file": qa, "source_line": li})
                index[qid] = light(recs[-1])
        qwrite(out, split, recs)
        stats[split] = _qstats(recs)
        del recs
    stats["_answers_unresolved_exact_but_case_insensitive_match"] = lower_rescue
    return stats, R, index


def _qstats(recs):
    """PER-REFERENCE gold semantics.  (2026-09-05 patch -- supersedes the cardinality test.)

    OLD, and forbidden:  n_all_gold_resolved counted a question only when
            len(gold_node_ids) == len(set(map(str, gold_refs)))
        which fails a question merely because ONE of its references is ambiguous and legitimately
        expands to more than one canonical node.

    NEW:  each gold_ref is resolved independently and is satisfied iff it maps to >= 1 canonical
        node.  A ref that resolves uniquely from official source context stores exactly that node;
        a GENUINELY ambiguous ref keeps its alternatives and is satisfied DISJUNCTIVELY
        (retrieved & {alternatives} != {}), never conjunctively.

    For metaqa / musique / squad every GoldResolver index is 1:1 by construction (see GoldResolver),
    so no ref can be ambiguous here and this change is semantics-only: the numbers do not move.
    It is applied for a single definition of gold presence across all six datasets.
    """
    n_with = n_all = n_any = 0
    refs_total = refs_resolved = refs_missing = refs_ambiguous = 0
    for r in recs:
        refs = list(dict.fromkeys(map(str, r["gold_refs"])))
        if not refs:
            continue
        n_with += 1
        miss = set(map(str, r.get("gold_refs_unresolved") or []))
        amb = set(map(str, r.get("gold_title_ambiguous") or []))
        m = sum(1 for x in refs if x in miss)
        refs_total += len(refs); refs_missing += m; refs_resolved += len(refs) - m
        refs_ambiguous += sum(1 for x in refs if x in amb)
        if m == 0: n_all += 1
        if r["gold_node_ids"]: n_any += 1
    return {"n": len(recs), "n_with_gold_refs": n_with,
            "n_all_gold_resolved": n_all,          # now: every REF resolved, not a cardinality match
            "n_any_gold_resolved": n_any,
            "gold_refs_total": refs_total,
            "gold_nodes_total": sum(len(r["gold_node_ids"]) for r in recs),
            "GOLD_REFS_TOTAL": refs_total, "GOLD_REFS_RESOLVED": refs_resolved,
            "MISSING_GOLD_REFS": refs_missing, "AMBIGUOUS_GOLD_REFS": refs_ambiguous,
            "ALL_GOLD_REFS_RESOLVE": refs_missing == 0}


def build_queries_2wiki(S, nodes, out):
    R = GoldResolver(nodes, "2wiki"); stats = {}; index = {}
    for split in S["splits"]:
        path = f"{S['base']}/{split}.json"; recs = []
        for qi, q in enumerate(iter_json_array(path)):
            sf = q.get("supporting_facts") or []
            sf_titles = list(dict.fromkeys(s[0] for s in sf))
            pairs = [(t, R.title(split, t)) for t in sf_titles]           # PER-REFERENCE resolution
            gold = [g for _, g in pairs if g]; unres = [t for t, g in pairs if not g]
            ctx_ids = [R.by_title.get(e[0]) for e in q["context"]]
            recs.append({"query_id": q["_id"], "dataset": "2wiki", "split": split, "question": q["question"],
                         "answer": q.get("answer"), "answers": [q["answer"]] if q.get("answer") else [],
                         "type": q.get("type"), "supporting_facts": sf, "evidences": q.get("evidences") or [],
                         "gold_refs": sf_titles, "gold_node_ids": gold, "context_node_ids": ctx_ids,
                         "gold_refs_unresolved": unres, "gold_title_ambiguous": [],
                         "source_file": path, "source_index": qi})
            index[q["_id"]] = light(recs[-1])
            if (qi + 1) % 50000 == 0: log(f"  [2wiki queries] {split} {qi+1}")
        qwrite(out, split, recs); stats[split] = _qstats(recs); del recs
    return stats, R, index


def build_queries_musique(S, nodes, out):
    R = GoldResolver(nodes, "musique"); stats = {}; index = {}
    for split in ("train", "dev", "test"):
        path = f"{S['base']}/{S['files'][split]}"; recs = []
        for qi, m in enumerate(iter_jsonl(path)):
            paras = m["paragraphs"]
            sup = [p for p in paras if p.get("is_supporting")]
            pairs = [({"title": p["title"], "idx": p.get("idx")},                    # PER-REFERENCE resolution
                      R.key(split, p["title"], p["paragraph_text"])) for p in sup]
            gold = [g for _, g in pairs if g]; unres = [ref for ref, g in pairs if not g]
            hop = m["id"].split("hop")[0] + "hop" if "hop" in m["id"] else None
            recs.append({"query_id": m["id"], "dataset": "musique", "split": split, "hop": hop, "question": m["question"],
                         "answer": m.get("answer"), "answer_aliases": m.get("answer_aliases") or [],
                         "answers": list(dict.fromkeys(([m["answer"]] if m.get("answer") else []) + list(m.get("answer_aliases") or []))),
                         "answerable": m.get("answerable"),
                         "gold_refs": [{"title": p["title"], "idx": p.get("idx")} for p in sup],
                         "gold_node_ids": gold,
                         "gold_refs_unresolved": unres, "gold_title_ambiguous": [],
                         "context_node_ids": [R.by_key.get(p["title"] + SEP + p["paragraph_text"]) for p in paras],
                         "question_decomposition": m.get("question_decomposition") or [],
                         "source_file": path, "source_index": qi})
            index[m["id"]] = light(recs[-1])
        qwrite(out, split, recs); stats[split] = _qstats(recs); del recs
    # auxiliary official single-hop questions (dev/test); query-only, no corpus effect
    sh = json.load(open(f"{S['base']}/{S['singlehop']}", encoding="utf-8"))
    stats["singlehop_dev_test_aux"] = {"n": len(sh), "file": S["singlehop"], "note": "kept as official aux file reference only; not tabled"}
    return stats, R, index


def build_queries_squad(S, nodes, out):
    R = GoldResolver(nodes, "squad"); stats = {}; index = {}
    for split in ("train", "dev"):
        path = f"{S['base']}/{S['files'][split]}"; recs = []; order = 0
        for ai, art in enumerate(iter_json_array_at(path, "data")):
            title = art.get("title", "")
            for pi, p in enumerate(art["paragraphs"]):
                nid = R.key(split, title, p["context"])
                for qa in p["qas"]:
                    imp = bool(qa.get("is_impossible", False))
                    spans = [{"text": a["text"], "answer_start": a["answer_start"]} for a in qa.get("answers", [])]
                    recs.append({"query_id": qa["id"], "dataset": "squad", "split": split, "question": qa["question"],
                                 "answers": list(dict.fromkeys(a["text"] for a in qa.get("answers", []))), "answer_spans": spans,
                                 "plausible_answers": [a["text"] for a in qa.get("plausible_answers", [])],
                                 "is_impossible": imp, "answerable": not imp, "title": title,
                                 "gold_refs": [f"{ai}:{pi}"], "gold_node_ids": [nid] if nid else [],
                                 "gold_refs_unresolved": ([] if nid else [f"{ai}:{pi}"]),   # PER-REFERENCE
                                 "gold_title_ambiguous": [],
                                 "source_file": path, "source_index": order})
                    index[qa["id"]] = light(recs[-1]); order += 1
        qwrite(out, split, recs); stats[split] = _qstats(recs); del recs
    return stats, R, index


QUERIES = {"metaqa": build_queries_metaqa, "2wiki": build_queries_2wiki, "musique": build_queries_musique, "squad": build_queries_squad}


# =====================================================================================  EVAL SUBSET

def legacy_subset(ds, index):
    """Recover the frozen legacy eval subset (cache rows -> legacy question ids -> official query ids)."""
    L = LEGACY_NAME[ds]; base = f"{WORK}/legacy/{L}"
    rows = json.load(open(f"{base}/legacy_eval_rows.json", encoding="utf-8"))
    lq = {r["node_id"]: r for r in iter_jsonl(f"{base}/questions.jsonl")}
    recs = []; misses = []; method = None
    if ds == "metaqa":
        method = "legacy id metaqa_q_{hop}hop_{split}_{global_counter}; counter reproduced over vanilla qa files in (hop 1..3) x (train,dev,test) order, skipping blank/no-tab lines; question text verified equal"
        counter = 0; want = {i: r for r, i in zip(rows["rows"], rows["ids"])}; hit = {}
        for hop in (1, 2, 3):
            for split in ("train", "dev", "test"):
                for li, line in enumerate(open(f"{SRC['metaqa']['qa_dir']}/{hop}-hop/vanilla/qa_{split}.txt", encoding="utf-8")):
                    s = line.strip()
                    if not s or "\t" not in s: continue
                    nid = f"metaqa_q_{hop}hop_{split}_{counter}"; counter += 1
                    if nid in want: hit[nid] = f"metaqa:{hop}hop:{split}:{li}"
        for lid, row in zip(rows["ids"], rows["rows"]):
            qid = hit.get(lid); node = lq.get(lid)
            if qid is None or qid not in index or (node and index[qid]["question"] != node["content"]):
                misses.append({"legacy_id": lid, "reason": "counter_or_text_mismatch"}); continue
            recs.append((row, lid, qid, node))
    elif ds == "squad":
        method = "legacy id squad_clean_q_{counter}; counter reproduced over official train-v2.0.json qas in file order; question text verified equal"
        want = set(rows["ids"]); hit = {}
        c = 0
        for art in iter_json_array_at(f"{SRC['squad']['base']}/train-v2.0.json", "data"):
            for p in art["paragraphs"]:
                for qa in p["qas"]:
                    nid = f"squad_clean_q_{c}"; c += 1
                    if nid in want: hit[nid] = qa["id"]
        for lid, row in zip(rows["ids"], rows["rows"]):
            qid = hit.get(lid); node = lq.get(lid)
            if qid is None or qid not in index or (node and index[qid]["question"] != node["content"]):
                misses.append({"legacy_id": lid, "reason": "counter_or_text_mismatch"}); continue
            recs.append((row, lid, qid, node))
    elif ds == "musique":
        method = "legacy id musique_clean_q_train_{i} (FlashRAG train_i) = official musique_ans_v1.0_train.jsonl line i; question text verified equal"
        train_ids = [m["id"] for m in iter_jsonl(f"{SRC['musique']['base']}/{SRC['musique']['files']['train']}")]
        for lid, row in zip(rows["ids"], rows["rows"]):
            m = re.match(r"musique_clean_q_train_(\d+)$", lid); node = lq.get(lid)
            i = int(m.group(1)) if m else -1
            qid = train_ids[i] if 0 <= i < len(train_ids) else None
            if qid is None or qid not in index or (node and index[qid]["question"] != node["content"]):
                misses.append({"legacy_id": lid, "reason": "index_or_text_mismatch"}); continue
            recs.append((row, lid, qid, node))
    elif ds == "2wiki":
        method = "legacy 2wiki_clean questions came from the FlashRAG 15k train sample (raw file no longer present); matched to official train/dev/test by EXACT question text; ambiguities resolved by supporting-fact title set == legacy gold title set"
        ldocs = {}
        for r in iter_jsonl(f"{base}/docs.jsonl"): ldocs[r["node_id"]] = r["title"]
        by_text = collections.defaultdict(list)
        for qid, r in index.items(): by_text[r["question"]].append(qid)
        for lid, row in zip(rows["ids"], rows["rows"]):
            node = lq.get(lid)
            if node is None: misses.append({"legacy_id": lid, "reason": "legacy_question_missing"}); continue
            cands = by_text.get(node["content"], [])
            gold_titles = sorted({ldocs.get(g, "") for g in node["neighbors"]})
            if len(cands) > 1:
                cands2 = [c for c in cands if sorted(set(index[c]["gold_refs"])) == gold_titles]
                if len(cands2) == 1: cands = cands2
            if len(cands) != 1:
                misses.append({"legacy_id": lid, "reason": "no_match" if not cands else "ambiguous", "n_candidates": len(cands)}); continue
            recs.append((row, lid, cands[0], node))
    full = fetch_full(out_dir_for_fetch[0], [qid for _, _, qid, _ in recs])
    out = []
    for row, lid, qid, node in recs:
        r = dict(full[qid]); r["legacy_query_id"] = lid; r["legacy_cache_row"] = row
        r["legacy_gold_ids"] = node["neighbors"] if node else None
        out.append(r)
    return out, {"method": method, "n_legacy": len(rows["ids"]), "n_recovered": len(out), "n_missed": len(misses), "misses": misses[:50],
                 "official_split_distribution": dict(collections.Counter(r["split"] for r in out))}


out_dir_for_fetch = [None]   # set in main() before any subset function runs (subset stage only; corpus never sees it)


def random_dev_subset(ds, index, seed, n=2000):
    rng = random.Random(seed)
    dev = [qid for qid, r in index.items() if r["split"] == "dev"]
    dev.sort()
    if ds == "metaqa":
        per = n // 3; sel = []
        for hop in (1, 2, 3):
            b = [q for q in dev if index[q]["hop"] == hop]; rng.shuffle(b); sel += b[:per]
    else:
        rng.shuffle(dev); sel = dev[:n]
    sel.sort()
    full = fetch_full(out_dir_for_fetch[0], sel)
    return [full[q] for q in sel], {"method": f"seeded random from official dev (seed={seed}, n={len(sel)}, metaqa hop-balanced)"}


def path_subset(path, index):
    ids = json.load(open(path, encoding="utf-8")) if path.endswith(".json") else [json.loads(l)["query_id"] for l in open(path, encoding="utf-8") if l.strip()]
    ids = [q for q in ids if q in index]
    full = fetch_full(out_dir_for_fetch[0], ids)
    return [full[q] for q in ids], {"method": f"query ids from {path}", "n_requested": len(ids), "n_found": len(full)}


# =====================================================================================  MAIN

def load_frozen_nodes(out):
    """--queries-only: reload the FROZEN nodes.jsonl instead of rebuilding the corpus.

    Returns the same (nodes, acc, ids, order_hash, corpus_hash, nodes_sha) the corpus stage would
    have produced, but read-only: nodes.jsonl is never opened for writing.  The recomputed count and
    hashes are ASSERTED against integrity_report.json, and the corpus accounting block is quoted
    forward from it because the corpus stage did not rerun.
    """
    prev_path = f"{out}/integrity_report.json"
    if not os.path.exists(prev_path):
        raise SystemExit(f"--queries-only needs an existing {prev_path} to assert against")
    prev = json.load(open(prev_path, encoding="utf-8"))
    nodes = list(iter_jsonl(f"{out}/nodes.jsonl"))
    ids = [n["node_id"] for n in nodes]
    order_hash = sha("\n".join(ids))
    corpus_hash = sha("\n".join(f"{n['node_id']}\t{n['content_hash']}" for n in nodes))
    nodes_sha = file_sha(f"{out}/nodes.jsonl")
    for k, got in (("canonical_node_count", len(ids)), ("NODE_ORDER_HASH", order_hash),
                   ("CORPUS_HASH", corpus_hash), ("nodes_jsonl_sha256", nodes_sha)):
        assert prev.get(k) == got, f"--queries-only: frozen corpus {k} changed: {prev.get(k)} -> {got}"
    log(f"[build] --queries-only: frozen corpus VERIFIED n={len(ids)} CORPUS_HASH={corpus_hash[:12]} "
        f"NODE_ORDER_HASH={order_hash[:12]} (nodes.jsonl untouched)")
    return nodes, prev["corpus_accounting_detail"], ids, order_hash, corpus_hash, nodes_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", choices=list(SRC))
    ap.add_argument("--eval-subset", default="legacy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None, help="name suffix for the eval subset file (default: legacy->eval_<N>; random_dev->eval_<N>_random_dev_s<seed>)")
    ap.add_argument("--queries-only", action="store_true",
                    help="rerun ONLY the query/gold, eval-subset and integrity stages against the FROZEN "
                         "nodes.jsonl. The corpus stage is skipped and nodes.jsonl is opened READ-ONLY; "
                         "its count and hashes are re-derived and asserted unchanged.")
    a = ap.parse_args()
    ds = a.ds; S = SRC[ds]; out = a.out or f"{ROOT}/{ds}"
    t0 = time.time(); started = time.strftime("%Y-%m-%dT%H:%M:%S")
    cmd = "python " + " ".join(sys.argv)
    log(f"[build {ds}] START out={out} eval_subset={a.eval_subset}")

    # ---- corpus (query-independent by construction) ----
    if a.queries_only:
        nodes, acc, ids, order_hash, corpus_hash, nodes_sha = load_frozen_nodes(out)
    else:
        nodes, acc = CORPUS[ds](S)
        ids, order_hash, corpus_hash = write_nodes(nodes, out)
        nodes_sha = file_sha(f"{out}/nodes.jsonl")
    log(f"[build {ds}] corpus: raw={acc['raw_source_records']} nodes={len(ids)} dup={acc['duplicates_collapsed']} "
        f"dropped={len(acc['dropped_records'])} corpus_hash={corpus_hash[:12]} order_hash={order_hash[:12]} ({time.time()-t0:.0f}s)")
    n_empty_text = sum(1 for n in nodes if not n["text"].strip()); n_empty_title = sum(1 for n in nodes if not str(n["title"]).strip())
    assert acc["raw_source_records"] == len(ids) + acc["duplicates_collapsed"] + len(acc["dropped_records"]), "accounting mismatch"

    # ---- queries (index into the frozen corpus) ----
    qstats, R, index = QUERIES[ds](S, nodes, out)
    log(f"[build {ds}] queries: " + ", ".join(f"{k}={v['n']}" for k, v in qstats.items() if isinstance(v, dict) and "n" in v) +
        f" unresolved_gold_refs={dict(R.unresolved)} ({time.time()-t0:.0f}s)")

    # ---- eval subset (never touches the corpus; nodes.jsonl already written) ----
    ev_info = None; ev_file = None
    out_dir_for_fetch[0] = out
    del nodes  # corpus objects are no longer needed; frees memory before the subset stage
    if a.eval_subset != "none":
        if a.eval_subset == "legacy":
            recs, ev_info = legacy_subset(ds, index); tag = a.tag or f"eval_{len(recs)}"
        elif a.eval_subset.startswith("random_dev:"):
            seed = int(a.eval_subset.split(":")[1]); n = 1998 if ds == "metaqa" else 2000
            recs, ev_info = random_dev_subset(ds, index, seed, n); tag = a.tag or f"eval_{len(recs)}_random_dev_s{seed}"
        else:
            recs, ev_info = path_subset(a.eval_subset, index); tag = a.tag or f"eval_{len(recs)}_custom"
        ev_file = f"{out}/{tag}.jsonl"
        with open(ev_file, "w", encoding="utf-8", newline="\n") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        ev_info["file"] = ev_file; ev_info["n"] = len(recs)
        ev_st = _qstats(recs)                                   # PER-REFERENCE, same rule as the splits
        ev_info["n_all_gold_resolved"] = ev_st["n_all_gold_resolved"]
        ev_info["n_with_gold_refs"] = ev_st["n_with_gold_refs"]
        ev_info["GOLD_REFS_TOTAL"] = ev_st["GOLD_REFS_TOTAL"]
        ev_info["MISSING_GOLD_REFS"] = ev_st["MISSING_GOLD_REFS"]
        ev_info["AMBIGUOUS_GOLD_REFS"] = ev_st["AMBIGUOUS_GOLD_REFS"]
        ev_info["ALL_GOLD_REFS_RESOLVE"] = ev_st["ALL_GOLD_REFS_RESOLVE"]
        log(f"[build {ds}] eval subset: {ev_info.get('method','')[:60]}... n={len(recs)} all_gold_resolved={ev_info['n_all_gold_resolved']}")

    # ---- integrity report ----
    splits_with_gold = [s for s, v in qstats.items() if isinstance(v, dict) and v.get("n_with_gold_refs")]
    all_gold_present = all(qstats[s]["ALL_GOLD_REFS_RESOLVE"] for s in splits_with_gold)
    gold_semantics = {
        "_semantics": ("gold ref -> {canonical node,...}; a ref resolves iff it maps to >=1 node; "
                       "ALL_GOLD_REFS_RESOLVE = every ref in every gold-bearing split resolves; ambiguous refs "
                       "are DISJUNCTIVE (retrieved & {alternatives} != {}), never conjunctive."),
        "_supersedes": ("the cardinality test len(gold_node_ids)==len(set(gold_refs)), which failed a question "
                        "merely because one of its refs legitimately expanded to >1 canonical node."),
        "ALL_GOLD_REFS_RESOLVE": bool(all_gold_present),
        "GOLD_REFS_TOTAL": sum(qstats[s]["GOLD_REFS_TOTAL"] for s in splits_with_gold),
        "MISSING_GOLD_REFS": sum(qstats[s]["MISSING_GOLD_REFS"] for s in splits_with_gold),
        "AMBIGUOUS_GOLD_REFS": sum(qstats[s]["AMBIGUOUS_GOLD_REFS"] for s in splits_with_gold),
        "per_split": {s: {k: qstats[s][k] for k in
                          ("GOLD_REFS_TOTAL", "GOLD_REFS_RESOLVED", "MISSING_GOLD_REFS",
                           "AMBIGUOUS_GOLD_REFS", "ALL_GOLD_REFS_RESOLVE")} for s in splits_with_gold},
        "_index_cardinality": ("GoldResolver builds a 1:1 ref->node index for every dataset in this builder, "
                               "so AMBIGUOUS_GOLD_REFS is 0 by construction here."),
    }
    integrity = {
        "dataset": ds, "dataset_version": VERSION, "built_at": started,
        "unique_node_ids": len(set(ids)) == len(ids), "canonical_node_count": len(ids),
        "raw_source_records": acc["raw_source_records"], "duplicates_collapsed": acc["duplicates_collapsed"],
        "dropped_records": len(acc["dropped_records"]), "drop_reasons": dict(collections.Counter(d["reason"] for d in acc["dropped_records"])),
        "accounting_identity": "RAW_SOURCE_RECORDS == CANONICAL_NODES + DUPLICATES_COLLAPSED + DROPPED_RECORDS (asserted)",
        "empty_text_nodes": n_empty_text, "empty_title_nodes": n_empty_title,
        "corpus_accounting_detail": acc,
        "query_stats": qstats, "splits_with_gold": splits_with_gold,
        "ALL_EVAL_GOLDS_PRESENT": bool(all_gold_present and (ev_info is None or ev_info["ALL_GOLD_REFS_RESOLVE"])),
        "GOLD_SEMANTICS": gold_semantics,
        "ALL_GOLD_REFS_RESOLVE": gold_semantics["ALL_GOLD_REFS_RESOLVE"],
        "MISSING_GOLD_REFS": gold_semantics["MISSING_GOLD_REFS"],
        "AMBIGUOUS_GOLD_REFS": gold_semantics["AMBIGUOUS_GOLD_REFS"],
        "unresolved_gold_refs": {"counts": dict(R.unresolved), "examples": R.unresolved_examples},
        "eval_subset": ev_info,
        "CORPUS_HASH": corpus_hash, "NODE_ORDER_HASH": order_hash, "nodes_jsonl_sha256": nodes_sha,
    }
    json.dump(integrity, open(f"{out}/integrity_report.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---- build info ----
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = None
    info = {"dataset": ds, "dataset_version": VERSION, "command": cmd, "builder": os.path.relpath(os.path.abspath(__file__)).replace("\\", "/"),
            "builder_sha256": file_sha(os.path.abspath(__file__)), "jsonstream_sha256": file_sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jsonstream.py")),
            "git_commit": commit, "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
            "peak_rss_mb": peak_rss_mb(), "eval_subset_arg": a.eval_subset, "out": out, "queries_only": bool(a.queries_only),
            "CORPUS_HASH": corpus_hash, "NODE_ORDER_HASH": order_hash, "nodes_jsonl_sha256": nodes_sha, "canonical_node_count": len(ids),
            "source_files": {p: {"sha256": file_sha(p), "bytes": os.path.getsize(p)} for p in _source_paths(ds)}}
    json.dump(info, open(f"{out}/build_info.json", "w", encoding="utf-8"), indent=2)
    log(f"[build {ds}] DONE nodes={len(ids)} CORPUS_HASH={corpus_hash[:12]} peak_rss_mb={info['peak_rss_mb']} ({info['seconds']}s)")


def _source_paths(ds):
    S = SRC[ds]
    if ds == "metaqa":
        ps = [S["dict"], S["kb"]]
        for hop in (1, 2, 3):
            for split in ("train", "dev", "test"):
                ps += [f"{S['qa_dir']}/{hop}-hop/vanilla/qa_{split}.txt", f"{S['qa_dir']}/{hop}-hop/qa_{split}_qtype.txt"]
        return ps
    if ds == "2wiki": return [f"{S['base']}/{s}.json" for s in S["splits"]]
    if ds == "musique": return [f"{S['base']}/{f}" for f in S["files"].values()] + [f"{S['base']}/{S['singlehop']}"]
    if ds == "squad": return [f"{S['base']}/{f}" for f in S["files"].values()]


if __name__ == "__main__":
    main()
