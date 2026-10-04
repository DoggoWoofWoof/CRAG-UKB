"""FINAL canonical builder for the big streaming corpora (Track B, canonical_v1): 2wiki + hotpotqa (+ webqsp, BLOCKED).

    python scratchpad/final_canonical_build/build_kb.py <ds> [--eval-subset legacy|none|random_held:<seed>|PATH]
                                                             [--out DIR] [--tag NAME] [--plan]

ds in {2wiki, webqsp, hotpotqa}.  Default --out = data/final_canonical/<ds>

WHY THIS IS A SEPARATE FILE FROM build.py
    build.py's sha256 is recorded inside the four ALREADY-ACCEPTED dataset manifests, so the two new corpora got
    their own module rather than editing it.  (2026-09-05: build.py was subsequently UNFROZEN by explicit user
    decision for the per-reference gold-semantics fix; its sha is no longer quoted here, it is read from
    build_info.json / MANIFEST.json so this comment cannot go stale.)
    Every contract build.py defines is reproduced here byte-for-byte:
      * node record schema      node_id, source_id, source_record_id, split_provenance, title, text,
                                source_file, content_hash, n_source_records (+ per-dataset extras)
      * node_id                 "<ds>:<sha256(identity_key)[:24]>"
      * content_hash            sha256(identity_key + US + text)      (US = \\x1f)
      * file order              lexicographic node_id
      * NODE_ORDER_HASH         sha256("\\n".join(node_ids))
      * CORPUS_HASH             sha256("\\n".join(f"{node_id}\\t{content_hash}"))
      * outputs                 nodes.jsonl, queries/<split>.jsonl, eval_<N>.jsonl,
                                integrity_report.json, build_info.json

INVARIANT (same as build.py, enforced structurally)
    QUERY SUBSETTING IS ALLOWED.  CORPUS SUBSETTING BY QUERY IS FORBIDDEN.
    build_corpus_*() receives ONLY official corpus source paths and never sees --eval-subset.
    webqsp additionally enforces this at the *column* level: the corpus pass opens the RoG parquets with
    columns=["graph"], so a_entity / q_entity / answer are not even materialised in memory.  That is the exact
    defect canonical_v1 removes from the Phase-C reference (scratchpad/c1c2_webqsp.py:32-33 injected the answer
    and topic entities into the corpus).

NORMALIZATION (per the approved SOURCE_CONTRACTs -- verbatim, nothing silently applied)
    2wiki     text = " ".join(official 'sentences' list) of a para_with_hyperlink article.  NO .strip() -- the
              already-encoded Phase-C universe (scratchpad/build_2wiki_universe.py:42) DID strip, so this build
              COUNTS how many records differ under that single step (that count bounds the re-encode requirement).
              Empty articles are KEPT.  This is exactly build.py's 2wiki text rule, applied to the FULL universe
              instead of the pooled question-context union.
              TEXTUALIZATION_REV 2 (2026-09-06): where that join is EMPTY, the record's own 'title' is used as the
              canonical text -- the SAME general source-level rule hotpotqa uses, not a second per-dataset policy.
              Fallback only: a record that already yields text is untouched and no title is prepended anywhere else.
              Forensics behind it (results/data_audit/final_canonical_v1/2wiki/EMPTY_TEXT_AUDIT.md): all 87,765
              empty-body records have 'sentences' == [] and a NON-EMPTY unique title, the schema holds no other text
              field ({id, title, sentences, mentions}), so TRULY_TEXTLESS = 0 and the title is the only source-native
              candidate.  20,524 of them are hyperlink targets, so leaving them as "" -- which gte-Qwen2 tokenizes to
              the ZERO-LENGTH sequence [] -- would push 87,765 degenerate points into the KNN/H4 structures built over
              this corpus.  This changes CORPUS_HASH (81fa7d1a5d4bbb24 -> see build_info) but NOT membership, node
              identity or node order; the user accepted that trade on 2026-09-06 rather than lock the defect into a
              5,989,847-node graph.
    webqsp    BLOCKED.  See data/final_canonical/_APPROVALS.json -> approvals.webqsp.
    hotpotqa  text = " ".join(non-empty sentences of the official 'text' list).  NO .strip() -- Phase-C
              (scratchpad/c1_hotpot_fullwiki.py:39) did strip, so this build COUNTS how many records differ under
              that single step and reports it (that count is exactly the dense/SPLADE re-encode requirement).
              Empty abstracts are KEPT (dropping them would make membership depend on a text-quality filter).
              TEXTUALIZATION_REV 2 (2026-09-05): where that join is EMPTY, the record's own 'title' is used as the
              canonical text.  Fallback only -- a record that already yields text is untouched, and no title is
              prepended anywhere else.  Motivation: "" tokenizes to the ZERO-LENGTH sequence [] under gte-Qwen2, so
              all such nodes previously collapsed onto one degenerate forward pass.  'text_with_links' is NOT used
              as the fallback: for every one of these records its readable residue after tag-stripping is empty
              (0/94 measured) and the source's own charoffset is [[]].  This changes CORPUS_HASH but NOT membership,
              node identity or node order.  2wiki received the SAME rule one day later (see above) -- the rule is
              general and source-level, so it is stated once and applied wherever a source has that shape.

NODE IDS
    2wiki     "2wiki:c<curid>"       readable, source-stable (the official 'id' of a para_with_hyperlink record)
    hotpotqa  "hotpotqa:c<curid>"    readable, source-stable (user decision, _APPROVALS.json)
    Neither is hashed: curid is already a source-stable identity, and hashing it only hurts debugging/remapping.

MEMORY
    2wiki     external merge sort, same machinery as hotpotqa; the query stage holds only the titles the official
              questions mention (~4e5), never the 6e6-title corpus index.
    hotpotqa  external merge sort: the corpus pass never holds more than SPILL node records; run files are merged
              with heapq.merge and deleted.  Peak RSS is independent of the 5.2M corpus size.
    No stage loads a whole parquet file, the whole tarball or the whole zip member.

GATES -- per-dataset, read from data/final_canonical/_APPROVALS.json (the all-or-nothing
_APPROVED_SOURCE_CONTRACTS file is GONE and must not be created):
    approvals.<ds>.status starting with "APPROVED"  -> may build
    approvals.webqsp.status BLOCKED_PENDING_...     -> REFUSED unconditionally; no flag can override it
    data/final_canonical/_GATE_HOTPOT_HEAVY_OK      -> additionally required for the two heavy corpora
                                                      (hotpotqa AND the full 2wiki universe build)
"""
import sys, os, json, hashlib, unicodedata, time, argparse, random, subprocess, collections, heapq, tarfile, bz2, shutil, glob, re, zipfile, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jsonstream import iter_jsonl, iter_json_array

VERSION = "canonical_v1"
ROOT = "data/final_canonical"
WORK = f"{ROOT}/_work"
SEP = "\x1f"
SPILL = 120_000          # external-sort run size (records); peak RSS is O(SPILL), not O(corpus)
HEAVY = {"hotpotqa", "2wiki"}   # corpora gated on _GATE_HOTPOT_HEAVY_OK (machine capacity)

SRC = {
    "2wiki": {
        "zip": "data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip",
        "member": "para_with_hyperlink.jsonl",
        "base": "data/original/2wiki/v1.0_ids_april2021",
        "splits": ["train", "dev", "test"],
        "held_out_split": "dev",      # official 2wiki TEST answers are HIDDEN
    },
    "webqsp": {
        "rog": [("train", "data/original/webqsp/rog_webqsp/train-00000-of-00002.parquet"),
                ("train", "data/original/webqsp/rog_webqsp/train-00001-of-00002.parquet"),
                ("validation", "data/original/webqsp/rog_webqsp/validation-00000-of-00001.parquet"),
                ("test", "data/original/webqsp/rog_webqsp/test-00000-of-00002.parquet"),
                ("test", "data/original/webqsp/rog_webqsp/test-00001-of-00002.parquet")],
        "official": {"train": "data/original/webqsp/WebQSP/data/WebQSP.train.json",
                     "test": "data/original/webqsp/WebQSP/data/WebQSP.test.json"},
        "held_out_split": "test",     # official MS test labels are PUBLIC (QUALITY_LOCKED lane)
    },
    "hotpotqa": {
        "tarball": "data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
        "queries": [("train", "data/original/hotpotqa/hf_distractor/train-00000-of-00002.parquet"),
                    ("train", "data/original/hotpotqa/hf_distractor/train-00001-of-00002.parquet"),
                    ("validation", "data/original/hotpotqa/hf_fullwiki/validation-00000-of-00001.parquet"),
                    ("test", "data/original/hotpotqa/hf_fullwiki/test-00000-of-00001.parquet")],
        "held_out_split": "validation",   # fullwiki dev; fullwiki TEST labels are hidden
    },
}
LEGACY_NAME = {"webqsp": "webqsp", "hotpotqa": "hotpotqa_clean", "2wiki": "2wiki_clean"}
LEGACY_QID_PREFIX = {"webqsp": "webqsp_q_", "hotpotqa": "hotpot_q_"}   # 2wiki's legacy ids do NOT embed the official id


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
    """Counts how many extra identity collapses lossy normalizations WOULD cause (never applied).
    Sampled for hotpotqa (curid identities never collide under normalization anyway); exact for webqsp."""
    def __init__(self, cap=None):
        self.cap = cap; self.n_seen = 0
        self.exact = set(); self.nfc_ = set(); self.ws_ = set(); self.cf = set()

    def add(self, key):
        if self.cap is not None and self.n_seen >= self.cap: return
        self.n_seen += 1
        self.exact.add(key); self.nfc_.add(nfc(key)); self.ws_.add(ws(key)); self.cf.add(ws(nfc(key)).casefold())

    def report(self):
        n = len(self.exact)
        r = {"exact_identities": n, "extra_collapses_if_NFC": n - len(self.nfc_),
             "extra_collapses_if_whitespace_collapse": n - len(self.ws_),
             "extra_collapses_if_NFC+ws+casefold": n - len(self.cf)}
        if self.cap is not None: r["sampled_first_n_identities"] = self.cap
        return r


# =====================================================================================  NODE WRITER
# Streams pre-sorted node dicts to nodes.jsonl while computing exactly the hashes build.py computes over the
# in-memory list ( sha("\n".join(...)) ), incrementally, so peak RSS is O(1) in the node count.

class NodeWriter:
    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.f = open(f"{out}/nodes.jsonl", "w", encoding="utf-8", newline="\n")
        self.h_ids = hashlib.sha256(); self.h_corpus = hashlib.sha256()
        self.n = 0; self.prev = None
        self.empty_text = 0; self.empty_title = 0

    def write(self, nd):
        nid = nd["node_id"]
        assert self.prev is None or nid > self.prev, f"nodes not strictly ordered: {self.prev!r} then {nid!r}"
        self.prev = nid
        pre = b"" if self.n == 0 else b"\n"
        self.h_ids.update(pre + nid.encode("utf-8"))
        self.h_corpus.update(pre + f"{nid}\t{nd['content_hash']}".encode("utf-8"))
        self.f.write(json.dumps(nd, ensure_ascii=False, sort_keys=True) + "\n")
        self.n += 1
        if not str(nd["text"]).strip(): self.empty_text += 1
        if not str(nd["title"]).strip(): self.empty_title += 1

    def close(self):
        self.f.close()
        return self.n, self.h_ids.hexdigest(), self.h_corpus.hexdigest()


# =====================================================================================  EXTERNAL SORT
# Shared by the two 5-6M-record corpora.  Records are spilled to sorted run files and merged with heapq.merge,
# so peak RSS is O(SPILL) and independent of the corpus size.  Emission is strictly lexicographic node_id order,
# byte-identical to what build.py::write_nodes would produce from the same node dicts held in memory.

class ExtSort:
    def __init__(self, tmp):
        if os.path.isdir(tmp): shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        self.tmp = tmp; self.buf = []; self.runs = []

    def add(self, nid, nd):
        self.buf.append((nid, json.dumps(nd, ensure_ascii=False, sort_keys=True)))
        if len(self.buf) >= SPILL: self.flush()

    def flush(self):
        if not self.buf: return
        self.buf.sort(key=lambda p: p[0])
        p = f"{self.tmp}/run_{len(self.runs):04d}.txt"
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for nid, line in self.buf: f.write(nid + "\t" + line + "\n")
        self.runs.append(p); self.buf.clear()

    @staticmethod
    def _read(p):
        with open(p, encoding="utf-8") as f:
            for l in f:
                nid, _, js = l.rstrip("\n").partition("\t")
                yield nid, js

    def merge_into(self, W):
        """Writes deduplicated, ordered nodes into NodeWriter W. Returns (n_duplicates, duplicate_examples)."""
        self.flush()
        dup = 0; dup_examples = []
        pend_id = pend_js = None; pend_dup = 0
        for nid, js in heapq.merge(*[self._read(p) for p in self.runs], key=lambda x: x[0]):
            if nid == pend_id:
                dup += 1; pend_dup += 1
                if len(dup_examples) < 20: dup_examples.append(nid)
                continue
            if pend_id is not None: W.write(_finish(pend_js, pend_dup))
            pend_id, pend_js, pend_dup = nid, js, 0
        if pend_id is not None: W.write(_finish(pend_js, pend_dup))
        shutil.rmtree(self.tmp, ignore_errors=True)
        return dup, dup_examples


# =====================================================================================  CORPUS: 2WIKI (FULL)

def build_corpus_2wiki(S, out):
    """The FULL official 2Wiki article universe: every record of para_with_hyperlink.jsonl inside its zip.

    This REPLACES the pooled question-context union (398,354 nodes, corpus_hash ec7fbbe8cd783040), which the user
    REJECTED on 2026-09-05 (_APPROVALS.json approvals.2wiki).  The official release describes this archive as all
    articles/paragraphs with hyperlink information except error paragraphs, i.e. the complete retrieval corpus.

    Streamed: the zip member is decompressed line by line; nothing but the sort buffer is held.
    NO question file is opened here -- the corpus is a pure function of the archive.
    """
    ZP, MEM = S["zip"], S["member"]
    es = ExtSort(f"{WORK}/2wiki_runs")
    raw = 0; dropped = []; empty_text = 0; strip_delta = 0; strip_examples = []
    title_fallback = 0; still_empty = 0
    n_sent_total = 0; n_mentions = 0; diag = Diag(cap=200_000)
    t0 = time.time()
    z = zipfile.ZipFile(ZP)
    with z.open(MEM) as fh:
        for li, rawline in enumerate(io.TextIOWrapper(fh, encoding="utf-8", newline="")):
            if not rawline.strip(): continue
            raw += 1
            try:
                r = json.loads(rawline)
            except Exception:
                dropped.append({"where": f"{MEM}:{li}", "reason": "unparsable_json_line"}); continue
            cid = r.get("id")
            if cid is None or str(cid) == "":
                dropped.append({"where": f"{MEM}:{li}", "reason": "missing_curid"}); continue
            cid = str(cid)
            sents = r.get("sentences") or []
            if isinstance(sents, str): sents = [sents]
            body = " ".join(sents)                     # NO .strip() -- see module docstring
            if body != body.strip():
                strip_delta += 1
                if len(strip_examples) < 20:
                    strip_examples.append({"curid": cid, "title": r.get("title", ""), "len": len(body),
                                           "len_stripped": len(body.strip())})
            if not body.strip():
                empty_text += 1
                # TEXTUALIZATION FALLBACK (contract revision 2026-09-06, TEXTUALIZATION_REV 2).  Identical rule to
                # hotpotqa's (build_corpus_hotpotqa below) -- one general source-level rule, not a per-dataset policy:
                # a record whose body is empty but whose SOURCE carries a non-empty title is textualized as that title.
                # Source-native (same JSON line as 'sentences'), deterministic, query-independent -- the
                # para_with_hyperlink schema is exactly {id, title, sentences, mentions} and holds no question, answer
                # or supporting_facts field, so no query artefact can reach the text.  It is also the ONLY candidate:
                # the forensic scan measured ALT_TEXT_NONEMPTY = 0 and TRULY_TEXTLESS = 0 over all 87,765 hits.
                # FALLBACK ONLY: records that already produce non-empty text are untouched, and no title is prepended
                # anywhere else in the corpus.
                tfb = r.get("title") or ""
                if isinstance(tfb, str) and tfb.strip():
                    body = tfb
                    title_fallback += 1
                else:
                    still_empty += 1
            n_sent_total += len(sents); n_mentions += len(r.get("mentions") or [])
            diag.add(cid)
            nid = f"2wiki:c{cid}"
            es.add(nid, {"node_id": nid, "source_id": cid, "source_record_id": f"{MEM}:{li}",
                         "split_provenance": ["full_hyperlink_universe"], "title": r.get("title", ""), "text": body,
                         "source_file": ZP, "content_hash": sha(cid + SEP + body), "n_source_records": 1,
                         "n_sentences": len(sents), "n_mentions": len(r.get("mentions") or [])})
            if raw % 1_000_000 == 0:
                log(f"  [2wiki corpus] {raw} records, {len(es.runs)} runs, {time.time()-t0:.0f}s")
    log(f"  [2wiki corpus] merging {len(es.runs)} runs ({raw} records)")
    W = NodeWriter(out)
    dup, dup_examples = es.merge_into(W)
    acc = {"raw_source_records": raw, "dropped_records": dropped, "duplicates_collapsed": dup,
           "identity": "official Wikipedia curid (record 'id'), exact string; node_id = 2wiki:c<curid>",
           "raw_source_record_unit": "one json article record (one line of para_with_hyperlink.jsonl)",
           "external_sort_runs": len(es.runs), "duplicate_curid_examples": dup_examples,
           "total_sentences": n_sent_total, "total_hyperlink_mentions": n_mentions,
           "text_construction": "text = ' '.join(official 'sentences' list) -- identical to build.py::build_corpus_2wiki's "
                                "rule (build.py:137), applied to the full article universe instead of the context union; "
                                "if that join is empty the source-native 'title' of the SAME record is used instead "
                                "(TEXTUALIZATION_REV 2, fallback only -- no title is prepended to any record that already "
                                "has text, and a record with an empty title stays empty)",
           "textualization_fallback": {
               "TEXTUALIZATION_REV": 2,
               "rule": "if ' '.join(official 'sentences' list) is empty -> canonical_text = record['title'] (unchanged otherwise)",
               "empty_before_fallback": empty_text,
               "title_fallback_applied": title_fallback,
               "still_empty_after_fallback": still_empty,
               "field_source": "record['title'] -- same JSON line as 'sentences'; the para_with_hyperlink schema is exactly "
                               "{id, title, sentences, mentions} and carries no question/answer/supporting_facts field",
               "query_independent": True,
               "same_rule_as": "hotpotqa (TEXTUALIZATION_REV 2, 2026-09-05) -- one general source-level rule, applied wherever "
                               "a source record yields an empty body but carries a non-empty title",
               "why": "the previous empty encoder input tokenizes to the ZERO-LENGTH sequence [] under gte-Qwen2, collapsing "
                      "every such node onto one degenerate forward pass; 20,524 of these nodes are hyperlink targets, so the "
                      "degenerate points would propagate into every KNN/H4 structure built over this corpus "
                      "(results/data_audit/final_canonical_v1/2wiki/EMPTY_TEXT_AUDIT.md)",
               "alternatives_rejected": "ALT_TEXT_NONEMPTY = 0 -- the schema has no other text field, so the title is the only "
                                        "source-native candidate; TRULY_TEXTLESS = 0, so no record needs to stay empty",
               "membership_impact": "NONE -- no node added, removed or reordered; identity remains the curid"},
           "phase_c_divergence": {
               "n_records_where_strip_changes_text": strip_delta, "strip_examples": strip_examples,
               "n_records_retextualized_from_title": title_fallback,
               "n_records_with_a_different_phase_c_encoder_input": strip_delta + title_fallback,
               "phase_c_reference": "data/canonical/2wiki_universe/documents.jsonl (5,989,847 rows, dense+SPLADE COMPLETE)",
               "meaning": "canonical_v1's text differs from the already-encoded 2wiki_universe text in exactly two situations and no "
                          "others: (a) scratchpad/build_2wiki_universe.py:42 applied .strip() to the same ' '.join(sentences) and "
                          "canonical_v1 does not; (b) TEXTUALIZATION_REV 2 retextualizes an empty body as the record's own title, "
                          "where the stored row holds \"\".  Every other record's encoder input is byte-identical -- hence "
                          "token-identical -- to the stored row.  This sum is therefore the exact upper bound on dense/SPLADE rows "
                          "needing a re-encode: measured, not assumed.  It is an UPPER bound because a retextualized title may still "
                          "collide with an already-encoded input (the forensic scan found exactly 1 such case, 2wiki:c320778 "
                          "\"Kagawa\"); reuse_map_kb.py decides that by frozen-tokenizer token-ID equality, not by this count."},
           "empty_text_nodes": {"n": empty_text, "policy": "KEPT -- dropping them would make corpus membership depend on a "
                                                           "text-quality filter (the 2wiki_universe manifest observed 87,765)",
                                "note": "counts records whose SOURCE body is empty. Since TEXTUALIZATION_REV 2 these are no longer "
                                        "empty in the node table: they carry the source-native title. The count of nodes with empty "
                                        "final text is integrity_report.empty_text_nodes.",
                                "n_still_empty_after_title_fallback": still_empty},
           "identity_normalization_diagnostics": diag.report(),
           "REPLACES": {"superseded_corpus": "pooled question-context union of train/dev/test 'context' fields",
                        "superseded_n": 398354, "superseded_corpus_hash_16": "ec7fbbe8cd783040",
                        "superseded_artifacts": "data/final_canonical/2wiki/_superseded_context_union_398354/"},
           "note": "This is the benchmark's own distributed article universe, not a question by-product: the record schema carries "
                   "no question/answer/supporting_facts field, and the 398,354 context titles are 6.7% of it."}
    return W, acc, None


# =====================================================================================  CORPUS: WEBQSP

def _triple_endpoints(g):
    """RoG 'graph' cell -> yields (head, tail) for each triple.  Tolerates list-of-list and struct encodings."""
    if g is None: return
    for tr in g:
        if tr is None: continue
        if isinstance(tr, dict):
            ks = list(tr)
            h, t = tr.get("h", tr.get(ks[0])), tr.get("t", tr.get(ks[-1]))
        else:
            tr = list(tr)
            if len(tr) < 3: continue
            h, t = tr[0], tr[2]
        yield h, t


def build_corpus_webqsp(S, out):
    """ONLY the 'graph' column of the RoG parquets is read.  No answer / topic entity is visible here."""
    import pyarrow.parquet as pq
    SPLIT_BIT = {"train": 1, "validation": 2, "test": 4}
    agg = {}                     # entity surface string -> [src_int, n_source_records, split_bits]
    raw = 0; dup = 0; dropped = []
    diag = Diag()
    triples = 0
    for fi, (split, path) in enumerate(S["rog"]):
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        assert "graph" in names, f"{path}: no 'graph' column (schema {sorted(names)})"
        bit = SPLIT_BIT[split]; ri = 0
        for batch in pf.iter_batches(batch_size=8, columns=["graph"]):
            for g in batch.column("graph").to_pylist():
                src = (fi << 24) | (ri & 0xFFFFFF)
                for h, t in _triple_endpoints(g):
                    triples += 1
                    for e in (h, t):
                        raw += 1
                        if not isinstance(e, str) or e == "":
                            dropped.append({"where": f"{os.path.basename(path)}:{ri}", "reason": "empty_or_nonstring_endpoint"})
                            continue
                        a = agg.get(e)
                        if a is None:
                            diag.add(e); agg[e] = [src, 1, bit]
                        else:
                            dup += 1; a[1] += 1; a[2] |= bit
                ri += 1
                if ri % 500 == 0:
                    log(f"  [webqsp corpus] {os.path.basename(path)} {ri} questions, {len(agg)} entities")
    log(f"  [webqsp corpus] emit: sorting {len(agg)} entity ids")
    order = sorted(((f"webqsp:{sha(e)[:24]}", e) for e in agg), key=lambda p: p[0])
    W = NodeWriter(out)
    names_out = []
    for nid, e in order:
        src, nq, bits = agg[e]
        fi, ri = src >> 24, src & 0xFFFFFF
        W.write({"node_id": nid, "source_id": e,
                 "source_record_id": f"{os.path.basename(S['rog'][fi][1])}:{ri}",
                 "split_provenance": [s for s in ("train", "validation", "test") if bits & SPLIT_BIT[s]],
                 "title": e, "text": e, "source_file": S["rog"][fi][1],
                 "content_hash": sha(e + SEP + e), "n_source_records": nq,
                 "n_rog_questions": nq, "is_raw_mid": bool(re.fullmatch(r"[mg]\.[0-9a-z_]+", e))})
        names_out.append(e)
    del order
    n_mid = sum(1 for e in names_out if re.fullmatch(r"[mg]\.[0-9a-z_]+", e))
    acc = {"raw_source_records": raw, "dropped_records": dropped, "duplicates_collapsed": dup,
           "identity": "Freebase entity SURFACE STRING at a RoG triple endpoint (exact); node_id = webqsp:<sha256(entity)[:24]>",
           "raw_source_record_unit": "one triple ENDPOINT mention (head and tail of every triple of every question's 'graph')",
           "rog_triples_read": triples,
           "query_independence_enforcement": "the corpus pass opens each parquet with columns=['graph']; a_entity/q_entity/answer "
                                             "are never read.  Phase-C (scratchpad/c1c2_webqsp.py:32-33) DID inject them -- that is "
                                             "the gold-conditioning defect canonical_v1 removes.",
           "identity_normalization_diagnostics": diag.report(),
           "raw_mid_nodes": {"n": n_mid, "fraction": round(n_mid / max(1, len(names_out)), 6),
                             "policy": "OFFICIAL VERBATIM -- MIDs kept as text; the ba6bd71 MID->name resolution is a graph-derived "
                                       "transform and is NOT applied (it would invalidate 1,316,466 encoded rows). Any resolution "
                                       "must be stored downstream as a separate derived field."},
           "note": "No question-independent Freebase corpus exists on disk or in the official release; a WebQSP corpus is necessarily "
                   "a union of per-question subgraphs. This one takes ALL 4,700 RoG questions of ALL THREE splits, so it is invariant "
                   "to the evaluated query subset. See SOURCE_CONTRACT.json WHY_THIS_IS_THE_FULL_RETRIEVAL_UNIVERSE (5)."}
    del agg
    return W, acc, set(names_out)


# =====================================================================================  CORPUS: HOTPOTQA

def _flat_sentences(t):
    """Official record 'text' is a list of sentence strings (occasionally nested).  Yields the leaf strings."""
    if isinstance(t, str):
        if t: yield t
    elif isinstance(t, list):
        for x in t:
            yield from _flat_sentences(x)


def build_corpus_hotpotqa(S, out):
    """Streams tar -> bz2 -> json lines; spills sorted runs to disk; merges them into lexicographic node_id order."""
    es = ExtSort(f"{WORK}/hotpot_runs")
    TB = S["tarball"]
    raw = 0; dropped = []; empty_abstract = 0; strip_delta = 0; strip_examples = []
    pc_fallback = 0; pc_fallback_examples = []
    title_fallback = 0; still_empty = 0
    diag = Diag(cap=200_000)
    t0 = time.time()

    tar = tarfile.open(TB, "r:bz2"); nm = 0
    for member in tar:
        if not member.isfile(): continue
        nm += 1
        mname = member.name.split("/", 1)[-1] if "/" in member.name else member.name
        try:
            blob = bz2.decompress(tar.extractfile(member).read()).decode("utf-8")
        except Exception as ex:
            dropped.append({"where": mname, "reason": f"member_unreadable:{type(ex).__name__}"}); continue
        for li, line in enumerate(blob.splitlines()):
            if not line.strip(): continue
            raw += 1
            try:
                r = json.loads(line)
            except Exception:
                dropped.append({"where": f"{mname}:{li}", "reason": "unparsable_json_line"}); continue
            cid = r.get("id")
            if cid is None or str(cid) == "":
                dropped.append({"where": f"{mname}:{li}", "reason": "missing_curid"}); continue
            cid = str(cid)
            sents = list(_flat_sentences(r.get("text")))
            body = " ".join(sents)                       # NO .strip() -- see module docstring
            if body != body.strip():
                strip_delta += 1
                if len(strip_examples) < 20:
                    strip_examples.append({"curid": cid, "title": r.get("title", ""), "len": len(body),
                                           "len_stripped": len(body.strip())})
            if not body.strip():
                empty_abstract += 1
                # Phase-C fell back to 'text_with_links' when 'text' was falsy (c1_hotpot_fullwiki.py:38); canonical_v1
                # never does. Where that fallback produced a non-empty body, the Phase-C encoder input DIFFERS from the
                # canonical text by more than a strip, so its stored row is NOT reusable. Counted, not assumed.
                if list(_flat_sentences(r.get("text_with_links"))):
                    pc_fallback += 1
                    if len(pc_fallback_examples) < 20:
                        pc_fallback_examples.append({"curid": cid, "title": r.get("title", "")})
                # TEXTUALIZATION FALLBACK (contract revision 2026-09-05, TEXTUALIZATION_REV 2).  A source record whose
                # abstract is empty would otherwise be handed to the dense encoder as "" -- which gte-Qwen2 tokenizes to
                # the ZERO-LENGTH sequence [], so every such node collapses onto one degenerate forward pass.  The same
                # source record carries a human-readable 'title', so use it.  Source-native (same JSON line as 'text'),
                # deterministic, and query-independent: the record schema holds no question/answer/supporting_facts
                # field.  'text_with_links' is NOT a candidate -- for these records it is empty <a></a> markup whose
                # readable residue measured 0/94 (results/data_audit/final_canonical_v1/hotpotqa/EMPTY_TEXT_94_AUDIT.md).
                # FALLBACK ONLY: records that already produce non-empty text are untouched, and no title is prepended
                # anywhere else in the corpus.
                tfb = r.get("title", "")
                if tfb.strip():
                    body = tfb
                    title_fallback += 1
                else:
                    still_empty += 1
            diag.add(cid)
            nid = f"hotpotqa:c{cid}"
            es.add(nid, {"node_id": nid, "source_id": cid, "source_record_id": f"{mname}:{li}",
                         "split_provenance": ["fullwiki_corpus"], "title": r.get("title", ""), "text": body,
                         "source_file": TB, "content_hash": sha(cid + SEP + body), "n_source_records": 1,
                         "n_sentences": len(sents), "url": r.get("url")})
        if nm % 5000 == 0:
            log(f"  [hotpot corpus] {nm} members, {raw} records, {len(es.runs)} runs, {time.time()-t0:.0f}s")
    tar.close()
    log(f"  [hotpot corpus] merging {len(es.runs)} runs ({raw} records)")
    W = NodeWriter(out)
    dup, dup_examples = es.merge_into(W)

    acc = {"raw_source_records": raw, "dropped_records": dropped, "duplicates_collapsed": dup,
           "identity": "official Wikipedia curid (record 'id'), exact string; node_id = hotpotqa:c<curid>",
           "raw_source_record_unit": "one json article record (one line inside a member .bz2 of the tarball)",
           "tar_members_read": nm, "external_sort_runs": len(es.runs), "duplicate_curid_examples": dup_examples,
           "text_construction": "text = ' '.join(non-empty sentences of the official 'text' list); 'text_with_links' NOT used; "
                                "if that join is empty the source-native 'title' of the SAME record is used instead "
                                "(TEXTUALIZATION_REV 2, fallback only -- no title is prepended to any record that already "
                                "has text, and a record with an empty title stays empty)",
           "textualization_fallback": {
               "TEXTUALIZATION_REV": 2,
               "rule": "if ' '.join(non-empty sentences of 'text') is empty -> canonical_text = record['title'] (unchanged otherwise)",
               "empty_before_fallback": empty_abstract,
               "title_fallback_applied": title_fallback,
               "still_empty_after_fallback": still_empty,
               "field_source": "record['title'] -- same JSON line as 'text'; schema carries no question/answer/supporting_facts field",
               "query_independent": True,
               "why": "the previous empty encoder input tokenizes to the ZERO-LENGTH sequence [] under gte-Qwen2, collapsing "
                      "every such node onto one degenerate forward pass; 'text_with_links' is not usable here (readable "
                      "residue measured 0/94 -- results/data_audit/final_canonical_v1/hotpotqa/EMPTY_TEXT_94_AUDIT.md)",
               "membership_impact": "NONE -- no node added, removed or reordered; identity remains the curid"},
           "phase_c_divergence": {
               "n_records_where_strip_changes_text": strip_delta, "strip_examples": strip_examples,
               "n_records_where_phase_c_fell_back_to_text_with_links": pc_fallback, "fallback_examples": pc_fallback_examples,
               "n_records_with_a_different_phase_c_encoder_input": strip_delta + pc_fallback,
               "meaning": "canonical_v1's text differs from the Phase-C text in exactly two situations, and in no others: (a) Phase-C "
                          "applied .strip() to the joined abstract (scratchpad/c1_hotpot_fullwiki.py:39) and canonical_v1 does not; "
                          "(b) Phase-C fell back to 'text_with_links' when 'text' was empty (line 38) and canonical_v1 never does. "
                          "Both joins are otherwise the same ' '.join over the same non-empty sentences, so every OTHER record's "
                          "encoder input is byte-identical -- hence token-identical -- to the already-computed row. The sum above is "
                          "therefore the exact upper bound on dense/SPLADE rows needing a re-encode: measured, not assumed."},
           "empty_abstract_nodes": {"n": empty_abstract, "policy": "KEPT -- dropping them would make corpus membership depend on a "
                                                                  "text-quality filter (Phase-C observed 94)",
                                    "note": "counts records whose SOURCE abstract is empty. Since TEXTUALIZATION_REV 2 these are "
                                            "no longer empty in the node table: they carry the source-native title. The count of "
                                            "nodes with empty final text is integrity_report.empty_text_nodes.",
                                    "n_still_empty_after_title_fallback": still_empty},
           "identity_normalization_diagnostics": diag.report(),
           "note": "This is a Wikipedia snapshot (enwiki-20171001 abstracts), not a benchmark by-product: the record schema carries no "
                   "question/answer/supporting_facts field and ~90.3% of its articles are never referenced by any HotpotQA question."}
    return W, acc, None


def _finish(js, n_dup):
    if not n_dup: return json.loads(js)
    nd = json.loads(js); nd["n_source_records"] = 1 + n_dup
    return nd


CORPUS = {"2wiki": build_corpus_2wiki, "webqsp": build_corpus_webqsp, "hotpotqa": build_corpus_hotpotqa}


# =====================================================================================  QUERY STAGE

def _qstats(recs):
    """PER-REFERENCE gold semantics.  (2026-09-05 patch -- supersedes the cardinality test.)

    OLD, and forbidden:  n_all_gold_resolved counted a question only when
            len(gold_node_ids) == len(set(map(str, gold_refs)))
        which fails a question merely because ONE of its references is ambiguous and legitimately
        expands to more than one canonical node.  (2wiki: "Unconquered" -> 2 curids, 8 train
        questions; gold_refs_total 404,170 vs gold_nodes_total 404,178, a SURPLUS not a shortfall.)

    NEW:  each gold_ref is resolved independently and is satisfied iff it maps to >= 1 canonical
        node.  A ref that resolves uniquely from official source context stores exactly that node
        (see _disambiguate_from_source); a ref that is GENUINELY ambiguous keeps its alternatives
        and is satisfied DISJUNCTIVELY -- retrieved & {alternatives} != {} -- never conjunctively.

    Per-record inputs: gold_refs, gold_node_ids, gold_refs_unresolved, gold_title_ambiguous.
    The legacy keys are retained so nothing downstream breaks; gold_refs_total keeps its old
    str()-keyed definition so the published totals stay comparable.
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


def qwrite(out, split, recs):
    os.makedirs(f"{out}/queries", exist_ok=True)
    with open(f"{out}/queries/{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def light(rec):
    return {"split": rec["split"], "question": rec["question"], "gold_refs": rec["gold_refs"], "hop": rec.get("hop")}


def fetch_full(out, qids):
    want = set(qids); got = {}
    for fn in sorted(os.listdir(f"{out}/queries")):
        if not fn.endswith(".jsonl"): continue
        for r in iter_jsonl(f"{out}/queries/{fn}"):
            if r["query_id"] in want: got[r["query_id"]] = r
    return got


class Unresolved:
    def __init__(self):
        self.counts = collections.Counter(); self.examples = []

    def miss(self, split, ref):
        self.counts[split] += 1
        if len(self.examples) < 50: self.examples.append({"split": split, "ref": ref})


def build_queries_webqsp(S, names, out):
    """Official Microsoft WebQSP questions.  gold = official answer entity NAMES resolved into the corpus.
    RoG a_entity/q_entity are read HERE (query stage) only, and reported as a second, cross-check coverage."""
    import pyarrow.parquet as pq
    rog_a = {}; rog_q = {}; rog_split = {}
    for split, path in S["rog"]:
        pf = pq.ParquetFile(path)
        cols = [c for c in ("id", "a_entity", "q_entity") if c in set(pf.schema_arrow.names)]
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            d = batch.to_pydict()
            for i, qid in enumerate(d["id"]):
                rog_split[qid] = split
                rog_a[qid] = [x for x in (d.get("a_entity") or [[]] * len(d["id"]))[i] or [] if isinstance(x, str)]
                rog_q[qid] = [x for x in (d.get("q_entity") or [[]] * len(d["id"]))[i] or [] if isinstance(x, str)]

    def nid_of(name):
        return f"webqsp:{sha(name)[:24]}" if name in names else None

    def ans_names(parses):
        out_ = []
        for p in parses or []:
            for a in p.get("Answers", []):
                nm = a.get("EntityName") or a.get("AnswerArgument")
                if nm: out_.append(nm)
        return list(dict.fromkeys(out_))

    def ans_mids(parses):
        out_ = []
        for p in parses or []:
            for a in p.get("Answers", []):
                if a.get("AnswerType") == "Entity" and a.get("AnswerArgument"): out_.append(a["AnswerArgument"])
        return list(dict.fromkeys(out_))

    U = Unresolved(); stats = {}; index = {}; rog_cov = collections.Counter()
    for split, path in S["official"].items():
        recs = []
        qs = json.load(open(path, encoding="utf-8"))["Questions"]
        for i, q in enumerate(qs):
            qid = q["QuestionId"]
            gnames = ans_names(q.get("Parses"))
            gold, missing = [], []
            for nm in gnames:
                n = nid_of(nm)
                if n is None: U.miss(split, nm); missing.append(nm)      # PER-REFERENCE: this ref did not resolve
                else: gold.append(n)
            gold = list(dict.fromkeys(gold))
            ra = rog_a.get(qid, [])
            rgold = list(dict.fromkeys([n for n in (nid_of(x) for x in ra) if n]))
            if ra:
                rog_cov["questions_with_rog_answers"] += 1
                # per-reference, not cardinality: every distinct RoG answer string must resolve
                if all(nid_of(x) is not None for x in set(ra)): rog_cov["rog_answers_fully_resolved"] += 1
                if rgold: rog_cov["rog_answers_any_resolved"] += 1
            topics = list(dict.fromkeys([p["TopicEntityName"] for p in (q.get("Parses") or []) if p.get("TopicEntityName")])) \
                     or rog_q.get(qid, [])
            recs.append({"query_id": qid, "dataset": "webqsp", "split": split,
                         "question": q.get("RawQuestion") or q.get("ProcessedQuestion"),
                         "question_processed": q.get("ProcessedQuestion"),
                         "answers": gnames, "official_answer_mids": ans_mids(q.get("Parses")),
                         "gold_refs": gnames, "gold_node_ids": gold,
                         "gold_refs_unresolved": missing, "gold_title_ambiguous": [],
                         "topic_entities": topics,
                         "topic_entity_node_ids": [n for n in (nid_of(t) for t in topics) if n],
                         "rog_split": rog_split.get(qid), "rog_answer_entities": ra,
                         "rog_gold_node_ids": rgold, "rog_topic_entities": rog_q.get(qid, []),
                         "parsing_eligible": qid in rog_split, "answerable": bool(gnames),
                         "source_file": path, "source_index": i})
            index[qid] = light(recs[-1])
        qwrite(out, split, recs); stats[split] = _qstats(recs); del recs
    stats["_rog_answer_cross_check"] = dict(rog_cov)
    stats["_gold_policy"] = ("gold_refs = OFFICIAL Microsoft answer entity names (Parses[].Answers[].EntityName, else AnswerArgument). "
                             "Resolution is exact-string into the corpus. Unresolved answers are REPORTED, never rescued by injecting "
                             "the answer into the corpus (that is the Phase-C defect). rog_gold_node_ids is a separate cross-check.")
    return stats, U, index


def _title_index(out, needed, tag):
    """Single stream over the FROZEN corpus, keeping only the titles the official questions mention.
    Returns (title -> sorted node_ids, ambiguous subset).  The corpus is never filtered by this."""
    t2i = collections.defaultdict(list); seen = 0
    for r in iter_jsonl(f"{out}/nodes.jsonl"):
        seen += 1
        t = r["title"]
        if t in needed: t2i[t].append(r["node_id"])
        if seen % 1_000_000 == 0: log(f"  [{tag} queries] pass B {seen} nodes scanned, {len(t2i)} titles matched")
    for t in t2i: t2i[t].sort()
    return t2i, {t: v for t, v in t2i.items() if len(v) > 1}


def _norm_ws(s):
    return " ".join((s or "").split())


def _disambiguate_from_source(out, ambiguous, question_contexts, tag):
    """Resolve an ambiguous gold reference using the OFFICIAL SOURCE CONTEXT of the question itself.

    A supporting-fact TITLE that matches several canonical curids is not automatically ambiguous:
    the official question record carries its own context paragraph for that title, which names the
    actual referent.  We compare that paragraph against each candidate node's frozen text and keep
    the candidate it uniquely identifies.  Only when no unique candidate survives does the ref stay
    ambiguous, and then the DISJUNCTIVE fallback in _qstats applies.

    Identity comes from the canonical source (official context paragraph + frozen node text) only.
    No question text, no answers, no gold paths, no neighbour heuristics.

    question_contexts: {(query_id, title) -> official context text}
    returns          : {(query_id, title) -> node_id}, plus a diagnostic record
    """
    diag = {"ambiguous_titles": len(ambiguous), "candidate_nodes": 0,
            "question_title_pairs": len(question_contexts), "resolved": 0, "unresolved": 0,
            "min_common_prefix_chars": 40, "decisions": []}
    if not ambiguous or not question_contexts:
        return {}, diag
    want = {i for v in ambiguous.values() for i in v}
    diag["candidate_nodes"] = len(want)
    texts = {}
    for r in iter_jsonl(f"{out}/nodes.jsonl"):                       # read-only; never rewritten
        if r["node_id"] in want:
            texts[r["node_id"]] = _norm_ws(r["text"])
            if len(texts) == len(want): break
    picked = {}
    for (qid, title), ctx in question_contexts.items():
        cands = ambiguous.get(title)
        if not cands:
            continue
        c = _norm_ws(ctx)
        scored = []
        for nid in cands:
            t = texts.get(nid, "")
            k = 0
            for x, y in zip(c, t):
                if x != y: break
                k += 1
            scored.append((k, nid))
        scored.sort(key=lambda z: (-z[0], z[1]))
        top = scored[0]
        unique = len(scored) == 1 or top[0] > scored[1][0]
        if unique and top[0] >= diag["min_common_prefix_chars"]:
            picked[(qid, title)] = top[1]
            diag["resolved"] += 1
        else:
            diag["unresolved"] += 1
        if len(diag["decisions"]) < 12:
            diag["decisions"].append({"query_id": qid, "title": title,
                                      "scores": [{"node_id": n, "common_prefix_chars": k} for k, n in scored],
                                      "picked": picked.get((qid, title)),
                                      "source_context_head": c[:120]})
    log(f"  [{tag} queries] source disambiguation: {diag['resolved']} resolved / "
        f"{diag['unresolved']} still ambiguous over {len(ambiguous)} ambiguous title(s)")
    return picked, diag


def build_queries_2wiki(S, out):
    """Official 2Wiki train/dev/test questions, resolved into the FULL article universe by supporting-fact TITLE.

    Three passes so the title->node_id map only ever holds the ~4e5 titles the questions actually mention, never
    the ~6e6-title corpus index.
    """
    tmpd = f"{out}/queries"; os.makedirs(tmpd, exist_ok=True)
    needed = set(); per_split = collections.Counter()

    # ---- pass A: parse the official question files, write partial records, collect needed titles ----
    for split in S["splits"]:
        path = f"{S['base']}/{split}.json"
        with open(f"{tmpd}/{split}.tmp", "w", encoding="utf-8", newline="\n") as h:
            for qi, q in enumerate(iter_json_array(path)):
                sf = q.get("supporting_facts") or []
                sf_titles = list(dict.fromkeys(s[0] for s in sf))
                ctx_titles = [e[0] for e in (q.get("context") or [])]
                needed.update(sf_titles); needed.update(ctx_titles)
                h.write(json.dumps({"query_id": q["_id"], "dataset": "2wiki", "split": split,
                                    "question": q["question"], "answer": q.get("answer"),
                                    "answers": [q["answer"]] if q.get("answer") else [],
                                    "type": q.get("type"), "supporting_facts": sf,
                                    "evidences": q.get("evidences") or [],
                                    "gold_refs": sf_titles, "context_titles": ctx_titles,
                                    "source_file": path, "source_index": qi},
                                   ensure_ascii=False, sort_keys=True) + "\n")
                per_split[split] += 1
                if per_split[split] % 50000 == 0:
                    log(f"  [2wiki queries] pass A {split} {per_split[split]}, {len(needed)} titles needed")
        log(f"  [2wiki queries] pass A {split} -> {per_split[split]} questions, {len(needed)} titles needed")

    # ---- pass B / C ----
    t2i, ambiguous = _title_index(out, needed, "2wiki")

    # ---- pass B2: OFFICIAL SOURCE CONTEXT for the (few) ambiguous titles ----
    # Only entered when a title genuinely matches >1 curid.  One extra stream of the official
    # question files; nothing else is read and nodes.jsonl is opened read-only.
    qctx = {}
    if ambiguous:
        for split in S["splits"]:
            for q in iter_json_array(f"{S['base']}/{split}.json"):
                sf_titles = {s[0] for s in (q.get("supporting_facts") or [])}
                hit = sf_titles & set(ambiguous)
                if not hit: continue
                for title, sents in (q.get("context") or []):
                    if title in hit:
                        qctx[(q["_id"], title)] = " ".join(sents or [])
    picked, disambig_diag = _disambiguate_from_source(out, ambiguous, qctx, "2wiki")

    U = Unresolved(); stats = {}; index = {}
    for split in S["splits"]:
        recs = []
        for r in iter_jsonl(f"{tmpd}/{split}.tmp"):
            gold, missing, still_amb = [], [], []
            for t in r["gold_refs"]:
                ids = t2i.get(t)
                if not ids:
                    U.miss(split, t); missing.append(t); continue
                if len(ids) > 1:                                  # resolve from official source context
                    p = picked.get((r["query_id"], t))
                    if p is not None: ids = [p]
                    else: still_amb.append(t)                     # genuinely ambiguous -> DISJUNCTIVE
                gold.extend(ids)
            r["gold_node_ids"] = list(dict.fromkeys(gold))
            r["context_node_ids"] = [(t2i[t][0] if t2i.get(t) else None) for t in r["context_titles"]]
            r["gold_refs_unresolved"] = missing
            r["gold_title_ambiguous"] = still_amb
            recs.append(r); index[r["query_id"]] = light(r)
        qwrite(out, split, recs); stats[split] = _qstats(recs); del recs
        os.remove(f"{tmpd}/{split}.tmp")
    stats["_gold_policy"] = ("gold_refs = official supporting_facts TITLES (deduplicated, order preserved); resolution is exact title -> "
                             "curid node via a single scan of the frozen corpus. PER-REFERENCE semantics: each ref must map to >=1 "
                             "canonical node. A title matching several curids is first disambiguated against the question's OWN official "
                             "context paragraph and, when that identifies one uniquely, exactly that node is stored; only a genuinely "
                             "ambiguous ref keeps its alternatives, and it is then satisfied DISJUNCTIVELY (retrieved & {alts} != {}), "
                             "never conjunctively. Unresolvable titles are REPORTED, never rescued by adding the article to the corpus.")
    stats["_title_resolution"] = {"titles_needed": len(needed), "titles_matched": len(t2i),
                                  "titles_unmatched": len(needed) - len(t2i),
                                  "ambiguous_titles": len(ambiguous),
                                  "ambiguous_examples": {k: v for k, v in list(ambiguous.items())[:10]}}
    stats["_source_disambiguation"] = disambig_diag
    return stats, U, index


def build_queries_hotpotqa(S, out):
    """Three passes so the title->node_id map only ever holds the titles the questions actually mention."""
    import pyarrow.parquet as pq
    tmpd = f"{out}/queries"; os.makedirs(tmpd, exist_ok=True)
    needed = set(); per_split = collections.defaultdict(int)

    # ---- pass A: parse the official question files, write partial records, collect needed titles ----
    handles = {}
    for split, path in S["queries"]:
        if split not in handles:
            handles[split] = open(f"{tmpd}/{split}.tmp", "w", encoding="utf-8", newline="\n")
        pf = pq.ParquetFile(path)
        have = set(pf.schema_arrow.names)
        cols = [c for c in ("id", "question", "answer", "type", "level", "supporting_facts", "context") if c in have]
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            for r in batch.to_pylist():
                sf = r.get("supporting_facts") or {}
                sf_t = list(sf.get("title") or []) if isinstance(sf, dict) else [x[0] for x in sf]
                sf_s = list(sf.get("sent_id") or []) if isinstance(sf, dict) else [x[1] for x in sf]
                gold_titles = list(dict.fromkeys(sf_t))
                ctx = r.get("context") or {}
                ctx_t = list(ctx.get("title") or []) if isinstance(ctx, dict) else [x[0] for x in (ctx or [])]
                needed.update(gold_titles); needed.update(ctx_t)
                rec = {"query_id": r["id"], "dataset": "hotpotqa", "split": split, "setting":
                       ("distractor_train_shared" if split == "train" else "fullwiki"),
                       "question": r.get("question"), "answer": r.get("answer"),
                       "answers": [r["answer"]] if r.get("answer") else [],
                       "type": r.get("type"), "level": r.get("level"),
                       "supporting_facts": [{"title": a, "sent_id": b} for a, b in zip(sf_t, sf_s)],
                       "gold_refs": gold_titles, "context_titles": ctx_t,
                       "source_file": path, "source_index": per_split[split]}
                handles[split].write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                per_split[split] += 1
        log(f"  [hotpot queries] pass A {os.path.basename(path)} -> {split} total {per_split[split]}, {len(needed)} titles needed")
    for h in handles.values(): h.close()

    # ---- pass B: single stream over the frozen corpus, keeping only the needed titles ----
    t2i, ambiguous = _title_index(out, needed, "hotpot")

    # ---- pass B2: OFFICIAL SOURCE CONTEXT for the (few) ambiguous titles ----
    qctx = {}
    if ambiguous:
        for split, path in S["queries"]:
            pf = pq.ParquetFile(path)
            have = set(pf.schema_arrow.names)
            for batch in pf.iter_batches(batch_size=256,
                                         columns=[c for c in ("id", "supporting_facts", "context") if c in have]):
                for r in batch.to_pylist():
                    sf = r.get("supporting_facts") or {}
                    sf_t = set(sf.get("title") or []) if isinstance(sf, dict) else {x[0] for x in sf}
                    hit = sf_t & set(ambiguous)
                    if not hit: continue
                    ctx = r.get("context") or {}
                    pairs = zip(ctx.get("title") or [], ctx.get("sentences") or []) if isinstance(ctx, dict) \
                        else [(x[0], x[1]) for x in (ctx or [])]
                    for title, sents in pairs:
                        if title in hit:
                            qctx[(r["id"], title)] = " ".join(sents or [])
    picked, disambig_diag = _disambiguate_from_source(out, ambiguous, qctx, "hotpot")

    # ---- pass C: resolve and finalize ----
    U = Unresolved(); stats = {}; index = {}
    for split in sorted(handles):
        recs = []
        for r in iter_jsonl(f"{tmpd}/{split}.tmp"):
            gold, missing, still_amb = [], [], []
            for t in r["gold_refs"]:
                ids = t2i.get(t)
                if not ids:
                    U.miss(split, t); missing.append(t); continue
                ids = sorted(ids)
                if len(ids) > 1:                                  # resolve from official source context
                    p = picked.get((r["query_id"], t))
                    if p is not None: ids = [p]
                    else: still_amb.append(t)                     # genuinely ambiguous -> DISJUNCTIVE
                gold.extend(ids)
            r["gold_node_ids"] = list(dict.fromkeys(gold))
            r["context_node_ids"] = [(sorted(t2i[t])[0] if t2i.get(t) else None) for t in r["context_titles"]]
            r["gold_refs_unresolved"] = missing
            r["gold_title_ambiguous"] = still_amb
            recs.append(r); index[r["query_id"]] = light(r)
        qwrite(out, split, recs); stats[split] = _qstats(recs); del recs
        os.remove(f"{tmpd}/{split}.tmp")
    stats["_gold_policy"] = ("gold_refs = official supporting_facts TITLES (deduplicated, order preserved); resolution is exact title -> "
                             "curid node via a single scan of the frozen corpus. PER-REFERENCE semantics: each ref must map to >=1 "
                             "canonical node; an ambiguous title is first disambiguated against the question's OWN official context "
                             "paragraph, and only a genuinely ambiguous ref keeps disjunctive alternatives.")
    stats["_title_resolution"] = {"titles_needed": len(needed), "titles_matched": len(t2i),
                                  "titles_unmatched": len(needed) - len(t2i),
                                  "ambiguous_titles": len(ambiguous),
                                  "ambiguous_examples": dict(list(ambiguous.items())[:10])}
    stats["_source_disambiguation"] = disambig_diag
    return stats, U, index


# =====================================================================================  EVAL SUBSET

out_dir_for_fetch = [None]


def _legacy_rows(ds):
    """rows/ids of the frozen legacy eval subset.  Recovered from the G2 cache + the UKB query id table
    (read-only); cached under _work/legacy/<L>/legacy_eval_rows.json, exactly as for the four built datasets."""
    L = LEGACY_NAME[ds]; base = f"{WORK}/legacy/{L}"; p = f"{base}/legacy_eval_rows.json"
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    import numpy as np
    cache = f"results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_{L}.npz"
    z = np.load(cache, allow_pickle=True)
    rows = [int(x) for x in z["rows"]]
    j = json.load(open(f"data/ukb_storage/{L}/gte_qwen/query_ids_all.json", encoding="utf-8"))
    ids = [j["ids"][r] for r in rows]
    d = {"rows": rows, "ids": ids, "hops": [None] * len(rows),
         "source": {"cache": cache, "rows_key": "rows", "ids_from": f"data/ukb_storage/{L}/gte_qwen/query_ids_all.json (ids[row])"}}
    os.makedirs(base, exist_ok=True)
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
    return d


def legacy_subset_2wiki(index):
    """2wiki_clean legacy ids do NOT embed the official question id (the substrate came from the FlashRAG 15k
    train sample, whose raw file is gone).  This reproduces build.py::legacy_subset's 2wiki branch verbatim:
    EXACT question-text match against the official questions, ties broken by supporting-fact title set."""
    L = "2wiki_clean"; base = f"{WORK}/legacy/{L}"
    rows = _legacy_rows("2wiki")
    lq = {r["node_id"]: r for r in iter_jsonl(f"{base}/questions.jsonl")}
    ldocs = {r["node_id"]: r["title"] for r in iter_jsonl(f"{base}/docs.jsonl")}
    by_text = collections.defaultdict(list)
    for qid, r in index.items(): by_text[r["question"]].append(qid)
    recs = []; misses = []
    for lid, row in zip(rows["ids"], rows["rows"]):
        node = lq.get(lid)
        if node is None:
            misses.append({"legacy_id": lid, "reason": "legacy_question_missing"}); continue
        cands = by_text.get(node["content"], [])
        gold_titles = sorted({ldocs.get(g, "") for g in node["neighbors"]})
        if len(cands) > 1:
            cands2 = [c for c in cands if sorted(set(index[c]["gold_refs"])) == gold_titles]
            if len(cands2) == 1: cands = cands2
        if len(cands) != 1:
            misses.append({"legacy_id": lid, "reason": "no_match" if not cands else "ambiguous",
                           "n_candidates": len(cands)}); continue
        recs.append((row, lid, cands[0], node))
    full = fetch_full(out_dir_for_fetch[0], [q for _, _, q, _ in recs])
    out = []
    for row, lid, qid, node in recs:
        r = dict(full[qid]); r["legacy_query_id"] = lid; r["legacy_cache_row"] = row
        r["legacy_gold_ids"] = node["neighbors"]
        out.append(r)
    return out, {"method": "legacy 2wiki_clean questions came from the FlashRAG 15k train sample (raw file no longer present); "
                           "matched to official train/dev/test by EXACT question text; ambiguities resolved by supporting-fact "
                           "title set == legacy gold title set (identical to build.py::legacy_subset's 2wiki branch)",
                 "n_legacy": len(rows["ids"]), "n_recovered": len(out), "n_missed": len(misses), "misses": misses[:50],
                 "official_split_distribution": dict(collections.Counter(r["split"] for r in out))}


def legacy_subset(ds, index):
    """Legacy ids for webqsp/hotpotqa EMBED the official question id, so recovery is an exact suffix map
    (no text matching needed).  Question text is verified when the legacy dump is present.
    2wiki_clean is different and is delegated to legacy_subset_2wiki."""
    if ds == "2wiki":
        return legacy_subset_2wiki(index)
    L = LEGACY_NAME[ds]; pre = LEGACY_QID_PREFIX[ds]
    rows = _legacy_rows(ds)
    lq = {}
    qf = f"{WORK}/legacy/{L}/questions.jsonl"
    if os.path.exists(qf):
        lq = {r["node_id"]: r for r in iter_jsonl(qf)}
    recs = []; misses = []; text_checked = 0; text_mismatch = 0
    for lid, row in zip(rows["ids"], rows["rows"]):
        if not lid.startswith(pre):
            misses.append({"legacy_id": lid, "reason": "unexpected_id_prefix"}); continue
        qid = lid[len(pre):]
        if qid not in index:
            misses.append({"legacy_id": lid, "reason": "official_query_id_not_found"}); continue
        node = lq.get(lid)
        if node is not None:
            text_checked += 1
            if index[qid]["question"] != node["content"]:
                text_mismatch += 1
                misses.append({"legacy_id": lid, "reason": "question_text_mismatch"}); continue
        recs.append((row, lid, qid, node))
    full = fetch_full(out_dir_for_fetch[0], [q for _, _, q, _ in recs])
    out = []
    for row, lid, qid, node in recs:
        r = dict(full[qid]); r["legacy_query_id"] = lid; r["legacy_cache_row"] = row
        r["legacy_gold_ids"] = node["neighbors"] if node else None
        out.append(r)
    return out, {"method": f"legacy id '{pre}<official_question_id>' -> official query id by exact prefix strip; "
                           f"question text verified against {qf} for {text_checked}/{len(rows['ids'])} ids "
                           f"({text_mismatch} mismatches)" + ("" if lq else " -- legacy question dump ABSENT, text NOT verified"),
                 "n_legacy": len(rows["ids"]), "n_recovered": len(out), "n_missed": len(misses), "misses": misses[:50],
                 "official_split_distribution": dict(collections.Counter(r["split"] for r in out))}


def random_held_subset(ds, index, seed, n=2000):
    """Second, DIFFERENT query subset used only to prove the corpus does not move.  Drawn from the dataset's
    held-out labelled split (webqsp: official MS test; hotpotqa: fullwiki dev) -- neither dataset has a 'dev'
    split in the build.py sense, so the flag is named random_held, not random_dev."""
    split = SRC[ds]["held_out_split"]
    pool = sorted(q for q, r in index.items() if r["split"] == split)
    rng = random.Random(seed); rng.shuffle(pool)
    sel = sorted(pool[:n])
    full = fetch_full(out_dir_for_fetch[0], sel)
    return [full[q] for q in sel], {"method": f"seeded random from official held-out split '{split}' (seed={seed}, n={len(sel)})"}


def path_subset(path, index):
    ids = json.load(open(path, encoding="utf-8")) if path.endswith(".json") else \
          [json.loads(l)["query_id"] for l in open(path, encoding="utf-8") if l.strip()]
    ids = [q for q in ids if q in index]
    full = fetch_full(out_dir_for_fetch[0], ids)
    return [full[q] for q in ids], {"method": f"query ids from {path}", "n_requested": len(ids), "n_found": len(full)}


# =====================================================================================  GATES

APPROVALS = f"{ROOT}/_APPROVALS.json"
NEVER_BUILD = {"webqsp"}      # hard-coded refusal; no flag, env var or argument can lift it


def approvals():
    if not os.path.exists(APPROVALS):
        sys.exit(f"REFUSING: {APPROVALS} does not exist -- there is no per-dataset approval record to read.")
    return json.load(open(APPROVALS, encoding="utf-8"))


def require_gates(ds):
    """PER-DATASET approval, read from _APPROVALS.json.

    The old all-or-nothing gate file _APPROVED_SOURCE_CONTRACTS is GONE: it must not be created and is never
    consulted.  webqsp is refused unconditionally -- the refusal is keyed on the dataset name AND re-checked
    against the recorded status, so neither editing the json nor any command-line flag can build it here.
    """
    A = approvals()
    st = ((A.get("approvals") or {}).get(ds) or {}).get("status")
    if ds in NEVER_BUILD:
        sys.exit(f"REFUSING: webqsp is BLOCKED_PENDING_FREEBASE_SOURCE and this builder will not build it under any "
                 f"argument.\n"
                 f"  Recorded status: {st!r} ({APPROVALS} -> approvals.webqsp).\n"
                 f"  The RoG all-question subgraph union is invariant to the evaluated subset but still QUERY-DERIVED, "
                 f"so it is not a corpus-independent retrieval universe.\n"
                 f"  Next step is a PROPOSAL ONLY: data/final_canonical/webqsp/FREEBASE_SOURCE_PROPOSAL.md. "
                 f"No webqsp nodes may be written.")
    if st is None:
        sys.exit(f"REFUSING: {APPROVALS} has no approvals.{ds} entry.")
    if not str(st).startswith("APPROVED"):
        sys.exit(f"REFUSING: approvals.{ds}.status = {st!r} does not start with 'APPROVED'. Nothing may be built.")
    rej = f"{ROOT}/_REJECTED_SOURCE_CONTRACTS"
    if os.path.exists(rej):
        sys.exit(f"REFUSING: {rej} exists -- the source contracts were REJECTED. Nothing may be built.")
    if ds in HEAVY:
        g = f"{ROOT}/_GATE_HOTPOT_HEAVY_OK"
        if not os.path.exists(g):
            sys.exit(f"REFUSING: {g} does not exist.\n"
                     f"  Streaming/hashing a ~5-6M-document corpus is gated on this machine-resource file.\n"
                     f"  Create it to allow the heavy pass; this builder must never create it itself.")
    return {"approval_source": APPROVALS, "approval_status": st,
            "_GATE_HOTPOT_HEAVY_OK": os.path.exists(f"{ROOT}/_GATE_HOTPOT_HEAVY_OK"),
            "_APPROVED_SOURCE_CONTRACTS_is_obsolete_and_not_consulted": True}


def _source_paths(ds):
    S = SRC[ds]
    if ds == "2wiki":
        return [S["zip"]] + [f"{S['base']}/{s}.json" for s in S["splits"]]
    if ds == "webqsp":
        return [p for _, p in S["rog"]] + list(S["official"].values())
    return [S["tarball"]] + [p for _, p in S["queries"]]


# =====================================================================================  MAIN

class _FrozenW:
    """Stand-in for the corpus writer when --queries-only is used: carries the frozen counters."""
    def __init__(self, empty_text, empty_title):
        self.empty_text = empty_text; self.empty_title = empty_title


def _frozen_corpus_state(out):
    """--queries-only: re-derive the corpus identity from the FROZEN nodes.jsonl, never rewriting it.

    Recomputes canonical_node_count, NODE_ORDER_HASH, CORPUS_HASH and nodes_jsonl_sha256 by streaming
    the existing file, then ASSERTS they equal the values already recorded in integrity_report.json.
    The corpus accounting block is quoted forward from that report because the corpus stage did not
    rerun -- it is not recomputed and not silently invented.
    """
    prev_path = f"{out}/integrity_report.json"
    if not os.path.exists(prev_path):
        raise SystemExit(f"--queries-only needs an existing {prev_path} to assert against")
    prev = json.load(open(prev_path, encoding="utf-8"))
    ho = hashlib.sha256(); hc = hashlib.sha256(); n = 0
    for r in iter_jsonl(f"{out}/nodes.jsonl"):
        nid = r["node_id"]
        if n:
            ho.update(b"\n"); hc.update(b"\n")
        ho.update(nid.encode("utf-8"))
        hc.update(f"{nid}\t{r['content_hash']}".encode("utf-8"))
        n += 1
    order_hash, corpus_hash = ho.hexdigest(), hc.hexdigest()
    nodes_sha = file_sha(f"{out}/nodes.jsonl")
    for k, got in (("canonical_node_count", n), ("NODE_ORDER_HASH", order_hash),
                   ("CORPUS_HASH", corpus_hash), ("nodes_jsonl_sha256", nodes_sha)):
        assert prev.get(k) == got, f"--queries-only: frozen corpus {k} changed: {prev.get(k)} -> {got}"
    acc = prev["corpus_accounting_detail"]
    log(f"[build_kb] --queries-only: frozen corpus VERIFIED n={n} CORPUS_HASH={corpus_hash[:12]} "
        f"NODE_ORDER_HASH={order_hash[:12]} (nodes.jsonl untouched)")
    return (_FrozenW(prev["empty_text_nodes"], prev["empty_title_nodes"]), acc, None,
            n, order_hash, corpus_hash, nodes_sha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", choices=list(SRC))
    ap.add_argument("--eval-subset", default="legacy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--plan", action="store_true", help="print sources + gate status and exit without reading any record")
    ap.add_argument("--queries-only", action="store_true",
                    help="rerun ONLY the query/gold, eval-subset and integrity stages against the FROZEN "
                         "nodes.jsonl. The corpus stage is skipped and nodes.jsonl is opened READ-ONLY; "
                         "its count and hashes are re-derived and asserted unchanged.")
    a = ap.parse_args()
    ds = a.ds; S = SRC[ds]; out = a.out or f"{ROOT}/{ds}"

    if a.plan:
        A = json.load(open(APPROVALS, encoding="utf-8")) if os.path.exists(APPROVALS) else {}
        print(json.dumps({"dataset": ds, "out": out, "eval_subset": a.eval_subset,
                          "gates": {"approvals_status": ((A.get("approvals") or {}).get(ds) or {}).get("status"),
                                    "refused_unconditionally": ds in NEVER_BUILD,
                                    "_REJECTED_SOURCE_CONTRACTS": os.path.exists(f"{ROOT}/_REJECTED_SOURCE_CONTRACTS"),
                                    "_GATE_HOTPOT_HEAVY_OK": os.path.exists(f"{ROOT}/_GATE_HOTPOT_HEAVY_OK"),
                                    "heavy": ds in HEAVY},
                          "sources": {p: {"exists": os.path.exists(p), "bytes": os.path.getsize(p) if os.path.exists(p) else None}
                                      for p in _source_paths(ds)}}, indent=2))
        return

    gates = require_gates(ds)
    t0 = time.time(); started = time.strftime("%Y-%m-%dT%H:%M:%S")
    cmd = "python " + " ".join(sys.argv)
    log(f"[build_kb {ds}] START out={out} eval_subset={a.eval_subset} gates={gates}")

    # ---- corpus (query-independent by construction) ----
    if a.queries_only:
        W, acc, names, n_nodes, order_hash, corpus_hash, nodes_sha = _frozen_corpus_state(out)
    else:
        W, acc, names = CORPUS[ds](S, out)
        n_nodes, order_hash, corpus_hash = W.close()
        nodes_sha = file_sha(f"{out}/nodes.jsonl")
    log(f"[build_kb {ds}] corpus: raw={acc['raw_source_records']} nodes={n_nodes} dup={acc['duplicates_collapsed']} "
        f"dropped={len(acc['dropped_records'])} corpus_hash={corpus_hash[:12]} order_hash={order_hash[:12]} ({time.time()-t0:.0f}s)")
    assert acc["raw_source_records"] == n_nodes + acc["duplicates_collapsed"] + len(acc["dropped_records"]), "accounting mismatch"

    # ---- queries (index into the frozen corpus) ----
    if ds == "webqsp":
        qstats, U, index = build_queries_webqsp(S, names, out)
    elif ds == "2wiki":
        qstats, U, index = build_queries_2wiki(S, out)
    else:
        qstats, U, index = build_queries_hotpotqa(S, out)
    del names
    log(f"[build_kb {ds}] queries: " + ", ".join(f"{k}={v['n']}" for k, v in qstats.items() if isinstance(v, dict) and "n" in v) +
        f" unresolved_gold_refs={dict(U.counts)} ({time.time()-t0:.0f}s)")

    # ---- eval subset (never touches the corpus; nodes.jsonl already written) ----
    ev_info = None; ev_file = None
    out_dir_for_fetch[0] = out
    if a.eval_subset != "none":
        if a.eval_subset == "legacy":
            recs, ev_info = legacy_subset(ds, index); tag = a.tag or f"eval_{len(recs)}"
        elif a.eval_subset.startswith("random_held:") or a.eval_subset.startswith("random_dev:"):
            seed = int(a.eval_subset.split(":")[1])
            recs, ev_info = random_held_subset(ds, index, seed)
            tag = a.tag or f"eval_{len(recs)}_random_{S['held_out_split']}_s{seed}"
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
        log(f"[build_kb {ds}] eval subset: n={len(recs)} all_gold_resolved={ev_info['n_all_gold_resolved']}")

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
        "source_disambiguation": qstats.get("_source_disambiguation"),
    }
    integrity = {
        "dataset": ds, "dataset_version": VERSION, "built_at": started,
        "unique_node_ids": True, "canonical_node_count": n_nodes,
        "raw_source_records": acc["raw_source_records"], "duplicates_collapsed": acc["duplicates_collapsed"],
        "dropped_records": len(acc["dropped_records"]),
        "drop_reasons": dict(collections.Counter(d["reason"] for d in acc["dropped_records"])),
        "accounting_identity": "RAW_SOURCE_RECORDS == CANONICAL_NODES + DUPLICATES_COLLAPSED + DROPPED_RECORDS (asserted)",
        "empty_text_nodes": W.empty_text, "empty_title_nodes": W.empty_title,
        "corpus_accounting_detail": acc,
        "query_stats": qstats, "splits_with_gold": splits_with_gold,
        "ALL_EVAL_GOLDS_PRESENT": bool(all_gold_present and (ev_info is None or ev_info["ALL_GOLD_REFS_RESOLVE"])),
        "GOLD_SEMANTICS": gold_semantics,
        "ALL_GOLD_REFS_RESOLVE": gold_semantics["ALL_GOLD_REFS_RESOLVE"],
        "MISSING_GOLD_REFS": gold_semantics["MISSING_GOLD_REFS"],
        "AMBIGUOUS_GOLD_REFS": gold_semantics["AMBIGUOUS_GOLD_REFS"],
        "gold_coverage_note": ("A value of false here is a RESULT, not a failure of the build: canonical_v1 refuses to inject answer "
                               "entities into the corpus to guarantee its own gold coverage. Per-split numbers are in query_stats."),
        "unresolved_gold_refs": {"counts": dict(U.counts), "examples": U.examples},
        "eval_subset": ev_info,
        "CORPUS_HASH": corpus_hash, "NODE_ORDER_HASH": order_hash, "nodes_jsonl_sha256": nodes_sha,
    }
    json.dump(integrity, open(f"{out}/integrity_report.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---- build info ----
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = None
    here = os.path.abspath(__file__)
    info = {"dataset": ds, "dataset_version": VERSION, "command": cmd,
            "builder": os.path.relpath(here).replace("\\", "/"), "builder_sha256": file_sha(here),
            "jsonstream_sha256": file_sha(os.path.join(os.path.dirname(here), "jsonstream.py")),
            "git_commit": commit, "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - t0, 1), "peak_rss_mb": peak_rss_mb(),
            "eval_subset_arg": a.eval_subset, "out": out, "gates": gates, "queries_only": bool(a.queries_only),
            "CORPUS_HASH": corpus_hash, "NODE_ORDER_HASH": order_hash, "nodes_jsonl_sha256": nodes_sha,
            "canonical_node_count": n_nodes,
            "source_files": {p: {"sha256": file_sha(p), "bytes": os.path.getsize(p)} for p in _source_paths(ds)}}
    json.dump(info, open(f"{out}/build_info.json", "w", encoding="utf-8"), indent=2)
    log(f"[build_kb {ds}] DONE nodes={n_nodes} CORPUS_HASH={corpus_hash[:12]} peak_rss_mb={info['peak_rss_mb']} ({info['seconds']}s)")


if __name__ == "__main__":
    main()
