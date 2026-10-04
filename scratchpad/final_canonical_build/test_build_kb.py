"""Synthetic self-test for scratchpad/final_canonical_build/build_kb.py.

Touches NO official dataset file.  Builds tiny fake sources with the same shapes and checks:
  webqsp   * corpus reads ONLY 'graph' -> an entity that appears solely in a_entity/q_entity MUST NOT become a node
           * dedup across questions/splits, split_provenance union, accounting identity
           * gold resolution reports the unresolved answer instead of inventing a node
  hotpotqa * external spill/merge produces strictly lexicographic node_id order
           * duplicate curid collapses with n_source_records incremented
           * NO .strip() applied, and the strip-delta diagnostic counts exactly the affected records
           * empty abstract kept
           * 3-pass title resolution, incl. an ambiguous title mapping to two curids
"""
import sys, os, json, io, bz2, tarfile, tempfile, shutil, hashlib
sys.path.insert(0, os.path.abspath("scratchpad/final_canonical_build"))
import build_kb as K
import pyarrow as pa, pyarrow.parquet as pq

TMP = tempfile.mkdtemp(prefix="kbtest_")
OK = []

# ISOLATE THE EXTERNAL-SORT SPILL DIRECTORY (2026-09-06).  build_kb.ExtSort uses a FIXED per-dataset work dir
# ({WORK}/2wiki_runs, {WORK}/hotpot_runs) and shutil.rmtree()s it in __init__, so this synthetic suite -- which
# calls the real build_corpus_* functions on tiny fixtures -- silently DESTROYS a concurrent real build of the
# same dataset.  That happened once for real: running this file while qi_test.py's 2wiki run B was merging wiped
# _work/2wiki_runs/ and the build died with FileNotFoundError on run_0000.txt.  Repointing the module global at a
# temp dir makes the suite safe to run at any time.  build_corpus_* read WORK at call time, so this is sufficient,
# and it is a change to the TEST, not to build_kb.py -- the builder's sha256 is unaffected.
K.WORK = f"{TMP}/_work"
os.makedirs(K.WORK, exist_ok=True)


def chk(name, cond, extra=""):
    OK.append((name, bool(cond), extra))
    print(("PASS " if cond else "FAIL ") + name + ((" | " + str(extra)) if extra else ""), flush=True)


# ------------------------------------------------------------------ webqsp
def mk_rog(path, rows):
    t = pa.table({"id": pa.array([r["id"] for r in rows], pa.string()),
                  "question": pa.array([r["q"] for r in rows], pa.string()),
                  "answer": pa.array([r["a"] for r in rows], pa.list_(pa.string())),
                  "q_entity": pa.array([r["qe"] for r in rows], pa.list_(pa.string())),
                  "a_entity": pa.array([r["ae"] for r in rows], pa.list_(pa.string())),
                  "graph": pa.array([r["g"] for r in rows], pa.list_(pa.list_(pa.string())))})
    pq.write_table(t, path)


wd = f"{TMP}/webqsp"; os.makedirs(wd)
mk_rog(f"{wd}/train.parquet", [
    {"id": "WebQTrn-0", "q": "q0", "a": ["Alpha"], "qe": ["Beta"], "ae": ["Alpha"],
     "g": [["Beta", "rel.a", "Alpha"], ["Beta", "rel.b", "m.0abc"]]},
    {"id": "WebQTrn-1", "q": "q1", "a": ["Gamma"], "qe": ["Beta"], "ae": ["Gamma", "ONLY_IN_A_ENTITY"],
     "g": [["Beta", "rel.a", "Gamma"]]},
])
mk_rog(f"{wd}/test.parquet", [
    {"id": "WebQTest-0", "q": "q2", "a": ["Alpha"], "qe": ["ONLY_IN_Q_ENTITY"], "ae": ["Alpha"],
     "g": [["Delta", "rel.c", "Alpha"], ["", "rel.d", "Delta"]]},
])
S_W = {"rog": [("train", f"{wd}/train.parquet"), ("test", f"{wd}/test.parquet")],
       "official": {"train": f"{wd}/WebQSP.train.json", "test": f"{wd}/WebQSP.test.json"},
       "held_out_split": "test"}
json.dump({"Questions": [
    {"QuestionId": "WebQTrn-0", "RawQuestion": "q0", "ProcessedQuestion": "q0",
     "Parses": [{"TopicEntityName": "Beta", "Answers": [{"AnswerType": "Entity", "AnswerArgument": "m.x", "EntityName": "Alpha"}]}]},
    {"QuestionId": "WebQTrn-1", "RawQuestion": "q1", "ProcessedQuestion": "q1",
     "Parses": [{"TopicEntityName": "Beta", "Answers": [{"AnswerType": "Entity", "AnswerArgument": "m.y", "EntityName": "Gamma"},
                                                        {"AnswerType": "Entity", "AnswerArgument": "m.z", "EntityName": "NEVER_IN_GRAPH"}]}]},
]}, open(f"{wd}/WebQSP.train.json", "w", encoding="utf-8"))
json.dump({"Questions": [
    {"QuestionId": "WebQTest-0", "RawQuestion": "q2", "ProcessedQuestion": "q2",
     "Parses": [{"TopicEntityName": "Delta", "Answers": [{"AnswerType": "Entity", "AnswerArgument": "m.x", "EntityName": "Alpha"}]}]},
]}, open(f"{wd}/WebQSP.test.json", "w", encoding="utf-8"))

out_w = f"{TMP}/out_webqsp"
W, acc, names = K.build_corpus_webqsp(S_W, out_w)
n, oh, ch = W.close()
nodes = [json.loads(l) for l in open(f"{out_w}/nodes.jsonl", encoding="utf-8")]
surf = {x["source_id"] for x in nodes}
chk("webqsp: a_entity-only entity EXCLUDED", "ONLY_IN_A_ENTITY" not in surf, sorted(surf))
chk("webqsp: q_entity-only entity EXCLUDED", "ONLY_IN_Q_ENTITY" not in surf)
chk("webqsp: expected node set", surf == {"Alpha", "Beta", "Gamma", "m.0abc", "Delta"}, sorted(surf))
chk("webqsp: empty endpoint dropped", len(acc["dropped_records"]) == 1, acc["dropped_records"])
chk("webqsp: accounting identity", acc["raw_source_records"] == n + acc["duplicates_collapsed"] + len(acc["dropped_records"]),
    (acc["raw_source_records"], n, acc["duplicates_collapsed"], len(acc["dropped_records"])))
alpha = [x for x in nodes if x["source_id"] == "Alpha"][0]
chk("webqsp: split_provenance union", alpha["split_provenance"] == ["train", "test"], alpha["split_provenance"])
chk("webqsp: n_source_records counts mentions", alpha["n_source_records"] == 2, alpha["n_source_records"])
chk("webqsp: MID flagged", [x for x in nodes if x["source_id"] == "m.0abc"][0]["is_raw_mid"] is True)
chk("webqsp: text == title == verbatim entity", all(x["text"] == x["title"] == x["source_id"] for x in nodes))
chk("webqsp: order lexicographic", [x["node_id"] for x in nodes] == sorted(x["node_id"] for x in nodes))
chk("webqsp: content_hash formula", alpha["content_hash"] == hashlib.sha256(("Alpha\x1fAlpha").encode()).hexdigest())

qstats, U, index = K.build_queries_webqsp(S_W, names, out_w)
tr = [json.loads(l) for l in open(f"{out_w}/queries/train.jsonl", encoding="utf-8")]
q1 = [r for r in tr if r["query_id"] == "WebQTrn-1"][0]
chk("webqsp: unresolvable official answer reported, not invented",
    q1["gold_refs"] == ["Gamma", "NEVER_IN_GRAPH"] and len(q1["gold_node_ids"]) == 1 and U.counts["train"] == 1,
    (q1["gold_node_ids"], dict(U.counts)))
chk("webqsp: rog cross-check present", q1["rog_answer_entities"] == ["Gamma", "ONLY_IN_A_ENTITY"] and len(q1["rog_gold_node_ids"]) == 1,
    (q1["rog_answer_entities"], q1["rog_gold_node_ids"]))
chk("webqsp: rog_split recorded", q1["rog_split"] == "train")
chk("webqsp: qstats honest (1 of 2 fully resolved)", qstats["train"]["n_all_gold_resolved"] == 1 and qstats["train"]["n_with_gold_refs"] == 2,
    qstats["train"])

K.out_dir_for_fetch[0] = out_w
sub, info = K.random_held_subset("webqsp", index, 7, n=5)
chk("webqsp: random_held draws from held-out split only", all(r["split"] == "test" for r in sub) and len(sub) == 1, info)

# corpus is invariant to the query subset: rebuild the corpus alone, compare hashes
out_w2 = f"{TMP}/out_webqsp2"
W2, _, _ = K.build_corpus_webqsp(S_W, out_w2)
n2, oh2, ch2 = W2.close()
chk("webqsp: corpus deterministic across rebuilds", (n, oh, ch) == (n2, oh2, ch2), (ch[:16], ch2[:16]))


# ------------------------------------------------------------------ hotpotqa
hd = f"{TMP}/hotpot"; os.makedirs(hd)
recs = []
for i in range(1, 61):
    recs.append({"id": str(1000 + i), "title": f"Art {i}", "url": f"u{i}",
                 "text": [f"Sentence {i} one.", f" tail {i}"], "text_with_links": ["<a>x</a>"]})
recs.append({"id": "2001", "title": "  Leading", "url": "u", "text": ["   starts with space"]})   # strip delta
recs.append({"id": "2002", "title": "Empty", "url": "u", "text": []})                              # empty abstract kept
recs.append({"id": "2005", "title": "Links only", "url": "u", "text": [],
             "text_with_links": ["<a href='x'>only here</a>"]})   # Phase-C would have fallen back -> not reusable
recs.append({"id": "2006", "title": "", "url": "u", "text": []})   # empty body AND empty title -> stays empty
recs.append({"id": "2003", "title": "Dup Title", "url": "u", "text": ["first"]})
recs.append({"id": "2004", "title": "Dup Title", "url": "u", "text": ["second"]})                  # ambiguous title
recs.append({"id": "1001", "title": "Art 1", "url": "u1", "text": ["Sentence 1 one.", " tail 1"]})  # duplicate curid
tb = f"{hd}/fake.tar.bz2"
with tarfile.open(tb, "w:bz2") as tar:
    for gi in range(0, len(recs), 12):
        chunk = recs[gi:gi + 12]
        blob = bz2.compress("\n".join(json.dumps(r) for r in chunk).encode("utf-8"))
        ti = tarfile.TarInfo(f"enwiki-fake/AA/wiki_{gi:02d}.bz2"); ti.size = len(blob)
        tar.addfile(ti, io.BytesIO(blob))


def mk_hq(path, rows):
    pq.write_table(pa.table({
        "id": pa.array([r["id"] for r in rows], pa.string()),
        "question": pa.array([r["q"] for r in rows], pa.string()),
        "answer": pa.array([r["a"] for r in rows], pa.string()),
        "type": pa.array(["comparison"] * len(rows), pa.string()),
        "level": pa.array(["hard"] * len(rows), pa.string()),
        "supporting_facts": pa.array([{"title": r["sf"], "sent_id": [0] * len(r["sf"])} for r in rows],
                                     pa.struct([("title", pa.list_(pa.string())), ("sent_id", pa.list_(pa.int32()))])),
        "context": pa.array([{"title": r["ctx"], "sentences": [["s"] for _ in r["ctx"]]} for r in rows],
                            pa.struct([("title", pa.list_(pa.string())), ("sentences", pa.list_(pa.list_(pa.string())))])),
    }), path)


mk_hq(f"{hd}/train.parquet", [{"id": "t1", "q": "Q1", "a": "A1", "sf": ["Art 1", "Art 2"], "ctx": ["Art 1", "Art 2", "Art 3"]},
                              {"id": "t2", "q": "Q2", "a": "A2", "sf": ["Dup Title"], "ctx": ["Dup Title", "Art 9"]}])
mk_hq(f"{hd}/val.parquet", [{"id": "v1", "q": "Q3", "a": "A3", "sf": ["Art 5", "MISSING TITLE"], "ctx": ["Art 5"]}])
mk_hq(f"{hd}/test.parquet", [{"id": "x1", "q": "Q4", "a": "", "sf": [], "ctx": ["Art 7"]}])
S_H = {"tarball": tb, "queries": [("train", f"{hd}/train.parquet"), ("validation", f"{hd}/val.parquet"),
                                  ("test", f"{hd}/test.parquet")], "held_out_split": "validation"}

out_h = f"{TMP}/out_hotpot"
K.SPILL = 7                              # force many spill runs + a real k-way merge
WH, accH, _ = K.build_corpus_hotpotqa(S_H, out_h)
nh, ohh, chh = WH.close()
hn = [json.loads(l) for l in open(f"{out_h}/nodes.jsonl", encoding="utf-8")]
byid = {x["source_id"]: x for x in hn}
chk("hotpot: many spill runs merged", accH["external_sort_runs"] >= 5, accH["external_sort_runs"])
chk("hotpot: strictly lexicographic node_id order", [x["node_id"] for x in hn] == sorted(x["node_id"] for x in hn))
chk("hotpot: duplicate curid collapsed", nh == 66 and accH["duplicates_collapsed"] == 1, (nh, accH["duplicates_collapsed"]))
chk("hotpot: dup bumps n_source_records", byid["1001"]["n_source_records"] == 2, byid["1001"]["n_source_records"])
chk("hotpot: accounting identity", accH["raw_source_records"] == nh + accH["duplicates_collapsed"] + len(accH["dropped_records"]))
chk("hotpot: NO strip applied", byid["2001"]["text"] == "   starts with space", repr(byid["2001"]["text"]))
pcd = accH["phase_c_divergence"]
chk("hotpot: strip delta counted exactly", pcd["n_records_where_strip_changes_text"] == 1, pcd)
chk("hotpot: Phase-C text_with_links fallback counted", pcd["n_records_where_phase_c_fell_back_to_text_with_links"] == 1 and
    pcd["n_records_with_a_different_phase_c_encoder_input"] == 2, pcd)
# --- TEXTUALIZATION_REV 2: empty body -> source-native title (fallback only) ---
tfb = accH["textualization_fallback"]
chk("hotpot: nonempty body UNCHANGED by the fallback", byid["1002"]["text"] == "Sentence 2 one.  tail 2" and
    byid["2001"]["text"] == "   starts with space", (repr(byid["1002"]["text"]), repr(byid["2001"]["text"])))
chk("hotpot: body empty + title nonempty -> TITLE fallback", byid["2005"]["text"] == "Links only" and
    byid["2002"]["text"] == "Empty", (repr(byid["2005"]["text"]), repr(byid["2002"]["text"])))
chk("hotpot: body empty + title empty -> STAYS empty", byid["2006"]["text"] == "", repr(byid["2006"]["text"]))
chk("hotpot: no title is PREPENDED to records that already have text",
    not byid["1002"]["text"].startswith(byid["1002"]["title"]) and "Art 2" not in byid["1002"]["text"])
chk("hotpot: empty abstract KEPT (membership unchanged)", all(k in byid for k in ("2002", "2005", "2006")) and
    accH["empty_abstract_nodes"]["n"] == 3, accH["empty_abstract_nodes"]["n"])
chk("hotpot: fallback counters exact", tfb["TEXTUALIZATION_REV"] == 2 and tfb["empty_before_fallback"] == 3 and
    tfb["title_fallback_applied"] == 2 and tfb["still_empty_after_fallback"] == 1, tfb)
chk("hotpot: content_hash follows the FINAL (post-fallback) text", byid["2002"]["content_hash"] ==
    hashlib.sha256(("2002\x1f" + "Empty").encode()).hexdigest())
chk("hotpot: only-empty-body records changed vs the pre-REV2 rule",
    sum(1 for x in hn if x["text"] == "") == 1, sum(1 for x in hn if x["text"] == ""))
chk("hotpot: text = ' '.join(non-empty sentences)", byid["1002"]["text"] == "Sentence 2 one.  tail 2", repr(byid["1002"]["text"]))
chk("hotpot: text_with_links NOT used", "<a>" not in json.dumps(hn))
chk("hotpot: content_hash formula", byid["1002"]["content_hash"] ==
    hashlib.sha256(("1002\x1f" + byid["1002"]["text"]).encode()).hexdigest())

qsH, UH, idxH = K.build_queries_hotpotqa(S_H, out_h)
trh = {r["query_id"]: r for r in (json.loads(l) for l in open(f"{out_h}/queries/train.jsonl", encoding="utf-8"))}
vah = {r["query_id"]: r for r in (json.loads(l) for l in open(f"{out_h}/queries/validation.jsonl", encoding="utf-8"))}
teh = {r["query_id"]: r for r in (json.loads(l) for l in open(f"{out_h}/queries/test.jsonl", encoding="utf-8"))}
chk("hotpot: gold titles resolved to curid nodes IN OFFICIAL supporting-fact order",
    trh["t1"]["gold_refs"] == ["Art 1", "Art 2"] and
    trh["t1"]["gold_node_ids"] == [byid["1001"]["node_id"], byid["1002"]["node_id"]],
    trh["t1"]["gold_node_ids"])
chk("hotpot: ambiguous title keeps BOTH curids and is flagged",
    len(trh["t2"]["gold_node_ids"]) == 2 and trh["t2"]["gold_title_ambiguous"] == ["Dup Title"], trh["t2"]["gold_node_ids"])
chk("hotpot: missing gold title reported", UH.counts["validation"] == 1 and len(vah["v1"]["gold_node_ids"]) == 1, dict(UH.counts))
chk("hotpot: hidden-label test split has no gold", teh["x1"]["gold_refs"] == [] and teh["x1"]["gold_node_ids"] == [])
chk("hotpot: context_node_ids aligned with context_titles",
    len(trh["t1"]["context_node_ids"]) == len(trh["t1"]["context_titles"]) == 3)
chk("hotpot: title resolution stats", qsH["_title_resolution"]["ambiguous_titles"] == 1 and
    qsH["_title_resolution"]["titles_unmatched"] == 1, qsH["_title_resolution"])
chk("hotpot: tmp query files removed", not [f for f in os.listdir(f"{out_h}/queries") if f.endswith(".tmp")])
chk("hotpot: spill run dir cleaned", not os.path.isdir(f"{K.WORK}/hotpot_runs"))

out_h2 = f"{TMP}/out_hotpot2"
K.SPILL = 5000                            # different spill size -> different run layout, same corpus
WH2, _, _ = K.build_corpus_hotpotqa(S_H, out_h2)
nh2, ohh2, chh2 = WH2.close()
chk("hotpot: corpus invariant to spill size (deterministic)", (nh, ohh, chh) == (nh2, ohh2, chh2), (chh[:16], chh2[:16]))
chk("hotpot: nodes.jsonl byte-identical across spill sizes",
    open(f"{out_h}/nodes.jsonl", "rb").read() == open(f"{out_h2}/nodes.jsonl", "rb").read())

# --- TEXTUALIZATION_REV 2: the title fallback must not open a query->text channel ---
out_h3 = f"{TMP}/out_hotpot3"
mk_hq(f"{hd}/other.parquet", [{"id": "z9", "q": "totally different", "a": "Empty",
                               "sf": ["Empty", "Links only"], "ctx": ["Empty", "Links only", "Art 4"]}])
S_H_alt = dict(S_H, queries=[("train", f"{hd}/other.parquet"), ("validation", f"{hd}/other.parquet"),
                             ("test", f"{hd}/other.parquet")])
WH3, accH3, _ = K.build_corpus_hotpotqa(S_H_alt, out_h3)
nh3, ohh3, chh3 = WH3.close()
chk("hotpot: title fallback is QUERY-INDEPENDENT (different query set -> identical corpus)",
    (nh, ohh, chh) == (nh3, ohh3, chh3), (chh[:16], chh3[:16]))
chk("hotpot: node ids + order unaffected by the fallback",
    [json.loads(l)["node_id"] for l in open(f"{out_h3}/nodes.jsonl", encoding="utf-8")] ==
    [x["node_id"] for x in hn])
chk("hotpot: fallback text comes from the SOURCE title, never the query/gold",
    accH3["textualization_fallback"]["title_fallback_applied"] == 2 and
    json.loads([l for l in open(f"{out_h3}/nodes.jsonl", encoding="utf-8")
                if '"source_id": "2005"' in l][0])["text"] == "Links only")

# ------------------------------------------------------------------ 2wiki (FULL para_with_hyperlink universe)
import zipfile

wd2 = f"{TMP}/2wiki"; os.makedirs(wd2)
ART = [
    {"id": "1001", "title": "Art A", "sentences": ["A one.", "A two."], "mentions": [{"ref_ids": ["1002"]}]},
    {"id": "1002", "title": "Art B", "sentences": ["B one."], "mentions": []},
    {"id": "1003", "title": "  Leading", "sentences": ["  starts with space"], "mentions": []},   # strip-delta
    {"id": "1004", "title": "Empty", "sentences": [], "mentions": []},        # empty body + title -> TITLE fallback
    {"id": "1005", "title": "Dup Title", "sentences": ["First of two."], "mentions": []},         # ambiguous title
    {"id": "1006", "title": "Dup Title", "sentences": ["Second of two."], "mentions": []},
    {"id": "1007", "title": "Never referenced", "sentences": ["Nobody asks about me."], "mentions": []},
    {"id": "1008", "title": "", "sentences": [], "mentions": []},             # empty body AND title -> stays empty
    {"id": "1005", "title": "Dup Title", "sentences": ["First of two."], "mentions": []},         # duplicate curid
]
zp = f"{wd2}/para_with_hyperlink.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("para_with_hyperlink.jsonl",
               "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in ART))
for split, qs in [("train", [
        {"_id": "w1", "question": "qw1", "answer": "a1", "type": "compositional",
         "supporting_facts": [["Art A", 0], ["Art B", 0]], "evidences": [],
         "context": [["Art A", ["A one.", "A two."]], ["Art B", ["B one."]], ["Missing Article", ["x"]]]},
        {"_id": "w2", "question": "qw2", "answer": "a2", "type": "comparison",
         "supporting_facts": [["Dup Title", 0]], "evidences": [], "context": [["Dup Title", ["First of two."]]]}]),
    ("dev", [{"_id": "w3", "question": "qw3", "answer": "a3", "type": "bridge",
              "supporting_facts": [["Art A", 0], ["Missing Article", 0]], "evidences": [],
              "context": [["Art A", ["A one."]]]}]),
    ("test", [{"_id": "w4", "question": "qw4", "answer": "", "type": "bridge",
               "supporting_facts": [], "evidences": [], "context": [["Art B", ["B one."]]]}])]:
    json.dump(qs, open(f"{wd2}/{split}.json", "w", encoding="utf-8"))
S_2 = {"zip": zp, "member": "para_with_hyperlink.jsonl", "base": wd2,
       "splits": ["train", "dev", "test"], "held_out_split": "dev"}

K.SPILL = 3
out_w = f"{TMP}/out_2wiki"
W2, A2, _ = K.build_corpus_2wiki(S_2, out_w)
n2, oh2, ch2 = W2.close()
nodes2 = [json.loads(l) for l in open(f"{out_w}/nodes.jsonl", encoding="utf-8")]
by2 = {n["source_id"]: n for n in nodes2}
chk("2wiki: node_id is readable 2wiki:c<curid>", by2["1001"]["node_id"] == "2wiki:c1001", by2["1001"]["node_id"])
chk("2wiki: strictly lexicographic node_id order",
    [n["node_id"] for n in nodes2] == sorted(n["node_id"] for n in nodes2))
chk("2wiki: duplicate curid collapsed", n2 == 8 and A2["duplicates_collapsed"] == 1, (n2, A2["duplicates_collapsed"]))
chk("2wiki: dup bumps n_source_records", by2["1005"]["n_source_records"] == 2, by2["1005"]["n_source_records"])
chk("2wiki: accounting identity",
    A2["raw_source_records"] == n2 + A2["duplicates_collapsed"] + len(A2["dropped_records"]))
chk("2wiki: text = ' '.join(sentences), NO strip", by2["1003"]["text"] == "  starts with space", repr(by2["1003"]["text"]))
chk("2wiki: strip delta counted exactly", A2["phase_c_divergence"]["n_records_where_strip_changes_text"] == 1,
    A2["phase_c_divergence"]["n_records_where_strip_changes_text"])
chk("2wiki: Phase-C divergence = strip delta + retextualized",
    A2["phase_c_divergence"]["n_records_retextualized_from_title"] == 1 and
    A2["phase_c_divergence"]["n_records_with_a_different_phase_c_encoder_input"] == 2,
    A2["phase_c_divergence"])
# --- TEXTUALIZATION_REV 2: empty body -> source-native title (fallback only), same rule as hotpotqa ---
tfb2 = A2["textualization_fallback"]
chk("2wiki: nonempty body UNCHANGED by the fallback", by2["1001"]["text"] == "A one. A two." and
    by2["1003"]["text"] == "  starts with space", (repr(by2["1001"]["text"]), repr(by2["1003"]["text"])))
chk("2wiki: body empty + title nonempty -> TITLE fallback", by2["1004"]["text"] == "Empty", repr(by2["1004"]["text"]))
chk("2wiki: body empty + title empty -> STAYS empty", by2["1008"]["text"] == "", repr(by2["1008"]["text"]))
chk("2wiki: no title is PREPENDED to records that already have text",
    not by2["1001"]["text"].startswith(by2["1001"]["title"]) and "Art A" not in by2["1001"]["text"])
chk("2wiki: empty article KEPT (membership unchanged)", all(k in by2 for k in ("1004", "1008")) and
    A2["empty_text_nodes"]["n"] == 2, A2["empty_text_nodes"]["n"])
chk("2wiki: only the title-less record is still empty", W2.empty_text == 1 and
    sum(1 for n in nodes2 if n["text"] == "") == 1, W2.empty_text)
chk("2wiki: fallback counters exact", tfb2["TEXTUALIZATION_REV"] == 2 and tfb2["empty_before_fallback"] == 2 and
    tfb2["title_fallback_applied"] == 1 and tfb2["still_empty_after_fallback"] == 1, tfb2)
chk("2wiki: content_hash follows the FINAL (post-fallback) text", by2["1004"]["content_hash"] ==
    K.sha("1004" + K.SEP + "Empty"))
chk("2wiki: content_hash formula", by2["1001"]["content_hash"] == K.sha("1001" + K.SEP + "A one. A two."))
chk("2wiki: multi-sentence join", by2["1001"]["text"] == "A one. A two.", by2["1001"]["text"])
chk("2wiki: article never referenced by a question IS a node (not a context union)", "1007" in by2)

qs2, U2, ix2 = K.build_queries_2wiki(S_2, out_w)
tr2 = {r["query_id"]: r for r in (json.loads(l) for l in open(f"{out_w}/queries/train.jsonl", encoding="utf-8"))}
dv2 = {r["query_id"]: r for r in (json.loads(l) for l in open(f"{out_w}/queries/dev.jsonl", encoding="utf-8"))}
te2 = {r["query_id"]: r for r in (json.loads(l) for l in open(f"{out_w}/queries/test.jsonl", encoding="utf-8"))}
chk("2wiki: gold titles resolved in official supporting-fact order",
    tr2["w1"]["gold_node_ids"] == ["2wiki:c1001", "2wiki:c1002"], tr2["w1"]["gold_node_ids"])
chk("2wiki: ambiguous title keeps BOTH curids and is flagged",
    tr2["w2"]["gold_node_ids"] == ["2wiki:c1005", "2wiki:c1006"] and tr2["w2"]["gold_title_ambiguous"] == ["Dup Title"],
    tr2["w2"]["gold_node_ids"])
chk("2wiki: missing gold title REPORTED, never injected into the corpus",
    U2.counts["dev"] == 1 and dv2["w3"]["gold_node_ids"] == ["2wiki:c1001"] and
    not any(n["title"] == "Missing Article" for n in nodes2), dict(U2.counts))
chk("2wiki: unlabelled test question carries no gold", te2["w4"]["gold_refs"] == [] and te2["w4"]["gold_node_ids"] == [])
chk("2wiki: context_node_ids aligned, missing context title -> None",
    tr2["w1"]["context_node_ids"] == ["2wiki:c1001", "2wiki:c1002", None], tr2["w1"]["context_node_ids"])
chk("2wiki: title resolution stats", qs2["_title_resolution"]["titles_unmatched"] == 1 and
    qs2["_title_resolution"]["ambiguous_titles"] == 1, qs2["_title_resolution"])
chk("2wiki: tmp query files removed", not [f for f in os.listdir(f"{out_w}/queries") if f.endswith(".tmp")])
chk("2wiki: spill run dir cleaned", not os.path.isdir(f"{K.WORK}/2wiki_runs"))

out_w2 = f"{TMP}/out_2wiki_b"
K.SPILL = 5000
W2b, _, _ = K.build_corpus_2wiki(S_2, out_w2)
n2b, oh2b, ch2b = W2b.close()
chk("2wiki: corpus invariant to spill size (deterministic)", (n2, oh2, ch2) == (n2b, oh2b, ch2b), (ch2[:16], ch2b[:16]))
chk("2wiki: nodes.jsonl byte-identical across spill sizes",
    open(f"{out_w}/nodes.jsonl", "rb").read() == open(f"{out_w2}/nodes.jsonl", "rb").read())

# --- TEXTUALIZATION_REV 2: the title fallback must not open a query->text channel ---
# The corpus pass never opens a question file, but the fallback is the first rule that CHOOSES text at build time,
# so prove constructively that a completely different question set leaves the corpus bit-identical.
for split, qs in [("train", [{"_id": "z1", "question": "totally different", "answer": "Empty",
                              "supporting_facts": [["Empty", 0], ["Never referenced", 0]], "evidences": [],
                              "context": [["Empty", ["x"]], ["Never referenced", ["y"]]]}]),
                  ("dev", []), ("test", [])]:
    json.dump(qs, open(f"{wd2}/alt_{split}.json", "w", encoding="utf-8"))
alt2 = f"{wd2}/alt"; os.makedirs(alt2, exist_ok=True)
for split in ("train", "dev", "test"):
    shutil.copy(f"{wd2}/alt_{split}.json", f"{alt2}/{split}.json")
S_2_alt = dict(S_2, base=alt2)
K.SPILL = 3
out_w3 = f"{TMP}/out_2wiki_c"
W2c, A2c, _ = K.build_corpus_2wiki(S_2_alt, out_w3)
n2c, oh2c, ch2c = W2c.close()
chk("2wiki: title fallback is QUERY-INDEPENDENT (different query set -> identical corpus)",
    (n2, oh2, ch2) == (n2c, oh2c, ch2c), (ch2[:16], ch2c[:16]))
chk("2wiki: node ids + order unaffected by the fallback",
    [json.loads(l)["node_id"] for l in open(f"{out_w3}/nodes.jsonl", encoding="utf-8")] ==
    [n["node_id"] for n in nodes2])
chk("2wiki: fallback text comes from the SOURCE title, never the query/gold",
    A2c["textualization_fallback"]["title_fallback_applied"] == 1 and
    json.loads([l for l in open(f"{out_w3}/nodes.jsonl", encoding="utf-8")
                if '"source_id": "1004"' in l][0])["text"] == "Empty")

# ------------------------------------------------------------------ NodeWriter == build.py::write_nodes
sys.path.insert(0, os.path.abspath("scratchpad/final_canonical_build"))
import build as B
outc = f"{TMP}/contract"; os.makedirs(outc)
ids_b, ohb, chb = B.write_nodes([dict(n) for n in nodes2], outc)
chk("contract: NodeWriter == build.py::write_nodes (hashes)", (ohb, chb) == (oh2, ch2), (chb[:16], ch2[:16]))
chk("contract: NodeWriter == build.py::write_nodes (bytes)",
    open(f"{outc}/nodes.jsonl", "rb").read() == open(f"{out_w}/nodes.jsonl", "rb").read())

# ------------------------------------------------------------------ per-dataset approval gate
import subprocess
def _gate(ds, extra=()):
    r = subprocess.run([sys.executable, "scratchpad/final_canonical_build/build_kb.py", ds,
                        "--eval-subset", "none", "--out", f"{TMP}/gate_{ds}", *extra],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"))
    return r.returncode, (r.stdout + r.stderr)

rc, msg = _gate("webqsp")
chk("gate: webqsp REFUSED unconditionally", rc != 0 and "REFUSING" in msg and "BLOCKED_PENDING_FREEBASE_SOURCE" in msg,
    msg.strip().splitlines()[0][:90] if msg.strip() else "")
chk("gate: webqsp refusal wrote no nodes", not os.path.exists(f"{TMP}/gate_webqsp/nodes.jsonl"))
_gate_src = open("scratchpad/final_canonical_build/build_kb.py", encoding="utf-8").read() \
    .split("def require_gates")[1].split("def _source_paths")[0]
chk("gate: the obsolete all-or-nothing file is never tested for existence",
    "os.path.exists(ok)" not in _gate_src and 'exists(f"{ROOT}/_APPROVED_SOURCE_CONTRACTS")' not in _gate_src)
chk("gate: refusal is keyed on the dataset name, not only on the json",
    "NEVER_BUILD" in _gate_src and K.NEVER_BUILD == {"webqsp"}, sorted(K.NEVER_BUILD))
chk("gate: heavy corpora both gated on _GATE_HOTPOT_HEAVY_OK", K.HEAVY == {"hotpotqa", "2wiki"}, sorted(K.HEAVY))
chk("gate: approvals file is the only approval source", K.APPROVALS.endswith("_APPROVALS.json"), K.APPROVALS)

shutil.rmtree(TMP, ignore_errors=True)
bad = [n for n, c, _ in OK if not c]
print(f"\n{sum(1 for _, c, _ in OK if c)}/{len(OK)} checks passed" + (f" | FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
