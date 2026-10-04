"""Build the multi-source Freebase QUESTION SET that maps onto the frozen 302M-node Freebase canonical tree.

  PYTHONHASHSEED=0 python scratchpad/_fbq_build_set.py [--smoke N] [--outdir DIR]

Outputs (write-once, supersede-never-edit; refuses to overwrite):
  results/FREEBASE_SCALE/FBQ_SET__v1.parquet   one row per question
  results/FREEBASE_SCALE/FBQ_SET__v1.json      record (counts, resolution, subsets, census, shas)
  results/FREEBASE_SCALE/FBQ_SET__v1.md        short summary

Sources: NSM release (data/final_canonical/webqsp/_acquisition/nsm/extracted/{webqsp/webqsp,CWQ/CWQ}) train_simple.json + dev_simple.json ONLY.
HELD-OUT RULE: test_simple.json is never parsed. Only its byte size, newline count (binary chunked read) and sha256 are recorded ('sealed, not read').
The NSM `subgraph` field is never parsed: the object prefix up to ', "subgraph": ' is closed with '}' and json-loaded.
Enrichment (same train/dev ids only, never test): RoG parquets (train + validation files only; columns id/question/answer/q_entity/a_entity -- the
huge `graph` column is not requested) and the original WebQSP.train.json (train questions only; Parses -> topic MID, chain, SPARQL).
Position rule: uid = hash(mid.encode('utf-8')) under PYTHONHASHSEED=0 -> merge-join against nodes/node_uid.npy (sorted int64) with an equality check;
kind from nodes/kind.npy. The join streams node_uid.npy in 64 MB blocks (no mmap, bounded RSS).
Nothing under data/final_canonical/** is modified."""
import argparse
import collections
import glob
import gzip  # noqa: F401  (kept for the .jsonl.gz fallback described in the task; pyarrow is available so unused)
import hashlib
import json
import os
import re
import sys
import time

assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSM_ROOT = os.path.join(ROOT, "data", "final_canonical", "webqsp", "_acquisition", "nsm", "extracted")
SRC = {"nsm_webqsp": os.path.join(NSM_ROOT, "webqsp", "webqsp"), "nsm_cwq": os.path.join(NSM_ROOT, "CWQ", "CWQ")}
FB = os.path.join(ROOT, "data", "final_canonical", "freebase")
ORIG_WQSP = os.path.join(ROOT, "data", "original", "webqsp", "WebQSP", "data", "WebQSP.train.json")
ROG = {"nsm_webqsp": os.path.join(ROOT, "data", "original", "webqsp", "rog_webqsp"), "nsm_cwq": os.path.join(ROOT, "data", "original", "cwq", "rog_cwq")}
POP = os.path.join(ROOT, "results", "L3_DEV", "l3w_population_webqsp__v1.json")
KINDS = ["ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER", "EXTERNAL_URI", "LITERAL"]
SUB = b', "subgraph": '
CODE_PATH = os.path.abspath(__file__)

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", type=int, default=0, help="rows per NSM file (0 = full run); smoke writes to --outdir (default: system temp), never to results/")
ap.add_argument("--outdir", default=None)
args = ap.parse_args()
SMOKE = args.smoke if args.smoke > 0 else None
OUTDIR = args.outdir or (os.path.join(os.environ.get("TEMP", "."), "fbq_smoke") if SMOKE else os.path.join(ROOT, "results", "FREEBASE_SCALE"))
os.makedirs(OUTDIR, exist_ok=True)
OUT_PQ = os.path.join(OUTDIR, "FBQ_SET__v1.parquet")
OUT_JSON = os.path.join(OUTDIR, "FBQ_SET__v1.json")
OUT_MD = os.path.join(OUTDIR, "FBQ_SET__v1.md")
for p in (OUT_PQ, OUT_JSON, OUT_MD):
    assert not os.path.exists(p), "write-once: refusing to overwrite " + p

peak = {"rss": 0}


def rss_note():
    mi = psutil.Process().memory_info()
    peak["rss"] = max(peak["rss"], getattr(mi, "peak_wset", mi.rss), mi.rss)


def sha_file(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_lines_and_sha(path, chunk=16 << 20):
    """Binary chunked newline count + sha256; the content is never decoded or parsed."""
    h = hashlib.sha256()
    n = 0
    last = b"\n"
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += b.count(b"\n")
            last = b[-1:]
    if last != b"\n":
        n += 1
    return n, h.hexdigest()


def is_mid(k):
    return isinstance(k, str) and k[:2] in ("m.", "g.")


def parse_nsm(path, source, split, limit, want_sha):
    sha = hashlib.sha256() if want_sha else None
    rows, n_lines, n_nosub, n_blank = [], 0, 0, 0
    with open(path, "rb") as f:
        for line in f:
            if sha is not None:
                sha.update(line)
            n_lines += 1
            if not line.strip():
                n_blank += 1
                continue
            if limit is not None and len(rows) >= limit:
                break
            i = line.find(SUB)
            if i < 0:
                n_nosub += 1
                d = json.loads(line)
                d.pop("subgraph", None)
            else:
                d = json.loads(line[:i] + b"}")
            for k in ("id", "question", "entities", "answers"):
                assert k in d, (path, n_lines, k)
            rows.append({"source": source, "split": split, "qid": d["id"], "question": d["question"], "ent_idx": d["entities"], "answers": d["answers"]})
    rss_note()
    return rows, {"file": os.path.relpath(path, ROOT).replace("\\", "/"), "bytes": os.path.getsize(path), "lines_read": n_lines, "blank_lines": n_blank,
                  "rows_parsed": len(rows), "lines_without_subgraph_key": n_nosub, "sha256": sha.hexdigest() if (sha is not None and limit is None) else None,
                  "sha256_note": None if limit is None else "smoke run: partial read, hash omitted"}


def norm1(q):
    return " ".join(q.lower().strip().split())


def norm2(q):
    return " ".join(re.sub(r"[\W_]+", " ", norm1(q), flags=re.UNICODE).split())


def gid(s):
    return "dq_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def resolve_positions(mids):
    """merge-join the unique MID strings against nodes/node_uid.npy (sorted int64). Returns (pos array aligned to mids, kind array aligned, info)."""
    q = np.fromiter((hash(m.encode("utf-8")) for m in mids), dtype=np.int64, count=len(mids))
    order = np.argsort(q, kind="stable")
    qs = q[order]
    n_uid_collisions_among_queries = int(len(qs) - len(np.unique(qs)))
    pos_sorted = np.full(len(qs), -1, dtype=np.int64)
    path = os.path.join(FB, "nodes", "node_uid.npy")
    mm = np.load(path, mmap_mode="r")
    off, n = int(mm.offset), int(mm.shape[0])
    assert mm.dtype == np.int64
    del mm
    h = hashlib.sha256()
    strictly_increasing = True
    prev_last = None
    block = 8_000_000
    with open(path, "rb") as f:
        h.update(f.read(off))
        f.seek(off)
        for lo in range(0, n, block):
            cnt = min(block, n - lo)
            blk = np.fromfile(f, dtype=np.int64, count=cnt)
            assert len(blk) == cnt
            if SMOKE is None:
                h.update(memoryview(blk))
            if len(blk) > 1 and not bool((blk[1:] > blk[:-1]).all()):
                strictly_increasing = False
            if prev_last is not None and not (blk[0] > prev_last):
                strictly_increasing = False
            prev_last = blk[-1]
            a = int(np.searchsorted(qs, blk[0], "left"))
            b = int(np.searchsorted(qs, blk[-1], "right"))
            if b > a:
                sub = qs[a:b]
                idx = np.searchsorted(blk, sub)
                idxc = np.minimum(idx, len(blk) - 1)
                hit = blk[idxc] == sub
                view = pos_sorted[a:b]
                view[hit] = lo + idxc[hit]
        rss_note()
    pos = np.full(len(q), -1, dtype=np.int64)
    pos[order] = pos_sorted
    kind_arr = np.full(len(q), -1, dtype=np.int8)
    kmm = np.load(os.path.join(FB, "nodes", "kind.npy"), mmap_mode="r")
    sel = np.flatnonzero(pos >= 0)
    if len(sel):
        o2 = sel[np.argsort(pos[sel])]
        kind_arr[o2] = kmm[pos[o2]]
    rss_note()
    info = {"n_unique_mids": int(len(mids)), "n_resolved": int((pos >= 0).sum()), "n_node_uid": n, "node_uid_strictly_increasing": strictly_increasing,
            "uid_collisions_among_query_mids": n_uid_collisions_among_queries, "node_uid_sha256_computed": h.hexdigest() if SMOKE is None else None}
    return pos, kind_arr, info


def load_rog(source):
    cols = ["id", "question", "answer", "q_entity", "a_entity"]
    out, dup = {}, 0
    per_split = collections.Counter()
    for sp in ("train", "validation"):  # NEVER test
        for fpath in sorted(glob.glob(os.path.join(ROG[source], sp + "-*.parquet"))):
            pf = pq.ParquetFile(fpath)
            for b in pf.iter_batches(batch_size=512, columns=cols):
                d = b.to_pydict()
                for i, qid in enumerate(d["id"]):
                    if qid in out:
                        dup += 1
                    out[qid] = {"rog_split": sp, "rog_question": d["question"][i], "rog_answer": d["answer"][i], "rog_q_entity": d["q_entity"][i], "rog_a_entity": d["a_entity"][i]}
                    per_split[sp] += 1
    rss_note()
    return out, {"rows_per_rog_split": dict(per_split), "duplicate_ids": dup}


def main():
    t_start = time.time()
    pop = json.load(open(POP, encoding="utf-8"))
    l3w = set(pop["query_ids"])
    assert len(l3w) == 786
    del pop

    # ---------------- 1. parse NSM train+dev (never test) ----------------
    rows, files = [], {}
    sealed = {}
    ents_by_src = {}
    for source, d in SRC.items():
        ents = [l.rstrip("\n") for l in open(os.path.join(d, "entities.txt"), encoding="utf-8")]
        ents_by_src[source] = len(ents)
        for split, fn in (("train", "train_simple.json"), ("dev", "dev_simple.json")):
            r, meta = parse_nsm(os.path.join(d, fn), source, split, SMOKE, want_sha=(SMOKE is None))
            for x in r:
                assert all(0 <= i < len(ents) for i in x["ent_idx"]), (source, x["qid"])
                x["topic_mids"] = [ents[i] for i in x["ent_idx"]]
                del x["ent_idx"]
            rows.extend(r)
            files[source + "/" + fn] = meta
        files[source + "/entities.txt"] = {"file": os.path.relpath(os.path.join(d, "entities.txt"), ROOT).replace("\\", "/"), "lines": len(ents),
                                           "bytes": os.path.getsize(os.path.join(d, "entities.txt")), "sha256": sha_file(os.path.join(d, "entities.txt")) if SMOKE is None else None}
        if SMOKE is None:
            fn = os.path.join(d, "test_simple.json")
            n, sh = count_lines_and_sha(fn)
            sealed[source + "/test_simple.json"] = {"file": os.path.relpath(fn, ROOT).replace("\\", "/"), "bytes": os.path.getsize(fn), "newline_count_rows": n, "sha256": sh,
                                                    "status": "sealed, not read (bytes counted and hashed only, never json-parsed, no id/question/answer touched)"}
        del ents
    print("parsed", len(rows), "rows", round(time.time() - t_start, 1), "s", flush=True)

    # ---------------- 2. resolve all MIDs ----------------
    mids = set()
    for x in rows:
        mids.update(x["topic_mids"])
        for a in x["answers"]:
            if is_mid(a.get("kb_id")):
                mids.add(a["kb_id"])
    mids = sorted(mids)
    pos, kind, rinfo = resolve_positions(mids)
    pmap = dict(zip(mids, pos.tolist()))
    kmap = dict(zip(mids, kind.tolist()))
    print("resolved", rinfo, round(time.time() - t_start, 1), "s", flush=True)

    # ---------------- 3. enrichment ----------------
    rog, rog_meta = {}, {}
    for s in SRC:
        rog[s], rog_meta[s] = load_rog(s)
    orig = {}
    oj = json.load(open(ORIG_WQSP, encoding="utf-8"))
    orig_meta = {"version": oj.get("Version"), "freebase_version": oj.get("FreebaseVersion"), "questions": len(oj["Questions"])}
    for q in oj["Questions"]:
        ps = q.get("Parses") or []
        tm = sorted({p["TopicEntityMid"] for p in ps if p.get("TopicEntityMid")})
        tn = sorted({p["TopicEntityName"] for p in ps if p.get("TopicEntityName")})
        am = sorted({a["AnswerArgument"] for p in ps for a in (p.get("Answers") or []) if a.get("AnswerType") == "Entity" and a.get("AnswerArgument")})
        chain = next((p["InferentialChain"] for p in ps if p.get("InferentialChain")), None)
        sparql = next((p["Sparql"] for p in ps if p.get("Sparql")), None)
        ncons = next((len(p.get("Constraints") or []) for p in ps if p.get("InferentialChain")), 0)
        orig[q["QuestionId"]] = {"orig_topic_mids": tm, "orig_topic_names": tn, "orig_answer_mids": am, "orig_inferential_chain": "|".join(chain) if chain else "", "orig_sparql": sparql or "",
                                 "orig_n_parses": len(ps), "orig_n_constraints": ncons}
    del oj
    rss_note()

    # ---------------- 4. per-row fields ----------------
    qid_re = re.compile(r"^(WebQ(Trn|Test)-\d+)_[0-9a-f]+$")
    lit_forms = collections.Counter()
    for x in rows:
        tp = [pmap[m] for m in x["topic_mids"]]
        tk = [kmap[m] for m in x["topic_mids"]]
        av, at, am_, ap_, ak = [], [], [], [], []
        for a in x["answers"]:
            k = a.get("kb_id")
            t = a.get("text")
            at.append(t if isinstance(t, str) else "")
            if is_mid(k):
                av.append(k)
                am_.append(True)
                ap_.append(pmap[k])
                ak.append(kmap[k])
            else:
                s = k if isinstance(k, str) else ("<null>" if k is None else repr(k))
                av.append(s)
                am_.append(False)
                ap_.append(-1)
                ak.append(-1)
                if k is None:
                    lit_forms["null"] += 1
                elif not isinstance(k, str):
                    lit_forms["non_string:" + type(k).__name__] += 1
                elif re.fullmatch(r"-?\d{4}(-\d\d(-\d\d)?)?", s):
                    lit_forms["date_like"] += 1
                elif re.fullmatch(r"-?\d+(\.\d+)?", s):
                    lit_forms["numeric"] += 1
                else:
                    lit_forms["other_string"] += 1
        x.update(topic_pos=tp, topic_kinds=tk, answer_values=av, answer_text=at, answer_is_mid=am_, answer_pos=ap_, answer_kinds=ak)
        x["n_topics"] = len(tp)
        x["n_answers"] = len(av)
        x["n_answer_literals"] = sum(1 for z in am_ if not z)
        x["n_unresolved_topic"] = sum(1 for p in tp if p < 0)
        x["n_unresolved_answer"] = sum(1 for z, p in zip(am_, ap_) if z and p < 0)
        x["n_unresolved"] = x["n_unresolved_topic"] + x["n_unresolved_answer"]
        x["fully_mappable"] = bool(x["n_topics"] >= 1 and x["n_unresolved_topic"] == 0 and x["n_answers"] >= 1 and x["n_answer_literals"] == 0 and x["n_unresolved_answer"] == 0)
        nq = norm1(x["question"])
        x["dup_group_id"] = gid(nq)
        x["dup_group_loose_id"] = gid(norm2(x["question"]))
    # eval flag
    for x in rows:
        if x["source"] == "nsm_webqsp":
            x["eval_flag"] = "DEV_L3W" if x["qid"] in l3w else ("NSM_DEV" if x["split"] == "dev" else "TRAIN_OTHER")
        else:
            x["eval_flag"] = "NSM_DEV" if x["split"] == "dev" else "TRAIN_OTHER"
        x["eval_eligible"] = x["eval_flag"] in ("DEV_L3W", "NSM_DEV")
    # duplicate groups
    cnt1 = collections.Counter(x["dup_group_id"] for x in rows)
    cnt2 = collections.Counter(x["dup_group_loose_id"] for x in rows)
    src_by_g = collections.defaultdict(set)
    for x in rows:
        src_by_g[x["dup_group_id"]].add(x["source"])
    for x in rows:
        x["dup_group_size"] = cnt1[x["dup_group_id"]]
        x["dup_group_loose_size"] = cnt2[x["dup_group_loose_id"]]
        x["dup_cross_source"] = len(src_by_g[x["dup_group_id"]]) > 1
    # CWQ -> WebQSP parent
    wq_rows = {(x["qid"]): x for x in rows if x["source"] == "nsm_webqsp"}
    wq_dups = collections.Counter(x["qid"] for x in rows if x["source"] == "nsm_webqsp")
    children = collections.Counter()
    n_id_nomatch = 0
    for x in rows:
        if x["source"] == "nsm_cwq":
            m = qid_re.match(x["qid"])
            if not m:
                n_id_nomatch += 1
                x["cwq_parent_id"], x["cwq_parent_kind"], x["parent_eval_flag"] = "", "NO_MATCH", ""
                continue
            x["cwq_parent_id"] = m.group(1)
            x["cwq_parent_kind"] = "WebQTrn" if m.group(2) == "Trn" else "WebQTest"
            if x["cwq_parent_kind"] == "WebQTest":
                x["parent_eval_flag"] = "PARENT_WEBQSP_TEST_SEALED"
            elif x["cwq_parent_id"] in wq_rows:
                x["parent_eval_flag"] = wq_rows[x["cwq_parent_id"]]["eval_flag"]
            else:
                x["parent_eval_flag"] = "PARENT_NOT_IN_NSM_WEBQSP"
            children[x["cwq_parent_id"]] += 1
        else:
            x["cwq_parent_id"], x["cwq_parent_kind"], x["parent_eval_flag"] = "", "", ""
    for x in rows:
        x["n_cwq_children"] = children.get(x["qid"], 0) if x["source"] == "nsm_webqsp" else 0
    # enrichment fields
    for x in rows:
        r = rog[x["source"]].get(x["qid"])
        x["rog_present"] = r is not None
        x["rog_split"] = r["rog_split"] if r else ""
        x["rog_q_entity"] = r["rog_q_entity"] if r else []
        x["rog_a_entity"] = r["rog_a_entity"] if r else []
        o = orig.get(x["qid"]) if x["source"] == "nsm_webqsp" else None
        x["orig_present"] = o is not None
        for k, dv in (("orig_topic_mids", []), ("orig_topic_names", []), ("orig_answer_mids", []), ("orig_inferential_chain", ""), ("orig_sparql", ""), ("orig_n_parses", 0), ("orig_n_constraints", 0)):
            x[k] = o[k] if o else dv
        if o:
            x["answers_match_orig"] = bool(set(v for v, z in zip(x["answer_values"], x["answer_is_mid"]) if z) == set(o["orig_answer_mids"]))
        else:
            x["answers_match_orig"] = False
    rss_note()

    # ---------------- 5. write parquet (write-once) ----------------
    L = pa.list_
    schema = pa.schema([
        ("source", pa.string()), ("split", pa.string()), ("qid", pa.string()), ("question", pa.string()),
        ("topic_mids", L(pa.string())), ("topic_pos", L(pa.int64())), ("topic_kinds", L(pa.int8())),
        ("answer_values", L(pa.string())), ("answer_text", L(pa.string())), ("answer_is_mid", L(pa.bool_())), ("answer_pos", L(pa.int64())), ("answer_kinds", L(pa.int8())),
        ("n_topics", pa.int32()), ("n_answers", pa.int32()), ("n_answer_literals", pa.int32()), ("n_unresolved_topic", pa.int32()), ("n_unresolved_answer", pa.int32()), ("n_unresolved", pa.int32()),
        ("fully_mappable", pa.bool_()), ("eval_flag", pa.string()), ("eval_eligible", pa.bool_()),
        ("dup_group_id", pa.string()), ("dup_group_size", pa.int32()), ("dup_cross_source", pa.bool_()), ("dup_group_loose_id", pa.string()), ("dup_group_loose_size", pa.int32()),
        ("cwq_parent_id", pa.string()), ("cwq_parent_kind", pa.string()), ("parent_eval_flag", pa.string()), ("n_cwq_children", pa.int32()),
        ("rog_present", pa.bool_()), ("rog_split", pa.string()), ("rog_q_entity", L(pa.string())), ("rog_a_entity", L(pa.string())),
        ("orig_present", pa.bool_()), ("orig_topic_mids", L(pa.string())), ("orig_topic_names", L(pa.string())), ("orig_answer_mids", L(pa.string())),
        ("orig_inferential_chain", pa.string()), ("orig_sparql", pa.string()), ("orig_n_parses", pa.int32()), ("orig_n_constraints", pa.int32()), ("answers_match_orig", pa.bool_()),
    ])
    table = pa.Table.from_pydict({f.name: [x[f.name] for x in rows] for f in schema}, schema=schema)
    with open(OUT_PQ, "xb") as fh:
        pq.write_table(table, fh, compression="zstd")
    pq_sha = sha_file(OUT_PQ)
    pq_bytes = os.path.getsize(OUT_PQ)
    rss_note()

    # ---------------- 6. statistics ----------------
    def frac(a, b):
        return None if b == 0 else round(a / b, 6)

    def stratum(n):
        return "zero" if n == 0 else "single(1)" if n == 1 else "small(2-5)" if n <= 5 else "large(>5)"

    def size_bin(n):
        for lim, lab in ((0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5"), (10, "6-10"), (20, "11-20"), (50, "21-50"), (100, "51-100"), (500, "101-500")):
            if n <= lim:
                return lab
        return ">500"

    BIN_ORDER = ["0", "1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "51-100", "101-500", ">500"]
    cells = collections.defaultdict(list)
    for x in rows:
        cells[(x["source"], x["split"])].append(x)

    counts = {"rows_total": len(rows), "per_source_split": {}, "per_source_split_eval_flag": {}, "per_eval_flag": dict(collections.Counter(x["eval_flag"] for x in rows)),
              "eval_eligible_rows": sum(x["eval_eligible"] for x in rows), "pool_only_rows": sum(not x["eval_eligible"] for x in rows)}
    resolution = {}
    for (s, sp), rs in sorted(cells.items()):
        k = s + "/" + sp
        counts["per_source_split"][k] = len(rs)
        counts["per_source_split_eval_flag"][k] = dict(collections.Counter(x["eval_flag"] for x in rs))
        tn = sum(x["n_topics"] for x in rs)
        tu = sum(x["n_unresolved_topic"] for x in rs)
        an_mid = sum(sum(x["answer_is_mid"]) for x in rs)
        an_unres = sum(x["n_unresolved_answer"] for x in rs)
        resolution[k] = {"questions": len(rs), "topic_mid_occurrences": tn, "topic_resolved": tn - tu, "topic_resolved_rate": frac(tn - tu, tn),
                         "answer_mid_occurrences": an_mid, "answer_mid_resolved": an_mid - an_unres, "answer_mid_resolved_rate": frac(an_mid - an_unres, an_mid),
                         "answer_literals": sum(x["n_answer_literals"] for x in rs), "answers_total": sum(x["n_answers"] for x in rs),
                         "questions_with_zero_topics": sum(1 for x in rs if x["n_topics"] == 0), "questions_with_zero_answers": sum(1 for x in rs if x["n_answers"] == 0),
                         "questions_all_topics_resolved": sum(1 for x in rs if x["n_topics"] >= 1 and x["n_unresolved_topic"] == 0),
                         "questions_any_literal_answer": sum(1 for x in rs if x["n_answer_literals"] > 0),
                         "questions_fully_mappable": sum(x["fully_mappable"] for x in rs), "fully_mappable_rate": frac(sum(x["fully_mappable"] for x in rs), len(rs))}
    # per-source overall resolution + unique-mid rates
    for s in SRC:
        rs = [x for x in rows if x["source"] == s]
        tn = sum(x["n_topics"] for x in rs)
        tu = sum(x["n_unresolved_topic"] for x in rs)
        an_mid = sum(sum(x["answer_is_mid"]) for x in rs)
        an_unres = sum(x["n_unresolved_answer"] for x in rs)
        ut = {m for x in rs for m in x["topic_mids"]}
        ua = {v for x in rs for v, z in zip(x["answer_values"], x["answer_is_mid"]) if z}
        resolution[s + "/ALL"] = {"questions": len(rs), "topic_resolved_rate": frac(tn - tu, tn), "answer_mid_resolved_rate": frac(an_mid - an_unres, an_mid),
                                  "unique_topic_mids": len(ut), "unique_topic_mids_resolved": sum(pmap[m] >= 0 for m in ut),
                                  "unique_answer_mids": len(ua), "unique_answer_mids_resolved": sum(pmap[m] >= 0 for m in ua),
                                  "questions_fully_mappable": sum(x["fully_mappable"] for x in rs), "fully_mappable_rate": frac(sum(x["fully_mappable"] for x in rs), len(rs))}
    allq = len(rows)
    resolution["ALL"] = {"questions": allq, "questions_fully_mappable": sum(x["fully_mappable"] for x in rows), "fully_mappable_rate": frac(sum(x["fully_mappable"] for x in rows), allq),
                         "unique_mids_looked_up": len(mids), "unique_mids_resolved": int((pos >= 0).sum())}

    # answer-set size distribution
    size_dist = {}
    for s in list(SRC) + ["ALL"]:
        rs = [x for x in rows if s == "ALL" or x["source"] == s]
        ns = np.array([x["n_answers"] for x in rs], dtype=np.int64)
        c = collections.Counter(size_bin(int(n)) for n in ns)
        size_dist[s] = {"bins": {b: c.get(b, 0) for b in BIN_ORDER}, "mean": round(float(ns.mean()), 4), "median": float(np.median(ns)), "p90": float(np.percentile(ns, 90)),
                        "max": int(ns.max()), "strata": dict(collections.Counter(stratum(int(n)) for n in ns))}

    # answer kinds
    kinds_dist = {}
    cvt = {}
    for s in list(SRC) + ["ALL"]:
        rs = [x for x in rows if s == "ALL" or x["source"] == s]
        c = collections.Counter()
        for x in rs:
            for z, p, k in zip(x["answer_is_mid"], x["answer_pos"], x["answer_kinds"]):
                c["LITERAL_VALUE" if not z else ("UNRESOLVED_MID" if p < 0 else KINDS[k])] += 1
        kinds_dist[s] = dict(c)
        # topic kinds
        ct = collections.Counter(("UNRESOLVED" if p < 0 else KINDS[k]) for x in rs for p, k in zip(x["topic_pos"], x["topic_kinds"]))
        kinds_dist[s + "__topics"] = dict(ct)
        ent_none = []
        n_all_nonentity = n_all_literal = n_all_unres = n_all_resolved_nonentity = n_mixed_nonentity = n_any_cvt = 0
        for x in rs:
            if x["n_answers"] == 0:
                continue
            kinds_row = ["L" if not z else ("U" if p < 0 else ("E" if k == 0 else "N")) for z, p, k in zip(x["answer_is_mid"], x["answer_pos"], x["answer_kinds"])]
            has_e = "E" in kinds_row
            if "N" in kinds_row:
                n_any_cvt += 1
            if not has_e:
                n_all_nonentity += 1
                if all(c_ == "L" for c_ in kinds_row):
                    n_all_literal += 1
                elif all(c_ == "U" for c_ in kinds_row):
                    n_all_unres += 1
                elif all(c_ == "N" for c_ in kinds_row):
                    n_all_resolved_nonentity += 1
                else:
                    n_mixed_nonentity += 1
        cvt[s] = {"questions_with_no_ENTITY_MID_resolved_answer(all answers literal/unresolved/non-entity kind)": n_all_nonentity, "of_which_all_literal": n_all_literal,
                  "of_which_all_unresolved_mid": n_all_unres, "of_which_all_resolved_non_entity_kind(CVT etc)": n_all_resolved_nonentity, "of_which_mixed_literal_unresolved_nonentity": n_mixed_nonentity,
                  "questions_with_at_least_one_resolved_non_entity_kind_answer": n_any_cvt}

    # unresolved MID census
    unres_ans = collections.Counter()
    unres_top = collections.Counter()
    for x in rows:
        for v, z, p in zip(x["answer_values"], x["answer_is_mid"], x["answer_pos"]):
            if z and p < 0:
                unres_ans[(x["source"], v)] += 1
        for v, p in zip(x["topic_mids"], x["topic_pos"]):
            if p < 0:
                unres_top[(x["source"], v)] += 1
    ent_sets = {}
    for s in SRC:
        need = {m for (ss, m) in list(unres_ans) + list(unres_top) if ss == s}
        found = set()
        with open(os.path.join(SRC[s], "entities.txt"), encoding="utf-8") as fh:
            for l in fh:
                l = l.rstrip("\n")
                if l in need:
                    found.add(l)
        ent_sets[s] = found
    census = {}
    for role, cc in (("answer", unres_ans), ("topic", unres_top)):
        for s in SRC:
            keys = [m for (ss, m) in cc if ss == s]
            occ = sum(v for (ss, m), v in cc.items() if ss == s)
            census[role + "/" + s] = {"unique_unresolved_mids": len(keys), "occurrences": occ,
                                      "unique_prefix_m.": sum(1 for m in keys if m.startswith("m.")), "unique_prefix_g.": sum(1 for m in keys if m.startswith("g.")),
                                      "unique_in_entities_txt_but_absent_from_node_uid": sum(1 for m in keys if m in ent_sets[s]),
                                      "unique_not_in_entities_txt": sum(1 for m in keys if m not in ent_sets[s]),
                                      "occurrences_prefix_m.": sum(v for (ss, m), v in cc.items() if ss == s and m.startswith("m.")),
                                      "occurrences_prefix_g.": sum(v for (ss, m), v in cc.items() if ss == s and m.startswith("g."))}
    census["answer_literal_forms"] = dict(lit_forms)
    ex = sorted(unres_ans.items(), key=lambda kv: -kv[1])[:10]
    census["top10_unresolved_answer_mids_by_occurrence"] = [{"source": s, "mid": m, "occurrences": v} for (s, m), v in ex]

    # duplicate-group stats
    def dup_stats(key_id, key_size):
        groups = collections.defaultdict(list)
        for x in rows:
            groups[x[key_id]].append(x)
        multi = {g: r for g, r in groups.items() if len(r) > 1}
        cross_src = {g: r for g, r in multi.items() if len({x["source"] for x in r}) > 1}
        within_wq = {g: r for g, r in multi.items() if len({x["source"] for x in r}) == 1 and r[0]["source"] == "nsm_webqsp"}
        within_cwq = {g: r for g, r in multi.items() if len({x["source"] for x in r}) == 1 and r[0]["source"] == "nsm_cwq"}
        cross_split = {g: r for g, r in multi.items() if len({(x["source"], x["split"]) for x in r}) > 1 and any(x["split"] == "dev" for x in r) and any(x["split"] == "train" for x in r)}
        ee_with_pool = sum(1 for x in rows if x["eval_eligible"] and any((not y["eval_eligible"]) for y in groups[x[key_id]]))
        return {"unique_normalised_questions": len(groups), "groups_with_size_gt1": len(multi), "rows_in_groups_gt1": sum(len(r) for r in multi.values()),
                "max_group_size": max(len(r) for r in groups.values()), "cross_source_groups(webqsp+cwq)": len(cross_src), "cross_source_rows": sum(len(r) for r in cross_src.values()),
                "within_webqsp_groups": len(within_wq), "within_cwq_groups": len(within_cwq), "groups_spanning_train_and_dev": len(cross_split),
                "eval_eligible_rows_sharing_a_group_with_a_pool_only_row": ee_with_pool,
                "cross_source_examples": [{"norm": norm1(r[0]["question"]), "members": [(x["source"], x["split"], x["qid"]) for x in r][:4]} for r in list(cross_src.values())[:5]]}
    dups = {"exact_norm(lowercase/strip/collapse-whitespace)": dup_stats("dup_group_id", "dup_group_size"),
            "loose_norm(+strip all non-alphanumerics, so a trailing '?' no longer separates NSM-WebQSP from CWQ)": dup_stats("dup_group_loose_id", "dup_group_loose_size"),
            "qid_duplicates_within_source": {s: int(sum(1 for q, c in collections.Counter(x["qid"] for x in rows if x["source"] == s).items() if c > 1)) for s in SRC},
            "webqsp_qids_in_both_train_and_dev": len({x["qid"] for x in rows if x["source"] == "nsm_webqsp" and x["split"] == "train"} & {x["qid"] for x in rows if x["source"] == "nsm_webqsp" and x["split"] == "dev"})}

    # CWQ -> WebQSP parent verification
    cw = [x for x in rows if x["source"] == "nsm_cwq"]
    pk = collections.Counter(x["cwq_parent_kind"] for x in cw)
    pk_split = collections.Counter((x["split"], x["cwq_parent_kind"]) for x in cw)
    pef = collections.Counter((x["split"], x["parent_eval_flag"]) for x in cw)
    ver_n = ver_hit = ver_hit_orig = ver_orig_n = 0
    for x in cw:
        pr = wq_rows.get(x["cwq_parent_id"])
        if pr is not None:
            ver_n += 1
            if set(x["topic_mids"]) & set(pr["topic_mids"]):
                ver_hit += 1
            if pr["orig_present"]:
                ver_orig_n += 1
                if set(x["topic_mids"]) & set(pr["orig_topic_mids"]):
                    ver_hit_orig += 1
    name_n = name_hit = 0
    for x in cw:
        pr = wq_rows.get(x["cwq_parent_id"])
        if pr is not None and pr["orig_topic_names"]:
            name_n += 1
            ql = x["question"].lower()
            if any(nm.lower() in ql for nm in pr["orig_topic_names"]):
                name_hit += 1
    parents = collections.Counter(x["cwq_parent_id"] for x in cw if x["cwq_parent_id"])
    cwq_parent = {"id_scheme": "CWQ qid = '<WebQSP parent id>_<32 hex>' (regex ^(WebQ(Trn|Test)-\\d+)_[0-9a-f]+$)", "ids_not_matching_scheme": n_id_nomatch,
                  "parent_kind_counts": dict(pk), "parent_kind_by_split": {"%s/%s" % k: v for k, v in pk_split.items()},
                  "parent_eval_flag_by_cwq_split": {"%s/%s" % k: v for k, v in pef.items()},
                  "distinct_parents": len(parents), "children_per_parent_max": max(parents.values()) if parents else 0, "children_per_parent_mean": round(sum(parents.values()) / max(len(parents), 1), 3),
                  "parents_present_in_nsm_webqsp_train_or_dev_rows": ver_n,
                  "verification_topic_mid_overlap_with_parent(NSM_topics)": {"checked": ver_n, "overlap": ver_hit, "rate": frac(ver_hit, ver_n)},
                  "verification_topic_mid_overlap_with_parent(original_WebQSP_TopicEntityMid)": {"checked": ver_orig_n, "overlap": ver_hit_orig, "rate": frac(ver_hit_orig, ver_orig_n)},
                  "verification_parent_topic_name_in_cwq_question(original_WebQSP_TopicEntityName, case-insensitive substring)": {"checked": name_n, "hit": name_hit, "rate": frac(name_hit, name_n)},
                  "note": "WebQTest parents can NOT be verified (that would read the sealed WebQSP test file); they are only counted from the id prefix."}

    # enrichment join census
    enrich = {"rog_meta": rog_meta, "orig_webqsp_train_json": orig_meta, "rog_join": {}, "orig_join": {}}
    for (s, sp), rs in sorted(cells.items()):
        k = s + "/" + sp
        enrich["rog_join"][k] = {"rows": len(rs), "rog_present": sum(x["rog_present"] for x in rs), "rog_split_counts": dict(collections.Counter(x["rog_split"] for x in rs))}
    for s in SRC:
        nsm_ids = {x["qid"] for x in rows if x["source"] == s}
        enrich["rog_join"][s + "/rog_ids_not_in_nsm_train_dev"] = int(len(set(rog[s]) - nsm_ids))
        enrich["rog_join"][s + "/nsm_ids_not_in_rog_train_validation"] = int(len(nsm_ids - set(rog[s])))
    for sp in ("train", "dev"):
        rs = cells[("nsm_webqsp", sp)]
        enrich["orig_join"][sp] = {"rows": len(rs), "orig_present": sum(x["orig_present"] for x in rs), "nsm_answers_equal_orig_union_of_parses": sum(x["answers_match_orig"] for x in rs),
                                   "nsm_topic_overlaps_orig_topic": sum(1 for x in rs if x["orig_present"] and set(x["topic_mids"]) & set(x["orig_topic_mids"]))}
    enrich["cwq_fields_not_available_locally"] = ("RoG-cwq carries only id/question/answer/q_entity/a_entity/graph/choices (names, no MIDs); compositionality_type, SPARQL, machine_question, "
                                                  "composition_answer and the original answer MIDs live only in the original ComplexWebQuestions release (see FBQ_EXTERNAL_SOURCES__v1.md)")

    # reproduction of the established probe numbers (FBQ_NSM_MAPPING_PROBE__v1.json)
    wd = cells[("nsm_webqsp", "dev")]
    d786 = [x for x in rows if x["source"] == "nsm_webqsp" and x["qid"] in l3w]
    d786_first = {}
    for x in d786:
        d786_first.setdefault(x["qid"], x)
    d786u = list(d786_first.values())
    repro = {"nsm_dev_simple_250": {"questions": len(wd), "topic_entities": sum(x["n_topics"] for x in wd), "topic_resolved": sum(x["n_topics"] - x["n_unresolved_topic"] for x in wd),
                                    "answer_mids": sum(sum(x["answer_is_mid"]) for x in wd), "answer_mids_resolved": sum(sum(x["answer_is_mid"]) - x["n_unresolved_answer"] for x in wd),
                                    "answers_non_mid_values": sum(x["n_answer_literals"] for x in wd), "questions_fully_mappable": sum(x["fully_mappable"] for x in wd),
                                    "probe_expected": {"questions": 250, "topic_entities": 254, "topic_resolved": 254, "answer_mids": 2990, "answer_mids_resolved": 2959, "answers_non_mid_values": 5, "questions_fully_mappable": 236}},
             "l3w_786": {"rows_joined": len(d786u), "rows_in_nsm_train": sum(1 for x in d786 if x["split"] == "train"), "rows_in_nsm_dev": sum(1 for x in d786 if x["split"] == "dev"),
                         "topic_entities": sum(x["n_topics"] for x in d786u), "topic_resolved": sum(x["n_topics"] - x["n_unresolved_topic"] for x in d786u),
                         "answer_mids": sum(sum(x["answer_is_mid"]) for x in d786u), "answer_mids_resolved": sum(sum(x["answer_is_mid"]) - x["n_unresolved_answer"] for x in d786u),
                         "questions_fully_mappable": sum(x["fully_mappable"] for x in d786u),
                         "probe_expected": {"rows_joined": 786, "topic_entities": 806, "topic_resolved": 806, "answer_mids": 8263, "answer_mids_resolved": 8195, "questions_fully_mappable": 773}},
             "nsm_dev_rows_also_in_l3w_786": sum(1 for x in wd if x["qid"] in l3w)}
    if SMOKE is None:
        e1, e2 = repro["nsm_dev_simple_250"], repro["l3w_786"]
        repro["probe_reproduced"] = bool(all(e1[k] == e1["probe_expected"][k] for k in e1["probe_expected"]) and all(e2[k] == e2["probe_expected"][k] for k in e2["probe_expected"]))

    # subsets
    def sub_counts(pred):
        rs = [x for x in rows if pred(x)]
        out = {"questions": len(rs), "by_source_split": {}, "by_eval_flag": dict(collections.Counter(x["eval_flag"] for x in rs)),
               "by_answer_stratum": dict(collections.Counter(stratum(x["n_answers"]) for x in rs)), "by_source_stratum": {}}
        for (s, sp), _ in sorted(cells.items()):
            out["by_source_split"][s + "/" + sp] = sum(1 for x in rs if x["source"] == s and x["split"] == sp)
        for s in SRC:
            out["by_source_stratum"][s] = dict(collections.Counter(stratum(x["n_answers"]) for x in rs if x["source"] == s))
        out["distinct_normalised_questions"] = len({x["dup_group_id"] for x in rs})
        out["total_gold_answer_mids"] = sum(sum(x["answer_is_mid"]) for x in rs)
        return out
    subsets = {
        "stratum_definition": "single = exactly 1 NSM answer; small = 2..5 answers; large = >=6 answers (zero = 0 answers); n_answers counts every NSM answer incl. literals",
        "SET_ALL_MAPPABLE": {"definition": "fully_mappable (all topic MIDs + all answer MIDs resolved to positions, >=1 topic, >=1 answer, no literal answers), any source/split", **sub_counts(lambda x: x["fully_mappable"])},
        "SET_EVAL_ELIGIBLE": {"definition": "fully_mappable AND eval_eligible (eval_flag DEV_L3W or NSM_DEV)", **sub_counts(lambda x: x["fully_mappable"] and x["eval_eligible"])},
        "SET_POOL_ONLY_MAPPABLE": {"definition": "fully_mappable AND NOT eval_eligible (eval_flag TRAIN_OTHER; may contain sealed split-B rows)", **sub_counts(lambda x: x["fully_mappable"] and not x["eval_eligible"])},
        "SET_EVAL_ELIGIBLE_EXCL_CWQ_WEBQSP_TEST_PARENT": {"definition": "informational: SET_EVAL_ELIGIBLE minus CWQ rows whose id-parent is a WebQTest-* question (leak-adjacent to the sealed WebQSP test)",
                                                          **sub_counts(lambda x: x["fully_mappable"] and x["eval_eligible"] and x["cwq_parent_kind"] != "WebQTest")},
    }
    cells_by_group = collections.defaultdict(list)
    for x in rows:
        cells_by_group[x["dup_group_id"]].append(x)
    subsets["SET_EVAL_ELIGIBLE_DEDUP_ACROSS_POOL"] = {
        "definition": "informational: SET_EVAL_ELIGIBLE minus rows whose exact-normalised question also occurs in a pool-only row",
        **sub_counts(lambda x: x["fully_mappable"] and x["eval_eligible"] and not any((not y["eval_eligible"]) for y in cells_by_group[x["dup_group_id"]]))}

    secs = round(time.time() - T0, 1)
    rss_note()
    rec = {
        "RECORD": "FBQ_SET", "version": "v1", "smoke": SMOKE, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Multi-source Freebase question set mapped onto the frozen 302M-node Freebase canonical positions (topic MIDs + answer MIDs -> position/kind). Freebase has no queries of its own (DATASET.json n_queries = 0).",
        "position_rule": "uid = hash(mid.encode('utf-8')) under PYTHONHASHSEED=0 (asserted); position = index in nodes/node_uid.npy with an equality check (merge-join, 64 MB blocks); kind = nodes/kind.npy (0 ENTITY_MID, 1 CVT_MEDIATOR, 2 SCHEMA_TYPE, 3 SCHEMA_PROPERTY, 4 SCHEMA_OTHER, 5 EXTERNAL_URI, 6 LITERAL)",
        "held_out_handling": {"parsed": ["nsm_webqsp train_simple", "nsm_webqsp dev_simple", "nsm_cwq train_simple", "nsm_cwq dev_simple"],
                              "sealed_not_read": sealed,
                              "rog_files_read": "train-* and validation-* parquets only, columns id/question/answer/q_entity/a_entity (graph column not requested); test-* parquets not opened",
                              "original_webqsp": "WebQSP.train.json only; WebQSP.test.json not opened",
                              "eval_flag_definition": {"DEV_L3W": "WebQSP row whose id is in the 786-row KB-L3 development population results/L3_DEV/l3w_population_webqsp__v1.json (split A) -> eval-eligible",
                                                       "NSM_DEV": "row from the NSM dev_simple file (webqsp or cwq) that is not DEV_L3W -> eval-eligible",
                                                       "TRAIN_OTHER": "every other NSM train_simple row -> POOL ONLY, NOT cleared as dev-eligible"},
                              "split_B_caveat": ("The 717 sealed WebQSP split-B rows (parity of sha1(query_id)[:8] over the 1,549-row train_holdout carve, per the l3w population record) are in NSM train_simple/dev_simple and are NOT identifiable "
                                                 "here without reading the sealed carve: they are inside TRAIN_OTHER (or NSM_DEV). TRAIN_OTHER is therefore not cleared for evaluation; only DEV_L3W and NSM_DEV rows are eval-eligible, and ALL remaining train rows are pool-only. "
                                                 "No attempt was made to reconstruct split B. The relationship of the 250 NSM dev_simple rows to split B is unchecked (the user ruled them eval-eligible).")},
        "inputs": {"nsm_files": files, "rog_parquets_read": {}, "original_webqsp_train_json": {"file": os.path.relpath(ORIG_WQSP, ROOT).replace("\\", "/"), "bytes": os.path.getsize(ORIG_WQSP), "sha256": sha_file(ORIG_WQSP) if SMOKE is None else None},
                   "l3w_population_json": {"file": os.path.relpath(POP, ROOT).replace("\\", "/"), "sha256": sha_file(POP)},
                   "node_uid.npy": {"bytes": os.path.getsize(os.path.join(FB, "nodes", "node_uid.npy")), "sha256_computed": rinfo["node_uid_sha256_computed"]},
                   "kind.npy": {"bytes": os.path.getsize(os.path.join(FB, "nodes", "kind.npy")), "sha256_computed": sha_file(os.path.join(FB, "nodes", "kind.npy")) if SMOKE is None else None},
                   "dataset_json_pins": {}},
        "counts": counts, "resolution": resolution, "resolve_join_info": rinfo, "answer_set_size_distribution": size_dist, "answer_kinds_distribution": kinds_dist,
        "questions_with_answers_all_nonentity": cvt, "unresolved_mid_census": census, "duplicate_groups": dups, "cwq_to_webqsp_parent": cwq_parent, "enrichment": enrich,
        "probe_reproduction": repro, "subsets": subsets,
        "output": {"parquet": os.path.relpath(OUT_PQ, ROOT).replace("\\", "/") if not SMOKE else OUT_PQ, "parquet_bytes": pq_bytes, "parquet_sha256": pq_sha, "rows": len(rows),
                   "columns": [f.name for f in schema], "compression": "zstd"},
        "code": {"scratchpad/_fbq_build_set.py": hashlib.sha256(open(CODE_PATH, "rb").read()).hexdigest()},
        "seconds": secs, "peak_rss_mb": round(peak["rss"] / 2 ** 20, 1),
    }
    # pins
    ds = json.load(open(os.path.join(FB, "DATASET.json"), encoding="utf-8"))
    pins = ds["nodes"]["index"]
    rec["inputs"]["dataset_json_pins"] = {"node_uid.sha256": pins["node_uid"]["sha256"], "kind.sha256": pins["kind"]["sha256"]}
    if SMOKE is None:
        rec["inputs"]["node_uid.npy"]["matches_pin"] = rinfo["node_uid_sha256_computed"] == pins["node_uid"]["sha256"]
        rec["inputs"]["kind.npy"]["matches_pin"] = rec["inputs"]["kind.npy"]["sha256_computed"] == pins["kind"]["sha256"]
    # RoG parquet shas (files actually read)
    for s in SRC:
        for sp in ("train", "validation"):
            for fpath in sorted(glob.glob(os.path.join(ROG[s], sp + "-*.parquet"))):
                rec["inputs"]["rog_parquets_read"][os.path.relpath(fpath, ROOT).replace("\\", "/")] = {"bytes": os.path.getsize(fpath), "sha256": sha_file(fpath) if SMOKE is None else None}
    rec["seconds"] = round(time.time() - T0, 1)
    rss_note()
    rec["peak_rss_mb"] = round(peak["rss"] / 2 ** 20, 1)
    with open(OUT_JSON, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1, default=str)
        fh.write("\n")
    write_md(rec)
    print("done", rec["seconds"], "s; peak RSS MB", rec["peak_rss_mb"], "rows", len(rows), flush=True)


def write_md(rec):
    c, r, sset = rec["counts"], rec["resolution"], rec["subsets"]
    L = []
    L.append("# FBQ_SET v1 -- multi-source Freebase question set (mapped onto the frozen 302M-node Freebase canonical)\n")
    L.append("Generated by `scratchpad/_fbq_build_set.py` (PYTHONHASHSEED=0); record `FBQ_SET__v1.json`, table `FBQ_SET__v1.parquet` (%d rows, %s bytes). Freebase itself has no queries (n_queries = 0); every question here is keyed to Freebase by MID through the NSM release.\n" % (rec["output"]["rows"], rec["output"]["parquet_bytes"]))
    L.append("Held-out rule: only NSM `train_simple` and `dev_simple` were parsed (WebQSP and CWQ). The NSM `test_simple` files were NOT parsed (bytes/newlines/sha256 only). RoG test parquets and `WebQSP.test.json` were not opened.\n")
    L.append("## Rows\n")
    L.append("| source/split | rows | fully mappable | rate | topic MID resolved | answer MID resolved |\n|---|---|---|---|---|---|")
    for k, v in r.items():
        if k.endswith("/ALL") or k == "ALL":
            continue
        L.append("| %s | %d | %d | %s | %s | %s |" % (k, v["questions"], v["questions_fully_mappable"], v["fully_mappable_rate"], v["topic_resolved_rate"], v["answer_mid_resolved_rate"]))
    L.append("| ALL | %d | %d | %s | | |" % (r["ALL"]["questions"], r["ALL"]["questions_fully_mappable"], r["ALL"]["fully_mappable_rate"]))
    L.append("\n## eval_flag\n")
    L.append("`DEV_L3W` (786-row WebQSP KB-L3 dev population) and `NSM_DEV` (NSM dev_simple rows) are eval-eligible; `TRAIN_OTHER` is pool-only and may contain sealed split-B rows (indistinguishable without reading the sealed carve).\n")
    L.append("| source/split | " + " | ".join(["DEV_L3W", "NSM_DEV", "TRAIN_OTHER"]) + " |\n|---|---|---|---|")
    for k, v in c["per_source_split_eval_flag"].items():
        L.append("| %s | %d | %d | %d |" % (k, v.get("DEV_L3W", 0), v.get("NSM_DEV", 0), v.get("TRAIN_OTHER", 0)))
    L.append("\n## Subsets\n")
    L.append("| subset | questions | definition |\n|---|---|---|")
    for k, v in sset.items():
        if isinstance(v, dict):
            L.append("| %s | %d | %s |" % (k, v["questions"], v["definition"]))
    L.append("\nStratum split of the main sets (single = 1 answer, small = 2..5, large = >=6):\n")
    L.append("| subset | single | small | large |\n|---|---|---|---|")
    for k in ("SET_ALL_MAPPABLE", "SET_EVAL_ELIGIBLE", "SET_POOL_ONLY_MAPPABLE"):
        v = sset[k]["by_answer_stratum"]
        L.append("| %s | %d | %d | %d |" % (k, v.get("single(1)", 0), v.get("small(2-5)", 0), v.get("large(>5)", 0)))
    cu = rec["unresolved_mid_census"]
    L.append("\n## Unresolved answer MID census\n")
    for k in ("answer/nsm_webqsp", "answer/nsm_cwq", "topic/nsm_webqsp", "topic/nsm_cwq"):
        v = cu[k]
        L.append("- %s: %d unique unresolved MIDs (%d occurrences); m.* %d / g.* %d; in the NSM entities.txt but absent from node_uid: %d; not in entities.txt: %d" % (k, v["unique_unresolved_mids"], v["occurrences"], v["unique_prefix_m."], v["unique_prefix_g."], v["unique_in_entities_txt_but_absent_from_node_uid"], v["unique_not_in_entities_txt"]))
    L.append("- literal answer forms (not MIDs, never positions): %s" % json.dumps(cu["answer_literal_forms"]))
    cp = rec["cwq_to_webqsp_parent"]
    L.append("\n## CWQ -> WebQSP parent\n")
    L.append("- id scheme `<WebQSP parent id>_<hash>`; parent kinds %s; topic-MID overlap with the parent (NSM): %s" % (json.dumps(cp["parent_kind_counts"]), json.dumps(cp["verification_topic_mid_overlap_with_parent(NSM_topics)"])))
    L.append("- parent eval_flag by CWQ split: %s" % json.dumps(cp["parent_eval_flag_by_cwq_split"]))
    dd = rec["duplicate_groups"]
    k1 = [k for k in dd if k.startswith("exact")][0]
    L.append("\n## Duplicates\n")
    L.append("- exact normalised question: %d unique, %d groups >1 (%d rows), cross-source groups %d" % (dd[k1]["unique_normalised_questions"], dd[k1]["groups_with_size_gt1"], dd[k1]["rows_in_groups_gt1"], dd[k1]["cross_source_groups(webqsp+cwq)"]))
    k2 = [k for k in dd if k.startswith("loose")][0]
    L.append("- loose (punctuation-insensitive): %d unique, %d groups >1, cross-source groups %d" % (dd[k2]["unique_normalised_questions"], dd[k2]["groups_with_size_gt1"], dd[k2]["cross_source_groups(webqsp+cwq)"]))
    L.append("\n## Not available locally\n")
    L.append("- " + rec["enrichment"]["cwq_fields_not_available_locally"])
    L.append("\nAnswer sets are the NSM answers (not the RoG gold-node sets): the retrieval target at Freebase scale is defined by these MIDs. Position `-1` = unresolved (MID absent from the cleaned freebase_v3 nodes) or literal answer. No question here has been used to tune anything.")
    with open(OUT_MD, "x", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
