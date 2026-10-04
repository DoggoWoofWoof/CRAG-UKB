"""FBQ_SET v2: v1 (NSM WebQSP + CWQ) + FreebaseQA train/dev + GrailQA train/dev, mapped onto the frozen 302M-node Freebase canonical,
with harmonised TYPE / HOPS, declared deterministic DEDUPE rules, family cap, leak-clean flags and a hop x type BALANCED set.

  PYTHONHASHSEED=0 python scratchpad/_fbq_build_set_v2.py --dry-run               # analysis only, prints the C table, writes NOTHING under results/
  PYTHONHASHSEED=0 python scratchpad/_fbq_build_set_v2.py --smoke 300              # tiny run, writes to a temp dir
  PYTHONHASHSEED=0 python scratchpad/_fbq_build_set_v2.py --c-recommended N        # final write-once outputs (N in 200/400/800/1600)

Outputs (write-once; supersede-never-edit; refuses to overwrite) in results/FREEBASE_SCALE/:
  FBQ_SET__v2__POOL.parquet           all rows that survive dedupe rules 1-2 (with every derived flag)
  FBQ_SET__v2__BALANCED.parquet       the recommended balanced set (all columns)
  FBQ_SET__v2__DEDUPE_AUDIT.parquet   one row per row dropped by a dedupe rule (rule, representative, group size)
  FBQ_SET__v2.json / FBQ_SET__v2.md   record (counts at every stage, tables, shas, seconds, peak RSS)

HELD-OUT RULES: NSM test_simple (WebQSP, CWQ), WebQSP.test.json, GrailQA test, FreebaseQA eval/partial, CWQ original test are never opened. Only
train/dev files are parsed. Nothing is tuned on retrieval results (none exist); every rule below is declared a priori and is deterministic.

HOPS (relation edges on the topic -> answer chain; a CVT hop counts as its two Freebase edges, same convention in every source):
  freebaseqa : len(InferentialChain.split('..')) per parse ('..' joins the two relations of a CVT); parses are ALTERNATIVE annotations -> hops = MIN over parses
  grailqa    : shortest undirected path (edges) from each ENTITY node of graph_query to the question node; entities are CONJUNCTIVE -> hops = MAX over entities
               (num_edge, the size of the whole query graph, is recorded separately); rows with no entity node have no topic and are not fully mappable
  cwq        : parsed from the original release SPARQL: triple patterns '<term> ns:rel <term>' (no literal object, no FILTER-EXISTS ?sk helper), undirected shortest path
               from each ns:m./ns:g. entity to ?x; MAX over entities (conjunctive). Fallback (no original match): unknown
  webqsp     : length of the InferentialChain of the primary parse (shortest chain, ties -> lowest ParseId) of WebQSP.train.json (train only);
               cross-checked against the shortest topic->answer path inside the NSM subgraph tuples (PPR-pruned graph, can overestimate); NSM path is the fallback
TYPE (first matching rule):
  freebaseqa : hops==1 -> simple else chain (constraints are not annotated)
  grailqa    : function count -> count; argmax/argmin -> superlative; > >= < <= -> comparative; (function none) literal node -> other; >=2 entity nodes -> conjunction;
               else hops==1 -> simple else chain
  cwq        : compositionality_type: composition->chain, conjunction, comparative, superlative; no original match -> 'cwq_type_unknown'
  webqsp     : primary parse: Order -> superlative; Equal/Entity constraint -> conjunction; any other constraint or Time -> other; else hops==1 simple else chain
               (type_hops_only keeps the by-hops-only labelling for every source)"""
import argparse
import collections
import glob  # noqa: F401
import gzip  # noqa: F401  (fallback only; pyarrow is available)
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata

assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FB = os.path.join(ROOT, "data", "final_canonical", "freebase")
ORIGD = os.path.join(ROOT, "data", "original")
RES = os.path.join(ROOT, "results", "FREEBASE_SCALE")
V1 = os.path.join(RES, "FBQ_SET__v1.parquet")
PROV = os.path.join(RES, "FBQ_DOWNLOAD_PROVENANCE__v1.json")
NSM_ROOT = os.path.join(ROOT, "data", "final_canonical", "webqsp", "_acquisition", "nsm", "extracted")
NSM_WQ = os.path.join(NSM_ROOT, "webqsp", "webqsp")
ORIG_WQSP = os.path.join(ORIGD, "webqsp", "WebQSP", "data", "WebQSP.train.json")
FQA = {"train": os.path.join(ORIGD, "freebaseqa", "FreebaseQA-train.json"), "dev": os.path.join(ORIGD, "freebaseqa", "FreebaseQA-dev.json")}
GRAIL = {"train": os.path.join(ORIGD, "grailqa", "grailqa_v1.0_train.json"), "dev": os.path.join(ORIGD, "grailqa", "grailqa_v1.0_dev.json")}
CWQO = {"train": os.path.join(ORIGD, "cwq_original", "ComplexWebQuestions_train.json"), "dev": os.path.join(ORIGD, "cwq_original", "ComplexWebQuestions_dev.json")}
KINDS = ["ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER", "EXTERNAL_URI", "LITERAL"]
SEED = "20261004"
CS = [200, 400, 800, 1600]
MIN_CELL = 50  # a-priori: comparative/superlative/count cells with fewer candidates than this are merged (per hop bucket)
FAMILY_CAP = 2
SRC_CAP = 0.5

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", type=int, default=0, help="rows per input file (0 = full); writes to the temp dir only")
ap.add_argument("--dry-run", action="store_true", help="full computation, no outputs under results/; writes a scratch summary to the temp dir")
ap.add_argument("--c-recommended", type=int, default=0)
ap.add_argument("--outdir", default=None)
args = ap.parse_args()
SMOKE = args.smoke if args.smoke > 0 else None
DRY = args.dry_run or SMOKE is not None
if not DRY:
    assert args.c_recommended in CS, "--c-recommended must be one of %s" % CS
OUTDIR = args.outdir or (os.path.join(os.environ.get("TEMP", "."), "fbq_v2_%s_%d" % ("smoke" if SMOKE else "dry", int(time.time()))) if DRY else RES)
os.makedirs(OUTDIR, exist_ok=True)
OUT = {k: os.path.join(OUTDIR, "FBQ_SET__v2" + s) for k, s in (("pool", "__POOL.parquet"), ("bal", "__BALANCED.parquet"), ("audit", "__DEDUPE_AUDIT.parquet"), ("json", ".json"), ("md", ".md"))}
for p in OUT.values():
    assert not os.path.exists(p), "write-once: refusing to overwrite " + p

peak = {"rss": 0}


def rss_note():
    mi = psutil.Process().memory_info()
    peak["rss"] = max(peak["rss"], getattr(mi, "peak_wset", mi.rss), mi.rss)


def say(*a):
    print("[%7.1fs rss %4d MB]" % (time.time() - T0, psutil.Process().memory_info().rss // 2 ** 20), *a, flush=True)


def sha_file(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def iter_json_array(path, block=1 << 20, limit=None):
    """stream the objects of a top-level JSON array (bounded memory)."""
    dec = json.JSONDecoder()
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(block)
        i = 0
        while buf[i:i + 1].isspace():
            i += 1
        assert buf[i] == "[", buf[:50]
        i += 1
        eof = False
        while True:
            while True:
                while i < len(buf) and (buf[i].isspace() or buf[i] == ","):
                    i += 1
                if i >= len(buf) and not eof:
                    break
                if i < len(buf) and buf[i] == "]":
                    return
                try:
                    obj, j = dec.raw_decode(buf, i)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                yield obj
                n += 1
                if limit is not None and n >= limit:
                    return
                i = j
            more = f.read(block)
            if not more:
                eof = True
                if i >= len(buf):
                    return
            buf = buf[i:] + more
            i = 0


def is_mid(k):
    return isinstance(k, str) and k[:2] in ("m.", "g.")


def norm_q(q):
    s = unicodedata.normalize("NFKD", q.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[\W_]+", " ", s, flags=re.UNICODE).split())


def rank_hex(qid):
    return hashlib.sha256((SEED + qid).encode("utf-8")).hexdigest()


def hop_bucket(h):
    return "unk" if h is None or h < 1 else ("4+" if h >= 4 else str(h))


def ans_bucket(n):
    return "0" if n == 0 else "1" if n == 1 else "2-5" if n <= 5 else "6+"


def resolve_positions(mids):
    """merge-join the unique MID strings against nodes/node_uid.npy (sorted int64) (copied from _fbq_build_set.py)."""
    q = np.fromiter((hash(m.encode("utf-8")) for m in mids), dtype=np.int64, count=len(mids))
    order = np.argsort(q, kind="stable")
    qs = q[order]
    n_coll = int(len(qs) - len(np.unique(qs)))
    pos_sorted = np.full(len(qs), -1, dtype=np.int64)
    path = os.path.join(FB, "nodes", "node_uid.npy")
    mm = np.load(path, mmap_mode="r")
    off, n = int(mm.offset), int(mm.shape[0])
    assert mm.dtype == np.int64
    del mm
    h = hashlib.sha256()
    strictly = True
    prev_last = None
    block = 8_000_000
    with open(path, "rb") as f:
        h.update(f.read(off))
        f.seek(off)
        for lo in range(0, n, block):
            cnt = min(block, n - lo)
            blk = np.fromfile(f, dtype=np.int64, count=cnt)
            assert len(blk) == cnt
            h.update(memoryview(blk))
            if len(blk) > 1 and not bool((blk[1:] > blk[:-1]).all()):
                strictly = False
            if prev_last is not None and not (blk[0] > prev_last):
                strictly = False
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
    info = {"n_unique_mids": int(len(mids)), "n_resolved": int((pos >= 0).sum()), "n_node_uid": n, "node_uid_strictly_increasing": strictly,
            "uid_collisions_among_query_mids": n_coll, "node_uid_sha256_computed": h.hexdigest()}
    return pos, kind_arr, info


# ---------------------------------------------------------------- source parsers
def parse_freebaseqa(split, limit):
    d = json.load(open(FQA[split], encoding="utf-8"))
    assert d["Dataset"] == "FreebaseQA-" + split, d["Dataset"]
    qs = d["Questions"]
    if limit:
        qs = qs[:limit]
    rows = []
    dis = 0
    for q in qs:
        ps = q["Parses"]
        tm = sorted({p["TopicEntityMid"] for p in ps})
        ans, names, seen = [], [], set()
        per_parse = []
        for p in ps:
            s = set()
            for a in p["Answers"]:
                m = a["AnswersMid"]
                s.add(m)
                if m not in seen:
                    seen.add(m)
                    ans.append(m)
                    names.append((a.get("AnswersName") or [""])[0])
            per_parse.append(frozenset(s))
        disagree = len(set(per_parse)) > 1
        dis += disagree
        hops = [len(p["InferentialChain"].split("..")) for p in ps]
        rows.append({"source": "freebaseqa", "split": split, "qid": q["Question-ID"], "question": q["RawQuestion"], "topic_mids": tm,
                     "answers": [(m, nm, True) for m, nm in zip(ans, names)], "hops_min": min(hops), "hops_max": max(hops), "fqa_n_parses": len(ps),
                     "fqa_parse_answers_disagree": disagree})
    return rows, {"file": os.path.relpath(FQA[split], ROOT).replace("\\", "/"), "questions": len(qs), "questions_with_parse_answer_disagreement": dis}


def grail_slim(q):
    gq = q["graph_query"]
    nodes = gq["nodes"]
    ents = [n for n in nodes if n["node_type"] == "entity"]
    n_lit = sum(1 for n in nodes if n["node_type"] == "literal")
    qn = [n["nid"] for n in nodes if n.get("question_node") == 1]
    adj = collections.defaultdict(list)
    for e in gq["edges"]:
        adj[e["start"]].append(e["end"])
        adj[e["end"]].append(e["start"])
    dists = []
    for en in ents:
        dist = {en["nid"]: 0}
        dq = [en["nid"]]
        for u in dq:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        dists.append(dist.get(qn[0]) if len(qn) == 1 else None)
    return ents, n_lit, dists


def parse_grailqa(split, limit):
    rows = []
    fn_c = collections.Counter()
    nonmid_ent = 0
    for q in iter_json_array(GRAIL[split], limit=limit):
        ents, n_lit, dists = grail_slim(q)
        tm = []
        for n in ents:
            if is_mid(n["id"]):
                tm.append(n["id"])
            else:
                nonmid_ent += 1
        tm = sorted(set(tm))
        ans, seen = [], set()
        for a in q["answer"]:
            key = (a["answer_type"], str(a["answer_argument"]))
            if key in seen:
                continue
            seen.add(key)
            ans.append((str(a["answer_argument"]), a.get("entity_name") or "", a["answer_type"] == "Entity" and is_mid(a["answer_argument"])))
        dd = [d for d in dists if d is not None]
        fn_c[q["function"]] += 1
        rows.append({"source": "grailqa", "split": split, "qid": str(q["qid"]), "question": q["question"], "topic_mids": tm, "answers": ans,
                     "hops_min": min(dd) if dd else None, "hops_max": max(dd) if dd else None, "grail_function": q["function"], "grail_num_edge": q["num_edge"],
                     "grail_num_node": q["num_node"], "grail_n_entity_nodes": len(ents), "grail_n_literal_nodes": n_lit, "grail_level": q.get("level") or "",
                     "grail_domains": "|".join(q.get("domains") or [])})
    return rows, {"file": os.path.relpath(GRAIL[split], ROOT).replace("\\", "/"), "questions": len(rows), "functions": dict(fn_c), "entity_nodes_with_non_mid_id": nonmid_ent}


TERM = r"(\?[A-Za-z0-9_]+|ns:[mg]\.[A-Za-z0-9_]+)"
TRIPLE = re.compile(TERM + r"\s+ns:([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)\s+(" + TERM[1:-1] + r"|\"[^\"]*\"(?:\^\^xsd:\w+|@\w+)?)")
ENT = re.compile(r"^ns:[mg]\.")


def cwq_sparql_hops(sparql):
    """(hops_min, hops_max, n_triples, n_entities) via undirected shortest path entity -> ?x over the SPARQL triple patterns."""
    lines = []
    for ln in sparql.split("\n"):
        if ln.lstrip().startswith("PREFIX"):
            continue
        if "#" in ln:
            ln = ln.split("#")[0]
        lines.append(ln)
    text = "\n".join(lines)
    adj = collections.defaultdict(set)
    ntrip = 0
    for m in TRIPLE.finditer(text):
        s, _rel, o = m.group(1), m.group(2), m.group(3)
        if o.startswith('"'):
            continue
        if s.startswith("?sk") or o.startswith("?sk"):
            continue
        adj[s].add(o)
        adj[o].add(s)
        ntrip += 1
    ents = [t for t in adj if ENT.match(t)]
    if "?x" not in adj or not ents:
        return None, None, ntrip, len(ents)
    ds = []
    for e in ents:
        dist = {e: 0}
        dq = [e]
        for u in dq:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        if "?x" in dist:
            ds.append(dist["?x"])
    if not ds:
        return None, None, ntrip, len(ents)
    return min(ds), max(ds), ntrip, len(ents)


def load_cwq_original(limit):
    out = {}
    meta = {}
    for sp, path in CWQO.items():
        n = 0
        for x in iter_json_array(path, limit=limit):
            hmin, hmax, ntrip, nent = cwq_sparql_hops(x["sparql"])
            am = sorted({a["answer_id"] for a in x["answers"] if is_mid(a.get("answer_id"))})
            out[x["ID"]] = {"orig_split": sp, "comp": x["compositionality_type"], "webqsp_ID": x["webqsp_ID"], "hops_min": hmin, "hops_max": hmax, "ntrip": ntrip,
                            "q": x["question"], "ans": am, "n_ans_nonmid": sum(1 for a in x["answers"] if not is_mid(a.get("answer_id")))}
            n += 1
        meta[sp] = {"file": os.path.relpath(path, ROOT).replace("\\", "/"), "rows": n}
    return out, meta


def load_webqsp_orig():
    oj = json.load(open(ORIG_WQSP, encoding="utf-8"))
    out = {}
    for q in oj["Questions"]:
        ps = q["Parses"]
        withc = [p for p in ps if p.get("InferentialChain")]
        if withc:
            p = min(withc, key=lambda p: (len(p["InferentialChain"]), p["ParseId"]))
            clen = len(p["InferentialChain"])
        else:
            p, clen = ps[0], None
        cons = p.get("Constraints") or []
        out[q["QuestionId"]] = {"chain_len": clen, "order": bool(p.get("Order")), "time": bool(p.get("Time")), "n_constraints": len(cons),
                                "ent_equal": any(c.get("Operator") == "Equal" and c.get("ArgumentType") == "Entity" for c in cons), "n_parses": len(ps)}
    return out


def nsm_webqsp_hops(rows):
    """shortest topic -> answer path inside the NSM subgraph tuples (undirected), WebQSP train_simple + dev_simple ONLY. Returns {qid: (min, max, n_reached, n_mid_answers)}."""
    want = {x["qid"]: x for x in rows if x["source"] == "nsm_webqsp"}
    need = set()
    for x in want.values():
        for v, z in zip(x["answer_values"], x["answer_is_mid"]):
            if z:
                need.add(v)
    idx = {}
    with open(os.path.join(NSM_WQ, "entities.txt"), encoding="utf-8") as fh:
        for i, l in enumerate(fh):
            l = l.rstrip("\n")
            if l in need:
                idx[l] = i
    res = {}
    for fn in ("train_simple.json", "dev_simple.json"):
        n = 0
        with open(os.path.join(NSM_WQ, fn), "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                if SMOKE is not None and n >= SMOKE:
                    break
                n += 1
                d = json.loads(line)
                x = want.get(d["id"])
                if x is None:
                    continue
                adj = collections.defaultdict(list)
                for h, _r, t in d["subgraph"]["tuples"]:
                    adj[h].append(t)
                    adj[t].append(h)
                dist = {}
                dq = []
                for t in d["entities"]:
                    dist[t] = 0
                    dq.append(t)
                for u in dq:
                    for v in adj.get(u, ()):
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            dq.append(v)
                am = [v for v, z in zip(x["answer_values"], x["answer_is_mid"]) if z]
                ds = [dist[idx[m]] for m in am if m in idx and idx[m] in dist]
                ds_pos = [z for z in ds if z >= 1]
                res[d["id"]] = (min(ds_pos) if ds_pos else None, max(ds_pos) if ds_pos else None, len(ds_pos), len(am))
    return res


def main():
    t_start = time.time()
    prov = json.load(open(PROV, encoding="utf-8"))
    # ---------------- 1. load v1 (NSM WebQSP + CWQ) ----------------
    cols = ["source", "split", "qid", "question", "topic_mids", "topic_pos", "topic_kinds", "answer_values", "answer_text", "answer_is_mid", "answer_pos", "answer_kinds",
            "n_topics", "n_answers", "n_answer_literals", "n_unresolved_topic", "n_unresolved_answer", "n_unresolved", "fully_mappable", "eval_flag", "eval_eligible",
            "cwq_parent_id", "cwq_parent_kind", "parent_eval_flag", "n_cwq_children"]
    v1 = pq.read_table(V1, columns=cols).to_pylist()
    if SMOKE:
        keep = collections.defaultdict(int)
        sm = []
        for x in v1:
            k = (x["source"], x["split"])
            if keep[k] < SMOKE:
                keep[k] += 1
                sm.append(x)
        v1 = sm
    say("v1 rows", len(v1))
    # ---------------- 2. parse new sources ----------------
    new_rows, parse_meta = [], {}
    for sp in ("train", "dev"):
        r, m = parse_freebaseqa(sp, SMOKE)
        new_rows += r
        parse_meta["freebaseqa/" + sp] = m
        r, m = parse_grailqa(sp, SMOKE)
        new_rows += r
        parse_meta["grailqa/" + sp] = m
        say("parsed new", sp, len(new_rows))
    rss_note()
    # ---------------- 3. resolve all MIDs ----------------
    mids = set()
    for x in v1:
        mids.update(x["topic_mids"])
        mids.update(v for v, z in zip(x["answer_values"], x["answer_is_mid"]) if z)
    for x in new_rows:
        mids.update(x["topic_mids"])
        mids.update(v for v, _t, z in x["answers"] if z)
    mids = sorted(mids)
    pos, kind, rinfo = resolve_positions(mids)
    pmap = dict(zip(mids, pos.tolist()))
    kmap = dict(zip(mids, kind.tolist()))
    del pos, kind
    say("resolved", {k: v for k, v in rinfo.items() if k != "node_uid_sha256_computed"})
    # v1 reproducibility: every v1 position/kind must be reproduced by the independent re-resolution
    mism = 0
    for x in v1:
        for m, p, k in zip(x["topic_mids"], x["topic_pos"], x["topic_kinds"]):
            if pmap[m] != p:
                mism += 1
        for m, z, p in zip(x["answer_values"], x["answer_is_mid"], x["answer_pos"]):
            if z and pmap[m] != p:
                mism += 1
    assert mism == 0, "v1 positions not reproduced: %d" % mism
    say("v1 positions reproduced exactly (0 mismatches)")
    # ---------------- 4. per-row schema for new rows ----------------
    for x in new_rows:
        tp = [pmap[m] for m in x["topic_mids"]]
        tk = [kmap[m] for m in x["topic_mids"]]
        av, at, am_, ap_, ak = [], [], [], [], []
        for v, t, z in x["answers"]:
            av.append(v)
            at.append(t)
            am_.append(z)
            ap_.append(pmap[v] if z else -1)
            ak.append(kmap[v] if z else -1)
        x.update(topic_pos=tp, topic_kinds=tk, answer_values=av, answer_text=at, answer_is_mid=am_, answer_pos=ap_, answer_kinds=ak)
        x["n_topics"] = len(tp)
        x["n_answers"] = len(av)
        x["n_answer_literals"] = sum(1 for z in am_ if not z)
        x["n_unresolved_topic"] = sum(1 for p in tp if p < 0)
        x["n_unresolved_answer"] = sum(1 for z, p in zip(am_, ap_) if z and p < 0)
        x["n_unresolved"] = x["n_unresolved_topic"] + x["n_unresolved_answer"]
        x["fully_mappable"] = bool(x["n_topics"] >= 1 and x["n_unresolved_topic"] == 0 and x["n_answers"] >= 1 and x["n_answer_literals"] == 0 and x["n_unresolved_answer"] == 0)
        x["eval_flag"] = "NEWSRC_DEV_UNRULED" if x["split"] == "dev" else "NEWSRC_TRAIN_POOL"
        x["eval_eligible"] = False
        x["cwq_parent_id"], x["cwq_parent_kind"], x["parent_eval_flag"], x["n_cwq_children"] = "", "", "", 0
        del x["answers"]
    rows = v1 + new_rows
    del v1, new_rows
    assert len({x["qid"] for x in rows}) == len(rows), "qid not globally unique"
    for x in rows:
        for k, dv in (("hops_min", None), ("hops_max", None), ("fqa_n_parses", 0), ("fqa_parse_answers_disagree", False), ("grail_function", ""), ("grail_num_edge", -1), ("grail_num_node", -1),
                      ("grail_n_entity_nodes", -1), ("grail_n_literal_nodes", -1), ("grail_level", ""), ("grail_domains", ""), ("cwq_comp_type", ""), ("cwq_webqsp_id", ""),
                      ("cwq_orig_present", False), ("wq_orig_chain_len", -1), ("wq_orig_n_constraints", -1), ("wq_nsm_hops_min", -1), ("wq_nsm_hops_max", -1), ("wq_nsm_answers_reached", -1)):
            x.setdefault(k, dv)
        x["dev_candidate_pending"] = x["eval_flag"] == "NEWSRC_DEV_UNRULED"
    say("rows", len(rows))

    # ---------------- 5. hops + type per source ----------------
    # 5a WebQSP (primary: original annotation; cross-check: NSM subgraph shortest path)
    wq_orig = load_webqsp_orig()
    nsm_h = nsm_webqsp_hops(rows)
    say("webqsp orig + nsm hops", len(wq_orig), len(nsm_h))
    cwq_o, cwq_meta = load_cwq_original(SMOKE)
    say("cwq original", {k: v["rows"] for k, v in cwq_meta.items()})
    wq_agree = collections.Counter()
    for x in rows:
        s = x["source"]
        if s == "nsm_webqsp":
            o = wq_orig.get(x["qid"])
            nh = nsm_h.get(x["qid"])
            if nh:
                x["wq_nsm_hops_min"] = -1 if nh[0] is None else nh[0]
                x["wq_nsm_hops_max"] = -1 if nh[1] is None else nh[1]
                x["wq_nsm_answers_reached"] = nh[2]
            if o:
                x["wq_orig_chain_len"] = -1 if o["chain_len"] is None else o["chain_len"]
                x["wq_orig_n_constraints"] = o["n_constraints"]
            if o and o["chain_len"]:
                x["hops"] = o["chain_len"]
                x["hops_method"] = "webqsp_orig_inferential_chain_primary_parse"
            elif nh and nh[0] is not None:
                x["hops"] = nh[0]
                x["hops_method"] = "webqsp_nsm_subgraph_shortest_path_fallback"
            else:
                x["hops"] = None
                x["hops_method"] = "unknown"
            if o and o["chain_len"] and nh and nh[0] is not None:
                wq_agree[(o["chain_len"], nh[0])] += 1
            x["hops_min"], x["hops_max"] = x["hops"], x["hops"]
            h = x["hops"]
            x["type_hops_only"] = "unknown" if h is None else ("simple" if h == 1 else "chain")
            if o is None:
                x["qtype"], x["type_method"], x["src_type_raw"] = x["type_hops_only"], "webqsp_hops_only(no_orig)", ""
            elif o["order"]:
                x["qtype"], x["type_method"], x["src_type_raw"] = "superlative", "webqsp_orig_Order", "Order"
            elif o["ent_equal"]:
                x["qtype"], x["type_method"], x["src_type_raw"] = "conjunction", "webqsp_orig_EntityEqual_constraint", "EntityEqual"
            elif o["n_constraints"] > 0 or o["time"]:
                x["qtype"], x["type_method"], x["src_type_raw"] = "other", "webqsp_orig_other_constraint_or_Time", "Time/ValueConstraint"
            else:
                x["qtype"], x["type_method"], x["src_type_raw"] = x["type_hops_only"], "webqsp_hops_only", ""
        elif s == "nsm_cwq":
            o = cwq_o.get(x["qid"])
            x["cwq_orig_present"] = o is not None
            if o:
                x["cwq_comp_type"], x["cwq_webqsp_id"] = o["comp"], o["webqsp_ID"]
                x["hops_min"], x["hops_max"] = o["hops_min"], o["hops_max"]
                x["hops"] = o["hops_max"]
                x["hops_method"] = "cwq_sparql_triple_path_max_over_entities" if o["hops_max"] is not None else "unknown"
                x["qtype"] = {"composition": "chain", "conjunction": "conjunction", "comparative": "comparative", "superlative": "superlative"}.get(o["comp"], "other")
                x["type_method"], x["src_type_raw"] = "cwq_compositionality_type", o["comp"]
                h = x["hops"]
                x["type_hops_only"] = "unknown" if h is None else ("simple" if h == 1 else "chain")
            else:
                x["hops"], x["hops_method"] = None, "unknown"
                x["qtype"], x["type_method"], x["src_type_raw"], x["type_hops_only"] = "cwq_type_unknown", "cwq_no_original_match", "", "unknown"
        elif s == "freebaseqa":
            x["hops"] = x["hops_min"]
            x["hops_method"] = "fqa_chain_segments_min_over_parses"
            x["type_hops_only"] = "simple" if x["hops"] == 1 else "chain"
            x["qtype"], x["type_method"], x["src_type_raw"] = x["type_hops_only"], "fqa_hops_only", ""
        else:  # grailqa
            x["hops"] = x["hops_max"]
            x["hops_method"] = "grail_graph_query_path_max_over_entities"
            f = x["grail_function"]
            h = x["hops"]
            x["type_hops_only"] = "unknown" if h is None else ("simple" if h == 1 else "chain")
            if f == "count":
                x["qtype"] = "count"
            elif f in ("argmax", "argmin"):
                x["qtype"] = "superlative"
            elif f in (">", ">=", "<", "<="):
                x["qtype"] = "comparative"
            elif x["grail_n_literal_nodes"] > 0:
                x["qtype"] = "other"
            elif x["grail_n_entity_nodes"] >= 2:
                x["qtype"] = "conjunction"
            elif h is None:
                x["qtype"] = "unknown"
            else:
                x["qtype"] = "simple" if h == 1 else "chain"
            x["type_method"], x["src_type_raw"] = "grail_function_and_graph", f
        x["hop_bucket"] = hop_bucket(x["hops"])
        x["ans_bucket"] = ans_bucket(x["n_answers"])
        kinds = [k for z, p, k in zip(x["answer_is_mid"], x["answer_pos"], x["answer_kinds"]) if z and p >= 0]
        if x["n_answers"] > 0 and x["n_answer_literals"] == x["n_answers"]:
            x["ans_kind"] = "literal"
        elif not kinds:
            x["ans_kind"] = "unresolved"
        elif any(k != 0 for k in kinds):
            x["ans_kind"] = "cvt_other"
        else:
            x["ans_kind"] = "entity"
    # CWQ id/parent consistency check against the explicit webqsp_ID
    cwq_parent_match = collections.Counter()
    for x in rows:
        if x["source"] == "nsm_cwq" and x["cwq_orig_present"]:
            cwq_parent_match[x["cwq_parent_id"] == x["cwq_webqsp_id"]] += 1
    # CWQ NSM vs original: question text and answers
    cwq_cmp = collections.Counter()
    for x in rows:
        if x["source"] == "nsm_cwq":
            o = cwq_o.get(x["qid"])
            if o is None:
                cwq_cmp["no_original_match"] += 1
                continue
            cwq_cmp["original_matched"] += 1
            cwq_cmp["question_text_equal"] += int(o["q"].strip() == x["question"].strip())
            nsm_am = sorted({v for v, z in zip(x["answer_values"], x["answer_is_mid"]) if z})
            cwq_cmp["answer_mid_sets_equal"] += int(nsm_am == o["ans"])
            cwq_cmp["split_equal"] += int(o["orig_split"] == x["split"])
    cwq_only_in_orig = len(set(cwq_o) - {x["qid"] for x in rows if x["source"] == "nsm_cwq"})
    rss_note()
    say("hops/type done")

    # ---------------- 6. flags: rank, leak_clean, family ----------------
    for x in rows:
        x["rank"] = rank_hex(x["qid"])
        x["norm_q"] = norm_q(x["question"])
        pk, pf = x["cwq_parent_kind"], x["parent_eval_flag"]
        x["leak_clean"] = not (pk == "WebQTest" or (pk == "WebQTrn" and pf == "DEV_L3W"))
        x["leak_clean_strict"] = x["leak_clean"] and not (pk == "WebQTrn" and pf in ("NSM_DEV", "DEV_L3W"))
        if x["source"] == "nsm_cwq":
            x["family_id"] = x["cwq_parent_id"] or ("cwq:" + x["qid"])
            x["family_role"] = "child" if x["cwq_parent_id"] else "single"
        elif x["source"] == "nsm_webqsp":
            x["family_id"] = x["qid"]
            x["family_role"] = "parent"
        else:
            x["family_id"] = x["source"] + ":" + x["qid"]
            x["family_role"] = "single"

    def prio(x):
        # representative of a duplicate group: (0) eval-eligible rows first, (1) then not-yet-ruled new-source dev rows (so a possibly-held-out row is not replaced by a train twin), (2) then the rest;
        # within a tier fully-mappable first, then lowest sha256(seed+qid) rank
        return (0 if x["eval_eligible"] else (1 if x["dev_candidate_pending"] else 2), 0 if x["fully_mappable"] else 1, x["rank"])

    stage = {"raw": collections.Counter((x["source"], x["split"]) for x in rows)}
    stage_fm = {"raw": collections.Counter((x["source"], x["split"]) for x in rows if x["fully_mappable"])}
    audit = []
    dropped = {}
    # ---------------- 7. dedupe rule 1: normalised question ----------------
    g1 = collections.defaultdict(list)
    for x in rows:
        g1[x["norm_q"]].append(x)
    r1 = {"groups_gt1": 0, "rows_in_groups_gt1": 0, "dropped": 0, "dropped_exact_text": 0, "dropped_norm_only": 0, "cross_source_groups": 0, "dropped_same_signature_as_rep": 0,
          "dropped_eval_eligible_rows": 0, "group_size_hist": collections.Counter(), "cross_source_pairs": collections.Counter(), "dropped_by_source_split": collections.Counter()}
    for g, mem in g1.items():
        if len(mem) < 2:
            continue
        r1["groups_gt1"] += 1
        r1["rows_in_groups_gt1"] += len(mem)
        r1["group_size_hist"][len(mem)] += 1
        srcs = sorted({m["source"] for m in mem})
        if len(srcs) > 1:
            r1["cross_source_groups"] += 1
            r1["cross_source_pairs"]["+".join(srcs)] += 1
        mem.sort(key=prio)
        rep = mem[0]
        for m in mem:
            m["r1_group_size"] = len(mem)
        for m in mem[1:]:
            dropped[m["qid"]] = "R1_normalised_question"
            exact = m["question"].strip() == rep["question"].strip()
            sig_same = (sorted(set(m["topic_mids"])) == sorted(set(rep["topic_mids"]))) and (sorted(set(m["answer_values"])) == sorted(set(rep["answer_values"])))
            r1["dropped"] += 1
            r1["dropped_exact_text"] += int(exact)
            r1["dropped_norm_only"] += int(not exact)
            r1["dropped_same_signature_as_rep"] += int(sig_same)
            r1["dropped_eval_eligible_rows"] += int(m["eval_eligible"])
            r1["dropped_by_source_split"][m["source"] + "/" + m["split"]] += 1
            audit.append({"qid": m["qid"], "source": m["source"], "split": m["split"], "rule": "R1_normalised_question", "rep_qid": rep["qid"], "rep_source": rep["source"], "group_size": len(mem),
                          "exact_text": exact, "same_signature_as_rep": sig_same, "eval_eligible": m["eval_eligible"], "question": m["question"]})
    for x in rows:
        x.setdefault("r1_group_size", 1)
    surv1 = [x for x in rows if x["qid"] not in dropped]
    stage["after_R1"] = collections.Counter((x["source"], x["split"]) for x in surv1)
    stage_fm["after_R1"] = collections.Counter((x["source"], x["split"]) for x in surv1 if x["fully_mappable"])
    say("R1 dropped", r1["dropped"])
    # ---------------- 8. dedupe rule 2: identical (topic set, answer set) signature ----------------
    def sig_of(x):
        if x["n_topics"] < 1 or x["n_answers"] < 1 or x["n_answer_literals"] > 0:
            return None
        return (tuple(sorted(set(x["topic_mids"]))), tuple(sorted(set(x["answer_values"]))))

    g2 = collections.defaultdict(list)
    for x in surv1:
        s = sig_of(x)
        if s is not None:
            g2[s].append(x)
    r2 = {"groups_gt1": 0, "rows_in_groups_gt1": 0, "dropped": 0, "cross_source_groups": 0, "dropped_eval_eligible_rows": 0, "group_size_hist": collections.Counter(),
          "cross_source_pairs": collections.Counter(), "dropped_by_source_split": collections.Counter(), "max_group_size": 0, "groups_by_composition": collections.Counter(),
          "cross_source_groups_by_pair_incl_dev_flag": collections.Counter()}
    sig_ids = {}
    for s, mem in g2.items():
        if len(mem) < 2:
            continue
        r2["groups_gt1"] += 1
        r2["rows_in_groups_gt1"] += len(mem)
        r2["group_size_hist"][len(mem)] += 1
        r2["max_group_size"] = max(r2["max_group_size"], len(mem))
        srcs = sorted({m["source"] for m in mem})
        sid = "sg_" + hashlib.sha1(repr(s).encode("utf-8")).hexdigest()[:12]
        cross = len(srcs) > 1
        if cross:
            r2["cross_source_groups"] += 1
            r2["cross_source_pairs"]["+".join(srcs)] += 1
        r2["groups_by_composition"]["+".join(sorted(collections.Counter(m["source"] for m in mem).elements()))] += 1
        mem.sort(key=prio)
        rep = mem[0]
        for m in mem:
            m["r2_group_size"] = len(mem)
            m["sig_id"] = sid
            m["sig_cross_source"] = cross
        for m in mem[1:]:
            dropped[m["qid"]] = "R2_signature"
            r2["dropped"] += 1
            r2["dropped_eval_eligible_rows"] += int(m["eval_eligible"])
            r2["dropped_by_source_split"][m["source"] + "/" + m["split"]] += 1
            audit.append({"qid": m["qid"], "source": m["source"], "split": m["split"], "rule": "R2_signature", "rep_qid": rep["qid"], "rep_source": rep["source"], "group_size": len(mem),
                          "exact_text": False, "same_signature_as_rep": True, "eval_eligible": m["eval_eligible"], "question": m["question"]})
    for x in rows:
        x.setdefault("r2_group_size", 1)
        x.setdefault("sig_id", "")
        x.setdefault("sig_cross_source", False)
    pool = [x for x in rows if x["qid"] not in dropped]
    stage["after_R2(pool)"] = collections.Counter((x["source"], x["split"]) for x in pool)
    stage_fm["after_R2(pool)"] = collections.Counter((x["source"], x["split"]) for x in pool if x["fully_mappable"])
    say("R2 dropped", r2["dropped"], "pool", len(pool))
    # family sizes in the pool
    fam_sz = collections.Counter(x["family_id"] for x in pool)
    for x in rows:
        x["family_size_pool"] = fam_sz.get(x["family_id"], 0)
        x["in_pool"] = x["qid"] not in dropped
        x["dropped_by"] = dropped.get(x["qid"], "")

    # ---------------- 9. balanced candidates ----------------
    def cand_base(x):
        return x["in_pool"] and x["fully_mappable"] and x["ans_kind"] != "literal" and x["hops"] is not None and x["qtype"] not in ("unknown", "cwq_type_unknown")

    c_mappable = [x for x in pool if x["fully_mappable"]]
    c_nolit = [x for x in c_mappable if x["ans_kind"] != "literal"]
    c_typed = [x for x in c_nolit if x["hops"] is not None and x["qtype"] not in ("unknown", "cwq_type_unknown")]
    c_leak = [x for x in c_typed if x["leak_clean"]]
    byfam = collections.defaultdict(list)
    for x in c_leak:
        byfam[x["family_id"]].append(x)
    keep = set()
    fam_cut = 0
    for f, mem in byfam.items():
        mem.sort(key=lambda m: (0 if m["family_role"] == "parent" else 1, m["rank"]))
        for m in mem[:FAMILY_CAP]:
            keep.add(m["qid"])
        fam_cut += max(0, len(mem) - FAMILY_CAP)
    cand = [x for x in c_leak if x["qid"] in keep]
    for x in rows:
        x["balanced_candidate"] = x["qid"] in keep
    cstage = {"pool": len(pool), "fully_mappable": len(c_mappable), "excl_literal_answer_kind": len(c_nolit), "hops_and_type_known": len(c_typed),
              "leak_clean(user rule)": len(c_leak), "after_family_cap(%d)" % FAMILY_CAP: len(cand), "rows_removed_by_family_cap": fam_cut}
    cstage_src = {k: collections.Counter(x["source"] for x in v) for k, v in (("pool", pool), ("fully_mappable", c_mappable), ("excl_literal", c_nolit), ("typed", c_typed), ("leak_clean", c_leak), ("family_cap", cand))}
    # sensitivity: balanced candidates without the leak_clean restriction
    byfam2 = collections.defaultdict(list)
    for x in c_typed:
        byfam2[x["family_id"]].append(x)
    n_unres = 0
    for f, mem in byfam2.items():
        n_unres += min(FAMILY_CAP, len(mem))
    cstage["sensitivity_candidates_without_leak_clean_restriction(family-capped)"] = n_unres
    say("candidates", cstage)

    # ---------------- 10. cells (hop bucket x type), with a-priori tiny-cell merge ----------------
    cnt_raw = collections.Counter((x["hop_bucket"], x["qtype"]) for x in cand)
    merge_types = ("comparative", "superlative", "count")
    merge_hb = set()
    for hb in ("1", "2", "3", "4+"):
        # a-priori rule: in a hop bucket, if ANY of comparative/superlative/count has 0 < candidates < MIN_CELL, the three are merged into 'comp_sup_count' for that bucket
        small = [t for t in merge_types if 0 < cnt_raw.get((hb, t), 0) < MIN_CELL]
        if small:
            merge_hb.add(hb)

    def cell_of(x):
        t = x["qtype"]
        if t in merge_types and x["hop_bucket"] in merge_hb:
            t = "comp_sup_count"
        return (x["hop_bucket"], t)

    cells = collections.defaultdict(list)
    for x in cand:
        cells[cell_of(x)].append(x)
    for c in cells.values():
        c.sort(key=lambda m: m["rank"])
    cell_srcs = {c: sorted({m["source"] for m in v}) for c, v in cells.items()}
    HB_ORDER = {"1": 0, "2": 1, "3": 2, "4+": 3}
    T_ORDER = {"simple": 0, "chain": 1, "conjunction": 2, "comparative": 3, "superlative": 4, "comp_sup_count": 5, "count": 6, "other": 7}

    def ckey(c):
        return (HB_ORDER.get(c[0], 9), T_ORDER.get(c[1], 9))

    cell_list = sorted(cells, key=ckey)

    def build_balanced(C):
        """per-cell cap C (first min(C, n) by sha256 rank), then the per-source cap (a source holds at most SRC_CAP of the final set):
        SWAP-ONLY (a cell never shrinks below min(C, n)): while some source s holds more than SRC_CAP of the rows of the MULTI-SOURCE cells (single-source cells are exempt from the cap and are excluded from
        numerator and denominator), take the multi-source cell with the highest in-cell share of s among those where an unselected row of another source exists, and replace s's worst-ranked selected row
        by the best-ranked unselected row of another source. Stops when no source is over the cap or no swap is possible anywhere (residual share is reported, not forced by shrinking cells)."""
        sel = {c: list(v[:C]) for c, v in cells.items()}
        multi = [c for c in cells if len(cell_srcs[c]) >= 2]
        before = collections.Counter(m["source"] for v in sel.values() for m in v)
        before_multi = collections.Counter(m["source"] for c in multi for m in sel[c])
        unsel = {c: {} for c in cells}
        for c, v in cells.items():
            chosen = {m["qid"] for m in sel[c]}
            for m in v:
                if m["qid"] not in chosen:
                    unsel[c].setdefault(m["source"], []).append(m)  # already rank-ascending
        swaps = 0
        stop_reason = "within_cap"
        for _ in range(200000):
            cnt = collections.Counter(m["source"] for c in multi for m in sel[c])
            tot = sum(cnt.values())
            over = [s for s, n in cnt.items() if n > SRC_CAP * tot]
            if not over:
                break
            s = max(over, key=lambda z: cnt[z])
            best_swap = None
            for c in multi:
                v = sel[c]
                ns = sum(1 for m in v if m["source"] == s)
                if ns == 0:
                    continue
                if any(u for src, u in unsel[c].items() if src != s):
                    key = (ns / len(v), ns, c)
                    if best_swap is None or key > best_swap[0]:
                        best_swap = (key, c)
            if best_swap is None:
                stop_reason = "no_swap_possible_for_%s" % s
                break
            c = best_swap[1]
            v = sel[c]
            victim = max((m for m in v if m["source"] == s), key=lambda m: m["rank"])
            new = min((u[0] for src, u in unsel[c].items() if src != s and u), key=lambda m: m["rank"])
            v.remove(victim)
            unsel[c][new["source"]].pop(0)
            v.append(new)
            unsel[c].setdefault(s, []).append(victim)
            unsel[c][s].sort(key=lambda m: m["rank"])
            swaps += 1
        for c in cells:
            sel[c].sort(key=lambda m: m["rank"])
        after = collections.Counter(m["source"] for v in sel.values() for m in v)
        after_multi = collections.Counter(m["source"] for c in multi for m in sel[c])
        return sel, before, after, before_multi, after_multi, swaps, stop_reason

    results = {}
    for C in CS:
        sel, before, after, before_multi, after_multi, swaps, stop_reason = build_balanced(C)
        tot_multi = sum(after_multi.values())
        tot = sum(len(v) for v in sel.values())
        per_cell = {}
        for c in cell_list:
            v = sel[c]
            per_cell["%s|%s" % c] = {"hop_bucket": c[0], "type": c[1], "candidates": len(cells[c]), "selected": len(v), "exhausted": len(cells[c]) < C,
                                     "sources_available": {s: sum(1 for m in cells[c] if m["source"] == s) for s in cell_srcs[c]},
                                     "sources_selected": dict(collections.Counter(m["source"] for m in v)), "answer_size_selected": dict(collections.Counter(m["ans_bucket"] for m in v)),
                                     "leak_clean_selected": sum(1 for m in v if m["leak_clean"]), "eval_eligible_selected": sum(1 for m in v if m["eval_eligible"])}
        n_exh = sum(1 for c in cell_list if len(cells[c]) < C)
        allv = [m for v in sel.values() for m in v]
        results[C] = {"total": tot, "cells_nonempty": len(cell_list), "cells_exhausted": n_exh, "cells_full": len(cell_list) - n_exh, "source_counts_before_cap": dict(before), "source_counts_after_cap": dict(after),
                      "source_share_after_cap": {s: round(n / tot, 4) for s, n in after.items()}, "swaps_in_cap_step": swaps, "cap_stop_reason": stop_reason, "multi_source_cells_rows": tot_multi,
                      "source_counts_multi_before_cap": dict(before_multi), "source_counts_multi_after_cap": dict(after_multi),
                      "source_share_multi_after_cap": {s: round(n / tot_multi, 4) for s, n in after_multi.items()} if tot_multi else {},
                      "answer_size_mix": dict(collections.Counter(m["ans_bucket"] for m in allv)), "eval_eligible_rows": sum(1 for m in allv if m["eval_eligible"]),
                      "leak_clean_strict_rows": sum(1 for m in allv if m["leak_clean_strict"]), "by_hop_bucket": dict(collections.Counter(m["hop_bucket"] for m in allv)),
                      "by_type": dict(collections.Counter(m["qtype"] for m in allv)), "cells": per_cell, "_sel": sel}
    say("balanced totals", {C: results[C]["total"] for C in CS})

    # ---------------- 11. power table ----------------
    def mde(n, pd_, z_a=1.959964, z_b=0.841621):
        lo, hi = 0.0, math.sqrt(pd_) - 1e-9
        for _ in range(100):
            d = (lo + hi) / 2
            need = (z_a * math.sqrt(pd_) + z_b * math.sqrt(max(pd_ - d * d, 1e-12))) ** 2 / (d * d)
            if need > n:
                lo = d
            else:
                hi = d
        return round((lo + hi) / 2, 4)

    power = {"method": "paired binary outcome (hit/miss per question), two systems, McNemar/normal approximation, 2-sided alpha=0.05, power 0.80; MDE = smallest true difference in hit rate detectable at n with discordant-pair rate p_d; Bonferroni column = alpha/14 (about 14 cells)",
             "table": {}}
    for n in (50, 100, 200, 400, 800, 1600, 3200, 6400):
        power["table"][str(n)] = {"pd=%.2f" % pd_: mde(n, pd_) for pd_ in (0.05, 0.10, 0.20, 0.30)}
        power["table"][str(n)]["pd=0.10_bonferroni14"] = mde(n, 0.10, z_a=3.2)

    # ---------------- 12. statistics for the record ----------------
    def frac(a, b):
        return None if b == 0 else round(a / b, 6)

    resolution = {}
    cells_sp = collections.defaultdict(list)
    for x in rows:
        cells_sp[(x["source"], x["split"])].append(x)
    for src in ("nsm_webqsp", "nsm_cwq", "freebaseqa", "grailqa"):
        for sp in ("train", "dev", "ALL"):
            rs = [x for x in rows if x["source"] == src and (sp == "ALL" or x["split"] == sp)]
            if not rs:
                continue
            tn = sum(x["n_topics"] for x in rs)
            tu = sum(x["n_unresolved_topic"] for x in rs)
            an_mid = sum(sum(x["answer_is_mid"]) for x in rs)
            an_un = sum(x["n_unresolved_answer"] for x in rs)
            resolution[src + "/" + sp] = {"questions": len(rs), "topic_mid_occurrences": tn, "topic_resolved_rate": frac(tn - tu, tn), "answer_mid_occurrences": an_mid,
                                          "answer_mid_resolved_rate": frac(an_mid - an_un, an_mid), "questions_zero_topics": sum(1 for x in rs if x["n_topics"] == 0),
                                          "questions_any_literal_answer": sum(1 for x in rs if x["n_answer_literals"] > 0), "questions_fully_mappable": sum(x["fully_mappable"] for x in rs),
                                          "fully_mappable_rate": frac(sum(x["fully_mappable"] for x in rs), len(rs))}
    # extra: freebaseqa lenient mappability (>=1 resolved topic and all answers resolved)
    fq = [x for x in rows if x["source"] == "freebaseqa"]
    fq_any = sum(1 for x in fq if any(p >= 0 for p in x["topic_pos"]) and x["n_unresolved_answer"] == 0 and x["n_answers"] >= 1)
    census = {}
    for role in ("topic", "answer"):
        for src in ("nsm_webqsp", "nsm_cwq", "freebaseqa", "grailqa"):
            occ, uniq = collections.Counter(), {}
            for x in rows:
                if x["source"] != src:
                    continue
                if role == "topic":
                    it = [(m, p) for m, p in zip(x["topic_mids"], x["topic_pos"])]
                else:
                    it = [(m, p) for m, p, z in zip(x["answer_values"], x["answer_pos"], x["answer_is_mid"]) if z]
                for m, p in it:
                    if p < 0:
                        occ[m] += 1
            census["%s/%s" % (role, src)] = {"unique_unresolved_mids": len(occ), "occurrences": sum(occ.values()), "unique_m.": sum(1 for m in occ if m.startswith("m.")), "unique_g.": sum(1 for m in occ if m.startswith("g.")),
                                             "occurrences_m.": sum(v for m, v in occ.items() if m.startswith("m.")), "occurrences_g.": sum(v for m, v in occ.items() if m.startswith("g."))}
    lit_forms = {}
    for src in ("nsm_webqsp", "nsm_cwq", "freebaseqa", "grailqa"):
        lit_forms[src] = sum(x["n_answer_literals"] for x in rows if x["source"] == src)

    def dist(rs, key):
        return dict(collections.Counter(x[key] for x in rs))

    hops_tab = {}
    type_tab = {}
    for src in ("nsm_webqsp", "nsm_cwq", "freebaseqa", "grailqa"):
        rs = [x for x in pool if x["source"] == src]
        hops_tab[src] = dist(rs, "hop_bucket")
        type_tab[src] = dist(rs, "qtype")
    pool_cells = collections.Counter((x["hop_bucket"], x["qtype"], x["source"]) for x in cand)
    # hops distribution of the raw (pre-dedupe) rows with the exact hop value
    hops_exact = {src: dict(sorted(collections.Counter(x["hops"] for x in rows if x["source"] == src).items(), key=lambda kv: (kv[0] is None, kv[0] or 0))) for src in ("nsm_webqsp", "nsm_cwq", "freebaseqa", "grailqa")}
    # method cross-checks
    xc = {"webqsp_orig_chain_len_vs_nsm_subgraph_min_path": {"%s|%s" % k: v for k, v in sorted(wq_agree.items())},
          "webqsp_agreement_rate": frac(sum(v for (a, b), v in wq_agree.items() if a == b), sum(wq_agree.values())),
          "webqsp_nsm_overestimates": frac(sum(v for (a, b), v in wq_agree.items() if b > a), sum(wq_agree.values())),
          "webqsp_nsm_underestimates": frac(sum(v for (a, b), v in wq_agree.items() if b < a), sum(wq_agree.values())),
          "webqsp_hops_method_counts": dist([x for x in rows if x["source"] == "nsm_webqsp"], "hops_method"),
          "cwq_parent_id_equals_explicit_webqsp_ID": {str(k): v for k, v in cwq_parent_match.items()},
          "cwq_nsm_vs_original": dict(cwq_cmp), "cwq_original_ids_not_in_nsm": cwq_only_in_orig, "cwq_original_files": cwq_meta,
          "cwq_hops_unknown_rows": sum(1 for x in rows if x["source"] == "nsm_cwq" and x["hops"] is None),
          "cwq_comp_type_counts": dist([x for x in rows if x["source"] == "nsm_cwq"], "cwq_comp_type"),
          "grail_function_counts": dist([x for x in rows if x["source"] == "grailqa"], "grail_function"),
          "grail_level_counts_dev": dist([x for x in rows if x["source"] == "grailqa" and x["split"] == "dev"], "grail_level"),
          "grail_num_edge_vs_path_hops": {"%s|%s" % k: v for k, v in sorted(collections.Counter((x["grail_num_edge"], x["hops"]) for x in rows if x["source"] == "grailqa").items(), key=lambda kv: str(kv[0]))},
          "grail_rows_without_entity_node": sum(1 for x in rows if x["source"] == "grailqa" and x["grail_n_entity_nodes"] == 0),
          "fqa_hops_min_vs_max_disagree_rows": sum(1 for x in rows if x["source"] == "freebaseqa" and x["hops_min"] != x["hops_max"]),
          "fqa_parse_answer_disagreement_rows": sum(1 for x in rows if x["source"] == "freebaseqa" and x["fqa_parse_answers_disagree"]),
          "fqa_lenient_mappable_any_parse_topic": fq_any, "fqa_strict_mappable_all_topics": sum(1 for x in fq if x["fully_mappable"])}
    eval_counts = {"eval_eligible_rows_raw": sum(1 for x in rows if x["eval_eligible"]), "eval_eligible_after_dedupe": sum(1 for x in pool if x["eval_eligible"]),
                   "eval_eligible_fully_mappable_pool": sum(1 for x in pool if x["eval_eligible"] and x["fully_mappable"]),
                   "eval_eligible_leak_clean_pool_mappable": sum(1 for x in pool if x["eval_eligible"] and x["fully_mappable"] and x["leak_clean"]),
                   "eval_eligible_leak_clean_strict_pool_mappable": sum(1 for x in pool if x["eval_eligible"] and x["fully_mappable"] and x["leak_clean_strict"]),
                   "eval_eligible_by_flag_pool_mappable_leak_clean": dist([x for x in pool if x["eval_eligible"] and x["fully_mappable"] and x["leak_clean"]], "eval_flag"),
                   "eval_eligible_removed_by_leak_clean_by_reason": {"cwq_child_of_WebQTest": sum(1 for x in pool if x["eval_eligible"] and x["fully_mappable"] and x["cwq_parent_kind"] == "WebQTest"),
                                                                       "cwq_child_of_DEV_L3W": sum(1 for x in pool if x["eval_eligible"] and x["fully_mappable"] and x["cwq_parent_kind"] == "WebQTrn" and x["parent_eval_flag"] == "DEV_L3W")},
                   "new_source_dev_rows_pending_ruling": {k: sum(1 for x in rows if x["source"] == k and x["split"] == "dev") for k in ("freebaseqa", "grailqa")},
                   "new_source_dev_rows_fully_mappable_in_pool": {k: sum(1 for x in pool if x["source"] == k and x["split"] == "dev" and x["fully_mappable"]) for k in ("freebaseqa", "grailqa")},
                   "pool_rows_not_leak_clean": sum(1 for x in pool if not x["leak_clean"]), "pool_rows_not_leak_clean_strict": sum(1 for x in pool if not x["leak_clean_strict"])}
    return dict(rows=rows, pool=pool, audit=audit, stage=stage, stage_fm=stage_fm, r1=r1, r2=r2, cstage=cstage, cstage_src=cstage_src, cells=cells, cell_list=cell_list, cell_srcs=cell_srcs,
                results=results, power=power, resolution=resolution, census=census, lit_forms=lit_forms, hops_tab=hops_tab, type_tab=type_tab, hops_exact=hops_exact, xc=xc, eval_counts=eval_counts,
                parse_meta=parse_meta, rinfo=rinfo, merge_hb=sorted(merge_hb), prov=prov, cnt_raw=cnt_raw, pool_cells=pool_cells, t_start=t_start)


def jsonable(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else ("|".join(map(str, k)) if isinstance(k, tuple) else str(k))): jsonable(v) for k, v in o.items() if k != "_sel"}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


COLS = [("source", pa.string()), ("split", pa.string()), ("qid", pa.string()), ("question", pa.string()), ("topic_mids", pa.list_(pa.string())), ("topic_pos", pa.list_(pa.int64())),
        ("topic_kinds", pa.list_(pa.int8())), ("answer_values", pa.list_(pa.string())), ("answer_text", pa.list_(pa.string())), ("answer_is_mid", pa.list_(pa.bool_())),
        ("answer_pos", pa.list_(pa.int64())), ("answer_kinds", pa.list_(pa.int8())), ("n_topics", pa.int32()), ("n_answers", pa.int32()), ("n_answer_literals", pa.int32()),
        ("n_unresolved_topic", pa.int32()), ("n_unresolved_answer", pa.int32()), ("n_unresolved", pa.int32()), ("fully_mappable", pa.bool_()), ("eval_flag", pa.string()),
        ("eval_eligible", pa.bool_()), ("dev_candidate_pending", pa.bool_()),
        ("hops", pa.int32()), ("hops_min", pa.int32()), ("hops_max", pa.int32()), ("hops_method", pa.string()), ("hop_bucket", pa.string()), ("qtype", pa.string()), ("type_method", pa.string()),
        ("src_type_raw", pa.string()), ("type_hops_only", pa.string()), ("ans_bucket", pa.string()), ("ans_kind", pa.string()),
        ("grail_function", pa.string()), ("grail_num_edge", pa.int32()), ("grail_num_node", pa.int32()), ("grail_n_entity_nodes", pa.int32()), ("grail_n_literal_nodes", pa.int32()),
        ("grail_level", pa.string()), ("grail_domains", pa.string()), ("fqa_n_parses", pa.int32()), ("fqa_parse_answers_disagree", pa.bool_()),
        ("cwq_comp_type", pa.string()), ("cwq_webqsp_id", pa.string()), ("cwq_orig_present", pa.bool_()), ("wq_orig_chain_len", pa.int32()), ("wq_orig_n_constraints", pa.int32()),
        ("wq_nsm_hops_min", pa.int32()), ("wq_nsm_hops_max", pa.int32()), ("wq_nsm_answers_reached", pa.int32()),
        ("norm_q", pa.string()), ("r1_group_size", pa.int32()), ("r2_group_size", pa.int32()), ("sig_id", pa.string()), ("sig_cross_source", pa.bool_()),
        ("family_id", pa.string()), ("family_role", pa.string()), ("family_size_pool", pa.int32()), ("cwq_parent_id", pa.string()), ("cwq_parent_kind", pa.string()),
        ("parent_eval_flag", pa.string()), ("n_cwq_children", pa.int32()), ("leak_clean", pa.bool_()), ("leak_clean_strict", pa.bool_()), ("rank", pa.string()),
        ("in_pool", pa.bool_()), ("dropped_by", pa.string()), ("balanced_candidate", pa.bool_())]


def to_table(rs, extra):
    sch = pa.schema([pa.field(n, t) for n, t in COLS] + [pa.field(n, t) for n, t in extra])
    data = {}
    for n, t in COLS:
        vals = [x.get(n) for x in rs]
        if n in ("hops", "hops_min", "hops_max"):
            vals = [-1 if v is None else v for v in vals]
        data[n] = vals
    for n, _t in extra:
        data[n] = [x.get(n) for x in rs]
    return pa.Table.from_pydict(data, schema=sch)


def write_outputs(R, rec_c):
    rows, pool, results = R["rows"], R["pool"], R["results"]
    sel = results[rec_c]["_sel"]
    selq = {m["qid"]: c for c, v in sel.items() for m in v}
    extra = [("balanced_cell", pa.string())] + [("in_balanced_C%d" % C, pa.bool_()) for C in CS]
    selsets = {C: {m["qid"] for v in results[C]["_sel"].values() for m in v} for C in CS}
    cellname = {m["qid"]: "%s|%s" % c for c, v in sel.items() for m in v}
    for x in pool:
        x["balanced_cell"] = cellname.get(x["qid"], "")
        for C in CS:
            x["in_balanced_C%d" % C] = x["qid"] in selsets[C]
    bal = [x for x in pool if x["qid"] in selq]
    bal.sort(key=lambda x: (ckey_global(x), x["rank"]))
    pool.sort(key=lambda x: (x["source"], x["split"], x["qid"]))
    out = {}
    for key, rs in (("pool", pool), ("bal", bal)):
        t = to_table(rs, extra)
        with open(OUT[key], "xb") as fh:
            pq.write_table(t, fh, compression="zstd")
        out[key] = {"path": os.path.relpath(OUT[key], ROOT).replace("\\", "/") if not DRY else OUT[key], "rows": len(rs), "bytes": os.path.getsize(OUT[key]), "sha256": sha_file(OUT[key])}
    aud = R["audit"]
    sch = pa.schema([("qid", pa.string()), ("source", pa.string()), ("split", pa.string()), ("rule", pa.string()), ("rep_qid", pa.string()), ("rep_source", pa.string()), ("group_size", pa.int32()),
                     ("exact_text", pa.bool_()), ("same_signature_as_rep", pa.bool_()), ("eval_eligible", pa.bool_()), ("question", pa.string())])
    t = pa.Table.from_pydict({f.name: [a[f.name] for a in aud] for f in sch}, schema=sch)
    with open(OUT["audit"], "xb") as fh:
        pq.write_table(t, fh, compression="zstd")
    out["audit"] = {"path": os.path.relpath(OUT["audit"], ROOT).replace("\\", "/") if not DRY else OUT["audit"], "rows": len(aud), "bytes": os.path.getsize(OUT["audit"]), "sha256": sha_file(OUT["audit"])}
    return out


_CELL_KEY = {}


def ckey_global(x):
    return _CELL_KEY.get(x["qid"], (9, 9))


def main_run():
    R = main()
    results = R["results"]
    rec_c = args.c_recommended if args.c_recommended else 400
    HB = {"1": 0, "2": 1, "3": 2, "4+": 3}
    TT = {"simple": 0, "chain": 1, "conjunction": 2, "comparative": 3, "superlative": 4, "comp_sup_count": 5, "count": 6, "other": 7}
    for c, v in results[rec_c]["_sel"].items():
        for m in v:
            _CELL_KEY[m["qid"]] = (HB.get(c[0], 9), TT.get(c[1], 9))
    files = write_outputs(R, rec_c)
    say("outputs written", files)
    rss_note()
    return R, files, rec_c


def tab_md(R, rec_c):
    L = []
    results, cell_list = R["results"], R["cell_list"]
    L.append("# FBQ_SET v2 -- merged, harmonised, deduped, balanced Freebase question set\n")
    L.append("Generated by `scratchpad/_fbq_build_set_v2.py` (PYTHONHASHSEED=0), downloads by `scratchpad/_fbq_download_v2.py` (provenance: `FBQ_DOWNLOAD_PROVENANCE__v1.json`). Record: `FBQ_SET__v2.json`. "
             "Held-out rule: NSM test_simple, WebQSP.test.json, GrailQA test (never extracted), FreebaseQA eval/partial, CWQ original test were never opened; nothing was tuned on retrieval results.\n")
    L.append("## Stage counts (rows)\n")
    srcs = ["nsm_webqsp/train", "nsm_webqsp/dev", "nsm_cwq/train", "nsm_cwq/dev", "freebaseqa/train", "freebaseqa/dev", "grailqa/train", "grailqa/dev"]
    L.append("| stage | " + " | ".join(srcs) + " | ALL |\n|---|" + "---|" * (len(srcs) + 1))
    for k in ("raw", "after_R1", "after_R2(pool)"):
        c = R["stage"][k]
        vals = [c.get(tuple(s.split("/")), 0) for s in srcs]
        L.append("| %s | %s | %d |" % (k, " | ".join(map(str, vals)), sum(vals)))
    for k in ("raw", "after_R1", "after_R2(pool)"):
        c = R["stage_fm"][k]
        vals = [c.get(tuple(s.split("/")), 0) for s in srcs]
        L.append("| %s, fully mappable | %s | %d |" % (k, " | ".join(map(str, vals)), sum(vals)))
    L.append("\nBalanced-candidate funnel (pool -> mappable -> literal-excluded -> hops/type known -> leak-clean -> family cap): `%s`\n" % json.dumps(R["cstage"]))
    L.append("## Dedupe\n")
    r1, r2 = R["r1"], R["r2"]
    L.append("- R1 normalised question (lowercase, accents stripped, punctuation removed, whitespace collapsed; representative = eval-eligible first, then not-yet-ruled new-source dev row, then the rest; within a tier fully mappable first, then lowest sha256(seed+qid)): groups>1 %d, rows in them %d, **removed %d** (exact-text %d, normalised-only %d), cross-source groups %d %s, removed rows with the same topic+answer signature as the representative %d, eval-eligible rows removed %d, removed by split %s"
             % (r1["groups_gt1"], r1["rows_in_groups_gt1"], r1["dropped"], r1["dropped_exact_text"], r1["dropped_norm_only"], r1["cross_source_groups"], json.dumps(dict(r1["cross_source_pairs"])), r1["dropped_same_signature_as_rep"], r1["dropped_eval_eligible_rows"], json.dumps(dict(r1["dropped_by_source_split"]))))
    L.append("- R2 identical (sorted topic MIDs, sorted answer MIDs) signature across different questions (rows with a literal answer or no topic skipped): groups>1 %d, rows in them %d, **removed %d**, max group %d, group-size histogram %s, cross-source (flagged) groups %d %s, eval-eligible rows removed %d, removed by split %s, group composition %s"
             % (r2["groups_gt1"], r2["rows_in_groups_gt1"], r2["dropped"], r2["max_group_size"], json.dumps(dict(sorted(r2["group_size_hist"].items()))), r2["cross_source_groups"], json.dumps(dict(r2["cross_source_pairs"])), r2["dropped_eval_eligible_rows"], json.dumps(dict(r2["dropped_by_source_split"])), json.dumps(dict(r2["groups_by_composition"].most_common(12)))))
    L.append("\n## Resolution (positions in the frozen 302M-node tree)\n")
    L.append("| source/split | questions | topic MID resolved | answer MID resolved | zero-topic rows | rows with a literal answer | fully mappable | rate |\n|---|---|---|---|---|---|---|---|")
    for k, v in R["resolution"].items():
        L.append("| %s | %d | %s | %s | %d | %d | %d | %s |" % (k, v["questions"], v["topic_resolved_rate"], v["answer_mid_resolved_rate"], v["questions_zero_topics"], v["questions_any_literal_answer"], v["questions_fully_mappable"], v["fully_mappable_rate"]))
    L.append("\nUnresolved MID census (unique m.* / g.*, occurrences m.* / g.*): " + "; ".join("%s: %d unique (%d m.* / %d g.*), occ %d (%d m.* / %d g.*)" % (k, v["unique_unresolved_mids"], v["unique_m."], v["unique_g."], v["occurrences"], v["occurrences_m."], v["occurrences_g."]) for k, v in R["census"].items()))
    L.append("\n## Hops / type (post-dedupe pool, all rows)\n")
    L.append("hop bucket by source: `%s`\n\ntype by source: `%s`\n" % (json.dumps(R["hops_tab"]), json.dumps(R["type_tab"])))
    L.append("## Balanced candidate cells (hop bucket x type, after mappable/leak-clean/family-cap), merged hop buckets for comparative/superlative/count: %s\n" % R["merge_hb"])
    L.append("| cell | candidates | sources available | candidates by eval flag (eval-eligible = DEV_L3W / NSM_DEV only; NEWSRC_DEV_UNRULED = FreebaseQA/GrailQA official dev, eligibility NOT ruled) |\n|---|---|---|---|")
    for c in cell_list:
        L.append("| %s / %s | %d | %s | %s |" % (c[0], c[1], len(R["cells"][c]), json.dumps({s: sum(1 for m in R["cells"][c] if m["source"] == s) for s in R["cell_srcs"][c]}),
                                                json.dumps(dict(collections.Counter(m["eval_flag"] for m in R["cells"][c]).most_common()))))
    L.append("\n## Balanced totals by cap C\n")
    L.append("Per-source cap rule: a source holds at most 50% of the rows of the multi-source cells (single-source cells exempt); enforced by swap-only replacement inside cells, never by shrinking a cell below min(C, candidates).\n")
    L.append("| C | total | non-empty cells | exhausted cells (< C) | full cells | source counts overall (before cap -> after cap) | source shares in multi-source cells after cap | cap swaps / stop | eval-eligible rows | leak-clean-strict rows | answer sizes |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for C in CS:
        r = results[C]
        L.append("| %d | %d | %d | %d | %d | %s -> %s | %s | %d / %s | %d | %d | %s |" % (C, r["total"], r["cells_nonempty"], r["cells_exhausted"], r["cells_full"], json.dumps(r["source_counts_before_cap"]), json.dumps(r["source_counts_after_cap"]),
                                                                                      json.dumps(r["source_share_multi_after_cap"]), r["swaps_in_cap_step"], r["cap_stop_reason"], r["eval_eligible_rows"], r["leak_clean_strict_rows"], json.dumps(r["answer_size_mix"])))
    for C in CS:
        r = results[C]
        L.append("\n### C = %d%s (total %d)\n" % (C, " (RECOMMENDED, written to BALANCED.parquet)" if C == rec_c else "", r["total"]))
        L.append("| cell | cand | selected | exhausted | source mix selected | answer size selected |\n|---|---|---|---|---|---|")
        for k, v in r["cells"].items():
            L.append("| %s / %s | %d | %d | %s | %s | %s |" % (v["hop_bucket"], v["type"], v["candidates"], v["selected"], "yes" if v["exhausted"] else "", json.dumps(v["sources_selected"]), json.dumps(v["answer_size_selected"])))
    L.append("\n## Paired-test resolution (MDE in hit-rate points at 80% power, alpha 0.05)\n")
    L.append("| n | " + " | ".join(next(iter(R["power"]["table"].values())).keys()) + " |\n|---|" + "---|" * len(next(iter(R["power"]["table"].values()))))
    for n, row in R["power"]["table"].items():
        L.append("| %s | %s |" % (n, " | ".join("%.1f" % (100 * v) for v in row.values())))
    L.append("\n" + R["power"]["method"])
    L.append("\n## Eval eligibility\n")
    L.append("`" + json.dumps(R["eval_counts"]) + "`")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    R, files, rec_c = main_run()
    rec = {"RECORD": "FBQ_SET", "version": "v2", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dry_or_smoke": DRY, "smoke_rows_per_file": SMOKE,
           "seed": SEED, "caps_C": CS, "recommended_C": rec_c, "family_cap": FAMILY_CAP, "source_share_cap": SRC_CAP, "min_cell_for_merge": MIN_CELL, "merged_hop_buckets_for_comp_sup_count": R["merge_hb"],
           "stage_counts": {k: {"%s/%s" % kk: v for kk, v in c.items()} for k, c in R["stage"].items()},
           "stage_counts_fully_mappable": {k: {"%s/%s" % kk: v for kk, v in c.items()} for k, c in R["stage_fm"].items()},
           "dedupe_R1": R["r1"], "dedupe_R2": R["r2"], "balanced_funnel": R["cstage"], "balanced_funnel_by_source": R["cstage_src"],
           "resolution": R["resolution"], "unresolved_mid_census": R["census"], "answer_literals_by_source": R["lit_forms"], "hops_bucket_by_source_pool": R["hops_tab"], "type_by_source_pool": R["type_tab"],
           "hops_exact_by_source_raw": R["hops_exact"], "cross_checks": R["xc"], "eval": R["eval_counts"], "parse_meta": R["parse_meta"], "resolve_info": R["rinfo"],
           "cells": {"%s|%s" % c: {"candidates": len(R["cells"][c]), "sources_available": {s: sum(1 for m in R["cells"][c] if m["source"] == s) for s in R["cell_srcs"][c]},
                                               "candidates_by_eval_flag": dict(collections.Counter(m["eval_flag"] for m in R["cells"][c]))} for c in R["cell_list"]},
           "balanced_by_C": jsonable(R["results"]), "power": R["power"], "outputs": files,
           "inputs": {"v1_parquet": {"sha256": sha_file(V1), "bytes": os.path.getsize(V1)},
                      "downloads": {k: {"sha256": v.get("sha256"), "bytes": v.get("measured_bytes")} for k, v in R["prov"]["files"].items()},
                      "grailqa_extracted": {k: {"sha256": v.get("sha256"), "bytes": v.get("bytes")} for k, v in R["prov"]["grailqa_archive_members"].items() if v.get("extracted")},
                      "webqsp_train_json": {"sha256": sha_file(ORIG_WQSP)}, "nsm_webqsp_train_simple": {"sha256": sha_file(os.path.join(NSM_WQ, "train_simple.json"))},
                      "nsm_webqsp_dev_simple": {"sha256": sha_file(os.path.join(NSM_WQ, "dev_simple.json"))}, "provenance_record": {"sha256": sha_file(PROV)}},
           "code": {"scratchpad/_fbq_build_set_v2.py": sha_file(os.path.abspath(__file__)), "scratchpad/_fbq_download_v2.py": sha_file(os.path.join(ROOT, "scratchpad", "_fbq_download_v2.py"))},
           "seconds": round(time.time() - T0, 1)}
    rss_note()
    rec["peak_rss_mb"] = round(peak["rss"] / 2 ** 20, 1)
    with open(OUT["json"], "x", encoding="utf-8", newline="\n") as fh:
        json.dump(jsonable(rec), fh, indent=1, default=str)
        fh.write("\n")
    with open(OUT["md"], "x", encoding="utf-8", newline="\n") as fh:
        fh.write(tab_md(R, rec_c) + "\nseconds %.1f, peak RSS %.1f MB\n" % (rec["seconds"], rec["peak_rss_mb"]))
    say("done", rec["seconds"], "s peak RSS MB", rec["peak_rss_mb"], "outdir", OUTDIR)
