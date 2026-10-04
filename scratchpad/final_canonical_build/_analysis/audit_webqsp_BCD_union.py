"""SECTIONS B + C + D -- official RoG union, endpoint classification, identity safety.

Streams every parquet record-batch of rmanluo/RoG-webqsp and rmanluo/RoG-cwq, unions and
deduplicates every graph triple, and classifies every distinct graph ENDPOINT.

Memory discipline (16 GB box): record-batch streaming + pyarrow dictionary_encode for
string->code at C speed, ONE shared entity vocab (~2.6M short strings) and relation vocab
(~7k) across both datasets, triples packed into one int64 (ent 22b | rel 13b | ent 22b)
and np.unique'd in bounded chunks.  WebQSP and CWQ are each scanned exactly once; the
union is np.union1d over the two packed sets, which is exact because they share a vocab.

Writes TEMP artifacts under scratchpad/final_canonical_build/_audit/ only.
"""
import glob
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = "C:/Users/Swastik/Desktop/CRAG/scratchpad/final_canonical_build/_audit"
os.makedirs(OUT, exist_ok=True)

WEBQSP_GLOB = "C:/Users/Swastik/Desktop/CRAG/data/original/webqsp/rog_webqsp/*.parquet"
CWQ_GLOB = "C:/Users/Swastik/Desktop/CRAG/data/original/cwq/rog_cwq/*.parquet"

ENT_BITS, REL_BITS = 23, 14   # 8,388,608 entities / 16,384 relations; 23+14+23 = 60 bits < 63
REL_SHIFT, HEAD_SHIFT = ENT_BITS, ENT_BITS + REL_BITS
ENT_MASK, REL_MASK = (1 << ENT_BITS) - 1, (1 << REL_BITS) - 1
FLUSH_AT = 30_000_000
BATCH_ROWS = 200

MID_RE = re.compile(r"^[mg]\.[0-9A-Za-z_]+$")
SCHEMA_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
DATEISH_RE = re.compile(r"^-?\d{1,4}(-\d{2}){0,2}(T[\d:]+Z?)?$")
NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def classify_endpoint(s):
    if s is None:
        return "OTHER", "null"
    t = s.strip()
    if t == "":
        return "OTHER", "empty"
    if MID_RE.match(t):
        return "FREEBASE_MID", "mid"
    if DATEISH_RE.match(t):
        return "OTHER", "date_or_year_literal"
    if NUMERIC_RE.match(t):
        return "OTHER", "numeric_literal"
    if SCHEMA_RE.match(t) and " " not in t:
        return "OTHER", "schema_path_string"
    return "HUMAN_READABLE_SURFACE", "surface"


class Vocab:
    """Shared entity + relation string vocabularies (first-seen order)."""

    def __init__(self):
        self.ent, self.ent_list = {}, []
        self.rel, self.rel_list = {}, []

    def _ids(self, d, lst, strings):
        out = np.empty(len(strings), dtype=np.int64)
        for i, s in enumerate(strings):
            v = d.get(s)
            if v is None:
                v = len(lst)
                d[s] = v
                lst.append(s)
            out[i] = v
        return out

    def ent_ids(self, strings):
        return self._ids(self.ent, self.ent_list, strings)

    def rel_ids(self, strings):
        return self._ids(self.rel, self.rel_list, strings)


class TripleSet:
    def __init__(self, vocab):
        self.v = vocab
        self.buf, self.buf_n = [], 0
        self.packed = np.zeros(0, dtype=np.int64)
        self.raw_instances = 0
        self.bad_arity = 0
        self.null_endpoints = 0

    def add(self, graph_arr):
        lvl1 = graph_arr.flatten()                       # ListArray<string>, one per triple
        if len(lvl1) == 0:
            return
        widths = pc.list_value_length(lvl1)
        wn = widths.to_numpy(zero_copy_only=False)
        if (wn != 3).any():
            self.bad_arity += int((wn != 3).sum())
            lvl1 = lvl1.take(pa.array(np.flatnonzero(wn == 3)))
            if len(lvl1) == 0:
                return
        flat = lvl1.flatten()
        n = len(flat)
        assert n % 3 == 0, n
        if flat.null_count:
            self.null_endpoints += flat.null_count
            flat = pc.fill_null(flat, "")
        enc = flat.dictionary_encode()
        codes = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
        vocab = enc.dictionary.to_pylist()
        m = n // 3
        self.raw_instances += m
        pos = np.arange(n) % 3
        ent_map = np.full(len(vocab), -1, dtype=np.int64)
        rel_map = np.full(len(vocab), -1, dtype=np.int64)
        ei = np.unique(codes[pos != 1])
        ri = np.unique(codes[pos == 1])
        ent_map[ei] = self.v.ent_ids([vocab[i] for i in ei])
        rel_map[ri] = self.v.rel_ids([vocab[i] for i in ri])
        if len(self.v.ent_list) > ENT_MASK or len(self.v.rel_list) > REL_MASK:
            raise OverflowError(f"vocab overflow: ents={len(self.v.ent_list)} rels={len(self.v.rel_list)}")
        c = codes.reshape(m, 3)
        self.buf.append((ent_map[c[:, 0]] << HEAD_SHIFT) | (rel_map[c[:, 1]] << REL_SHIFT) | ent_map[c[:, 2]])
        self.buf_n += m
        if self.buf_n >= FLUSH_AT:
            self.compact()

    def compact(self):
        if self.buf:
            self.packed = np.unique(np.concatenate([self.packed] + self.buf))
            self.buf, self.buf_n = [], 0


def scan(paths, tag, vocab):
    ts = TripleSet(vocab)
    qe, ae = set(), set()
    t0 = time.time()
    for pi, p in enumerate(paths):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=BATCH_ROWS, columns=["graph"]):
            ts.add(b.column("graph"))
        tb = pq.read_table(p, columns=["q_entity", "a_entity"])
        for col, bag in (("q_entity", qe), ("a_entity", ae)):
            for v in tb.column(col).combine_chunks().flatten().to_pylist():
                if v is not None:
                    bag.add(v)
        del tb
        print(f"  [{tag}] {pi+1}/{len(paths)} {os.path.basename(p)[:34]:36s} "
              f"ents={len(vocab.ent_list):,} rels={len(vocab.rel_list):,} "
              f"uniq={ts.packed.size:,}(+{ts.buf_n:,}) raw={ts.raw_instances:,} "
              f"{time.time()-t0:.0f}s", flush=True)
    ts.compact()
    return ts, qe, ae


def ents_of(packed):
    return np.union1d(np.unique((packed >> HEAD_SHIFT) & ENT_MASK), np.unique(packed & ENT_MASK))


def rels_of(packed):
    return np.unique((packed >> REL_SHIFT) & REL_MASK)


def main():
    wq = sorted(glob.glob(WEBQSP_GLOB))
    cq = sorted(glob.glob(CWQ_GLOB))
    print(f"webqsp parquets={len(wq)} cwq parquets={len(cq)}", flush=True)
    if len(wq) != 5:
        print(f"!! WEBQSP INCOMPLETE: {len(wq)}/5", flush=True)
        return 2
    if len(cq) != 24:
        print(f"!! CWQ INCOMPLETE: {len(cq)}/24 shards -- refusing to report a union "
              f"(a WebQSP-only union is the REJECTED object)", flush=True)
        return 2

    V = Vocab()
    res = {"_generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "_packing": f"int64 head({ENT_BITS}) rel({REL_BITS}) tail({ENT_BITS})"}

    wts, wq_q, wq_a = scan(wq, "webqsp", V)
    cts, cq_q, cq_a = scan(cq, "cwq", V)

    w_p, c_p = wts.packed, cts.packed
    u_p = np.union1d(w_p, c_p)
    w_e, c_e, u_e = ents_of(w_p), ents_of(c_p), ents_of(u_p)
    w_r, c_r, u_r = rels_of(w_p), rels_of(c_p), rels_of(u_p)

    res["ROG_WEBQSP"] = {"files": [os.path.basename(x) for x in wq],
                         "ENTITY_N": int(w_e.size), "RELATION_N": int(w_r.size), "TRIPLE_N": int(w_p.size),
                         "raw_triple_instances": wts.raw_instances,
                         "duplicate_triple_instances_collapsed": wts.raw_instances - int(w_p.size),
                         "bad_arity_rows": wts.bad_arity, "null_endpoints": wts.null_endpoints,
                         "q_entity_distinct": len(wq_q), "a_entity_distinct": len(wq_a)}
    res["ROG_CWQ"] = {"files": [os.path.basename(x) for x in cq],
                      "ENTITY_N": int(c_e.size), "RELATION_N": int(c_r.size), "TRIPLE_N": int(c_p.size),
                      "raw_triple_instances": cts.raw_instances,
                      "duplicate_triple_instances_collapsed": cts.raw_instances - int(c_p.size),
                      "bad_arity_rows": cts.bad_arity, "null_endpoints": cts.null_endpoints,
                      "q_entity_distinct": len(cq_q), "a_entity_distinct": len(cq_a)}
    res["UNION"] = {"ENTITY_N": int(u_e.size), "RELATION_N": int(u_r.size), "TRIPLE_N": int(u_p.size),
                    "triples_shared": int(w_p.size + c_p.size - u_p.size),
                    "triples_only_webqsp": int(u_p.size - c_p.size),
                    "triples_only_cwq": int(u_p.size - w_p.size),
                    "entities_shared": int(np.intersect1d(w_e, c_e).size),
                    "entities_only_webqsp": int(np.setdiff1d(w_e, c_e).size),
                    "entities_only_cwq": int(np.setdiff1d(c_e, w_e).size),
                    "relations_shared": int(np.intersect1d(w_r, c_r).size),
                    "self_loops": int((((u_p >> HEAD_SHIFT) & ENT_MASK) == (u_p & ENT_MASK)).sum())}
    for k in ("ROG_WEBQSP", "ROG_CWQ", "UNION"):
        print(k, {a: b for a, b in res[k].items() if isinstance(b, int)}, flush=True)

    # ---------------- SECTION B: endpoint classification ----------------
    EL = V.ent_list
    cls, sub = [None] * len(EL), [None] * len(EL)
    for i, s in enumerate(EL):
        cls[i], sub[i] = classify_endpoint(s)

    def tally(idx):
        return {"N": int(idx.size),
                "classes": dict(Counter(cls[i] for i in idx.tolist())),
                "subtypes": dict(Counter(sub[i] for i in idx.tolist()))}

    res["B_endpoint_classification"] = {"ROG_WEBQSP": tally(w_e), "ROG_CWQ": tally(c_e), "UNION": tally(u_e),
                                        "_rule": {"FREEBASE_MID": MID_RE.pattern,
                                                  "OTHER": "null | empty | date/year literal | numeric literal "
                                                           "| freebase schema-path string",
                                                  "HUMAN_READABLE_SURFACE": "everything else"}}
    ex = {"FREEBASE_MID": [], "HUMAN_READABLE_SURFACE": [], "OTHER": {}}
    for i in u_e.tolist():
        c = cls[i]
        if c == "OTHER":
            ex["OTHER"].setdefault(sub[i], [])
            if len(ex["OTHER"][sub[i]]) < 6:
                ex["OTHER"][sub[i]].append(EL[i])
        elif len(ex[c]) < 6:
            ex[c].append(EL[i])
    res["B_examples"] = ex
    res["B_relation_examples"] = [V.rel_list[i] for i in u_r[:12].tolist()]

    # ---------------- SECTION D: identity safety ----------------
    surf = [i for i in u_e.tolist() if cls[i] == "HUMAN_READABLE_SURFACE"]
    mids = [i for i in u_e.tolist() if cls[i] == "FREEBASE_MID"]
    norm = Counter()
    for i in surf:
        norm[" ".join(EL[i].split()).casefold()] += 1
    dup = {k: v for k, v in norm.items() if v > 1}
    res["D_identity_safety"] = {
        "UNIQUE_SURFACE_N": len(surf),
        "UNIQUE_MID_N": len(mids),
        "EXACT_STRING_DUPLICATE_GROUPS": 0,
        "MAX_COLLISION_SIZE_observable_in_union": 0,
        "_exact_note": "0 BY CONSTRUCTION. The union is keyed on the endpoint string itself, so any two "
                       "Freebase objects that share a surface were ALREADY merged upstream by RoG. The "
                       "collapse is not observable in this artifact; it must be estimated from a "
                       "MID-bearing source (D2).",
        "CASEFOLD_WHITESPACE_DUPLICATE_GROUPS": len(dup),
        "CASEFOLD_MAX_COLLISION_SIZE": max(dup.values()) if dup else 0,
        "CASEFOLD_SURFACES_INVOLVED": int(sum(dup.values())),
        "casefold_examples": sorted(dup.items(), key=lambda kv: -kv[1])[:12],
        "rog_q_entity_that_are_MIDs": {"webqsp": sum(1 for x in wq_q if MID_RE.match(x)), "webqsp_total": len(wq_q),
                                       "cwq": sum(1 for x in cq_q if MID_RE.match(x)), "cwq_total": len(cq_q)},
        "rog_a_entity_that_are_MIDs": {"webqsp": sum(1 for x in wq_a if MID_RE.match(x)), "webqsp_total": len(wq_a),
                                       "cwq": sum(1 for x in cq_a if MID_RE.match(x)), "cwq_total": len(cq_a)},
    }

    # ---------------- gold/topic injection diagnostic (feeds A) ----------------
    w_set = {EL[i] for i in w_e.tolist()}
    res["A_gold_injection_diagnostic"] = {
        "_what": "c1c2_webqsp.py:32-33 add a_entity and q_entity to the DOC set. These counts are the "
                 "RoG-webqsp answer/topic strings that are NOT graph endpoints, i.e. rows that exist in "
                 "the 1,316,466 corpus ONLY because of gold/topic injection.",
        "webqsp_a_entity_not_a_graph_endpoint": sum(1 for x in wq_a if x not in w_set),
        "webqsp_q_entity_not_a_graph_endpoint": sum(1 for x in wq_q if x not in w_set),
        "webqsp_either_not_a_graph_endpoint": len({x for x in (wq_a | wq_q) if x not in w_set}),
    }

    # ---------------- persist TEMP artifacts for section E ----------------
    def dump(path, idx, arr):
        with open(path, "w", encoding="utf-8") as fh:
            for i in idx.tolist():
                fh.write(arr[i].replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r") + "\n")

    dump(OUT + "/union_entities.txt", u_e, EL)
    dump(OUT + "/webqsp_entities.txt", w_e, EL)
    dump(OUT + "/cwq_entities.txt", c_e, EL)
    dump(OUT + "/union_relations.txt", u_r, V.rel_list)
    with open(OUT + "/union_entity_class.txt", "w", encoding="utf-8") as fh:
        for i in u_e.tolist():
            fh.write(cls[i] + "\n")
    np.save(OUT + "/union_packed_triples.npy", u_p)
    np.save(OUT + "/union_entity_ids.npy", u_e)

    with open(OUT + "/BCD_union.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=str)
    print("WROTE " + OUT + "/BCD_union.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
