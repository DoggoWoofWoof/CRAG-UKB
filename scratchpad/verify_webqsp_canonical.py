# -*- coding: utf-8 -*-
"""
WEBQSP_CANONICAL_V1_VERIFICATION
================================
Checks the ARTIFACTS the WebQSP build wrote, not the sources it read. The build already
verified its inputs; this reads what landed on disk and asks whether it is internally
consistent, because a build report is the builder's own account of itself.

THE CHECK THAT MATTERS MOST IS 3
    The encoder work order is deduplicated: 2,592,894 nodes collapse to 1,791,533 distinct
    texts, and encoder_row_of_node maps a node to a row of that list. Every WebQSP vector will
    be attached to a node through that map. If it is wrong by even one position, vectors are
    silently attached to the wrong nodes and nothing downstream can detect it -- the shapes all
    still agree, the counts all still agree, and retrieval just quietly returns the wrong
    documents.

    So the map is not spot-checked. Every one of the 2,592,894 nodes has its text compared,
    exactly, against the text of the encoder row it points at.

WHAT THIS DOES NOT CHECK
    Vector content -- there are no WebQSP vectors yet. This gate is what must pass BEFORE
    spending GPU time, because it is the part that would make that spend worthless.
"""
import collections
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

D = "data/final_canonical/webqsp"
OUT = D + "/CANONICAL_V1_VERIFICATION.json"
fail = []


def check(name, ok, detail):
    fail.append(name) if not ok else None
    print("  %-34s %s  %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)
    return {"verdict": "PASS" if ok else "FAIL", "detail": detail}


def main():
    t0 = time.time()
    rep = {"RECORD": "WEBQSP_CANONICAL_V1_VERIFICATION",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "checks": {}}
    build = json.load(io.open(D + "/CANONICAL_V1_BUILD.json", encoding="utf-8"))
    n_claimed = int(build["nodes"]["n"])

    # ---- 1. nodes.jsonl ------------------------------------------------------------------
    n = 0
    bad_id = bad_pos = empty = 0
    seen = set()
    texts = []
    rowmap_texts = []
    with io.open(D + "/nodes.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            o = json.loads(ln)
            n += 1
            if o.get("node_id") != "webqsp:n%d" % i:
                bad_id += 1
            if o.get("position") != i:
                bad_pos += 1
            t = o.get("text") or ""
            if not t.strip():
                empty += 1
            texts.append(t)
            seen.add(o.get("node_id"))
    rep["checks"]["N1_row_count"] = check(
        "N1_row_count", n == n_claimed,
        "%s rows on disk vs %s claimed" % (format(n, ","), format(n_claimed, ",")))
    rep["checks"]["N2_node_id_is_position"] = check(
        "N2_node_id_is_position", bad_id == 0 and bad_pos == 0,
        "node_id != webqsp:n<line>: %d, position != line: %d" % (bad_id, bad_pos))
    rep["checks"]["N3_node_id_unique"] = check(
        "N3_node_id_unique", len(seen) == n, "%s distinct of %s" % (format(len(seen), ","),
                                                                   format(n, ",")))
    rep["checks"]["N4_no_empty_text"] = check(
        "N4_no_empty_text", empty == 0, "%d empty/whitespace texts" % empty)

    # ---- 2. encoder work order ------------------------------------------------------------
    rows = []
    with io.open(D + "/encoder_inputs.jsonl", encoding="utf-8") as f:
        for j, ln in enumerate(f):
            o = json.loads(ln)
            rows.append(o)
            rowmap_texts.append(o.get("text") or "")
    seq_ok = all(r["row"] == j for j, r in enumerate(rows))
    rep["checks"]["E1_rows_are_0_to_N"] = check(
        "E1_rows_are_0_to_N", seq_ok,
        "%s encoder rows, row field is 0..N-1: %s" % (format(len(rows), ","), seq_ok))
    dist = len(set(rowmap_texts))
    rep["checks"]["E2_texts_distinct"] = check(
        "E2_texts_distinct", dist == len(rows),
        "%s distinct texts of %s rows -- a duplicate here means paying twice for one vector"
        % (format(dist, ","), format(len(rows), ",")))

    # ---- 3. THE JOIN: every node's text == the text of the row it points at ---------------
    z = np.load(D + "/encoder_row_of_node.npz")
    m = z[list(z.keys())[0]]
    inrange = bool(m.min() >= 0 and m.max() < len(rows))
    rep["checks"]["M1_map_shape_and_range"] = check(
        "M1_map_shape_and_range", m.size == n and inrange,
        "map size %s vs %s nodes; values in [%d, %d], rows=%s"
        % (format(m.size, ","), format(n, ","), m.min(), m.max(), format(len(rows), ",")))

    mism = 0
    first = None
    if m.size == n and inrange:
        for i in range(n):
            if texts[i] != rowmap_texts[m[i]]:
                mism += 1
                if first is None:
                    first = i
    rep["checks"]["M2_every_node_text_matches_its_row"] = check(
        "M2_every_node_text_matches_its_row", mism == 0,
        "%d of %s nodes disagree with the encoder row they point at%s"
        % (mism, format(n, ","), "" if first is None else " (first: node %d)" % first))
    used = len(set(m.tolist()))
    rep["checks"]["M3_every_row_is_used"] = check(
        "M3_every_row_is_used", used == len(rows),
        "%s of %s encoder rows are referenced -- an unused row is a vector paid for and thrown "
        "away" % (format(used, ","), format(len(rows), ",")))

    # ---- 4. graph -------------------------------------------------------------------------
    gm = json.load(io.open(D + "/graph/GRAPH_MANIFEST.json", encoding="utf-8"))
    fams = gm.get("families") or {}
    gres = {}
    for fam, spec in fams.items():
        if not spec.get("present"):
            # An ABSENT family is a fact about this dataset, not a failure. webqsp is a KB
            # graph with no prose to run NER over and no embeddings yet for kNN, and the
            # standing rule is that absence is recorded rather than filled in with invented
            # edges -- so the verifier records it too instead of demanding a file.
            gres[fam] = {"present": False, "why": spec.get("why")}
            print("  %-34s SKIP  absent by design: %s" % ("G_%s" % fam, spec.get("why")),
                  flush=True)
            continue
        p = os.path.join(os.path.dirname(D), os.path.basename(D), spec.get("file", ""))
        if not os.path.exists(p):
            p = D + "/graph/%s.npz" % fam
        g = np.load(p)
        src, dst = g["src"], g["dst"]
        rel = g["rel"] if "rel" in g else None
        oob = int(((src < 0) | (src >= n) | (dst < 0) | (dst >= n)).sum())
        # relation_vocabulary is PER FAMILY here, not global -- reading it from the top level
        # yields an empty list and then every relation id looks out of range.
        nrel = len(spec.get("relation_vocabulary") or [])
        rel_oob = 0 if rel is None else int(((rel < 0) | (rel >= nrel)).sum())
        gres[fam] = {"present": True, "n_edges": int(src.size),
                     "endpoints_out_of_range": oob, "n_relations_declared": nrel,
                     "relation_ids_out_of_range": rel_oob,
                     "declared_n_edges": spec.get("n_edges")}
        rep["checks"]["G_%s" % fam] = check(
            "G_%s" % fam,
            oob == 0 and rel_oob == 0 and (spec.get("n_edges") in (None, int(src.size))),
            "%s edges, %d endpoints out of [0,%s), %d bad relation ids, manifest says %s"
            % (format(int(src.size), ","), oob, format(n, ","), rel_oob,
               format(spec.get("n_edges"), ",") if spec.get("n_edges") else "-"))
    rep["graph"] = gres

    # ---- 5. queries -----------------------------------------------------------------------
    qn = 0
    gp_oob = idmismatch = 0
    cov = collections.Counter()
    qids = set()
    for sp in sorted(os.listdir(D + "/queries")):
        if not sp.endswith(".jsonl"):
            continue
        for ln in io.open(D + "/queries/" + sp, encoding="utf-8"):
            o = json.loads(ln)
            qn += 1
            qids.add(o["query_id"])
            cov[o.get("gold_coverage")] += 1
            for a, b in zip(o.get("gold_positions") or [], o.get("gold_node_ids") or []):
                if not (0 <= a < n):
                    gp_oob += 1
                elif b != "webqsp:n%d" % a:
                    idmismatch += 1
            if len(o.get("gold_positions") or []) != len(o.get("gold_node_ids") or []):
                idmismatch += 1
    rep["checks"]["Q1_query_ids_unique"] = check(
        "Q1_query_ids_unique", len(qids) == qn,
        "%s distinct of %s" % (format(len(qids), ","), format(qn, ",")))
    rep["checks"]["Q2_gold_positions_in_range"] = check(
        "Q2_gold_positions_in_range", gp_oob == 0, "%d out of [0,%s)" % (gp_oob, format(n, ",")))
    rep["checks"]["Q3_gold_node_ids_match_positions"] = check(
        "Q3_gold_node_ids_match_positions", idmismatch == 0,
        "%d rows where the id string and the position disagree" % idmismatch)
    rep["queries"] = {"n": qn, "gold_coverage": dict(sorted(cov.items()))}

    # ---- 6. hashes ------------------------------------------------------------------------
    hs = {}
    for rel_p in ["nodes.jsonl", "encoder_inputs.jsonl", "encoder_row_of_node.npz",
                  "graph/GRAPH_MANIFEST.json"]:
        p = os.path.join(D, rel_p)
        if not os.path.exists(p):
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                b = f.read(8 << 20)
                if not b:
                    break
                h.update(b)
        hs[rel_p] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
    rep["hashes"] = hs

    rep["VERDICT"] = "ALL_PASS" if not fail else "FAIL"
    rep["failed_checks"] = fail
    rep["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(rep, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nVERDICT: %s   %.0fs" % (rep["VERDICT"], rep["elapsed_s"]))
    print("wrote", OUT)
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
