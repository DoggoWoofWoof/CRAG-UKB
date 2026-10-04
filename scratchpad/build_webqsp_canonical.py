"""
WEBQSP -> canonical_v1  (dataset 6, everything except the encoder)
==================================================================
Builds nodes, graph, queries and the encoder work order for WebQSP in the same canonical
positional space the other five datasets use. It does NOT encode; the encoder bill it emits is
measured, not estimated, and is the input to whatever runs the encoder.

WHICH WEBQSP THIS IS
    The RoG WebQSP+CWQ graph as resolved by the V1 track: 2,592,894 nodes and 8,309,195 edges,
    with node_kind FROZEN (hash 5ea22695...) and retrieval_role = ALL_NODE decided by the user
    on 2026-09-06 -- every node is retrieval-eligible, node_kind selects the SERIALIZER and
    never a permission.

    It is NOT the old 781,485-node legacy substrate (which was sample-conditioned and is
    rejected as canonical), and it is NOT the 1,316,466-node Phase-C tree. Those are different
    corpora; the Phase-C webqsp encodings therefore do not apply here and are not reused.

THE ONE OPEN AXIS, AND WHAT IS ASSUMED
    V1_RETRIEVAL_ROLE_DECISION left entity text undecided between NAME_ONLY and
    NAME_PLUS_FACTS. This build takes NAME_ONLY, which is that record's own DEFAULT arm and its
    stated recommendation: under ALL_NODE the CVT record already carries the composition, so
    NAME_PLUS_FACTS would serialise every CVT argument a second time inside each adjacent
    entity for 2.554x the tokens. Both columns are already materialised in entity_text.parquet,
    so this is a column choice and NAME_PLUS_FACTS remains available as the ablation arm it was
    always intended to be. The choice is recorded in the manifest as an ASSUMPTION.

POSITIONAL SPACE
    node_uid is asserted to be exactly 0..N-1, so canonical position == node_uid == nodes.jsonl
    line number == graph endpoint. The edge table's uids are then usable as positions directly,
    with no remap and no join.

GOLD IS JOINED EXACTLY, NOT NORMALISED
    Gold and topic entities are RoG surface strings. All 2,592,894 source_rog_endpoint values
    are distinct, so an exact match is a bijection onto nodes and cannot be ambiguous. Stripping
    whitespace first would gain exactly ONE additional gold reference while making two surfaces
    collide, so it is not done. 49 endpoints legitimately carry leading or trailing spaces from
    the source and are preserved verbatim.

WHAT THE COVERAGE NUMBERS MEAN
    Reference-level gold coverage is 62.4%, which sounds alarming and is not the number that
    matters: the misses concentrate in list-answer questions that carry dozens of gold entities
    each. Per QUERY, 94.6% have every gold answer present and 96.9% have at least one. Both
    numbers are written to the manifest, because quoting only the flattering one would be the
    whole problem with reporting a single coverage figure.

    A POSITIVE CONTROL is what makes those numbers readable at all: topic entities join at
    99.92%, so the join reaches the population and the gold shortfall is a property of the RoG
    graph's contents rather than a broken key.
"""
import collections
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

V1 = "data/final_canonical/webqsp/v1"
OUT = "data/final_canonical/webqsp"
PHASE_C_QUERIES = "data/canonical/webqsp/queries.jsonl"
ENTITY_TEXT_COLUMN = "text_name_only"      # the open axis; see the docstring
BUF = 8 << 20


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(BUF)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    rep = {"RECORD": "CANONICAL_V1_WEBQSP_BUILD",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source": V1, "entity_text_column": ENTITY_TEXT_COLUMN}

    # ---------- nodes -------------------------------------------------------------------
    nt = pq.read_table(V1 + "/nodes.parquet",
                       columns=["node_uid", "node_key", "source_rog_endpoint", "node_kind",
                                "display_name", "display_name_disambiguated", "source_mid"])
    uid = np.array([int(x) for x in nt["node_uid"].to_pylist()], dtype=np.int64)
    n = int(uid.size)
    order = np.argsort(uid, kind="stable")
    if not np.array_equal(uid[order], np.arange(n, dtype=np.int64)):
        raise SystemExit("node_uid is not exactly 0..N-1; the positional space assumption fails")
    # The check above only proves node_uid is a PERMUTATION of 0..N-1. Every other column is
    # read in parquet row order, so unless the file is already sorted by node_uid, using row
    # order as position would silently attach every node's text to a different node. Reorder
    # explicitly when it is not already sorted, rather than assuming it is.
    in_order = bool(np.array_equal(uid, np.arange(n, dtype=np.int64)))
    print("nodes=%s  node_uid is a permutation of 0..N-1; already_sorted=%s"
          % (format(n, ","), in_order), flush=True)

    def col(name):
        v = nt[name].to_pylist()
        return v if in_order else [v[j] for j in order]

    key = col("node_key")
    ep = col("source_rog_endpoint")
    kind = col("node_kind")
    disp = col("display_name")
    dis2 = col("display_name_disambiguated")
    mid = col("source_mid")
    del nt

    text = [None] * n
    et = pq.read_table(V1 + "/entity_text.parquet",
                       columns=["node_uid", ENTITY_TEXT_COLUMN])
    for u, tx in zip(et["node_uid"].to_pylist(), et[ENTITY_TEXT_COLUMN].to_pylist()):
        text[int(u)] = tx
    n_ent = et.num_rows
    del et
    ct = pq.read_table(V1 + "/cvt_text.parquet", columns=["node_uid", "text_cvt_record"])
    for u, tx in zip(ct["node_uid"].to_pylist(), ct["text_cvt_record"].to_pylist()):
        text[int(u)] = tx
    n_cvt = ct.num_rows
    del ct
    missing = sum(1 for x in text if x is None)
    empty = sum(1 for x in text if x is not None and not x.strip())
    if missing:
        raise SystemExit("%d nodes have no rendered text" % missing)
    print("text: entity=%s cvt=%s  missing=0  whitespace_only=%d"
          % (format(n_ent, ","), format(n_cvt, ","), empty), flush=True)

    os.makedirs(OUT + "/queries", exist_ok=True)
    p_nodes = OUT + "/nodes.jsonl"
    with io.open(p_nodes, "w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            f.write(json.dumps({
                # the field MUST be node_id: pointer_resolver._node_ids reads that key, and
                # all five existing datasets use it. A different name here would make
                # CanonicalEmbeddings("webqsp", ...) fail at construction.
                "node_id": "webqsp:n%d" % i,
                "position": i,
                "text": text[i],
                "display_name": disp[i],
                "display_name_disambiguated": dis2[i],
                "node_kind": kind[i],
                "node_key": key[i],
                "source_rog_endpoint": ep[i],
                "source_mid": mid[i]}, ensure_ascii=False) + "\n")
    rep["nodes"] = {"n": n, "file": p_nodes, "bytes": os.path.getsize(p_nodes),
                    "sha256": sha256_file(p_nodes),
                    "by_node_kind": dict(collections.Counter(kind))}
    print("wrote %s  %.2f GB  %.0fs"
          % (p_nodes, rep["nodes"]["bytes"] / 1e9, time.time() - t0), flush=True)

    # ---------- the encoder work order ---------------------------------------------------
    # Distinct texts only. The pointer index expands them back to all N nodes, so the encoder
    # never sees the same input twice -- the same reuse mechanism the other five datasets use.
    uniq = {}
    row_of = np.empty(n, dtype=np.int64)
    for i in range(n):
        t = text[i]
        j = uniq.get(t)
        if j is None:
            j = len(uniq)
            uniq[t] = j
        row_of[i] = j
    nu = len(uniq)
    p_uniq = OUT + "/encoder_inputs.jsonl"
    inv = [None] * nu
    for t, j in uniq.items():
        inv[j] = t
    with io.open(p_uniq, "w", encoding="utf-8", newline="\n") as f:
        for j in range(nu):
            f.write(json.dumps({"row": j, "text": inv[j]}, ensure_ascii=False) + "\n")
    np.savez(OUT + "/encoder_row_of_node.npz", row=row_of.astype(np.int32))
    chars = int(sum(len(x) for x in inv))
    rep["encoder_work_order"] = {
        "n_nodes": n, "n_distinct_texts": nu,
        "dedup_ratio": round(n / float(nu), 4),
        "distinct_chars": chars,
        "est_tokens_chars_over_4": chars // 4,
        "inputs": p_uniq, "inputs_bytes": os.path.getsize(p_uniq),
        "inputs_sha256": sha256_file(p_uniq),
        "row_of_node": OUT + "/encoder_row_of_node.npz",
        "note": ("encode encoder_inputs.jsonl in row order under the frozen contract, then the "
                 "pointer index for node i is row_of_node[i]. Nothing else needs encoding.")}
    print("encoder bill: %s nodes -> %s distinct texts (%.2fx), %s chars, ~%s tokens"
          % (format(n, ","), format(nu, ","), n / float(nu), format(chars, ","),
             format(chars // 4, ",")), flush=True)

    # ---------- graph -------------------------------------------------------------------
    os.makedirs(OUT + "/graph", exist_ok=True)
    et = pq.read_table(V1 + "/edges.parquet")
    src = np.array([int(x) for x in et["src_uid"].to_pylist()], dtype=np.int64)
    dst = np.array([int(x) for x in et["dst_uid"].to_pylist()], dtype=np.int64)
    rel = np.array([int(x) for x in et["relation_uid"].to_pylist()], dtype=np.int64)
    del et
    bad = int(((src < 0) | (src >= n) | (dst < 0) | (dst >= n)).sum())
    if bad:
        raise SystemExit("%d edges have an endpoint outside 0..N-1" % bad)
    rt = pq.read_table(V1 + "/relations.parquet",
                       columns=["relation_uid", "relation_key", "relation_label_qualified"])
    rmap = {int(u): (k, l) for u, k, l in zip(rt["relation_uid"].to_pylist(),
                                              rt["relation_key"].to_pylist(),
                                              rt["relation_label_qualified"].to_pylist())}
    nrel = int(rel.max()) + 1
    if nrel > 32767:
        raise SystemExit("relation ids exceed int16")
    np.savez(OUT + "/graph/structural.npz",
             src=src.astype(np.int32), dst=dst.astype(np.int32), rel=rel.astype(np.int16))
    n_edges = int(src.size)
    n_self = int((src == dst).sum())
    deg = np.zeros(n, dtype=np.int64)
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    isolated = int((deg == 0).sum())
    # relation_vocabulary is a LIST indexed by relation id -- the convention pointer_resolver
    # and the other five datasets already use. A dict keyed by string would not index.
    vocab = [None] * nrel
    for rid, (rkey, rlab) in rmap.items():
        if 0 <= rid < nrel:
            vocab[rid] = rlab or rkey
    gm = {"RECORD": "CANONICAL_V1_GRAPH_MANIFEST",
          "dataset": "webqsp", "n_nodes": n,
          "endpoint_space": "canonical positions == node_uid == nodes.jsonl line numbers",
          "endpoint_resolution": ("none needed: node_uid is already 0..N-1 and the edge table "
                                  "is expressed in the same ids, so endpoints are used as-is"),
          "direction_policy": "as found in the source; NOT symmetrised",
          "family_policy": ("webqsp carries structural only. Absence is recorded, never filled "
                            "in with invented edges."),
          "nodes_unrepresented_in_graph": [],
          "families": {
              "structural": {"present": True, "file": "graph/structural.npz",
                             "n_edges": int(n_edges), "n_edges_in_source": int(n_edges),
                             "unresolved_endpoint_edges": 0, "unresolved_examples": [],
                             "attributes": "rel(int16) + relation_vocabulary",
                             "relation_vocabulary": vocab, "n_relations": nrel,
                             "self_loops": n_self, "directed": True,
                             "source": V1 + "/edges.parquet",
                             "provenance": "RoG WebQSP+CWQ graph triples, V1-resolved",
                             "edge_subtype": "kb_relation"},
              "ner": {"present": False,
                      "why": "webqsp is a KB graph; there is no document prose to run NER over. "
                             "No edge was invented to match the text datasets."},
              "knn": {"present": False,
                      "why": "not built. kNN needs the embeddings, which are the one thing this "
                             "build deliberately does not produce."}},
          "isolated_nodes": isolated}
    json.dump(gm, io.open(OUT + "/graph/GRAPH_MANIFEST.json", "w", encoding="utf-8"), indent=1)
    rep["graph"] = {"structural_edges": int(src.size), "n_relations": nrel,
                    "unresolved_endpoints": 0, "isolated_nodes": isolated}
    print("graph: %s edges, %d relations, 0 unresolved endpoints, %s isolated nodes  %.0fs"
          % (format(int(src.size), ","), nrel, format(isolated, ","), time.time() - t0),
          flush=True)
    del src, dst, rel

    # ---------- queries + gold ------------------------------------------------------------
    pos_of = {}
    for i in range(n):
        pos_of[ep[i]] = i
    if len(pos_of) != n:
        raise SystemExit("source_rog_endpoint is not unique; the exact join is not a bijection")

    rows = collections.defaultdict(list)
    g_tot = g_hit = t_tot = t_hit = 0
    # Three distinct outcomes, deliberately NOT collapsed into one "no gold" bucket:
    #   FULL / PARTIAL   the corpus holds all / some of the gold entities
    #   NONE_IN_CORPUS   gold entities were given and NONE of them is in the corpus
    #   NO_GOLD_GIVEN    the benchmark itself marks the question unanswerable
    # Merging the last two would blame the corpus for the benchmark's own unanswerables and
    # overstate the coverage shortfall.
    full = part = zero = nogold = 0
    unans_by_split = collections.Counter()
    for ln in io.open(PHASE_C_QUERIES, encoding="utf-8"):
        o = json.loads(ln)
        gold = o.get("gold_entity_names") or []
        topic = o.get("topic_entities") or []
        gp = [pos_of[a] for a in gold if a in pos_of]
        tp = [pos_of[a] for a in topic if a in pos_of]
        g_tot += len(gold); g_hit += len(gp)
        t_tot += len(topic); t_hit += len(tp)
        # An empty gold list is only excusable if the BENCHMARK says the question is
        # unanswerable. Asserting that correspondence would be worthless -- it is exactly the
        # thing that could silently stop being true -- so it is checked in both directions on
        # every row. If a gold list is ever empty for any other reason, that is a join failure
        # wearing an unanswerable costume and it must not pass as "not a corpus gap".
        if bool(gold) == (o.get("answerability") is False):
            raise SystemExit("query %s: gold=%d but answerability=%r -- the empty-gold and "
                             "unanswerable populations are not the same set"
                             % (o["query_id"], len(gold), o.get("answerability")))
        if not gold:
            cov = "NO_GOLD_GIVEN"
            nogold += 1
            unans_by_split[o["official_split"]] += 1
        elif len(gp) == len(gold):
            cov = "FULL"
            full += 1
        elif gp:
            cov = "PARTIAL"
            part += 1
        else:
            cov = "NONE_IN_CORPUS"
            zero += 1
        rows[o["official_split"]].append({
            "query_id": o["query_id"],
            "split": o["official_split"],
            "question": o["question"],
            "answers": o.get("answers") or [],
            "gold_entity_names": gold,
            "gold_positions": gp,
            "gold_node_ids": ["webqsp:n%d" % i for i in gp],
            "topic_entity_names": topic,
            "topic_positions": tp,
            "gold_coverage": cov,
            "answerability": o.get("answerability"),
            "original_index": o.get("original_index")})
    qsum = {}
    for sp, rs in sorted(rows.items()):
        p = "%s/queries/%s.jsonl" % (OUT, sp)
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        qsum[sp] = {"n": len(rs), "file": p, "sha256": sha256_file(p)}
    nq = sum(v["n"] for v in qsum.values())
    rep["queries"] = {
        "n": nq, "by_split": qsum,
        "join": "EXACT match of RoG surface strings onto source_rog_endpoint (a bijection)",
        "POSITIVE_CONTROL_topic_entities": {
            "refs": t_tot, "joined": t_hit,
            "pct": round(100.0 * t_hit / t_tot, 2) if t_tot else None,
            "why": ("the control is what makes the gold shortfall readable: the key reaches the "
                    "population, so what gold misses is absent from the RoG graph rather than "
                    "unmatched by a bad join")},
        "gold_reference_level": {"refs": g_tot, "joined": g_hit,
                                 "pct": round(100.0 * g_hit / g_tot, 2) if g_tot else None},
        "gold_query_level": {
            "FULL": full, "PARTIAL": part, "NONE_IN_CORPUS": zero,
            "NO_GOLD_GIVEN": nogold,
            "answerable_queries": nq - nogold,
            "pct_full_of_answerable": round(100.0 * full / (nq - nogold), 2),
            "pct_at_least_one_of_answerable":
                round(100.0 * (full + part) / (nq - nogold), 2),
            "NO_GOLD_GIVEN_by_split": dict(sorted(unans_by_split.items())),
            "NO_GOLD_GIVEN_is_not_a_corpus_gap": (
                "every one of these %d rows carries the benchmark's own answerability=False "
                "flag, and every flagged row has an empty gold list -- checked in both "
                "directions per row, so this is measured rather than assumed. Folding them "
                "into NONE_IN_CORPUS would blame the corpus for the benchmark's design."
                % nogold)},
        "both_numbers_are_reported_because": (
            "reference-level coverage is %.1f%% and query-level (at least one gold, over "
            "answerable queries) is %.1f%%. The gap is list-answer questions carrying dozens of "
            "gold entities each: missing 30 of 40 answers to one question costs 30 references "
            "and 0 queries. Quoting either number alone would misrepresent the corpus, so both "
            "are computed here rather than written down."
            % (100.0 * g_hit / g_tot, 100.0 * (full + part) / (nq - nogold)))}
    print("queries=%d  gold refs %s/%s (%.2f%%)"
          % (nq, format(g_hit, ","), format(g_tot, ","), 100.0 * g_hit / g_tot), flush=True)
    print("  per-query FULL=%d PARTIAL=%d NONE_IN_CORPUS=%d | NO_GOLD_GIVEN=%d %s"
          % (full, part, zero, nogold, dict(sorted(unans_by_split.items()))), flush=True)
    print("  at least one gold in corpus: %d/%d answerable (%.2f%%)"
          % (full + part, nq - nogold, 100.0 * (full + part) / (nq - nogold)), flush=True)
    print("  POSITIVE CONTROL topic entities %s/%s (%.2f%%)"
          % (format(t_hit, ","), format(t_tot, ","), 100.0 * t_hit / t_tot), flush=True)

    rep["ASSUMPTIONS"] = [
        ("entity text = NAME_ONLY. V1_RETRIEVAL_ROLE_DECISION left this axis open and "
         "recommended NAME_ONLY as the default arm under ALL_NODE; NAME_PLUS_FACTS is already "
         "materialised in entity_text.parquet and stays available as the ablation arm."),
        ("WebQSP was previously scoped as WEBQSP_ROG_RESOLVED, which explicitly excluded the "
         "encoder. The later instruction to complete WebQSP as dataset 6 lists embeddings among "
         "the completeness criteria, so the encoder is back in scope. This build stops just "
         "short of it and hands over a measured work order.")]
    rep["NOT_DONE_HERE"] = ["dense and splade encodings", "pointer index", "retrieval cache",
                            "kNN graph (needs the embeddings)"]
    rep["elapsed_s"] = round(time.time() - t0, 1)
    p = OUT + "/CANONICAL_V1_BUILD.json"
    json.dump(rep, io.open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nwrote %s  %.1fs" % (p, rep["elapsed_s"]))


if __name__ == "__main__":
    main()
