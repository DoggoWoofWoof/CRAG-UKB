"""PHASE 7 (four-way comparison) + PHASE 8 (embedding-reuse projection).

PHASE 7 compares four objects:
  1. legacy 781,485-node substrate  data/processed/master_nodes_webqsp.json
  2. Phase-C 1,316,466 store        data/canonical/webqsp/documents.jsonl
  3. released RoG mixed-endpoint union (2,592,894)  -- _audit/ temp artifacts
  4. reconstructed MID-level graph  -- NOT BUILT (no Freebase source; see Phase 2)

The legacy neighbour-name assignment bug is verified DIRECTLY, not cited: loader_webqsp.py's
`deduce()` is re-implemented here from the same TEST parquet shards the loader read, and the
re-derived titles are compared against the stored ones.

PHASE 8 projects Dense/SPLADE reuse under the canonical text policy.  Reuse is decided by
token-ID equality under the frozen tokenizers; here it is computed at the *text-identity* level
plus the two measured tokenizer correction terms from the prior audit, because a text that
changes cannot possibly reuse and a text that is byte-identical always does.  NOTHING IS ENCODED.

Reads only; writes only _audit/.
"""
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = "C:/Users/Swastik/Desktop/CRAG"
A = os.path.join(ROOT, "scratchpad/final_canonical_build/_audit")
OUT = os.path.join(A, "phase78_compare_reuse.json")

LEGACY_781K = f"{ROOT}/data/processed/master_nodes_webqsp.json"
PHASEC = f"{ROOT}/data/canonical/webqsp/documents.jsonl"
TEST_PARQUETS = [f"{ROOT}/data/raw/full/kb/webqsp_test0.parquet",
                 f"{ROOT}/data/raw/full/kb/webqsp_test1.parquet"]

MID_RE = re.compile(r"^[mg]\.[0-9a-z_]+$")          # loader_webqsp.py:33 -- lowercase only
MID_RE_AUDIT = re.compile(r"^[mg]\.[0-9A-Za-z_]+$")  # audit classifier
LABEL = {0: "NOT_BARE_MID", 1: "CVT_MEDIATOR", 2: "NAME_RESOLVABLE", 3: "MIXED", 4: "UNKNOWN"}


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    res = {"_what": "Phase 7 four-way comparison + Phase 8 reuse projection",
           "_generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---------------- object 3: released RoG union ---------------------------
    with open(os.path.join(A, "union_entities.jsonl"), encoding="utf-8") as fh:
        U = [json.loads(l) for l in fh]
    with open(os.path.join(A, "union_entity_class.txt"), encoding="utf-8") as fh:
        UC = [l.rstrip("\n") for l in fh]
    lab = np.load(os.path.join(A, "union_mid_label_final.npy"))
    upos = {s: i for i, s in enumerate(U)}
    log(f"[union] {len(U):,} endpoints  {time.time()-t0:.0f}s")

    # ---------------- object 1: legacy 781k ----------------------------------
    import pyarrow.parquet as pq
    triples = set()
    for p in TEST_PARQUETS:
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=200, columns=["graph"]):
            for g in b.column("graph").to_pylist():
                for t in g:
                    triples.add((str(t[0]), str(t[1]), str(t[2])))
    adj = defaultdict(list)
    ents = set()
    for h, rel, tl in triples:
        adj[h].append((rel, tl)); adj[tl].append((rel, h)); ents.add(h); ents.add(tl)
    entities = sorted(ents)
    log(f"[legacy] re-derived from TEST shards: triples={len(triples):,} entities={len(entities):,} "
        f"{time.time()-t0:.0f}s")

    is_mid_l = lambda x: bool(MID_RE.match(x))
    named_of = {}

    def deduce(x):                       # verbatim re-implementation of loader_webqsp.py:52-60
        if not is_mid_l(x):
            return x
        if x in named_of:
            return named_of[x]
        nm = next((o for _, o in adj.get(x, []) if not is_mid_l(o)), None)
        named_of[x] = nm
        return nm

    rederived_title = [(e if not is_mid_l(e) else (deduce(e) or "")) for e in entities]

    with open(LEGACY_781K, encoding="utf-8") as fh:
        legacy = json.load(fh)
    if isinstance(legacy, dict):
        legacy = legacy.get("nodes", legacy.get("data", list(legacy.values())))
    log(f"[legacy] stored nodes = {len(legacy):,}")

    stored_title, stored_text = {}, {}
    for nd in legacy:
        nid = nd.get("node_id") or nd.get("id")
        md = nd.get("metadata") or {}
        stored_title[nid] = md.get("title", "")
        stored_text[nid] = nd.get("content") or nd.get("text") or ""

    n_leg = len(entities)
    fabricated, title_match, mid_nodes, mid_no_neighbour = 0, 0, 0, 0
    fabricated_examples, collide_with_real = [], 0
    named_surface_set = {e for e in entities if not is_mid_l(e)}
    for i, e in enumerate(entities):
        nid = f"webqsp_doc_{i}"
        st = stored_title.get(nid)
        rt = rederived_title[i]
        if st == rt:
            title_match += 1
        if is_mid_l(e):
            mid_nodes += 1
            if rt == "":
                mid_no_neighbour += 1
            else:
                fabricated += 1
                if rt in named_surface_set:
                    collide_with_real += 1
                if len(fabricated_examples) < 8:
                    fabricated_examples.append(
                        {"endpoint_MID": e, "node_id": nid, "fabricated_title": rt,
                         "stored_title": st, "stored_text": (stored_text.get(nid) or "")[:110]})
    res["PHASE7_legacy_781k"] = {
        "path": LEGACY_781K,
        "stored_node_count": len(legacy),
        "rederived_entity_count": n_leg,
        "graph_source": "RoG-webqsp TEST shards only (loader_webqsp.py:118-123)",
        "rederived_triples": len(triples),
        "node_id_scheme": "webqsp_doc_<positional index into sorted(entities)>",
        "MID_endpoints": mid_nodes,
        "human_readable_endpoints": n_leg - mid_nodes,
        "FABRICATED_NAMES_rederived": fabricated,
        "MID_with_no_named_neighbour_left_blank": mid_no_neighbour,
        "fabricated_title_equals_a_real_entitys_own_name": collide_with_real,
        "stored_title_matches_rederived_deduce": title_match,
        "stored_title_match_pct": round(100.0 * title_match / max(n_leg, 1), 4),
        "BUG_VERIFIED": bool(fabricated > 0 and title_match > 0.99 * n_leg),
        "_bug": "loader_webqsp.py:52-60 deduce() returns the FIRST non-MID neighbour in adjacency "
                "order and loader_webqsp.py:68 uses it as the node's own name/title.",
        "examples": fabricated_examples,
    }
    log("[legacy] " + json.dumps({k: v for k, v in res["PHASE7_legacy_781k"].items()
                                  if isinstance(v, (int, float, bool))}))
    del legacy, stored_text, adj, triples, named_of

    # ---------------- object 2: Phase-C 1,316,466 ----------------------------
    pc_ids, pc_text_eq_id, pc_id_rule_ok, pc_n = set(), 0, 0, 0
    pc_mid, pc_named = 0, 0
    with open(PHASEC, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            did = r.get("id") or r.get("doc_id") or r.get("node_id")
            txt = r.get("text", "")
            src = r.get("original_source_id", r.get("source_id"))
            pc_n += 1
            pc_ids.add(src if src is not None else txt)
            if src is not None and txt == src:
                pc_text_eq_id += 1
            key = src if src is not None else txt
            if did == "webqsp_ent_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]:
                pc_id_rule_ok += 1
            if MID_RE_AUDIT.match(key):
                pc_mid += 1
            else:
                pc_named += 1
    log(f"[phasec] rows={pc_n:,} text==src={pc_text_eq_id:,} id-rule ok={pc_id_rule_ok:,} "
        f"{time.time()-t0:.0f}s")

    # ---------------- cross-object overlap -----------------------------------
    legacy_ep = set(entities)
    union_ep = set(U)
    res["PHASE7_four_way"] = {
        "objects": {
            "1_legacy_781k": {"N": n_leg, "identity": "positional webqsp_doc_<i>",
                              "MID_recoverable": "YES via sorted(entities) reconstruction, "
                                                 "but NOT stored anywhere in the artifact",
                              "fabricated_names": fabricated,
                              "surface_collisions_manufactured": collide_with_real},
            "2_phaseC_1316466": {"N": pc_n, "identity": "webqsp_ent_<sha256(surface)[:16]>",
                                 "text_equals_source_id_rows": pc_text_eq_id,
                                 "id_rule_reproduces": pc_id_rule_ok,
                                 "MID_endpoints": pc_mid, "named_endpoints": pc_named,
                                 "MID_recoverable": "only for the 59.8% already stored AS MIDs",
                                 "fabricated_names": 0},
            "3_released_RoG_union": {"N": len(U),
                                     "identity": "released endpoint string (mixed MID/surface)",
                                     "MID_endpoints": int(sum(1 for c in UC if c == "FREEBASE_MID")),
                                     "named_endpoints": int(sum(1 for c in UC if c == "HUMAN_READABLE_SURFACE")),
                                     "other": int(sum(1 for c in UC if c == "OTHER")),
                                     "MID_recoverable": "NO for the 925,761 named endpoints",
                                     "fabricated_names": 0},
            "4_reconstructed_MID_graph": {"STATUS": "NOT BUILT",
                                          "reason": "no benchmark Freebase source on this machine "
                                                    "(Phase 2: 1,996,229 files scanned, 0 hits) and "
                                                    "downloading was out of scope"},
        },
        "overlap": {
            "legacy781k_in_union": len(legacy_ep & union_ep),
            "legacy781k_not_in_union": len(legacy_ep - union_ep),
            "phaseC_in_union": len(pc_ids & union_ep),
            "phaseC_not_in_union": len(pc_ids - union_ep),
            "legacy781k_in_phaseC": len(legacy_ep & pc_ids),
            "union_not_in_phaseC": len(union_ep - pc_ids),
        },
    }
    log("[overlap] " + json.dumps(res["PHASE7_four_way"]["overlap"]))

    # ---------------- PHASE 8: reuse projection ------------------------------
    # Canonical text policy under MID-level identity:
    #   named endpoint  -> Freebase display_name (+aliases/types once a dump exists).
    #                      With NO dump the best available string is the released surface,
    #                      i.e. BYTE-IDENTICAL to what Phase-C already encoded  -> REUSABLE.
    #   CVT / mediator  -> deterministic schema text                            -> NEW
    #   name-resolvable / mixed bare MID -> needs Freebase metadata             -> NEW
    #   OTHER literal   -> literal verbatim                                     -> REUSABLE
    in_pc = np.fromiter((U[i] in pc_ids for i in range(len(U))), bool, len(U))
    named = np.array([c == "HUMAN_READABLE_SURFACE" for c in UC])
    other = np.array([c == "OTHER" for c in UC])
    is_cvt = lab == 1
    needs_meta = (lab == 2) | (lab == 3)

    text_unchanged = named | other
    reuse_base = int((in_pc & text_unchanged).sum())
    lost_cvt = int((in_pc & is_cvt).sum())
    lost_meta = int((in_pc & needs_meta).sum())
    baseline_dense, baseline_splade = 1_299_535, 1_302_766

    res["PHASE8_reuse_projection"] = {
        "_rule": "token-ID equality under the frozen tokenizers. A text that changes cannot reuse; "
                 "a byte-identical text always does. The two measured tokenizer correction terms "
                 "from the prior audit (dense +8 over raw-text equality; SPLADE +3,231 from its "
                 "uncased WordPiece) are carried and flagged, not re-derived. NOTHING ENCODED.",
        "baseline_released_endpoint_text": {
            "DENSE_REUSE_N": baseline_dense, "DENSE_REUSE_PCT": 50.12,
            "SPLADE_REUSE_N": baseline_splade, "SPLADE_REUSE_PCT": 50.24,
        },
        "under_MID_level_canonical_text": {
            "endpoints_whose_text_is_unchanged": int(text_unchanged.sum()),
            "of_those_already_encoded_in_PhaseC": reuse_base,
            "reuse_lost_because_node_becomes_CVT_schema_text": lost_cvt,
            "reuse_lost_because_node_needs_Freebase_metadata": lost_meta,
            "DENSE_REUSE_N": reuse_base + 8,
            "DENSE_REENCODE_N": len(U) - (reuse_base + 8),
            "SPLADE_REUSE_N": reuse_base + 3231,
            "SPLADE_REENCODE_N": len(U) - (reuse_base + 3231),
            "_correction_note": "+8 / +3,231 are the prior audit's measured tokenizer-collision "
                                "surpluses over raw-text equality; they apply to the surface rows, "
                                "which are exactly the rows kept here.",
        },
        "delta_vs_baseline": {
            "DENSE": reuse_base + 8 - baseline_dense,
            "SPLADE": reuse_base + 3231 - baseline_splade,
        },
        "encoder_work_is_bounded_by_DISTINCT_texts_not_nodes": {
            "CVT_nodes": int(is_cvt.sum()),
            "_see": "phase45_mid_cvt_classification.json -> PHASE5_CVT_TEXT",
        },
        "_caveat": "This projection is CONDITIONAL. Real canonical text (display_name + aliases + "
                   "types) requires a Freebase metadata source that is not present. Once it lands, "
                   "the 925,761 named endpoints get aliases/types appended and their text CHANGES "
                   "too, which would drive dense reuse toward ~0. Correct representation has "
                   "priority over reuse.",
    }
    log("[phase8] " + json.dumps(res["PHASE8_reuse_projection"]["under_MID_level_canonical_text"]))

    res["_elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    log("WROTE " + OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
