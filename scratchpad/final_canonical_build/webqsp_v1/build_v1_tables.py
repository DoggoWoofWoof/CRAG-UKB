"""V1 §2/§9-11 -- persist the reproduced released-RoG graph as nodes/relations/edges parquet.

    python scratchpad/final_canonical_build/webqsp_v1/build_v1_tables.py

Operates on the EXACT released RoG object reproduced by rog_union.py (§1 passed: 8,309,195
triples / 7,058 relations / 2,592,894 endpoints, all deltas 0).  Nothing from the NSM branch
enters here: WEBQSP_NSM_MID returned SIBLING_EXTRACTION and is a measurement instrument only.

Identity model (§2):
    node_uid              int64, rank in a deterministic total order over the endpoint strings
    node_key              stable string key, sha256(endpoint)[:16] -- survives set changes
    source_rog_endpoint   the released endpoint string, VERBATIM, never normalised
Edges reference node_uid.  Readable surfaces are never a key: node_key is derived from the endpoint
bytes, and display fields (§3/§6) are added later and are never used for lookup.

Resolution/provenance columns are created here and left null with resolution_status =
PENDING_SECTION_3, so §3 fills a declared schema rather than inventing one.

Speed: §1 looped in Python over 158.8M raw triples (1,112 s).  This pass dictionary-encodes each
row group and does the string->id work only over that group's UNIQUE strings, then packs triples
into int64 and dedupes with numpy.
"""
import hashlib, json, os, time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = {"webqsp": "data/original/webqsp/rog_webqsp", "cwq": "data/original/cwq/rog_cwq"}
OUTD = "data/final_canonical/webqsp/v1"
REPORT = "data/final_canonical/webqsp/V1_TABLES_REPORT.json"

EXP_TRIPLES, EXP_RELATIONS, EXP_ENDPOINTS = 8309195, 7058, 2592894
EP_BITS, REL_BITS = 22, 14
EP_MASK, REL_MASK = (1 << EP_BITS) - 1, (1 << REL_BITS) - 1
CONSOLIDATE_EVERY = 10


def relation_label(key):
    """Section 7: readable label; the exact key is preserved separately.

    people.person.place_of_birth -> "place of birth"
    """
    seg = key.rsplit(".", 1)[-1] if "." in key else key
    return seg.replace("_", " ").strip()


def main():
    t0 = time.time()
    os.makedirs(OUTD, exist_ok=True)
    ep, rel = {}, {}                 # string -> global id
    ep_list, rel_list = [], []
    parts = []

    def consolidate():
        if len(parts) > 1:
            u = np.unique(np.concatenate(parts))
            parts.clear()
            parts.append(u)

    raw = 0
    for ds, d in SRC.items():
        for sh in sorted(f for f in os.listdir(d) if f.endswith(".parquet")):
            pf = pq.ParquetFile(f"{d}/{sh}")
            for rg in range(pf.metadata.num_row_groups):
                col = pf.read_row_group(rg, columns=["graph"]).column("graph")
                for chunk in col.chunks:
                    flat = chunk.flatten().flatten()          # list<list<str>> -> flat string values
                    if len(flat) == 0:
                        continue
                    enc = pc.dictionary_encode(flat)
                    idx = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
                    dic = enc.dictionary.to_pylist()
                    h_i, r_i, t_i = idx[0::3], idx[1::3], idx[2::3]
                    raw += len(h_i)
                    # map local dictionary -> global ids, touching only strings actually used per role
                    lut_e = np.full(len(dic), -1, dtype=np.int64)
                    for li in np.unique(np.concatenate([h_i, t_i])):
                        s = dic[li]
                        g = ep.get(s)
                        if g is None:
                            g = ep[s] = len(ep_list)
                            ep_list.append(s)
                        lut_e[li] = g
                    lut_r = np.full(len(dic), -1, dtype=np.int64)
                    for li in np.unique(r_i):
                        s = dic[li]
                        g = rel.get(s)
                        if g is None:
                            g = rel[s] = len(rel_list)
                            rel_list.append(s)
                        lut_r[li] = g
                    if len(ep_list) > EP_MASK or len(rel_list) > REL_MASK:
                        raise SystemExit(f"bit budget exceeded: eps={len(ep_list)} rels={len(rel_list)}")
                    codes = (lut_e[h_i] << (REL_BITS + EP_BITS)) | (lut_r[r_i] << EP_BITS) | lut_e[t_i]
                    parts.append(np.unique(codes))
                    if len(parts) >= CONSOLIDATE_EVERY:
                        consolidate()
            print(f"[{ds}] {sh} eps={len(ep_list):,} rels={len(rel_list):,} raw={raw:,} "
                  f"t={time.time() - t0:.0f}s", flush=True)

    codes = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
    parts.clear()
    n_tri = int(codes.size)
    print(f"[union] triples={n_tri:,} eps={len(ep_list):,} rels={len(rel_list):,} "
          f"t={time.time() - t0:.0f}s", flush=True)

    # ---- deterministic total order: UTF-8 byte order over the verbatim endpoint strings ----
    order = np.asarray(sorted(range(len(ep_list)), key=lambda i: ep_list[i].encode("utf-8")),
                       dtype=np.int64)
    gid_to_uid = np.empty(len(ep_list), dtype=np.int64)
    gid_to_uid[order] = np.arange(len(ep_list), dtype=np.int64)
    endpoints_sorted = [ep_list[i] for i in order]

    rorder = np.asarray(sorted(range(len(rel_list)), key=lambda i: rel_list[i].encode("utf-8")),
                        dtype=np.int64)
    rgid_to_uid = np.empty(len(rel_list), dtype=np.int64)
    rgid_to_uid[rorder] = np.arange(len(rel_list), dtype=np.int64)
    relations_sorted = [rel_list[i] for i in rorder]

    src = gid_to_uid[(codes >> (REL_BITS + EP_BITS)) & EP_MASK]
    rid = rgid_to_uid[(codes >> EP_BITS) & REL_MASK]
    dst = gid_to_uid[codes & EP_MASK]
    del codes

    # edges sorted by (src, relation, dst) so the TABLE is deterministic, not merely its content
    eorder = np.lexsort((dst, rid, src))
    src, rid, dst = src[eorder], rid[eorder], dst[eorder]
    del eorder

    # ---- stable keys + order/content hashes ----
    keys = []
    h_order, h_pairs = hashlib.sha256(), hashlib.sha256()
    for s in endpoints_sorted:
        k = hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
        keys.append(k)
        h_order.update(k.encode())
        h_order.update(b"\n")
        h_pairs.update(k.encode())
        h_pairs.update(b"\t")
        h_pairs.update(s.encode("utf-8"))
        h_pairs.update(b"\n")
    node_order_hash, node_content_hash = h_order.hexdigest(), h_pairs.hexdigest()
    key_collisions = len(keys) - len(set(keys))

    n = len(endpoints_sorted)
    nulls = pa.nulls(n, pa.string())
    nodes = pa.table({
        "node_uid": pa.array(np.arange(n, dtype=np.int64)),
        "node_key": pa.array(keys, pa.string()),
        "source_rog_endpoint": pa.array(endpoints_sorted, pa.string()),
        "node_kind": nulls,
        "canonical_name": nulls,
        "display_name": nulls,
        "display_name_disambiguated": nulls,
        "display_source": nulls,
        "source_mid": nulls,
        "retrieval_role": nulls,
        "resolution_status": pa.array(["PENDING_SECTION_3"] * n, pa.string()),
    })
    rels_tbl = pa.table({
        "relation_uid": pa.array(np.arange(len(relations_sorted), dtype=np.int32)),
        "relation_key": pa.array(relations_sorted, pa.string()),
        "relation_label": pa.array([relation_label(r) for r in relations_sorted], pa.string()),
    })
    edges = pa.table({
        "src_uid": pa.array(src),
        "relation_uid": pa.array(rid.astype(np.int32)),
        "dst_uid": pa.array(dst),
    })

    for tbl, name in ((nodes, "nodes"), (rels_tbl, "relations"), (edges, "edges")):
        tmp = f"{OUTD}/{name}.parquet.tmp"
        pq.write_table(tbl, tmp, compression="zstd")
        os.replace(tmp, f"{OUTD}/{name}.parquet")

    # ---- verification (§11 dangling, §15 topology) ----
    dangling = int(((src < 0) | (src >= n) | (dst < 0) | (dst >= n)).sum())
    rel_dangling = int(((rid < 0) | (rid >= len(relations_sorted))).sum())
    self_loops = int((src == dst).sum())
    labels = rels_tbl.column("relation_label").to_pylist()
    lbl_dupes = len(labels) - len(set(labels))

    doc = {
        "schema": "V1_TABLES_REPORT/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "released RoG WebQSP union CWQ, 29 official parquet shards",
        "section": "V1 §2 identity + §9-11 tables",
        "nsm_branch_excluded": "WEBQSP_NSM_MID returned SIBLING_EXTRACTION; nothing from it enters V1",
        "counts": {"nodes": n, "relations": len(relations_sorted), "edges": int(src.size),
                   "raw_triple_occurrences": raw},
        "expected": {"nodes": EXP_ENDPOINTS, "relations": EXP_RELATIONS, "edges": EXP_TRIPLES},
        "TOPOLOGY_MATCHES_SECTION_1": bool(n == EXP_ENDPOINTS
                                           and len(relations_sorted) == EXP_RELATIONS
                                           and int(src.size) == EXP_TRIPLES),
        "DANGLING_ENDPOINTS": dangling,
        "DANGLING_RELATIONS": rel_dangling,
        "NODE_KEY_COLLISIONS": key_collisions,
        "SELF_LOOPS": self_loops,
        "RELATION_LABEL_NON_UNIQUE": lbl_dupes,
        "relation_label_note": "relation_label is a READABLE surface and is deliberately NOT unique; "
                               "relation_key stays the identity. Non-unique labels are expected.",
        "NODE_ORDER_HASH": node_order_hash,
        "NODE_CONTENT_HASH": node_content_hash,
        "node_order_rule": "ascending UTF-8 byte order over source_rog_endpoint, verbatim "
                           "(no strip, no case folding, no NFC)",
        "node_key_rule": "sha256(source_rog_endpoint utf-8)[:16]",
        "edge_order_rule": "lexsort by (src_uid, relation_uid, dst_uid)",
        "files": {k: {"bytes": os.path.getsize(f"{OUTD}/{k}.parquet")}
                  for k in ("nodes", "relations", "edges")},
        "resolution_fields_pending": ["node_kind", "canonical_name", "display_name",
                                      "display_name_disambiguated", "display_source", "source_mid",
                                      "retrieval_role"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({k: doc[k] for k in ("counts", "expected", "TOPOLOGY_MATCHES_SECTION_1",
                                          "DANGLING_ENDPOINTS", "DANGLING_RELATIONS",
                                          "NODE_KEY_COLLISIONS", "SELF_LOOPS", "NODE_ORDER_HASH",
                                          "files", "elapsed_s")}, indent=1))


if __name__ == "__main__":
    main()
