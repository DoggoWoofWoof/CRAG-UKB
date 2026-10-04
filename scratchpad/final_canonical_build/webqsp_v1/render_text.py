"""V1 section 3 step 5 -- render display names and the three text views.

    python scratchpad/final_canonical_build/webqsp_v1/render_text.py

Rules frozen in data/final_canonical/webqsp/V1_CVT_GATE_PREREGISTRATION.json.  node_kind is frozen
(hash 5ea22695...) and is READ ONLY here.

Produces:
  nodes.parquet          canonical_name / display_name / display_name_disambiguated / display_source
  entity_text.parquet    text_name_only + text_name_plus_facts, for every non-CVT node
  cvt_text.parquet       structural record text, for all 1,588,085 CVT mediators

Display surfaces, by node_kind:
  READABLE_ENTITY    the RoG surface verbatim               display_source = ROG_SURFACE
  VALUE_LITERAL      the literal verbatim                   display_source = LITERAL_VERBATIM
  CVT_MEDIATOR       structural label from asserted types   display_source = STRUCTURAL_SCHEMA_LABEL
  MID_NAMED_ENTITY   structural descriptor from incident schema, NEVER the MID and never a guessed
  UNRESOLVED_OTHER   name                                   display_source = STRUCTURAL_SCHEMA_LABEL

A bare MID is never emitted as a display surface (directive section 13).  Where a node has no
recoverable name, the honest output is a schema-derived descriptor, and a gold answer that is that
node's true name will correctly match nothing.
"""
import hashlib, json, os, time
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

D = "data/final_canonical/webqsp/v1"
PREREG = "data/final_canonical/webqsp/V1_CVT_GATE_PREREGISTRATION.json"
FROZEN = "data/final_canonical/webqsp/V1_NODE_KIND_FROZEN.json"
OUT = "data/final_canonical/webqsp/V1_RENDER_REPORT.json"

CVT = "CVT_MEDIATOR"


def natural(s):
    return " ".join(s.replace("_", " ").replace("#", " ").split())


def main():
    t0 = time.time()
    json.load(open(PREREG, encoding="utf-8"))          # must exist before rendering
    frz = json.load(open(FROZEN, encoding="utf-8"))

    nodes = pq.read_table(f"{D}/nodes.parquet")
    eps = nodes.column("source_rog_endpoint").to_pylist()
    kind = nodes.column("node_kind").to_pylist()
    N = len(eps)

    h = hashlib.sha256()
    for k in kind:
        h.update(k.encode())
        h.update(b"\n")
    assert h.hexdigest() == frz["NODE_KIND_HASH"], "node_kind changed since it was frozen"

    rels = pq.read_table(f"{D}/relations.parquet")
    rkeys = rels.column("relation_key").to_pylist()
    rqual = rels.column("relation_label_qualified").to_pylist()
    rtype = [k.rsplit(".", 1)[0] if "." in k else k for k in rkeys]

    e = pq.read_table(f"{D}/edges.parquet")
    src = e.column("src_uid").to_numpy()
    rid = e.column("relation_uid").to_numpy()
    dst = e.column("dst_uid").to_numpy()
    del e
    E = src.size
    print(f"[load] N={N:,} E={E:,} t={time.time()-t0:.0f}s", flush=True)

    is_cvt = np.array([k == CVT for k in kind])

    # ---------- asserted types per node, for structural labels ----------
    # outgoing type is the primary signal; nodes with no outgoing edge fall back to the relations
    # pointing AT them, which for g. leaves is the only signal that exists at all.
    out_types = defaultdict(set)
    for s, r in zip(src.tolist(), rid.tolist()):
        out_types[s].add(rtype[r])
    in_rels = defaultdict(set)
    for d, r in zip(dst.tolist(), rid.tolist()):
        in_rels[d].add(r)
    print(f"[types] t={time.time()-t0:.0f}s", flush=True)

    def type_display(ty):
        """Freebase type -> readable name.

        Display surfaces are not keys, so the domain prefix is dropped: sports.sports_team_roster
        renders 'sports team roster', not 'sports.sports team roster'. Two types may collapse to the
        same display string; that is fine here and is what display_name_disambiguated exists for.
        """
        return natural(ty.rsplit(".", 1)[-1] if "." in ty else ty)

    def struct_label(i, is_cvt_node):
        ts = out_types.get(i)
        if not ts:
            rs = in_rels.get(i)
            ts = {rtype[r] for r in rs} if rs else None
        if not ts:
            return "unlabelled record" if is_cvt_node else "unnamed entity"
        body = ", ".join(sorted({type_display(x) for x in ts}))
        return f"{body} record" if is_cvt_node else f"unnamed {body} entity"

    canonical, display, dsource = [None] * N, [None] * N, [None] * N
    for i in range(N):
        k = kind[i]
        if k == "READABLE_ENTITY":
            canonical[i] = display[i] = eps[i]
            dsource[i] = "ROG_SURFACE"
        elif k == "VALUE_LITERAL":
            display[i] = eps[i]
            dsource[i] = "LITERAL_VERBATIM"
        else:
            display[i] = struct_label(i, k == CVT)
            dsource[i] = "STRUCTURAL_SCHEMA_LABEL"
    print(f"[display] t={time.time()-t0:.0f}s", flush=True)

    # ---------- disambiguated display: only where a surface is shared ----------
    cnt = Counter(display)
    seen = Counter()
    disamb = [None] * N
    for i in range(N):
        d = display[i]
        if cnt[d] == 1:
            disamb[i] = d
        else:
            seen[d] += 1
            disamb[i] = f"{d} #{seen[d]}"
    shared_surfaces = sum(1 for v in cnt.values() if v > 1)
    nodes_sharing = sum(v for v in cnt.values() if v > 1)
    display_empty = sum(1 for d in display if not d or not d.strip())
    mid_visible = sum(1 for i in range(N)
                      if display[i] == eps[i] and kind[i] not in ("READABLE_ENTITY", "VALUE_LITERAL"))

    # ---------- CVT record text ----------
    # group every incident edge by the CVT, then emit role-labelled arguments in a deterministic
    # order.  Both directions are included: a CVT is a record whose arguments point outward, but
    # some schemas attach the record to its subject via an incoming edge.
    cvt_args = defaultdict(list)
    for s, r, d in zip(src.tolist(), rid.tolist(), dst.tolist()):
        if is_cvt[s]:
            cvt_args[s].append((rqual[r], d))
        if is_cvt[d] and not is_cvt[s]:
            cvt_args[d].append((rqual[r], s))
    print(f"[cvt args] t={time.time()-t0:.0f}s", flush=True)

    cvt_uids = np.flatnonzero(is_cvt)
    cvt_text = []
    for i in cvt_uids.tolist():
        args = sorted({(lbl, display[o]) for lbl, o in cvt_args.get(i, ())})
        body = " ".join(f"{lbl}: {val}." for lbl, val in args)
        cvt_text.append(f"{display[i]}. {body}".strip())
    pq.write_table(pa.table({
        "node_uid": pa.array(cvt_uids.astype(np.int64)),
        "display_name": pa.array([display[i] for i in cvt_uids.tolist()], pa.string()),
        "text_cvt_record": pa.array(cvt_text, pa.string()),
    }), f"{D}/cvt_text.parquet.tmp", compression="zstd")
    os.replace(f"{D}/cvt_text.parquet.tmp", f"{D}/cvt_text.parquet")
    print(f"[cvt text] {len(cvt_text):,} t={time.time()-t0:.0f}s", flush=True)

    # ---------- entity text: NAME_ONLY and NAME_PLUS_FACTS ----------
    # NAME_PLUS_FACTS = own surface + direct non-CVT facts + the arguments of every adjacent CVT
    # (either direction) with the entity itself removed.  That flattening is the whole point: it is
    # the strongest form of the substitution hypothesis the secondary cell is testing.
    ent_facts = defaultdict(list)
    for s, r, d in zip(src.tolist(), rid.tolist(), dst.tolist()):
        if not is_cvt[s] and not is_cvt[d]:
            ent_facts[s].append((rqual[r], d))
    for i in cvt_uids.tolist():
        args = cvt_args.get(i, ())
        for _, o in args:
            if is_cvt[o]:
                continue
            for lbl2, o2 in args:
                if o2 != o and not is_cvt[o2]:
                    ent_facts[o].append((lbl2, o2))
    print(f"[entity facts] t={time.time()-t0:.0f}s", flush=True)

    ent_uids = np.flatnonzero(~is_cvt)
    name_only, name_facts = [], []
    for i in ent_uids.tolist():
        name_only.append(display[i])
        args = sorted({(lbl, display[o]) for lbl, o in ent_facts.get(i, ())})
        body = " ".join(f"{lbl}: {val}." for lbl, val in args)
        name_facts.append(f"{display[i]}. {body}".strip() if body else display[i])
    pq.write_table(pa.table({
        "node_uid": pa.array(ent_uids.astype(np.int64)),
        "node_kind": pa.array([kind[i] for i in ent_uids.tolist()], pa.string()),
        "text_name_only": pa.array(name_only, pa.string()),
        "text_name_plus_facts": pa.array(name_facts, pa.string()),
    }), f"{D}/entity_text.parquet.tmp", compression="zstd")
    os.replace(f"{D}/entity_text.parquet.tmp", f"{D}/entity_text.parquet")
    print(f"[entity text] {len(name_only):,} t={time.time()-t0:.0f}s", flush=True)

    # ---------- persist display fields ----------
    tbl = nodes.drop_columns(["canonical_name", "display_name", "display_name_disambiguated",
                              "display_source"])
    tbl = tbl.append_column("canonical_name", pa.array(canonical, pa.string()))
    tbl = tbl.append_column("display_name", pa.array(display, pa.string()))
    tbl = tbl.append_column("display_name_disambiguated", pa.array(disamb, pa.string()))
    tbl = tbl.append_column("display_source", pa.array(dsource, pa.string()))
    tbl = tbl.set_column(tbl.schema.get_field_index("resolution_status"), "resolution_status",
                         pa.array(["RENDERED"] * N, pa.string()))
    pq.write_table(tbl, f"{D}/nodes.parquet.tmp", compression="zstd")
    os.replace(f"{D}/nodes.parquet.tmp", f"{D}/nodes.parquet")

    d_cvt = Counter(cvt_text)
    d_name = Counter(name_only)
    doc = {
        "schema": "V1_RENDER_REPORT/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preregistration": PREREG,
        "node_kind_hash_verified": True,
        "DISPLAY_NAME_EMPTY_N": display_empty,
        "MID_VISIBLE_AS_DISPLAY_N": mid_visible,
        "display_source_counts": dict(Counter(dsource).most_common()),
        "shared_display_surfaces": shared_surfaces,
        "nodes_sharing_a_display_surface": nodes_sharing,
        "disambiguation_note": "display_name_disambiguated appends a deterministic ordinal ONLY "
                               "where a surface is shared. It is a display aid, never a key, and no "
                               "node was merged (directive section 8).",
        "cvt_text": {"rows": len(cvt_text), "distinct_texts": len(d_cvt),
                     "compression_ratio": round(len(cvt_text) / max(len(d_cvt), 1), 2),
                     "samples": [t for t, _ in d_cvt.most_common(5)]},
        "entity_text": {"rows": len(name_only), "distinct_name_only": len(d_name)},
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(doc, indent=1)[:2500])


if __name__ == "__main__":
    main()
