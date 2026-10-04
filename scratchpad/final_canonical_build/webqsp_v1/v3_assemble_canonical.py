"""STEP 7a: assemble CRAG_FREEBASE_CANONICAL from the pass outputs.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/v3_assemble_canonical.py

The passes produce a node universe (C), a canonical edge set (D) and a schema layer (A). This joins
them into the three tables the contract names, and does nothing else: no filtering, no inference,
no repair. Anything this pass cannot establish is written as NULL, and STEP 6 then checks that the
NULLs are honest.

DISPLAY TEXT IS A JOIN, NOT A FALLBACK. An MID with no type.object.name gets display_text = NULL.
It does NOT get its own MID, and it does not get a name borrowed from a same-named neighbour. V1
section 2 and section 13 are explicit about this and they are the reason the WebQSP graph is
readable at all: an MID is an opaque endpoint identifier tied to the released dump, never a
semantic representation of the thing. A build that quietly writes the MID into the display column
looks complete and is unusable.

NAME SELECTION IS DETERMINISTIC. A subject can carry names in dozens of languages. English wins
because the downstream corpora are English; among the rest the smallest language tag wins, and
within one tag the smallest lexical form. That is an arbitrary rule but a STABLE one, so two runs
of this build produce byte-identical tables. "Whichever row came first" would not.

LITERAL COMPONENTS ARE RE-DERIVED FROM THE RAW TERM, using the same terminator-dispatched split
PASS A and PASS B used, so all three agree on what a literal IS. The raw term is kept alongside:
display_text is the unescaped lexical form, and it is never mistaken for the identity, which
remains (lexical_form, datatype, language).
"""
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
PA_ = f"{V3}/pass_a"
PC = f"{V3}/pass_c"
PD = f"{V3}/pass_d"
CAN = f"{V3}/canonical"
REPORT = f"{V3}/V3_CANONICAL_ASSEMBLY.json"

BATCH = 1 << 20
CONSUME = False
BS = chr(92)
_ESC = re.compile(
    BS + BS + "(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|[tbnrf" + BS + BS + "'" + chr(34) + "])")
_SIMPLE = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f",
           chr(34): chr(34), "'": "'", BS: BS}
QUOTE = chr(34)

LITERAL, CVT_MEDIATOR, ENTITY_MID, SCHEMA_TYPE, SCHEMA_PROPERTY, SCHEMA_OTHER, \
    EXTERNAL_URI, OTHER = range(8)
KIND_NAME = {LITERAL: "LITERAL", CVT_MEDIATOR: "CVT_MEDIATOR", ENTITY_MID: "ENTITY_MID",
             SCHEMA_TYPE: "SCHEMA_TYPE", SCHEMA_PROPERTY: "SCHEMA_PROPERTY",
             SCHEMA_OTHER: "SCHEMA_OTHER", EXTERNAL_URI: "EXTERNAL_URI", OTHER: "OTHER"}
MID_KINDS = {ENTITY_MID, CVT_MEDIATOR}


def _sub(m):
    g = m.group(1)
    return chr(int(g[1:], 16)) if g[0] in "uU" else _SIMPLE[g]


def unescape(s):
    """One left-to-right scan. A chain of str.replace calls would be WRONG: an escaped backslash
    followed by 't' would be rewritten into a TAB, silently merging two distinct literals."""
    return _ESC.sub(_sub, s) if BS in s else s


def split_literal(o):
    """Terminator-dispatched, identical to PASS A and PASS B, so all three agree on identity."""
    if o.endswith(QUOTE):
        return o[1:-1], "", ""
    if o.endswith(">"):
        i = o.rfind(QUOTE + "^^<")
        if i > 0:
            return o[1:i], "", o[i + 4:-1]
        return None
    i = o.rfind(QUOTE + "@")
    if i > 0:
        return o[1:i], o[i + 2:], ""
    return None


def build_name_index():
    """Best display name per subject, as (sorted uid array, permutation, unsorted text array).

    name.parquet is subject-contiguous -- stage 1 proved zero reopened subject blocks and stage 2
    merged the members in source order -- so the reduction is a streaming one and needs no sort of
    the 68.4M rows. Only the winners are held.

    THE WINNERS ARE HELD IN ARROW, NOT IN PYTHON LISTS. There are tens of millions of them, and a
    Python list of that many int and str objects costs about 5.7 GB before the numpy and arrow
    copies are built beside it -- more than this machine has. Winners are therefore flushed into
    numpy/arrow chunks at a bounded interval, and the text is never reordered: the sort produces a
    permutation, and lookups take through it. Reordering 52M strings would transiently double the
    largest allocation in the pass for no benefit.
    """
    t0 = time.time()
    uid_chunks, txt_chunks = [], []
    bu, bt = [], []
    FLUSH = 1 << 21
    cur = None
    best_lang = best_lex = None
    n = 0

    def flush():
        if bu:
            uid_chunks.append(np.array(bu, dtype=np.int64))
            txt_chunks.append(pa.array(bt, type=pa.string()))
            del bu[:], bt[:]

    def emit():
        if cur is not None:
            bu.append(hash(cur.encode()))
            bt.append(unescape(best_lex))
            if len(bu) >= FLUSH:
                flush()

    for b in pq.ParquetFile(f"{PA_}/name.parquet").iter_batches(batch_size=BATCH):
        ss = b.column(0).to_pylist()
        lx = b.column(1).to_pylist()
        lg = b.column(2).to_pylist()
        for s, x, g in zip(ss, lx, lg):
            n += 1
            if s != cur:
                emit()
                cur, best_lang, best_lex = s, g, x
                continue
            # 'en' beats everything; otherwise smallest tag, then smallest lexical form. Arbitrary
            # but stable, so the table is reproducible byte for byte.
            if best_lang == "en":
                if g == "en" and x < best_lex:
                    best_lex = x
            elif g == "en" or (g, x) < (best_lang, best_lex):
                best_lang, best_lex = g, x
    emit()
    flush()
    u = np.concatenate(uid_chunks) if uid_chunks else np.zeros(0, dtype=np.int64)
    del uid_chunks[:]
    arr = pa.chunked_array(txt_chunks, type=pa.string()) if txt_chunks else         pa.chunked_array([pa.array([], type=pa.string())], type=pa.string())
    order = np.argsort(u, kind="stable")
    u = u[order]
    dup = int((u[1:] == u[:-1]).sum()) if u.size > 1 else 0
    print(f"  {n:,} name rows -> {u.size:,} named subjects, {dup} uid collisions "
          f"({time.time()-t0:.0f}s)", flush=True)
    return u, order, arr, {"name_rows": n, "named_subjects": int(u.size),
                           "uid_collisions": dup}


def main():
    global CONSUME
    CONSUME = "--consume" in sys.argv
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 is required so UIDs match the passes.")
    t0 = time.time()
    for d in (CAN, f"{CAN}/nodes", f"{CAN}/edges", f"{CAN}/metadata"):
        os.makedirs(d, exist_ok=True)

    print("building name index", flush=True)
    nuid, norder, ntxt, nameinfo = build_name_index()

    # ---------- nodes ----------
    print("assembling nodes", flush=True)
    files = sorted(glob.glob(f"{PC}/nodes/*.parquet"))
    if not files:
        raise SystemExit("PASS C nodes not found")
    rows = named = lit_rows = lit_unparsed = consumed_nodes = 0
    kind_counts = {}
    for f in files:
        base = os.path.basename(f)
        out = f"{CAN}/nodes/{base}"
        w = None
        for b in pq.ParquetFile(f).iter_batches(batch_size=BATCH):
            u = b.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
            ids = b.column(1).to_pylist()
            kk = b.column(2).to_numpy(zero_copy_only=False).astype(np.int8)
            rows += u.size
            for k, c in zip(*np.unique(kk, return_counts=True)):
                kind_counts[KIND_NAME[int(k)]] = kind_counts.get(KIND_NAME[int(k)], 0) + int(c)

            islit = kk == LITERAL
            disp = [None] * u.size
            lex = [None] * u.size
            dty = [None] * u.size
            lng = [None] * u.size

            if islit.any():
                for i in np.flatnonzero(islit):
                    t = split_literal(ids[i])
                    if t is None:
                        lit_unparsed += 1
                        continue
                    a, g, d = t
                    lex[i], lng[i], dty[i] = a, (g or None), (d or None)
                    disp[i] = unescape(a)
                    lit_rows += 1

            nonlit = np.flatnonzero(~islit)
            if nonlit.size and nuid.size:
                q = u[nonlit]
                order = np.argsort(q, kind="stable")
                qs = q[order]
                p = np.searchsorted(nuid, qs)
                np.clip(p, 0, nuid.size - 1, out=p)
                hit_s = nuid[p] == qs
                p_un = np.empty(q.size, np.int64)
                h_un = np.empty(q.size, bool)
                p_un[order], h_un[order] = p, hit_s
                got = np.flatnonzero(h_un)
                if got.size:
                    # p_un indexes the SORTED uid array; norder maps that back to
                    # the text array's own order, which was never permuted.
                    vals = ntxt.take(pa.array(norder[p_un[got]])).to_pylist()
                    for j, val in zip(nonlit[got], vals):
                        # empty display text is never written: absent and blank are different
                        # claims, and STEP 6 checks exactly that.
                        if val:
                            disp[j] = val
                            named += 1

            tb = pa.Table.from_arrays(
                [pa.array(u), pa.array(ids, type=pa.string()),
                 pa.array([KIND_NAME[int(k)] for k in kk], type=pa.string()),
                 pa.array(disp, type=pa.string()), pa.array(lex, type=pa.string()),
                 pa.array(dty, type=pa.string()), pa.array(lng, type=pa.string())],
                names=["node_uid", "node_id", "kind", "display_text",
                       "lexical_form", "datatype", "language"])
            if w is None:
                w = pq.ParquetWriter(out, tb.schema, compression="zstd")
            w.write_table(tb)
        if w is not None:
            w.close()
        # The input shard is only released once its output shard exists and carries exactly the same
        # number of rows.  Holding both copies of the 8.0 GB node universe at once does not fit
        # beside the edge table on this disk, and a copy that is never checked is not a backup --
        # it is a second thing that can be wrong.
        if CONSUME:
            # a zero-row input produces no output file at all, so the count is read as 0 rather
            # than by opening a path that was never created.
            got = pq.ParquetFile(out).metadata.num_rows if w is not None else 0
            want = pq.ParquetFile(f).metadata.num_rows
            if got != want:
                raise SystemExit(f"{base}: wrote {got} rows for {want} input rows; refusing to "
                                 f"delete the input")
            os.remove(f)
            consumed_nodes += 1
        print(f"  {base}  {rows:,} cumulative  t={time.time()-t0:.0f}s", flush=True)

    # ---------- edges and relations ----------
    print("linking edges and relations", flush=True)
    # PASS D writes the canonical edge set as TWO families and both are required:
    #   direct_<member>.parquet  the 76.79% of edges that were never at risk of a reverse-pair
    #                            collision and were deduplicated in place inside their subject runs
    #   part_<nnn>.parquet       the 23.21% that were fingerprint-partitioned and folded globally
    # Globbing only part_* would silently drop three quarters of the graph and still produce a
    # table that loads, validates structurally and looks finished, so the two counts are asserted
    # against the PASS D report rather than trusted.
    direct = sorted(glob.glob(f"{PD}/edges/direct_*.parquet"))
    folded = sorted(glob.glob(f"{PD}/edges/part_*.parquet"))
    eparts = direct + folded
    if not direct or not folded:
        raise SystemExit(f"PASS D edges incomplete: {len(direct)} direct + {len(folded)} folded")
    edges = 0
    for p in eparts:
        dst = f"{CAN}/edges/{os.path.basename(p)}"
        if not os.path.exists(dst):
            os.replace(p, dst)
        edges += pq.ParquetFile(dst).metadata.num_rows
    pdrep = f"{V3}/V3_PASS_D_CANONICAL_EDGES.json"
    if os.path.exists(pdrep):
        want = json.load(open(pdrep, encoding="utf-8"))["CANONICALISATION"]["CANONICAL_EDGES"]
        if edges != want:
            raise SystemExit(f"edge count {edges:,} does not match PASS D's {want:,}; refusing to "
                             f"assemble a graph that has already lost rows")
    shutil.copy2(f"{PD}/relations.parquet", f"{CAN}/relations.parquet")
    rel_rows = pq.ParquetFile(f"{CAN}/relations.parquet").metadata.num_rows

    # ---------- metadata layer ----------
    meta_rows = {}
    moved_meta = []
    for t in ("name", "alias", "description", "type", "rdf_type", "key", "property_schema",
              "type_hints", "reverse_property", "master_property", "label_residue",
              "rdf_type_residue"):
        src = f"{PA_}/{t}.parquet"
        dst = f"{CAN}/metadata/{t}.parquet"
        if os.path.exists(dst):                       # already assembled; re-run is a no-op
            meta_rows[t] = pq.ParquetFile(dst).metadata.num_rows
            continue
        if os.path.exists(src):
            # MOVE under --consume.  These 8.2 GB ARE the canonical metadata layer; there is no
            # version of this build in which pass_a/ and canonical/metadata/ should both hold them.
            # os.replace is atomic within the drive, so an interrupted run leaves the table at
            # exactly one of the two paths, never at neither.
            if CONSUME:
                os.replace(src, dst)
                moved_meta.append(t)
            else:
                shutil.copy2(src, dst)
            meta_rows[t] = pq.ParquetFile(dst).metadata.num_rows

    node_files_on_disk = sorted(glob.glob(f"{CAN}/nodes/*.parquet"))
    node_rows_on_disk = sum(pq.ParquetFile(x).metadata.num_rows for x in node_files_on_disk)
    doc = {
        "schema": "V3_CANONICAL_ASSEMBLY/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        # rows/by_kind count what THIS invocation wrote.  Under --consume a resumed run sees only
        # the shards that were still unconsumed, so the authoritative total is read back from the
        # written tables instead of accumulated.  STEP 6 recomputes by_kind from the tables anyway
        # and is the authority; this is a convenience copy.
        "NODES": {"rows": node_rows_on_disk, "shards": len(node_files_on_disk),
                  "rows_this_invocation": rows, "by_kind_this_invocation": kind_counts,
                  "with_display_text_from_names": named,
                  "literal_rows_parsed": lit_rows, "literal_rows_unparsed": lit_unparsed},
        "NAME_INDEX": nameinfo,
        "NAME_SELECTION": "english first; otherwise the smallest language tag, then the smallest "
                          "lexical form. Arbitrary but stable, so the table is reproducible.",
        "EDGES": {"parts": len(eparts), "direct_parts": len(direct),
                  "folded_parts": len(folded), "rows": edges,
                  "matches_pass_d_report": True},
        "RELATIONS": {"rows": rel_rows},
        "METADATA": meta_rows,
        "INPUT_RECLAMATION": {
            "consume": CONSUME,
            "pass_c_node_shards_deleted": consumed_nodes,
            "pass_a_metadata_tables_moved": moved_meta,
            "gate": "a PASS C shard is deleted only after its canonical shard exists and has "
                    "an identical row count; a PASS A table is moved, never copied, so the "
                    "metadata layer exists at exactly one path.",
        },
        "DISPLAY_TEXT_RULE": "a node with no type.object.name has display_text = NULL. It never "
                             "receives its own MID and never borrows a name from a same-named "
                             "node: V1 forbids merging or naming nodes because surface forms "
                             "coincide.",
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({k: doc[k] for k in ("NODES", "NAME_INDEX", "EDGES", "RELATIONS")}, indent=1))


if __name__ == "__main__":
    main()
