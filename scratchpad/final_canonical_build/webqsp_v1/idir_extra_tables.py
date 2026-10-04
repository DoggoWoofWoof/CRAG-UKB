"""Exhaust the IDIR archive before it is deleted.

object_names.csv has already been probed and returned 2,259 nodes. Deletion of the 13.2 GB zip is
irreversible, so the two remaining tables that could plausibly carry a human-readable identity for
an unresolved node are probed first, and the yields recorded, so the decision to delete rests on
measurement rather than on an assumption about what was left inside.

  entities_id_label.csv   1,933,710,254 bytes   MID,label,numeric_id
        A DIFFERENT table from object_names: one canonical label per entity rather than a
        multilingual /type/object/name dump, so it can hold a label for an entity object_names
        missed. Commas inside the label are backslash-escaped, and the numeric id is comma-free,
        so the row splits unambiguously from both ends.

  object_ids.csv            161,903,657 bytes   MID,/type/object/id,"path"
        Internal namespace paths (/user/...). These are IDENTIFIERS, not names, so a hit here is
        recorded under its own provenance and is not eligible for a name tier -- it is measured to
        establish what the archive held, not to inflate the recovery count.

Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, json, glob, time, subprocess, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ZIP = f"{V3}/_acquisition/idir/idirlab-freebases.zip"
SEVENZ = r"C:\Program Files\7-Zip\7z.exe"
CACHE = f"{V3}/_acquisition/_unresolved_population.npz"
KIND = ["ENTITY_MID", "CVT_MEDIATOR", "OTHER"]
PROBE = 1_000_000
QUOTE = '"'
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
z = np.load(CACHE)
U, KC = z["U"], z["KC"]
NU = len(U)
print(f"unresolved population: {NU:,} "
      f"(ENTITY_MID {int((KC==0).sum()):,}, CVT {int((KC==1).sum()):,})", flush=True)


def run(entry, parse, tag):
    """Stream one zip entry, batch-test subjects against the unresolved set, keep the hits."""
    p = subprocess.Popen([SEVENZ, "x", "-so", ZIP, entry],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22)
    rows = mids = 0
    hits = {}
    b_sub, b_val = [], []

    def probe():
        nonlocal hits
        if not b_sub:
            return
        nid = [s[1:].replace("/", ".", 1) for s in b_sub]
        u = np.fromiter(map(hash, nid), dtype=np.int64, count=len(nid))
        j = np.searchsorted(U, u)
        np.clip(j, 0, NU - 1, out=j)
        for i in np.flatnonzero(U[j] == u).tolist():
            v = b_val[i]
            if v and int(u[i]) not in hits:
                hits[int(u[i])] = (nid[i], v)

    for raw in io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace", newline=""):
        rows += 1
        r = parse(raw.rstrip("\r\n"))
        if r is None:
            continue
        mids += 1
        b_sub.append(r[0]); b_val.append(r[1])
        if len(b_sub) >= PROBE:
            probe()
            b_sub, b_val = [], []
            print(f"  {tag} {rows:,} rows  mid-rows={mids:,}  hits={len(hits):,} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    probe()
    p.stdout.close()
    p.wait()
    print(f"{tag}: rows={rows:,} mid-rows={mids:,} distinct hits={len(hits):,} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rows, mids, hits


def parse_label(line):
    a, _, rest = line.partition(",")
    if not (a.startswith("/m/") or a.startswith("/g/")) or not rest:
        return None
    lab, _, _num = rest.rpartition(",")       # the numeric id is comma-free
    if not lab:
        lab = rest
    lab = lab.replace("\\,", ",").strip().strip(QUOTE).strip()
    return (a, lab) if lab else None


def parse_objid(line):
    a, _, rest = line.partition(",")
    if not (a.startswith("/m/") or a.startswith("/g/")) or not rest:
        return None
    _, _, val = rest.partition(",")
    val = val.strip().strip(QUOTE).strip()
    return (a, val) if val else None


rec = {"schema": "IDIR_EXTRA_TABLES/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. 301,977,131 nodes / "
                       "2,062,430,072 edges unchanged; OVERLAY_V1 manifest hash unchanged."),
       "WHY": ("the zip is about to be deleted, which is irreversible, so every remaining table "
               "that could carry a readable identity for an unresolved node is probed first and "
               "its yield recorded."),
       "unresolved_nodes_probed": int(NU),
       "unresolved_entity_mids": int((KC == 0).sum()),
       "tables": {}}

for entry, fn, tag, is_name in (
        ("idirlab-freebases/Metadata/entities_id_label.csv", parse_label, "entities_id_label", True),
        ("idirlab-freebases/Metadata/object_ids.csv", parse_objid, "object_ids", False)):
    rows, mids, hits = run(entry, fn, tag)
    kc = KC[np.searchsorted(U, np.array(sorted(hits), dtype=np.int64))] if hits else np.array([])
    by = {KIND[i]: int((kc == i).sum()) for i in range(3)} if hits else {}
    if hits:
        ks = sorted(hits)
        pq.write_table(pa.table({
            "node_uid": pa.array(ks, pa.int64()),
            "node_id": pa.array([hits[k][0] for k in ks]),
            "value": pa.array([hits[k][1] for k in ks]),
            "kind": pa.array([KIND[c] for c in kc.tolist()]).dictionary_encode()}),
            f"{V3}/_acquisition/idir_{tag}.parquet", compression="zstd")
    rec["tables"][tag] = {
        "entry": entry, "rows": rows, "mid_rows": mids, "distinct_hits": len(hits),
        "hits_by_kind": by,
        "ELIGIBLE_AS_A_NAME": is_name,
        "provenance_if_used": "IDIR_CANONICAL_LABEL" if is_name else
                              "IDIR_OBJECT_ID (an identifier, not a name; not name-eligible)"}
    print(f"  {tag} by kind: {by}", flush=True)

rec["elapsed_s"] = round(time.time() - t0, 1)
with io.open(f"{V3}/V3_IDIR_EXTRA_TABLES.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps(rec["tables"], indent=1))
print(f"{time.time()-t0:.0f}s")
