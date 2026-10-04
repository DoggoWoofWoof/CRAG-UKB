"""Display names for the 52,272,634 CVT mediators -- cheap, no edge pass.

A mediator's readable identity IS its type: /m/0abcxyz is "Film performance". PASS C already
decided CVT-ness by testing type.object.type against the 1,649 MEDIATOR_TYPE_PATHS, so recovering
which mediator type each CVT has is a 1,649-value filter over type.parquet, not a join against a
52M-row subject set. The type path is then given a human name from the frozen name table, which
carries names for schema paths as well as entities.
"""
import json, time
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"
frz = json.load(open(f"{V3}/V3_PASS_A_SCHEMA_FREEZE.json", encoding="utf-8"))
MED = frz["DECLARATIONS"]["MEDIATOR_TYPE_PATHS"]
medset = pa.array(sorted(MED))
print(f"mediator type paths: {len(MED):,}", flush=True)

# --- names for the type paths themselves ---------------------------------------------------------
t0 = time.time()
pf = pq.ParquetFile(f"{V3}/canonical/metadata/name.parquet")
pathname = {}
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=["subject", "lexical", "lang"])
    m = pc.is_in(t.column("subject"), value_set=medset)
    if pc.sum(m).as_py():
        f = t.filter(m)
        for s, lx, lg in zip(f.column("subject").to_pylist(),
                             f.column("lexical").to_pylist(),
                             f.column("lang").to_pylist()):
            if s not in pathname or (lg == "en" and pathname[s][1] != "en"):
                pathname[s] = (lx, lg)
print(f"mediator type paths WITH a name: {len(pathname):,} / {len(MED):,} ({time.time()-t0:.0f}s)",
      flush=True)

def prettify(path):
    """fallback when the type path has no name row: film.performance -> 'Film performance'."""
    last = path.split(".")[-1].replace("_", " ").strip()
    return last[:1].upper() + last[1:] if last else path

label = {p: (pathname[p][0] if p in pathname else prettify(p)) for p in MED}
from_name = sum(1 for p in MED if p in pathname)

# --- which mediator type does each CVT carry? ----------------------------------------------------
t0 = time.time()
pf = pq.ParquetFile(f"{V3}/canonical/metadata/type.parquet")
parts, rows = [], 0
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=["subject", "type"])
    m = pc.is_in(t.column("type"), value_set=medset)
    n = pc.sum(m).as_py()
    if n:
        parts.append(t.filter(m).combine_chunks()); rows += n
tab = pa.concat_tables(parts); del parts
print(f"CVT type rows: {rows:,} ({time.time()-t0:.0f}s)", flush=True)

# a CVT can carry more than one mediator type; take the lexicographically smallest for determinism
t0 = time.time()
tab = tab.sort_by([("subject", "ascending"), ("type", "ascending")])
subj = tab.column("subject").combine_chunks()
typ = tab.column("type").combine_chunks()
import numpy as np
sv = subj.to_numpy(zero_copy_only=False)
first = np.empty(len(sv), dtype=bool); first[0] = True
np.not_equal(sv[1:], sv[:-1], out=first[1:])
idx = np.flatnonzero(first)
u_subj = subj.take(pa.array(idx)); u_typ = typ.take(pa.array(idx))
print(f"distinct CVT subjects: {len(idx):,} ({time.time()-t0:.0f}s)", flush=True)

lab = pa.array([label[t] for t in u_typ.to_pylist()], type=pa.string())
out = pa.table({"node_id": u_subj, "mediator_type": u_typ, "display_name": lab})
pq.write_table(out, f"{V3}/_acquisition/cvt_names.parquet", compression="zstd")

json.dump({
    "schema": "CVT_DISPLAY_NAMES/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "mediator_type_paths": len(MED),
    "type_paths_with_a_freebase_name": from_name,
    "type_paths_prettified_from_path": len(MED) - from_name,
    "cvt_type_rows": rows,
    "distinct_cvt_subjects_named": int(len(idx)),
    "cvt_nodes_total": 52272634,
    "tie_break": "a CVT carrying more than one mediator type takes the lexicographically smallest, "
                 "so the result is deterministic and reproducible.",
    "examples": out.slice(0, 12).to_pylist(),
}, open(f"{V3}/V3_CVT_DISPLAY_NAMES.json", "w", encoding="utf-8"), indent=1)
print(f"\ncovered {len(idx):,} of 52,272,634 CVT nodes")
print("examples:", out.slice(0, 6).to_pylist())
