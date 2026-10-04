"""SEMANTIC_KIND v2.1 -- the empirical grade re-measured at type-SET granularity.

    PYTHONHASHSEED=0 python .../semantic_kind_v2_1.py

WHAT CHANGES FROM v2
  v2 graded a node EMPIRICALLY_NAMELESS when every type it carries has zero named instances across
  the whole snapshot. typeset_rate.py has since measured the exact type SIGNATURE of every one of the
  112.8M typed nodes in the graph, so the grade can be asked the precise question: "does any node
  with exactly this set of types have a name anywhere in Freebase?" The type-level rule implies the
  set-level rule (a named node with the set would be a named instance of each type), so v2.1 can
  only move nodes INTO the empirical grade, never out, and the hard SOURCE_DECLARED count is untouched.

WHAT DOES NOT CHANGE
  Exact zero is still the only thing that earns EMPIRICALLY_NAMELESS. The census showed that the
  large infrastructure sets are NEAR zero, not zero -- {common.document} alone has 17,742 named
  instances in 5.52M, {type.content_import} 5 in 1.93M, {type.content} 28 in 1.40M. Those nodes stay
  in the hunt. What v2.1 adds for them is a `set_named_rate_band` (ZERO / NEAR_ZERO <1e-5 /
  LOW <1e-2 / MID <0.5 / HIGH) so that expensive archive work is spent first on classes whose
  population actually exhibits names, which is the order the user asked for. The band is a
  prioritisation, not a grade; whether NEAR_ZERO ever becomes a grade is a decision recorded
  separately, with the named exceptions listed, not made here.

OUTPUT (append-only; v1 and v2 records are left as written)
  _acquisition/semantic_kind_v2_1.parquet   node_uid, nameless_grade, recovery_class,
                                             set_named, set_total, set_named_rate_band
  V3_SEMANTIC_KIND_V2_1.json                 counts, the hunt population by band and class
"""
import sys, io, os, json, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()


def dict_codes(col):
    """(codes int array, list of labels) for a dictionary column, without materialising strings."""
    a = col.combine_chunks()
    if pa.types.is_dictionary(a.type):
        return a.indices.to_numpy(zero_copy_only=False).astype(np.int64), a.dictionary.to_pylist()
    labels = sorted(set(a.to_pylist()))
    idx = {l: i for i, l in enumerate(labels)}
    return np.fromiter((idx[x] for x in a.to_pylist()), np.int64, len(a)), labels


sk = pq.read_table(f"{ACQ}/semantic_kind_v2.parquet",
                   columns=["node_uid", "nameless_grade", "recovery_class", "has_recovered_name"])
uid = sk["node_uid"].to_numpy()
order = np.argsort(uid, kind="stable")
uid = uid[order]
g_code, g_lab = dict_codes(sk["nameless_grade"]); g_code = g_code[order]
c_code, c_lab = dict_codes(sk["recovery_class"]); c_code = c_code[order]
named = sk["has_recovered_name"].to_numpy(zero_copy_only=False)[order]
N = len(uid)
print(f"v2 loaded: {N:,} rows ({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ the set census, joined per node
rs = pq.read_table(f"{ACQ}/_residue_typeset_sig.parquet")
r_uid = rs["node_uid"].to_numpy()
r_sig = rs["sig"].to_numpy()
ts = pq.read_table(f"{ACQ}/_typeset_named_rate.parquet", columns=["sig", "total", "named"])
s_sig = ts["sig"].to_numpy(); s_tot = ts["total"].to_numpy(); s_named = ts["named"].to_numpy()
so = np.argsort(s_sig); s_sig, s_tot, s_named = s_sig[so], s_tot[so], s_named[so]
pos = np.searchsorted(s_sig, r_sig)
assert np.all(s_sig[pos] == r_sig), "every residue signature must exist in the census"
r_tot, r_named = s_tot[pos], s_named[pos]
# join onto the v2 rows
p2 = np.searchsorted(uid, r_uid)
assert np.all(uid[p2] == r_uid), "every typed residue node must be a v2 row"
set_total = np.zeros(N, np.int64); set_named = np.full(N, -1, np.int64)   # -1 = untyped
set_total[p2] = r_tot; set_named[p2] = r_named
typed = set_named >= 0
print(f"typed residue nodes joined: {typed.sum():,} of {N:,} ({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ grades and bands
G = {l: i for i, l in enumerate(g_lab)}
C = {l: i for i, l in enumerate(c_lab)}
hard = g_code == G["SOURCE_DECLARED_NAMELESS"]
inferred = g_code == G["INFERRED_NAMELESS"]
emp_v2 = g_code == G["EMPIRICALLY_NAMELESS"]
emp_set = typed & (set_named == 0) & ~named
emp = (~hard) & (~inferred) & (~named) & (emp_v2 | emp_set)
rate = np.where(typed & (set_total > 0), set_named / np.maximum(set_total, 1), np.nan)
BANDS = ["UNTYPED", "ZERO", "NEAR_ZERO", "LOW", "MID", "HIGH"]
band = np.zeros(N, np.int8)
band[typed & (set_named == 0)] = 1
band[typed & (set_named > 0) & (rate < 1e-5)] = 2
band[typed & (rate >= 1e-5) & (rate < 1e-2)] = 3
band[typed & (rate >= 1e-2) & (rate < 0.5)] = 4
band[typed & (rate >= 0.5)] = 5

NG = ["NOT_NAMELESS", "SOURCE_DECLARED_NAMELESS", "EMPIRICALLY_NAMELESS", "INFERRED_NAMELESS"]
ng = np.zeros(N, np.int8)
ng[hard] = 1; ng[emp] = 2; ng[inferred & ~hard] = 3
# recovery class: v2 classes, with nodes newly graded empirical moved to EMPIRICALLY_NAMELESS_INTERNAL
rc = c_code.copy()
moved = emp & ~emp_v2 & ~named
rc[moved] = C["EMPIRICALLY_NAMELESS_INTERNAL"]
HUNT = {"LIKELY_REAL_ENTITY", "LIKELY_CREATIVE_WORK", "LIKELY_MEDIA_ENTITY", "LIKELY_PLACE",
        "LIKELY_PERSON", "UNTYPED_CANDIDATE"}
hunt_codes = np.array([C[h] for h in HUNT if h in C])
in_hunt = np.isin(rc, hunt_codes)
print(f"newly EMPIRICALLY_NAMELESS at set level: {moved.sum():,}; hunt V2.1 = {in_hunt.sum():,} "
      f"({time.time()-t0:.0f}s)", flush=True)

pq.write_table(pa.table({
    "node_uid": pa.array(uid, pa.int64()),
    "nameless_grade": pa.DictionaryArray.from_arrays(pa.array(ng, pa.int8()), pa.array(NG)),
    "recovery_class": pa.DictionaryArray.from_arrays(pa.array(rc.astype(np.int8), pa.int8()), pa.array(c_lab)),
    "set_named": pa.array(set_named, pa.int64()),
    "set_total": pa.array(set_total, pa.int64()),
    "set_named_rate_band": pa.DictionaryArray.from_arrays(pa.array(band, pa.int8()), pa.array(BANDS))}),
    f"{ACQ}/semantic_kind_v2_1.parquet", compression="zstd")


def bc(codes, labels, mask=None):
    m = codes if mask is None else codes[mask]
    c = np.bincount(m, minlength=len(labels))
    return {labels[i]: int(c[i]) for i in np.argsort(-c) if c[i]}


hunt_by_band = bc(band, BANDS, in_hunt)
hunt_by_class = bc(rc, c_lab, in_hunt)
cross = {}
for b in range(len(BANDS)):
    m = in_hunt & (band == b)
    if m.any():
        cross[BANDS[b]] = bc(rc, c_lab, m)
# the biggest near-zero sets inside the hunt, with their exact counts: the evidence for any later decision
sig_of_hunt = np.full(N, 0, np.uint64); sig_of_hunt[p2] = r_sig
m = in_hunt & (band == 2)
u_s, cnt = np.unique(sig_of_hunt[m], return_counts=True)
top = np.argsort(-cnt)[:25]
_tt = pq.read_table(f"{ACQ}/_typeset_named_rate.parquet", columns=["sig", "types"])
types_by_sig = dict(zip(_tt["sig"].to_numpy().tolist(), _tt["types"].to_pylist()))
near_zero_top = []
for i in top:
    s = int(u_s[i]); k = np.searchsorted(s_sig, np.uint64(s))
    near_zero_top.append({"types": types_by_sig[s], "hunt_nodes": int(cnt[i]),
                          "graph_total": int(s_tot[k]), "graph_named": int(s_named[k])})

rec = {"schema": "SEMANTIC_KIND_OVERLAY/v2.1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "SUPERSEDES_NOTHING": "V3_SEMANTIC_KIND.json and V3_SEMANTIC_KIND_V2.json are left as written; new record.",
       "APPEND_ONLY": "node_kind and CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 untouched.",
       "population": int(N),
       "EMPIRICAL_RULE": "exact type SET of the node has zero named instances across all 112.8M typed nodes (typeset_rate.py); type-level rule of v2 is implied and retained",
       "by_nameless_grade": bc(ng, NG),
       "HARD_COUNT_SOURCE_DECLARED_NAMELESS": int(hard.sum()),
       "EMPIRICALLY_NAMELESS_v2_type_level": int(emp_v2.sum()),
       "EMPIRICALLY_NAMELESS_v2_1_set_level": int(emp.sum()),
       "moved_into_empirical_by_set_level": int(moved.sum()),
       "INFERRED_CVT_HELD_APART": int((ng == 3).sum()),
       "by_recovery_class": bc(rc, c_lab),
       "TRUE_NAME_RECOVERY_RESIDUE_V2_1": int(in_hunt.sum()),
       "TRUE_NAME_RECOVERY_RESIDUE_V2_for_comparison": int(np.isin(c_code, hunt_codes).sum()),
       "hunt_by_set_named_rate_band": hunt_by_band,
       "BANDS": {"ZERO": "set never named (would be empirical, but node is named/declared otherwise)",
                 "NEAR_ZERO": "0 < named/total < 1e-5", "LOW": "< 1e-2", "MID": "< 0.5", "HIGH": ">= 0.5",
                 "UNTYPED": "node carries no type at all"},
       "hunt_by_class": hunt_by_class,
       "hunt_band_x_class": cross,
       "near_zero_sets_in_hunt_top25": near_zero_top,
       "ORDER_OF_WORK": "HIGH -> MID -> LOW -> UNTYPED -> NEAR_ZERO: archive spend goes first to populations that demonstrably carry names",
       "NEAR_ZERO_IS_NOT_A_GRADE": "the named exceptions of each near-zero set are listed by typeset_named_examples.py; admitting NEAR_ZERO as a nameless grade is a recorded decision, not a computation",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_SEMANTIC_KIND_V2_1.json", "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=1)
print(json.dumps({k: v for k, v in rec.items() if k not in ("near_zero_sets_in_hunt_top25", "hunt_band_x_class")}, indent=1))
print("\nhunt band x class:")
for b, d in cross.items():
    print(f"  {b}: {d}")
print("\nnear-zero sets in the hunt (top 25):")
for r in near_zero_top:
    print(f"  {r['hunt_nodes']:>10,}  named {r['graph_named']:>6,} / {r['graph_total']:>10,}  {r['types'][:100]}")
print(f"{time.time()-t0:.0f}s")
