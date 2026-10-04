# -*- coding: utf-8 -*-
"""Analyse the floor evidence (scratchpad/fb4/evidence/*) so the GENERATED_FLOOR naming rules are
written against measured shapes, not guessed ones.  Reads only; writes scratchpad/fb4/FLOOR_ANALYSIS.json.

Per floor node: kind, tier (INFERRED_REJECTED / NO_INFERENCE_ROW), declared types, description /
alias / key reach, out- and in-relation multisets, named-neighbour availability (a neighbour is
'named' when it is not itself a floor node: every non-floor node resolves to ORIGINAL / RECOVERED /
URI_SELF / KEY_SEMANTIC / INFERRED under names_v4).
"""
import collections
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCR = "scratchpad/fb4"
EV = SCR + "/evidence"
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


def member(sorted_arr, q):
    p = np.searchsorted(sorted_arr, q)
    p2 = np.minimum(p, len(sorted_arr) - 1)
    return sorted_arr[p2] == q


floor = np.load(SCR + "/floor_uids.npy")
ftier = np.load(SCR + "/floor_tier.npy")
fkind = np.load(SCR + "/floor_kind.npy")
NF = len(floor)
TIERS = ["ORIGINAL", "RECOVERED", "URI_SELF", "KEY_SEMANTIC", "INFERRED_ADMISSIBLE", "INFERRED_REJECTED", "NO_INFERENCE_ROW"]
KINDS = ["LITERAL", "EXTERNAL_URI", "ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER"]

ids = pq.read_table(EV + "/floor_ids.parquet")
assert ids.num_rows == NF
iu = ids["node_uid"].to_numpy()
irow = np.searchsorted(floor, iu)
assert np.array_equal(floor[irow], iu)
node_id = np.empty(NF, dtype=object)
node_id[irow] = np.array(ids["node_id"].to_pylist(), dtype=object)
mid_prefix = collections.Counter(s.split(".")[0] if "." in s else "<nodot>" for s in node_id.tolist())
log("ids", NF, "prefixes", mid_prefix.most_common(8))

out = {"RECORD": "FLOOR_ANALYSIS", "floor_nodes": NF,
       "by_tier": {TIERS[t]: int(c) for t, c in zip(*np.unique(ftier, return_counts=True))},
       "by_kind": {KINDS[k]: int(c) for k, c in zip(*np.unique(fkind, return_counts=True))},
       "id_prefix": dict(mid_prefix.most_common(10))}

# ---- types ---------------------------------------------------------------------------------------
ty = pq.read_table(EV + "/floor_types.parquet")
tu = ty["node_uid"].to_numpy()
trow = np.searchsorted(floor, tu)
assert np.array_equal(floor[trow], tu)
tname = np.array(ty["type"].to_pylist(), dtype=object)
n_types = np.bincount(trow, minlength=NF)
type_counter = collections.Counter(tname.tolist())
# type-set signature per node
order = np.argsort(trow, kind="stable")
sig = {}
r_sorted, t_sorted = trow[order], tname[order]
bounds = np.flatnonzero(np.diff(r_sorted)) + 1
starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(r_sorted)]])
sig_counter = collections.Counter()
node_sig = np.empty(NF, dtype=object)
for s, e in zip(starts, ends):
    key = "|".join(sorted(t_sorted[s:e].tolist()))
    sig_counter[key] += 1
    node_sig[r_sorted[s]] = key
untyped = n_types == 0
out["TYPES"] = {"typed_nodes": int((~untyped).sum()), "untyped_nodes": int(untyped.sum()),
                "untyped_by_tier": {TIERS[t]: int(c) for t, c in zip(*np.unique(ftier[untyped], return_counts=True))},
                "type_rows": ty.num_rows, "distinct_types": len(type_counter),
                "top_types": type_counter.most_common(60), "top_typesets": sig_counter.most_common(60),
                "distinct_typesets": len(sig_counter)}
log("types", out["TYPES"]["typed_nodes"], "typed;", out["TYPES"]["untyped_nodes"], "untyped")

# ---- description / alias / key --------------------------------------------------------------------
def reach(name, cols):
    t = pq.read_table(EV + "/floor_%s.parquet" % name)
    u = t["node_uid"].to_numpy()
    r = np.searchsorted(floor, u)
    assert np.array_equal(floor[r], u)
    has = np.zeros(NF, dtype=bool); has[r] = True
    d = {"rows": t.num_rows, "nodes": int(has.sum()),
         "by_tier": {TIERS[k]: int(c) for k, c in zip(*np.unique(ftier[has], return_counts=True))} if has.any() else {}}
    return t, r, has, d

desc_t, desc_r, has_desc, out["DESCRIPTION"] = reach("desc", ["lexical", "lang"])
if desc_t.num_rows:
    out["DESCRIPTION"]["lang_top"] = collections.Counter(desc_t["lang"].to_pylist()).most_common(8)
    out["DESCRIPTION"]["samples"] = [(node_id[desc_r[i]], desc_t["lexical"][i].as_py()[:120]) for i in range(0, desc_t.num_rows, max(1, desc_t.num_rows // 12))][:12]
alias_t, alias_r, has_alias, out["ALIAS"] = reach("alias", ["lexical", "lang"])
if alias_t.num_rows:
    out["ALIAS"]["samples"] = [(node_id[alias_r[i]], alias_t["lexical"][i].as_py()[:80]) for i in range(0, alias_t.num_rows, max(1, alias_t.num_rows // 12))][:12]
key_t, key_r, has_key, out["KEY"] = reach("keys", ["key"])
if key_t.num_rows:
    keys = key_t["key"].to_pylist()
    ns = collections.Counter("/".join(k.split("/")[:3]) for k in keys)
    out["KEY"]["namespace_top"] = ns.most_common(40)
    out["KEY"]["samples"] = [(node_id[key_r[i]], keys[i][:100]) for i in range(0, len(keys), max(1, len(keys) // 25))][:25]
log("desc", out["DESCRIPTION"]["nodes"], "alias", out["ALIAS"]["nodes"], "key", out["KEY"]["nodes"])

# ---- edges ---------------------------------------------------------------------------------------
pf = pq.ParquetFile(EV + "/floor_edges.parquet")
n_out = np.zeros(NF, dtype=np.int64); n_in = np.zeros(NF, dtype=np.int64)
named_nb_out = np.zeros(NF, dtype=bool); named_nb_in = np.zeros(NF, dtype=bool)
rel_out = collections.Counter(); rel_in = collections.Counter()
rel_out_named = collections.Counter(); rel_in_named = collections.Counter()
sig_out = collections.defaultdict(collections.Counter)  # typeset -> out relation multiset counter
sig_in = collections.defaultdict(collections.Counter)
nb_floor_count = 0
rows = 0
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg)
    u = t["node_uid"].to_numpy(); r = np.searchsorted(floor, u); assert np.array_equal(floor[r], u)
    other = t["other_uid"].to_numpy()
    other_floor = member(floor, other)
    nb_floor_count += int(other_floor.sum())
    d = np.array(t["direction"].to_pylist()); rel = np.array(t["relation"].to_pylist(), dtype=object)
    is_out = d == "out"
    np.add.at(n_out, r[is_out], 1); np.add.at(n_in, r[~is_out], 1)
    named = ~other_floor
    named_nb_out[r[is_out & named]] = True; named_nb_in[r[~is_out & named]] = True
    rel_out.update(rel[is_out].tolist()); rel_in.update(rel[~is_out].tolist())
    rel_out_named.update(rel[is_out & named].tolist()); rel_in_named.update(rel[~is_out & named].tolist())
    for rr, rl, io_ in zip(r.tolist(), rel.tolist(), is_out.tolist()):
        (sig_out if io_ else sig_in)[node_sig[rr] or "<untyped>"][rl] += 1
    rows += t.num_rows
    if rg % 20 == 0:
        log("edges rg", rg, "/", pf.num_row_groups, "rows", rows)
deg = n_out + n_in
named_any = named_nb_out | named_nb_in
out["EDGES"] = {
    "rows": rows, "neighbour_is_floor_rows": nb_floor_count,
    "isolated_nodes": int((deg == 0).sum()),
    "nodes_with_out": int((n_out > 0).sum()), "nodes_with_in": int((n_in > 0).sum()),
    "nodes_with_named_neighbour": int(named_any.sum()),
    "nodes_with_named_neighbour_by_tier": {TIERS[k]: int(c) for k, c in zip(*np.unique(ftier[named_any], return_counts=True))},
    "nodes_without_named_neighbour_by_typeset": collections.Counter((node_sig[i] or "<untyped>") for i in np.flatnonzero(~named_any)).most_common(30),
    "deg_percentiles": {p: float(np.percentile(deg, p)) for p in (50, 90, 99, 99.9)}, "deg_max": int(deg.max()),
    "out_relations_top": rel_out.most_common(60), "in_relations_top": rel_in.most_common(60),
    "out_relations_to_named_top": rel_out_named.most_common(40), "in_relations_from_named_top": rel_in_named.most_common(40),
    "typeset_out_relations": {k: v.most_common(12) for k, v in sorted(sig_out.items(), key=lambda kv: -sum(kv[1].values()))[:40]},
    "typeset_in_relations": {k: v.most_common(12) for k, v in sorted(sig_in.items(), key=lambda kv: -sum(kv[1].values()))[:40]},
}
# joint availability
out["AVAILABILITY"] = {
    "typed": int((~untyped).sum()),
    "typed_and_named_neighbour": int(((~untyped) & named_any).sum()),
    "typed_no_named_neighbour": int(((~untyped) & ~named_any).sum()),
    "untyped_and_named_neighbour": int((untyped & named_any).sum()),
    "untyped_no_named_neighbour": int((untyped & ~named_any).sum()),
    "untyped_no_named_neighbour_with_key": int((untyped & ~named_any & has_key).sum()),
    "untyped_no_named_neighbour_isolated": int((untyped & ~named_any & (deg == 0)).sum()),
    "nothing_at_all(untyped,no edge,no desc/alias/key)": int((untyped & (deg == 0) & ~has_desc & ~has_alias & ~has_key).sum()),
}
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(SCR + "/FLOOR_ANALYSIS.json", "w"), indent=1, ensure_ascii=False)
log("done", json.dumps(out["AVAILABILITY"]))
