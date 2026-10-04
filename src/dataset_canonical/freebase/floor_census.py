# -*- coding: utf-8 -*-
"""Census of the served-name FLOOR of CRAG_FREEBASE_CANONICAL before the name-completion pass.

For every one of the 301,977,131 nodes decide which tier the NAME_HIERARCHY_CONTRACT_V1 rule
resolves it to, and additionally where the 534,532 cascade names (V3_ACTUAL_NAME_LAYER_FROZEN)
would land -- they are ACTUAL names but the V1 hierarchy never read cascade_names.parquet, so
under that rule they resolve to the floor.

Tiers assigned here (per node, disjoint):
    ORIGINAL            overlay_v1.is_original_name (literal self, current/deleted exact)
    RECOVERED           in _acquisition/cascade_names.parquet (source-derived, not in overlay)
    URI_SELF            overlay display_name_source == URI_DERIVED (the node IS the resource)
    KEY_SEMANTIC        overlay FREEBASE_KEY_EXACT (readable key rendered; not the object's name)
    INFERRED_ADMISSIBLE inference_overlay_v1 row with reject_reason 0 and not in the v2 amendment
    INFERRED_REJECTED   inference_overlay_v1 row rejected (bookkeeping head / unreadable payload)
    NO_INFERENCE_ROW    handed to the inference layer, no rendering produced

FLOOR (what the completion pass must name) = INFERRED_REJECTED + NO_INFERENCE_ROW.
Writes: <out>/floor_uids.npy (int64, sorted), floor_kind.npy (int8 codes), the census json,
and a stratified sample with semantic_text for rule design.  Reads only; nothing frozen changes.
"""
import glob
import io
import json
import os
import sys
import time
from collections import Counter

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

FB = "data/final_canonical/freebase_v3"
OUT = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/fb4"
os.makedirs(OUT, exist_ok=True)
KINDS = ["LITERAL", "EXTERNAL_URI", "ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER"]
KIND_CODE = {k: i for i, k in enumerate(KINDS)}
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


# ---- inference layer: uid -> (admissible?) ----------------------------------------------------
inf_files = sorted(glob.glob(FB + "/inference_overlay_v1/part_*.parquet"))
adm_files = sorted(glob.glob(FB + "/inference_admissibility_v1/part_*.parquet"))
assert len(inf_files) == len(adm_files) == 32, (len(inf_files), len(adm_files))
uids, rej, srcs = [], [], []
src_dict = {}
for f, g in zip(inf_files, adm_files):
    t = pq.read_table(f, columns=["node_uid", "inference_source"])
    a = pq.read_table(g, columns=["reject_reason"])
    assert t.num_rows == a.num_rows, (f, t.num_rows, a.num_rows)
    uids.append(t["node_uid"].to_numpy())
    rej.append(a["reject_reason"].to_numpy().astype(np.int8))
    s = pc.cast(t["inference_source"], "string").combine_chunks().dictionary_encode()
    d = s.dictionary.to_pylist()
    m = np.array([src_dict.setdefault(x, len(src_dict)) for x in d], np.int8)
    srcs.append(m[s.indices.to_numpy(zero_copy_only=False)])
inf_uid = np.concatenate(uids)
inf_rej = np.concatenate(rej)
inf_src = np.concatenate(srcs)
del uids, rej, srcs
amend = json.load(io.open(FB + "/inference_admissibility_v2_amendment.json", encoding="utf-8"))
am_uids = {int(r["node_uid"]) for r in amend["REJECTED_ROWS"]}
assert len(am_uids) == amend["SCALE"]["rows_rejected_by_this_amendment"] == 36
log("inference rows", len(inf_uid), "admissible v1", int((inf_rej == 0).sum()), "v2 amendment uids", len(am_uids))
o = np.argsort(inf_uid, kind="stable")
inf_uid, inf_rej, inf_src = inf_uid[o], inf_rej[o], inf_src[o]
del o
assert np.all(np.diff(inf_uid) > 0), "inference node_uid not unique"
if am_uids:
    pos = np.searchsorted(inf_uid, np.array(sorted(am_uids), np.int64))
    hit = inf_uid[np.minimum(pos, len(inf_uid) - 1)] == np.array(sorted(am_uids), np.int64)
    inf_rej[pos[hit]] = np.where(inf_rej[pos[hit]] == 0, 2, inf_rej[pos[hit]])
    log("amendment applied to", int(hit.sum()), "rows")
log("admissible effective", int((inf_rej == 0).sum()))

# ---- cascade ----------------------------------------------------------------------------------
c = pq.read_table(FB + "/_acquisition/cascade_names.parquet", columns=["node_uid", "source"])
cas_uid = c["node_uid"].to_numpy()
cas_src = pc.cast(c["source"], "string").to_pylist()
o = np.argsort(cas_uid, kind="stable")
cas_uid = cas_uid[o]
cas_src = [cas_src[i] for i in o]
assert np.all(np.diff(cas_uid) > 0)
log("cascade rows", len(cas_uid))


def member(sorted_arr, q):
    p = np.searchsorted(sorted_arr, q)
    p2 = np.minimum(p, len(sorted_arr) - 1)
    return sorted_arr[p2] == q, p2


# ---- overlay x nodes pass ---------------------------------------------------------------------
TIERS = ["ORIGINAL", "RECOVERED", "URI_SELF", "KEY_SEMANTIC", "INFERRED_ADMISSIBLE", "INFERRED_REJECTED", "NO_INFERENCE_ROW"]
TC = {t: i for i, t in enumerate(TIERS)}
tab = Counter()           # (kind, overlay_source, tier) -> n
floor_str = Counter()     # overlay display string on FLOOR nodes
floor_uids, floor_kind, floor_tier = [], [], []
sample = []
rng = np.random.default_rng(20260912)
ov_files = sorted(glob.glob(FB + "/overlay_v1/*.parquet"))
assert len(ov_files) == 262
n_total = 0
for i, f in enumerate(ov_files):
    name = os.path.basename(f)
    ov = pq.read_table(f, columns=["node_uid", "display_name", "display_name_source", "is_original_name"])
    nd = pq.read_table(FB + "/canonical/nodes/" + name, columns=["node_uid", "kind"])
    assert ov.num_rows == nd.num_rows
    u = ov["node_uid"].to_numpy()
    assert np.array_equal(u, nd["node_uid"].to_numpy()), "overlay/nodes misaligned: " + name
    n_total += len(u)
    kind = pc.cast(nd["kind"], "string").combine_chunks().dictionary_encode()
    kd = [KIND_CODE[x] for x in kind.dictionary.to_pylist()]
    kcode = np.array(kd, np.int8)[kind.indices.to_numpy(zero_copy_only=False)]
    src = pc.cast(ov["display_name_source"], "string").combine_chunks().dictionary_encode()
    sd = src.dictionary.to_pylist()
    scode = src.indices.to_numpy(zero_copy_only=False)
    orig = ov["is_original_name"].to_numpy(zero_copy_only=False)
    tier = np.full(len(u), -1, np.int8)
    tier[orig] = TC["ORIGINAL"]
    hit, p = member(cas_uid, u)
    tier[hit & (tier < 0)] = TC["RECOVERED"]
    uri = np.array([s == "URI_DERIVED" for s in sd])[scode]
    tier[uri & (tier < 0)] = TC["URI_SELF"]
    key = np.array([s == "FREEBASE_KEY_EXACT" for s in sd])[scode]
    tier[key & (tier < 0)] = TC["KEY_SEMANTIC"]
    ihit, ip = member(inf_uid, u)
    adm = ihit & (inf_rej[ip] == 0)
    tier[adm & (tier < 0)] = TC["INFERRED_ADMISSIBLE"]
    tier[ihit & ~adm & (tier < 0)] = TC["INFERRED_REJECTED"]
    tier[tier < 0] = TC["NO_INFERENCE_ROW"]
    # cross-tab
    key3 = kcode.astype(np.int64) * 10000 + scode.astype(np.int64) * 100 + tier
    vals, cnts = np.unique(key3, return_counts=True)
    for v, n in zip(vals.tolist(), cnts.tolist()):
        tab[(KINDS[v // 10000], sd[(v // 100) % 100], TIERS[v % 100])] += n
    fl = (tier == TC["INFERRED_REJECTED"]) | (tier == TC["NO_INFERENCE_ROW"])
    if fl.any():
        floor_uids.append(u[fl]); floor_kind.append(kcode[fl]); floor_tier.append(tier[fl])
        dn = ov["display_name"].combine_chunks().take(np.flatnonzero(fl)).dictionary_encode()
        dd = dn.dictionary.to_pylist()
        di = dn.indices.to_numpy(zero_copy_only=False)
        bc = np.bincount(di, minlength=len(dd))
        for j in np.flatnonzero(bc):
            floor_str[dd[j]] += int(bc[j])
        # stratified sample: up to 4 per shard
        idx = np.flatnonzero(fl)
        pick = rng.choice(idx, size=min(4, len(idx)), replace=False)
        sem = pq.read_table(FB + "/semantic_v1/" + name, columns=["node_uid", "semantic_text"])
        assert np.array_equal(sem["node_uid"].to_numpy()[pick], u[pick])
        nid = pq.read_table(FB + "/canonical/nodes/" + name, columns=["node_id"])["node_id"]
        for j in pick.tolist():
            rr = {"node_uid": int(u[j]), "node_id": nid[j].as_py(), "kind": KINDS[kcode[j]], "overlay_source": sd[scode[j]],
                  "overlay_display": ov["display_name"][j].as_py(), "tier": TIERS[tier[j]],
                  "semantic_text": sem["semantic_text"][j].as_py()}
            if ihit[j]:
                rr["inference_source"] = [k for k, v in src_dict.items() if v == inf_src[ip[j]]][0]
                rr["reject_reason"] = int(inf_rej[ip[j]])
            sample.append(rr)
    if i % 20 == 0:
        log("shard", i, name, "rows so far", n_total, "floor so far", sum(len(x) for x in floor_uids))

floor_uids = np.concatenate(floor_uids); floor_kind = np.concatenate(floor_kind); floor_tier = np.concatenate(floor_tier)
o = np.argsort(floor_uids, kind="stable")
floor_uids, floor_kind, floor_tier = floor_uids[o], floor_kind[o], floor_tier[o]
np.save(OUT + "/floor_uids.npy", floor_uids)
np.save(OUT + "/floor_kind.npy", floor_kind)
np.save(OUT + "/floor_tier.npy", floor_tier)
by_tier = Counter()
by_kind_tier = Counter()
for (k, s, t), n in tab.items():
    by_tier[t] += n
    by_kind_tier[k + "|" + t] += n
rec = {
    "RECORD": "V4_FLOOR_CENSUS", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "nodes_total": n_total, "CHECK_total": n_total == 301977131,
    "by_tier": dict(sorted(by_tier.items(), key=lambda x: -x[1])),
    "by_kind_and_tier": dict(sorted(by_kind_tier.items(), key=lambda x: -x[1])),
    "by_kind_source_tier": {"|".join(k): v for k, v in sorted(tab.items(), key=lambda x: -x[1])},
    "FLOOR": {"n": int(len(floor_uids)), "definition": "INFERRED_REJECTED + NO_INFERENCE_ROW; URI_SELF and KEY_SEMANTIC are NOT floor here (the node's own identifier / readable key)",
              "by_kind": {KINDS[k]: int(n) for k, n in zip(*np.unique(floor_kind, return_counts=True))},
              "by_tier": {TIERS[k]: int(n) for k, n in zip(*np.unique(floor_tier, return_counts=True))},
              "top_overlay_strings": floor_str.most_common(60), "distinct_overlay_strings": len(floor_str)},
    "inference_sources": src_dict, "cascade_rows": int(len(cas_uid)),
    "NOTE": "cascade names are ACTUAL names (V3_ACTUAL_NAME_LAYER_FROZEN) that NAME_HIERARCHY_CONTRACT_V1 never read; under that contract they resolve to the floor. Counted here as RECOVERED.",
    "seconds": round(time.time() - t0, 1),
}
json.dump(rec, io.open(OUT + "/V4_FLOOR_CENSUS.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(sample, io.open(OUT + "/floor_sample.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("done; floor", len(floor_uids), "by_tier", dict(by_tier))
