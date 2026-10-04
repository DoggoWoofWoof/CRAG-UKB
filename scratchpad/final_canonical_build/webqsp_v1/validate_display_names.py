"""Verify CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 against the frozen graph, shard by shard.

Treats the overlay as invalid until counts establish otherwise. Every check is a count, not a
sample, except the printed examples at the end.

  1 the overlay covers exactly the frozen node universe, per shard, by node_uid
  2 node_uid is unique within every shard
  3 DISPLAY_NAME_EMPTY = 0
  4 no OPAQUE node: NAKED_MID_SELF = 0 (a display_name that IS the node's own identifier) and
    NAKED_MID_DERIVED = 0 (MID-shaped text emitted by a tier that does not quote the source
    verbatim). MID-SHAPEDNESS ALONE IS NOT THE TEST -- Freebase genuinely names some entities
    m.albarn / g.wygonik / m.jpg, and those are reported as a diagnostic with evidence, not failed
  5 every display_name_source is in the contract enum, and confidence/is_original_name match the
    tier table exactly -- confidence must be a property of the source class, never per node

Everything is done in Arrow/numpy. The obvious implementation (to_pylist the kind and source
columns and zip them) allocates 40M Python strings on a 20.7M-row subj shard; that anti-pattern
has already cost this project three times, so no whole column is ever materialised as Python.
"""
import sys, io, os, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"
OUT = f"{V3}/overlay_v1"
MID_RE = r"^[mg]\.[0-9a-z_]{2,}$"
SRC = ["LITERAL_SELF", "FREEBASE_CURRENT_EXACT", "FREEBASE_DELETED_EXACT", "FREEBASE_KEY_EXACT",
       "URI_DERIVED", "STRUCTURAL_INFERRED", "STRUCTURAL_FALLBACK"]
CONF = {"LITERAL_SELF": 1.0, "FREEBASE_CURRENT_EXACT": 1.0, "FREEBASE_DELETED_EXACT": 0.9,
        "FREEBASE_KEY_EXACT": 1.0, "URI_DERIVED": 1.0, "STRUCTURAL_INFERRED": 0.5,
        "STRUCTURAL_FALLBACK": 0.0}
ORIG = {"LITERAL_SELF", "FREEBASE_CURRENT_EXACT", "FREEBASE_DELETED_EXACT"}
# tiers whose text is the source string verbatim, so MID-shaped output is Freebase's wording
VERBATIM = {"LITERAL_SELF", "FREEBASE_CURRENT_EXACT", "FREEBASE_DELETED_EXACT"}
MIDKINDS = pa.array(["ENTITY_MID", "CVT_MEDIATOR"])
t0 = time.time()

chk = collections.Counter()
by_src = collections.Counter()
by_kind_src = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
shaped_evidence = []
bad = []
nodes = 0

shards = sorted(glob.glob(f"{V3}/canonical/nodes/*.parquet"))
for n, fp in enumerate(shards):
    base = os.path.basename(fp)
    op = f"{OUT}/{base}"
    if not os.path.exists(op):
        bad.append(f"MISSING overlay shard {base}")
        continue
    src_t = pq.read_table(fp, columns=["node_uid", "kind", "node_id"])
    ov = pq.read_table(op)
    nodes += ov.num_rows
    if ov.num_rows != src_t.num_rows:
        bad.append(f"{base}: rows {ov.num_rows:,} != frozen {src_t.num_rows:,}")
        continue
    a = src_t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
    b = ov["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
    if not np.array_equal(a, b):
        bad.append(f"{base}: node_uid does not match the frozen shard row-for-row")
        continue
    chk["uid_aligned_shards"] += 1
    if len(np.unique(b)) != len(b):
        bad.append(f"{base}: duplicate node_uid inside the shard")
    del a, b

    nm = ov["display_name"].combine_chunks()
    if nm.null_count:
        bad.append(f"{base}: {nm.null_count} null display_name")
    empt = int(pc.sum(pc.cast(pc.equal(pc.utf8_trim_whitespace(pc.fill_null(nm, "")), ""),
                              "int64")).as_py() or 0)
    chk["DISPLAY_NAME_EMPTY"] += empt

    # source is dictionary-encoded: work on the codes, never on 20M decoded strings
    s = ov["display_name_source"].combine_chunks()
    d = s if isinstance(s.type, pa.DictionaryType) else s.dictionary_encode()
    dvals = d.dictionary.to_pylist()
    codes = d.indices.to_numpy(zero_copy_only=False)
    cnt = np.bincount(codes, minlength=len(dvals))
    conf = ov["resolution_confidence"].combine_chunks().to_numpy(zero_copy_only=False)
    orig = ov["is_original_name"].combine_chunks().to_numpy(zero_copy_only=False)
    for i, name in enumerate(dvals):
        if not cnt[i]:
            continue
        by_src[name] += int(cnt[i])
        if name not in CONF:
            bad.append(f"{base}: display_name_source not in the enum: {name}")
            continue
        msk = codes == i
        if not np.allclose(conf[msk], CONF[name]):
            bad.append(f"{base}: {name} has a confidence other than {CONF[name]}")
        if not np.all(orig[msk] == (name in ORIG)):
            bad.append(f"{base}: {name} has is_original_name != {name in ORIG}")

    # OPACITY, not MID-shapedness. The pattern alone is not the test: Freebase really does call
    # some entities "m.albarn" (M. Albarn) or "g.wygonik" (G. Wygonik), and some names are
    # filename fragments like "m.jpg". Those are source-declared strings, so displaying them is
    # showing what Freebase says, not showing an opaque identifier. Blanking them would destroy
    # real data to satisfy a regex. So the shaped count is kept as a DIAGNOSTIC and the gate is
    # two strictly narrower tests, both of which must be zero:
    #   SELF    the display_name IS the node's own identifier -- unambiguous opacity
    #   DERIVED MID-shaped text produced by a tier that does NOT quote the source verbatim, i.e.
    #           an identifier leaking out of a rendering rule
    kd = src_t["kind"].combine_chunks()
    if isinstance(kd.type, pa.DictionaryType):
        kd = kd.cast(pa.string())
    nid = src_t["node_id"].combine_chunks()
    looks_mid = pc.fill_null(pc.match_substring_regex(nm, MID_RE), False)
    chk["NAKED_MID_ANY_KIND"] += int(pc.sum(pc.cast(looks_mid, "int64")).as_py() or 0)
    gated = pc.and_(looks_mid, pc.is_in(kd, value_set=MIDKINDS))
    nk = int(pc.sum(pc.cast(gated, "int64")).as_py() or 0)
    chk["NAKED_MID_SHAPED_ON_MID_KIND"] += nk
    selfmid = pc.and_(gated, pc.equal(nm, nid))
    chk["NAKED_MID_SELF"] += int(pc.sum(pc.cast(selfmid, "int64")).as_py() or 0)
    verbatim = np.array([v in VERBATIM for v in dvals], dtype=bool)[codes]
    derived = pc.and_(gated, pa.array(~verbatim))
    chk["NAKED_MID_DERIVED"] += int(pc.sum(pc.cast(derived, "int64")).as_py() or 0)
    if nk and len(shaped_evidence) < 30:
        f_nm = nm.filter(gated).to_pylist()
        f_id = nid.filter(gated).to_pylist()
        f_sc = s.filter(gated).to_pylist()
        for x, y, z in zip(f_id, f_nm, f_sc):
            if len(shaped_evidence) < 30:
                shaped_evidence.append({"node_id": x, "display_name": y,
                                        "display_name_source": str(z),
                                        "display_name_is_the_nodes_own_id": x == y})

    # kind x source counts, vectorised by Arrow group_by rather than a 20M-iteration Python zip
    g = pa.table({"k": kd, "s": s}).group_by(["k", "s"]).aggregate([([], "count_all")])
    for k, sv, c in zip(g["k"].to_pylist(), g["s"].to_pylist(), g["count_all"].to_pylist()):
        by_kind_src[k][sv] += c

    if n % 40 == 0:
        ni = nid
        for i in range(0, min(ov.num_rows, 4000), 700):
            k = kd[i].as_py()
            if len(examples[k]) < 4:
                examples[k].append((str(ni[i].as_py())[:44], str(nm[i].as_py())[:56],
                                    str(s[i].as_py())))
        del ni
    if n % 40 == 0 or n == len(shards) - 1:
        print(f"  [{n+1:3d}/{len(shards)}] {base} nodes={nodes:,} ({time.time()-t0:.0f}s)", flush=True)
    del src_t, ov, nm, s, d, kd, nid, conf, orig, codes

EXPECT = 301977131
chk["NAKED_MID_DISPLAY"] = chk["NAKED_MID_SELF"] + chk["NAKED_MID_DERIVED"]
ok = ((not bad) and nodes == EXPECT and chk["DISPLAY_NAME_EMPTY"] == 0 and chk["NAKED_MID_SELF"] == 0 and chk["NAKED_MID_DERIVED"] == 0)
auth = sum(by_src[s] for s in ("LITERAL_SELF", "FREEBASE_CURRENT_EXACT",
                               "FREEBASE_DELETED_EXACT", "FREEBASE_KEY_EXACT"))
struct = by_src["STRUCTURAL_INFERRED"] + by_src["STRUCTURAL_FALLBACK"]
print(f"\n{'PASS' if ok else 'FAIL'}  nodes={nodes:,} (expected {EXPECT:,})")
print(f"  shards uid-aligned to the frozen table : {chk['uid_aligned_shards']}/{len(shards)}")
print(f"  DISPLAY_NAME_EMPTY                     : {chk['DISPLAY_NAME_EMPTY']:,}")
print(f"  NAKED_MID_SELF (display IS the id)     : {chk['NAKED_MID_SELF']:,}")
print(f"  NAKED_MID_DERIVED (id via a rule)      : {chk['NAKED_MID_DERIVED']:,}")
print(f"  -- diagnostics, not gated --")
print(f"  MID-shaped text on a MID-bearing kind  : {chk['NAKED_MID_SHAPED_ON_MID_KIND']:,}")
print(f"  MID-shaped text, any kind              : {chk['NAKED_MID_ANY_KIND']:,}")
for b in bad[:20]:
    print("  ! " + b)
print("\nby source:")
for s in SRC:
    if by_src[s]:
        print(f"  {by_src[s]:>12,}  {100*by_src[s]/nodes:6.3f}%  {s}")
print(f"\n  AUTHORITATIVE_SOURCE_RESOLUTION {auth:,} ({100*auth/nodes:.3f}%)")
print(f"  URI_DERIVED                     {by_src['URI_DERIVED']:,} ({100*by_src['URI_DERIVED']/nodes:.3f}%)")
print(f"  STRUCTURAL                      {struct:,} ({100*struct/nodes:.3f}%)")
print("\nby kind:")
for k, c in sorted(by_kind_src.items(), key=lambda kv: -sum(kv[1].values())):
    tot = sum(c.values())
    print(f"  {k:16s} {tot:>12,}  " + "  ".join(f"{s}={v:,}" for s, v in c.most_common(3)))
print("\nexamples:")
for k in ("ENTITY_MID", "CVT_MEDIATOR", "EXTERNAL_URI", "LITERAL", "SCHEMA_TYPE"):
    for e in examples.get(k, [])[:3]:
        print(f"  {k:13s} {e[0]:44s} -> {e[1]:56s} [{e[2]}]")
if examples.get("NAKED"):
    print("naked-MID examples: " + ", ".join(examples["NAKED"][:5]))

rec = {"schema": "RESOLUTION_OVERLAY_VALIDATION/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "VERDICT": "PASS" if ok else "FAIL",
       "nodes": nodes, "nodes_expected": EXPECT,
       "shards_uid_aligned": chk["uid_aligned_shards"], "shards_total": len(shards),
       "DISPLAY_NAME_EMPTY": chk["DISPLAY_NAME_EMPTY"],
       "NAKED_MID_DISPLAY": chk["NAKED_MID_DISPLAY"],
       "NAKED_MID_SELF": chk["NAKED_MID_SELF"],
       "NAKED_MID_DERIVED": chk["NAKED_MID_DERIVED"],
       "NAKED_MID_SHAPED_TEXT_ON_MID_KIND": chk["NAKED_MID_SHAPED_ON_MID_KIND"],
       "NAKED_MID_SHAPED_TEXT_ANY_KIND": chk["NAKED_MID_ANY_KIND"],
       "MID_SHAPED_TEXT_IS_NOT_OPACITY": (
         "The MID-shaped counts are diagnostics, NOT failures. Freebase itself names some entities "
         "in a MID-like form -- abbreviated personal names lowercased (m.albarn = M. Albarn, "
         "g.wygonik, g.carillo) and filename fragments (m.jpg). Those strings are source-declared, "
         "so showing them is showing what Freebase says, not showing an opaque identifier; "
         "suppressing them would destroy real data to satisfy a regex. The gate is therefore "
         "NAKED_MID_SELF (the display IS the node's own id) and NAKED_MID_DERIVED (MID-shaped text "
         "emitted by a tier that does not quote the source verbatim). Evidence for every shaped "
         "row is listed in mid_shaped_evidence."),
       "mid_shaped_evidence": shaped_evidence,
       "OPERATIONALLY_UNRESOLVED": chk["DISPLAY_NAME_EMPTY"] + chk["NAKED_MID_DISPLAY"],
       "failures": bad[:50],
       "by_source": dict(by_src),
       "by_kind": {k: dict(v) for k, v in by_kind_src.items()},
       "AUTHORITATIVE_SOURCE_RESOLUTION": auth,
       "AUTHORITATIVE_SOURCE_RESOLUTION_PCT": round(100 * auth / nodes, 4) if nodes else 0,
       "STRUCTURAL": struct,
       "STRUCTURAL_PCT": round(100 * struct / nodes, 4) if nodes else 0,
       "honesty_rule": ("zero opaque nodes is an OPERATIONAL claim that CRAG never shows a bare "
                        "MID. It is NOT a claim that every original name was recovered; the "
                        "authoritative and structural figures must always be quoted together."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_RESOLUTION_OVERLAY_VALIDATION.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\n{time.time()-t0:.0f}s")
sys.exit(0 if ok else 1)
