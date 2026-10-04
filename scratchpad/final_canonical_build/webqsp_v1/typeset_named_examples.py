"""The named exceptions of the near-zero type sets: what ARE the 5 named {type.content_import} nodes?

    PYTHONHASHSEED=0 python .../typeset_named_examples.py

WHY
  typeset_rate.py found that the big infrastructure sets are near zero, not zero: {common.document}
  has 17,742 named instances in 5.52M, {type.content_import} 5 in 1.93M, {type.content} 28 in 1.40M,
  {type.permission} 22 in 221K. Whether NEAR_ZERO can ever be admitted as a nameless grade depends on
  what those exceptions are -- a real title on a document is one thing, the literal string "null" or a
  test topic is another. This pass lists them, so the decision is made on the evidence and recorded.

HOW
  Target sets: every set with 0 < named <= 200 and total >= 50,000, plus {common.document}. A second
  bucketed pass over type.parquet (same signature as the census) collects the node_uids of NAMED
  nodes (not in the residue) whose signature is a target; then name.parquet (68.4M rows) is scanned
  once, hashing each subject, to attach the names. For {common.document} a deterministic sample of
  300 is kept.

OUTPUT (append-only)
  _acquisition/_typeset_near_zero_exceptions.json   per set: total, named, the names (or a sample),
                                                     and how many of the names are placeholders
"""
import sys, io, os, re, json, time, shutil, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
TMP = f"{ACQ}/_typeset_buckets_ex"
NB = 16
t0 = time.time()

U = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
ts = pq.read_table(f"{ACQ}/_typeset_named_rate.parquet")
T = {r["types"]: r for r in ts.to_pylist()}
targets = {k: v for k, v in T.items() if 0 < v["named"] <= 200 and v["total"] >= 50000}
targets["common.document"] = T["common.document"]
print(f"target sets: {len(targets)}", flush=True)
for k, v in sorted(targets.items(), key=lambda kv: -kv[1]["total"])[:30]:
    print(f"  named {v['named']:>6,} / {v['total']:>10,}  {k[:110]}")

# ------------------------------------------------------------------ pass 1 (same as the census)
f = pq.ParquetFile(f"{V3}/canonical/metadata/type.parquet")
NRG = f.metadata.num_row_groups
type_id = {}
shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
fh = [open(f"{TMP}/b{b:02d}.bin", "wb") for b in range(NB)]
for g in range(NRG):
    rg = f.read_row_group(g)
    d = pc.dictionary_encode(rg.column("subject").combine_chunks())
    suid = np.fromiter(map(hash, d.dictionary.to_pylist()), np.int64, len(d.dictionary))
    uid = suid[d.indices.to_numpy()]
    ty = pc.dictionary_encode(rg.column("type").combine_chunks())
    tmap = np.fromiter((type_id.setdefault(p, len(type_id)) for p in ty.dictionary.to_pylist()),
                       np.int64, len(ty.dictionary))
    tid = tmap[ty.indices.to_numpy()]
    b = (uid & (NB - 1)).astype(np.int64)
    o = np.argsort(b, kind="stable")
    cuts = np.searchsorted(b[o], np.arange(NB + 1))
    rec = np.empty(len(uid), dtype=[("u", np.int64), ("t", np.int32)])
    rec["u"], rec["t"] = uid[o], tid[o]
    for k in range(NB):
        if cuts[k + 1] > cuts[k]:
            fh[k].write(rec[cuts[k]:cuts[k + 1]].tobytes())
    if g % 100 == 0:
        print(f"  pass1 rg {g}/{NRG} ({time.time()-t0:.0f}s)", flush=True)
for h in fh:
    h.close()
rng = np.random.default_rng(20260907)
type_rnd = rng.integers(1, 2**63, size=len(type_id), dtype=np.uint64)
# target signatures, recomputed the census's way from the type ids of this run
tsig = {}
for k in targets:
    ids = [type_id[p] for p in k.split("|") if p in type_id]
    if len(ids) == len(k.split("|")):
        tsig[int(type_rnd[np.array(ids)].sum(dtype=np.uint64))] = k
tsig_arr = np.array(sorted(tsig), np.uint64)

# ------------------------------------------------------------------ pass 2: named nodes in target sets
found = collections.defaultdict(list)          # types -> [node_uid]
for k in range(NB):
    rec = np.fromfile(f"{TMP}/b{k:02d}.bin", dtype=[("u", np.int64), ("t", np.int32)])
    o = np.lexsort((rec["t"], rec["u"]))
    su, st = rec["u"][o], rec["t"][o].astype(np.int64)
    del rec, o
    keep = np.r_[True, (su[1:] != su[:-1]) | (st[1:] != st[:-1])]
    su, st = su[keep], st[keep]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    sigs = np.add.reduceat(type_rnd[st], starts).astype(np.uint64)
    uids = su[starts]
    pos = np.searchsorted(tsig_arr, sigs); pos[pos >= len(tsig_arr)] = 0
    hit = tsig_arr[pos] == sigs
    pu = np.searchsorted(U, uids); pu[pu >= len(U)] = 0
    is_named = U[pu] != uids
    for s, u in zip(sigs[hit & is_named].tolist(), uids[hit & is_named].tolist()):
        found[tsig[s]].append(u)
    print(f"  pass2 bucket {k}/{NB} named-in-target so far {sum(len(v) for v in found.values()):,} "
          f"({time.time()-t0:.0f}s)", flush=True)
shutil.rmtree(TMP, ignore_errors=True)
want = set()
for k, v in found.items():
    v.sort()
    if len(v) > 300:
        v[:] = [v[i] for i in np.linspace(0, len(v) - 1, 300).astype(int)]
    want |= set(v)
want_arr = np.array(sorted(want), np.int64)
print(f"named nodes to look up: {len(want_arr):,}", flush=True)

# ------------------------------------------------------------------ names, one scan of name.parquet
names = collections.defaultdict(list)
nf = pq.ParquetFile(f"{V3}/canonical/metadata/name.parquet")
for g in range(nf.metadata.num_row_groups):
    rg = nf.read_row_group(g)
    d = pc.dictionary_encode(rg.column("subject").combine_chunks())
    dh = np.fromiter(map(hash, d.dictionary.to_pylist()), np.int64, len(d.dictionary))
    h = dh[d.indices.to_numpy()]
    pos = np.searchsorted(want_arr, h); pos[pos >= len(want_arr)] = 0
    m = want_arr[pos] == h
    if m.any():
        idx = np.flatnonzero(m)
        subj = rg.column("subject").take(pa.array(idx)).to_pylist()
        lex = rg.column("lexical").take(pa.array(idx)).to_pylist()
        lang = rg.column("lang").take(pa.array(idx)).to_pylist()
        for a, b, c, hh in zip(subj, lex, lang, h[idx].tolist()):
            names[hh].append((a, b, c))
    if g % 50 == 0:
        print(f"  names rg {g}/{nf.metadata.num_row_groups} matched {len(names):,} ({time.time()-t0:.0f}s)", flush=True)

PLACEHOLDER = re.compile(r"^(null|none|n/a|na|unknown|untitled|test.*|tmp.*|temp.*|xxx+|asdf.*|\?+|-+|\.+|\d+)$", re.I)
out = {}
for k, v in sorted(found.items(), key=lambda kv: -targets[kv[0]]["total"]):
    rows = []
    for u in v:
        for subj, lex, lang in names.get(u, []):
            rows.append({"node_id": subj, "name": lex, "lang": lang})
    ph = sum(1 for r in rows if PLACEHOLDER.match((r["name"] or "").strip()))
    out[k] = {"graph_total": targets[k]["total"], "graph_named": targets[k]["named"],
              "listed": len(rows), "placeholder_like": ph,
              "lang_dist": dict(collections.Counter(r["lang"] for r in rows).most_common(6)),
              "examples": rows[:60]}
rec = {"schema": "TYPESET_NEAR_ZERO_EXCEPTIONS/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "target_rule": "0 < named <= 200 and total >= 50,000, plus common.document (sample of 300)",
       "sets": out, "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{ACQ}/_typeset_near_zero_exceptions.json", "w", encoding="utf-8") as fo:
    json.dump(rec, fo, indent=1, ensure_ascii=False)
for k, v in out.items():
    print(f"\n{k[:100]}  named {v['graph_named']:,}/{v['graph_total']:,}  listed {v['listed']}  placeholder-like {v['placeholder_like']}  {v['lang_dist']}")
    for r in v["examples"][:12]:
        print(f"    {r['node_id']:<14} {r['lang']:<4} {r['name']!r}")
print(f"{time.time()-t0:.0f}s")
