"""Cascade step 1: IDIR object_names against every node we could not name from the IA snapshot.

IDIR's source population is NOT identical to the Internet Archive snapshot we built from -- the
oracle comparison put the overlap at 95.662%, symmetric. That was a provenance nuisance; for NAME
RECOVERY it is an asset, because names IDIR holds for MIDs our snapshot lost are genuinely new
evidence.

Recovered names are labelled IDIR_FREEBASE_NAME, never FREEBASE_CURRENT_EXACT: the two sources
differ, so the provenance must say which one attested the name.

Streamed straight out of the zip through 7-Zip's stdout -- no 3.73 GB CSV is ever written, and the
pipe is driven by Popen rather than a shell pipeline because MSYS throttles bulk shell pipes ~20x.

THE LOOKUP IS BATCHED, NOT PER-ROW
  A first version called np.searchsorted once per input line against the 69.8M-element uid array.
  A scalar searchsorted pays numpy's whole dispatch cost per call, which measured ~78 us/row and
  put throughput at ~13k rows/s -- slower than the 7-Zip decompressor feeding it. Rows are now
  buffered and tested one vector at a time, and the label is parsed only for rows that actually
  hit, so ~99% of lines never pay for the string cleanup at all.

Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, json, glob, time, subprocess, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
OVL = f"{V3}/overlay_v1"
ZIP = f"{V3}/_acquisition/idir/idirlab-freebases.zip"
ENTRY = "idirlab-freebases/Metadata/object_names.csv"
SEVENZ = r"C:\Program Files\7-Zip\7z.exe"
CACHE = f"{V3}/_acquisition/_unresolved_population.npz"
STRUCT = {"STRUCTURAL_INFERRED", "STRUCTURAL_FALLBACK"}
KIND = ["ENTITY_MID", "CVT_MEDIATOR", "OTHER"]
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

# ---------------------------------------------------------------- 1. the unresolved population
# Cached: it is a pure function of two frozen inputs, so rebuilding it on every restart of a
# recovery step is 289s of pointless re-reading of 524 parquet files.
if os.path.exists(CACHE):
    z = np.load(CACHE)
    U, KC = z["U"], z["KC"]
    print(f"unresolved population from cache  ({time.time()-t0:.0f}s)", flush=True)
else:
    uids, kinds = [], []
    for fp in sorted(glob.glob(f"{OVL}/*.parquet")):
        base = os.path.basename(fp)
        ov = pq.read_table(fp, columns=["node_uid", "display_name_source"])
        s = ov["display_name_source"].combine_chunks()
        d = s if isinstance(s.type, pa.DictionaryType) else s.dictionary_encode()
        want = np.array([v in STRUCT for v in d.dictionary.to_pylist()], dtype=bool)
        m = want[d.indices.to_numpy(zero_copy_only=False)]
        if not m.any():
            continue
        kd = pq.read_table(f"{V3}/canonical/nodes/{base}", columns=["kind"])["kind"].combine_chunks()
        if isinstance(kd.type, pa.DictionaryType):
            kd = kd.cast(pa.string())
        kv = kd.to_numpy(zero_copy_only=False)[m]
        kc = np.full(len(kv), 2, dtype=np.uint8)
        kc[kv == "ENTITY_MID"] = 0
        kc[kv == "CVT_MEDIATOR"] = 1
        uids.append(ov["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)[m])
        kinds.append(kc)
        del ov, s, d, kd
    U = np.concatenate(uids); KC = np.concatenate(kinds)
    del uids, kinds
    o = np.argsort(U, kind="stable"); U, KC = U[o], KC[o]
    del o
    np.savez(CACHE, U=U, KC=KC)
print(f"unresolved nodes: {len(U):,}  "
      f"(ENTITY_MID {int((KC==0).sum()):,}, CVT {int((KC==1).sum()):,}, other {int((KC==2).sum()):,})"
      f"  ({time.time()-t0:.0f}s)", flush=True)
NU = len(U)

# ---------------------------------------------------------------- 2. stream IDIR object_names
p = subprocess.Popen([SEVENZ, "x", "-so", ZIP, ENTRY],
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22)
rows = mids = matched = 0
# Bounded memory: matches are flushed to parquet parts and deduped at the end, rather than held in
# a dict that would reach ~3 GB while the semantic pass already holds ~3 GB.
PARTS = f"{V3}/_acquisition/_idir_parts"
os.makedirs(PARTS, exist_ok=True)
for old in glob.glob(f"{PARTS}/*.parquet"):
    os.remove(old)
bu, bn, bi, br = [], [], [], []
part = 0
FLUSH = 2_000_000
PROBE = 1_000_000          # rows tested per vectorised searchsorted
lang_hit = collections.Counter()
LANGRANK = {"en": 0}
QUOTE = '"'


def flush():
    global part, bu, bn, bi, br
    if not bu:
        return
    pq.write_table(pa.table({"node_uid": pa.array(bu, pa.int64()),
                             "node_id": pa.array(bi, pa.string()),
                             "display_name": pa.array(bn, pa.string()),
                             "rank": pa.array(br, pa.int8())}),
                   f"{PARTS}/p{part:04d}.parquet", compression="zstd")
    part += 1
    bu, bn, bi, br = [], [], [], []


def probe(sub, rest):
    """Test a whole buffer of subjects at once; parse labels only for the ones that hit."""
    global matched
    nid = [s[1:].replace("/", ".", 1) for s in sub]
    u = np.fromiter(map(hash, nid), dtype=np.int64, count=len(nid))
    j = np.searchsorted(U, u)
    np.clip(j, 0, NU - 1, out=j)
    idx = np.flatnonzero(U[j] == u)
    if not idx.size:
        return
    for i in idx.tolist():
        # SUBJECT,/type/object/name,"LABEL"@lang -- the label may itself contain commas, so the
        # split is bounded and the language suffix is taken off the tail.
        _, _, obj = rest[i].partition(",")
        if not obj:
            continue
        lit, _, lang = obj.rpartition("@")
        if not lit:
            lit, lang = obj, ""
        lit = lit.strip().strip(QUOTE).strip()
        if not lit:
            continue
        matched += 1
        lang_hit[lang] += 1
        bu.append(int(u[i])); bn.append(lit); bi.append(nid[i])
        br.append(LANGRANK.get(lang, 1))
    if len(bu) >= FLUSH:
        flush()


b_sub, b_rest = [], []
for raw in io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace", newline=""):
    rows += 1
    a, _, rest = raw.rstrip("\r\n").partition(",")
    if not a.startswith("/m/") and not a.startswith("/g/"):
        continue
    mids += 1
    b_sub.append(a); b_rest.append(rest)
    if len(b_sub) >= PROBE:
        probe(b_sub, b_rest)
        b_sub, b_rest = [], []
        print(f"  {rows:,} rows  mid-rows={mids:,}  matched={matched:,} "
              f"({time.time()-t0:.0f}s)", flush=True)
if b_sub:
    probe(b_sub, b_rest)
flush()
p.stdout.close()
p.wait()
print(f"object_names rows: {rows:,}  MID rows: {mids:,}  matching name-rows: {matched:,}"
      f"  ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- 3. dedupe, classify, write
distinct = 0
by_kind = {}
if matched:
    t = pq.read_table(sorted(glob.glob(f"{PARTS}/*.parquet")))
    # one name per node: English wins, then lexicographically smallest -- deterministic either way
    t = t.sort_by([("node_uid", "ascending"), ("rank", "ascending"),
                   ("display_name", "ascending")])
    mu = t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
    keep = np.empty(len(mu), dtype=bool)
    keep[0] = True
    keep[1:] = mu[1:] != mu[:-1]
    t = t.filter(pa.array(keep))
    mu = t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
    distinct = len(mu)
    kc = KC[np.searchsorted(U, mu)]
    out = pa.table({"node_uid": t["node_uid"], "node_id": t["node_id"],
                    "display_name": t["display_name"],
                    "kind": pa.array([KIND[c] for c in kc.tolist()]).dictionary_encode()})
    pq.write_table(out, f"{V3}/_acquisition/idir_names.parquet", compression="zstd")
    by_kind = {KIND[i]: int((kc == i).sum()) for i in range(3)}
    del t, out
    for old in glob.glob(f"{PARTS}/*.parquet"):
        os.remove(old)
    os.rmdir(PARTS)

n_ent = int((KC == 0).sum())
rec = {"schema": "IDIR_NAME_RECOVERY/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. 301,977,131 nodes / "
                       "2,062,430,072 edges unchanged; OVERLAY_V1 manifest hash unchanged."),
       "QUESTION": ("does IDIR's object_names table hold real Freebase names for nodes our "
                    "Internet Archive snapshot could not name?"),
       "WHY_IDIR_CAN_HELP": ("the IDIR source population is not identical to the IA snapshot "
                             "(oracle overlap 95.662%, symmetric), so names it holds for MIDs our "
                             "snapshot lost are new evidence rather than a duplicate of what we have"),
       "source": {"artifact": "idirlab-freebases.zip", "entry": ENTRY,
                  "declared_uncompressed_bytes": 4006052812},
       "object_names_rows": rows,
       "object_names_mid_rows": mids,
       "unresolved_nodes_probed": int(NU),
       "unresolved_entity_mids": n_ent,
       "matching_name_rows": matched,
       "recovered": distinct,
       "recovered_by_kind": by_kind,
       "entity_mid_recovery_pct": round(100 * by_kind.get("ENTITY_MID", 0) / n_ent, 4) if n_ent else 0,
       "note_rows_vs_nodes": ("matching_name_rows counts rows (a node may have many languages); "
                              "recovered counts distinct nodes."),
       "languages": dict(lang_hit.most_common(20)),
       "PROVENANCE_RULE": ("recovered names are IDIR_FREEBASE_NAME, NOT FREEBASE_CURRENT_EXACT. "
                           "IDIR and the IA snapshot are different source populations; the tier "
                           "must record which one attested the name."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_IDIR_NAME_RECOVERY.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\nrecovered {distinct:,} distinct nodes from {matched:,} matching name-rows")
for k, v in by_kind.items():
    print(f"   {v:>12,}  {k}")
print(f"ENTITY_MID recovery: {rec['entity_mid_recovery_pct']}% of {n_ent:,}")
print(f"{time.time()-t0:.0f}s")
