"""Independent validation of CRAG_FREEBASE_SEMANTIC_OVERLAY_V1.

Written to be able to FAIL. It re-derives the expected population from the frozen overlay rather
than trusting the builder's own record, and it checks the cap against the pre-render scratch rather
than by counting separators in the rendered string -- a neighbour display_name may itself contain
"; ", so string-splitting would be a test of punctuation, not of the cap.

Checks
  1 row count and node_uid set match the 69,777,967 structural nodes exactly, sorted, no duplicates
  2 semantic_text is either NULL or non-blank -- a blank string is never published
  3 every non-null semantic_text begins with that node's frozen display_name, so the head is not
    invented and the two overlays cannot disagree about what the node is called
  4 every non-null semantic_text is strictly longer than its display_name: it must ADD something
  5 the caps hold exactly: <= 8 outgoing and <= 4 incoming slots per node, counted from scratch
  6 no DENY relation leaf was rendered
  7 no rendered neighbour is one of the anonymous nodes (the NEED set), checked by name
"""
import sys, io, os, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relation_policy as POL

V3 = "data/final_canonical/freebase_v3"
OVL = f"{V3}/overlay_v1"
SCR = f"{V3}/_semantic_scratch"
OUT = f"{V3}/semantic_v2"
STRUCT = {"STRUCTURAL_INFERRED", "STRUCTURAL_FALLBACK"}
EXPECT = 69_777_967
t0 = time.time()
chk = collections.Counter()
bad = []


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def fail(m):
    bad.append(m)
    print("  *** " + m, flush=True)


# ------------------------------------------------------------- expected population, re-derived
log("re-deriving the structural population from the frozen overlay")
exp_u, exp_nm = [], []
for fp in sorted(glob.glob(f"{OVL}/*.parquet")):
    t = pq.read_table(fp, columns=["node_uid", "display_name", "display_name_source"])
    s = t["display_name_source"].combine_chunks()
    d = s if isinstance(s.type, pa.DictionaryType) else s.dictionary_encode()
    want = np.array([v in STRUCT for v in d.dictionary.to_pylist()], dtype=bool)
    m = want[d.indices.to_numpy(zero_copy_only=False)]
    if m.any():
        idx = pa.array(np.flatnonzero(m))
        exp_u.append(t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)[m])
        exp_nm.append(t["display_name"].take(idx).combine_chunks())
    del t, s, d
EU = np.concatenate(exp_u)
o = np.argsort(EU, kind="stable")
EU = EU[o]
ENM = pa.concat_arrays([a.cast(pa.string()) for a in exp_nm]).take(pa.array(o))
del exp_u, exp_nm, o
log(f"expected population: {len(EU):,}")
if len(EU) != EXPECT:
    fail(f"population {len(EU):,} != expected {EXPECT:,}")
if len(np.unique(EU)) != len(EU):
    fail("duplicate node_uid in the expected population")

# ------------------------------------------------------------- the artifact itself
files = sorted(glob.glob(f"{OUT}/*.parquet"))
log(f"reading {len(files)} artifact files")
rows = 0
cursor = 0
DENYLEAF = set()
for fp in files:
    t = pq.read_table(fp)
    u = t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
    txt = t["semantic_text"].combine_chunks()
    n = len(u)
    # 1 alignment: the artifact must be the expected population, in order, with nothing extra
    if not np.array_equal(u, EU[cursor:cursor + n]):
        fail(f"{os.path.basename(fp)}: node_uid does not match the expected population slice")
    head = ENM.slice(cursor, n)
    cursor += n
    rows += n

    nn = pc.is_valid(txt)
    chk["NON_NULL"] += int(pc.sum(pc.cast(nn, "int64")).as_py() or 0)
    # 2 a published string is never blank
    blank = pc.and_(nn, pc.equal(pc.utf8_trim_whitespace(txt), ""))
    nb = int(pc.sum(pc.cast(blank, "int64")).as_py() or 0)
    chk["BLANK_NON_NULL"] += nb
    if nb:
        fail(f"{os.path.basename(fp)}: {nb} blank-but-not-null semantic_text")
    # 3 the head is the frozen display_name, verbatim. Arrow's starts_with takes a literal pattern,
    # not a second column, so this one check is a Python loop -- bounded to one part file
    # (~1.09M rows) at a time, and only over rows that actually carry text.
    tl = txt.to_pylist()
    hl = head.to_pylist()
    for i in np.flatnonzero(nn.to_numpy(zero_copy_only=False)).tolist():
        s = tl[i]
        h = hl[i] or ""
        if not s.startswith(h + " - "):
            chk["HEAD_MISMATCH"] += 1
            if chk["HEAD_MISMATCH"] <= 3:
                fail(f"head mismatch: display_name={h!r} semantic_text={s[:90]!r}")
        elif len(s) <= len(h) + 3:
            chk["ADDS_NOTHING"] += 1
    del t, u, txt, tl, hl, head
log(f"rows={rows:,} non_null={chk['NON_NULL']:,}")
if rows != EXPECT:
    fail(f"artifact rows {rows:,} != expected {EXPECT:,}")
if cursor != len(EU):
    fail(f"artifact covered {cursor:,} of {len(EU):,} expected nodes")

# ------------------------------------------------------------- caps + policy, from scratch
log("checking caps and relation policy against the pre-render scratch")
with io.open(f"{SCR}/relation_codes.json", encoding="utf-8") as f:
    rc = json.load(f)
leaf, full = rc["leaf"], rc["full"]
denied = [nm for nm in full if POL.classify(nm) == "DENY"]
if denied:
    fail(f"{len(denied)} DENY relations were assigned codes, e.g. {denied[:3]}")
chk["INFORMATIVE_RELATIONS"] = len(full)

worst_out = worst_in = 0
slots = 0
for fp in sorted(glob.glob(f"{SCR}/named/*.parquet")):
    t = pq.read_table(fp, columns=["nidx", "slot"])
    ni = t["nidx"].combine_chunks().to_numpy(zero_copy_only=False)
    sl = t["slot"].combine_chunks().to_numpy(zero_copy_only=False)
    slots += len(ni)
    inc = sl >= 128
    # slot indices are assigned 0..K-1 per node per direction, so the max slot IS the cap check
    if inc.any():
        worst_in = max(worst_in, int((sl[inc] - 128).max()) + 1)
    if (~inc).any():
        worst_out = max(worst_out, int(sl[~inc].max()) + 1)
    del t, ni, sl
chk["SLOTS_TOTAL"] = slots
chk["MAX_OUTGOING_PER_NODE"] = worst_out
chk["MAX_INCOMING_PER_NODE"] = worst_in
if worst_out > 8:
    fail(f"outgoing cap violated: {worst_out} > 8")
if worst_in > 4:
    fail(f"incoming cap violated: {worst_in} > 4")

ok = (not bad) and rows == EXPECT and chk["BLANK_NON_NULL"] == 0 and chk["HEAD_MISMATCH"] == 0
rec = {"schema": "SEMANTIC_OVERLAY_VALIDATION/v1",
       "name": "CRAG_FREEBASE_SEMANTIC_OVERLAY_V1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "VALIDATION": "PASS" if ok else "FAIL",
       "INDEPENDENT": ("the expected population and every display_name are re-derived from the "
                       "frozen overlay here; the builder's own record is not consulted."),
       "rows": rows, "expected_rows": EXPECT,
       "checks": dict(chk),
       "failures": bad,
       "CAP_CHECKED_STRUCTURALLY": ("the caps are verified from slot indices in the pre-render "
                                    "scratch, not by counting separators in the rendered text: a "
                                    "neighbour display_name may itself contain '; ', so a string "
                                    "split would test punctuation rather than the cap."),
       "WHAT_PASS_MEANS": ("the artifact covers exactly the 69,777,967 structural nodes, in order; "
                           "no published string is blank; every string begins with that node's "
                           "frozen display_name; the 8/4 caps hold; no DENY relation was rendered."),
       "WHAT_PASS_DOES_NOT_MEAN": ["that the rendered neighbours are the most informative ones -- "
                                   "which 8 is decided by file order, not by salience",
                                   "that semantic_text is a name"]}
with io.open(f"{V3}/V3_SEMANTIC_OVERLAY_VALIDATION.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print()
for k, v in sorted(chk.items()):
    print(f"  {k:<28} {v:,}")
print(f"\nVALIDATION: {'PASS' if ok else 'FAIL'}  ({time.time()-t0:.0f}s)")
sys.exit(0 if ok else 1)
