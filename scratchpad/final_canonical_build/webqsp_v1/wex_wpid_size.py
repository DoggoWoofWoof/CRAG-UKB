"""WEX wpid: guid -> Wikipedia page id.  Downloaded and verified 2026-09-07, never joined.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/wex_wpid_size.py

WHY SIZE IT BEFORE FETCHING ANYTHING
  This is the second unconverted exact source found on disk in one sitting -- fb2w was the first,
  and converting that one doubled the Wikidata route.  The decision here is whether to spend a
  Wikipedia crawl, so the question that has to be answered first is not "how many residue nodes
  carry a page id" but "how many carry one AND are still unnamed after the current cascade".  The
  first number justifies nothing on its own; a page id on a node we already name adds a corroborating
  surface, not an identity.

  No network in this pass.  It writes the mapping and the counts, and stops.

WHAT A PAGE ID IS WORTH, AND WHAT IT IS NOT
  A Wikipedia page id resolves by ?curid=<id> to exactly one article.  That is an EXACT identifier
  lookup -- no name matching, no disambiguation, nothing to get wrong -- which is why it is in scope
  while generic web recovery is not.  But the string it yields is a Wikipedia article title, not a
  Freebase /type/object/name, so it could never be is_original_name true.

  It also does NOT revive the refuted WIKIPEDIA_TITLE tier.  That refutation was specific and stands:
  ~15 residue nodes hold any Wikipedia title key IN THE CURRENT GRAPH (V3_KEY_NAMESPACE_CENSUS).
  This is a different artifact -- a 2010 WEX table keyed by guid -- so it is not evidence against
  that finding and that finding is not evidence against this.  Whether it earns a tier depends
  entirely on the unnamed count this pass measures.
"""
import sys, io, os, re, json, time, bz2, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
SRC = f"{ACQ}/raw/wex_wpid_freebase-wex-2010-07-05-freebase_wpid.tsv.bz2"
PREFIX = "9202a8c04000641f"
ALPHA = "0123456789bcdfghjklmnpqrstvwxyz_"
t0 = time.time()

UNIV = np.load(f"{V3}/canonical/_node_uids_sorted.npy", mmap_mode="r")
RES = np.load(f"{ACQ}/_unresolved_population.npz")["U"]


def guid_to_mid(g):
    """The proven-injective decode. Verified over 3,038,202 guids earlier in this campaign."""
    n = int(g[16:], 16) - 0x8000000000000000
    if n < 0:
        return None
    s = ""
    while n:
        s = ALPHA[n & 31] + s
        n >>= 5
    return "m.0" + s


rows, n_lines, bad = [], 0, 0
with bz2.open(SRC, "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        n_lines += 1
        p = line.rstrip("\n").split("\t")
        if len(p) != 2 or not p[0].startswith(PREFIX) or not p[1].isdigit():
            bad += 1
            continue
        m = guid_to_mid(p[0])
        if m:
            rows.append((m, int(p[1])))
        else:
            bad += 1
print(f"wex wpid: {n_lines:,} lines, {len(rows):,} decoded, {bad:,} unparsed "
      f"({time.time()-t0:.0f}s)", flush=True)
if not rows:
    sys.exit("POSITIVE CONTROL FAILED: decoded 0 guids. Refusing to write a zero.")

seen = {}
for m, w in rows:
    seen.setdefault(m, w)
mids = sorted(seen)
uids = np.fromiter((hash(m) for m in mids), np.int64, len(mids))
o = np.argsort(uids, kind="stable")
su = uids[o]
i = np.searchsorted(UNIV, su)
ing = int((np.take(UNIV, np.minimum(i, len(UNIV) - 1)) == su).sum())
j = np.searchsorted(RES, su)
hm = np.take(RES, np.minimum(j, len(RES) - 1)) == su
sel = o[hm]
print(f"distinct mids {len(mids):,}   in graph {ing:,}   on residue {int(hm.sum()):,}  "
      f"({time.time()-t0:.0f}s)", flush=True)
if ing == 0:
    sys.exit("POSITIVE CONTROL FAILED: 0 of the decoded mids are in the 302M universe. The decode "
             "or the join key is wrong; this is not a result.")

hit_mids = [mids[k] for k in sel]
hit_uids = uids[sel]
hit_wpid = [seen[m] for m in hit_mids]

# THE NUMBER THAT ACTUALLY DECIDES THIS
named = set(pq.read_table(f"{ACQ}/cascade_names.parquet",
                          columns=["node_uid"])["node_uid"].to_pylist())
unnamed = [k for k in range(len(hit_mids)) if int(hit_uids[k]) not in named]
print(f"of those, STILL UNNAMED after the current cascade: {len(unnamed):,}  "
      f"(already named: {len(hit_mids)-len(unnamed):,})", flush=True)

os.makedirs(f"{ACQ}/wex_wpid", exist_ok=True)
pq.write_table(pa.table({
    "node_uid": pa.array([int(hit_uids[k]) for k in range(len(hit_mids))], pa.int64()),
    "node_id": pa.array(hit_mids),
    "wikipedia_en_page_id": pa.array(hit_wpid, pa.int64()),
    "already_named_by_cascade": pa.array([int(hit_uids[k]) in named
                                          for k in range(len(hit_mids))], pa.bool_())}),
    f"{ACQ}/wex_wpid/part_00000.parquet", compression="zstd")

rec = {
 "RECORD": "V3_WEX_WPID_SIZING",
 "WHAT": "the 2010 WEX guid->Wikipedia-page-id table, decoded and joined. Downloaded and checksum-"
         "verified 2026-09-07 and never joined until now. No network was used in this pass.",
 "JOIN_CONTROL": {"SOURCE_ROWS": n_lines, "VALID_ID_ROWS": len(mids),
                  "IN_GRAPH_POSITIVE_CONTROL_N": ing, "RESIDUE_HIT_N": int(hm.sum()),
                  "JOIN_NORMALIZATION": "guid -> mid by base32 decode of int(guid[16:],16) - "
                                        "0x8000000000000000, alphabet 0123456789bcdfghjklmnpqrstvwxyz_",
                  "unparsed_lines": bad,
                  "DIRECTION": "INBOUND", "CONTROL_CLASS": "NON_CIRCULAR_RAW_SIDE"},
 "THE_DECIDING_NUMBER": {
    "residue_nodes_with_a_page_id": int(hm.sum()),
    "of_those_STILL_UNNAMED_after_the_current_cascade": len(unnamed),
    "of_those_already_named": len(hit_mids) - len(unnamed),
    "WHY_THIS_AND_NOT_THE_REACH": "a page id on a node we already name is a corroborating surface, "
                                  "not an identity. Only the unnamed count can justify a crawl."},
 "WHAT_A_PAGE_ID_YIELDS": {
    "resolution": "?curid=<id> resolves to exactly one article. Exact identifier lookup, no name "
                  "matching, which is why it is in scope while generic web recovery is not.",
    "is_original_name": False,
    "REASON": "the string is a Wikipedia article title, not a Freebase /type/object/name."},
 "DOES_NOT_REVIVE_THE_REFUTED_TIER":
    "V3_KEY_NAMESPACE_CENSUS refuted WIKIPEDIA_TITLE on the grounds that ~15 residue nodes hold any "
    "Wikipedia title key IN THE CURRENT GRAPH. That is a different artifact from a 2010 WEX table "
    "keyed by guid. Neither finding is evidence about the other, and the refutation stands as "
    "written.",
 "elapsed_s": round(time.time() - t0, 1)}
b = json.dumps(rec, indent=1, ensure_ascii=False).encode()
rec["record_sha256"] = hashlib.sha256(b).hexdigest()
io.open(f"{V3}/V3_WEX_WPID_SIZING.json", "w", encoding="utf-8").write(
    json.dumps(rec, indent=1, ensure_ascii=False))
print(f"wrote V3_WEX_WPID_SIZING.json  sha {rec['record_sha256'][:16]}  ({rec['elapsed_s']}s)")
