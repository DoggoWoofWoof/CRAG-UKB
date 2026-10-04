"""Per-node predicate SIGNATURES for the 5,832,113 UNTYPED_CANDIDATE nodes.

    PYTHONHASHSEED=0 python .../untyped_signature.py

WHY A SECOND EDGE PASS
  V3_UNTYPED_CENSUS gave the marginals and they are lopsided: dataworld.gardening_hint.* and
  type.object.permission account for almost every edge the class touches, and 3,057,145 nodes have
  out-degree exactly 2 against 3,052,894 gardening_hint.last_referenced_by out-edges.  That LOOKS
  like three million Freebase housekeeping records sitting in the name hunt.  It cannot be concluded
  from marginals: totals over a class do not tell you which node carries which predicate, and the
  campaign rule is that an artifact is invalid until counts establish otherwise.

  This pass computes, per node, the exact SET of predicates on it, so the claim becomes a count of
  nodes with a given signature rather than an inference from two column totals.

HOW THE SIGNATURE IS EXACT AND STILL FITS IN MEMORY
  The 63 predicates that touch the class most often get one bit each; a node's signature is the OR
  of its predicates' bits, plus one overflow bit set when it touches anything outside the 63.
  A set-OR is order independent and duplicate insensitive, which is exactly what a predicate SET is,
  so identical signatures mean identical predicate sets (up to the overflow bit).  In and out are
  kept separate because direction is what distinguishes "is a gardening hint" from "is referenced by
  one".  Cost: two int64 arrays over 5.83M nodes.

WHAT IT DECIDES
  A node whose entire signature is inside the bookkeeping set never had a name to lose: it is a
  permission record or a gardening hint, not a topic.  A node carrying any content predicate is a
  different case and stays in the hunt.  Both are counted here; neither is applied.  Reclassifying
  is a new overlay record with its own hash, never an edit of the frozen one.

OUTPUT (append-only)
  _acquisition/_untyped_signatures.parquet   node_uid, sig_in, sig_out, deg_in, deg_out
  V3_UNTYPED_SIGNATURE.json                  the signature frequency table, decoded to names
"""
import sys, io, os, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()
NBIT = 63

# Freebase's own bookkeeping. A node touching only these is a maintenance record, not a topic.
BOOKKEEPING = {
    "type.object.permission",
    "dataworld.gardening_hint.last_referenced_by",
    "dataworld.gardening_hint.replaced_by",
}

sk = pq.read_table(f"{ACQ}/semantic_kind_v2_1.parquet", columns=["node_uid", "recovery_class"])
rc = sk["recovery_class"]
rc = pc.cast(rc, pa.string()) if pa.types.is_dictionary(rc.type) else rc
U = np.sort(pc.filter(sk["node_uid"], pc.equal(rc, "UNTYPED_CANDIDATE")).to_numpy())
del sk, rc
print(f"UNTYPED_CANDIDATE nodes: {len(U):,} ({time.time()-t0:.0f}s)", flush=True)

# the 63 bit-worthy predicates come from the census that already ran, so this pass does not
# have to guess which ones matter
cen = pq.read_table(f"{ACQ}/_untyped_predicate_census.parquet")
tot = (cen["n_edges_into_untyped"].to_numpy() + cen["n_edges_out_of_untyped"].to_numpy())
names = cen["relation"].to_pylist()
order = np.argsort(-tot)[:NBIT]
BITREL = [names[i] for i in order.tolist()]
BIT = {r: 1 << i for i, r in enumerate(BITREL)}
OVER = 1 << 63
print(f"bit-assigned predicates: {len(BITREL)} (top by total edges touching the class)", flush=True)

rel_t = pq.read_table(f"{V3}/canonical/relations.parquet", columns=["rel_uid", "relation"])
RU = rel_t["rel_uid"].to_numpy()
RN = rel_t["relation"].to_pylist()
o = np.argsort(RU)
RU = RU[o]
RN = [RN[i] for i in o.tolist()]
NR = len(RU)
# dense rel id -> signature bit (int64 so the OR is a plain numpy op); overflow uses the sign bit
relbit = np.zeros(NR + 1, np.uint64)
for i, r in enumerate(RN):
    relbit[i] = np.uint64(BIT.get(r, OVER))
relbit[NR] = np.uint64(OVER)          # rel_uid not found in relations.parquet

sig_in = np.zeros(len(U), np.uint64)
sig_out = np.zeros(len(U), np.uint64)
deg_in = np.zeros(len(U), np.int64)
deg_out = np.zeros(len(U), np.int64)

files = sorted(glob.glob(f"{V3}/canonical/edges/*.parquet"))
print(f"edge files: {len(files)}", flush=True)
tot_e = 0
for fi, fp in enumerate(files):
    pf = pq.ParquetFile(fp)
    for g in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(g, columns=["src", "rel", "dst"])
        src = rg["src"].to_numpy(); rel = rg["rel"].to_numpy(); dst = rg["dst"].to_numpy()
        tot_e += len(src)
        for arr, sg, dg in ((dst, sig_in, deg_in), (src, sig_out, deg_out)):
            p = np.searchsorted(U, arr); np.clip(p, 0, len(U) - 1, out=p)
            hit = U[p] == arr
            if not hit.any():
                continue
            q = np.searchsorted(RU, rel[hit]); np.clip(q, 0, NR - 1, out=q)
            q = np.where(RU[q] == rel[hit], q, NR)
            ph = p[hit]
            b = relbit[q]
            # np.bitwise_or.at is the unbuffered scatter-OR; duplicates on one node accumulate
            np.bitwise_or.at(sg, ph, b)
            dg += np.bincount(ph, minlength=len(U))
    if fi % 20 == 0:
        print(f"  file {fi}/{len(files)} edges {tot_e:,} ({time.time()-t0:.0f}s)", flush=True)

pq.write_table(pa.table({"node_uid": pa.array(U, pa.int64()),
                         "sig_in": pa.array(sig_in.astype(np.int64), pa.int64()),
                         "sig_out": pa.array(sig_out.astype(np.int64), pa.int64()),
                         "deg_in": pa.array(deg_in, pa.int64()),
                         "deg_out": pa.array(deg_out, pa.int64())}),
                f"{ACQ}/_untyped_signatures.parquet", compression="zstd")


def decode(s):
    s = int(s)
    out = [BITREL[i] for i in range(NBIT) if s & (1 << i)]
    if s & OVER:
        out.append("<other>")
    return out


book_bits = 0
for r in BOOKKEEPING:
    book_bits |= BIT.get(r, 0)
missing = [r for r in BOOKKEEPING if r not in BIT]
both = (sig_in | sig_out).astype(np.int64)
only_book = (both != 0) & ((both & ~np.int64(book_bits)) == 0)
noedge = (deg_in == 0) & (deg_out == 0)

cnt = collections.Counter()
for a, b in zip(sig_in.tolist(), sig_out.tolist()):
    cnt[(a, b)] += 1
top = cnt.most_common(40)

rec = {"schema": "UNTYPED_SIGNATURE/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "new record; frozen graph read-only, frozen overlay not reclassified.",
       "population": int(len(U)),
       "edges_scanned": int(tot_e),
       "bit_assigned_predicates": NBIT,
       "bookkeeping_set": sorted(BOOKKEEPING),
       "bookkeeping_predicates_not_seen_on_this_class": missing,
       "nodes_touching_ONLY_bookkeeping_predicates": int(only_book.sum()),
       "nodes_with_no_edge_at_all": int(noedge.sum()),
       "WHAT_THE_BOOKKEEPING_COUNT_MEANS": "these nodes carry no predicate other than Freebase's own "
                                           "maintenance properties. They are permission records and "
                                           "gardening hints, not topics that lost a name. Proposed "
                                           "for a terminal bucket; NOT applied here.",
       "top_40_signatures": [
           {"n_nodes": n, "in": decode(a), "out": decode(b)} for (a, b), n in top],
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_UNTYPED_SIGNATURE.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in rec.items() if k != "top_40_signatures"}, indent=1))
print("\ntop signatures:")
for d in rec["top_40_signatures"][:20]:
    print(f"  {d['n_nodes']:>10,}  in={d['in']}  out={d['out']}")
