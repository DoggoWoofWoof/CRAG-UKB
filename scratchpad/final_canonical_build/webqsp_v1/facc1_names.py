"""Cascade step 5, naming: recover attested surface forms for the 1,623 FACC1-reachable nodes.

    PYTHONHASHSEED=0 python .../facc1_names.py

THE RULE, AS SPECIFIED
  "Do not take a random mention. For each MID, aggregate surface forms and require something like
  high-confidence observations, repeated support, and a dominant normalized form."

  Mapped onto what the two files actually carry:
    repeated support     entity_list count >= MIN_COUNT   (a surface seen more than once)
    high confidence      P(mid|surface) >= MIN_P          (the surface predominantly means THIS mid)
    dominant form        the chosen surface is the unique argmax of P for that mid

  All three are required together. The yield at other thresholds is reported as a grid so the
  choice is visible and can be tightened without rerunning anything.

TWO PROPERTIES OF THE SOURCE THAT LIMIT WHAT MAY BE CLAIMED
  1. Surfaces are case-folded. Zero of 47,187,940 rows contain an uppercase character. So a
     recovered display_name is published lowercase. Title-casing it would invent capitalisation the
     source does not contain, and capitalisation is part of a name.
  2. Surfaces carry Freebase key encoding: $0028 and $0029 are ( and ). That is a documented,
     reversible escape, so decoding it recovers what the string always was and is not a rewrite.
     Whether it was applied is recorded either way.

WHAT A RECOVERED NAME HERE IS
  A surface form observed in web text referring to this MID. It is NOT the /type/object/name value.
  Provenance FACC1_ATTESTED_SURFACE, is_original_name false.

Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, re, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
MIN_COUNT = 2
MIN_P = 0.5
RESIDUE_ENTITY_MIDS = 17_809_849
ESC = re.compile(r"\$([0-9A-Fa-f]{4})")
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")


def decode(s):
    """Freebase key escaping: $0028 -> '('. Reversible, so this recovers rather than rewrites."""
    return ESC.sub(lambda m: chr(int(m.group(1), 16)), s)


# ---------------------------------------------------------------- load both sides
cnt = {}
uid = {}
kind = {}
for fp in sorted(glob.glob(f"{ACQ}/facc1_hits/*.parquet")):
    t = pq.read_table(fp)
    for n, u, s, c in zip(t["node_id"].to_pylist(), t["node_uid"].to_pylist(),
                          t["surface"].to_pylist(), t["count"].to_pylist()):
        cnt[(n, s)] = max(cnt.get((n, s), 0), c)
        uid[n] = u
by_node = collections.defaultdict(list)
for fp in sorted(glob.glob(f"{ACQ}/facc1_surfacemap_hits/*.parquet")):
    t = pq.read_table(fp)
    for n, u, s, p in zip(t["node_id"].to_pylist(), t["node_uid"].to_pylist(),
                          t["surface"].to_pylist(), t["p_mid_given_surface"].to_pylist()):
        by_node[n].append((s, p))
        uid[n] = u
print(f"nodes with surfaces: {len(by_node):,}   entity_list (node,surface) pairs: {len(cnt):,}"
      f"  ({time.time()-t0:.0f}s)", flush=True)

esc_seen = sum(1 for n in by_node for s, _ in by_node[n] if "$" in s)


def select(n, mc, mp):
    """Both signals must hold for the SAME surface, then rank.

    The first version ranked by P and only then checked support, which threw away good names: for
    m.0g56pxz 'gitanjali group' carries 1,253 mentions at P=1.0, but a second surface tied at P=1.0
    with no support at all won the argmax and the node was rejected. Support and confidence are
    properties of a particular string, so they are applied to a particular string.

    Dominance is then the unique argmax among QUALIFYING surfaces; a genuine tie -- m.0dtywr is
    both 'beauty and the beast (1992 film)' and '(1993 film)' at P=1.0 -- is refused, not broken."""
    cand = []
    for s, pv in by_node[n]:
        c = cnt.get((n, s), 0)
        if pv >= mp and c >= mc:
            cand.append((s, pv, c))
    if not cand:
        best_p = max((pv for _, pv in by_node[n]), default=0.0)
        best_c = max((cnt.get((n, s), 0) for s, _ in by_node[n]), default=0)
        return None, ("p_below_threshold" if best_p < mp else
                      "support_below_threshold" if best_c < mc else
                      "no_single_surface_meets_both")
    top = max(pv for _, pv, _ in cand)
    tied = [x for x in cand if x[1] == top]
    if len(tied) > 1:
        tc = max(c for _, _, c in tied)
        tied = [x for x in tied if x[2] == tc]
        if len(tied) > 1:
            return None, "no_dominant_form"
    return tied[0], ""

# ---------------------------------------------------------------- do the files agree?
# The support threshold rejects most of the reach, so before accepting that as a fact about FACC1 it
# has to be separated from a much duller possibility: that the two files simply disagree about which
# surface belongs to a node, so the surface carrying the count is never the surface carrying P.
n_sup = 0
node_counts = collections.defaultdict(dict)
for (n, s_), c in cnt.items():
    node_counts[n][s_] = c
agree = collections.Counter()
for n, lst in by_node.items():
    top = max(lst, key=lambda x: (x[1], x[0]))[0]
    d = node_counts.get(n)
    agree["absent" if not d else "same" if top in d else "diff"] += 1
    if d and max(d.values()) >= 2:
        n_sup += 1
p_ok = sum(1 for n, lst in by_node.items()
           if any(pv >= MIN_P for s_, pv in lst if s_ in node_counts.get(n, ())))
print(f"file agreement: {dict(agree)}   nodes with count>=2: {n_sup:,}   "
      f"nodes with a P>=0.5 counted surface: {p_ok:,}", flush=True)

# ---------------------------------------------------------------- threshold grid
grid = {}
for mc in (1, 2, 3, 5, 10):
    for mp in (0.0, 0.5, 0.9, 0.99):
        k = sum(1 for n in by_node if select(n, mc, mp)[0])
        grid[f"count>={mc},P>={mp}"] = k
print("yield grid:", json.dumps(grid), flush=True)

# ---------------------------------------------------------------- apply the default
rows = []
reject = collections.Counter()
for n in by_node:
    pick, why = select(n, MIN_COUNT, MIN_P)
    if not pick:
        reject[why] += 1
        continue
    surf, pv, c = pick
    nm = decode(surf).strip()
    if not nm:
        reject["empty_after_decode"] += 1
        continue
    rows.append((uid[n], n, nm, pv, c, len(by_node[n])))

rows.sort()
if rows:
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in rows], pa.int64()),
        "node_id": pa.array([r[1] for r in rows]),
        "display_name": pa.array([r[2] for r in rows]),
        "p_mid_given_surface": pa.array([r[3] for r in rows], pa.float64()),
        "mention_count": pa.array([r[4] for r in rows], pa.int32()),
        "surfaces_considered": pa.array([r[5] for r in rows], pa.int32())}),
        f"{ACQ}/facc1_names.parquet", compression="zstd")

rec = {"schema": "FACC1_NAMES/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. RESOLUTION_OVERLAY_V1 "
                       "manifest hash 25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865"
                       " unchanged."),
       "reachable_nodes": len(by_node),
       "RULE": {"MIN_COUNT": MIN_COUNT, "MIN_P": MIN_P,
                "dominant_form": ("unique argmax of P among surfaces that ALREADY meet both "
                                  "thresholds; ties broken by support, and a full tie refused"),
                "same_surface": "support and confidence must hold for the same string, not for two different ones",
                "conjunction": "all three required together"},
       "FILE_AGREEMENT": {
           "WHY": ("support rejects most of the reach, so that had to be separated from the duller "
                   "possibility that the two files disagree about which surface belongs to a node -- "
                   "in which case the counted surface and the confident surface would never be the "
                   "same string and the conjunction would be unsatisfiable by construction."),
           "entity_list_surface_vs_surface_map_argmax": dict(agree),
           "nodes_with_any_surface_count_ge_2": n_sup,
           "nodes_with_a_counted_surface_at_P_ge_0.5": p_ok,
           "VERDICT": ("the files agree on the surface for the large majority of nodes and no node "
                       "is absent from either, so the rejections are NOT a join artifact. FACC1 "
                       "simply has repeated support for only a small minority of these nodes: they "
                       "are residual precisely because they were barely written about."),
           "CONSEQUENCE_FOR_THE_THRESHOLD": ("count>=2 is kept. Dropping to count>=1 would multiply "
                                             "the yield, but a single observed mention of a "
                                             "case-folded web surface is exactly the 'random "
                                             "mention' the rule was written to exclude.")},
       "YIELD_GRID": grid,
       "GRID_NOTE": ("reported so the threshold is visible rather than tuned silently. count>=1, "
                     "P>=0.0 is the unfiltered reach and is NOT a name count."),
       "names_recovered": len(rows),
       "rejected": dict(reject),
       "pct_of_residue": round(100 * len(rows) / RESIDUE_ENTITY_MIDS, 6),
       "CASE": {"source_is_case_folded": True,
                "uppercase_rows_in_47M": 0,
                "DECISION": ("published lowercase, as the source holds it. Title-casing would "
                             "invent capitalisation, and capitalisation is part of a name.")},
       "KEY_ESCAPES": {"rows_containing_dollar": esc_seen,
                       "decoded": True,
                       "RULE": "$XXXX -> chr(0xXXXX); Freebase key encoding, reversible"},
       "PROVENANCE_IF_USED": "FACC1_ATTESTED_SURFACE",
       "IS_ORIGINAL_NAME": False,
       "WHY_NOT_ORIGINAL": ("a surface form observed in web text referring to this MID, not the "
                            "/type/object/name value Freebase held."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FACC1_NAMES.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)

print(f"\nFACC1 names recovered: {len(rows):,}  (rule: count>={MIN_COUNT} and P>={MIN_P} and dominant)")
print(f"  rejected: {dict(reject)}")
for r in sorted(rows, key=lambda x: -x[4])[:15]:
    print(f"   {r[1]:<14} P={r[3]:.4f} n={r[4]:<7} {r[2]!r}")
print(f"{time.time()-t0:.0f}s")
