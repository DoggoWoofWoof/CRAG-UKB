"""Why is webqsp's name-join exactly zero? Test the anti-join hypothesis.

WHAT IS ESTABLISHED

  WEBQSP_NAME_JOIN_CONTROL put two draws through ONE pass over name.parquet:
    A  webqsp-agnostic ENTITY_MID keys from subj_000   15,000 -> 13,757 named  (91.713%)
    B  the same keys restricted to webqsp MIDs         15,000 ->      0 named  ( 0.000%)
  So the join code is sound and the zero is a property of the population, not a bug.

WHAT IS REFUTED

  Namespace. g.* ids are far less named than m.* ids, so a g.-heavy draw could produce a low
  rate -- but webqsp's structural residue is 405,357 g. against 1,247,261 m. (24.5% g.), and
  draw A was 14,996 m. of 15,000.  A 75%-m. population cannot join at 0% when m. joins at
  ~91.7%; that predicts roughly 10,000 of 15,000.  Namespace is not the mechanism.

THE HYPOTHESIS THIS TESTS

  webqsp already resolved MIDs to names ("webqsp: resolve all raw Freebase MIDs to names in
  doc text").  If it drew those names from this same Freebase name source, then the nodes
  still labelled CVT_MEDIATOR / MID_NAMED_ENTITY / UNRESOLVED_OTHER are BY CONSTRUCTION the
  ones that source could not name.  The 0% would then be a tautology -- the residue is the
  anti-join of the very table being queried -- and not a discovery about V3.

  The discriminating test is the complement: webqsp's READABLE_ENTITY nodes.  If the residue
  is the anti-join, the resolved nodes must join at a HIGH rate through identical code.

    residue 0% and readable high    -> ANTIJOIN_CONFIRMED. A webqsp V2 built on this name
                                       table gains nothing because the naming already ran
                                       against it.  Nothing is broken and nothing is owed.
    residue 0% and readable also 0% -> the two populations share some other blocker; the
                                       anti-join story is wrong and this stays open.

  This is the same shape as ZERO_JOIN_INVARIANT, applied one level up: a zero is only
  interpretable once a population that SHOULD hit is shown to hit.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/webqsp_antijoin_test.py
"""

import collections
import json
import os
import sys

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODES = os.path.join(ROOT, "data", "final_canonical", "webqsp", "nodes.jsonl")
NAME = os.path.join(ROOT, "data", "final_canonical", "freebase_v3", "canonical",
                    "metadata", "name.parquet")
OUT = os.path.join(ROOT, "data", "final_canonical", "webqsp", "WEBQSP_ANTIJOIN_TEST.json")

N = 15000
RESIDUE = ("CVT_MEDIATOR", "MID_NAMED_ENTITY", "UNRESOLVED_OTHER")


def draw():
    """Two key sets from webqsp's own node table, both MIDs, split by whether webqsp resolved
    the node to readable text."""
    resid, readable = {}, {}
    kinds = collections.Counter()
    mid_by_kind = collections.Counter()
    with open(NODES, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            k = d["node_kind"]
            kinds[k] += 1
            m = d.get("source_mid") or ""
            if m[:2] not in ("m.", "g."):
                continue
            mid_by_kind[k] += 1
            if k in RESIDUE:
                if len(resid) < N:
                    resid[m] = k
            else:
                if len(readable) < N:
                    readable[m] = k
    return resid, readable, kinds, mid_by_kind


def join(sets):
    hit = {k: set() for k in sets}
    rows = 0
    f = pq.ParquetFile(NAME)
    for b in f.iter_batches(batch_size=1 << 19, columns=["subject"]):
        rows += b.num_rows
        for s in b.column("subject").to_pylist():
            for k, ks in sets.items():
                if s in ks:
                    hit[k].add(s)
    return hit, rows


def main():
    print("drawing residue and readable MID sets from webqsp nodes.jsonl ...", flush=True)
    resid, readable, kinds, mid_by_kind = draw()
    print("  webqsp node_kind totals: %s" % dict(kinds), flush=True)
    print("  of which carry a MID:    %s" % dict(mid_by_kind), flush=True)
    print("  |residue draw|=%d  |readable draw|=%d" % (len(resid), len(readable)), flush=True)
    if not readable:
        print("  NOTE: no non-residue node carries a source_mid -- see verdict", flush=True)

    print("joining both through ONE pass over name.parquet ...", flush=True)
    hit, nrows = join({"residue": set(resid), "readable": set(readable)})

    res = {}
    print("")
    print("%-12s %8s %8s %9s" % ("population", "keys", "named", "pct"))
    for k, ks in (("residue", resid), ("readable", readable)):
        h = len(hit[k])
        pct = 100.0 * h / max(1, len(ks))
        res[k] = {"keys": len(ks), "named": h, "pct": round(pct, 3),
                  "kinds": dict(collections.Counter(ks.values()))}
        print("%-12s %8d %8d %8.3f%%" % (k, len(ks), h, pct))

    rp, dp = res["residue"]["pct"], res["readable"]["pct"]
    if not readable:
        v = ("NO_READABLE_MIDS -- webqsp keeps source_mid only on the residue kinds, so the "
             "complement cannot be drawn from this file. The anti-join remains the leading "
             "explanation but is NOT demonstrated here; it would need webqsp's own name-"
             "resolution log")
    elif rp < 1 and dp > 50:
        v = ("ANTIJOIN_CONFIRMED -- webqsp's readable nodes join this name table at %.1f%% "
             "while its structural residue joins at %.1f%%. The residue IS the set this "
             "source could not name, so a webqsp V2 over this table gains 0 names by "
             "construction. Nothing is broken" % (dp, rp))
    elif rp < 1 and dp < 1:
        v = ("ANTIJOIN_REFUTED -- neither webqsp population joins, so the zero is not "
             "explained by prior resolution and the cause is still open. Do NOT price a V2 "
             "gain from these numbers")
    else:
        v = "INCONCLUSIVE at these rates -- read the table, do not summarise it"
    print("")
    print("VERDICT: %s" % v)

    rec = {"RECORD": "WEBQSP_ANTIJOIN_TEST",
           "_what": "asks whether webqsp's 0%% name-join is a tautology: is the structural "
                    "residue the anti-join of the very name table being queried? "
                    "MEASUREMENT ONLY -- no dataset was modified.",
           "established": {"draw_A_agnostic_pct": 91.713, "draw_B_webqsp_pct": 0.0,
                           "source": "WEBQSP_NAME_JOIN_CONTROL, one pass, identical code"},
           "refuted": {"namespace": "webqsp residue is 405357 g. / 1247261 m. (24.5%% g.) and "
                                     "draw A was 14996 m. of 15000; a 75%%-m. population cannot "
                                     "join at 0%% when m. joins at ~91.7%%"},
           "method": {"n_per_draw": N, "name_rows_scanned": nrows,
                      "residue_kinds": list(RESIDUE)},
           "webqsp_node_kind_totals": dict(kinds),
           "webqsp_mid_carrying_by_kind": dict(mid_by_kind),
           "results": res,
           "VERDICT": v}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
