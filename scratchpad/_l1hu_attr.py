"""PART A/B -- attribution ladder + per-query rescue classification.

Pure L0: every number here already sits in halo/COVERAGE_<ds>.json (BASE+F6 lanes x
{C0_METIS,C1_HYPER_UNIVERSAL} x {O0_CORE,O4_FULL_C_b0.5}, each with a per-query _ind_ALL array).
Nothing is re-evaluated; this only tabulates and cross-tabulates what Phase B already measured.

PART A -- the 8-cell ladder (F6 = the frozen SAFE selector, the shipped lane):

  A0 METIS  no-halo BASE     A1 METIS  no-halo SAFE(F6)
  A2 H4     no-halo BASE     A3 H4     no-halo SAFE(F6)
  A4 METIS  b0.5    BASE     A5 METIS  b0.5    SAFE(F6)
  A6 H4     b0.5    BASE     A7 H4     b0.5    SAFE(F6)   <- current shipped system

  HYPERGRAPH_BASE_EFFECT = A2-A0   SAFE_ON_METIS_EFFECT = A1-A0   SAFE_ON_H4_EFFECT = A3-A2
  HALO_ON_METIS_BASE = A4-A0       HALO_ON_METIS_SAFE = A5-A1
  HALO_ON_H4_BASE    = A6-A2       HALO_ON_H4_SAFE     = A7-A3

PART B -- classify every query on the SAFE(F6) lane using
  a=A1(METIS,noHalo)  b=A3(H4,noHalo)  c=A5(METIS,Halo)  d=A7(H4,Halo, FINAL SHIPPED)

  BASE_ALREADY_SOLVED       a=1, b=1                    (H4 never threatened it; halo irrelevant)
  H4_LOSS_HALO_RESCUE       a=1, b=0, d=1                (H4 alone would regress it; halo repairs)
  REGRESSED                 a=1, b=0, d=0                (uncompensated regression -- not a spec
                                                           name, kept for exhaustiveness; should be
                                                           rare/zero given the phase's own gate)
  STILL_UNSOLVED            a=0, d=0
  H4_RESCUE_HALO_REDUNDANT  a=0, d=1, b=1, c=1           (either mechanism alone would rescue it)
  H4_ONLY_RESCUE            a=0, d=1, b=1, c=0
  HALO_ONLY_RESCUE          a=0, d=1, b=0, c=1
  H4_AND_HALO_RESCUE        a=0, d=1, b=0, c=0           (true synergy -- neither alone suffices)

This is an interpretive choice (the spec named the 7 buckets, not their formal boolean definition);
it is documented here exactly so it can be corrected if a different reading was intended, and it is
provably exhaustive + mutually exclusive over all 16 (a,b,c,d) combinations (b=1 => d=1 always,
since a halo-augmented fetch is a strict superset of the same core's no-halo fetch, so no (a,b,c,d)
combination with b=1,d=0 can occur).

  python scratchpad/_l1hu_attr.py ladder
  python scratchpad/_l1hu_attr.py classify
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1hu_hard as HH
import _l1ep_pu as PU

OUT = HH.OUT
CELLS = [("A0", "C0_METIS", "O0_CORE", "BASE"), ("A1", "C0_METIS", "O0_CORE", "F6"),
        ("A2", "C1_HYPER_UNIVERSAL", "O0_CORE", "BASE"), ("A3", "C1_HYPER_UNIVERSAL", "O0_CORE", "F6"),
        ("A4", "C0_METIS", "O4_FULL_C_b0.5", "BASE"), ("A5", "C0_METIS", "O4_FULL_C_b0.5", "F6"),
        ("A6", "C1_HYPER_UNIVERSAL", "O4_FULL_C_b0.5", "BASE"),
        ("A7", "C1_HYPER_UNIVERSAL", "O4_FULL_C_b0.5", "F6")]
DELTAS = [("HYPERGRAPH_BASE_EFFECT", "A2", "A0"), ("SAFE_ON_METIS_EFFECT", "A1", "A0"),
         ("SAFE_ON_H4_EFFECT", "A3", "A2"), ("HALO_ON_METIS_BASE", "A4", "A0"),
         ("HALO_ON_METIS_SAFE", "A5", "A1"), ("HALO_ON_H4_BASE", "A6", "A2"),
         ("HALO_ON_H4_SAFE", "A7", "A3")]


def load_cells(ds):
    cov = json.load(open("%s/halo/COVERAGE_%s.json" % (OUT, ds)))
    out = {}
    for name, core, ocell, lane in CELLS:
        try:
            c = cov[core][ocell][lane]
        except KeyError:
            return None
        out[name] = {"val": c["ALL_REQUIRED_FETCHED"], "ind": np.asarray(c["_ind_ALL"], np.int8)}
    return out


def ladder():
    report = {}
    for ds in HH.DS:
        cells = load_cells(ds)
        if cells is None:
            continue
        floor = HH.FLOOR.get(ds, 0.0)
        row = {name: c["val"] for name, c in cells.items()}
        for dname, hi, lo in DELTAS:
            g, l, p = HH.mcnemar(cells[lo]["ind"], cells[hi]["ind"])
            d = round(row[hi] - row[lo], 4)
            row[dname] = {"delta": d, "gained": g, "lost": l, "p": p,
                         "sig": bool(p < 0.05 and abs(d) > floor)}
        report[ds] = row
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open("%s/attribution/A_LADDER.json" % _ensure(OUT), "w"), indent=1)
    for ds, row in report.items():
        print("\n== %s ==" % ds)
        print("  A0=%.4f A1=%.4f A2=%.4f A3=%.4f A4=%.4f A5=%.4f A6=%.4f A7=%.4f" % tuple(
            row[k] for k in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")))
        for dname, *_ in DELTAS:
            v = row[dname]
            print("  %-22s %+.4f  (gained=%d lost=%d p=%.3g sig=%s)" % (
                dname, v["delta"], v["gained"], v["lost"], v["p"], v["sig"]))
    return report


def _ensure(base):
    p = "%s/attribution" % base
    os.makedirs(p, exist_ok=True)
    return base


BUCKETS = ["BASE_ALREADY_SOLVED", "H4_LOSS_HALO_RESCUE", "REGRESSED", "STILL_UNSOLVED",
          "H4_RESCUE_HALO_REDUNDANT", "H4_ONLY_RESCUE", "HALO_ONLY_RESCUE", "H4_AND_HALO_RESCUE"]


def bucket_of(a, b, c, d):
    if a and b:
        return "BASE_ALREADY_SOLVED"
    if a and not b:
        return "H4_LOSS_HALO_RESCUE" if d else "REGRESSED"
    # a == 0
    if not d:
        return "STILL_UNSOLVED"
    if b and c:
        return "H4_RESCUE_HALO_REDUNDANT"
    if b and not c:
        return "H4_ONLY_RESCUE"
    if not b and c:
        return "HALO_ONLY_RESCUE"
    return "H4_AND_HALO_RESCUE"


def classify():
    g, gptr, rows, hops = None, None, None, None
    report = {}
    for ds in HH.DS:
        cells = load_cells(ds)
        if cells is None:
            continue
        a, b, c, d = (cells[k]["ind"] for k in ("A1", "A3", "A5", "A7"))
        nq = len(a)
        assert len(b) == len(c) == len(d) == nq
        lab = np.array([bucket_of(int(a[i]), int(b[i]), int(c[i]), int(d[i])) for i in range(nq)])
        g_, gptr_, rows_, hops_ = PU.gold_rows(ds)
        need_n = np.array([len(set(int(x) for x in g_[gptr_[qi]:gptr_[qi + 1]])) for qi in range(nq)])
        counts = {bk: int((lab == bk).sum()) for bk in BUCKETS}
        assert sum(counts.values()) == nq, (ds, sum(counts.values()), nq)
        per_bucket = {}
        for bk in BUCKETS:
            m = lab == bk
            if not m.any():
                per_bucket[bk] = {"n": 0, "frac": 0.0}
                continue
            per_bucket[bk] = {"n": int(m.sum()), "frac": round(float(m.mean()), 4),
                              "mean_required_nodes": round(float(need_n[m].mean()), 2),
                              "median_required_nodes": float(np.median(need_n[m]))}
            if ds == "metaqa" and hops_ is not None:
                h = np.asarray(hops_)[:nq]
                per_bucket[bk]["by_hop"] = {"hop%d" % k: int(((lab == bk) & (h == k)).sum())
                                            for k in (1, 2, 3)}
        report[ds] = {"nq": nq, "buckets": per_bucket,
                      "sanity_total_matches_nq": sum(counts.values()) == nq}
    json.dump(report, open("%s/attribution/B_QUERY_CLASSES.json" % _ensure(OUT), "w"), indent=1)
    for ds, r in report.items():
        print("\n== %s (nq=%d) ==" % (ds, r["nq"]))
        for bk in BUCKETS:
            v = r["buckets"][bk]
            extra = (" mean_req=%.1f" % v["mean_required_nodes"]) if v["n"] else ""
            print("  %-26s n=%-5d (%5.1f%%)%s" % (bk, v["n"], 100 * v["frac"], extra))
    return report


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "classify":
        classify()
    else:
        ladder()
