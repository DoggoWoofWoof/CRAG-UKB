"""Measure the literal deficit that IDIR's preprocessing filter would impose on CRAG.

    python scratchpad/final_canonical_build/webqsp_v1/literal_deficit.py

IDIR keeps a triple only when BOTH endpoints match '^/m/|^/g/' (verified verbatim in
DataPreparationScripts/FB1.sh), so every literal-valued fact is dropped before their graph
variants are built. The CRAG question is how many facts the benchmark actually needs that such a
filter destroys.

WHICH INSTRUMENT. Three candidate substrates, and only one can see literals at all:

  NSM sibling   MID-preserving, but its vocabulary is 99.99% MIDs -- 149 non-MID of 1,441,420
                (webqsp) and 226 of 2,429,346 (CWQ). NSM ALSO strips literals, so it cannot
                measure their loss. Recorded here because that fact matters on its own.
  IDIR          strips them by construction. The thing under test.
  RoG V1        collapsed names onto MIDs, but it KEPT literal-valued objects as strings. A date
                or a number is identifiable by shape regardless of the name collapse, and section 3
                already classified 14,515 of them as VALUE_LITERAL under a frozen rule.

So RoG V1 is the instrument, with a stated floor: literal STRINGS (a description, a name-shaped
value) are indistinguishable from collapsed entity names there, so every count below is a LOWER
BOUND covering pattern-identifiable literals only.

Reads only. Writes one report.
"""
import json, os, re, time
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq

D = "data/final_canonical/webqsp/v1"
SRC = {"webqsp": "data/original/webqsp/rog_webqsp", "cwq": "data/original/cwq/rog_cwq"}
NSM = "data/final_canonical/webqsp/_acquisition/nsm/extracted"
OUT = "data/final_canonical/webqsp/V3_LITERAL_DEFICIT.json"

MID_RE = re.compile(r"^[mg]\.[0-9A-Za-z_]+$")


def main():
    t0 = time.time()

    # ---- why NSM is not the instrument ----
    nsm = {}
    for ds, p in (("webqsp", f"{NSM}/webqsp/webqsp/entities.txt"),
                  ("CWQ", f"{NSM}/CWQ/CWQ/entities.txt")):
        with open(p, encoding="utf-8") as fh:
            ents = [ln.rstrip("\n") for ln in fh]
        non = [e for e in ents if not MID_RE.match(e)]
        nsm[ds] = {"vocab": len(ents), "non_mid": len(non),
                   "non_mid_pct": round(100.0 * len(non) / len(ents), 4),
                   "examples": non[:10]}
    print(f"[nsm] {json.dumps({k: v['non_mid'] for k, v in nsm.items()})}", flush=True)

    # ---- the instrument: V1 ----
    nodes = pq.read_table(f"{D}/nodes.parquet", columns=["node_kind", "display_name"])
    kind = nodes.column("node_kind").to_pylist()
    disp = nodes.column("display_name").to_pylist()
    N = len(kind)
    is_lit = np.array([k == "VALUE_LITERAL" for k in kind])
    is_cvt = np.array([k == "CVT_MEDIATOR" for k in kind])
    lit_surfaces = {disp[i] for i in range(N) if is_lit[i]}

    e = pq.read_table(f"{D}/edges.parquet", columns=["src_uid", "dst_uid"])
    src = e.column("src_uid").to_numpy()
    dst = e.column("dst_uid").to_numpy()
    del e
    E = src.size

    dst_lit = is_lit[dst]
    src_lit = is_lit[src]
    dropped = dst_lit | src_lit
    print(f"[v1] E={E:,} literal-incident={int(dropped.sum()):,} t={time.time()-t0:.0f}s", flush=True)

    # ---- CVT damage: mediators whose arguments are literal-valued ----
    args_tot = defaultdict(int)
    args_lit = defaultdict(int)
    for s, d_, dl in zip(src.tolist(), dst.tolist(), dst_lit.tolist()):
        if is_cvt[s]:
            args_tot[s] += 1
            if dl:
                args_lit[s] += 1
    cvt_all = cvt_some = 0
    for s, tot in args_tot.items():
        l = args_lit.get(s, 0)
        if l == tot and tot > 0:
            cvt_all += 1
        elif l > 0:
            cvt_some += 1
    cvt_with_args = len(args_tot)
    cvt_total = int(is_cvt.sum())

    # ---- gold answers that are literals ----
    gold, occ = set(), Counter()
    for d_ in SRC.values():
        for sh in sorted(f for f in os.listdir(d_) if f.endswith(".parquet")):
            for a in pq.read_table(f"{d_}/{sh}", columns=["answer"]).column("answer").to_pylist():
                for s in a:
                    gold.add(s)
                    occ[s] += 1
    gold_lit = {g for g in gold if g in lit_surfaces}
    gold_lit_occ = sum(occ[g] for g in gold_lit)

    doc = {
        "schema": "V3_LITERAL_DEFICIT/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "how many WebQSP/CWQ-relevant facts would the IDIR filter destroy?",
        "idir_filter_verified_verbatim": "WHERE subject REGEXP '^/m/|^/g/' AND object REGEXP "
                                         "'^/m/|^/g/' -- confirmed in DataPreparationScripts/FB1.sh",

        "WHY_NSM_IS_NOT_THE_INSTRUMENT": {
            "finding": "the NSM sibling release also strips literals; its vocabulary is ~99.99% "
                       "MIDs.",
            "per_dataset": nsm,
            "consequence": "NSM cannot measure literal loss because it already suffered it. This is "
                           "a finding in its own right: no existing WebQSP-lineage graph preserves "
                           "literal-valued facts. RoG collapsed them into strings indistinguishable "
                           "from entity names, NSM dropped them, IDIR filters them by regex.",
        },

        "INSTRUMENT": "RoG V1, whose VALUE_LITERAL class was frozen in section 3.",
        "LOWER_BOUND_DISCLAIMER": "counts pattern-identifiable literals (date-shaped and "
                                  "numeric-shaped) only. A literal STRING is indistinguishable from "
                                  "a collapsed entity name in RoG, so the true deficit is strictly "
                                  "larger than every number below.",

        "EDGES": {
            "total": E,
            "literal_incident": int(dropped.sum()),
            "literal_incident_pct": round(100.0 * int(dropped.sum()) / E, 4),
            "object_is_literal": int(dst_lit.sum()),
            "subject_is_literal": int(src_lit.sum()),
        },
        "NODES": {
            "total": N,
            "value_literal": int(is_lit.sum()),
            "distinct_literal_surfaces": len(lit_surfaces),
        },
        "CVT_DAMAGE": {
            "cvt_total": cvt_total,
            "cvt_with_at_least_one_argument": cvt_with_args,
            "cvt_losing_every_argument_under_the_filter": cvt_all,
            "cvt_losing_some_arguments_under_the_filter": cvt_some,
            "why_this_matters": "a mediator that loses all its arguments is not a smaller record, "
                                "it is an empty one. The composition that makes a CVT worth "
                                "retrieving is exactly what a literal-stripping filter removes.",
        },
        "GOLD_ANSWERS_THAT_ARE_LITERALS": {
            "distinct": len(gold_lit),
            "distinct_pct_of_gold": round(100.0 * len(gold_lit) / len(gold), 4),
            "occurrences": gold_lit_occ,
            "occurrence_pct": round(100.0 * gold_lit_occ / sum(occ.values()), 4),
            "gold_total_distinct": len(gold),
            "examples": sorted(gold_lit)[:15],
            "reading": "these answers are unreachable in a graph whose literal-valued facts were "
                       "filtered out. Those questions are not made harder, they are made "
                       "unanswerable.",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
