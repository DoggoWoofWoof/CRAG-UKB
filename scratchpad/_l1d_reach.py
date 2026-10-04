"""L1 DEVELOPMENT diagnostic (gold-aware, never a candidate): where does a localization variant lose the gold -- in
ACTIVATION (a gold node gets no localization score, L = 0, so it is outside the LOC order) or in RANKING (every gold node is
scored but some sit beyond M in the LOC order)?  Exact partition of the queries at every budget M (LOC arm):
    ALL@M              the variant's LOC@M serves every gold node (the record's positions)
    activation_miss@M  not ALL@M, and some gold node is unscored (outside the LOC order)
    ranking_miss@M     not ALL@M, and every gold node is scored (served too late)
    ALL + activation_miss + ranking_miss = 1.  (A gold outside the LOC order is still served by the FLAT fill when the LOC
    order is shorter than M; 'ALL with an unscored gold' counts those queries.)
For reference, the whole-block ceiling of each cell and hierarchy level (_l1d_ceiling.py): the share of queries whose
distinct gold blocks (regions) hold <= M nodes -- what block-contiguous serving (whole blocks, the served P50) could reach;
node orders that interleave FLAT rank inside a block can serve a gold without its whole block, so it bounds only the
block-contiguous arms.  Reads only the records' npz, the population's gold nodes and the frozen partitions; no query
embedding.

Usage: python scratchpad/_l1d_reach.py <tag> <stem>[,<stem>...] [--dir=<dir>]  (reads <dir>/<stem>_<dataset>__<tag>.npz;
       default dir results/L1_DEV)  ->  <dir>/reach_SUMMARY__<tag>.json (write-once) + markdown.
"""
import json
import os
import sys

import numpy as np

import _l1d_lib as D
import _l1d_hier as H

log = D.log


def main():
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag, stems = pos_args[0], pos_args[1].split(",")
    DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), D.OUT)
    fo = os.path.join(DIR, "reach_SUMMARY__%s.json" % tag)
    assert not os.path.exists(fo), "write-once: %s exists" % fo
    res = {"mode": "L1_DEVELOPMENT_REACH_DECOMPOSITION (gold-aware diagnostic; never a candidate)", "definitions": __doc__,
           "code": {"path": D.rel(os.path.abspath(__file__)), "sha256": D.sha_file(os.path.abspath(__file__)),
                    "hier": {"path": "scratchpad/_l1d_hier.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_hier.py"))},
                    "lib": {"path": "scratchpad/_l1d_lib.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))}},
           "inputs": {}, "datasets": {}}
    md = []
    for ds in ("metaqa", "musique", "squad"):
        recs = [(s, os.path.join(DIR, "%s_%s__%s.npz" % (s, ds, tag))) for s in stems]
        recs = [(s, p) for s, p in recs if os.path.exists(p)]
        if not recs:
            continue
        cd = D.AD.CanonicalDataset(ds)
        nrec = {len(np.load(p)["rows"]) for _, p in recs}
        assert len(nrec) == 1, nrec
        pop = D.Population(cd, rows_limit=nrec.pop())
        parts = {c: D.Part(cd, D.TAG_OF[c]) for c in D.CELLS[ds]}
        S = H.HierSpec(cd, None, parts, log_only=True)
        need = {}                                                   # (cell, level) -> per-query gold-region node total
        for c in parts:
            h = S.H[c]
            for l in range(len(h["maps"])):
                nm, rsz = h["nodemaps"][l], h["rsz"][l]
                need[(c, l)] = np.array([int(rsz[np.unique(nm[g])].sum()) for g in pop.golds], np.int64)
        gptr, nq = pop.gptr, pop.nq
        E = {"n_rows": nq, "variants": {}}
        md.append("\n### %s (%d queries)\n" % (ds, nq))
        E["block_ceiling"] = {"%s:L%d" % k: {str(M): D.q4((v <= M).mean()) for M in D.M_CURVE} for k, v in need.items()}
        md.append("| record | variant | gold scored | queries all-scored | " + " | ".join(
            "M=%d ALL / act / rank" % M for M in D.M_CURVE) + " |")
        md.append("|---|---|---|---|" + "---|" * len(D.M_CURVE))
        for stem, p in recs:
            z = np.load(p)
            assert (z["rows"][:nq] == pop.rows).all() and (z["gptr"][:nq + 1] == gptr).all(), (stem, ds)
            res["inputs"]["%s_%s" % (stem, ds)] = {"path": D.rel(p), "sha256": D.sha_file(p)}
            for k in z.files:
                if not k.startswith("lpos__"):
                    continue
                name = k[len("lpos__"):]
                scored = z[k] >= 0
                ploc = z["pos_LOC__" + name]
                allsc = np.logical_and.reduceat(scored, gptr[:-1])
                e = {"gold_nodes_scored": D.q4(scored.mean()), "queries_all_gold_scored": D.q4(allsc.mean()), "M": {}}
                cells_md = []
                for M in D.M_CURVE:
                    allM = np.logical_and.reduceat(ploc < M, gptr[:-1])
                    am, rm = ~allM & ~allsc, ~allM & allsc
                    e["M"][str(M)] = {"ALL": D.q4(allM.mean()), "activation_miss": D.q4(am.mean()), "ranking_miss": D.q4(rm.mean()),
                                      "ALL_with_an_unscored_gold (FLAT fill)": int((allM & ~allsc).sum())}
                    cells_md.append("%.3f / %.3f / %.3f" % (allM.mean(), am.mean(), rm.mean()))
                E["variants"]["%s:%s" % (stem, name)] = e
                md.append("| %s | %s | %.3f | %.3f | %s |" % (stem, name, scored.mean(), allsc.mean(), " | ".join(cells_md)))
        md.append("\nwhole-block ceiling (block-contiguous serving): " + "; ".join(
            "%s %s" % (k, " ".join("%.3f" % x for x in v.values())) for k, v in E["block_ceiling"].items()))
        res["datasets"][ds] = E
        log("%s done" % ds)
    D.G.S.wj(fo, res)
    print("\n".join(md))
    print("-> %s sha256 %s" % (fo, D.sha_file(fo)))


if __name__ == "__main__":
    main()
