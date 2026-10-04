"""L1 DEVELOPMENT diagnostic (gold-aware; never a candidate): the DEGREE-MATCHED control for the edge diagnostic's
neighbour-degree table (_l1d_edgediag.py, records results/L1_DEV/edgediag_<ds>__<tag>.json).

The edge diagnostic reports, per edge family F and neighbour-degree bin d (undirected degree of u in F), the per-entry
enrichment  P(u gold | u in N_F(s), deg_F(u) in d) / P(u gold)  against a degree-BLIND random node.  Hubs can be gold-rich
in themselves (a question distribution that asks for genres/years/languages makes the hub nodes answers), so a high bin
enrichment need not mean that ADJACENCY localizes.  This reads the record's raw sums and divides instead by the
degree-MATCHED prior of the same stratum s (ALL / per hop):
    gold (exact):      P(u gold | deg_F(u) in d)  =  sum_q |G_q n d| / (n_q * n_d)
    missing@M (approx): P(u missing@M | deg_F(u) in d, u outside FLAT@M)
                        ~=  sum_q |G_q^miss(M) n d| / (n_q * n_d * (1 - M/N))     (FLAT@M taken as degree-neutral)
    enrichment_dm(d) = observed / (entries_d * prior_d)        (entries outside FLAT@M for the missing columns)
The expected counts use the stratum-level prior, i.e. they ignore the within-stratum correlation between a query's entries
in bin d and its golds in bin d.  Also reported per bin: node share n_d / N and gold share sum_q |G_q n d| / sum_q |G_q|
(their ratio = how gold-rich the degree class is, adjacency aside).

Reads the edge-diagnostic record, the population's gold nodes, the v1 FLAT gold ranks (loc_<ds>__v1.npz pos_FLAT) and
the graph families (degrees only; the same key construction as _l1d_edgediag.build_families).  Writes nothing under data/.
Usage: python scratchpad/_l1d_degprior.py <dataset> <tag>  ->  results/L1_DEV/degprior_<ds>__<tag>.json (write-once)
"""
import collections
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_edgediag as E

log = D.log


def family_degree_bins(cd, N):
    """node -> degree bin per family, exactly as _l1d_edgediag.build_families (deg_und of the family's undirected keys)."""
    s, d, r, ent = cd.family("structural")
    s, d, r = (np.asarray(x, np.int64) for x in (s, d, r))
    vocab = ent.get("relation_vocabulary") or []
    SD = E.dkeys(s, d, N)
    und = collections.OrderedDict()
    STRUCT = E.und_of_directed(SD, N)
    und["STRUCT_out"] = und["STRUCT_in"] = und["STRUCT"] = STRUCT
    ks, kd, _, _ = cd.family("knn")
    KNN = E.ukeys(ks, kd, N)
    ns_, nd_, _, _ = cd.family("ner")
    NER = E.ukeys(ns_, nd_, N)
    titles = [x.get("title") for x in cd.ds.nodes()]
    grp = collections.defaultdict(list)
    for i, t in enumerate(titles):
        if t:
            grp[t].append(i)
    tp = [(a, b) for v in grp.values() if len(v) > 1 for ia, a in enumerate(v) for b in v[ia + 1:]]
    TITLE = E.ukeys([a for a, _ in tp], [b for _, b in tp], N) if tp else np.empty(0, np.int64)
    del titles, grp, tp
    und["KNN"], und["NER"] = KNN, NER
    if len(TITLE):
        und["TITLE"] = TITLE
    und["SK"] = und["OUT+KNN"] = np.union1d(STRUCT, KNN)
    und["SKN"] = np.union1d(und["SK"], NER)
    if len(vocab) > 1:
        for ri, nm in enumerate(vocab):
            m = r == ri
            und[nm + "_out"] = und[nm + "_in"] = E.und_of_directed(E.dkeys(s[m], d[m], N), N)
    out, memo = collections.OrderedDict(), {}
    for nm, k in und.items():
        if id(k) not in memo:
            deg = np.bincount(k // np.int64(N), minlength=N) + np.bincount(k % np.int64(N), minlength=N)
            memo[id(k)] = E.dbin(deg).astype(np.int64)
        out[nm] = memo[id(k)]
    return out


def main():
    ds, tag = sys.argv[1], sys.argv[2]
    t0 = time.time()
    fo = os.path.join(D.OUT, "degprior_%s__%s.json" % (ds, tag))
    assert not os.path.exists(fo), "write-once: %s exists" % fo
    fr = os.path.join(D.OUT, "edgediag_%s__%s.json" % (ds, tag))
    with open(fr, encoding="utf-8") as fh:
        R = json.load(fh)
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    assert R["N"] == N
    pop = D.Population(cd, None)
    nq, gptr = pop.nq, pop.gptr
    assert nq == R["n_rows"] and pop.ng_tot == R["n_gold_nodes"]
    z1 = np.load(os.path.join(D.OUT, "loc_%s__%s.npz" % (ds, E.V1_TAG)))
    assert (z1["rows"][:nq] == pop.rows).all() and int(z1["gptr"][nq]) == pop.ng_tot
    pos = z1["pos_FLAT"][:pop.ng_tot].astype(np.int64)
    snames = R["results"]["strata"]
    hops = sorted(set(int(x) for x in pop.hops))
    assert snames == ["ALL"] + ["hop%d" % x for x in hops], snames
    sidx = {x: 1 + i for i, x in enumerate(hops)}
    ns = len(snames)
    nqs = np.zeros(ns)
    nqs[0] = nq
    for x in hops:
        nqs[sidx[x]] = float((np.asarray(pop.hops) == x).sum())
    NB = len(E.DEG_LABELS)
    MS = list(E.MS)
    DB = family_degree_bins(cd, N)
    log("degree bins built for %d families (%.0fs)" % (len(DB), time.time() - t0))
    res = {"dataset": ds, "tag": tag, "mode": "L1_DEVELOPMENT_DEGREE_MATCHED_CONTROL (gold-aware; never a candidate)",
           "status": "DEVELOPMENT (descriptive)", "definitions": __doc__,
           "inputs": {"edgediag": {"path": D.rel(fr), "sha256": D.sha_file(fr)},
                      "loc_v1": {"path": D.rel(os.path.join(D.OUT, "loc_%s__%s.npz" % (ds, E.V1_TAG))),
                                 "sha256": D.sha_file(os.path.join(D.OUT, "loc_%s__%s.npz" % (ds, E.V1_TAG)))},
                      "population": pop.record},
           "code": {"path": D.rel(os.path.abspath(__file__)), "sha256": D.sha_file(os.path.abspath(__file__)),
                    "edgediag": {"path": "scratchpad/_l1d_edgediag.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_edgediag.py"))},
                    "lib": {"path": "scratchpad/_l1d_lib.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))}},
           "families": {}}
    md = ["\n### %s: neighbour-degree bins, enrichment vs degree-BLIND random -> vs degree-MATCHED prior (gold exact; missing@M approx)" % ds]
    for fam, db in DB.items():
        key = "edge|%s|nbr_degree" % fam
        if key not in R["results"]["sums"]:
            continue
        A = np.asarray(R["results"]["sums"][key], np.float64)          # (strata, bins, 12): EDGE_FIELDS
        assert A.shape == (ns, NB, len(E.EDGE_FIELDS)), (fam, A.shape)
        f_ = {n: i for i, n in enumerate(E.EDGE_FIELDS)}
        n_d = np.bincount(db, minlength=NB).astype(np.float64)
        gcnt = np.zeros((ns, NB))
        mcnt = np.zeros((len(MS), ns, NB))
        for j in range(nq):
            g = pop.golds[j]
            b = db[g]
            pj = pos[gptr[j]:gptr[j + 1]]
            for s_ in (0, sidx[int(pop.hops[j])]):
                gcnt[s_] += np.bincount(b, minlength=NB)
                for k, M in enumerate(MS):
                    mcnt[k, s_] += np.bincount(b[pj >= M], minlength=NB)
        F = {}
        for si, sn in enumerate(snames):
            rows = {}
            for bi, bl in enumerate(E.DEG_LABELS):
                cnt = A[si, bi, f_["nbrs"]]
                if cnt <= 0 or n_d[bi] <= 0:
                    continue
                pg = gcnt[si, bi] / (nqs[si] * n_d[bi])
                e = {"entries": int(cnt), "share_of_entries": D.q4(cnt / max(A[si, :, f_["nbrs"]].sum(), 1.0)),
                     "nodes_in_bin": int(n_d[bi]), "node_share": D.q4(n_d[bi] / N),
                     "gold_share": D.q4(gcnt[si, bi] / max(gcnt[si].sum(), 1.0)),
                     "enrich_gold_blind": D.q4(A[si, bi, f_["gold"]] / A[si, bi, f_["exp_gold"]]) if A[si, bi, f_["exp_gold"]] > 0 else None,
                     "enrich_gold_degree_matched": D.q4((A[si, bi, f_["gold"]] / cnt) / pg) if pg > 0 else None}
                for k, M in enumerate(MS):
                    out_ = A[si, bi, f_["out@%d" % M]]
                    pm = mcnt[k, si, bi] / (nqs[si] * n_d[bi] * (1.0 - M / float(N)))
                    ex = A[si, bi, f_["exp_miss@%d" % M]]
                    e["enrich_miss@%d_blind" % M] = D.q4(A[si, bi, f_["miss@%d" % M]] / ex) if ex > 0 else None
                    e["enrich_miss@%d_degree_matched_approx" % M] = (D.q4((A[si, bi, f_["miss@%d" % M]] / out_) / pm)
                                                                      if out_ > 0 and pm > 0 else None)
                rows[bl] = e
            F[sn] = rows
        res["families"][fam] = F
        for sn in snames:
            md.append("%-18s %-5s %s" % (fam, sn, "  ".join(
                "%s: %s->%s / m5k %s->%s (e%.2f n%.3f g%.3f)" % (
                    bl, v["enrich_gold_blind"], v["enrich_gold_degree_matched"], v["enrich_miss@5000_blind"],
                    v["enrich_miss@5000_degree_matched_approx"], v["share_of_entries"], v["node_share"], v["gold_share"])
                for bl, v in F[sn].items())))
    res["seconds"] = round(time.time() - t0, 1)
    res["process_peak_rss_mb"] = D.peak_rss_mb()
    D.G.S.wj(fo, res)
    print("\n".join(md))
    log("done (%.0fs) -> %s sha256 %s" % (res["seconds"], fo, D.sha_file(fo)[:12]))


if __name__ == "__main__":
    main()
