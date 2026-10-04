"""PHASE A -- universalise the HYPERGRAPH REPRESENTATION.  The partitioner never changes.

The previous phase established the asymmetry this program exists to fix.  A true hypergraph core
(Mt-KaHyPar on closed neighbourhoods) is the best representation found so far -- MetaQA ALL@50
+0.1286, hop3 0.2583 -> 0.5916, WebQSP +0.0599, HotpotQA +0.0150 -- but it is NOT universal,
because SQuAD significantly regresses.  The single fixed hyperedge cap of 25 is why: SQuAD's mean
STRUCT closed neighbourhood is 108.3 nodes against 5.2-14.6 everywhere else, so the one cap that
keeps 47-80% of structural pins on the other five keeps 0.87% of SQuAD's.  Every rule below is
ONE rule applied identically to all six corpora; only the number it produces follows the corpus.

  H0_FIXED_CAP        keep e iff |e| <= 25, drop the rest whole      (the previous build, exact)
  H1_FULL_WEIGHTED    every e, nothing dropped, w(e) = 1/(|e|-1)
  H2_Q90/Q95/Q99      cap = percentile_a of THAT corpus's raw size distribution, then truncate
  H3_B05/B10/B20      cap = c * target_block_size, target_block_size = N/k ~= 100
  H4_SPLIT_PRESERVE   oversized e decomposed into anchor-preserving chunks; NO pin discarded

Hyperedge = closed neighbourhood {u} u N(u) of a family graph, so the raw size is deg(u)+1 and
the raw size distribution is the family degree distribution shifted by one.  Truncation, where a
rule calls for it, keeps the members with the strongest STATIC local evidence: members are ranked
by |N(u) n N(v)| descending (local mutual connectivity), then deg(v) ascending (a low-degree
neighbour is more specific than a hub -- the same anti-hub principle as the frozen 1/df NER
weight), then node id ascending so the result is deterministic.  No query, no gold label and no
retrieval score enters any of this.  Hyperedges are emitted in family order and then anchor id
order, which for H0 is byte-identical to the previous build's emission order.

Families are uniform: STRUCT, KNN and NER are all closed neighbourhoods of their own frozen edge
set.  RECORDED LIMITATION -- the NER family on disk (ner_edges_w_df25.pkl) is already clique
expanded, so N_NER(u) is the UNION of the entity groups containing u rather than one hyperedge
per entity.  The per-entity groups are not on disk and recovering them would be a new NER build.

  python scratchpad/_l1hu_build.py audit
  python scratchpad/_l1hu_build.py build <ds> <RULE> [FAMS=SK]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kn_sub as KS
import _l1ep_part as PP
import _l1ep_pu as PU

OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_HYPERGRAPH_UNIVERSAL"
EXP = "scratchpad/_l1hu/graphs"
WSCALE = 1000                     # frozen: the quantisation hyper_clique() already uses
FIXED_CAP = PP.HYPER_CAP          # 25 -- the cap the previous, non-universal build used
FAMSETS = {"S": ("STRUCT",), "SK": ("STRUCT", "KNN"), "SN": ("STRUCT", "NER"),
           "SKN": ("STRUCT", "KNN", "NER")}
RULES = {
    "H0_FIXED_CAP":      {"mode": "drop",  "cap": ("fixed", FIXED_CAP), "w": False},
    "H1_FULL_WEIGHTED":  {"mode": "none",  "cap": None,                 "w": True},
    "H2_Q90":            {"mode": "trunc", "cap": ("pct", 90.0),        "w": True},
    "H2_Q95":            {"mode": "trunc", "cap": ("pct", 95.0),        "w": True},
    "H2_Q99":            {"mode": "trunc", "cap": ("pct", 99.0),        "w": True},
    "H3_B05":            {"mode": "trunc", "cap": ("blk", 0.5),         "w": True},
    "H3_B10":            {"mode": "trunc", "cap": ("blk", 1.0),         "w": True},
    "H3_B20":            {"mode": "trunc", "cap": ("blk", 2.0),         "w": True},
    "H4_SPLIT_PRESERVE": {"mode": "split", "cap": ("blk", 1.0),         "w": True},
}
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
cm = lambda x: "{:,}".format(int(x))


def famkeys(ds, log=log):
    N, S, K, _X = KS.keysets(ds, log)
    _, NER_RAW = KS.raw_ner_keys(ds, log)
    return int(N), {"STRUCT": S, "KNN": K, "NER": NER_RAW}


def _cn_counts(nb, xadj, adj, mark):
    """|N(u) n N(v)| for every v in N(u), vectorised through a shared marker array."""
    st = xadj[nb].astype(np.int64)
    ln = (xadj[nb + 1] - xadj[nb]).astype(np.int64)
    tot = int(ln.sum())
    if tot == 0:
        return np.zeros(len(nb), np.int64)
    base = np.repeat(st - np.concatenate(([np.int64(0)], np.cumsum(ln)[:-1])), ln)
    allnb = adj[base + np.arange(tot, dtype=np.int64)]
    own = np.repeat(np.arange(len(nb), dtype=np.int64), ln)
    return np.bincount(own[mark[allnb]], minlength=len(nb)).astype(np.int64)


def ordered(u, xadj, adj, deg, mark):
    """N(u) in the static retention order: mutual connectivity desc, deg asc, node id asc."""
    nb = adj[xadj[u]:xadj[u + 1]].astype(np.int64)
    mark[nb] = True
    cn = _cn_counts(nb, xadj, adj, mark)
    mark[nb] = False
    return nb[np.lexsort((nb, deg[nb], -cn))]


def _scatter(eidx, dst, anchors, xadj, adj, deg):
    """write [u] + N(u) for many anchors at once; dst[i] is where anchor i starts."""
    eidx[dst] = anchors
    ln = deg[anchors]
    tot = int(ln.sum())
    if tot == 0:
        return
    off = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(ln) - ln, ln)
    eidx[np.repeat(dst + 1, ln) + off] = adj[np.repeat(xadj[anchors], ln) + off]


def resolve_cap(rule, raw, tbs):
    R = RULES[rule]["cap"]
    if R is None:
        return None
    if R[0] == "fixed":
        return int(R[1])
    if R[0] == "pct":
        return max(2, int(np.percentile(raw, R[1])))
    return max(2, int(round(R[1] * tbs)))


def build(ds, rule, famset="SK", log=log):
    assert rule in RULES and famset in FAMSETS
    R, fams = RULES[rule], FAMSETS[famset]
    N, ks = famkeys(ds, log)
    _, npart = PU.load_assignment(ds, "CURRENT")
    k = int(npart)
    tbs = int(round(N / k))
    C = {}
    for f in fams:
        xadj, adj = PP._csr_from_keys(ks[f], N)
        C[f] = (xadj, adj.astype(np.int64), np.diff(xadj).astype(np.int64))
    cap = resolve_cap(rule, np.concatenate([C[f][2][C[f][2] >= 1] + 1 for f in fams]), tbs)

    mark = np.zeros(N, bool)
    parts, szs, fstat = [], [], {}
    for f in fams:
        xadj, adj, deg = C[f]
        anc = np.nonzero(deg >= 1)[0]
        raw = deg[anc] + 1
        pre_n, pre_p = int(len(anc)), int(raw.sum())
        big = raw > cap if cap is not None else np.zeros(len(anc), bool)
        nbig = int(big.sum())

        if R["mode"] == "drop":
            anc, raw, big = anc[~big], raw[~big], big[~big]
            sz = raw.copy()
        elif R["mode"] == "trunc":
            sz = np.minimum(raw, cap) if cap is not None else raw.copy()
        elif R["mode"] == "none":
            sz = raw.copy()
        else:                                             # split: 1 edge per chunk
            per = cap - 1
            nch = np.where(big, (deg[anc] + per - 1) // per, 1)
            sz = np.empty(int(nch.sum()), np.int64)

        if R["mode"] != "split":
            eptr = np.zeros(len(sz) + 1, np.int64)
            eptr[1:] = np.cumsum(sz)
            eidx = np.empty(int(eptr[-1]), np.int64)
            small = ~big
            _scatter(eidx, eptr[:-1][small], anc[small], xadj, adj, deg)
            for i in np.nonzero(big)[0]:
                u = int(anc[i])
                eidx[eptr[i]] = u
                eidx[eptr[i] + 1:eptr[i + 1]] = ordered(u, xadj, adj, deg, mark)[:cap - 1]
            kept, dup = int(eptr[-1]), 0
        else:
            eoff = np.zeros(len(nch) + 1, np.int64)
            eoff[1:] = np.cumsum(nch)
            sz[eoff[:-1][~big]] = raw[~big]
            for i in np.nonzero(big)[0]:
                d = int(deg[anc[i]])
                c = np.full(int(nch[i]), per + 1, np.int64)
                r = d % per
                if r:
                    c[-1] = r + 1
                sz[eoff[i]:eoff[i + 1]] = c
            eptr = np.zeros(len(sz) + 1, np.int64)
            eptr[1:] = np.cumsum(sz)
            eidx = np.empty(int(eptr[-1]), np.int64)
            _scatter(eidx, eptr[eoff[:-1][~big]], anc[~big], xadj, adj, deg)
            for i in np.nonzero(big)[0]:
                u = int(anc[i])
                om = ordered(u, xadj, adj, deg, mark)
                for j, s in enumerate(range(0, len(om), per)):
                    t = eoff[i] + j
                    eidx[eptr[t]] = u
                    eidx[eptr[t] + 1:eptr[t + 1]] = om[s:s + per]
            kept, dup = int(eptr[-1]), int((nch - 1).sum())

        parts.append(eidx)
        szs.append(sz)
        fstat[f] = {"hyperedges_precap": pre_n, "pins_precap": pre_p,
                    "anchors_over_cap": nbig, "hyperedges_out": int(len(sz)),
                    "pins_written": kept, "anchor_duplication_pins": dup,
                    "pins_discarded": pre_p - (kept - dup),
                    "pin_retention": round((kept - dup) / max(pre_p, 1), 6),
                    "size_max_out": int(sz.max()) if len(sz) else 0}
        log("  %s %s %s: %s anchors (%s over cap) -> %s hyperedges, %s/%s pins (%.4f)"
            % (ds, rule, f, cm(pre_n), cm(nbig), cm(len(sz)), cm(kept - dup), cm(pre_p),
               fstat[f]["pin_retention"]))

    sz = np.concatenate(szs)
    eptr = np.zeros(len(sz) + 1, np.int64)
    eptr[1:] = np.cumsum(sz)
    eidx = np.concatenate(parts)
    assert eidx.min() >= 0 and eidx.max() < N and len(eidx) == eptr[-1]
    w = np.maximum(1, np.rint(WSCALE / (sz - 1)).astype(np.int64))
    os.makedirs(EXP, exist_ok=True)
    fp = "%s/%s__%s__%s.npz" % (EXP, ds, rule, famset)
    kw = dict(eptr=eptr, eidx=eidx.astype(np.int32), N=np.array([N]), k=np.array([k]))
    if R["w"]:
        kw["ew"] = w.astype(np.int32)
    np.savez_compressed(fp, **kw)
    pre = sum(v["pins_precap"] for v in fstat.values())
    dup = sum(v["anchor_duplication_pins"] for v in fstat.values())
    meta = {"ds": ds, "rule": rule, "famset": famset, "families": list(fams), "N": N, "k": k,
            "target_block_size": tbs, "cap": cap, "cap_rule": RULES[rule]["cap"],
            "mode": R["mode"], "weighted": R["w"], "weight_scale": WSCALE,
            "hyperedges": int(len(sz)), "pins": int(len(eidx)),
            "pins_precap_total": pre, "anchor_duplication_pins": dup,
            "pin_retention_total": round((int(len(eidx)) - dup) / max(pre, 1), 6),
            "size_min": int(sz.min()), "size_max": int(sz.max()),
            "size_mean": round(float(sz.mean()), 2),
            "weight_min": int(w.min()), "weight_max": int(w.max()),
            "weight_sum": int(w.sum()), "per_family": fstat,
            "file": fp, "bytes": os.path.getsize(fp)}
    log("  %s %s %s: cap=%s  %s hyperedges  %s pins  retention %.4f  w[%d,%d]  %.1f MB"
        % (ds, rule, famset, cap, cm(meta["hyperedges"]), cm(meta["pins"]),
           meta["pin_retention_total"], meta["weight_min"], meta["weight_max"],
           meta["bytes"] / 1e6))
    return meta


def audit(log=log):
    """A1 -- the raw distribution, per corpus per family, with NO truncation anywhere."""
    Q = [50, 75, 90, 95, 99, 99.9]
    rec = {"DEFINITION": "hyperedge = closed neighbourhood {u} u N(u); raw size = deg(u)+1",
           "NOTE": "no cap, no truncation, no drop -- this measures the input distribution only",
           "FIXED_CAP": int(FIXED_CAP), "PER_CORPUS": {}}
    for ds in KS.DS:
        N, ks = famkeys(ds, log)
        _, npart = PU.load_assignment(ds, "CURRENT")
        tbs = int(round(N / int(npart)))
        e = {"N": N, "k": int(npart), "target_block_size": tbs, "families": {}}
        pool = {}
        for f in ("STRUCT", "KNN", "NER"):
            xadj, adj = PP._csr_from_keys(ks[f], N)
            deg = np.diff(xadj).astype(np.int64)
            sz = deg[deg >= 1] + 1
            if not len(sz):
                continue
            pool[f] = sz
            ok = sz <= FIXED_CAP
            e["families"][f] = dict(
                {"hyperedges": int(len(sz)), "pins": int(sz.sum()),
                 "mean": round(float(sz.mean()), 2), "max": int(sz.max()),
                 "hyperedges_kept_at_fixed_cap": int(ok.sum()),
                 "pins_kept_at_fixed_cap": int(sz[ok].sum()),
                 "pin_retention_at_fixed_cap": round(float(sz[ok].sum() / sz.sum()), 6)},
                **{("p%g" % q): float(np.percentile(sz, q)) for q in Q})
        for tag, fs in (("POOLED_ALL", ("STRUCT", "KNN", "NER")), ("POOLED_SK", ("STRUCT", "KNN"))):
            a = np.concatenate([pool[f] for f in fs if f in pool])
            ok = a <= FIXED_CAP
            e[tag] = dict({"hyperedges": int(len(a)), "pins": int(a.sum()),
                           "mean": round(float(a.mean()), 2), "max": int(a.max()),
                           "pin_retention_at_fixed_cap": round(float(a[ok].sum() / a.sum()), 6)},
                          **{("p%g" % q): float(np.percentile(a, q)) for q in Q})
            e["CAP_IF_RULE_" + tag[7:]] = {r: resolve_cap(r, a, tbs) for r in sorted(RULES)}
        for f in e["families"]:
            e["families"][f]["frac_of_corpus_pins"] = round(
                e["families"][f]["pins"] / float(e["POOLED_ALL"]["pins"]), 5)
        rec["PER_CORPUS"][ds] = e
        p = e["POOLED_SK"]
        log("%-16s SK mean %6.2f p90 %6.1f p95 %7.1f p99 %8.1f max %9s tbs %3d  "
            "cap25 keeps %.4f of SK pins" % (ds, p["mean"], p["p90"], p["p95"], p["p99"],
                                             cm(p["max"]), tbs, p["pin_retention_at_fixed_cap"]))
    os.makedirs("%s/pin_retention" % OUT, exist_ok=True)
    fp = "%s/pin_retention/A1_PRECAP_DISTRIBUTION.json" % OUT
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "audit":
        audit()
    elif a and a[0] == "build":
        os.makedirs("%s/hypergraph_build" % OUT, exist_ok=True)
        fp = "%s/hypergraph_build/BUILD_MANIFEST.json" % OUT
        rec = json.load(open(fp)) if os.path.exists(fp) else {}
        for ds in a[1].split(","):
            for rule in a[2].split(","):
                for fs in ((a[3] if len(a) > 3 else "SK")).split(","):
                    rec["%s__%s__%s" % (ds, rule, fs)] = build(ds, rule, fs)
                    json.dump(rec, open(fp, "w"), indent=1)
        log("wrote", fp)
    else:
        print(__doc__)
