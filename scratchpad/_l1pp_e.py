"""ROUND E -- STEP 1 executed literally: one vote per EVIDENCE FAMILY.

The frozen F6 scores a candidate with three terms

    1/(K0+canonical_rank) + 1/(K0+structural_rank) + 1/(K0+retrieval_rank)

where `canonical` is RRF(PR_dense, PR_splade) at PARTITION granularity and `retrieval` is the
node-level Dense+SPLADE RRF mapped to partitions.  Those are the same evidence twice, which is the
duplication this phase was asked to remove.  ROUND B removed it by DELETION (ES) and that costs
five of six corpora, so deletion is not the answer.  This round asks the sharper question: can the
duplication be removed WITHOUT throwing the second granularity's ordering away?

    F6      canonical + structural + retrieval          (3 votes, lexical counted twice)
    ES      canonical + structural                      (2 votes, node granularity DELETED)
    LEXONE  RRF(canonical, retrieval) + structural      (2 votes, ONE lexical vote, nothing lost)
    DS3     dense + splade + structural                 (the directive's literal channel set)
    DS4     dense + splade + structural + retrieval     (DS3 control with the node view restored)

Every variant is the same protected-core competition, is parameter-free, uses no dataset identity
and no gold, and emits EXACTLY 50 partitions.

  python scratchpad/_l1pp_e.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1pp_b as B
import _l1kb_core as KB
import _l1kb_router as JR

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
K0, P = PP.K0, PP.P


def lex_one(cpos, rpos):
    """fuse the two lexical granularities into ONE ranking, then vote once."""
    u = set(cpos) | set(rpos)
    sc = [(-(1.0 / (K0 + cpos[p]) if p in cpos else 0.0)
           - (1.0 / (K0 + rpos[p]) if p in rpos else 0.0), cpos.get(p, 10 ** 6), p) for p in u]
    sc.sort()
    return {p: r for r, (_, _, p) in enumerate(sc)}


def select(c, Bv, mode, dpos, spos_lex):
    cands = c["bnd"] + [p for p in dict.fromkeys(
        list(c["spos"]) + list(c["rpos"])) if p not in c["bnd"] and p not in c["prot_set"]]
    lex = lex_one(c["cpos"], c["rpos"]) if mode == "LEXONE" else None
    sc = []
    for p in cands:
        s = 0.0
        if mode in ("F6", "ES"):
            s += 1.0 / (K0 + c["cpos"][p]) if p in c["cpos"] else 0.0
        elif mode == "LEXONE":
            s += 1.0 / (K0 + lex[p]) if p in lex else 0.0
        else:                                            # DS3 / DS4
            s += 1.0 / (K0 + dpos[p]) + 1.0 / (K0 + spos_lex[p])
        if p in c["spos"]:
            s += 1.0 / (K0 + c["spos"][p])
        if mode in ("F6", "DS4") and p in c["rpos"]:
            s += 1.0 / (K0 + c["rpos"][p])
        sc.append((-s, c["cpos"].get(p, 10 ** 6), p))
    sc.sort()
    fs = c["prot"] + [p for _, _, p in sc[:Bv]]
    assert len(set(fs)) == P
    return fs


def run_ds(ds, OUT):
    z, meta = PP.load(ds)
    nq = meta["n_dev_queries"]; goldp = PP.goldparts(z, meta); psz = z["part_sizes"]
    hops = z["hops"]
    topo = TA.load_topology(ds, log=lambda *a: None)
    ch = PP.channels(ds, z, meta, topo)
    npart = ch["npart"]
    C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
    dpos = PP.rank_pos(ch["PR_d"], npart)
    spos = PP.rank_pos(ch["PR_s"], npart)
    V = {}
    for Bv in (6, 8):
        ctxs = KB.contexts(z, meta, C, Bv)
        if Bv == 6:
            V["BASE"] = np.array([c["prot"] + c["bnd"] for c in ctxs], np.int32)
        for mode in ("F6", "ES", "LEXONE", "DS3", "DS4"):
            V[f"{mode}_B{Bv}"] = np.array(
                [select(c, Bv, mode, dpos[qi], spos[qi]) for qi, c in enumerate(ctxs)], np.int32)
    base = PP.score_top50(V["BASE"], goldp, psz)
    f6 = PP.score_top50(V["F6_B6"], goldp, psz)
    res = {}
    for name, top in V.items():
        s = PP.score_top50(top, goldp, psz)
        e = {"ALL": round(s["ALL"], 4), "dALL": round(s["ALL"] - base["ALL"], 4),
             "dALL_vs_F6": round(s["ALL"] - f6["ALL"], 4),
             "vs_BASE": PP.mcnemar(s["ind"], base["ind"]),
             "vs_F6": PP.mcnemar(s["ind"], f6["ind"]),
             "scope_mean": round(s["scope_mean"], 1)}
        if ds == "metaqa":
            e["per_hop"] = {str(h): {
                "n": int((hops == h).sum()),
                "ALL": round(float(s["ind"][hops == h].mean()), 4),
                "dALL": round(float(s["ind"][hops == h].mean() - base["ind"][hops == h].mean()), 4),
                "dALL_vs_F6": round(float(s["ind"][hops == h].mean() - f6["ind"][hops == h].mean()), 4),
                "vs_BASE": PP.mcnemar(s["ind"][hops == h], base["ind"][hops == h]),
                "vs_F6": PP.mcnemar(s["ind"][hops == h], f6["ind"][hops == h])}
                for h in (1, 2, 3) if (hops == h).sum()}
        res[name] = e
        log("%-15s %-12s ALL %.4f  dBASE %+.4f (p %.4f)  dF6 %+.4f (p %.4f)" % (
            ds, name, e["ALL"], e["dALL"], e["vs_BASE"]["mcnemar_p"], e["dALL_vs_F6"],
            e["vs_F6"]["mcnemar_p"]))
    OUT.setdefault("RESULTS", {})[ds] = res


def main():
    dsets = sys.argv[1:] or PP.DSETS
    fp = f"{PP.PPD}/diag/round_e.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in dsets:
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote round_e.json")


if __name__ == "__main__":
    main()
