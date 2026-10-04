"""STEP 4/5 -- SET-AWARE FIXED-P50 PARTITION ROUTING (KB multi-hop phase).

    final = base_rank[:50-B]  u  X,   X subset of (boundary u challengers),  |X| = B  ==> 50 out.

Every family below chooses X as a SET.  The candidate items are

    each boundary incumbent                                   (singleton, always available)
    each retrieval-continuation challenger                    (singleton, families that allow it)
    each structural PATH GROUP                                (ATOMIC: all its payable partitions)

and the option X = boundary (ABSTAIN, change nothing) is always in the enumeration, so the router
can always decline.  Selection is exhaustive over subsets of the top-G path groups; the remaining
slots are filled with the best singletons.  This is combinatorial search over a frozen,
parameter-free score -- no weights, no thresholds, no learned anything.

Families
  BASE   keep the boundary, never swap                         (lower control)
  F6     the frozen safe universal router, called directly     (upper control / parity gate)
  F6E    enumeration with singletons only, no groups, no gate  (machinery parity: must equal F6)
  J1     structural challengers may enter ONLY as complete chains; no retrieval singletons
  J1R    J1 + retrieval-continuation singletons                (anticipates J4)
  J3x    incumbent-aware GROUP SWAP gates on top of J1R/F6E:
           DOM   min(score over admitted) > max(score over evicted)
           PAIR  rank-matched: i-th best admitted > i-th best evicted, for all i
           MAJ   admitted set wins on >= 2 of {canonical, structural, retrieval} channel sums

  python scratchpad/_l1kb_router.py <round>
"""
import os, sys, json, time, hashlib, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT
import _l1kb_core as KB

ROUND = sys.argv[1] if len(sys.argv) > 1 else "round1"
G_TOP = 8            # global constant: how many path groups enter the exhaustive subset search
INF = 10 ** 6
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ path groups
def build_groups(z, PZ, meta, hard, M_struct):
    """per query: deduplicated ordered partition chains with their deterministic evidence."""
    nq = meta["n_dev_queries"]
    sn = z["s_node"]; sd = z["s_sdir"]
    out = []
    for qi in range(nq):
        pr = PZ["p_root"][qi]; pa = PZ["p_a"][qi]; pb = PZ["p_b"][qi]
        pl = PZ["p_len"][qi]; ss = PZ["p_seedslot"][qi]
        seen = {}
        for j in range(min(sn.shape[1], M_struct)):
            v = int(sn[qi, j])
            if v < 0:
                break
            ps = []
            for x in (int(pr[j]), int(pa[j]), int(pb[j]), v):
                if x < 0:
                    continue
                p = int(hard[x])
                if p >= 0 and (not ps or ps[-1] != p):
                    ps.append(p)
            if not ps:
                continue
            key = tuple(ps)
            g = seen.get(key)
            if g is None:
                seen[key] = {"parts": key, "term": int(hard[v]), "depth": int(pl[j]),
                             "jbest": j, "n_nodes": 1, "sdir": float(sd[qi, j]),
                             "seedslots": {int(ss[j])}}
            else:
                g["n_nodes"] += 1
                g["sdir"] = max(g["sdir"], float(sd[qi, j]))
                g["seedslots"].add(int(ss[j]))
        gs = sorted(seen.values(), key=lambda g: (g["jbest"], g["parts"]))
        for g in gs:
            g["seedslots"] = sorted(g["seedslots"])
        out.append(gs)
    return out


def payable(g, prot_set):
    return frozenset(p for p in g["parts"] if p not in prot_set)


# ------------------------------------------------------------------ set selection
def channel_scores(c):
    """the three frozen channel contributions, per candidate partition."""
    cpos, spos, rpos = c["cpos"], c["spos"], c["rpos"]

    def f(p):
        return (1.0 / (KB.K0 + cpos[p]) if p in cpos else 0.0,
                1.0 / (KB.K0 + spos[p]) if p in spos else 0.0,
                1.0 / (KB.K0 + rpos[p]) if p in rpos else 0.0)
    return f


def gate_ok(rule, A, D, sc3, tot):
    """A = admitted partitions, D = evicted incumbents (|A| == |D|)."""
    if not A:
        return True
    if rule is None:
        return True
    if rule == "DOM":
        return min(tot[p] for p in A) > max(tot[p] for p in D)
    if rule == "PAIR":
        a = sorted((tot[p] for p in A), reverse=True)
        d = sorted((tot[p] for p in D), reverse=True)
        return all(x > y for x, y in zip(a, d))
    if rule == "MAJ":
        w = 0
        for k in range(3):
            sa = sum(sc3[p][k] for p in A); sdd = sum(sc3[p][k] for p in D)
            w += int(sa > sdd)
        return w >= 2
    raise ValueError(rule)


def struct_channel(c, qi, groups_all, mode):
    """the structural channel's partition ordering.  All three are symmetric (incumbents keep
    their structural vote) and parameter-free.

      S4   frozen MULTI_SIGNAL_RRF over the partitions of the structural residual nodes
      S1T  the same partitions ordered by best structural node rank alone (isolates S4)
      CP   CHAIN-PROPAGATED: every partition ON a provenance chain inherits that chain's best
           structural node rank, so path intermediates -- which no independent channel scores at
           all -- become rankable.  This is the INDEPENDENT-scoring control for path groups.
    """
    if mode == "S4":
        return c["spos"]
    if mode == "S1T":
        agg = c["sagg"]
        return {p: r for r, p in enumerate(sorted(agg, key=lambda p: (agg[p][0], p)))}
    if mode == "CP":
        best = {}
        for g in groups_all[qi]:
            for p in g["parts"]:
                if p not in best or g["jbest"] < best[p]:
                    best[p] = g["jbest"]
        return {p: r for r, p in enumerate(sorted(best, key=lambda p: (best[p], p)))}
    raise ValueError(mode)


# family -> (struct singletons allowed, retrieval singletons allowed, path groups allowed)
FAM = {"F6E": (1, 1, 0), "ES": (1, 0, 0), "ER": (0, 1, 0),
       "J1": (0, 0, 1), "J1R": (0, 1, 1), "J1S": (1, 0, 1), "J1SR": (1, 1, 1)}


def make_selector(fam, gate=None, groups_all=None, schan="S4", g_top=G_TOP):
    """returns selector(c, qi, extra) -> list of exactly B partitions."""
    allow_struct_single, allow_ret_single, use_groups = FAM[fam]

    def sel(c, qi, extra):
        B = len(c["bnd"])
        bnd = list(c["bnd"]); prot = c["prot_set"]; b50 = c["base50"]
        spos = struct_channel(c, qi, groups_all, schan)
        rpos = c["rpos"]; cpos = c["cpos"]

        def f(p):
            return (1.0 / (KB.K0 + cpos[p]) if p in cpos else 0.0,
                    1.0 / (KB.K0 + spos[p]) if p in spos else 0.0,
                    1.0 / (KB.K0 + rpos[p]) if p in rpos else 0.0)
        sc3 = {}; tot = {}

        def add(p):
            if p not in sc3:
                sc3[p] = f(p); tot[p] = sum(sc3[p])
        for p in bnd:
            add(p)

        singles = list(bnd)
        if allow_struct_single:
            singles += [p for p in spos if p not in b50]
        if allow_ret_single:
            singles += [p for p in rpos if p not in b50]
        singles = list(dict.fromkeys(singles))
        for p in singles:
            add(p)
        singles.sort(key=lambda p: (-tot[p], cpos.get(p, INF), p))

        gopts = []
        if use_groups:
            for g in groups_all[qi][:g_top]:
                pay = payable(g, prot)
                if 1 <= len(pay) <= B and any(p not in b50 for p in pay):
                    for p in pay:
                        add(p)
                    gopts.append(pay)
            gopts = list(dict.fromkeys(gopts))[:g_top]

        # exhaustive over subsets of the top-G path groups, pruned by |U| <= B (a union that
        # already overspends the budget can never be extended into a legal one).  X = boundary
        # (ABSTAIN) is the r=0 leaf, so declining is always in the enumeration.
        best = [None]; seen = set(); bset = set(bnd); nb = len(gopts)

        def consider(U):
            if U in seen:
                return
            seen.add(U)
            X = list(U)
            for p in singles:
                if len(X) >= B:
                    break
                if p not in U:
                    X.append(p)
            if len(X) != B:
                return
            XS = set(X)
            A = sorted(XS - b50); D = sorted(bset - XS)
            if not gate_ok(gate, A, D, sc3, tot):
                return
            key = (-sum(tot[p] for p in X), tuple(sorted(cpos.get(p, INF) for p in X)),
                   tuple(sorted(X)))
            if best[0] is None or key < best[0][0]:
                best[0] = (key, X)

        def dfs(i, U):
            consider(U)
            for j in range(i, nb):
                V = U | gopts[j]
                if len(V) <= B:
                    dfs(j + 1, V)
        dfs(0, frozenset())
        return bnd if best[0] is None else best[0][1]
    return sel


# ------------------------------------------------------------------ J4 -- path completion
def make_complete_selector(admit, groups_all, g_top=G_TOP, frontier=False, single=None,
                           mode="BOTH"):
    """J4 / STEP 7.  Start from the FROZEN F6 selection, then COMPLETE any structural chain that
    F6 already anchored: if a selected partition lies on a provenance chain whose other payable
    partitions were left out, pull them in and displace the weakest selected partitions that are
    not on that chain.

    This is a repair operator, not a competing objective, which is why it is universality-safe:
    on a corpus whose chains are depth-1 the payable set is already inside X and the operator is
    a no-op, so the frozen text behaviour is preserved bit-for-bit.

    admit -- the deterministic admissibility test for one completion:
      FREE    always complete when affordable                (mechanism upper bound)
      ANCHOR  every displaced partition must score strictly below the chain's own anchor, i.e.
              below the member that already won a slot on its own merit
      SUM     the incoming chain remainder must outscore the partitions it displaces (J3 applied
              to the completion: challenger SET vs incumbent SET)
    frontier -- J2: at most one completion per distinct structural frontier (originating seed)
    single   -- J5: predicate a depth-1 chain must satisfy to be completed at all
    mode     -- which half of the mechanism is enabled:
      ADMIT   complete only with partitions from OUTSIDE the base top-50 (pure co-admission)
      RETAIN  complete only by re-admitting boundary partitions F6 dropped (chain preservation,
              STEP 7: do not evict P_b just because some isolated P_c scores higher)
      BOTH    both
    """
    def sel(c, qi, extra):
        B = len(c["bnd"])
        bnd = list(c["bnd"]); prot = c["prot_set"]; b50 = c["base50"]
        spos, rpos, cpos = c["spos"], c["rpos"], c["cpos"]

        def f(p):
            return ((1.0 / (KB.K0 + cpos[p]) if p in cpos else 0.0)
                    + (1.0 / (KB.K0 + spos[p]) if p in spos else 0.0)
                    + (1.0 / (KB.K0 + rpos[p]) if p in rpos else 0.0))
        X, _ = KB.f6_select(bnd, c["chal"], spos, rpos, cpos, B)
        tot = {p: f(p) for p in X}
        XS = set(X)
        used_frontier = set()
        for g in groups_all[qi][:g_top]:
            pay = payable(g, prot)
            if not pay or len(pay) > B:
                continue
            anch = pay & XS
            miss = pay - XS
            if not anch or not miss:
                continue
            if mode == "ADMIT" and any(p in b50 for p in miss):
                continue
            if mode == "RETAIN" and any(p not in b50 for p in miss):
                continue
            if frontier and used_frontier & set(g["seedslots"]):
                continue
            if single is not None and g["depth"] <= 1 and not single(g, c, qi):
                continue
            for p in pay:
                if p not in tot:
                    tot[p] = f(p)
            rem = sorted((p for p in XS if p not in pay),
                         key=lambda p: (tot[p], -cpos.get(p, INF), -p))
            if len(rem) < len(miss):
                continue
            drop = rem[:len(miss)]
            if admit == "ANCHOR":
                a = max(tot[p] for p in anch)
                if not all(tot[d] < a for d in drop):
                    continue
            elif admit == "SUM":
                if not sum(tot[p] for p in miss) > sum(tot[d] for d in drop):
                    continue
            elif admit != "FREE":
                raise ValueError(admit)
            XS = (XS - set(drop)) | miss
            used_frontier |= set(g["seedslots"])
        return sorted(XS)
    return sel


def multi_seed(g, c, qi):
    """J5: a depth-1 structural candidate is path-stable only if two distinct retrieval seeds
    independently reach it."""
    return len(g["seedslots"]) >= 2


def strong_dir(g, c, qi):
    """J5: ... or if it is the query-local best directional candidate of its own frontier."""
    return g["jbest"] == 0


# ------------------------------------------------------------------ evaluation
def folds(ds, z):
    rows = z["rows"]
    return np.array([int(hashlib.sha1(f"{ds}:{int(r)}".encode()).hexdigest(), 16) % 2
                     for r in rows], np.int8)


def score_run(res, ind_base, hops, fold, churn):
    ind = res["ind"]
    m = RT.mcnemar(ind, ind_base)
    o = {"ALL": round(float(ind.mean()), 4),
         "dALL": round(float(ind.mean() - ind_base.mean()), 4),
         "ANY": round(float(res["ind_any"].mean()), 4),
         "newly_covered": m["gained"], "newly_uncovered": m["lost"], "net": m["net"],
         "mcnemar_p": m["mcnemar_p"], "sig": m["sig"],
         "gold_admitted": res["gold_admitted"], "gold_evicted": res["gold_evicted"],
         "net_gold": res["gold_admitted"] - res["gold_evicted"],
         "churn_mean": round(float(churn.mean()), 3),
         "churn_hist": {str(k): int(v) for k, v in zip(*np.unique(churn, return_counts=True))}}
    for nm, msk in (("DISCOVERY", fold == 0), ("VALIDATION", fold == 1)):
        mm = RT.mcnemar(ind[msk], ind_base[msk])
        o[nm] = {"n": int(msk.sum()), "ALL": round(float(ind[msk].mean()), 4),
                 "dALL": round(float(ind[msk].mean() - ind_base[msk].mean()), 4),
                 "net": mm["net"], "mcnemar_p": mm["mcnemar_p"], "sig": mm["sig"]}
    if (hops >= 0).any():
        o["per_hop"] = {}
        for h in sorted(set(int(x) for x in hops if x >= 0)):
            hm = hops == h
            mm = RT.mcnemar(ind[hm], ind_base[hm])
            o["per_hop"][str(h)] = {"n": int(hm.sum()),
                                    "BASE": round(float(ind_base[hm].mean()), 4),
                                    "ALL": round(float(ind[hm].mean()), 4),
                                    "dALL": round(float(ind[hm].mean() - ind_base[hm].mean()), 4),
                                    "newly_covered": mm["gained"],
                                    "newly_uncovered": mm["lost"], "net": mm["net"],
                                    "mcnemar_p": mm["mcnemar_p"]}
    return o


def build_sel(spec, GRP):
    """one dispatch point: a grid entry is a dict describing exactly one universal router."""
    k = spec["kind"]
    if k == "BASE":
        return KB.sel_base
    if k == "F6":
        return KB.sel_f6
    if k == "SET":
        return make_selector(spec["fam"], spec.get("gate"), GRP, spec.get("schan", "S4"))
    if k == "COMPLETE":
        return make_complete_selector(spec["admit"], GRP, frontier=spec.get("frontier", False),
                                      single={"MSEED": multi_seed, "SDIR": strong_dir}.get(
                                          spec.get("single")),
                                      mode=spec.get("mode", "BOTH"))
    raise ValueError(k)


def C4(admit, B=6, **kw):
    nm = "J4_" + admit + ("_" + kw["mode"] if kw.get("mode", "BOTH") != "BOTH" else "")          + ("_FRONT" if kw.get("frontier") else "")          + ("_" + kw["single"] if kw.get("single") else "") + f"_B{B}"
    return dict(name=nm, kind="COMPLETE", admit=admit, B=B, **kw)


GRIDS = {
    # ROUND 1 -- J1 (path-complete promotion) and J3 (incumbent-aware group swap gates), with the
    # controls that separate the two things a path group can be doing:
    #   E_CP   independent scoring, chain-propagated structural channel  -> pure REACH
    #   J1     atomic whole-chain promotion                              -> REACH + ATOMICITY
    # (fam, gate, B, struct-channel)
    "round1": [("BASE", None, 6, "S4"), ("F6", None, 6, "S4"), ("F6E", None, 6, "S4"),
               ("F6E", None, 6, "S1T"), ("F6E", None, 6, "CP"),
               ("ES", None, 6, "CP"), ("ER", None, 6, "S4"),
               ("J1", None, 6, "S4"), ("J1", None, 4, "S4"), ("J1", None, 2, "S4"),
               ("J1R", None, 6, "S4"), ("J1R", None, 4, "S4"), ("J1R", None, 2, "S4"),
               ("J1SR", None, 6, "S4"),
               ("J1", "DOM", 6, "S4"), ("J1", "PAIR", 6, "S4"), ("J1", "MAJ", 6, "S4"),
               ("J1R", "DOM", 6, "S4"), ("J1R", "PAIR", 6, "S4"), ("J1R", "MAJ", 6, "S4"),
               ("F6E", "DOM", 6, "S4"), ("F6E", "PAIR", 6, "S4"), ("F6E", "MAJ", 6, "S4")],
    # ROUND 2 -- J4 (structural chain + retrieval complement) and J2 (frontier coverage).
    # Observed round-1 failure: J1 wins MetaQA by forbidding retrieval singletons, which is
    # exactly what makes it lose on text; J1R restores text and loses the KB gain; and J1SR
    # (both allowed) collapses EXACTLY onto F6, because under a modular objective a group is
    # redundant whenever its members can be taken individually.  Hypothesis: the mechanism has
    # to be a REPAIR on top of F6, not a rival objective -- complete a chain F6 already anchored.
    "round2": [dict(name="BASE_B6", kind="BASE", B=6), dict(name="F6_B6", kind="F6", B=6),
               dict(name="J1_B6", kind="SET", fam="J1", B=6),
               dict(name="J1R_B6", kind="SET", fam="J1R", B=6),
               C4("FREE"), C4("ANCHOR"), C4("SUM"),
               C4("FREE", mode="ADMIT"), C4("FREE", mode="RETAIN"),
               C4("ANCHOR", mode="ADMIT"), C4("ANCHOR", mode="RETAIN"),
               C4("SUM", mode="ADMIT"), C4("SUM", mode="RETAIN"),
               C4("ANCHOR", frontier=True), C4("FREE", frontier=True),
               C4("ANCHOR", single="MSEED"), C4("ANCHOR", single="SDIR"),
               C4("FREE", B=4), C4("ANCHOR", B=4), C4("ANCHOR", B=2),
               C4("SUM", B=4)],
}


def main():
    grid = GRIDS[ROUND]
    SB = {}
    for ds in KB.DSETS:
        z, meta = KB.load(ds)
        nq = meta["n_dev_queries"]; hard = z["hard"]; hops = z["hops"]
        goldp = KB.goldparts(z, meta)
        C, GRP = KB.substrate(ds, z, meta, build_groups)
        ind_base = C["ind_base"]
        fold = folds(ds, z)
        ctxs = {B: KB.contexts(z, meta, C, B) for B in sorted({g["B"] for g in grid})}
        ref = RT.evaluate(ds, z, meta, dict(KB.BASE_CFG), C); ref.pop("_ind")
        res = {}
        for spec in grid:
            name = spec["name"]; B = spec["B"]
            if name in res:
                continue
            t = time.time()
            sel = build_sel(spec, GRP)
            r = KB.run_selector(ctxs[B], sel, goldp, ind_base)
            res[name] = score_run(r, ind_base, hops, fold, r["churn"])
            res[name]["sec"] = round(time.time() - t, 1)
        assert res["F6_B6"]["ALL"] == ref["ALL"], f"{ds}: F6 replay drift"
        if "F6E_B6" in res:
            assert res["F6E_B6"]["ALL"] == ref["ALL"], \
                f"{ds}: enumeration parity broken ({res['F6E_B6']['ALL']} vs {ref['ALL']})"
        assert res["BASE_B6"]["dALL"] == 0.0
        SB[ds] = {"meta": {"n": nq, "BASE_ALL": round(float(ind_base.mean()), 4),
                           "BASE_ANY": meta["BASE_ANY_P50"]}, "configs": res}
        log(f"{ds:15s} " + "  ".join(f"{k}:{v['dALL']:+.4f}" for k, v in res.items()
                                     if k not in ("BASE_B6",)))

    names = sorted(set.intersection(*[set(SB[d]["configs"]) for d in SB]))
    rows = []
    for n in names:
        d = {ds: SB[ds]["configs"][n] for ds in SB}
        mh = d["metaqa"].get("per_hop", {})
        rows.append({"config": n,
                     "worst_dALL": round(min(v["dALL"] for v in d.values()), 4),
                     "macro_dALL": round(float(np.mean([v["dALL"] for v in d.values()])), 4),
                     "n_sig_regressions": sum(1 for v in d.values() if v["sig"] and v["dALL"] < 0),
                     "metaqa_hop2_d": mh.get("2", {}).get("dALL"),
                     "metaqa_hop3_d": mh.get("3", {}).get("dALL"),
                     "webqsp_dALL": d["webqsp"]["dALL"], "webqsp_ANY": d["webqsp"]["ANY"],
                     "churn": round(float(np.mean([v["churn_mean"] for v in d.values()])), 3),
                     "net_gold": {ds: d[ds]["net_gold"] for ds in d},
                     "per_ds": {ds: d[ds]["dALL"] for ds in d}})
    rows.sort(key=lambda r: (r["n_sig_regressions"], -(r["metaqa_hop3_d"] or -9),
                             -(r["metaqa_hop2_d"] or -9), -r["worst_dALL"], -r["macro_dALL"]))
    fp = f"{KB.KBD}/scoreboard_{ROUND}.json"
    json.dump({"round": ROUND, "G_TOP": G_TOP, "SCOREBOARD": SB, "RANKING": rows},
              open(fp, "w"), indent=1)
    print(f"\n{'config':18s} {'worst':>8s} {'macro':>8s} {'reg':>4s} {'mq_h2':>7s} {'mq_h3':>7s} "
          f"{'webq':>8s} {'churn':>6s}")
    for r in rows:
        print(f"{r['config']:18s} {r['worst_dALL']:+8.4f} {r['macro_dALL']:+8.4f} "
              f"{r['n_sig_regressions']:4d} {r['metaqa_hop2_d']:+7.4f} {r['metaqa_hop3_d']:+7.4f} "
              f"{r['webqsp_dALL']:+8.4f} {r['churn']:6.2f}")
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
