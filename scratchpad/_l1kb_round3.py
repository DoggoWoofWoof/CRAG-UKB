"""ROUND 3 -- MODALITY DE-DUPLICATION + QUERY-LOCAL DEPTH/NEED ALLOCATION (STEP 6, J5).

OBSERVED FAILURE (round 2 + the struct-only control):
  F6 scores a candidate with THREE channels -- canonical, structural, retrieval -- but the
  canonical and retrieval channels are the SAME MODALITY at two granularities:
  _l1ps_router.ret_challengers reads z["ret_rrf"], the dense+SPLADE rrf200 NODE continuation,
  and z["base_rank"] is that identical rrf200 evidence aggregated to partitions.  So a
  boundary-adjacent partition collects two correlated lexical votes while a structural
  challenger collects one.  Measured consequence on MetaQA:
        F6  hop2 +0.0060  hop3 +0.0015
        ES  hop2 +0.0180  hop3 +0.0105   (identical router, retrieval channel removed)
        J1  hop2 +0.0195  hop3 +0.0120   (ES + atomic path groups)
  i.e. 87.5% of the recoverable hop3 gain is CHANNEL REDUNDANCY, not path atomicity.
  But ES is not universal (2wiki -0.0020, hotpot +0.0055 vs F6 +0.0160): dropping retrieval
  globally trades the text gain away.

HYPOTHESIS: the KB/text tension is modality DOUBLE-COUNTING, not "structure vs retrieval".
  Removing the redundant lexical vote -- parameter-free, per candidate -- should recover the
  KB gain without the text loss, because on text the structural channel simply has little
  evidence to promote with the freed weight.

FAMILIES (all parameter-free; no dataset identity, no fitted threshold, no gold, no learning)

  lexical term for candidate p, given canonical rank cpos[p] and retrieval rank rpos[p]:
    SUM      1/(K0+cpos) + 1/(K0+rpos)            FROZEN F6 (double count)
    MINRANK  1/(K0+min(cpos,rpos))                one modality, best granularity wins
    CANON    1/(K0+cpos) if canonical else ret    canonical precedence
    NOVEL    SUM, but the retrieval term counts only when p has NO canonical rank
                                                  (retrieval must DISAGREE to get a vote)

  query-local gate `when` -- which queries the de-duplication is applied to at all.  When the
  condition is false the router falls back to EXACTLY F6, so ABSTENTION is the default action:
    ALWAYS   unconditional
    DEEP     the query has >=1 structure-only challenger (no canonical, no retrieval rank)
             whose minimum TRAVERSAL hop is >= 2  -- query-local multi-hop demand.  This is the
             graph traversal depth already used by frozen S4, never a benchmark hop label.
    SEED     J5 single-node confidence: >=1 challenger whose chain is reached from >=2 distinct
             retrieval seed slots, or which is the query-local best (jbest==0) of its frontier.
             Deliberately NOT "needs >=2 structural nodes".
    NOVELQ   the retrieval channel nominates >=1 out-of-P50 partition the canonical channel
             does not rank at all -- query-local evidence that retrieval is not redundant here.

  PRIO variant: instead of reweighting, give the structure-only deep class lexicographic first
  refusal on slots, F6 order inside each class.

  crossed with the frozen abstention gates DOM / PAIR / MAJ and B in {1,2,4,6}.

  python scratchpad/_l1kb_round3.py [diag|run]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT
import _l1kb_core as KB
import _l1kb_router as JR

MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
INF = JR.INF
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ query-local predicates
def deep_struct_only(c):
    """structure-only challengers (no canonical rank, no retrieval rank) reached at traversal
    hop >= 2.  Pure discovery: no lexical channel scores them at all."""
    b50 = c["base50"]; sagg = c["sagg"]
    return [p for p in c["spos"]
            if p not in b50 and p not in c["cpos"] and p not in c["rpos"]
            and p in sagg and sagg[p][3] >= 2]


def seed_confident(c, qi, groups_all):
    """J5: challengers whose provenance chain is reached from >=2 distinct retrieval seed slots,
    or which are the query-local best candidate of their own frontier."""
    b50 = c["base50"]; out = set()
    for g in groups_all[qi]:
        if len(g["seedslots"]) >= 2 or g["jbest"] == 0:
            out |= {p for p in g["parts"] if p not in b50}
    return out


def ret_is_novel(c):
    """retrieval nominates something the canonical channel does not rank at all."""
    b50 = c["base50"]
    return any(p not in b50 and p not in c["cpos"] for p in c["rpos"])


# ------------------------------------------------------------------ selector
def make_dsel(lex="MINRANK", when="ALWAYS", gate=None, prio=False, groups_all=None):
    def sel(c, qi, extra):
        B = len(c["bnd"])
        cpos, spos, rpos = c["cpos"], c["spos"], c["rpos"]
        deep = deep_struct_only(c)
        if when == "ALWAYS":
            on = True
        elif when == "DEEP":
            on = bool(deep)
        elif when == "SEED":
            on = bool(seed_confident(c, qi, groups_all))
        elif when == "NOVELQ":
            on = ret_is_novel(c)
        else:
            raise ValueError(when)
        if not on:
            return KB.sel_f6(c, qi, extra)

        def lexterm(p):
            cr = cpos.get(p); rr = rpos.get(p)
            if lex == "SUM":
                return (1.0 / (KB.K0 + cr) if cr is not None else 0.0) + \
                       (1.0 / (KB.K0 + rr) if rr is not None else 0.0)
            if lex == "MINRANK":
                rs = [x for x in (cr, rr) if x is not None]
                return 1.0 / (KB.K0 + min(rs)) if rs else 0.0
            if lex == "CANON":
                if cr is not None:
                    return 1.0 / (KB.K0 + cr)
                return 1.0 / (KB.K0 + rr) if rr is not None else 0.0
            if lex == "NOVEL":
                s = 1.0 / (KB.K0 + cr) if cr is not None else 0.0
                if rr is not None and cr is None:
                    s += 1.0 / (KB.K0 + rr)
                return s
            raise ValueError(lex)

        cands = c["bnd"] + [p for p in dict.fromkeys(c["chal"]) if p not in c["bnd"]]
        tot = {}; sc3 = {}
        for p in cands:
            lt = lexterm(p)
            st = 1.0 / (KB.K0 + spos[p]) if p in spos else 0.0
            tot[p] = lt + st
            sc3[p] = (lt, st, 0.0)
        dset = set(deep) if prio else set()
        order = sorted(cands, key=lambda p: (0 if p in dset else 1, -tot[p],
                                             cpos.get(p, INF), p))
        X = order[:B]
        if gate is not None:
            A = [p for p in X if p not in c["bnd"]]
            D = [p for p in c["bnd"] if p not in X]
            if A and not JR.gate_ok(gate, A, D, sc3, tot):
                return list(c["bnd"])
        return X
    return sel


# ------------------------------------------------------------------ grid
def N(lex, when="ALWAYS", gate=None, prio=False, B=6):
    nm = lex + ("" if when == "ALWAYS" else "_" + when) + ("_PRIO" if prio else "") \
         + ("" if gate is None else "_" + gate) + "_B%d" % B
    return dict(name=nm, kind="D", lex=lex, when=when, gate=gate, prio=prio, B=B)


GRID = [dict(name="BASE_B6", kind="BASE", B=6), dict(name="F6_B6", kind="F6", B=6),
        dict(name="SUM_B6", kind="D", lex="SUM", when="ALWAYS", gate=None, prio=False, B=6),
        dict(name="ES_B6", kind="SET", fam="ES", B=6),
        dict(name="J1_B6", kind="SET", fam="J1", B=6)]
for lex in ("MINRANK", "CANON", "NOVEL"):
    for B in (6, 4, 2):
        GRID.append(N(lex, B=B))
    for when in ("DEEP", "SEED", "NOVELQ"):
        GRID.append(N(lex, when=when, B=6))
for gate in ("DOM", "PAIR", "MAJ"):
    GRID.append(N("MINRANK", gate=gate, B=6))
GRID += [N("MINRANK", prio=True, B=6), N("MINRANK", when="DEEP", prio=True, B=6),
         N("SUM", prio=True, B=6), N("MINRANK", B=1)]


def build(spec, GRP):
    k = spec["kind"]
    if k == "BASE":
        return KB.sel_base
    if k == "F6":
        return KB.sel_f6
    if k == "SET":
        return JR.make_selector(spec["fam"], spec.get("gate"), GRP, "S4")
    if k == "D":
        return make_dsel(spec["lex"], spec["when"], spec["gate"], spec["prio"], GRP)
    raise ValueError(k)


# ------------------------------------------------------------------ channel-redundancy diagnostic
def diag():
    out = {}
    for ds in KB.DSETS:
        z, meta = KB.load(ds)
        C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
        ctxs = KB.contexts(z, meta, C, 6)
        nq = len(ctxs)
        n_ret = n_ret_in_canon = 0
        q_novel = q_deep = q_seed = 0
        n_deep = []; ncan = []
        for qi, c in enumerate(ctxs):
            b50 = c["base50"]
            rr = [p for p in c["rpos"] if p not in b50]
            n_ret += len(rr)
            n_ret_in_canon += sum(1 for p in rr if p in c["cpos"])
            q_novel += int(ret_is_novel(c))
            d = deep_struct_only(c)
            n_deep.append(len(d)); q_deep += int(bool(d))
            q_seed += int(bool(seed_confident(c, qi, GRP)))
            ncan.append(len(c["cpos"]))
        out[ds] = {"n": nq, "ret_chal_total": n_ret,
                   "frac_ret_chal_already_canonical": round(n_ret_in_canon / max(1, n_ret), 4),
                   "frac_q_ret_novel": round(q_novel / nq, 4),
                   "frac_q_has_deep_struct_only": round(q_deep / nq, 4),
                   "mean_deep_struct_only_per_q": round(float(np.mean(n_deep)), 3),
                   "frac_q_seed_confident": round(q_seed / nq, 4),
                   "mean_canonical_ranked": round(float(np.mean(ncan)), 1)}
        log("%-15s ret_chal_in_canon %.4f  q_ret_novel %.4f  q_deep %.4f (%.2f/q)  q_seed %.4f"
            % (ds, out[ds]["frac_ret_chal_already_canonical"], out[ds]["frac_q_ret_novel"],
               out[ds]["frac_q_has_deep_struct_only"], out[ds]["mean_deep_struct_only_per_q"],
               out[ds]["frac_q_seed_confident"]))
    os.makedirs(f"{KB.KBD}/diag", exist_ok=True)
    fp = f"{KB.KBD}/diag/channel_redundancy.json"
    json.dump(out, open(fp, "w"), indent=1)
    log("wrote", fp)


def main():
    SB = {}
    for ds in KB.DSETS:
        z, meta = KB.load(ds)
        nq = meta["n_dev_queries"]; hops = z["hops"]
        goldp = KB.goldparts(z, meta)
        C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
        ind_base = C["ind_base"]
        fold = JR.folds(ds, z)
        ctxs = {B: KB.contexts(z, meta, C, B) for B in sorted({g["B"] for g in GRID})}
        ref = RT.evaluate(ds, z, meta, dict(KB.BASE_CFG), C); ref.pop("_ind")
        res = {}
        for spec in GRID:
            if spec["name"] in res:
                continue
            t = time.time()
            r = KB.run_selector(ctxs[spec["B"]], build(spec, GRP), goldp, ind_base)
            res[spec["name"]] = JR.score_run(r, ind_base, hops, fold, r["churn"])
            res[spec["name"]]["sec"] = round(time.time() - t, 1)
        assert res["F6_B6"]["ALL"] == ref["ALL"], f"{ds}: F6 replay drift"
        assert res["SUM_B6"]["ALL"] == ref["ALL"], \
            f"{ds}: D-selector parity broken ({res['SUM_B6']['ALL']} vs {ref['ALL']})"
        assert res["BASE_B6"]["dALL"] == 0.0
        SB[ds] = {"meta": {"n": nq, "BASE_ALL": round(float(ind_base.mean()), 4),
                           "BASE_ANY": meta["BASE_ANY_P50"]}, "configs": res}
        log("%-15s " % ds + "  ".join("%s:%+.4f" % (k, v["dALL"]) for k, v in res.items()
                                      if k != "BASE_B6"))
    fp = f"{KB.KBD}/scoreboard_round3.json"
    json.dump({"ROUND": "round3", "GRID": GRID, "SCOREBOARD": SB}, open(fp, "w"), indent=1)
    log("wrote", fp)


if __name__ == "__main__":
    (diag if MODE == "diag" else main)()
