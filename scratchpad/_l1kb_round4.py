"""ROUND 4 -- MODALITY-BALANCED SLOT ALLOCATION.

OBSERVED FAILURE (round 3):
  De-duplicating the redundant lexical vote works exactly as predicted on the KB axis --
  MetaQA hop3 +0.0015 (F6) -> +0.0135 (CANON_DEEP), hop2 +0.0060 -> +0.0255, beating both
  ES (-retrieval, +0.0105/+0.0180) and J1 (+0.0120/+0.0195).  But EVERY de-dup variant pays
  on HotpotQA: F6 +0.0160 -> MINRANK +0.0025 -> CANON -0.0015.  The channel-redundancy
  diagnostic says why: the fraction of retrieval challengers the canonical channel already
  ranks is 1.0000 on musique/squad, 0.9986 on 2wiki, 0.9534 metaqa, 0.9378 webqsp -- but only
  0.8148 on hotpot, where 56% of queries get a genuinely NEW retrieval nomination.  So on
  hotpot the second lexical vote is real CORROBORATION, not double counting, and removing it
  destroys evidence.  Re-weighting the score is therefore the wrong instrument: it cannot be
  right for both corroboration and redundancy at once.

HYPOTHESIS: the quantity that has to be shared fairly is not SCORE WEIGHT but SLOTS.  Leave
  the frozen F6 score untouched -- so corroboration keeps its full value wherever it is real --
  and instead let the two MODALITIES alternate as they claim the B boundary slots.  Structure
  then cannot be crowded out of every slot by lexical evidence (the KB failure), and lexical
  evidence keeps ceil(B/2) slots it can spend on corroborated candidates (the text gain).
  1:1 is the parameter-free fair split; nothing is tuned.

FAMILIES (all parameter-free; no dataset identity, no fitted threshold, no gold, no learning)

  ALT    alternate modality while filling the B slots.  A LEXICAL round takes the unclaimed
         candidate with the best lexical evidence; a STRUCTURAL round takes the unclaimed
         candidate with the best structural evidence.  Within a round, candidates are ordered
         by the FROZEN F6 total, restricted to those the round's modality actually scores; a
         round whose modality has no unclaimed evidence falls through to the other one, so the
         router never leaves a slot empty and never invents a candidate.
    lead LEX  lexical moves first (default: lexical is the incumbent modality)
    lead STR  structural moves first
  DISC   structural rounds consider ONLY structure-only discovery candidates (no canonical
         rank, no retrieval rank) -- the partitions no lexical channel can see at all.
  HALF   non-alternating split: the first ceil(B/2) slots are filled by the lexical order and
         the rest by the structural order.

  crossed with the round-3 query-local gates (ALWAYS / DEEP / SEED), the frozen abstention
  gates DOM / PAIR / MAJ, and B in {1,2,4,6}.  Every gate falls back to EXACTLY F6.

  python scratchpad/_l1kb_round4.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT
import _l1kb_core as KB
import _l1kb_router as JR
import _l1kb_round3 as R3

INF = JR.INF
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def make_alt(sched="ALT", lead="LEX", disc=False, when="ALWAYS", gate=None, groups_all=None):
    def sel(c, qi, extra):
        B = len(c["bnd"])
        cpos, spos, rpos = c["cpos"], c["spos"], c["rpos"]
        if when == "ALWAYS":
            on = True
        elif when == "DEEP":
            on = bool(R3.deep_struct_only(c))
        elif when == "SEED":
            on = bool(R3.seed_confident(c, qi, groups_all))
        else:
            raise ValueError(when)
        if not on:
            return KB.sel_f6(c, qi, extra)

        cands = c["bnd"] + [p for p in dict.fromkeys(c["chal"]) if p not in c["bnd"]]
        tot = {}; sc3 = {}
        for p in cands:
            lc = 1.0 / (KB.K0 + cpos[p]) if p in cpos else 0.0
            lr = 1.0 / (KB.K0 + rpos[p]) if p in rpos else 0.0
            st = 1.0 / (KB.K0 + spos[p]) if p in spos else 0.0
            tot[p] = lc + lr + st                      # FROZEN F6 score, untouched
            sc3[p] = (lc + lr, st, 0.0)
        b50 = c["base50"]
        dsc = set(R3.deep_struct_only(c)) if disc else None
        key = lambda p: (-tot[p], cpos.get(p, INF), p)
        lexo = sorted([p for p in cands if p in cpos or p in rpos], key=key)
        stro = sorted([p for p in cands
                       if p in spos and (dsc is None or p in dsc or p in b50)], key=key)
        if sched == "HALF":
            turns = ["LEX"] * ((B + 1) // 2) + ["STR"] * (B // 2)
            if lead == "STR":
                turns = ["STR"] * ((B + 1) // 2) + ["LEX"] * (B // 2)
        else:
            first = lead
            other = "STR" if first == "LEX" else "LEX"
            turns = [first if i % 2 == 0 else other for i in range(B)]
        X = []; taken = set()
        it = {"LEX": iter(lexo), "STR": iter(stro)}
        pend = {"LEX": None, "STR": None}

        def nxt(m):
            if pend[m] is not None and pend[m] not in taken:
                return pend[m]
            for p in it[m]:
                if p not in taken:
                    pend[m] = p
                    return p
            pend[m] = None
            return None
        for t in turns:
            p = nxt(t)
            if p is None:
                p = nxt("STR" if t == "LEX" else "LEX")
            if p is None:
                rest = [q for q in sorted(cands, key=key) if q not in taken]
                if not rest:
                    break
                p = rest[0]
            X.append(p); taken.add(p); pend[t] = None
        if gate is not None:
            A = [p for p in X if p not in c["bnd"]]
            D = [p for p in c["bnd"] if p not in X]
            if A and not JR.gate_ok(gate, A, D, sc3, tot):
                return list(c["bnd"])
        return X
    return sel


def N(sched="ALT", lead="LEX", disc=False, when="ALWAYS", gate=None, B=6):
    nm = sched + ("_STRLEAD" if lead == "STR" else "") + ("_DISC" if disc else "") \
         + ("" if when == "ALWAYS" else "_" + when) + ("" if gate is None else "_" + gate) \
         + "_B%d" % B
    return dict(name=nm, kind="ALT", sched=sched, lead=lead, disc=disc, when=when,
                gate=gate, B=B)


GRID = [dict(name="BASE_B6", kind="BASE", B=6), dict(name="F6_B6", kind="F6", B=6),
        dict(name="MINRANK_SEED_B6", kind="D", lex="MINRANK", when="SEED", gate=None,
             prio=False, B=6),
        dict(name="CANON_DEEP_B6", kind="D", lex="CANON", when="DEEP", gate=None,
             prio=False, B=6)]
for sched in ("ALT", "HALF"):
    for lead in ("LEX", "STR"):
        for B in (6, 4, 2):
            GRID.append(N(sched, lead, B=B))
for lead in ("LEX", "STR"):
    GRID.append(N("ALT", lead, disc=True, B=6))
    for when in ("DEEP", "SEED"):
        GRID.append(N("ALT", lead, when=when, B=6))
for gate in ("DOM", "PAIR", "MAJ"):
    GRID.append(N("ALT", "LEX", gate=gate, B=6))
GRID.append(N("ALT", "LEX", disc=True, when="SEED", B=6))
GRID.append(N("ALT", "LEX", B=1))


def build(spec, GRP):
    k = spec["kind"]
    if k == "BASE":
        return KB.sel_base
    if k == "F6":
        return KB.sel_f6
    if k == "D":
        return R3.make_dsel(spec["lex"], spec["when"], spec["gate"], spec["prio"], GRP)
    if k == "ALT":
        return make_alt(spec["sched"], spec["lead"], spec["disc"], spec["when"],
                        spec["gate"], GRP)
    raise ValueError(k)


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
        assert res["BASE_B6"]["dALL"] == 0.0
        SB[ds] = {"meta": {"n": nq, "BASE_ALL": round(float(ind_base.mean()), 4),
                           "BASE_ANY": meta["BASE_ANY_P50"]}, "configs": res}
        log("%-15s " % ds + "  ".join("%s:%+.4f" % (k, v["dALL"]) for k, v in res.items()
                                      if k != "BASE_B6"))
    fp = f"{KB.KBD}/scoreboard_round4.json"
    json.dump({"ROUND": "round4", "GRID": GRID, "SCOREBOARD": SB}, open(fp, "w"), indent=1)
    log("wrote", fp)


if __name__ == "__main__":
    main()
