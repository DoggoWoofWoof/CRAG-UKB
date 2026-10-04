"""STEP 10 -- PPR damping ROBUSTNESS check (not a search).

The one global contract is alpha=0.85, iters=20.  Alpha is never selected on gold.  This script
only shows that the phase's conclusion about partition PPR does not hinge on that constant: it
re-runs the two informative corpora at alpha in {0.50, 0.70, 0.85, 0.95} and reports whether the
sign of the verdict changes.  If a different alpha were better we would report it as a finding,
not adopt it.

  python scratchpad/_l1pp_alpha.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1pp_b as B
import _l1pp_d as D
import _l1kb_core as KB
import _l1kb_router as JR

log = lambda *a: print(*a, flush=True)
OUT = {}
for ds in ("metaqa", "hotpotqa_clean", "webqsp"):
    z, meta = PP.load(ds); nq = meta["n_dev_queries"]
    goldp = PP.goldparts(z, meta); psz = z["part_sizes"]
    topo = TA.load_topology(ds, log=lambda *a: None)
    ch = PP.channels(ds, z, meta, topo); npart, tie, hard = ch["npart"], ch["base_rank_replay"], ch["hard"]
    G = PP.partition_graph(ds, topo, log=lambda *a: None)
    C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
    sfull = C["sfull"][(B.M_STRUCT, "S4")]
    S = PP.seed_personalization(z, hard, npart, nq)
    Pm = PP.transition(G, npart)
    Cd = B.contrib_full(ch["PR_d"], npart); Cs = B.contrib_full(ch["PR_s"], npart)
    base = PP.score_top50(B.direct_top50(Cd + Cs, tie), goldp, psz)
    ctxs = KB.contexts(z, meta, C, 6)
    s4 = PP.score_top50(D.swap(ctxs, [D.list_channel(sfull)], True, 6), goldp, psz)
    row = {"BASE": 0.0, "S4_F6_B6": round(s4["ALL"] - base["ALL"], 4)}
    for a in (0.50, 0.70, 0.85, 0.95):
        m = B.global_ppr(S, Pm, alpha=a)
        Cp, _ = B.contrib_mass(m, tie, npart)
        d = PP.score_top50(B.direct_top50(Cd + Cs + Cp, tie), goldp, psz)
        sw = PP.score_top50(D.swap(ctxs, [D.mass_channel(m, tie, npart, nq)], True, 6), goldp, psz)
        row[f"DIRECT_PPR_a{a}"] = round(d["ALL"] - base["ALL"], 4)
        row[f"SWAP_PPR_F6_a{a}"] = round(sw["ALL"] - base["ALL"], 4)
    OUT[ds] = row
    log(ds, json.dumps(row))
json.dump(OUT, open(f"{PP.PPD}/diag/alpha_sensitivity.json", "w"), indent=1)
log("wrote alpha_sensitivity.json")
