"""DEV_A only: do the two mechanisms that moved musique (directional block signal Cdir; extra point 2q - mu under the
interleave) add up when combined?  Evaluated through the candidate module's channels so the numbers are the candidate's.
Combos are exploration (DEV_A); whatever is pre-registered afterwards is decided on DEV_B once."""
import os
import sys
import time

import _l1g_candidate as CAND
import _l1g_core as G

name = sys.argv[1] if len(sys.argv) > 1 else "musique"
t0 = time.time()
D = G.Data(name, dense_fp32=True)
ch, gdiag = CAND.channels(D)
del D.E
npart = D.npart
res = {"cache": name, "partition": G.PARTITION_OF[name], "n_DEV_A": int(D.A.sum()), "BASE_A": D.r_base["ALL_split"]["A"], "arms": {}}
G.log("%s [%s] DEV_A n=%d BASE %.4f (channels in %.0fs)" % (name, G.PARTITION_OF[name], int(D.A.sum()), res["BASE_A"]["ALL"], time.time() - t0))
Cd, Cs, Cg, Cdir, Cmax = ch["Cd"], ch["Cs"], ch["Cg"], ch["Cdir"], ch["Cmax"]
arms = {
    "BASE F0(Cd,Cs)": G.F0([Cd, Cs], npart),
    "D2d F0(Cd,Cs,Cdir)": G.F0([Cd, Cs, Cdir], npart),
    "G2_IL IL(Cd,Cs,Cg)": G.F_interleave([Cd, Cs, Cg], npart),
    "X1 F0(Cd,Cs,Cg,Cdir)": G.F0([Cd, Cs, Cg, Cdir], npart),
    "X2 IL(Cd,Cs,Cdir)": G.F_interleave([Cd, Cs, Cdir], npart),
    "X3 IL(Cd,Cs,Cg,Cdir)": G.F_interleave([Cd, Cs, Cg, Cdir], npart),
    "X4 F0(F0(Cd,Cs), Cdir)  [BASE rank as one channel]": G.F0([G.F0([Cd, Cs], npart), Cdir], npart),
    "QMAX F0(Cd,Cs,Cmax)": G.F0([Cd, Cs, Cmax], npart),
    "X5 F0(Cd,Cs,Cmax,Cdir)": G.F0([Cd, Cs, Cmax, Cdir], npart),
}
allv = {}
for tag, rank in arms.items():
    out, v = D.eval_rank(rank, tag)
    res["arms"][tag] = out["A"]
    allv[tag] = v.astype(bool)
m = D.A
res["pairwise_DEV_A"] = {}
for a_, b_ in (("D2d F0(Cd,Cs,Cdir)", "QMAX F0(Cd,Cs,Cmax)"), ("X1 F0(Cd,Cs,Cg,Cdir)", "D2d F0(Cd,Cs,Cdir)"), ("X5 F0(Cd,Cs,Cmax,Cdir)", "D2d F0(Cd,Cs,Cdir)"),
              ("X3 IL(Cd,Cs,Cg,Cdir)", "D2d F0(Cd,Cs,Cdir)")):
    g, l, p = G.X.mcnemar(allv[b_][m], allv[a_][m])
    res["pairwise_DEV_A"]["%s vs %s" % (a_, b_)] = {"gained": g, "lost": l, "p": p}
    G.log("  paired %-28s vs %-28s +%d/-%d p=%.1e" % (a_[:28], b_[:28], g, l, p))
G.S.wj(os.path.join(G.OUT, "combo_A_%s.json" % name), res)
G.log("done %.0fs" % (time.time() - t0))
