"""Post-hoc VIEW of the ROUTE development records (read-only; REPORT section 34), over route_SUMMARY__<tag>.{json,npz} and the
nine records it merged (route / route2 / route3 x metaqa / musique / squad).  No new ranking or count: it re-reads the per-query
ALL-served intervals of a PANEL of named arms and reports
  (A) few vs many: every count's exact mean fan-out B_P(q) per cell at B_N 1000 and its MuSiQue / MetaQA ratios, next to the
      fan-out each cell needs (the smallest fixed B_P of a ranking within 0.01 of unrouted, and the envelope's B_P);
  (B) the panel at every B_N: ALL gold (exact counts), the delta vs unrouted, the mean fan-out and its fraction of the shards,
      paired gained / lost vs unrouted (McNemar p, descriptive), and the router information each arm needs;
  (C) the B_N-coupled router-side arm ES|ESTAR_BN (route3's diagnostic, recounted by the summary) on the same footing.
The panel was named AFTER the summary existed (FLAGGED: post-hoc; it collects the proposal's items as written, the rounds'
endpoints and the summary's top arms -- a selection, not a new result).  Every ALL here is asserted equal to the summary's
exact count.
Exact counts: shares with Decimal ROUND_HALF_UP to 3 decimals, means to 1, ratios to 2.
Usage: python scratchpad/_l1d_route_view.py <tag> [--dir=<summary dir> --out=<dir> --md=<path>]
       -> <out>/route_VIEW__<tag>.json (write-once; out defaults to dir, dir to results/L1_DEV)
DEVELOPMENT numbers (user rulings 2026-09-26/27): descriptive, no verdicts."""
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

import numpy as np

import _l1d_lib as D
import _l1d_adaptbp as AB
import _l1d_route as RT
import _l1d_route2 as RT2
import _l1d_route3 as RT3

TAG = sys.argv[1]
MD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--md=")), None)
DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), D.OUT)
OUTD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), DIR)
STEMS = ("route", "route2", "route3")
MODS = {"route": RT, "route2": RT2, "route3": RT3}
CELL_ORDER = [("metaqa", "metaqa"), ("metaqa", "metaqa_phg"), ("musique", "musique"), ("squad", "squad"), ("squad", "squad_phg")]
M_MAIN = 1000
M_PAIRED = (500, 1000, 2000)
PSTAR = "PSTAR_BN"
# the router information of an arm is its ranking's (the summary's ranking_information; the counts read the same evidence)
PANEL = [
    ("TOP3", "NEFF_TOP3", "section 33's OWN.TOP3.NEFF (the proposal's item 1)"),
    ("MAX", "NEFF_MAX", "section 33's OWN.MAX.NEFF (item 1)"),
    ("SUM", "NEFF_SUM", "section 33's OWN.SUM.NEFF; SUM == greedy marginal node coverage on disjoint shards (item 2)"),
    ("SDIV", "KNEE_SDIV", "seed-diversity coverage (item 3), knee of its own scores"),
    ("SC_L", "SC_L_COVER", "greedy seed cover run to saturation (items 2-3)"),
    ("FL_L", "FL_L_KNEEG", "facility location, greedy, knee of its marginal gains (item 4)"),
    ("FL_L", "FL_L_COVER", "facility location, greedy, run to saturation (item 4)"),
    ("PD", "NEFF_PD", "one-hop partition-graph diffusion (item 5; FLAGGED: a second hop at shard granularity)"),
    ("PC_L", "PC_L_KNEEG", "noisy-OR coverage (round 2's saturation mitigation)"),
    ("NSUM", "NEFF_NSUM", "additive facility location (round 2's saturation mitigation)"),
    ("ES", "KNEE_SDIV", "text endpoint A (the router's own order O')"),
    ("SDE", "BPI_SDIV", "KB endpoint B"),
    ("SDK", "BPI_SDIV", "KB endpoint B on the partition sketch alone"),
    ("MXS_L4", "MA_KS_L4", "round 3: lam4 mixture of A and B (native)"),
    ("UM_KS_L4", "UM_KS_L4", "round 3: lam4 union of A and B (native)"),
    ("UM_KS_L4", "MA_KS_L4", "first of the B_N 500 universality list (cross-record)"),
    ("IL_S_T3", "MG_KS_L2", "first of the B_N 1000 universality list (cross-record)"),
    ("IL_S_T3", "UM_KT_L4", "B_N 1000 lossless table, delta 0.05 (cross-record)"),
    ("S", "KNEE_TOP3", "B_N 1000 lossless table, delta 0.005"),
    ("S", PSTAR, "the B_N-coupled lossless reference (S at PSTAR == unrouted)"),
]
ESTAR_ARM = ("ES", "ESTAR_BN", "round 3: B_N-coupled router-side estimate of PSTAR (not in the factorial)")


def rh(fr, nd=3):
    d = Decimal(fr.numerator) / Decimal(fr.denominator)
    v = d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return str(abs(v) if v == 0 else v)


def sh(num, n, nd=3):
    return rh(Fraction(int(num), int(n)), nd)


def dl(a, b, n, nd=3):
    s = rh(Fraction(int(a) - int(b), int(n)), nd)
    return s if s.startswith("-") else "+" + s


fs = os.path.join(DIR, "route_SUMMARY__%s.json" % TAG)
S = json.load(open(fs, encoding="utf-8"))
fzs = os.path.join(DIR, S["npz"]["path"])
assert D.sha_file(fzs) == S["npz"]["sha256"], "summary npz changed"
ZS = np.load(fzs)
RANKS, COUNTS = [str(x) for x in ZS["ranks"]], [str(x) for x in ZS["counts"]]
assert [str(x) for x in ZS["cells"]] == [c for _, c in CELL_ORDER] and [int(m) for m in ZS["m_curve"]] == list(D.M_CURVE)
for s_ in STEMS:
    assert S["harness_sha256"][s_] == D.sha_file(os.path.abspath(MODS[s_].__file__)), "harness %s changed" % s_
mi0 = D.M_CURVE.index(M_MAIN)
view = {"tag": TAG, "status": "DEVELOPMENT (descriptive; post-hoc view, no new arm)", "summary": {"path": D.rel(fs), "sha256": D.sha_file(fs)},
        "rounding": "shares from exact query counts, Decimal ROUND_HALF_UP to 3 decimals (means to 1, ratios to 2)",
        "panel (FLAGGED: named after the summary existed)": [{"arm": "%s|%s" % (r_, k_), "router_information": S["ranking_information"][r_],
                                                              "role": w_} for r_, k_, w_ in PANEL + [ESTAR_ARM]],
        "few_vs_many": {}, "panel": {}}

# ---------------- (A) few vs many: every count's exact mean fan-out at B_N 1000, the MuSiQue / MetaQA ratios, the fan-out each cell needs
NQ = {c: S["cells"][c]["n_rows"] for _, c in CELL_ORDER}
NP = {c: S["cells"][c]["npart"] for _, c in CELL_ORDER}
fv = {}
for ki, cn in enumerate(COUNTS):
    e = {c: [int(ZS["BPSUM__" + c][ki, mi0]), NQ[c]] for _, c in CELL_ORDER}
    mean_ = {c: Fraction(*e[c]) for c in e}
    fv[cn] = {"B_P_sum_and_n (B_N 1000)": e,
              "musique_over_metaqa": [int((mean_["musique"] / mean_["metaqa"]).numerator), int((mean_["musique"] / mean_["metaqa"]).denominator)],
              "musique_over_metaqa_phg": [int((mean_["musique"] / mean_["metaqa_phg"]).numerator), int((mean_["musique"] / mean_["metaqa_phg"]).denominator)]}
need = {}
for _, c in CELL_ORDER:
    rr = S["cells"][c]["rankings"]
    env = S["cells"][c]["envelope (best fixed point of any ranking; gold-chosen hindsight)"][str(M_MAIN)]
    need[c] = {"envelope": env, "smallest_B_P_within_0.01_of_unrouted": {rk: rr[rk][str(M_MAIN)]["smallest_B_P_within_0.01_of_unrouted"]
                                                                        for rk in ("S", "ES", "TOP3", "SDIV", "SDE")},
               "best_fixed": {rk: rr[rk][str(M_MAIN)]["best_fixed"] for rk in ("S", "ES", "TOP3", "SDIV", "SDE")}}
view["few_vs_many"] = {"counts": fv, "need_at_B_N_1000": need}

# ---------------- (B) the panel: per-query served sets re-read from the records (the summary's merge, asserted)
REC, Z = {}, {}
for ds in ("metaqa", "musique", "squad"):
    REC[ds], Z[ds] = {}, {}
    for s_ in STEMS:
        inp = S["inputs"][ds][s_]
        fj, fz = os.path.join(D.REPO, inp["json"]["path"]), os.path.join(D.REPO, inp["npz"]["path"])
        assert D.sha_file(fj) == inp["json"]["sha256"] and D.sha_file(fz) == inp["npz"]["sha256"], "record changed: %s" % fj
        REC[ds][s_], Z[ds][s_] = json.load(open(fj, encoding="utf-8")), np.load(fz)


def interval(ds, c, rk):
    for s_ in STEMS:
        z = Z[ds][s_]
        zr = [str(x) for x in z["ranks"]]
        if rk in zr:
            i = zr.index(rk)
            return z["LO__" + c][:, i].astype(np.int64), z["HI__" + c][:, i, :].astype(np.int64)
    raise KeyError(rk)


def count_of(ds, c, cn, mi):
    if cn == PSTAR:
        return Z[ds]["route"]["PSTAR__" + c][:, mi].astype(np.int64)
    if cn == "ESTAR_BN":
        return Z[ds]["route3"]["ESTAR__" + c][:, mi].astype(np.int64)
    for s_ in STEMS:
        z = Z[ds][s_]
        zc = [str(x) for x in z["counts"]]
        if cn in zc:
            return z["CNT__" + c][:, zc.index(cn)].astype(np.int64)
    raise KeyError(cn)


for ds, c in CELL_ORDER:
    nq, npart = NQ[c], NP[c]
    LOs, HIs = interval(ds, c, "S")
    unr = {M: AB.served_at(LOs, HIs[:, mi], npart) for mi, M in enumerate(D.M_CURVE)}
    for mi, M in enumerate(D.M_CURVE):
        assert int(unr[M].sum()) == S["cells"][c]["unrouted_ALL"][str(M)]["n"]
    pc = {}
    for rk, cn, _ in PANEL + [ESTAR_ARM]:
        LO, HI = interval(ds, c, rk)
        e = {}
        for mi, M in enumerate(D.M_CURVE):
            b = count_of(ds, c, cn, mi)
            a_ = AB.served_at(LO, HI[:, mi], b)
            n_ = int(a_.sum())
            if cn != "ESTAR_BN":
                assert n_ == int(ZS["ALLN__" + c][RANKS.index(rk), COUNTS.index(cn), mi]), "ALL differs from the summary: %s|%s %s %d" % (rk, cn, c, M)
                assert int(b.sum()) == int(ZS["BPSUM__" + c][COUNTS.index(cn), mi])
            else:
                assert n_ == S["route3_diagnostics"]["cells"][c]["ES|ESTAR_BN (exact counts)"][str(M)]["n"]
            e[str(M)] = {"n": n_, "unrouted_n": int(unr[M].sum()), "B_P_sum": int(b.sum()), "nq": nq, "npart": npart}
            if M in M_PAIRED:
                e[str(M)]["paired_vs_unrouted"] = D.paired(unr[M], a_)
        pc["%s|%s" % (rk, cn)] = e
    view["panel"][c] = pc

fj = os.path.join(OUTD, "route_VIEW__%s.json" % TAG)
assert not os.path.exists(fj), "write-once"
view["code"] = {"path": "scratchpad/_l1d_route_view.py", "sha256": D.sha_file(os.path.abspath(__file__))}
D.G.S.wj(fj, view)
print("wrote %s sha256 %s" % (fj, D.sha_file(fj)[:12]))

# ---------------- markdown
C = [c for _, c in CELL_ORDER]
md = ["# ROUTE view (%s; summary %s)" % (TAG, view["summary"]["sha256"][:12]), "",
      "## (A) Few vs many: mean B_P(q) per cell at B_N 1000 and the MuSiQue / MetaQA ratio (counts ordered by the ratio over H4_SK MetaQA)", "",
      "Need at B_N 1000 (smallest fixed B_P within 0.01 of unrouted: S / ES / TOP3 / SDIV / SDE; envelope B_P):", ""]
for c in C:
    n_ = need[c]
    md.append("- %s (%d shards): %s; envelope %s at %s %d" % (c, NP[c], " / ".join(str(n_["smallest_B_P_within_0.01_of_unrouted"][rk])
                                                                         for rk in ("S", "ES", "TOP3", "SDIV", "SDE")),
                                                          sh(n_["envelope"]["n"], NQ[c]) if "n" in n_["envelope"] else n_["envelope"],
                                                          n_["envelope"].get("ranking", "?"), n_["envelope"].get("B_P", -1)))
md += ["", "| count | " + " | ".join(C) + " | MuSiQue / MetaQA | MuSiQue / MetaQA PHG |", "|---|" + "---|" * (len(C) + 2)]
order = sorted(COUNTS, key=lambda k: -Fraction(*fv[k]["musique_over_metaqa"]))
for cn in order:
    e = fv[cn]["B_P_sum_and_n (B_N 1000)"]
    md.append("| %s | %s | %s | %s |" % (cn, " | ".join(rh(Fraction(*e[c]), 1) for c in C), rh(Fraction(*fv[cn]["musique_over_metaqa"]), 2),
                                        rh(Fraction(*fv[cn]["musique_over_metaqa_phg"]), 2)))
md += ["", "## (B) Panel (FLAGGED: named after the summary existed): arm -- role; router information of its ranking", ""]
for rk, cn, w_ in PANEL + [ESTAR_ARM]:
    md.append("- %s\\|%s -- %s; %s" % (rk, cn, w_, S["ranking_information"][rk]))
md.append("")
for M in D.M_CURVE:
    md += ["### B_N = %d: ALL [delta vs unrouted] @ mean B_P [fraction of shards]%s" % (
        M, "; paired gained / lost vs unrouted (McNemar p)" if M in M_PAIRED else ""), "",
           "| arm | " + " | ".join(C) + " |", "|---|" + "---|" * len(C)]
    md.append("| unrouted | %s |" % " | ".join(sh(view["panel"][c]["S|" + PSTAR][str(M)]["unrouted_n"], NQ[c]) for c in C))
    for rk, cn, _ in PANEL + [ESTAR_ARM]:
        cells_ = []
        for c in C:
            e = view["panel"][c]["%s|%s" % (rk, cn)][str(M)]
            s_ = "%s [%s] @ %s [%s]" % (sh(e["n"], e["nq"]), dl(e["n"], e["unrouted_n"], e["nq"]), rh(Fraction(e["B_P_sum"], e["nq"]), 1),
                                        sh(e["B_P_sum"], e["nq"] * e["npart"]))
            if "paired_vs_unrouted" in e:
                p_ = e["paired_vs_unrouted"]
                s_ += "; +%d/-%d (%s)" % (p_["gained"], p_["lost"], ("%.2g" % p_["p"]) if p_["p"] is not None else "-")
            cells_.append(s_)
        md.append("| %s\\|%s | %s |" % (rk, cn, " | ".join(cells_)))
    md.append("")
if MD:
    assert not os.path.exists(MD)
    open(MD, "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("wrote", MD)
