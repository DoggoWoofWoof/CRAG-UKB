"""STEPS 1-7 of the exact-P50 structural contract audit.  Diagnostic only, nothing is promoted.

The oracle at budget B is exact and needs no search.  With FINAL P = 50 held fixed,

    prot_B = base_rank[:50-B]      bnd_B = base_rank[50-B:50]      final = prot_B | X, |X| = B

so a query is coverable at budget B under evidence universe U iff

    need = goldp - prot_B     satisfies    need subset (bnd_B | chal_U)    and    |need| <= B

because the oracle may spend its B slots on exactly the needed partitions.  Both failure modes are
therefore separated by construction: `need` not inside the candidate universe is an EVIDENCE/REACH
failure, `|need| > B` is a CAPACITY failure, and they are reported apart.

  python scratchpad/_l1ca_run.py ds
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1ps_router as RT
import _l1sr_eval as EV
import _l1sr_diag as DG
import _l1ca_core as CA

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def blocks_of(hops, nq):
    b = [("ALL", np.ones(nq, bool))]
    hs = sorted(set(int(x) for x in hops)) if hops is not None else []
    if hs and min(hs) >= 0:
        b += [(f"hop{h}", np.asarray(hops)[:nq] == h) for h in hs]
    return b


def run(ds="metaqa", nq_max=None):
    S, U, work, parity = CA.universes(ds, log, nq_max)
    nq = len(U["n_adm"])
    hard = S["hard"]; goldp = S["goldp"]; ctxs = S["ctxs"]
    hops = S["hops"]; base_rank = np.asarray(S["z"]["base_rank"])
    gr, _ = DG.gold_rows_of(ds, S["z"])
    Gs = [set(int(x) for x in g) for g in gr]
    BL = blocks_of(hops, nq)
    OUT = {"ds": ds, "nq": nq, "STEP7_WORK": work, "PARITY": work["PARITY"],
           "STEP1": {}, "STEP2": {}, "STEP3": {}, "STEP45": {}, "STEP6": {}}
    log(f"frozen replay parity {work['PARITY']}   "
        f"admitted/q {work['admitted']}  visited/q {work['visited']}")

    # ---------------- STEP 1: the three nested evidence universes -------------------------------
    SF = {u: [] for u in CA.UNIVERSES}
    naggs = {u: 0 for u in CA.UNIVERSES}
    for qi in range(nq):
        nd, hp, sd, ct = U["node"][qi], U["hop"][qi], U["sdir"][qi], U["cnt"][qi]
        # M64 must stop at the admitted count: the frozen array is -1 padded past it, so a query
        # with fewer than 64 admitted nodes must NOT pull pruned nodes into the M64 prefix.
        for u, dep in (("S4_M64", min(EV.M_STRUCT, int(U["n_adm"][qi]))),
                       ("S4_FULL_ADMITTED", int(U["n_adm"][qi])),
                       ("S4_FULL_VISITED", int(U["n_vis"][qi]))):
            r, na = CA.sf_prefix(nd, hp, sd, ct, hard, dep)
            SF[u].append(r); naggs[u] += na
    # gate: the M64 prefix must reproduce the frozen structural ranking exactly
    frozen = [RT.order_struct(c["sagg"], "S4") for c in ctxs]   # full length: EV.evaluate needs it
    ok = sum(int(SF["S4_M64"][qi] == frozen[qi]) for qi in range(nq))
    OUT["STEP1"]["M64_MATCHES_FROZEN"] = f"{ok}/{nq}"
    for u in CA.UNIVERSES:
        OUT["STEP1"][u] = {"partitions_per_query": round(naggs[u] / nq, 1),
                           "nodes_per_query": round(float(
                               EV.M_STRUCT if u == "S4_M64" else
                               (U["n_adm"].mean() if u == "S4_FULL_ADMITTED"
                                else U["n_vis"].mean())), 1)}
    log(f"STEP1 M64-vs-frozen {ok}/{nq}   partitions/q " +
        "  ".join(f"{u}={OUT['STEP1'][u]['partitions_per_query']}" for u in CA.UNIVERSES))

    # ---------------- STEP 2: CORE vs NOVEL, classified without gold ----------------------------
    for u in CA.UNIVERSES:
        acc = {nm: {"gold_nodes_CORE": 0, "gold_nodes_NOVEL": 0, "gold_parts_CORE": 0,
                    "gold_parts_NOVEL": 0, "gold_parts_CORE_hit": 0, "gold_parts_NOVEL_hit": 0,
                    "parts_CORE": 0, "parts_NOVEL": 0} for nm, _ in BL}
        for qi in range(nq):
            b50 = set(int(x) for x in base_rank[qi][:EV.P])
            sfs = set(SF[u][qi])
            dep = (min(EV.M_STRUCT, int(U["n_adm"][qi])) if u == "S4_M64" else
                   (int(U["n_adm"][qi]) if u == "S4_FULL_ADMITTED" else int(U["n_vis"][qi])))
            nodes = U["node"][qi][:dep]
            gp = set(int(hard[v]) for v in Gs[qi] if int(hard[v]) >= 0)
            gn_core = sum(1 for v in nodes if int(v) in Gs[qi] and int(hard[v]) in b50)
            gn_nov = sum(1 for v in nodes if int(v) in Gs[qi] and int(hard[v]) not in b50)
            for nm, mk in BL:
                if not mk[qi]:
                    continue
                a = acc[nm]
                a["gold_nodes_CORE"] += gn_core
                a["gold_nodes_NOVEL"] += gn_nov
                a["parts_CORE"] += len(sfs & b50)
                a["parts_NOVEL"] += len(sfs - b50)
                a["gold_parts_CORE"] += len(gp & b50)
                a["gold_parts_NOVEL"] += len(gp - b50)
                a["gold_parts_CORE_hit"] += len(gp & b50 & sfs)
                a["gold_parts_NOVEL_hit"] += len((gp - b50) & sfs)
        for nm in acc:
            a = acc[nm]
            a["NOVEL_GOLD_PARTITION_RECALL"] = round(
                a["gold_parts_NOVEL_hit"] / max(1, a["gold_parts_NOVEL"]), 4)
            a["CORE_GOLD_PARTITION_RECALL"] = round(
                a["gold_parts_CORE_hit"] / max(1, a["gold_parts_CORE"]), 4)
            a["frac_evidence_spent_on_CORE"] = round(
                a["parts_CORE"] / max(1, a["parts_CORE"] + a["parts_NOVEL"]), 4)
        OUT["STEP2"][u] = acc
        log(f"STEP2 {u:18s} NOVEL_GOLD_PARTITION_RECALL {acc['ALL']['NOVEL_GOLD_PARTITION_RECALL']:.4f}"
            f"   CORE {acc['ALL']['CORE_GOLD_PARTITION_RECALL']:.4f}"
            f"   evidence on CORE {acc['ALL']['frac_evidence_spent_on_CORE']:.4f}")

    # ---------------- STEP 3: what the needed partitions actually have --------------------------
    for nm, mk in BL:
        row = {"needed": 0, "in_visited_universe": 0, "in_admitted": 0, "in_S4_M64": 0,
               "in_S4_FULL_ADMITTED": 0, "in_S4_FULL_VISITED": 0, "in_canonical_or_retrieval": 0,
               "in_NOTHING": 0,
               # the FROZEN contract candidate set, exactly as the previous phase counted it:
               # an incumbent boundary partition, or a challenger from SF(M64) or RF
               "in_frozen_candidate_set": 0, "in_RF": 0, "in_boundary": 0,
               "in_canonical_top50": 0, "in_NOTHING_frozen_contract": 0}
        for qi in range(nq):
            if not mk[qi]:
                continue
            c = ctxs[qi]
            need = set(goldp[qi]) - c["prot_set"]
            vis_p = set(int(hard[v]) for v in U["node"][qi])
            adm_p = set(int(hard[v]) for v in U["node"][qi][:int(U["n_adm"][qi])])
            s64, sfa, sfv = (set(SF[u][qi]) for u in CA.UNIVERSES)
            cr = set(c["cpos"]) | set(c["_RF"])
            rf = set(c["_RF"]); bnd = set(c["bnd"])
            b50q = set(int(x) for x in base_rank[qi][:EV.P])
            froz = bnd | (s64 - b50q) | (rf - b50q)
            for p in need:
                row["needed"] += 1
                row["in_visited_universe"] += int(p in vis_p)
                row["in_admitted"] += int(p in adm_p)
                row["in_S4_M64"] += int(p in s64)
                row["in_S4_FULL_ADMITTED"] += int(p in sfa)
                row["in_S4_FULL_VISITED"] += int(p in sfv)
                row["in_canonical_or_retrieval"] += int(p in cr)
                row["in_NOTHING"] += int(p not in vis_p and p not in cr)
                row["in_frozen_candidate_set"] += int(p in froz)
                row["in_RF"] += int(p in rf)
                row["in_boundary"] += int(p in bnd)
                row["in_canonical_top50"] += int(p in b50q)
                row["in_NOTHING_frozen_contract"] += int(p not in froz)
        n = max(1, row["needed"])
        row.update({f"frac_{k}": round(row[k] / n, 4) for k in
                    ("in_visited_universe", "in_admitted", "in_S4_M64", "in_S4_FULL_ADMITTED",
                     "in_S4_FULL_VISITED", "in_canonical_or_retrieval", "in_NOTHING",
                     "in_frozen_candidate_set", "in_RF", "in_boundary", "in_canonical_top50",
                     "in_NOTHING_frozen_contract")})
        OUT["STEP3"][nm] = row
    for nm in OUT["STEP3"]:
        r = OUT["STEP3"][nm]
        log(f"STEP3 {nm:6s} needed {r['needed']:6d}  visited {r['frac_in_visited_universe']:.4f}"
            f"  admitted {r['frac_in_admitted']:.4f}  M64 {r['frac_in_S4_M64']:.4f}"
            f"  FULLVIS {r['frac_in_S4_FULL_VISITED']:.4f}"
            f"  canon/ret {r['frac_in_canonical_or_retrieval']:.4f}"
            f"  NOTHING {r['frac_in_NOTHING']:.4f}")

    # ---------------- STEP 4/5: the oracle ceiling matrix ---------------------------------------
    i_safe, _ = EV.evaluate(S, frozen, "A_F6", (EV.P,))
    i_safe = i_safe[EV.P][:nq]
    OUT["SAFE"] = {nm: round(float(i_safe[mk].mean()), 4) for nm, mk in BL}
    needsz = np.zeros((len(CA.BS), nq), np.int64)
    for u in CA.UNIVERSES:
        OUT["STEP45"][u] = {}
        for bi, Bv in enumerate(CA.BS):
            cov = np.zeros(nq, np.int8)
            fail_cap = np.zeros(nq, np.int8); fail_ev = np.zeros(nq, np.int8)
            for qi in range(nq):
                c = ctxs[qi]
                b50 = [int(x) for x in base_rank[qi][:EV.P]]
                prot = set(b50[:EV.P - Bv]); bnd = set(b50[EV.P - Bv:])
                chal = set(p for p in SF[u][qi] if p not in set(b50))
                chal |= set(p for p in c["_RF"] if p not in set(b50))
                need = set(goldp[qi]) - prot
                if u == CA.UNIVERSES[-1]:
                    needsz[bi, qi] = len(need)
                enough = len(need) <= Bv
                reach = need <= (bnd | chal)
                cov[qi] = int(enough and reach)
                fail_cap[qi] = int(not enough)
                fail_ev[qi] = int(enough and not reach)
            OUT["STEP45"][u][f"B{Bv}"] = {
                **{nm: round(float(cov[mk].mean()), 4) for nm, mk in BL},
                "fail_capacity": {nm: int(fail_cap[mk].sum()) for nm, mk in BL},
                "fail_evidence": {nm: int(fail_ev[mk].sum()) for nm, mk in BL}}
        log(f"STEP45 {u:18s} " + "  ".join(
            f"B{b}={OUT['STEP45'][u][f'B{b}']['ALL']:.4f}" for b in CA.BS))
    # ---- REALISABILITY CONTROL: the UNCHANGED frozen F6 selector fed each evidence universe -----
    # Not a new selector and not a promotion.  The previous phase showed a large oracle gain at the
    # read stage converting to ~0 under the real selector, so an oracle ceiling means nothing until
    # this is measured.  B stays 6, S4 stays S4, F6 stays F6; only SF changes.
    OUT["REALISABLE_F6"] = {}
    for u in CA.UNIVERSES:
        ind, churn = EV.evaluate(S, SF[u] + frozen[nq:], "A_F6", (EV.P,))
        ind = ind[EV.P][:nq]
        m = PP.mcnemar(ind, i_safe)
        row = {**{nm: round(float(ind[mk].mean()), 4) for nm, mk in BL},
               "churn": round(float(churn[:nq].mean()), 3),
               **{f"vs_SAFE_{k}": v for k, v in m.items()}}
        for nm, mk in BL:
            if nm == "ALL":
                continue
            row[f"vs_SAFE_{nm}"] = PP.mcnemar(ind[mk], i_safe[mk])
        OUT["REALISABLE_F6"][u] = row
        log(f"F6-REAL {u:18s} " + "  ".join(f"{nm} {row[nm]:.4f}" for nm, _ in BL) +
            f"   net {m['net']:+d} p={m['mcnemar_p']:.3g}{' SIG' if m['sig'] else ''}"
            f"   churn {row['churn']}")

    OUT["STEP4_NEED_DISTRIBUTION"] = {}
    for nm, mk in BL:
        gsz = np.array([len(set(goldp[qi]) - set(int(x) for x in base_rank[qi][:EV.P - 6]))
                        for qi in range(nq)])
        OUT["STEP4_NEED_DISTRIBUTION"][nm] = {
            "mean_need_at_B6": round(float(gsz[mk].mean()), 2),
            **{f"queries_needing_gt_{b}": int((gsz[mk] > b).sum()) for b in (6, 8, 12, 20)},
            "queries": int(mk.sum())}

    # ---------------- STEP 6: the full-universe bookend and the decomposition --------------------
    full = np.array([int(len(goldp[qi]) <= EV.P) for qi in range(nq)])
    OUT["STEP6"]["FULL_UNIVERSE_P50_ORACLE"] = {nm: round(float(full[mk].mean()), 4)
                                                for nm, mk in BL}
    for nm, mk in BL:
        fu = float(full[mk].mean())
        fv50 = OUT["STEP45"]["S4_FULL_VISITED"]["B50"][nm]
        fv6 = OUT["STEP45"]["S4_FULL_VISITED"]["B6"][nm]
        sa = OUT["SAFE"][nm]
        OUT["STEP6"][nm] = {
            "FULL_UNIVERSE_P50": round(fu, 4),
            "FULL_VISITED_B50": fv50, "FULL_VISITED_B6": fv6, "SAFE": sa,
            "reach_gap": round(fu - fv50, 4),
            "B_capacity_gap": round(fv50 - fv6, 4),
            "ranking_selection_gap": round(fv6 - sa, 4),
            "total_gap": round(fu - sa, 4)}
    fp = f"{CA.CAD}/diag/audit_{ds}.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")
    return OUT


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "metaqa",
        int(sys.argv[2]) if len(sys.argv) > 2 else None)
