"""L1 CANDIDATE ADMISSION PHASE -- STEP 7 (no-atom loss audit) + STEP 8 (query-local applicability).

STEP 7 asks the one question the previous phase left open: does hybrid admission PRESERVE the
required partitions that carry no core-exit atom -- the entire loss population of plain G4?

STEP 8 asks whether the gain regime is distinguishable from the loss regime using only
inference-safe per-query statistics, with NO threshold search and NO dataset identity.  If the
distributions separate, a universal applicability rule may exist; if they overlap, SRC assignment is
simply not universally safe and no gate can rescue it.

  python scratchpad/_l1ca_audit.py <ds> [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1bc_core as BC
import _l1bc_ledger as LG
import _l1bc_diag as DG
import _l1ca_admit as AD

P, B, MISS = BC.P, BC.B, BC.MISS
CAD = AD.CAD
STATS = ["src_atoms", "src_targets", "src_conc", "src_safe_overlap", "src_dense_agree",
         "src_splade_agree", "src_targets_per_source", "frac_miss_with_src"]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
# the comparison that isolates ADMISSION (same selector both sides) and the headline comparison
CMPS = [("A2_HYBRID_ADMIT", "G4", "SAFE_POOL", "G4"), ("A2_HYBRID_ADMIT", "G4", "SAFE_POOL", "F6"),
        ("A2_HYBRID_ADMIT", "F6", "SAFE_POOL", "F6"), ("A3_INTERLEAVE", "F6", "SAFE_POOL", "F6"),
        ("A4_SUBSUME", "F6", "SAFE_POOL", "F6"), ("A4_SUBSUME", "G4", "SAFE_POOL", "F6")]


def run(ds, log=log):
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)
    audit = {c: {"GAIN": [], "LOSS": []} for c in CMPS}
    ev = {c: {"gain": [], "loss": [], "same": []} for c in CMPS}
    stat = np.zeros((nq, len(STATS)))
    preserve = {"g4_lost_partitions": 0, "of_which_no_src": 0, "a2g4_preserved": 0,
                "a2g4_preserved_no_src": 0}
    for qi in range(nq):
        c = ctxs[qi]
        r = BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        prot = [int(p) for p in c["prot"]]
        pset = set(prot)
        cpos, spos, rpos = c["cpos"], c["spos"], c["rpos"]
        X, cands, sc = LG.safe_pick(c)
        cands = [int(p) for p in cands if int(p) in ixp]
        safe_ord = {int(p): i for i, (_, _, p) in enumerate(sc)}
        K = len(cands)
        REQ = set(int(p) for p in goldp[qi])
        a1 = AD.a1_pool(r, cpos, K, pset)
        a2, _ = AD.a2_pool(r, ixp, cands, cpos, K, prot, a1)
        a3 = AD.a3_pool(cands, K, a1)
        a4, _ = AD.a4_pool(r, ixp, cands, K, a1, safe_ord)
        pools = {"SAFE_POOL": cands, "A1_SRC_ASSIGN": a1, "A2_HYBRID_ADMIT": a2,
                 "A3_INTERLEAVE": a3, "A4_SUBSUME": a4}
        sa = AD.src_atoms(r)
        tgt = {}
        owner = {}
        for Pi, tg in sa:
            for j in tg:
                p = int(r["pid"][j])
                tgt.setdefault(p, []).append(Pi)
                owner.setdefault(p, Pi)
        srcset = set(tgt)

        # ---- STEP 8 statistics, all inference safe
        cnt = np.array([len(tg) for _, tg in sa], np.float64)
        tot = cnt.sum()
        dn = set(p for p in srcset if r["r_dense"][ixp[p]] < MISS)
        sp = set(p for p in srcset if r["r_splade"][ixp[p]] < MISS)
        stat[qi] = [len(sa), len(srcset),
                    float((cnt / tot) @ (cnt / tot)) if tot else 0.0,
                    len(srcset & set(cands)) / max(len(srcset), 1),
                    len(dn) / max(len(srcset), 1), len(sp) / max(len(srcset), 1),
                    float(cnt.mean()) if len(cnt) else 0.0, 0.0]

        pk, hitv = {}, {}
        for k, pl in pools.items():
            pk[(k, "F6")] = AD.f6_on_pool(pl, spos, rpos, cpos)[:B]
            pk[(k, "G4")] = AD.g4_on_pool(r, ixp, prot, pl, safe_ord)[0][:B]
            for s in ("F6", "G4"):
                hitv[(k, s)] = int(REQ <= (pset | set(pk[(k, s)])))
        miss_safe = REQ - (pset | set(pk[("SAFE_POOL", "F6")]))
        stat[qi, 7] = (sum(1 for p in miss_safe if p in srcset) / len(miss_safe)) if miss_safe else 0.0

        # ---- the preservation question: what plain G4 lost, does A2+G4 keep?
        lost_g4 = (REQ & set(pk[("SAFE_POOL", "F6")])) - set(pk[("SAFE_POOL", "G4")])
        for p in lost_g4:
            preserve["g4_lost_partitions"] += 1
            ns = p not in srcset
            preserve["of_which_no_src"] += int(ns)
            kept = p in set(pk[("A2_HYBRID_ADMIT", "G4")])
            preserve["a2g4_preserved"] += int(kept)
            preserve["a2g4_preserved_no_src"] += int(kept and ns)

        for cm in CMPS:
            ka, sa_, kb, sb = cm
            h1, h0 = hitv[(ka, sa_)], hitv[(kb, sb)]
            ev[cm]["gain" if h1 > h0 else ("loss" if h1 < h0 else "same")].append(qi)
            if h1 == h0:
                continue
            ent = [p for p in pk[(ka, sa_)] if p not in pk[(kb, sb)]]
            lft = [p for p in pk[(kb, sb)] if p not in pk[(ka, sa_)]]
            if h1 > h0:
                key = [p for p in ent if p in REQ] or ent
                p0 = key[0]
                audit[cm]["GAIN"].append({
                    "q": qi, "hop": int(hops[qi]), "entered_required": [int(x) for x in ent if x in REQ],
                    "newly_admitted": [int(x) for x in ent if x not in set(cands)],
                    "src_source_partition": int(owner.get(p0, -1)),
                    "n_src_sources_for_it": len(tgt.get(p0, [])),
                    "was_in_SAFE_pool": bool(p0 in set(cands)), "displaced": [int(x) for x in lft]})
            else:
                key = [p for p in lft if p in REQ] or lft
                p0 = key[0]
                audit[cm]["LOSS"].append({
                    "q": qi, "hop": int(hops[qi]), "lost_required": [int(x) for x in lft if x in REQ],
                    "has_SRC": bool(p0 in srcset),
                    "has_dense": bool(r["r_dense"][ixp[p0]] < MISS) if p0 in ixp else False,
                    "has_splade": bool(r["r_splade"][ixp[p0]] < MISS) if p0 in ixp else False,
                    "has_struct": bool(p0 in spos), "has_retcont": bool(p0 in rpos),
                    "still_in_pool": bool(p0 in set(pools[ka])),
                    "took_its_slot": [int(x) for x in ent]})
        if (qi + 1) % 500 == 0:
            log(f"   {ds} audit {qi+1}/{nq}")
    return dict(ds=ds, nq=nq, hops=hops, audit=audit, ev=ev, stat=stat, preserve=preserve)


def report(R):
    out = {"ds": R["ds"], "nq": R["nq"], "preserve": R["preserve"], "audit": {}, "applicability": {}}
    pv = R["preserve"]
    out["preserve"]["frac_no_src_of_g4_losses"] = round(
        pv["of_which_no_src"] / max(pv["g4_lost_partitions"], 1), 4)
    out["preserve"]["frac_preserved_by_A2_G4"] = round(
        pv["a2g4_preserved"] / max(pv["g4_lost_partitions"], 1), 4)
    out["preserve"]["frac_no_src_preserved_by_A2_G4"] = round(
        pv["a2g4_preserved_no_src"] / max(pv["of_which_no_src"], 1), 4)
    for cm in CMPS:
        nm = f"{cm[0]}+{cm[1]} vs {cm[2]}+{cm[3]}"
        A = R["audit"][cm]
        lo = A["LOSS"]
        out["audit"][nm] = {
            "n_gain": len(A["GAIN"]), "n_loss": len(lo),
            "gain_newly_admitted": sum(1 for x in A["GAIN"] if x["newly_admitted"]),
            "gain_was_already_in_SAFE_pool": sum(1 for x in A["GAIN"] if x["was_in_SAFE_pool"]),
            "loss_has_SRC": sum(1 for x in lo if x["has_SRC"]),
            "loss_no_SRC": sum(1 for x in lo if not x["has_SRC"]),
            "loss_still_in_pool": sum(1 for x in lo if x["still_in_pool"]),
            "loss_evidence": {k: sum(1 for x in lo if x["has_" + k])
                              for k in ("dense", "splade", "struct", "retcont")},
            "examples": (A["GAIN"][:3] + lo[:3])}
    st = R["stat"]
    for cm in CMPS:
        nm = f"{cm[0]}+{cm[1]} vs {cm[2]}+{cm[3]}"
        g, l, s = (np.array(R["ev"][cm][k], np.int64) for k in ("gain", "loss", "same"))
        d = {"n_gain": len(g), "n_loss": len(l), "n_same": len(s)}
        for i, k in enumerate(STATS):
            d[k] = {"gain": round(float(st[g, i].mean()), 4) if len(g) else None,
                    "loss": round(float(st[l, i].mean()), 4) if len(l) else None,
                    "same": round(float(st[s, i].mean()), 4) if len(s) else None,
                    "AUC_gain_vs_loss": round(DG._auc(st[g, i], st[l, i]), 4)
                    if len(g) and len(l) else None}
        out["applicability"][nm] = d
    return out


if __name__ == "__main__":
    os.makedirs(f"{CAD}/diag", exist_ok=True)
    for ds in (sys.argv[1:] or ["metaqa"]):
        R = run(ds)
        rp = report(R)
        json.dump(rp, open(f"{CAD}/diag/audit_{ds}.json", "w"), indent=1)
        np.savez_compressed(f"{CAD}/diag/stat_{ds}.npz", stat=R["stat"], hops=R["hops"],
                            **{f"ev_{c[0]}_{c[1]}_{c[2]}_{c[3]}_{k}": np.array(v, np.int64)
                               for c in CMPS for k, v in R["ev"][c].items()})
        log(ds, json.dumps({"preserve": rp["preserve"], "audit":
                            {k: {a: b for a, b in v.items() if a != "examples"}
                             for k, v in rp["audit"].items()}}, indent=1))
