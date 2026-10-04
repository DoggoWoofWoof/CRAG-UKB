"""L1 CANDIDATE ADMISSION PHASE -- STEP 0: fix the B accounting before anything else runs.

Two different B questions were reported under one label.  This module separates them and adds the
full B ladder so the answer is not a two-point comparison.

    B_CURRENT_HEADROOM            = O2 - O1   B under the CURRENT candidate universe
    B_AFTER_FULL_VISITED_HEADROOM = O4 - O3   B once admission/reach is repaired

and, because O2/O4 relax B all the way to 50 (which necessarily un-protects the canonical core), the
intermediate ladder at a FIXED P=50:

    oracle_B(U, b) : protect base_rank[:50-b], choose the remaining b perfectly from U
                     b=6  reproduces O1 (U=SAFE) and O3 (U=FULL_VISITED)
                     b=50 reproduces O2 (U=SAFE) and O4 (U=FULL_VISITED)

Per-query monotonicity of the chain is verified, not assumed -- the loss decomposition is only
mutually exclusive if O0 <= O1 <= O3 <= O4 <= O5 holds for every single query.

  python scratchpad/_l1ca_step0.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1bc_core as BC
import _l1bc_ledger as LG

P, B = BC.P, BC.B
BLAD = [6, 8, 10, 14, 20, 30, 50]
DS = ["metaqa", "webqsp", "musique_clean", "2wiki_clean", "hotpotqa_clean", "squad_clean"]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def run(ds):
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)
    O = {k: np.zeros(nq, np.int8) for k in ("O0", "O1", "O2", "O3", "O4", "O5")}
    LAD = {(u, b): np.zeros(nq, np.int8) for u in ("SAFE", "FULL") for b in BLAD}
    for qi in range(nq):
        c = ctxs[qi]
        r = BC.rows(T, qi)
        pid = r["pid"]
        REQ = set(int(p) for p in goldp[qi])
        prot = set(c["prot"])
        X, cands, _ = LG.safe_pick(c)
        candset = set(int(p) for p in cands)
        reach = set(int(p) for p, f in zip(pid, r["in_reach"]) if f)
        sel = prot | set(int(p) for p in X)
        out = REQ - prot
        O["O0"][qi] = int(REQ <= sel)
        O["O1"][qi] = int(len(out) <= B and out <= candset)
        O["O2"][qi] = int(len(REQ) <= P and REQ <= (prot | candset))
        O["O3"][qi] = int(len(out) <= B and out <= reach)
        O["O4"][qi] = int(len(REQ) <= P and REQ <= (prot | reach))
        O["O5"][qi] = int(len(REQ) <= P)
        base = [p for p, _ in sorted(c["cpos"].items(), key=lambda kv: kv[1])]
        for u, U in (("SAFE", prot | candset), ("FULL", prot | reach)):
            for b in BLAD:
                keep = set(base[:P - b])
                need = REQ - keep
                LAD[(u, b)][qi] = int(len(need) <= b and need <= U)
    # per-query monotonicity of the mutually exclusive chain
    mono = {"O0<=O1": int((O["O0"] <= O["O1"]).all()), "O1<=O3": int((O["O1"] <= O["O3"]).all()),
            "O3<=O4": int((O["O3"] <= O["O4"]).all()), "O4<=O5": int((O["O4"] <= O["O5"]).all()),
            "O1<=O2": int((O["O1"] <= O["O2"]).all()), "O2<=O4": int((O["O2"] <= O["O4"]).all())}
    return dict(ds=ds, nq=nq, hops=hops, O=O, LAD=LAD, mono=mono)


def report(R):
    ds, nq = R["ds"], R["nq"]
    out = {"ds": ds, "nq": nq, "chain_monotone_per_query": R["mono"], "ladder": {}, "B": {}}
    for nm, m in LG.masks(ds, R["hops"], nq):
        o = {k: float(R["O"][k][m].mean()) for k in ("O0", "O1", "O2", "O3", "O4", "O5")}
        out["ladder"][nm] = {k: round(v, 4) for k, v in o.items()}
        out["B"][nm] = {
            "B_CURRENT_HEADROOM_O2_O1": round(o["O2"] - o["O1"], 4),
            "B_AFTER_FULL_VISITED_HEADROOM_O4_O3": round(o["O4"] - o["O3"], 4),
            "SAFE_universe_B_ladder": {b: round(float(R["LAD"][("SAFE", b)][m].mean()), 4)
                                       for b in BLAD},
            "FULL_VISITED_B_ladder": {b: round(float(R["LAD"][("FULL", b)][m].mean()), 4)
                                      for b in BLAD}}
    return out


if __name__ == "__main__":
    os.makedirs(f"{BC.BCD}/diag", exist_ok=True)
    allr = {}
    for ds in (sys.argv[1:] or DS):
        rp = report(run(ds))
        allr[ds] = rp
        json.dump(rp, open(f"{BC.BCD}/diag/step0_{ds}.json", "w"), indent=1)
    json.dump(allr, open(f"{BC.BCD}/diag/step0_ALL.json", "w"), indent=1)
    for ds, rp in allr.items():
        log(ds, "monotone", rp["chain_monotone_per_query"])
    print()
    print("ORACLE LADDER")
    print(f'{"":22}' + "".join(f"{k:>9}" for k in ("O0", "O1", "O2", "O3", "O4", "O5")))
    for ds, rp in allr.items():
        for nm in rp["ladder"]:
            if ds != "metaqa" and nm != "ALL":
                continue
            print(f"{ds+'/'+nm:22}" + "".join(f"{rp['ladder'][nm][k]:9.4f}" for k in
                                              ("O0", "O1", "O2", "O3", "O4", "O5")))
    print()
    print("B HEADROOM, the two different questions")
    print(f'{"":22}{"O2-O1 (now)":>14}{"O4-O3 (after)":>16}')
    for ds, rp in allr.items():
        for nm in rp["B"]:
            if ds != "metaqa" and nm != "ALL":
                continue
            b = rp["B"][nm]
            print(f"{ds+'/'+nm:22}{b['B_CURRENT_HEADROOM_O2_O1']:14.4f}"
                  f"{b['B_AFTER_FULL_VISITED_HEADROOM_O4_O3']:16.4f}")
    print()
    print("B LADDER at fixed P=50: protect base_rank[:50-b], choose b perfectly from the universe")
    for ds in allr:
        for nm in allr[ds]["B"]:
            if ds != "metaqa" and nm != "ALL":
                continue
            b = allr[ds]["B"][nm]
            print(f"  {ds+'/'+nm:20} b=" + "".join(f"{x:>9}" for x in BLAD))
            print(f"  {'SAFE universe':22}" + "".join(
                f"{b['SAFE_universe_B_ladder'][x]:9.4f}" for x in BLAD))
            print(f"  {'FULL_VISITED universe':22}" + "".join(
                f"{b['FULL_VISITED_B_ladder'][x]:9.4f}" for x in BLAD))
