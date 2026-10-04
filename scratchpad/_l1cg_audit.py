"""STEPS 1-3 -- proposal-recall audit and marginal complementarity.

Every number is about DISCOVERY only: can a family put the required partition into the candidate
pool at all?  No family's score is used anywhere and the selector is untouched.

Target set per query:
    miss    = gold partitions outside the protected core   (what must be swapped in)
    outside = miss \\ (bnd u chal)                          (what the current pool cannot see)
A query counts as pool-limited when it is P50-feasible and `outside` is non-empty.

The primary metric is ALL_REQUIRED_PROPOSED@M -- outside subset of F@M.  Recovering one partition
of a 3-partition chain does not solve a multi-hop query, so ANY is reported but never ranked on.

  python scratchpad/_l1cg_audit.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1cg_core as CG

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
M_MARGINAL = 64


def slices(ds, S):
    nq = S["nq"]
    out = {"ALL": np.ones(nq, bool)}
    if ds == "metaqa":
        for h in (1, 2, 3):
            out[f"hop{h}"] = S["hops"] == h
    return out


def run_ds(ds, OUT):
    S = CG.substrate(ds)
    O = CG.proposals(ds, S)
    T, nq = S["T"], S["nq"]
    npart = int(CG.part_graph(ds)["npart"])
    tgt = np.array([bool(t["feasible"] and t["outside"]) for t in T])
    row = {"npart": npart, "nq": nq, "n_pool_limited": int(tgt.sum()),
           "PROPOSAL_SET_SIZE": {}, "RECALL": {}, "POOL_ORACLE": {}}
    for f in CG.FAMS:
        row["PROPOSAL_SET_SIZE"][f] = {
            "mean_available": round(float(np.mean([len(O[f][qi]) for qi in range(nq)])), 1),
            "frac_of_universe_at_M256": round(float(np.mean(
                [min(256, len(O[f][qi])) for qi in range(nq)]) / max(1, npart - CG.P)), 3)}

    SL = slices(ds, S)
    for f in CG.FAMS:
        row["RECALL"][f] = {}
        row["POOL_ORACLE"][f] = {}
        for M in CG.MS:
            ex = [set(O[f][qi][:M]) for qi in range(nq)]
            po = CG.pool_oracle(S, T, ex)
            row["POOL_ORACLE"][f][str(M)] = {
                k: round(float(po[m].mean()), 4) for k, m in SL.items()}
            e = {}
            for k, m in SL.items():
                sel = np.where(m & tgt)[0]
                if not len(sel):
                    continue
                nall = sum(1 for qi in sel if T[qi]["outside"] <= ex[qi])
                nany = sum(1 for qi in sel if T[qi]["outside"] & ex[qi])
                tot = sum(len(T[qi]["outside"]) for qi in sel)
                got = sum(len(T[qi]["outside"] & ex[qi]) for qi in sel)
                e[k] = {"n_target": int(len(sel)),
                        "ALL_REQUIRED_PROPOSED": round(nall / len(sel), 4),
                        "ANY_REQUIRED_PROPOSED": round(nany / len(sel), 4),
                        "micro_partition_recall": round(got / max(1, tot), 4)}
            row["RECALL"][f][str(M)] = e
        k0 = "hop3" if ds == "metaqa" else "ALL"
        r = row["RECALL"][f]
        log("   %-20s ALL@M %s | pool-oracle(%s) %s"
            % (f, " ".join("%.3f" % r[str(M)][k0]["ALL_REQUIRED_PROPOSED"] for M in CG.MS),
               k0, " ".join("%.3f" % row["POOL_ORACLE"][f][str(M)][k0] for M in CG.MS)))

    # ---------------- STEP 3: marginal complementarity at a fixed budget
    EX = {f: [set(O[f][qi][:M_MARGINAL]) for qi in range(nq)] for f in CG.FAMS}
    rec = {f: {(qi, p) for qi in np.where(tgt)[0] for p in (T[qi]["outside"] & EX[f][qi])}
           for f in CG.FAMS}
    solved = {f: {int(qi) for qi in np.where(tgt)[0] if T[qi]["outside"] <= EX[f][qi]}
              for f in CG.FAMS}
    uniq = {f: len(rec[f] - set().union(*[rec[g] for g in CG.FAMS if g != f]))
            for f in CG.FAMS}
    uqs = {f: len(solved[f] - set().union(*[solved[g] for g in CG.FAMS if g != f]))
           for f in CG.FAMS}
    jac = {f: {g: round(len(rec[f] & rec[g]) / max(1, len(rec[f] | rec[g])), 3)
               for g in CG.FAMS} for f in CG.FAMS}
    # greedy marginal ordering, judged ONLY on discovery
    order, cur, chosen = [], set(), []
    for _ in range(len(CG.FAMS)):
        best, gain = None, -1
        for f in CG.FAMS:
            if f in chosen:
                continue
            g = len(solved[f] | cur) - len(cur)
            if g > gain:
                best, gain = f, g
        chosen.append(best); cur |= solved[best]
        order.append({"family": best, "marginal_queries_solved": gain,
                      "cumulative_queries_solved": len(cur),
                      "cumulative_frac_of_pool_limited": round(len(cur) / max(1, int(tgt.sum())), 4)})
    row["MARGINAL"] = {
        "M": M_MARGINAL, "n_pool_limited": int(tgt.sum()),
        "per_family": {f: {"missing_partitions_recovered": len(rec[f]),
                           "unique_missing_partitions": uniq[f],
                           "queries_fully_solved": len(solved[f]),
                           "unique_queries_fully_solved": uqs[f]} for f in CG.FAMS},
        "jaccard_overlap": jac, "greedy_order": order}
    log("   greedy: " + " -> ".join(f"{o['family'].split('_')[0]}+{o['marginal_queries_solved']}"
                                    for o in order if o["marginal_queries_solved"] > 0))
    OUT[ds] = row


def main():
    os.makedirs(f"{CG.CGD}/diag", exist_ok=True)
    fp = f"{CG.CGD}/diag/audit.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in (sys.argv[1:] or CG.DSETS):
        log(ds)
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote audit.json")


if __name__ == "__main__":
    main()
