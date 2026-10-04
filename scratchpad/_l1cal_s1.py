"""STEP 1 -- reproduce the exact scoring failure, then ask what each rule would do about it.

Under the frozen R0 selector every query produces an eviction set E = boundary \\ final and an
admission set A = final \\ boundary.  Gold labels those decisions POST HOC only:

    GOOD     an admitted challenger that carries gold
    BAD      an evicted incumbent that carries gold
    NEUTRAL  everything else

The specific failure the phase exists to fix is the cross-list comparison: an incumbent holding
CANONICAL EVIDENCE ONLY is compared against a challenger holding ONE non-canonical channel, and the
challenger wins purely because 1/(K0+r_challenger) > 1/(K0+r_incumbent) -- two reciprocal ranks read
off lists of completely different depth (200 vs 64 vs 32).

The last block is the one that matters: for the exact (query, partition) pairs R0 gets wrong, does
each calibrated rule retain the gold-bearing incumbent, and does it still admit the gold-bearing
challengers R0 gets right?

  python scratchpad/_l1cal_s1.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1cal_core as CC

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
K0, P, B = CC.K0, CC.P, 6


def prof(c, p):
    return ("C" if p in c["cpos"] else "-") + ("S" if p in c["spos"] else "-") \
           + ("R" if p in c["rpos"] else "-")


def run_ds(ds, OUT):
    S = CC.substrate(ds, B)
    ctxs, KS, goldp, nq = S["ctxs"], S["KS"], S["goldp"], S["nq"]
    ev_prof, ad_prof = {}, {}
    good = bad = nswap = 0
    decisive = decisive_bad = 0
    examples = []
    bad_pairs, good_pairs = [], []
    for qi, c in enumerate(ctxs):
        X = CC.sel_R0(c, KS[qi], B)
        fs = set(X)
        E = [p for p in c["bnd"] if p not in fs]
        A = [p for p in X if p not in c["bnd"]]
        nswap += len(A)
        g = goldp[qi]
        for p in E:
            k = prof(c, p)
            d = ev_prof.setdefault(k, {"n": 0, "gold": 0})
            d["n"] += 1; d["gold"] += int(p in g)
            if p in g:
                bad += 1
                bad_pairs.append((qi, int(p)))
        for p in A:
            k = prof(c, p)
            d = ad_prof.setdefault(k, {"n": 0, "gold": 0})
            d["n"] += 1; d["gold"] += int(p in g)
            if p in g:
                good += 1
                good_pairs.append((qi, int(p)))
        # ---- the decisive cross-list pattern
        conly = [p for p in E if prof(c, p) == "C--"]
        one_nc = [p for p in A if p not in c["cpos"]
                  and (prof(c, p) in ("-S-", "--R"))]
        if conly and one_nc:
            decisive += 1
            hit = [p for p in conly if p in g]
            if hit:
                decisive_bad += 1
                if len(examples) < 8:
                    i0, a0 = hit[0], one_nc[0]
                    ch = "S" if a0 in c["spos"] else "R"
                    r_ch = c["spos"][a0] if ch == "S" else c["rpos"][a0]
                    L_ch = len(c["spos"]) if ch == "S" else len(c["rpos"])
                    examples.append({
                        "qi": qi, "gold_incumbent": int(i0),
                        "incumbent_canonical_rank": int(c["cpos"][i0]),
                        "incumbent_raw": round(1.0 / (K0 + c["cpos"][i0]), 6),
                        "incumbent_unit": round(KS[qi]["unit"][0].get(i0, 0.0), 4),
                        "challenger": int(a0), "challenger_channel": ch,
                        "challenger_rank": int(r_ch), "challenger_list_len": int(L_ch),
                        "challenger_raw": round(1.0 / (K0 + r_ch), 6),
                        "challenger_unit": round(
                            (KS[qi]["unit"][1] if ch == "S" else KS[qi]["unit"][2]).get(a0, 0.0), 4)})
    # ---- counterfactual: what do the other rules do with R0's own decisions?
    cf = {}
    bad_set = {}
    for qi, p in bad_pairs:
        bad_set.setdefault(qi, set()).add(p)
    good_set = {}
    for qi, p in good_pairs:
        good_set.setdefault(qi, set()).add(p)
    for name in CC.RULES:
        keep = lose = 0
        for qi, ps in bad_set.items():
            fs = set(CC.SEL[name](ctxs[qi], KS[qi], B))
            keep += len(ps & fs); lose += len(ps - fs)
        akeep = alose = 0
        for qi, ps in good_set.items():
            fs = set(CC.SEL[name](ctxs[qi], KS[qi], B))
            akeep += len(ps & fs); alose += len(ps - fs)
        cf[name] = {"R0_BAD_evictions_now_RETAINED": keep, "still_evicted": lose,
                    "R0_GOOD_admissions_still_ADMITTED": akeep, "now_missed": alose,
                    "net_gold_slots_vs_R0": keep - alose}
    OUT[ds] = {"nq": nq, "swaps": nswap, "GOOD_admissions": good, "BAD_evictions": bad,
               "GOOD_over_BAD": round(good / max(1, bad), 3),
               "evicted_profiles": ev_prof, "admitted_profiles": ad_prof,
               "queries_with_decisive_cross_list_pattern": decisive,
               "of_those_evicting_a_gold_partition": decisive_bad,
               "examples": examples, "COUNTERFACTUAL": cf}
    log("%-15s swaps %5d  GOOD %4d  BAD %4d  (ratio %.2f)  decisive-pattern %4d (%d bad)"
        % (ds[:14], nswap, good, bad, good / max(1, bad), decisive, decisive_bad))
    for name in CC.RULES:
        v = cf[name]
        log("      %-26s retains %4d/%4d bad-evicted gold | keeps %4d/%4d good-admitted | net %+d"
            % (name, v["R0_BAD_evictions_now_RETAINED"], bad,
               v["R0_GOOD_admissions_still_ADMITTED"], good, v["net_gold_slots_vs_R0"]))


def main():
    OUT = {}
    fp = f"{CC.CALD}/diag/step1_failure.json"
    for ds in (sys.argv[1:] or CC.DSETS):
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote step1_failure.json")


if __name__ == "__main__":
    main()
