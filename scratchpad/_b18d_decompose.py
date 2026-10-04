"""B1.8d — DECOMPOSING THE HEADROOM: domain-level scalar vs query-level discrimination.

B1.8c found a dissociation that reframes the whole B1 problem:
    in-domain end-to-end policy captures 73.2% (MetaQA) / 38.7% (2Wiki) of oracle,
    while its per-query R^2(D4) is ~0.003 -- i.e. it predicts essentially NOTHING per query.
Hypothesis H1: the in-domain gain is carried almost entirely by a PER-DOMAIN CONSTANT reserve R,
    not by query-conditional discrimination.

Test: compare, on the same eval half,
    (a) BEST CONSTANT R      -- one scalar per domain, chosen on the train half, ZERO query information
    (b) learned in-domain policy (B1.8c)  -- query-conditional, trained in-domain (cheats on domain)
    (c) per-query ORACLE R   -- upper bound of any per-query policy
Decomposition:  DOMAIN-LEVEL headroom = (a) - 0 ;  QUERY-LEVEL headroom = (c) - (a).

If (a) ~ (b) and (c) - (a) is small relative to (a), the reserve decision is a DOMAIN-LEVEL SCALAR and
per-query modelling (more features, attention, more source datasets) cannot be the lever.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _b14_twochannel import load
from _b17_marginal import utilities, R_VALS

DSES = ["metaqa", "2wiki_clean", "squad_clean"]
OUT = "results/GENERALIZATION/_g2_b18d_decompose.json"


def main():
    res = {"HYPOTHESIS": "H1: the in-domain reserve gain is a per-domain CONSTANT R, not query discrimination."}
    tab = {}
    for ds in DSES:
        d = load(ds)
        UR, D = utilities(d)
        nq = len(UR); tr = np.arange(nq) % 2 == 0; te = ~tr
        # (a) best constant R chosen on the TRAIN half only
        const_net_tr = [float(UR[tr][:, i].sum()) for i in range(4)]
        bi = int(np.argmax(const_net_tr)); bR = R_VALS[bi]
        const_net_te = float(UR[te][:, bi].sum())
        # (c) per-query oracle on eval half
        orc_net = float(UR[te].max(1).sum())
        orcR = np.array([R_VALS[int(np.argmax(UR[i] >= UR[i].max() - 1e-12))] for i in np.where(te)[0]])
        # per-R eval curve
        curve = {str(R_VALS[i]): round(float(UR[te][:, i].sum()), 1) for i in range(4)}
        tab[ds] = {
            "eval_n": int(te.sum()),
            "NET_by_constant_R_eval": curve,
            "best_constant_R_from_train": bR,
            "NET_best_constant_R": round(const_net_te, 1),
            "NET_learned_in_domain_policy": None,          # filled from b18c
            "NET_per_query_oracle": round(orc_net, 1),
            "mean_oracle_R": round(float(orcR.mean()), 3),
            "frac_oracle_R0": round(float((orcR == 0).mean()), 3),
            "DOMAIN_LEVEL_headroom": round(const_net_te, 1),
            "QUERY_LEVEL_headroom": round(orc_net - const_net_te, 1),
            "query_share_of_total_headroom": round(float((orc_net - const_net_te) / max(orc_net, 1e-9)), 4)}
        print(f"[b18d] {ds:13s} curve {curve} | bestconstR {bR} NET {const_net_te:.0f} | "
              f"oracle {orc_net:.0f} | query-share {tab[ds]['query_share_of_total_headroom']:.1%}", flush=True)

    # pull the learned in-domain numbers from b18c for the head-to-head
    p = "results/GENERALIZATION/_g2_b18c_ceiling.json"
    if os.path.exists(p):
        c = json.load(open(p))["IN_DOMAIN_CEILING"]
        for ds in DSES:
            best = max(["COMP", "P0", "P0+COMP"], key=lambda k: c[ds][k]["capture_frac_of_oracle"])
            tab[ds]["NET_learned_in_domain_policy"] = c[ds][best]["in_domain_NET"]
            tab[ds]["learned_in_domain_variant"] = best
            tab[ds]["learned_minus_best_constant"] = round(
                c[ds][best]["in_domain_NET"] - tab[ds]["NET_best_constant_R"], 1)

    res["DECOMPOSITION"] = tab
    res["VERDICT"] = {
        "H1_supported": {ds: (tab[ds]["NET_learned_in_domain_policy"] is not None and
                              tab[ds]["learned_minus_best_constant"] <= 0.10 * max(tab[ds]["NET_best_constant_R"], 1))
                         for ds in DSES},
        "query_share_of_total_headroom": {ds: tab[ds]["query_share_of_total_headroom"] for ds in DSES}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    print("\n=== HEADROOM DECOMPOSITION (eval half) ===")
    print(f"  {'dataset':14s} {'bestConstR':>10s} {'NET(const)':>10s} {'NET(learned)':>12s} {'NET(oracle)':>11s} "
          f"{'learned-const':>13s} {'query share':>11s}")
    for ds in DSES:
        t = tab[ds]
        print(f"  {ds:14s} {t['best_constant_R_from_train']:10d} {t['NET_best_constant_R']:10.0f} "
              f"{str(t['NET_learned_in_domain_policy']):>12s} {t['NET_per_query_oracle']:11.0f} "
              f"{str(t['learned_minus_best_constant']):>13s} {t['query_share_of_total_headroom']:10.1%}")
    print("\n[b18d] wrote", OUT)


if __name__ == "__main__":
    main()
