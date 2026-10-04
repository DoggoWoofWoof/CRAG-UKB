"""L1 DEVELOPMENT -- CARRY-FORWARD record of the L1 routing rule (ruling 2026-09-29; REPORT section 37).

Hash-freezes the spec the advisor ruled and the user accepted ("Record the ruling only": write up the ruling and hash-freeze the
spec, but read no held-out rows yet):
    L1a  IR_L1 node localisation (sections 32-34)
    L1b  ES shard ranking on the deployed K map; KNEE_SDIV coarse fan-out B100(q) on the same partitioner's K = 100 map;
         B_P(q, K) = min(K, ceil(B100(q) * (K / 100)^0.75))
and copies, from the pinned section 36 records, the evidence the ruling rests on: the alpha 0.75 systems rows (recall at B_N,
fan-out B_P and rho, local scan CMASS / N, load skew), the tolerance verdicts over K >= 250, structural vs balanced random, the
K = 100 anchor and the largest-K contrast with ES.KNEE_SDIV and alpha 1.0.
Reads only development records (results/L1_DEV/kscale_*__v1.json, their npz by sha, the section 36 summary, the loc records' code
pins and the population file by sha).  Reads no held-out row, no split B row, no TEST row; writes nothing under data/.
Every copied number is the summary's own string (Decimal ROUND_HALF_UP from exact counts) or an exact count difference.

    python -u scratchpad/_l1d_carry.py      -> results/L1_DEV/L1_ROUTING_CARRY_FORWARD__v1.json (write-once)
"""
import hashlib
import io
import json
import os
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEV = os.path.join(REPO, "results", "L1_DEV")
OUTP = os.path.join(DEV, "L1_ROUTING_CARRY_FORWARD__v1.json")
DATASETS = ("metaqa", "musique", "squad")
FAMILIES = {"metaqa": ("PHG", "MTK"), "musique": ("PHG",), "squad": ("PHG", "MTK")}
BASE, ARM, A10 = "ES|KNEE_SDIV", "ES|KSCALE_a0.75", "ES|KSCALE_a1.0"
ALPHA = "0.75"
BNS = ("100", "250", "500", "1000", "2000", "5000")
M = "−"

# the section 36 pins every downstream claim rests on (asserted against the records and the files on disk)
KSCALE_SUMMARY_SHA = "db15a9e0"
MODULES = {
    "scratchpad/_l1d_kscale.py": "b47cfaf4ae8198361fb261a4c1ae39dd3580c2a77fa8674bfaf4f149811448a2",
    "scratchpad/_l1d_kscale_summary.py": "125ce9dc1639f8b832d7da1f0affb02b306ad55fede7f46fe027b4202024d9d1",
    "scratchpad/_l1d_scale_summary.py": "08696c8b4e192191c3bad1a6c420d1da6f2e4123bf102cf74b6f674efcc8e188",
    "scratchpad/_l1d_scale.py": "ba0a18377da04a918602ac9f9b5ea1df103d8138cb47abe4ed93a77315b53989",
    "scratchpad/_l1d_arms.py": "f5f52b79a2137a6e1f764d6438cdd77600783bb075589309bc2216e0ade46fad",
    "scratchpad/_l1d_lib.py": "dafe39b2a0d77c23ae7eed2c8d8ccf93c88dfc02ebbceaefe143ce0a9a6c275a",
    "scratchpad/_l1d_node1h.py": "8265d6bc43c36b652aa9a01f1344505e04f75775e416829daa9df4b03caae438",
    "scratchpad/_l1d_edgediag.py": "874bfc495a2d8e9b6d5aed3a5b0e6aa5bb5fd3080edbf5840351b9ebd6b2eec3",
    "scratchpad/_l1d_adaptbp.py": "984e3aeffa3aed8ad446637c1bbe4afd4df789947a2e12ada9cc7cd73acf88da",
    "scratchpad/_l1d_route.py": "7b420e5401f31f1f0f0459bf0b7c10b90becfdfe31f1b50489ac4b00ba4ad9c2",
    "scratchpad/_l1d_route2.py": "890566c71c64b3118ec399ecc5375471736376a92b5134e940043c048c8d2ed9",
    "scratchpad/_l1d_route3.py": "283a711b6fc007a720c6d800ea6cc0fa1be177114ccfaf14da27cf169b8e80b6",
    "scratchpad/_l1d_scale_parts.py": "906a41c2a8b7f023d56e28ba0ab09b03e66c5051b31bac2ee328085a2e79d5ec",
    "scratchpad/_l1d_loc.py": "d4570b9e47efba89060c29c0c1a9c1ddc18dec54677581fe7f111677174f29b4",
}
POPULATION = ("results/L1_DEV/loc_population.json", "c7f7080639d66f1e8f5c2fb90aff7da49dfda5f63a5fe558fe8c266bd5fef47e")
TOL_BOUND = {"0.01": {1998: 19, 2000: 20}, "0.005": {1998: 9, 2000: 10}}   # floor(tol x rows): rows below unrouted allowed

ADVISOR = [   # verbatim from the advisor's ruling as the user forwarded it on 2026-09-29
    "Given §36, I’d choose (b) with α=0.75 as the current L1 routing rule.",
    "At B_N=1000, α=0.75 is the smallest scaling exponent that stays within 0.01 of unrouted recall across the structural maps "
    "for K≥250, while still preserving the desirable property that ρ=B_P/K falls as the graph is partitioned more finely.",
    "B_P(q,K) = ⌈B_100(q) (K/100)^0.75⌉ where B_100(q) is the `KNEE_SDIV` count computed against the K=100 map.",
    "I would not try to fix the B_N=5000 problem in L1",
    "The issue is: anchor-count underestimation at large B_N not K scaling.",
    "I'd like to see the corresponding α=.75 load number prominently in the final report too, because this is a "
    "distributed-systems property, not merely retrieval quality.",
    "We ultimately care about: Recall fan-out local scan load skew not recall alone.",
    "Carry forward: IR_L1, ES shard ranking, KNEE_SDIV coarse fan-out, B_P(q,K)= ⌈B_100(q)(K/100)^0.75⌉ and treat "
    "B_N=5000 parity as a known limitation of the coarse-anchor count, not something to tune away now.",
    "Then stop L1 development.",
]
USER = [      # verbatim from the user's answer on 2026-09-29 (it overrides the advisor's "next = held-out")
    "Yes — do not spend the held-out evaluation yet.",
    "Given where the system stands, I would choose “Record the ruling only” for now",
    "I would not use the reserved MetaQA/SQuAD/MuSiQue rows now to decide how to fix MetaQA.",
    "We already have plenty of legacy-exposed development data. Use that aggressively.",
    "Do not read the held-out rows yet.",
    "I think improving the KB side should be the highest priority now, rather than spending more time squeezing another 0.2% "
    "out of MuSiQue routing.",
]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rj(p):
    return json.load(io.open(p, encoding="utf-8"))


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def sgn(n):
    return "0" if n == 0 else ("+%d" % n if n > 0 else M + "%d" % (-n))


def kcells(D, fam):
    return sorted((c for c in D["cells"] if c.startswith(fam + "_k")), key=lambda c: int(c.split("_k")[1]))


def systems_row(D, c, arm):
    a = D["cells"][c]["arms"][arm]
    return {"K": D["cells"][c]["K"], "native": D["cells"][c]["native"],
            "B_N": {b: {"n": a["B_N"][b]["n"], "unrouted": D["unrouted_n"][b], "delta_rows": sgn(a["B_N"][b]["n"] - D["unrouted_n"][b]),
                        "ALL": a["B_N"][b]["ALL"]} for b in BNS},
            "B_P_mean": a["B_P_mean"], "B_P_p90 (nearest rank)": a["B_P_p90 (nearest rank)"], "B_P_max": a["B_P_max"], "rho_mean": a["rho_mean"],
            "CMASS_over_N_mean": a["CMASS_over_N_mean"], "busiest_shard_contact_share": a["load"]["largest_contact_rate"],
            "peak_over_mean": a["load"]["peak_over_mean"], "never_contacted": a["load"]["never_contacted"]}


def main():
    assert not os.path.exists(OUTP), "write-once"
    code = {}
    for p, want in MODULES.items():
        got = sha_file(os.path.join(REPO, p))
        assert got == want, (p, got)
        code[p] = got
    code["scratchpad/_l1d_carry.py"] = sha_file(os.path.abspath(__file__))
    assert sha_file(os.path.join(REPO, POPULATION[0])) == POPULATION[1]
    sp = os.path.join(DEV, "kscale_SUMMARY__v1.json")
    ssha = sha_file(sp)
    assert ssha.startswith(KSCALE_SUMMARY_SHA), ssha
    S = rj(sp)
    for k, v in S["code"].items():
        assert MODULES[v["path"]] == v["sha256"], k

    evidence, pins = {}, {}
    for ds in DATASETS:
        rp = os.path.join(DEV, "kscale_%s__v1.json" % ds)
        R = rj(rp)
        assert R["dataset"] == ds and R["mode"] == "L1_DEVELOPMENT_KSCALE"
        assert R["population"]["file"]["sha256"] == POPULATION[1]
        assert R["code"]["harness"]["sha256"] == MODULES["scratchpad/_l1d_kscale.py"]
        assert R["code"]["arms"]["sha256"] == MODULES["scratchpad/_l1d_arms.py"] and R["code"]["lib"]["sha256"] == MODULES["scratchpad/_l1d_lib.py"]
        for p, s in R["structures"]["imports"].items():
            assert MODULES[p] == s, p
        npz = os.path.join(DEV, R["npz"]["path"])
        assert sha_file(npz) == R["npz"]["sha256"]
        L = rj(os.path.join(DEV, "loc_%s__v1.json" % ds))
        assert L["code"]["sha256"] == MODULES["scratchpad/_l1d_loc.py"] and L["population"]["file"]["sha256"] == POPULATION[1]
        st = R["structures"]
        C = st["cells"]
        pins[ds] = {"record": {"path": rel(rp), "sha256": sha_file(rp), "npz": rel(npz), "npz_sha256": R["npz"]["sha256"]},
                    "population": {"n_rows": R["n_rows"], "query_ids_sha256": R["population"]["query_ids_sha256"], "label": R["population"]["label"]},
                    "served_cells": st["served_cells"], "native_K": st["native_K"], "K_grid": st["K_grid"],
                    "maps": {c: ({"file": C[c]["file"], "sha256": C[c]["sha256"]} if "file" in C[c] else
                                 {"rule": C[c]["rule"], "sha256_int64_vector": C[c]["sha256_int64_vector"], "seed": C[c]["seed"]})
                             | {"partitioner": C[c]["partitioner"], "K": C[c]["K"]} for c in sorted(C)},
                    "absent_cells": st["absent_cells"]}
        D = S["datasets"][ds]
        assert D["n_rows"] == R["n_rows"] and str(D["K_REF"]) == "100" and ARM in D["arms"]
        n = D["n_rows"]
        E = {"n_rows": n, "unrouted_n": D["unrouted_n"], "families": {}}
        for fam in FAMILIES[ds]:
            cs = kcells(D, fam)
            rows = {c: systems_row(D, c, ARM) for c in cs}
            big = cs[-1]
            holds = {}
            for b in BNS:
                ks = [c for c in cs if D["cells"][c]["K"] >= 250]
                d = [(D["cells"][c]["arms"][ARM]["B_N"][b]["n"] - D["unrouted_n"][b], D["cells"][c]["K"]) for c in ks]
                worst = min(d)
                h = D["holds_over_K"][fam][b]["K>=250"]
                mine = {t: worst[0] >= -TOL_BOUND[t][n] for t in TOL_BOUND}
                assert all(mine[t] == h["arms"][ARM]["holds"][t] for t in TOL_BOUND), (ds, fam, b)
                holds[b] = {"worst_delta_rows": sgn(worst[0]), "worst_at_K": worst[1], "holds_tol_0.01": mine["0.01"], "holds_tol_0.005": mine["0.005"],
                            "smallest_grid_alpha_holding_tol_0.01": h["by_tol"]["0.01"]["smallest_alpha"],
                            "base_ES_KNEE_SDIV_holds_tol_0.01": h["by_tol"]["0.01"]["base_holds"]}
            anchor = {b: sgn(D["cells"][fam + "_k100"]["arms"][ARM]["B_N"][b]["n"] - D["unrouted_n"][b]) for b in BNS}
            contrast = {lab: {k: v for k, v in systems_row(D, big, arm).items() if k != "B_N"} | {"B_N_1000_delta_rows": systems_row(D, big, arm)["B_N"]["1000"]["delta_rows"]}
                        for arm, lab in ((BASE, "ES.KNEE_SDIV (section 35)"), (ARM, "alpha 0.75 (carried forward)"), (A10, "alpha 1.0"))}
            svr = {c: D["structural_vs_random"][c][ARM] for c in cs if c in D["structural_vs_random"] and ARM in D["structural_vs_random"][c]}
            E["families"][fam] = {"alpha_0.75_systems_rows": rows, "tolerance_over_K_ge_250": holds, "K100_anchor_delta_rows": anchor,
                                  "largest_K_contrast": {"K": D["cells"][big]["K"], "arms": contrast},
                                  "structural_vs_balanced_random": svr}
        E["per_hop_at_1000"] = D["per_hop_at_1000"]
        evidence[ds] = E

    rec = {
        "RECORD": "L1_ROUTING_CARRY_FORWARD", "version": 1, "date": "2026-09-29", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "STATUS": "CARRIED_FORWARD: the current L1 routing rule by ruling (advisor and user, 2026-09-29). Development evidence only; NOT "
                  "held-out confirmed; the architecture is not final (KB multi-hop, L2 and L3 are open). L1 routing development is stopped.",
        "ruling": {"advisor_verbatim": ADVISOR, "user_verbatim": USER,
                   "chosen_option": "Record the ruling only: write up the ruling and hash-freeze the spec, but read no held-out rows yet"},
        "spec": {
            "L1": "ONE static A^T x per query (no recursive traversal, no learned parameters); partitions are routing shards",
            "L1a_node_localisation": "IR_L1 = RRF(FLAT, LOC), K0 60 (sections 32-34; FLAT = dense + SPLADE RRF; LOC = the static one-hop "
                                     "node localisation); served order: the first B_N nodes of IR_L1 inside the contacted partitions",
            "L1b_shard_ranking": "ES on the deployed K map",
            "L1b_coarse_fan_out": "B100(q) = KNEE_SDIV(q) on the same partitioner's K = 100 map (section 35 arithmetic, unchanged)",
            "L1b_fan_out": "B_P(q, K) = min(K, ceil(B100(q) * (K / 100)^0.75))",
            "alpha": ALPHA, "K_REF": 100,
            "min_never_binds": "at K >= 100: B100(q) <= 100 and (K / 100)^0.75 <= K / 100, so B100(q) (K / 100)^0.75 <= K",
            "no_localised_evidence": "a query without localised evidence contacts every partition",
            "router_state": "two node -> shard tables of the same partitioner: the deployed K map and the K = 100 map (one small integer "
                            "per node each); the K maps are not nested in the K = 100 map",
            "B_N": "not fixed by this ruling; the held-out criterion's B_N is undecided (user: no preference)",
        },
        "measures": ["recall: ALL-gold at B_N against unrouted (IR_L1 on the whole corpus, same B_N)",
                     "fan-out: B_P and rho = B_P / K", "local scan: CMASS / N (nodes held by the contacted shards over N)",
                     "load skew: busiest-shard contact share and peak / mean shard load; identity peak / mean = busiest share / rho"],
        "evidence": evidence,
        "pins": {"kscale_summary": {"path": rel(sp), "sha256": ssha}, "population": {"path": POPULATION[0], "sha256": POPULATION[1]}, "datasets": pins},
        "limitations": [
            "alpha 0.75 is the smallest alpha of the tested grid {0.65, 0.75, 0.85, 1.0} holding tol 0.01 at B_N 1000 on every "
            "structural map for K >= 250; alphas between 0.65 and 0.75 were never run (grid minimality, not a continuous optimum)",
            "tolerance by B_N (alpha 0.75, structural maps, K >= 250): holds tol 0.005 up to B_N 500 and tol 0.01 up to B_N 1000; fails "
            "tol 0.01 at B_N 2000 and 5000 (per-family worst deltas in evidence.<ds>.families.<fam>.tolerance_over_K_ge_250)",
            "B_N 5000 parity is a known limitation of the coarse-anchor count (advisor ruling): the K = 100 anchor is already below "
            "unrouted there (evidence K100_anchor_delta_rows); the cause is attributed to anchor-count underestimation, not isolated by "
            "an experiment",
            "load skew still grows with K under alpha 0.75: at the largest K the busiest shard is contacted by nearly every query, so "
            "peak / mean is close to 1 / rho (evidence largest_K_contrast); alpha 1.0 flattens it by contacting ~a third of the shards",
            "structural beats balanced random at every K and B_N >= 500 on MetaQA and MuSiQue (every seed); on SQuAD the margin is small "
            "at K <= 500 and within 0.002 either way at K >= 1000 (evidence structural_vs_balanced_random)",
            "the anchor count is estimated on a different partition (K = 100) than the one served (K); the K maps are not nested in it",
            "development populations only (legacy-exposed, freely reusable), three mapped corpora, the grid K plus the native K; "
            "MetaQA PHG K 5000 is PARTITION_INVALID and absent",
            "MetaQA hop-2 / hop-3 ALL-gold stays low under the one-hop L1 (evidence per_hop_at_1000); by ruling this is for "
            "partition-local completion, L2 or L3 to recover, never for recursion inside L1",
        ],
        "held_out_plan": {
            "status": "PLANNED, NOT SELECTED, NOT READ (no held-out row has been selected or read)",
            "A_in_domain_later": {"MetaQA": "666 x 3 hops = 1,998 never-used dev rows", "SQuAD": "2,000 never-used dev rows",
                                  "MuSiQue": "2,000 never-used train rows (every dev row is already used)",
                                  "when": "once the full L1 -> L2 -> L3 system is stable"},
            "B_transfer_later_sealed": {"WebQSP": 717, "HotpotQA": 969, "2Wiki": 996, "status": "sealed"},
            "criterion_B_N": "undecided",
        },
        "next": ["MetaQA development on legacy-exposed rows: hop-2 / hop-3 ALL-gold, 5-10 and 11+ answer queries, golds outside one-hop "
                 "L1 but inside routed partitions, golds needing another STRUCT hop -> decide partition-local completion vs L2 vs L3",
                 "WebQSP shard maps as development / transfer infrastructure (results/L1_HOST lane); split B stays sealed",
                 "finalize L1 + L2 + L3", "then held-out A; later held-out B"],
        "guards": {"held_out_rows_read": 0, "split_B_rows_read": 0, "TEST_rows_read": 0, "data_written": False},
        "code": code,
    }
    with io.open(OUTP + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUTP + ".tmp", OUTP)
    print(rel(OUTP), sha_file(OUTP))
    for ds in DATASETS:
        for fam, F in evidence[ds]["families"].items():
            print(ds, fam, {b: (v["worst_delta_rows"], v["worst_at_K"], v["holds_tol_0.01"], v["holds_tol_0.005"])
                            for b, v in F["tolerance_over_K_ge_250"].items()})


if __name__ == "__main__":
    main()
