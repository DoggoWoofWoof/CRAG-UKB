"""Cross-cache summary of the pre-registered partition-prototype ladder (PREREGISTRATION_PROTOTYPE_LADDER.json).

Reads only the five write-once records results/L1_COVPART/proto_A_<cache>.json and applies the pre-registered rule:
per level, m* = the smallest ladder m at which FPS_m PRESERVES the QMAX chain on ALL five DEV_A caches (PRESERVES =
net loss vs QMAX at that level <= max(2, round(0.5 % of n_A)) and no significant loss at 0.05, exact McNemar); the
headline answer is the PRIMARY level (the composite end to end).  Also: the centroid and RAND rows, the recall
hierarchy per cache (centroid -> prototypes -> node votes -> QMAX), the outlier and cohesion diagnostics side by side,
and the mechanical check of every stated prediction.  No new number is computed from data.
    python -u scratchpad/_l1c_proto_summary.py        -> results/L1_COVPART/proto_SUMMARY.json (write-once)
"""
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results", "L1_COVPART")
PRE = os.path.join(OUT, "PREREGISTRATION_PROTOTYPE_LADDER.json")
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
LADDER = (1, 2, 4, 8, 16, 32)
LEVELS = ("ALONE", "F0", "SAFE", "PRIMARY")
CENTS = ("CENT_MEAN", "CENT_NORM")
DIAG_VARIANTS = ("CENT_MEAN", "CENT_NORM", "FPS_1", "FPS_4", "FPS_8", "FPS_32", "RAND_8")
K_DROP = "dropped_gold_nodes (served by QMAX, not by the variant, in lost queries)"
K_KEEP = "retained_gold_nodes (served by both, all queries)"
K_PCT = "centroid_distance_percentile_median (0 = most central node of its block, 1 = most peripheral)"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


pre = json.load(open(PRE, encoding="utf-8"))
pre_sha = sha_file(PRE)
for mod, pn in pre["code"]["new_modules"].items():
    assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
fp_out = os.path.join(OUT, "proto_SUMMARY.json")
assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
recs = {}
for c in CACHES:
    p = os.path.join(OUT, "proto_A_%s.json" % c)
    assert os.path.exists(p), "missing record %s" % p
    r = json.load(open(p, encoding="utf-8"))
    assert r["code"]["sha256"] == pre["code"]["new_modules"]["_l1c_proto.py"]["sha256"], c
    assert r["preregistration"]["sha256"] == pre_sha, c
    recs[c] = r


def var(c, L, v):
    return recs[c]["levels"][L]["variants"][v]


def smallest_all(L, tag):
    ok = [k for k in LADDER if all(var(c, L, "%s_%d" % (tag, k))["PRESERVES"] for c in CACHES)]
    return (min(ok) if ok else None), ok


def binding(L, ms):
    if ms == LADDER[0]:
        return []
    k = LADDER[-1] if ms is None else LADDER[LADDER.index(ms) - 1]
    return [c for c in CACHES if not var(c, L, "FPS_%d" % k)["PRESERVES"]]


levels = {}
for L in LEVELS:
    ms, ok = smallest_all(L, "FPS")
    mr, okr = smallest_all(L, "RAND")
    per_cache = {c: recs[c]["levels"][L]["smallest_preserving_FPS_m"] for c in CACHES}
    levels[L] = {
        "label": ("FPS_PRESERVES_QMAX_AT_m=%d" % ms) if ms else "NO_FPS_LADDER_POINT_PRESERVES_QMAX (m* > 32)",
        "m_star_FPS (all five caches)": ms, "FPS_m_preserving_on_all_five": ok,
        "FPS_preserving_on_all_five_from_m_star_upward": bool(ms) and ok == [k for k in LADDER if k >= ms],
        "smallest_preserving_FPS_m_per_cache": per_cache,
        "binding_caches (fail at the ladder point just below m*, or at m = 32 when there is no m*)": binding(L, ms),
        "m_star_RAND (control)": mr, "RAND_m_preserving_on_all_five": okr,
        "centroids_preserve (per cache)": {v: {c: var(c, L, v)["PRESERVES"] for c in CACHES} for v in CENTS},
        "table (ALL / net loss vs QMAX / PRESERVES)": {c: {"QMAX": recs[c]["levels"][L]["ALL_reference"],
                                                           **{v: [var(c, L, v)["ALL"], var(c, L, v)["vs_QMAX"]["net_loss"], var(c, L, v)["PRESERVES"]]
                                                              for v in list(CENTS) + ["FPS_%d" % k for k in LADDER] + ["RAND_%d" % k for k in LADDER]}}
                                                      for c in CACHES},
        "tolerance_net_loss_per_cache": {c: recs[c]["levels"][L]["tolerance_net_loss"] for c in CACHES}}

# ---------------------------------------------------------------- the recall hierarchy the request asked for (centroid -> prototypes -> node votes -> QMAX)
hier = {}
for c in CACHES:
    r = recs[c]
    al = r["alone_references (IVF-like endpoint comparators)"]
    hier[c] = {
        "ALONE (routing by one channel; P50)": {**{v: var(c, "ALONE", v)["ALL"] for v in list(CENTS) + ["FPS_%d" % k for k in LADDER]},
                                               "DENSE_VOTE": al["DENSE_VOTE_ALONE"], "SPLADE_VOTE": al["SPLADE_VOTE_ALONE"],
                                               "NODE_VOTE (L1 = RRF of the two)": al["L1"], "QMAX": al["QMAX_ALONE"]},
        "recall_blocks@50 vs QMAX_ALONE": {**{v: var(c, "ALONE", v)["recall_blocks@50 (mean |P50_alone & P50_QMAX_alone| / 50)"]
                                              for v in list(CENTS) + ["FPS_%d" % k for k in LADDER]}, **al["recall_blocks@50 vs QMAX_ALONE"]},
        "F0 (node votes + stage 2; P50)": {**{v: var(c, "F0", v)["ALL"] for v in list(CENTS) + ["FPS_%d" % k for k in LADDER]},
                                           "no stage 2 (L1)": r["levels"]["F0"]["ALL_no_stage2_reference"], "QMAX": r["levels"]["F0"]["ALL_reference"]},
        "recall_blocks@50 vs QMAX_F0": {v: var(c, "F0", v)["overlap_with_QMAX_chain"]["recall_blocks@50 (mean |P50 & P50_QMAX_F0| / 50)"]
                                        for v in list(CENTS) + ["FPS_%d" % k for k in LADDER]},
        "PRIMARY (the composite end to end)": {**{v: var(c, "PRIMARY", v)["ALL"] for v in list(CENTS) + ["FPS_%d" % k for k in LADDER]},
                                               "no stage 2 (BALANCED_H2_PATCH1)": r["levels"]["PRIMARY"]["ALL_no_stage2_reference"],
                                               "QMAX": r["levels"]["PRIMARY"]["ALL_reference"]},
        "vectors_scored_per_query": r["representatives"]["vectors_scored_per_query"]}

# ---------------------------------------------------------------- the outlier diagnostic and the cohesion diagnostic side by side
outl = {}
for c in CACHES:
    outl[c] = {}
    for L in ("ALONE", "F0", "PRIMARY"):
        outl[c][L] = {}
        for v in DIAG_VARIANTS:
            e = recs[c]["outlier_diagnostic"]["per_level"][L][v]
            d, k = e[K_DROP], e[K_KEEP]
            row = {"lost_queries": e["lost_queries (QMAX covers, variant not)"], "gained_queries": e["gained_queries (variant covers, QMAX not)"],
                   "dropped_n": d["n"], "retained_n": k["n"]}
            if d["n"]:
                row["percentile_median dropped / retained"] = [d[K_PCT], k.get(K_PCT)]
                row["peripheral_quartile_share dropped / retained"] = [d["share_in_peripheral_quartile (percentile >= 0.75)"], k.get("share_in_peripheral_quartile (percentile >= 0.75)")]
                row["block_max_node_share dropped / retained"] = [d["share_that_is_the_block_max_node_for_its_query"], k.get("share_that_is_the_block_max_node_for_its_query")]
                if "share_that_is_a_representative" in d:
                    row["representative_share dropped / retained"] = [d["share_that_is_a_representative"], k.get("share_that_is_a_representative")]
                mw = [x for x in e if x.startswith("mann_whitney_p")]
                if mw:
                    row["mann_whitney_p"] = e[mw[0]]
            outl[c][L][v] = row
coh = {}
for c in CACHES:
    ch = recs[c]["cohesion"]
    ra = ch["actual_partition"]
    rr = [ch[k] for k in ch if k.startswith("size_matched_random_partition")][0]
    coh[c] = {"global_resultant_length": ch["global_resultant_length (all nodes; the anisotropy floor)"],
              "resultant_length actual / random": [ra["resultant_length_size_weighted_mean (1 = identical vectors)"], rr["resultant_length_size_weighted_mean (1 = identical vectors)"]],
              "mean_pairwise_cosine actual / random": [ra["mean_pairwise_cosine_inside_a_block_size_weighted"], rr["mean_pairwise_cosine_inside_a_block_size_weighted"]],
              "covering_radius_r1 actual / random": [ra["fps_covering_radius_size_weighted_mean_by_m (Euclidean, stored vectors)"]["1"], rr["fps_covering_radius_size_weighted_mean_by_m (Euclidean, stored vectors)"]["1"]],
              "r4_over_r1 actual / random": [ra["covering_radius_ratio_r_m_over_r_1_mean"]["4"], rr["covering_radius_ratio_r_m_over_r_1_mean"]["4"]],
              "r32_over_r1 actual / random": [ra["covering_radius_ratio_r_m_over_r_1_mean"]["32"], rr["covering_radius_ratio_r_m_over_r_1_mean"]["32"]]}

# ---------------------------------------------------------------- the mechanical check of every stated prediction
mP, mF, mA = (levels[L]["m_star_FPS (all five caches)"] for L in ("PRIMARY", "F0", "ALONE"))
BIG = 10 ** 6                                                   # "no ladder point" sorts above every m


def nz(x):
    return BIG if x is None else x


drop_mu = recs["musique"]["outlier_diagnostic"]["per_level"]["ALONE"]["CENT_NORM"]
checks = {
    "P1": not var("musique", "F0", "CENT_NORM")["PRESERVES"],
    "P2a": not any(var("musique", "PRIMARY", v)["PRESERVES"] for v in ("CENT_MEAN", "CENT_NORM", "FPS_1")),
    "P2b": all(var(c, "PRIMARY", v)["PRESERVES"] for c in ("metaqa", "metaqa_phg") for v in ("CENT_MEAN", "CENT_NORM", "FPS_1")),
    "P2c": not any(var(c, "F0", "CENT_NORM")["PRESERVES"] for c in ("squad", "squad_phg")),
    "P3a": mP in (16, 32),
    "P3b": nz(mF) >= nz(mP) and nz(mA) >= nz(mF),
    "P3c": mA is None,
    "P4": all(nz(levels[L]["m_star_FPS (all five caches)"]) <= nz(levels[L]["m_star_RAND (control)"]) for L in ("F0", "PRIMARY")),
    "P5": bool(drop_mu[K_DROP]["n"] and drop_mu[K_KEEP]["n"] and drop_mu[K_DROP][K_PCT] > drop_mu[K_KEEP][K_PCT]),
    "P6": all(coh[c]["resultant_length actual / random"][0] > coh[c]["resultant_length actual / random"][1] for c in CACHES)}
pred = {k: {"stated": pre["predictions_stated_now"][k], "held": bool(v)} for k, v in checks.items()}

res = {"RECORD": "PROTOTYPE_LADDER_SUMMARY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "question": pre["question"], "preregistration": {"path": "results/L1_COVPART/PREREGISTRATION_PROTOTYPE_LADDER.json", "sha256": pre_sha},
       "records": {c: {"path": "results/L1_COVPART/proto_A_%s.json" % c, "sha256": sha_file(os.path.join(OUT, "proto_A_%s.json" % c)), "n_split_A": recs[c]["n_split_A"]} for c in CACHES},
       "code": {"path": "scratchpad/_l1c_proto_summary.py", "sha256": sha_file(os.path.abspath(__file__))},
       "HEADLINE (PRIMARY level, the composite end to end)": levels["PRIMARY"]["label"],
       "labels_by_level": {L: levels[L]["label"] for L in LEVELS},
       "levels": levels, "recall_hierarchy": hier, "outlier_diagnostic": outl, "cohesion": coh, "predictions": pred,
       "what_this_does_not_decide": pre["decision_rule_pre_registered"]["what_is_not_decided"]}
with open(fp_out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=True)
print("HEADLINE %s | by level %s" % (res["HEADLINE (PRIMARY level, the composite end to end)"], json.dumps(res["labels_by_level"])))
print("predictions %s" % json.dumps({k: v["held"] for k, v in pred.items()}))
print("-> %s sha256 %s" % (os.path.relpath(fp_out, REPO), sha_file(fp_out)[:12]))
