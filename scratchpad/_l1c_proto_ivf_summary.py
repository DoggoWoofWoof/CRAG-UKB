"""IVF-centred reading of the partition-prototype ladder (PREREGISTRATION_PROTOTYPE_LADDER_IVF_ADDENDUM.json), written
after the user's clarification of 2026-09-25: "the point of this wasnt to make qmax better it was to see if we can instead of
node voting do an ivfesque approach for selecting the parititions".

Question: can an IVF-style coarse quantizer over the served graph partitions -- score q against a few fixed, query-free
representatives per block, probe the top 50 blocks (nprobe = 50 = the P50 block budget), serve every node inside them --
replace node voting (the dense and the SPLADE top-100 hits vote for their own block and for the blocks of their directed
STRUCT out-neighbours, rr(sum) + rr(max) per channel, frozen RRF of the two channels = L1)?
The IVF endpoint is the ladder's ALONE level; every ALONE arm was already paired with node voting (L1) and with the dense
node vote alone (DENSE_VOTE_ALONE) by an exact McNemar inside the write-once proto_A_<cache>.json records.  This script reads
only those five records and computes nothing from data.
    python -u scratchpad/_l1c_proto_ivf_summary.py       -> results/L1_COVPART/proto_IVF_SUMMARY.json (write-once)
"""
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results", "L1_COVPART")
PRE = os.path.join(OUT, "PREREGISTRATION_PROTOTYPE_LADDER.json")
ADD = os.path.join(OUT, "PREREGISTRATION_PROTOTYPE_LADDER_IVF_ADDENDUM.json")
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
KBS = ["metaqa", "metaqa_phg"]
TEXT = ["squad", "squad_phg", "musique"]
LADDER = (1, 2, 4, 8, 16, 32)
CENTS = ["CENT_MEAN", "CENT_NORM"]
FPS = ["FPS_%d" % k for k in LADDER]
RND = ["RAND_%d" % k for k in LADDER]
LIMIT = "QMAX_ALONE"                                          # the ladder's limit: every node of a block is a representative
IVF_ARMS = CENTS + FPS + RND + [LIMIT]
NODE_VOTE, DENSE_VOTE, SPLADE_VOTE = "L1", "DENSE_VOTE_ALONE", "SPLADE_VOTE_ALONE"
ALPHA_LOSS = 0.05


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


add = json.load(open(ADD, encoding="utf-8"))
add_sha = sha_file(ADD)
pre_sha = sha_file(PRE)
assert pre_sha == add["reads_the_runs_of"]["sha256"], "the original pre-registration changed"
for mod, pn in add["code"]["new_modules"].items():
    assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "%s changed since the addendum" % mod
fp_out = os.path.join(OUT, "proto_IVF_SUMMARY.json")
assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
recs = {}
for c in CACHES:
    p = os.path.join(OUT, "proto_A_%s.json" % c)
    assert os.path.exists(p), "missing record %s" % p
    r = json.load(open(p, encoding="utf-8"))
    assert r["preregistration"]["sha256"] == pre_sha and r["code"]["sha256"] == add["reads_the_runs_of"]["ladder_module_sha256"], c
    recs[c] = r


def arm_name(v):
    return v if v == LIMIT else v + "_ALONE"


def preserves(o, tol):
    net = o["lost"] - o["gained"]
    return bool(net <= tol and not (o["lost"] > o["gained"] and o["p"] < ALPHA_LOSS))


def paired(o):
    return {"gained": o["gained"], "lost": o["lost"], "net_queries": o["gained"] - o["lost"], "p": o["p"], "label": o["label"]}


per_cache = {}
for c in CACHES:
    r = recs[c]
    A = r["arms"]
    nA = r["n_split_A"]
    tol = r["levels"]["ALONE"]["tolerance_net_loss"]
    vec = r["representatives"]["vectors_scored_per_query"]
    ivf = {}
    for v in IVF_ARMS:
        a = A[arm_name(v)]
        o, od = a["ALL_vs_" + NODE_VOTE], a["ALL_vs_" + DENSE_VOTE]
        ivf[v] = {"ALL": a["ALL"], "ANY": a["ANY"],
                  "points_vs_node_vote": round(100.0 * (o["gained"] - o["lost"]) / nA, 2),
                  "vs_node_vote": paired(o), "PRESERVES_NODE_VOTE": preserves(o, tol),
                  "vs_dense_vote": paired(od), "PRESERVES_DENSE_VOTE": preserves(od, tol),
                  "per_hop_ALL": {h: e["ALL"] for h, e in a["per_hop"].items()},
                  "prototype_vectors_scored_per_query": vec["QMAX" if v == LIMIT else v]}
    fok = [k for k in LADDER if ivf["FPS_%d" % k]["PRESERVES_NODE_VOTE"]]
    rok = [k for k in LADDER if ivf["RAND_%d" % k]["PRESERVES_NODE_VOTE"]]
    fok_d = [k for k in LADDER if ivf["FPS_%d" % k]["PRESERVES_DENSE_VOTE"]]
    if fok:
        status = "IVF_PRESERVES_NODE_VOTE_AT_m=%d" % min(fok)
    elif ivf[LIMIT]["PRESERVES_NODE_VOTE"]:
        status = "ONLY_THE_LADDER_LIMIT_PRESERVES (every node a representative; no FPS_m with m <= 32)"
    else:
        status = "NOT_EVEN_THE_LADDER_LIMIT_PRESERVES_NODE_VOTE"
    per_cache[c] = {
        "status": status, "n_split_A": nA, "tolerance_net_loss": tol, "N": r["N"], "npart": r["npart"],
        "block_size_min_median_max": r["block_size_min_median_max"],
        "node_vote (L1)": A[NODE_VOTE]["ALL"], "dense_vote_alone": A[DENSE_VOTE]["ALL"], "splade_vote_alone": A[SPLADE_VOTE]["ALL"],
        "node_vote_per_hop_ALL": {h: e["ALL"] for h, e in A[NODE_VOTE]["per_hop"].items()},
        "smallest_FPS_m_preserving_node_vote": min(fok) if fok else None, "FPS_m_preserving_node_vote": fok,
        "FPS_preserving_from_smallest_m_upward": bool(fok) and fok == [k for k in LADDER if k >= min(fok)],
        "smallest_RAND_m_preserving_node_vote": min(rok) if rok else None,
        "centroids_preserve_node_vote": {v: ivf[v]["PRESERVES_NODE_VOTE"] for v in CENTS},
        "ladder_limit_preserves_node_vote": ivf[LIMIT]["PRESERVES_NODE_VOTE"],
        "smallest_FPS_m_preserving_dense_vote (dense-only comparison)": min(fok_d) if fok_d else None,
        "ivf": ivf,
        "recall_as_we_move_left (ALL; points vs node vote)": [[v, ivf[v]["ALL"], ivf[v]["points_vs_node_vote"]] for v in CENTS + FPS + [LIMIT]]
                                                             + [["DENSE_VOTE", A[DENSE_VOTE]["ALL"], None], ["SPLADE_VOTE", A[SPLADE_VOTE]["ALL"], None],
                                                                ["NODE_VOTE (L1)", A[NODE_VOTE]["ALL"], 0.0]]}


def m_star(caches, tag="FPS"):
    ok = [k for k in LADDER if all(per_cache[c]["ivf"]["%s_%d" % (tag, k)]["PRESERVES_NODE_VOTE"] for c in caches)]
    return (min(ok) if ok else None), ok


mt, okt = m_star(TEXT)
mtr, _ = m_star(TEXT, "RAND")
ma, oka = m_star(CACHES)
text_verdict = {"label": ("IVF_REPLACES_NODE_VOTING_ON_TEXT_AT_m=%d" % mt) if mt else "IVF_DOES_NOT_REPLACE_NODE_VOTING_ON_TEXT (no FPS_m with m <= 32 preserves it on all three)",
                "m_star_text_FPS": mt, "FPS_m_preserving_on_all_three": okt,
                "from_m_star_upward": bool(mt) and okt == [k for k in LADDER if k >= mt],
                "m_star_text_RAND (control)": mtr,
                "binding_text_caches (fail at the ladder point just below m*, or at m = 32 when there is no m*)":
                    [] if mt == LADDER[0] else [c for c in TEXT if not per_cache[c]["ivf"]["FPS_%d" % (LADDER[-1] if mt is None else LADDER[LADDER.index(mt) - 1])]["PRESERVES_NODE_VOTE"]],
                "ladder_limit_preserves_on_all_three": all(per_cache[c]["ladder_limit_preserves_node_vote"] for c in TEXT)}
universal = {"label (observed on the two metaqa caches before the addendum; not a test)":
             ("IVF_REPLACES_NODE_VOTING_ON_ALL_FIVE_AT_m=%d" % ma) if ma else "IVF_DOES_NOT_REPLACE_NODE_VOTING_UNIVERSALLY",
             "m_star_all_five_FPS": ma, "kb_status": {c: per_cache[c]["status"] for c in KBS}}
chk = {"T1": not any(per_cache[c]["centroids_preserve_node_vote"]["CENT_NORM"] for c in ("squad", "squad_phg")),
       "T2": not per_cache["musique"]["centroids_preserve_node_vote"]["CENT_NORM"],
       "T3": all(per_cache[c]["smallest_FPS_m_preserving_node_vote"] is not None for c in ("squad", "squad_phg")),
       "T4": per_cache["musique"]["smallest_FPS_m_preserving_node_vote"] is None and mt is None}
pred = {k: {"stated": add["predictions_stated_now (text caches, unseen)"][k], "held": bool(v)} for k, v in chk.items()}

res = {"RECORD": "PROTOTYPE_LADDER_IVF_SUMMARY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "question": add["question"], "addendum": {"path": "results/L1_COVPART/PREREGISTRATION_PROTOTYPE_LADDER_IVF_ADDENDUM.json", "sha256": add_sha},
       "records": {c: {"path": "results/L1_COVPART/proto_A_%s.json" % c, "sha256": sha_file(os.path.join(OUT, "proto_A_%s.json" % c))} for c in CACHES},
       "code": {"path": "scratchpad/_l1c_proto_ivf_summary.py", "sha256": sha_file(os.path.abspath(__file__))},
       "HEADLINE (text caches; pre-registered in the addendum)": text_verdict["label"],
       "text_verdict": text_verdict, "universal_verdict": universal,
       "status_by_cache": {c: per_cache[c]["status"] for c in CACHES},
       "per_cache": per_cache, "predictions": pred,
       "what_this_does_not_decide": add["what_is_not_decided"]}
with open(fp_out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=True)
print("HEADLINE %s | status %s" % (res["HEADLINE (text caches; pre-registered in the addendum)"], json.dumps(res["status_by_cache"])))
print("predictions %s" % json.dumps({k: v["held"] for k, v in pred.items()}))
print("-> %s sha256 %s" % (os.path.relpath(fp_out, REPO), sha_file(fp_out)[:12]))
