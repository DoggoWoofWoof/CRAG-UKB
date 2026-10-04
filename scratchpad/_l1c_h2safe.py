"""H2_AS_SAFE_CANDIDATE_GENERATOR -- the pinned MICRO_L3_H2 beam used ONLY as a candidate source for the canonical SAFE structure
swap (ruling of 2026-09-15: "H2 does not obtain its own RRF channel. Its output is merely another candidate source for the existing
structure swap ... H2 candidate block rank = best beam rank of any node in that block ... evaluate it against canonical SAFE").
STATUS = POSTHOC_MECHANISM_TEST (DEV_A already read; pre-registered in PREREGISTRATION_H2_SAFE_DEV_A.json before any run).

    canonical L1 (base_rank, unchanged)  ->  core = base_rank[:44] locked  ->  boundary = base_rank[44:50]
    candidates = boundary u static structural challengers (frozen expand_dir nodes, S4 order) u retrieval challengers (rrf200
                 continuation) u H2 blocks (own blocks of the beam's visited nodes in first-visit order, seeds excluded)
    F6 score   = 1/(K0+cpos) + 1/(K0+spos') + 1/(K0+rpos), spos'[b] = min(static structural rank, H2 block rank); ties by canonical
                 rank then block id; exactly 6 chosen (unchanged _l1kb_core.f6_select); final = core u 6 = exactly 50
Arms: L1 (served BASE) | SAFE (canonical B6_S4_F6_Ms64_Mr32, asserted equal to the frozen replay records on DEV_A rows) |
      SAFE_H2 (primary).  The beam is asserted identical to section 14.1 by rebuilding MICRO_L3_H2 and matching its record.
    python -u _l1c_h2safe.py <cache>      -> results/L1_COVPART/h2safe_A_<cache>.json
    python -u _l1c_h2safe.py --summary    -> results/L1_COVPART/h2safe_A_SUMMARY.json (the pre-registered rule over the five records)
"""
import hashlib
import json
import os
import sys

import numpy as np

import _l1g_core as G
import _l1kb_core as KB
import _l1ps_router as RT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PINNED_SHA = "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5"
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json",
               "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json",
               "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}
CFG = dict(KB.BASE_CFG)                                      # B=6, M_struct=64, M_ret=32, S4, F6 -- unchanged
B = CFG["B"]


def summary():
    recs = {c: json.load(open(os.path.join(OUT, "h2safe_A_%s.json" % c), encoding="utf-8")) for c in CACHES}
    text_ok = {c: not recs[c]["arms"]["SAFE_H2"]["significant_loss_vs_SAFE"] for c in ("squad", "squad_phg", "musique")}
    mq = {}
    for c in ("metaqa", "metaqa_phg"):
        h = recs[c]["arms"]["SAFE_H2"]["per_hop"]["hop2"]["vs_SAFE"]
        mq[c] = {"gained": h["gained"], "lost": h["lost"], "p": h["p"], "pass": bool(h["gained"] > h["lost"] and h["p"] < 0.01)}
    verdict = "PASS" if all(text_ok.values()) and all(v["pass"] for v in mq.values()) else "FAIL"
    S = {"RECORD": "H2_AS_SAFE_CANDIDATE_GENERATOR_DEV_A_SUMMARY", "STATUS": "POSTHOC_MECHANISM_TEST",
         "rule": "text safety on squad/squad_phg/musique (no significant loss vs SAFE) AND hop-2 gain vs SAFE with p < 0.01 on both metaqa caches",
         "text_safety": text_ok, "metaqa_hop2_vs_SAFE": mq, "verdict": verdict,
         "per_cache": {c: {"L1": recs[c]["arms"]["L1"]["BASE_ALL"], "SAFE": recs[c]["arms"]["SAFE"]["BASE_ALL"], "SAFE_H2": recs[c]["arms"]["SAFE_H2"]["BASE_ALL"],
                           "SAFE_H2_vs_SAFE": recs[c]["arms"]["SAFE_H2"]["BASE_ALL_vs_SAFE"], "SAFE_H2_vs_L1": recs[c]["arms"]["SAFE_H2"]["BASE_ALL_vs_L1"],
                           "recovered_fraction_of_H2_RRF_hop2_gain": recs[c]["arms"]["SAFE_H2"].get("recovery_of_section14_gain", {}).get("hop2", {}).get("fraction")}
                       for c in CACHES}}
    G.S.wj(os.path.join(OUT, "h2safe_A_SUMMARY.json"), S)
    G.log("SUMMARY verdict %s text_ok %s metaqa %s" % (verdict, text_ok, mq))


if "--summary" in sys.argv:
    summary()
    sys.exit(0)

name = sys.argv[1]
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
assert hashlib.sha256(raw).hexdigest() == PINNED_SHA, "pinned uL3 module changed"
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
exec(head)                                                   # D, C, beams1, beams2, repair_channel, hard, hops, m, rowsA ... identical beam

# ---------------------------------------------------------------- identical beam: rebuild section 14.1's MICRO_L3_H2 and match its record
rep2, S2 = repair_channel([beams1[qi] + beams2[qi] for qi in range(nq)])
base_rank = np.asarray(C.base_rank, np.int64)
allv_rrf, _ = C.cover(X.fuse(C, base_rank, rep2, "RRF"))
allv_rrf = allv_rrf.astype(bool)
rec14 = json.load(open(os.path.join(OUT, "microl3_A_%s.json" % name), encoding="utf-8"))
a0 = np.asarray(D.base_all, bool)
g14, l14, _ = X.mcnemar(a0[m], allv_rrf[m])
assert round(float(allv_rrf[m].mean()), 4) == rec14["arms"]["MICRO_L3_H2"]["BASE_ALL"], "beam differs from section 14.1"
assert (g14, l14) == tuple(rec14["arms"]["MICRO_L3_H2"]["BASE_ALL_vs_L1"][k] for k in ("gained", "lost")), "beam differs from section 14.1"
G.log("identical beam: MICRO_L3_H2 rebuilt = %.4f (+%d/-%d) == section 14.1 record" % (allv_rrf[m].mean(), g14, l14))

# ---------------------------------------------------------------- canonical SAFE machinery (frozen code path), on the raw cache arrays
z0 = np.load(C.path, allow_pickle=True)
zz = {k: z0[k] for k in z0.files if k != "meta_json"}
meta = C.meta
assert int(meta["n_dev_queries"]) == nq
Cc = RT.build_cache(name, zz, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
ctxs = KB.contexts(zz, meta, Cc, B, CFG)
goldp = Cc["goldp"]
ind_base = np.asarray(Cc["ind_base"], bool)
assert (ind_base[m] == a0[m]).all(), "SAFE machinery BASE differs from the lane's served BASE"
SFs = Cc["sfull"][(CFG["M_struct"], CFG["agg"])]
RFs = Cc["rfull"][CFG["M_ret"]]

# H2 candidate source: own blocks of the visited nodes in first-visit order (hop-1 beam, then hop-2 beam); seeds excluded
h2list = [[] for _ in range(nq)]
h2pos = [{} for _ in range(nq)]
for qi in rowsA:
    bl = list(dict.fromkeys(int(hard[v]) for v in (beams1[qi] + beams2[qi])))
    h2list[qi] = bl
    h2pos[qi] = {b: r for r, b in enumerate(bl)}


def select(qi, with_h2):
    c = ctxs[qi]
    if not with_h2:
        Xs, _ = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
        return Xs
    chal = list(c["chal"]) + [b for b in h2list[qi] if b not in c["base50"]]
    spos = dict(c["spos"])
    for b, r in h2pos[qi].items():
        spos[b] = min(spos.get(b, 10 ** 9), r)
    Xs, _ = KB.f6_select(c["bnd"], chal, spos, c["rpos"], c["cpos"], B)
    return Xs


finals = {"L1": [], "SAFE": [], "SAFE_H2": []}
h2_credit = np.zeros(nq, np.int32)          # selected out-of-P50 blocks whose H2 rank beat their static structural rank (or had none)
h2_only = np.zeros(nq, np.int32)            # selected out-of-P50 blocks absent from BOTH frozen challenger lists
for qi in range(nq):
    c = ctxs[qi]
    finals["L1"].append(set(c["base50"]))
    finals["SAFE"].append(c["prot_set"] | set(select(qi, False)))
    X2 = select(qi, True)
    finals["SAFE_H2"].append(c["prot_set"] | set(X2))
    for b in X2:
        if b in c["base50"]:
            continue
        hp = h2pos[qi].get(b, 10 ** 9)
        if hp < c["spos"].get(b, 10 ** 9):
            h2_credit[qi] += 1
        if b not in c["spos"] and b not in c["rpos"]:
            h2_only[qi] += 1
for k, fs in finals.items():
    assert all(len(s) == P_MAIN for s in fs), k

# ---------------------------------------------------------------- canonical SAFE reproduction against the frozen replay record (DEV_A rows)
recS = json.load(open(os.path.join(G.X.REPO, SAFE_RECORD[name]), encoding="utf-8"))
assert int(recS["population"]["nq"]) == nq and os.path.basename(recS["cache"]["file"]) == os.path.basename(C.path), (recS["cache"]["file"], C.path)
ind = {k: np.array([int(goldp[qi] <= fs[qi]) for qi in range(nq)], bool) for k, fs in finals.items()}
ind_any = {k: np.array([int(bool(goldp[qi] & fs[qi])) for qi in range(nq)], bool) for k, fs in finals.items()}
assert (ind["L1"][m] == np.asarray(recS["_ind_BASE"], bool)[m]).all(), "BASE differs from the frozen replay record"
assert (ind["SAFE"][m] == np.asarray(recS["_ind_SAFE"], bool)[m]).all(), "canonical SAFE not reproduced on DEV_A rows"
G.log("canonical SAFE reproduced on DEV_A rows against %s (SAFE %.4f, BASE %.4f)" % (os.path.basename(SAFE_RECORD[name]), ind["SAFE"][m].mean(), ind["L1"][m].mean()))


# ---------------------------------------------------------------- evaluation
def mc(a, b, s):
    g_, l_, p_ = X.mcnemar(a[s], b[s])
    return {"gained": g_, "lost": l_, "p": p_}


def evaluate(arm):
    fs, iv, ia = finals[arm], ind[arm], ind_any[arm]
    churn = np.array([len(fs[qi] - finals["L1"][qi]) for qi in rowsA])
    adm = sum(len(goldp[qi] & (fs[qi] - finals["L1"][qi])) for qi in rowsA)
    ev = sum(len(goldp[qi] & (finals["L1"][qi] - fs[qi])) for qi in rowsA)
    o = {"arm": arm, "BASE_ALL": round(float(iv[m].mean()), 4), "ANY": round(float(ia[m].mean()), 4),
         "churn_per_query_mean": round(float(churn.mean()), 2), "queries_with_zero_churn": int((churn == 0).sum()),
         "gold_blocks_admitted_vs_base50": int(adm), "gold_blocks_evicted_vs_base50": int(ev), "per_hop": {}}
    if arm == "SAFE_H2":
        o["H2_credited_admissions_per_query_mean"] = round(float(h2_credit[rowsA].mean()), 2)
        o["queries_with_any_H2_credited_admission"] = int((h2_credit[rowsA] > 0).sum())
        o["H2_only_admissions_per_query_mean"] = round(float(h2_only[rowsA].mean()), 2)
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        o["per_hop"]["hop%d" % h] = {"n": int(s.sum()), "BASE_ALL": round(float(iv[s].mean()), 4)}
        if arm != "L1":
            o["per_hop"]["hop%d" % h]["vs_L1"] = mc(ind["L1"], iv, s)
        if arm == "SAFE_H2":
            o["per_hop"]["hop%d" % h]["vs_SAFE"] = mc(ind["SAFE"], iv, s)
    if arm != "L1":
        o["BASE_ALL_vs_L1"] = mc(ind["L1"], iv, m)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_vs_L1"] = {"n": int(s.sum()), **mc(ind["L1"], iv, s), "pooled_L1": round(float(ind["L1"][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
    if arm == "SAFE_H2":
        o["BASE_ALL_vs_SAFE"] = mc(ind["SAFE"], iv, m)
        o["significant_loss_vs_SAFE"] = bool(o["BASE_ALL_vs_SAFE"]["lost"] > o["BASE_ALL_vs_SAFE"]["gained"] and o["BASE_ALL_vs_SAFE"]["p"] < 0.05)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_vs_SAFE"] = {"n": int(s.sum()), **mc(ind["SAFE"], iv, s), "pooled_SAFE": round(float(ind["SAFE"][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
            # fraction of section 14.1's MICRO_L3_H2 gain over L1 that SAFE_H2 recovers over SAFE (reported, not thresholded)
            rec_h2 = rec14["arms"]["MICRO_L3_H2"]
            recov = {}
            for key, s in [("overall", m)] + [(h, m & (hops == int(h[3:]))) for h in o["per_hop"]] + [("hop2_hop3_pooled", m & (hops >= 2))]:
                d14 = float(allv_rrf[s].mean() - ind["L1"][s].mean())
                d_ = float(iv[s].mean() - ind["SAFE"][s].mean())
                recov[key] = {"H2_RRF_minus_L1_pts": round(100 * d14, 1), "SAFE_H2_minus_SAFE_pts": round(100 * d_, 1), "fraction": (round(d_ / d14, 3) if abs(d14) > 1e-12 else None)}
            o["recovery_of_section14_gain"] = recov
    return o


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": nA, "status": "POSTHOC_MECHANISM_TEST",
       "beam_module_sha256": PINNED_SHA, "selector": {"cfg": CFG, "code": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select (unchanged)"},
       "SAFE_record": SAFE_RECORD[name], "identical_beam_check": {"MICRO_L3_H2_rebuilt": round(float(allv_rrf[m].mean()), 4), "gained": g14, "lost": l14},
       "H2_source": "own blocks of the beam's visited nodes in first-visit order (hop-1 beam then hop-2 beam, seeds excluded); spos'[b] = min(static structural rank, H2 block rank); no fourth channel",
       "H2_blocks_per_query_mean": round(float(np.mean([len(h2list[qi]) for qi in rowsA])), 1),
       "static_structural_blocks_per_query_mean": round(float(np.mean([len(SFs[qi]) for qi in rowsA])), 1),
       "retrieval_challenger_blocks_per_query_mean": round(float(np.mean([len(RFs[qi]) for qi in rowsA])), 1),
       "arms": {}}
for arm in ("L1", "SAFE", "SAFE_H2"):
    o = evaluate(arm)
    res["arms"][arm] = o
    G.log("%-8s %s" % (arm, json.dumps({k: o[k] for k in o if k not in ("per_hop", "recovery_of_section14_gain", "arm")})))
    if o["per_hop"]:
        G.log("         per hop %s" % json.dumps(o["per_hop"]))
    if arm == "SAFE_H2" and o.get("recovery_of_section14_gain"):
        G.log("         recovery %s" % json.dumps(o["recovery_of_section14_gain"]))

# ---------------------------------------------------------------- attribution: flips vs SAFE, and candidate-generator recall of the missing gold blocks
flips = {"newly_covered_vs_SAFE": {"n": 0, "gold_blocks_entering": {"H2_credited": 0, "static_structural": 0, "retrieval": 0, "boundary_reorder": 0}},
         "newly_lost_vs_SAFE": {"n": 0, "gold_blocks_evicted": 0, "displacing_blocks": {"H2_credited": 0, "static_or_retrieval": 0}}}
for qi in rowsA:
    c = ctxs[qi]
    fS, fH = finals["SAFE"][qi], finals["SAFE_H2"][qi]
    if ind["SAFE_H2"][qi] and not ind["SAFE"][qi]:
        flips["newly_covered_vs_SAFE"]["n"] += 1
        for b in goldp[qi] & (fH - fS):
            hp = h2pos[qi].get(b, 10 ** 9)
            if hp < c["spos"].get(b, 10 ** 9):
                flips["newly_covered_vs_SAFE"]["gold_blocks_entering"]["H2_credited"] += 1
            elif b in c["spos"]:
                flips["newly_covered_vs_SAFE"]["gold_blocks_entering"]["static_structural"] += 1
            elif b in c["rpos"]:
                flips["newly_covered_vs_SAFE"]["gold_blocks_entering"]["retrieval"] += 1
            else:
                flips["newly_covered_vs_SAFE"]["gold_blocks_entering"]["boundary_reorder"] += 1
    if ind["SAFE"][qi] and not ind["SAFE_H2"][qi]:
        flips["newly_lost_vs_SAFE"]["n"] += 1
        flips["newly_lost_vs_SAFE"]["gold_blocks_evicted"] += len(goldp[qi] & (fS - fH))
        for b in fH - fS:
            hp = h2pos[qi].get(b, 10 ** 9)
            if hp < c["spos"].get(b, 10 ** 9):
                flips["newly_lost_vs_SAFE"]["displacing_blocks"]["H2_credited"] += 1
            else:
                flips["newly_lost_vs_SAFE"]["displacing_blocks"]["static_or_retrieval"] += 1
res["attribution_flips"] = flips


def bucket(r):
    return "0-5" if r <= 5 else "6-24" if r <= 24 else "25-49" if r <= 49 else "50+"


recall = {}
for gname, s in ([("hop%d" % h, m & (hops == h)) for h in sorted(set(int(x) for x in hops[m] if x >= 0))] or [("all", m)]):
    o = {"n_queries_with_gold_outside_P50": 0, "worst_missing_gold_block": {"proposed_by_static_structural": 0, "proposed_by_retrieval": 0, "proposed_by_H2": 0,
                                                                            "proposed_by_none": 0, "H2_position": {"0-5": 0, "6-24": 0, "25-49": 0, "50+": 0}},
         "all_missing_gold_blocks_proposed_by_H2": 0, "static_structural_rank_of_worst_missing": {"0-5": 0, "6-24": 0, "25-49": 0, "50+": 0}}
    for qi in np.nonzero(s)[0]:
        c = ctxs[qi]
        miss = [b for b in goldp[qi] if b not in c["base50"]]
        if not miss:
            continue
        o["n_queries_with_gold_outside_P50"] += 1
        sset, rset = c["spos"], c["rpos"]
        # worst = the missing gold block that is hardest for the generators: largest best-available rank across the three lists
        def best_rank(b):
            return min(sset.get(b, 10 ** 9), rset.get(b, 10 ** 9), h2pos[qi].get(b, 10 ** 9))
        w = max(miss, key=lambda b: (best_rank(b), b))
        ws = o["worst_missing_gold_block"]
        ws["proposed_by_static_structural"] += int(w in sset)
        ws["proposed_by_retrieval"] += int(w in rset)
        ws["proposed_by_H2"] += int(w in h2pos[qi])
        ws["proposed_by_none"] += int(w not in sset and w not in rset and w not in h2pos[qi])
        if w in h2pos[qi]:
            ws["H2_position"][bucket(h2pos[qi][w])] += 1
        if w in sset:
            o["static_structural_rank_of_worst_missing"][bucket(sset[w])] += 1
        o["all_missing_gold_blocks_proposed_by_H2"] += int(all(b in h2pos[qi] for b in miss))
    recall[gname] = o
res["candidate_generator_recall"] = recall
G.log("flips %s" % json.dumps(flips))
G.log("recall %s" % json.dumps(recall))
G.S.wj(os.path.join(OUT, "h2safe_A_%s.json" % name), res)
G.log("done -> h2safe_A_%s.json" % name)
