"""CORRECTED SELECTION -- STEP 9 key with criterion 1 enforced against BOTH baselines, + LODO.

  1  no statistically significant ALL regression on any corpus, vs BASE *and* vs the safe
     universal baseline B6_S4_F6_Ms64_Mr32   (the per-round scoreboards only tested vs BASE,
     which lets a rule that gives up a trusted gain look clean)
  2  maximize MetaQA hop3 dALL
  3  maximize MetaQA hop2 dALL
  4  maximize WebQSP dALL
  5  preserve the text gains: maximize the worst text (dALL - F6 dALL)
  6  maximize worst-corpus dALL
  7  maximize macro dALL
  8  minimize gold-bearing evictions
  9  minimize churn

  python scratchpad/_l1kb_select2.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kb_core as KB

TEXT = ["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
SB = json.load(open(f"{KB.KBD}/scoreboard_consolidated.json"))["SCOREBOARD"]
NAMES = sorted(set.intersection(*[set(SB[d]["configs"]) for d in SB]))


def key(name, scope):
    d = {ds: SB[ds]["configs"][name] for ds in scope}
    mh = d["metaqa"]["per_hop"] if "metaqa" in d else {}
    txt = [ds for ds in scope if ds in TEXT]
    return (sum(1 for v in d.values() if (v["sig"] and v["dALL"] < 0)
                or (v["vs_F6"]["sig"] and v["dALL_vs_F6"] < 0)),
            -mh.get("3", {}).get("dALL", 0.0), -mh.get("2", {}).get("dALL", 0.0),
            -(d["webqsp"]["dALL"] if "webqsp" in d else 0.0),
            -(min(d[ds]["dALL_vs_F6"] for ds in txt) if txt else 0.0),
            -min(v["dALL"] for v in d.values()),
            -float(np.mean([v["dALL"] for v in d.values()])),
            sum(v["gold_evicted"] for v in d.values()),
            float(np.mean([v["churn_mean"] for v in d.values()])))


def detail(name):
    print(f"\n=== {name} ===")
    print("%-15s %8s %8s %9s %9s %8s %8s %6s" % ("corpus", "ALL", "dBASE", "p(BASE)",
                                                 "dF6", "p(F6)", "churn", "abst"))
    for ds in KB.DSETS:
        c = SB[ds]["configs"][name]
        print("%-15s %8.4f %+8.4f %9.4f %+9.4f %8.4f %8.3f %6.3f" % (
            ds, c["ALL"], c["dALL"], c["mcnemar_p"], c["dALL_vs_F6"],
            c["vs_F6"]["mcnemar_p"], c["churn_mean"], c["abstain_frac"]))
    c = SB["metaqa"]["configs"][name]
    b = SB["metaqa"]["configs"]["BASE_B6"]
    print("  MetaQA per hop:")
    for h in ("1", "2", "3"):
        ph = c["per_hop"][h]; pb = b["per_hop"][h]
        print("    hop%s n=%d BASE %.4f -> %.4f  dBASE %+.4f (p %.4f)  dF6 %+.4f (p %.4f)" % (
            h, ph["n"], pb["ALL"], ph["ALL"], ph["dALL"], ph["mcnemar_p"],
            ph["dALL_vs_F6"], ph["vs_F6"]["mcnemar_p"]))
    print("  folds: " + "  ".join(
        "%s D %+.4f(p%.3f) V %+.4f(p%.3f)" % (ds[:7], SB[ds]["configs"][name]["DISCOVERY"]["dALL"],
                                              SB[ds]["configs"][name]["DISCOVERY"]["mcnemar_p"],
                                              SB[ds]["configs"][name]["VALIDATION"]["dALL"],
                                              SB[ds]["configs"][name]["VALIDATION"]["mcnemar_p"])
        for ds in ("metaqa", "webqsp")))
    print("  churn hist metaqa: %s" % c["churn_hist"])


def main():
    allds = list(SB)
    rank = sorted(NAMES, key=lambda n: key(n, allds))
    win = rank[0]
    print("=== CORRECTED lexicographic winner over all 6 corpora: %s ===" % win)
    for n in rank[:6]:
        k = key(n, allds)
        print("  %-24s reg %d  h3 %+.4f  h2 %+.4f  wq %+.4f  keepTXT %+.4f  macro %+.4f"
              % (n, k[0], -k[1], -k[2], -k[3], -k[4], -k[6]))
    for n in (win, "ALT_STRLEAD_SEED_B6", "F6_B6", "CANON_DEEP_B6"):
        detail(n)

    print("\n=== LODO under the corrected criterion ===")
    lodo = {}
    for held in allds:
        sc = [d for d in allds if d != held]
        pick = min(NAMES, key=lambda n: key(n, sc))
        h = SB[held]["configs"][pick]
        lodo[held] = {"pick": pick, "held_dALL": h["dALL"], "held_p": h["mcnemar_p"],
                      "held_dALL_vs_F6": h["dALL_vs_F6"],
                      "held_p_vs_F6": h["vs_F6"]["mcnemar_p"], "same_as_global": pick == win}
        print("held-out %-15s -> %-22s  dBASE %+.4f (p%.4f)  dF6 %+.4f (p%.4f)%s" % (
            held, pick, h["dALL"], h["mcnemar_p"], h["dALL_vs_F6"],
            h["vs_F6"]["mcnemar_p"], "  [= global]" if pick == win else ""))
    json.dump({"WINNER": win, "RANKING": rank[:30], "LODO": lodo},
              open(f"{KB.KBD}/selection_corrected.json", "w"), indent=1)
    print("\nwrote %s/selection_corrected.json" % KB.KBD)


if __name__ == "__main__":
    main()
