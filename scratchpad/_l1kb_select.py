"""STEP 9/10 -- LEXICOGRAPHIC SELECTION, NESTED VALIDATION, LEAVE-ONE-DATASET-OUT SANITY.

Selection key (STEP 9), applied to ONE global config -- never a per-corpus choice:

  1 no statistically significant ALL regression on any corpus in scope
  2 maximize MetaQA hop3 dALL
  3 maximize MetaQA hop2 dALL
  4 maximize WebQSP dALL
  5 preserve the text gains: maximize the worst (text dALL - F6 text dALL)
  6 maximize worst-corpus dALL
  7 maximize macro dALL
  8 minimize gold-bearing evictions
  9 minimize churn

LODO: run exactly the same selection on five corpora and read off the held-out sixth.  A rule that
only survives because MetaQA was in the selection set is not universal, and this is what shows it.

  python scratchpad/_l1kb_select.py <scoreboard.json> [more.json ...]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kb_core as KB

TEXT = ["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
KBS = ["metaqa", "webqsp"]


def merge(paths):
    """union of every config measured in the given scoreboards (same name == same rule)."""
    SB = {}
    for p in paths:
        j = json.load(open(p))["SCOREBOARD"]
        for ds, d in j.items():
            SB.setdefault(ds, {"meta": d["meta"], "configs": {}})
            for k, v in d["configs"].items():
                SB[ds]["configs"][k] = v
    names = sorted(set.intersection(*[set(SB[d]["configs"]) for d in SB]))
    return SB, names


def key(SB, name, scope, ref="F6_B6"):
    d = {ds: SB[ds]["configs"][name] for ds in scope}
    r = {ds: SB[ds]["configs"][ref] for ds in scope}
    mh = d["metaqa"]["per_hop"] if "metaqa" in d else {}
    h3 = mh.get("3", {}).get("dALL", 0.0)
    h2 = mh.get("2", {}).get("dALL", 0.0)
    wq = d["webqsp"]["dALL"] if "webqsp" in d else 0.0
    txt = [ds for ds in scope if ds in TEXT]
    keep = min([d[ds]["dALL"] - r[ds]["dALL"] for ds in txt]) if txt else 0.0
    return (sum(1 for v in d.values() if v["sig"] and v["dALL"] < 0),
            -h3, -h2, -wq, -keep,
            -min(v["dALL"] for v in d.values()),
            -float(np.mean([v["dALL"] for v in d.values()])),
            sum(v["gold_evicted"] for v in d.values()),
            float(np.mean([v["churn_mean"] for v in d.values()])))


def summarize(SB, name, scope=None):
    scope = scope or list(SB)
    d = {ds: SB[ds]["configs"][name] for ds in scope}
    mh = d.get("metaqa", {}).get("per_hop", {})
    return {"config": name,
            "worst_dALL": round(min(v["dALL"] for v in d.values()), 4),
            "macro_dALL": round(float(np.mean([v["dALL"] for v in d.values()])), 4),
            "n_sig_regressions": sum(1 for v in d.values() if v["sig"] and v["dALL"] < 0),
            "metaqa_hop1": mh.get("1", {}).get("ALL"), "metaqa_hop2": mh.get("2", {}).get("ALL"),
            "metaqa_hop3": mh.get("3", {}).get("ALL"),
            "metaqa_hop2_d": mh.get("2", {}).get("dALL"),
            "metaqa_hop3_d": mh.get("3", {}).get("dALL"),
            "webqsp_dALL": d.get("webqsp", {}).get("dALL"),
            "webqsp_ANY": d.get("webqsp", {}).get("ANY"),
            "churn": round(float(np.mean([v["churn_mean"] for v in d.values()])), 3),
            "gold_evicted": sum(v["gold_evicted"] for v in d.values()),
            "gold_admitted": sum(v["gold_admitted"] for v in d.values()),
            "per_ds": {ds: d[ds]["dALL"] for ds in d},
            "per_ds_sig": {ds: d[ds]["sig"] for ds in d},
            "DISCOVERY": {ds: d[ds]["DISCOVERY"]["dALL"] for ds in d},
            "VALIDATION": {ds: d[ds]["VALIDATION"]["dALL"] for ds in d}}


def main(paths):
    SB, names = merge(paths)
    allds = list(SB)
    rank = sorted(names, key=lambda n: key(SB, n, allds))
    out = {"scoreboards": paths, "n_configs": len(names),
           "RANKING": [summarize(SB, n) for n in rank]}

    win = rank[0]
    print(f"=== lexicographic winner over all {len(allds)} corpora: {win}\n")
    hdr = f"{'config':26s} {'reg':>3s} {'mq_h3':>7s} {'mq_h2':>7s} {'webq':>7s} {'worst':>8s} " \
          f"{'macro':>8s} {'chn':>5s} {'ev':>5s}"
    print(hdr)
    for n in rank[:16]:
        s = summarize(SB, n)
        print(f"{n:26s} {s['n_sig_regressions']:3d} {s['metaqa_hop3_d']:+7.4f} "
              f"{s['metaqa_hop2_d']:+7.4f} {s['webqsp_dALL']:+7.4f} {s['worst_dALL']:+8.4f} "
              f"{s['macro_dALL']:+8.4f} {s['churn']:5.2f} {s['gold_evicted']:5d}")

    print("\n=== nested folds (winner) ===")
    w = summarize(SB, win)
    for ds in allds:
        c = SB[ds]["configs"][win]
        print(f"{ds:15s} FULL {c['dALL']:+.4f} (p={c['mcnemar_p']:.4f})  "
              f"DISCOVERY {c['DISCOVERY']['dALL']:+.4f} (p={c['DISCOVERY']['mcnemar_p']:.4f})  "
              f"VALIDATION {c['VALIDATION']['dALL']:+.4f} (p={c['VALIDATION']['mcnemar_p']:.4f})")

    print("\n=== LODO: select on five corpora, read off the held-out sixth ===")
    lodo = {}
    for held in allds:
        sc = [d for d in allds if d != held]
        pick = min(names, key=lambda n: key(SB, n, sc))
        h = SB[held]["configs"][pick]
        lodo[held] = {"selected_on_other_five": pick, "held_out_dALL": h["dALL"],
                      "held_out_sig": h["sig"], "held_out_p": h["mcnemar_p"],
                      "held_out_net": h["net"],
                      "same_as_global_winner": pick == win}
        print(f"held-out {held:15s} -> pick {pick:26s} held-out dALL {h['dALL']:+.4f} "
              f"(net {h['net']:+d}, p={h['mcnemar_p']:.4f}){'  [= global]' if pick == win else ''}")
    out["WINNER"] = win
    out["WINNER_SUMMARY"] = w
    out["LODO"] = lodo
    fp = f"{KB.KBD}/selection.json"
    json.dump(out, open(fp, "w"), indent=1)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main(sys.argv[1:] or [f"{KB.KBD}/scoreboard_round1.json"])
