"""Assemble L2_CONTROLLER_RESULTS.json from saved C*.json + _analysis.json. VAL only."""
import json, os
OUT = "results/L2/_ctrl"
COLS = ["MRR", "NDCG@50", "R@1", "R@5", "R@10", "R@20", "R@50", "ANY@10", "ANY@20", "ANY@50", "ALL@10", "ALL@20", "ALL@50"]
ROWS = ["C0", "C0_norel", "C1", "C2", "C3", "C4", "C5"]


def traj(j):
    """Trained-epoch trajectory summary (excludes init epoch -1): what training actually did."""
    h = j.get("history")
    if not h:
        return None
    tr = [e for e in h if e.get("epoch", -1) >= 0 and e.get("val_pooled")]
    if not tr:
        return None
    nd = [(e["epoch"], e["val_pooled"]["NDCG@50"], e["val_pooled"]["ALL@50"]) for e in tr]
    best = max(nd, key=lambda x: x[1]); worst = min(nd, key=lambda x: x[1])
    return {"init_ndcg": h[0]["val_pooled"]["NDCG@50"],
            "trained_best_ndcg": round(best[1], 4), "trained_best_epoch": best[0],
            "trained_worst_ndcg": round(worst[1], 4), "trained_worst_epoch": worst[0],
            "final_epoch_ndcg": round(nd[-1][1], 4), "final_epoch_all50": round(nd[-1][2], 4),
            "selected": "init(=C0)" if best[1] < h[0]["val_pooled"]["NDCG@50"] + 1e-9 else f"epoch{best[0]}"}


def load(tag):
    p = f"{OUT}/{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    tbl = {}
    for tag in ROWS:
        j = load(tag)
        if j is None:
            tbl[tag] = None; continue
        row = {"POOLED": {c: j["POOLED"].get(c) for c in COLS}}
        for ds in ("2wiki_clean", "musique_clean"):
            row[ds] = {c: j[ds].get(c) for c in COLS}
        if "gate" in j: row["gate"] = j["gate"]
        if "train_meta" in j: row["train_meta"] = j["train_meta"]
        t = traj(j)
        if t: row["trained_trajectory"] = t
        tbl[tag] = row
    analysis = load("_analysis") if os.path.exists(f"{OUT}/_analysis.json") else None
    out = {"level": "VAL_ONLY", "primary_metric": "NDCG@50",
           "selection_rule": "primary NDCG@50, tie-break ALL@10 then ALL@50 (declared pre-training); init-inclusive (C0 floor)",
           "loss": "multi-positive pairwise RankNet (every gold-neg pair equal weight)",
           "table": tbl, "analysis": analysis}
    json.dump(out, open("results/L2/L2_CONTROLLER_RESULTS.json", "w"), indent=1, default=str)
    # console preview
    print(f"{'row':10s} {'sel-NDCG':>8s} {'MRR':>7s} {'ALL@50':>7s}  {'trainedBest':>11s} {'trainedWorst':>12s}  sel")
    for tag in ROWS:
        r = tbl[tag]
        if r is None: print(f"{tag:10s}  (missing)"); continue
        p = r["POOLED"]; t = r.get("trained_trajectory")
        tb = f"{t['trained_best_ndcg']:.4f}@e{t['trained_best_epoch']}" if t else "  --  "
        tw = f"{t['trained_worst_ndcg']:.4f}@e{t['trained_worst_epoch']}" if t else "  --  "
        sel = t["selected"] if t else "-"
        print(f"{tag:10s} {p['NDCG@50']:8.4f} {p['MRR']:7.4f} {p['ALL@50']:7.4f}  {tb:>11s} {tw:>12s}  {sel}")
    print("wrote results/L2/L2_CONTROLLER_RESULTS.json")


if __name__ == "__main__":
    main()
