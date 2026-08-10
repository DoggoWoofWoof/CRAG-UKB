import json, glob, os
results = []
for f in sorted(glob.glob("results/hnm_ablation/comparison_*_hnm.json")):
    ds = os.path.basename(f).split("_")[1]
    data = json.load(open(f))
    best_r1, best_key = 0, ""
    for k, v in data.items():
        if "kl_div" in k and "test" in v:
            r1 = v["test"].get("recall@1", 0)
            if r1 > best_r1:
                best_r1, best_key = r1, k
    if best_key:
        t = data[best_key]["test"]
        hnm_k = best_key.split("_hnm_")[1]
        results.append(f"{ds}|{best_key}|{hnm_k}|{t['recall@1']}|{t['recall@20']}|{t['mrr']}")

with open("_best_hnm.txt", "w") as f:
    f.write("\n".join(results))
print("DONE")
