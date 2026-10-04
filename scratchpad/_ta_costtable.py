"""Regenerate the cross-corpus cost table for the audit report from the sweep JSONs."""
import json

DS = [("metaqa", "MetaQA"), ("2wiki_clean", "2wiki"), ("musique_clean", "MuSiQue"), ("squad_clean", "SQuAD")]
CFG = ["A1a_M32", "A2_M32", "A1a_M256", "A2_M256"]


def tot(v):
    if "dense" in v:
        d, s = v["dense"], v["splade"]
        return (d["edges_scanned"] + s["edges_scanned"],
                d["structural_nodes"] + s["structural_nodes"],
                round(d["sec"] + s["sec"] + v.get("vote_sec", 0), 1))
    return v.get("edges_scanned", 0), v.get("structural_nodes", 0), round(v.get("sec", 0) + v.get("vote_sec", 0), 1)


rows = ["| corpus | cfg | edges scanned | structural nodes | seconds | × BASE | ALL@P50 net |",
        "|---|---|--:|--:|--:|--:|--:|"]
for ds, name in DS:
    j = json.load(open(f"results/GENERALIZATION/_g2_ta_{ds}.json"))
    C, R = j["COST"], j["CONFIGS"]
    b = C["BASE"]["sec"]
    rows.append(f"| **{name}** (n={j['n_dev_queries']}) | BASE | 0 | 0 | {b:.1f} | 1.0 | — |")
    for k in CFG:
        if k not in C:
            continue
        e, n, sc = tot(C[k])
        net = R[k]["paired_vs_BASE_ALL"]["net"]
        rows.append(f"| | {k} | {e:,} | {n:,} | {sc:.1f} | {sc/b:.1f} | {net:+d} |")
print("\n".join(rows))
