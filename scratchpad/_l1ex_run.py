"""L1 EQUAL-EXPOSURE AUDIT -- STEPS 1-8.  Emits diag/exposure.json.

  python scratchpad/_l1ex_run.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ex_core as EX

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
METHODS = ["CANON", "SAFE", "SAFE_POOL", "U_PC5"]
FINE = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 12500, 15000,
        20000, 25000, 30000, 40000, 60000, 100000, 200000, 500000, 10 ** 7]
TARGETS = [0.30, 0.40, 0.50, 0.60]


def entry(E, cov, pr, nd, SL):
    n = E["n_nodes"]
    e = {"partitions": EX.stats(pr), "nodes": EX.stats(nd),
         "frac_partitions": round(float(pr.mean()) / E["n_parts"], 4),
         "EXPOSED_FRACTION": round(float(nd.mean()) / n, 4),
         "NODE_COMPRESSION_RATIO": (round(n / float(nd.mean()), 2) if nd.mean() > 0 else None)}
    for k, m in SL.items():
        e[k] = round(float(cov[m].mean()), 4)
    return e


def run_ds(ds, OUT):
    E = EX.build(ds, log)
    SL = EX.slices(E)
    row = {"corpus_nodes": E["n_nodes"], "corpus_partitions": E["n_parts"],
           "mean_partition_size": round(E["mean_part_size"], 1), "nq": E["nq"],
           "PART_BUDGET": {}, "NODE_BUDGET": {}, "FULL": {}, "FINE": {}}

    for m in METHODS:
        cov, pr, nd = EX.at_part_budget(E, m, 10 ** 9)
        row["FULL"][m] = entry(E, cov, pr, nd, SL)
        for k in EX.PART_BUDGETS:
            if m == "CANON" or (m == "U_PC5" and k >= 16):
                c2, p2, n2 = EX.at_part_budget(E, m, k if m == "CANON" else EX.P + k)
                row["PART_BUDGET"].setdefault(f"{m}@{k}", entry(E, c2, p2, n2, SL))
        if m in ("SAFE", "SAFE_POOL"):
            row["PART_BUDGET"][m] = row["FULL"][m]
        for b in EX.NODE_BUDGETS:
            c2, p2, n2 = EX.at_node_budget(E, m, b)
            row["NODE_BUDGET"].setdefault(str(b), {})[m] = entry(E, c2, p2, n2, SL)
        row["FINE"][m] = {}
        for b in FINE:
            c2, p2, n2 = EX.at_node_budget(E, m, b)
            row["FINE"][m][str(b)] = {"nodes_mean": round(float(n2.mean()), 1),
                                      **{k: round(float(c2[msk].mean()), 4)
                                         for k, msk in SL.items()}}
        f = row["FULL"][m]
        log(f"   {ds:15s} {m:10s} parts {f['partitions']['mean']:6.1f} "
            f"nodes {f['nodes']['mean']:9.1f}  exposed {f['EXPOSED_FRACTION']:7.2%}  "
            f"compression {f['NODE_COMPRESSION_RATIO']:7.2f}x  ALL {f['ALL']:.4f}"
            + (f"  hop3 {f['hop3']:.4f}" if "hop3" in f else ""))

    # ---- STEP 7 / 8: nodes required to reach a coverage target, by any existing method
    row["TARGETS"] = {}
    for k in SL:
        row["TARGETS"][k] = {}
        for t in TARGETS:
            best = None
            for m in METHODS:
                for b in FINE:
                    v = row["FINE"][m][str(b)]
                    if v[k] >= t and (best is None or v["nodes_mean"] < best[1]):
                        best = (m, v["nodes_mean"], v[k])
                        break
            row["TARGETS"][k][f"{t:.2f}"] = (
                None if best is None else
                {"method": best[0], "nodes_mean": best[1],
                 "EXPOSED_FRACTION": round(best[1] / E["n_nodes"], 4), "coverage": best[2]})

    # ---- STEP 5: Pareto frontier over (mean nodes exposed, ALL coverage)
    pts = []
    for src in ("PART_BUDGET", "FULL"):
        for name, e in row[src].items():
            q = {"config": name if src == "PART_BUDGET" else f"{name}@full",
                 "nodes": e["nodes"]["mean"], "ALL": e["ALL"],
                 "EXPOSED_FRACTION": e["EXPOSED_FRACTION"],
                 "NODE_COMPRESSION_RATIO": e["NODE_COMPRESSION_RATIO"]}
            if "hop3" in e:
                q["hop3"] = e["hop3"]
            pts.append(q)
    seen, uniq = set(), []
    for p in sorted(pts, key=lambda x: (x["nodes"], -x["ALL"])):
        if p["config"] in seen:
            continue
        seen.add(p["config"]); uniq.append(p)
    for p in uniq:
        p["dominated_by"] = [q["config"] for q in uniq
                             if q["config"] != p["config"] and q["ALL"] >= p["ALL"]
                             and q["nodes"] <= p["nodes"]
                             and (q["ALL"] > p["ALL"] or q["nodes"] < p["nodes"])]
        p["pareto"] = len(p["dominated_by"]) == 0
    row["PARETO"] = uniq
    OUT[ds] = row


def main():
    os.makedirs(f"{EX.EXD}/diag", exist_ok=True)
    fp = f"{EX.EXD}/diag/exposure.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in (sys.argv[1:] or EX.DSETS):
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote exposure.json")


if __name__ == "__main__":
    main()
