"""Modal build cost for the hypergraph phase, and the no-GPU audit.

Reads the runner source (so the GPU claim is checked against the file, not remembered) and the
per-job JSON lines the matrix entrypoint printed.  Also records the mid-phase interruption: the
workspace the first large fan-out ran in exhausted its monthly grant and was disabled, which
killed 17 of 18 in-flight jobs.  That is a cost fact, so it is recorded rather than quietly
re-run.

  python scratchpad/_l1hu_cost.py
"""
import os, sys, json, re, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_HYPERGRAPH_UNIVERSAL"
LOGS = {"scratchpad/_l1hu/modal_sm.log": ("swathihrao28", 8, 16),
        "scratchpad/_l1hu/modal_lg.log": ("swathihrao28", 16, 96),
        "scratchpad/_l1hu/modal_hot.log": ("spanishorgay", 16, 64),
        "scratchpad/_l1hu/modal_web.log": ("pes1ug23cs623", 16, 64)}


def main():
    src = open("scratchpad/modal_hyper.py", encoding="utf-8").read()
    rec = {"MODAL_NO_GPU_ANYWHERE": "gpu=" not in src,
           "gpu_mentions": [l.strip() for l in src.splitlines() if "gpu=" in l],
           "PARTITIONER_RUNS_IN_CHILD_PROCESS": "subprocess.run" in src,
           "accounts": {}, "jobs": [], "INTERRUPTION": {
               "what": "workspace disabled mid fan-out (monthly grant exhausted)",
               "account": "swathihrao28",
               "error": "ConflictError: workspace ac-P23AaSSfe5DXQnmoxFs4Y1 is disabled",
               "jobs_lost": 17, "jobs_saved": 1,
               "response": ("re-ran the lost cells on two fresh pool accounts, one corpus each, "
                            "with a lean image (no from-source KaHIP compile, unused here) and "
                            "the large memory class cut from 96 GB to 64 GB against a measured "
                            "34.1 GB peak")}}
    tot_cpu_h = tot_gb_h = 0.0
    for lg, (acct, cpu, gb) in LOGS.items():
        if not os.path.exists(lg):
            continue
        n = ok = 0
        ch = gh = 0.0
        for line in open(lg, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line.startswith('{"ds"'):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            n += 1
            ok += int(d.get("status") == "OK")
            w = float(d.get("wall_seconds") or 0)
            ch += cpu * w / 3600.0
            gh += gb * w / 3600.0
            rec["jobs"].append({"account": acct, "ds": d["ds"], "tag": d["tag"],
                                "status": d.get("status"), "wall_seconds": w,
                                "peak_rss_mb": d.get("peak_rss_mb"), "cpu": cpu, "mem_gb": gb})
        rec["accounts"][lg] = {"account": acct, "cpu": cpu, "mem_gb": gb, "jobs": n, "ok": ok,
                               "cpu_hours": round(ch, 2), "gb_hours": round(gh, 1)}
        tot_cpu_h += ch
        tot_gb_h += gh
    peaks = [j["peak_rss_mb"] for j in rec["jobs"] if j.get("peak_rss_mb")]
    rec["MODAL_TOTAL_CPU_HOURS"] = round(tot_cpu_h, 2)
    rec["MODAL_TOTAL_GB_HOURS"] = round(tot_gb_h, 1)
    rec["MODAL_JOBS"] = len(rec["jobs"])
    rec["PEAK_RSS_MB_MAX"] = max(peaks) if peaks else None
    rec["ACCOUNTS_USED"] = sorted({j["account"] for j in rec["jobs"]})
    os.makedirs("%s/hypergraph_build" % OUT, exist_ok=True)
    fp = "%s/hypergraph_build/MODAL_COST.json" % OUT
    json.dump(rec, open(fp, "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("MODAL_NO_GPU_ANYWHERE", "MODAL_JOBS",
                                          "MODAL_TOTAL_CPU_HOURS", "MODAL_TOTAL_GB_HOURS",
                                          "PEAK_RSS_MB_MAX", "ACCOUNTS_USED")}, indent=1))
    print("wrote", fp)
    return rec


if __name__ == "__main__":
    main()
