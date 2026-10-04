"""PHASE 18 -- Modal CPU cost.  Parses the runner logs; no cost is estimated by hand.

CPU-seconds are wall_seconds x the container CPU request, which is what Modal bills.  No GPU
was requested anywhere in this program -- that is asserted against the runner source, not
merely stated.
"""
import os, re, sys, json, glob
OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_UNIVERSAL_PARTITION_SEARCH"
LOGS = "scratchpad/_l1ov"
RUNNERS = ("scratchpad/modal_partition.py", "scratchpad/modal_partition_hi.py")
ROW = re.compile(r'\{"ds": "(?P<ds>[a-z0-9_]+)", "method": "(?P<method>[A-Z0-9_]+)", '
                 r'"graph": "(?P<graph>[A-Z0-9_]+)", "status": "(?P<status>[A-Z]+)", '
                 r'"wall_seconds": (?P<wall>[0-9.]+), "peak_rss_mb": (?P<rss>[0-9.]+)')


def main():
    probe = json.load(open(f"{OUT}/modal/PROBE.json"))
    cpus = float(probe.get("cpu_request_partition") or 16.0)
    seen, rows = set(), []
    for lf in sorted(glob.glob(f"{LOGS}/log_mp_*.txt")):
        for m in ROW.finditer(open(lf, encoding="utf-8", errors="replace").read()):
            d = m.groupdict()
            key = (d["ds"], d["method"], d["graph"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"ds": d["ds"], "method": d["method"], "graph": d["graph"],
                         "status": d["status"], "wall_seconds": float(d["wall"]),
                         "peak_rss_mb": float(d["rss"]),
                         "cpu_seconds": round(float(d["wall"]) * cpus, 1), "log": os.path.basename(lf)})
    # both runners are audited: the _hi variant differs only in the memory/timeout
    # request HotpotQA needed, and must be just as GPU-free as the original.
    src = "".join(open(f, encoding="utf-8").read() for f in RUNNERS
                  if os.path.exists(f))
    by_m = {}
    for r in rows:
        b = by_m.setdefault(r["method"], {"cells": 0, "wall_seconds": 0.0, "cpu_seconds": 0.0,
                                          "max_peak_rss_mb": 0.0, "max_wall_seconds": 0.0})
        b["cells"] += 1
        b["wall_seconds"] = round(b["wall_seconds"] + r["wall_seconds"], 1)
        b["cpu_seconds"] = round(b["cpu_seconds"] + r["cpu_seconds"], 1)
        b["max_peak_rss_mb"] = max(b["max_peak_rss_mb"], r["peak_rss_mb"])
        b["max_wall_seconds"] = max(b["max_wall_seconds"], r["wall_seconds"])
    tot_cpu = round(sum(r["cpu_seconds"] for r in rows), 1)
    res = {"NO_GPU_ANYWHERE": ("gpu=" not in src) and ("gpu =" not in src),
           "GPU_TOKENS_IN_RUNNER": src.count("gpu="),
           "RUNNERS_AUDITED": [f for f in RUNNERS if os.path.exists(f)],
           "CPU_REQUEST": cpus, "MEMORY_REQUEST_MB": probe.get("memory_request_mb"),
           "CELLS": len(rows), "TOTAL_WALL_SECONDS": round(sum(r["wall_seconds"] for r in rows), 1),
           "TOTAL_CPU_SECONDS": tot_cpu, "TOTAL_CPU_HOURS": round(tot_cpu / 3600.0, 2),
           "BY_METHOD": by_m, "PER_CELL": sorted(rows, key=lambda r: -r["wall_seconds"]),
           "SINGLE_THREADED_NOTE": ("KaHIP kaffpa is single-threaded but the container still requests 16 CPUs, so the "
                                    "H1/H2 cpu_seconds above are what Modal bills, not work done; their "
                                    "useful compute is closer to wall_seconds x 1."),
           "ABORTED_NOT_COUNTED": ("cells that were launched and stopped before returning contribute no row here, "
                                   "so TOTAL_CPU_HOURS understates spend; see "
                                   "hard_partitions/IN_FLIGHT.json STOPPED_BY_USER."),
           "IMAGE_BUILD": "KaHIP compiled once into a cached image layer; never rebuilt per corpus",
           "NOTE": ("wall x cpu_request is what Modal bills; the one-off image build that compiles "
                    "KaHIP is not in these per-cell numbers and is amortised over every cell")}
    os.makedirs(f"{OUT}/modal", exist_ok=True)
    json.dump(res, open(f"{OUT}/modal/COST.json", "w"), indent=1)
    print(f"cells {res['CELLS']}  wall {res['TOTAL_WALL_SECONDS']}s  "
          f"cpu-hours {res['TOTAL_CPU_HOURS']}  NO_GPU={res['NO_GPU_ANYWHERE']}")
    for m, b in sorted(by_m.items()):
        print(f"  {m:32s} cells {b['cells']:2d}  wall {b['wall_seconds']:8.1f}s  "
              f"slowest {b['max_wall_seconds']:7.1f}s  peak RSS {b['max_peak_rss_mb']:9.1f} MB")
    return res


if __name__ == "__main__":
    main()
