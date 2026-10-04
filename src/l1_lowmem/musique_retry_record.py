"""Records the single, isolated Mt-KaHyPar retry for musique (spec step 7) and its classification.

    python -u src/l1_lowmem/musique_retry_record.py   -> results/L1_LOWMEM/MUSIQUE_MTKAHYPAR_RETRY.json
The canonical partition.py rewrote data/l1_canonical/musique/parts/H4_SK.FAILED.json in place (its own behaviour for run
outcomes); the previous attempt's numbers are preserved here from the record as it stood before the retry.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, wj, rj, pin, log  # noqa: E402

FAILED = os.path.join(REPO, "data", "l1_canonical", "musique", "parts", "H4_SK.FAILED.json")


def main():
    cur = rj(FAILED)
    rec = {
        "RECORD": "MUSIQUE_MTKAHYPAR_RETRY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ruling": "exactly one isolated Mt-KaHyPar retry after host contention ended; if it fails again on memory -> RESOURCE_INFEASIBLE_LOCAL, no further retries",
        "previous_attempt": {"utc": "2026-09-13T13:56:05Z", "threads": 8, "rss_cap_gb": 6.0, "host_available_gb_at_start": 7.71, "peak_rss_kb": 6002284,
                             "killed": True, "rc": 137, "wall_seconds": 801.2, "STATUS": "FAILED_MEMORY_CAP",
                             "source": "H4_SK.FAILED.json as it stood before the retry (rewritten in place by partition.py); preserved verbatim as "
                                       "results/L1_LOWMEM/FREIGHT_RUNS_musique.json baseline_mtkahypar.record"},
        "retry": {"command": "python -u src/l1_canonical/partition.py musique --threads 4 --cap-gb 6.0",
                  "why_4_threads": "DETERMINISTIC_QUALITY output is thread-count independent (contract note; local == Modal bit-identical), and flow "
                                   "refinement memory scales with threads -> the retry used the lowest-memory configuration of the SAME frozen recipe "
                                   "(preset, objective, eps, seed, weights, k rule unchanged)",
                  "why_cap_6_0": "HARD_CAP_GB of partition.py: the WSL VM ceiling is 7.6 GB and .wslconfig must not be modified; 6.0 GB leaves the VM's OS its headroom",
                  "isolation": "no FREIGHT run, no other partitioner, nothing else inside the WSL VM; the cap is on the worker's own RSS inside the VM. "
                               "A foreign host process (scripts/m3b_run.py --stage screen, ~9.7 GB private, started 20:29 local) was running on the Windows "
                               "host and was left untouched -- it does not share the VM's memory budget, so the kill is attributable to the worker's own "
                               "footprint exceeding the cap, not to host contention",
                  "record": pin(FAILED), "STATUS": cur and cur.get("STATUS"), "guard": cur and cur.get("guard"), "threads": cur and cur["contract"]["threads"],
                  "wall_seconds": cur and cur["guard"]["wall_seconds"], "peak_rss_kb_at_kill": cur and cur["guard"]["peak_rss_kb"]},
        "CLASSIFICATION": "RESOURCE_INFEASIBLE_LOCAL",
        "consequence": "the musique Mt-KaHyPar baseline column reads RESOURCE_INFEASIBLE_LOCAL; the FREIGHT arm for musique is reported alone "
                       "(BASELINE_ABSENT downstream); the canonical musique H4_SK partition still moves to the external lane or to a new explicit ruling; "
                       "the algorithm is not swapped",
    }
    wj(os.path.join(OUT, "MUSIQUE_MTKAHYPAR_RETRY.json"), rec)
    log("musique retry:", rec["retry"]["STATUS"], "->", rec["CLASSIFICATION"])


if __name__ == "__main__":
    main()
