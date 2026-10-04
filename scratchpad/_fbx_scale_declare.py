"""Write the FREEBASE_SCALE compute-stage declaration (write-once); later stages are dated addenda, never edits.

Usage: python scratchpad/_fbx_scale_declare.py            (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
OUT = "results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1.json"
CODE = ["scratchpad/_fbx_stats.py", "scratchpad/_host_yield.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    rec = {
        "stage": "FBX_SCALE",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S+05:30", time.localtime()),
        "what": "the FREEBASE_SCALE systems-validation compute on the shared host 'gpu': the read-only tree data/final_canonical/freebase (302M nodes, 2.06B edges) verified byte for byte by FBX_HOST_VERIFY__v1.json",
        "role": "a large, source-derived, query-independent scale substrate; NOT a seventh benchmark dataset, NOT a WebQSP replacement, NOT a development set",
        "user_direction (verbatim, chat, 2026-09-30)": [
            "Yes — I’d finish FREEBASE_SCALE first now.",
            "run the intended large-K partitioning so we finally test the thousands-of-shards regime;",
            "record partition balance, cut/replication statistics, build time, peak RAM/disk, and partition sizes;",
            "run the existing non-learning L1 router against that partition map if the pinned query workload can be replayed;",
            "report absolute fan-out B_P and especially fraction of shards contacted B_P/K;",
            "measure routing/local retrieval latency and bytes/nodes touched;",
            "do no new mechanism tuning on Freebase — this lane should validate scalability, not become another development set.",
            "I would keep the current L1 candidate fixed for that run: IR_L1 node localization -> ES partition routing -> KNEE_SDIV and use Freebase to answer scale, not 'which router scores best.'"
        ],
        "standing_rulings_in_force": [
            "I would not change the partitioning algorithm just because the current machine has 16 GB RAM... We should not silently swap Mt-KaHyPar for something cheaper.",
            "PHG contract: NP = 4 frozen; if 4 ranks no longer fit, record the resource failure and get a separate explicit rank-count ruling (2026-09-14)",
            "the six datasets' substrate is closed (DATA_FREEZE_VFINAL); nothing under data/final_canonical is written",
            "gpu should be free if you need; you have the second highest priority (mpr first) -- every job runs under scratchpad/_host_yield.py; foreign processes are never touched",
            "do not read the held-out rows; do not spend the held-out evaluation"
        ],
        "stage_1_FBX_STATS": {
            "job": "scratchpad/_fbx_stats.py run <workers>  (read-only mmap of the served CSR; streaming chunks; writes only results/FREEBASE_SCALE/FBX_GRAPH_STATS__v1.json)",
            "answers": ["graph size, self loops, unique undirected pairs (the STRUCT key set), multiplicity",
                        "degree distributions and hubs", "exact pin count of the STRUCT-only H4_SPLIT_PRESERVE hypergraph (pins = anchors + 2F)",
                        "oversize hyperedges the split rule must chop, for K in the grid", "relation histogram / type-mirror share"],
            "not": "no partitioning, no embedding, no routing, no queries; a measurement of the input the frozen recipe would consume",
            "checked_before_launch": "vectorised chunk result == brute-force Python sets on nodes [0, 3000): undirected degree sum 23,675 identical, top hubs identical (2026-09-30)"
        },
        "next_stages (declared as intent; each needs its own dated addendum before it runs)": [
            "stage 2: partition feasibility of the frozen PHG contract at this pin count (memory model from the 40 measured cells; no attempt that could exhaust the shared WSL VM)",
            "stage 3: a partition map at thousands of blocks, ONLY under the contract the user rules for (see stage-2 result)",
            "stage 4: shard-routing measurements on the WebQSP split-A population, with whichever localisation the substrate supports (the frozen IR_L1 needs dense+SPLADE vectors of all 302M nodes, which do not exist)"
        ],
        "refusals_kept": ["TEST never read; split B stays sealed", "no encoder training, no LLM", "no new router or mechanism tuned on Freebase",
                          "corpus subsetting by query is forbidden (a scale measurement uses the whole graph)", "nothing under data/final_canonical is written; no credential on the host",
                          "no attempt that could exhaust the shared host's WSL VM or another user's job"],
        "priority_policy": {"wrapper": "python -u scratchpad/_host_yield.py run -- <cmd>", "scheduler_reserve_untouched": "2 CPU / 8 GB"},
        "code": {p: sha(p) for p in CODE},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
