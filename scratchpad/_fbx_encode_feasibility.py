"""FREEBASE_SCALE stage 2b -- what the frozen IR_L1 (dense + SPLADE node vectors) would cost on 302M nodes.

A calculation from records only: measured host-GPU encoding speed (results/L3_HOST/ENC_REPRO__v1.json), the
frozen vector formats (results/... six datasets: dense fp16 1536-d, SPLADE CSR) and the host disk.  Nothing runs.

  python scratchpad/_fbx_encode_feasibility.py       (write-once: results/FREEBASE_SCALE/FBX_LOCALISATION_FEASIBILITY__v1.json)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
OUT = "results/FREEBASE_SCALE/FBX_LOCALISATION_FEASIBILITY__v1.json"
REPRO = "results/L3_HOST/ENC_REPRO__v1.json"
STATS = "results/FREEBASE_SCALE/FBX_GRAPH_STATS__v1.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    R = json.load(io.open(REPRO, encoding="utf-8"))
    N = json.load(io.open(STATS, encoding="utf-8"))["stats"]["nodes"]
    nd, nk = R["dense_docs"]["rows"], R["splade_docs"]["nnz_host"]
    ds, ss = R["dense_docs_seconds"], R["splade_docs_seconds"]
    dense_rate, splade_rate = nd / ds, nd / ss
    dense_bytes = N * 1536 * 2
    nnz_per_doc = nk / nd
    splade_nnz = N * nnz_per_doc
    free_gb = 386.8
    rec = {
        "stage": "FBX_SCALE / stage 2b (cost of the frozen node localisation on 302M nodes; a calculation, nothing run)",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "DEVELOPMENT (systems)",
        "inputs": {REPRO: sha(REPRO), STATS: sha(STATS)},
        "measured_on_host_gpu": {"device": R["env"]["device"], "priority": R["priority"], "dense_docs": nd, "dense_seconds": ds, "dense_docs_per_s": round(dense_rate, 1),
                                 "splade_seconds": ss, "splade_docs_per_s": round(splade_rate, 1),
                                 "note": "MetaQA documents (names); the GPU was shared and the job ran at BELOW_NORMAL; a Freebase name is of similar length but this is not verified"},
        "freebase_nodes": N,
        "dense_encode_hours": round(N / dense_rate / 3600, 1), "dense_encode_days": round(N / dense_rate / 86400, 1),
        "splade_encode_hours": round(N / splade_rate / 3600, 1),
        "dense_store_gb_fp16_1536d": round(dense_bytes / 1e9, 1),
        "splade_nonzeros_per_doc_measured": round(nnz_per_doc, 2), "splade_nonzeros_total_billion": round(splade_nnz / 1e9, 2),
        "splade_store_gb_at_6_bytes_per_nonzero": round(splade_nnz * 6 / 1e9, 1),
        "host_c_drive_free_gb": free_gb,
        "dense_store_over_free_disk": round(dense_bytes / 1e9 / free_gb, 2),
        "verdict": "FBX_IRL1_FULL_REPLAY_INFEASIBLE_ON_HOST",
        "basis": "the dense vectors alone need %.0f GB against %.0f GB free (%.1fx), and the dense encode is about %.1f days of a shared GPU; the frozen encoder, dimension and precision are part of the frozen substrate and were not changed" % (
            dense_bytes / 1e9, free_gb, dense_bytes / 1e9 / free_gb, N / dense_rate / 86400),
        "consequence": "the frozen IR_L1 cannot localise on the 302M-node graph; a query workload can only be replayed with the localisation the WebQSP corpus already has (its own frozen retrieval cache), mapped to Freebase positions through bridge/webqsp_positions.npy",
        "code": {"scratchpad/_fbx_encode_feasibility.py": sha("scratchpad/_fbx_encode_feasibility.py")},
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote", OUT, sha(OUT))
    print(json.dumps({k: rec[k] for k in rec if k.startswith(("dense_", "splade_", "verdict"))}, indent=1))


if __name__ == "__main__":
    main()
