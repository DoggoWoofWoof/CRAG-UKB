"""FBX_SCALE Track C / encoding contract -- what the canonical Qwen encoder costs on Freebase node names (a measurement for the contract, nothing scientific is decided here).

  python -u scratchpad/_fbx_enc_probe.py SAMPLE     # env crag-ner (pyarrow), CPU: a random sample of DISTINCT names (first occurrences) from 40 random row-group units -> data/freebase_scale/enc/probe_sample.json (write-once)
  python -u scratchpad/_fbx_enc_probe.py ENCODE     # env mpr-cu128, GPU: token-length distribution, docs/s at several batch sizes, batch-size numerics -> results/FREEBASE_SCALE/enc/FBX_ENC_PROBE__v1.json (write-once)

The encoder is scratchpad/_enc_host.py's (the canonical C5 setup, placement-tested BIT-for-BIT-close on MetaQA: results/L3_HOST/ENC_REPRO__v1.json): gte-Qwen2-1.5B-instruct rev a9af15a6..., documents plain,
fp16, normalised, use_cache off.  sentence-transformers sorts every encode() call by character length, so the batches are consecutive in length order.
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TREE = os.path.join(REPO, "data", "final_canonical", "freebase")
NAMES = os.path.join(REPO, "data", "freebase_scale", "names")
WORK = os.path.join(REPO, "data", "freebase_scale", "enc")
RECS = os.path.join(REPO, "results", "FREEBASE_SCALE", "enc")
SAMPLE = os.path.join(WORK, "probe_sample.json")
OUT = os.path.join(RECS, "FBX_ENC_PROBE__v1.json")
N_UNITS, PER_UNIT = 40, 400
CAP = 512                    # candidate max_seq_length for the contract (tokens); the probe reports how many sampled names exceed it
THROUGHPUT_N = 6144
BATCHES = (32, 64, 128, 256)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wj(path, obj):
    assert not os.path.exists(path), "write-once: %s exists" % path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=True))
    os.replace(path + ".tmp", path)


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def cmd_sample():
    import pyarrow.parquet as pq
    assert not os.path.exists(SAMPLE), "write-once"
    ptr = np.load(os.path.join(NAMES, "ptr.npy"), mmap_mode="r")
    fpos = np.load(os.path.join(NAMES, "first_pos.npy"), mmap_mode="r")
    units = []
    for f in sorted(x for x in os.listdir(os.path.join(TREE, "nodes")) if x.startswith("shard_") and x.endswith(".parquet")):
        md = pq.ParquetFile(os.path.join(TREE, "nodes", f)).metadata
        units += [(int(f[6:11]), g) for g in range(md.num_row_groups)]
    rng = np.random.RandomState(0)
    pick = sorted(rng.choice(len(units), size=N_UNITS, replace=False).tolist())
    rows = []
    for i in pick:
        s, g = units[i]
        tb = pq.ParquetFile(os.path.join(TREE, "nodes", "shard_%05d.parquet" % s)).read_row_group(g, columns=["position", "name", "name_kind"])
        pos = tb.column("position").to_numpy()
        first = np.asarray(fpos[np.asarray(ptr[pos])]) == pos           # the first occurrence of its name: exactly the positions that are encoded
        idx = np.flatnonzero(first)
        idx = np.sort(rng.choice(idx, size=min(PER_UNIT, len(idx)), replace=False))
        nm, nk = tb.column("name").to_pylist(), tb.column("name_kind").to_numpy()
        rows += [{"pos": int(pos[j]), "name": nm[j] or "", "kind": int(nk[j])} for j in idx]
        log("unit %d (shard %d rg %d): %d first-occurrence positions, sampled %d" % (i, s, g, len(idx), min(PER_UNIT, len(idx))))
    os.makedirs(WORK, exist_ok=True)
    wj(SAMPLE, {"seed": 0, "units": pick, "rows": rows})
    log("sample: %d names, %s" % (len(rows), sha_file(SAMPLE)[:16]))


def wait_gpu(need_gib):
    import torch
    t0 = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        if free >= need_gib:
            return free
        assert time.time() - t0 < 4 * 3600, "GPU never had %.0f GiB free" % need_gib
        log("waiting for GPU memory (%.1f GiB free, need %.0f)" % (free, need_gib))
        time.sleep(60)


def pct(a, qs=(50, 90, 99, 99.9, 99.99)):
    return {("p%s" % q): float(np.percentile(a, q)) for q in qs} | {"max": float(np.max(a)), "mean": float(np.mean(a))}


def cmd_encode():
    import _enc_host as EH
    import torch
    global OUT
    k = 1
    while os.path.exists(OUT):                        # every measurement is kept: the GPU is shared, so a later attempt can be cleaner
        k += 1
        OUT = os.path.join(RECS, "FBX_ENC_PROBE__v1_attempt%d.json" % k)
    t00 = time.time()
    smp = json.load(io.open(SAMPLE, encoding="utf-8"))
    rows = smp["rows"]
    names = [r["name"] for r in rows]
    kinds = np.array([r["kind"] for r in rows])
    log("sample %d names; authentic labels (kinds 0-2): %.1f %%" % (len(names), 100.0 * (kinds <= 2).mean()))
    free = wait_gpu(10)
    rec = {"RECORD": "FBX_ENC_PROBE", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sample_sha256": sha_file(SAMPLE), "n_sample": len(names),
           "priority": EH.lower_priority(), "threads": EH.set_threads(), "env": EH.env_versions(), "gpu_free_gib_at_start": round(free, 1), "module_sha256": sha_file(os.path.abspath(__file__)),
           "encoder": {"model": EH.GTE, "revision": EH.GTE_REV}}
    m = EH.load_dense()
    rec["default_max_seq_length"] = int(m.max_seq_length)
    rec["tokenizer_padding_side"] = m.tokenizer.padding_side
    tok = m.tokenizer
    ntok = np.array([len(x) for x in tok([n if n else " " for n in names], add_special_tokens=True, truncation=False)["input_ids"]])
    nch = np.array([len(n) for n in names])
    rec["tokens"] = {"all": pct(ntok), "kind_0_2": pct(ntok[kinds <= 2]), "kind_3_plus": pct(ntok[kinds > 2]) if (kinds > 2).any() else None,
                     "over": {str(c): int((ntok > c).sum()) for c in (64, 128, 256, 512, 1024)}, "chars": pct(nch), "kind_counts": {str(k): int((kinds == k).sum()) for k in sorted(set(kinds.tolist()))}}
    log("tokens: %s" % json.dumps(rec["tokens"]["all"]))
    m.max_seq_length = CAP
    rec["probe_max_seq_length"] = CAP
    rng = np.random.RandomState(1)
    sub = rng.choice(len(names), size=THROUGHPUT_N, replace=False)
    subn = [names[i] for i in sub]
    EH.dense_encode(m, subn[:256], 32)                                              # warm-up
    thr = {}
    vecs = {}
    for bs in BATCHES:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        v = EH.dense_encode(m, subn, bs)
        torch.cuda.synchronize()
        dt = time.time() - t
        thr[str(bs)] = {"docs_per_s": round(len(subn) / dt, 1), "seconds": round(dt, 2), "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        vecs[bs] = v
        log("batch %d: %s" % (bs, json.dumps(thr[str(bs)])))
    rec["throughput"] = thr
    D_DISTINCT = 245350737
    rec["extrapolated_hours_for_245350737_distinct_names"] = {b: round(D_DISTINCT / thr[b]["docs_per_s"] / 3600.0, 1) for b in thr}
    # batch-composition numerics: the same names encoded under different batch sizes (the vector depends on the padding of its batch only through fp16 arithmetic)
    ref = vecs[32].astype(np.float32)
    num = {}
    for bs in BATCHES[1:]:
        v = vecs[bs].astype(np.float32)
        cos = (ref * v).sum(axis=1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(v, axis=1)
        num["batch%d_vs_32" % bs] = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "cos_p01": float(np.percentile(cos, 1)), "rows_bitwise_equal": int((vecs[bs].view(np.uint16) == vecs[32].view(np.uint16)).all(axis=1).sum()),
                                      "max_abs_diff": float(np.abs(ref - v).max())}
    rec["batch_numerics"] = num
    rec["seconds"] = round(time.time() - t00, 1)
    wj(OUT, rec)
    log("ENCODE done: %s" % json.dumps({"thr": thr, "hours": rec["extrapolated_hours_for_245350737_distinct_names"], "num": num}))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "SAMPLE":
        cmd_sample()
    elif a and a[0] == "ENCODE":
        cmd_encode()
    else:
        raise SystemExit(__doc__)
