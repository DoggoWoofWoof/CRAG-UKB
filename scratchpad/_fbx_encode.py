"""FBX_SCALE Track C -- the Freebase Qwen encoding: deterministic, resumable chunks converted IMMEDIATELY to the validated compact representation (OPQ rotation + 8-bit product quantisation).

The encoder is scratchpad/_enc_host.py's (canonical C5 setup: gte-Qwen2-1.5B-instruct rev a9af15a6..., documents plain, fp16, normalised, use_cache off); the text of a node is its Freebase `name` string verbatim.
The unit of encoding is the DISTINCT NAME (results/FREEBASE_SCALE/names/FBX_NAME_DEDUP__v1.json: ptr[p] = row of position p, first_pos[r] = lowest position of row r; rows are numbered by ascending first
position).  Chunk c holds rows [c * CHUNK_ROWS, (c + 1) * CHUNK_ROWS); a row's vector is the vector of its name; every position inherits its row's vector through ptr.  No float matrix is ever kept except the training
sample and the holdout (both small, written once); every chunk's float batch is quantised and discarded.

  python -u scratchpad/_fbx_encode.py CONTRACT <m> <cap_tokens> <batch> <calibration_eval_record>   # write-once record BEFORE the first vector; everything below refuses to run without it
  python -u scratchpad/_fbx_encode.py TRAIN                                                          # encode the training sample + the holdout (float, kept), train OPQ+PQ, freeze the codebook
  python -u scratchpad/_fbx_encode.py CHUNKS <first> <last_exclusive>                                # resumable: skips chunks whose record + checksum verify
  python -u scratchpad/_fbx_encode.py VERIFY                                                         # every chunk present, checksummed, row ranges contiguous, total = distinct names
  python -u scratchpad/_fbx_encode.py TEST                                                           # 2 tiny chunks end to end in data/freebase_scale/enc_test (GPU, host)
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
NAMES_REC = os.path.join(REPO, "results", "FREEBASE_SCALE", "names", "FBX_NAME_DEDUP__v1.json")
DIM = 1536
CHUNK_ROWS = 131072
TRAIN_ROWS = 131072
HOLDOUT_ROWS = 50000
OPQ_NITER = 20
STATE = {"work": os.path.join(REPO, "data", "freebase_scale", "enc"), "recs": os.path.join(REPO, "results", "FREEBASE_SCALE", "enc"), "chunk_rows": CHUNK_ROWS, "train_rows": TRAIN_ROWS, "holdout_rows": HOLDOUT_ROWS}


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def wj(path, obj, overwrite=False):
    assert overwrite or not os.path.exists(path), "write-once: %s exists" % path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=True))
    os.replace(path + ".tmp", path)


def save_npy(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        np.save(f, a)
    os.replace(path + ".tmp", path)


def contract_path():
    return os.path.join(STATE["recs"], "FBX_QWEN_ENCODING_CONTRACT__v1.json")


def load_contract():
    p = contract_path()
    assert os.path.exists(p), "no encoding contract: run CONTRACT first"
    c = json.load(io.open(p, encoding="utf-8"))
    return c, sha_file(p)


# ---------------------------------------------------------------------------------------------------------------- names
class NodeNames:
    """position -> name, reading the node shards' row groups (cache of the last one; chunks are read in ascending position order)."""

    def __init__(self):
        import pyarrow.parquet as pq
        self.pq = pq
        self.files, self.starts, self.loc = [], [], []
        pos = 0
        for f in sorted(x for x in os.listdir(os.path.join(TREE, "nodes")) if x.startswith("shard_") and x.endswith(".parquet")):
            p = os.path.join(TREE, "nodes", f)
            md = pq.ParquetFile(p).metadata
            for g in range(md.num_row_groups):
                self.starts.append(pos)
                self.loc.append((p, g))
                pos += md.row_group(g).num_rows
        self.starts = np.array(self.starts, np.int64)
        self.N = pos
        self.cache = (None, None)

    def group(self, g):
        if self.cache[0] == g:
            return self.cache[1]
        p, gg = self.loc[g]
        tb = self.pq.ParquetFile(p).read_row_group(gg, columns=["position", "name"])
        pos = tb.column("position").to_numpy()
        assert int(pos[0]) == int(self.starts[g]) and int(pos[-1]) == int(self.starts[g]) + len(pos) - 1, "row group %d does not cover positions %d.." % (g, self.starts[g])
        self.cache = (g, tb.column("name"))
        return self.cache[1]

    def get(self, positions):
        positions = np.asarray(positions, np.int64)
        assert bool((np.diff(positions) > 0).all()), "positions must ascend"
        gi = np.searchsorted(self.starts, positions, side="right") - 1
        out = [None] * len(positions)
        for g in np.unique(gi):
            sel = np.flatnonzero(gi == g)
            col = self.group(int(g))
            import pyarrow as pa
            vals = col.take(pa.array(positions[sel] - self.starts[g])).to_pylist()
            for k, v in zip(sel.tolist(), vals):
                assert v is not None, "node at position %d has no name" % positions[k]
                out[k] = v
        return out


def distinct():
    fpos = np.load(os.path.join(NAMES, "first_pos.npy"), mmap_mode="r")
    return fpos


def names_digest(names):
    h = hashlib.sha256()
    for s in names:
        b = s.encode("utf-8", "surrogatepass")
        h.update(len(b).to_bytes(4, "little"))
        h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- encoder
def load_encoder(contract):
    import _enc_host as EH
    m = EH.load_dense()
    m.max_seq_length = int(contract["encoding"]["max_seq_length"])
    return EH, m


def wait_gpu(need_gib):
    import torch
    t0 = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        if free >= need_gib:
            return free
        assert time.time() - t0 < 6 * 3600, "GPU never had %.0f GiB free" % need_gib
        log("waiting for GPU memory (%.1f GiB free, need %.0f)" % (free, need_gib))
        time.sleep(60)


def encode_rows(EH, enc, contract, rows, fpos, nn):
    names = nn.get(np.asarray(fpos[rows], np.int64))
    v = EH.dense_encode(enc, names, int(contract["encoding"]["batch_size"]))
    return v, names


# ---------------------------------------------------------------------------------------------------------------- codebook
def train_codebook(Xtrain, m, nthreads):
    import faiss
    faiss.omp_set_num_threads(nthreads)
    x = Xtrain.astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    opq = faiss.OPQMatrix(DIM, m)
    opq.verbose = False
    opq.niter = OPQ_NITER
    opq.train(x)
    pq = faiss.ProductQuantizer(DIM, m, 8)
    pq.train(opq.apply(x))
    A = faiss.vector_to_array(opq.A).reshape(DIM, DIM)
    cb = faiss.vector_to_array(pq.centroids).reshape(m, 256, DIM // m)
    assert np.allclose(opq.apply(x[:64]), x[:64] @ A.T, atol=1e-4), "A is not the OPQ rotation as applied"
    return A, cb


def make_pq(cb):
    import faiss
    m = cb.shape[0]
    pq = faiss.ProductQuantizer(DIM, m, 8)
    faiss.copy_array_to_vector(np.ascontiguousarray(cb, np.float32).ravel(), pq.centroids)
    return pq


def quantise(A, pq, V, chunk=32768):
    """codes (uint8, n x m) of the fp16 vectors V: fp32 renormalise, rotate (y = x A^T), product-quantise."""
    m = pq.M
    out = np.empty((len(V), m), np.uint8)
    At = np.ascontiguousarray(A.T, np.float32)
    for a in range(0, len(V), chunk):
        x = V[a:a + chunk].astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        out[a:a + chunk] = pq.compute_codes(np.ascontiguousarray(x @ At))
    return out


def recon_cosine(A, pq, V, codes):
    x = V.astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = x @ A.T
    r = pq.decode(np.ascontiguousarray(codes))
    c = (y * r).sum(1) / (np.linalg.norm(y, axis=1) * np.maximum(np.linalg.norm(r, axis=1), 1e-12))
    return {"mean": float(c.mean()), "p05": float(np.percentile(c, 5)), "min": float(c.min()), "n": int(len(c))}


# ---------------------------------------------------------------------------------------------------------------- stages
def cmd_contract(m, cap, batch, eval_rec):
    out = contract_path()
    assert not os.path.exists(out), "write-once"
    import _enc_host as EH
    D = int(json.load(io.open(NAMES_REC, encoding="utf-8"))["n_distinct_names"])
    nn = NodeNames()
    assert len(distinct()) == D, "first_pos.npy does not have one entry per distinct name"
    rec = {"RECORD": "FBX_QWEN_ENCODING_CONTRACT", "version": 1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "status": "frozen before the first Freebase vector is encoded",
           "encoding": {"model": EH.GTE, "revision": EH.GTE_REV, "sentence_transformers_call": "SentenceTransformer(snapshot, trust_remote_code=True, local_files_only=True).half(); encode(names, batch_size, normalize_embeddings=True).astype(float16)",
                        "config_use_cache": False, "documents": "plain (no instruction prefix)", "text": "the Freebase node `name` string, verbatim (UTF-8), one text per distinct name", "max_seq_length": int(cap), "batch_size": int(batch),
                        "batching": "sentence-transformers sorts each encode() call by character length and cuts consecutive batches; one call per chunk", "dim": DIM, "dtype": "float16", "normalisation": "unit L2 (normalize_embeddings=True)",
                        "tokenizer": "the model's own (padding side as shipped)", "hf_offline": True},
           "ordering": {"unit": "distinct name; row r has first position first_pos[r]; ptr[p] = row of position p (results/FREEBASE_SCALE/names/FBX_NAME_DEDUP__v1.json); node-ID ordering = position = canonical row order of the node shards",
                        "n_positions": int(nn.N), "n_distinct_rows": D, "chunk_rows": STATE["chunk_rows"], "n_chunks": (D + STATE["chunk_rows"] - 1) // STATE["chunk_rows"], "chunk_c_rows": "[c * chunk_rows, min((c + 1) * chunk_rows, n_distinct_rows))",
                        "ptr_sha256": sha_file(os.path.join(NAMES, "ptr.npy")), "first_pos_sha256": sha_file(os.path.join(NAMES, "first_pos.npy")), "name_dedup_record_sha256": sha_file(NAMES_REC)},
           "representation": {"kind": "OPQ rotation (faiss OPQMatrix(1536, m), niter %d) + ProductQuantizer(1536, m, 8 bit)" % OPQ_NITER, "m_bytes_per_row": int(m), "train_rows": STATE["train_rows"], "train_sample": "RandomState(0).choice(n_distinct_rows, train_rows, replace=False), sorted",
                              "holdout": "RandomState(1) draw of min(n_distinct_rows, %d) rows, training rows removed, first %d kept, sorted (float fp16 kept for measurement only)" % (STATE["holdout_rows"] + 10000, STATE["holdout_rows"]),
                              "quantise": "fp16 -> fp32 renormalise -> y = x A^T -> pq.compute_codes", "float_policy": "no float chunk is kept: a chunk's vectors are quantised and discarded; only the training sample and the holdout stay as fp16",
                              "calibration_evidence": {"eval_record": eval_rec, "eval_record_sha256": sha_file(eval_rec) if os.path.exists(eval_rec) else None, "note": "the WebQSP end-to-end calibration (addenda 6/6A) accepts this m"}},
           "chunk_record": "per chunk: row range, vector count, names sha256 (length-prefixed UTF-8), codes sha256, model revision, contract sha256, codebook sha256, seconds; files codes_<c>.npy + chunk_<c>.json under data/freebase_scale/enc/chunks/",
           "code": {"scratchpad/_fbx_encode.py": sha_file(os.path.abspath(__file__)), "scratchpad/_enc_host.py": sha_file(os.path.join(HERE, "_enc_host.py"))}}
    wj(out, rec)
    log("contract written: %s" % sha_file(out))


def train_paths():
    w = STATE["work"]
    return {"train": os.path.join(w, "train_emb.npy"), "train_rows": os.path.join(w, "train_rows.npy"), "hold": os.path.join(w, "holdout_emb.npy"), "hold_rows": os.path.join(w, "holdout_rows.npy"),
            "codebook": os.path.join(w, "codebook_opq_pq.npz"), "rec": os.path.join(STATE["recs"], "FBX_QWEN_CODEBOOK__v1.json")}


def cmd_train(nthreads=6):
    c, csha = load_contract()
    P = train_paths()
    assert not os.path.exists(P["rec"]), "write-once"
    fpos = distinct()
    D = int(c["ordering"]["n_distinct_rows"])
    assert len(fpos) >= D
    tr = np.sort(np.random.RandomState(0).choice(D, STATE["train_rows"], replace=False))
    ho = np.random.RandomState(1).choice(D, min(D, STATE["holdout_rows"] + 10000), replace=False)
    ho = np.sort(ho[~np.isin(ho, tr)][:STATE["holdout_rows"]])
    t0 = time.time()
    wait_gpu(8)
    EH, enc = load_encoder(c)
    nn = NodeNames()
    out = {}
    for key, rows in (("train", tr), ("hold", ho)):
        if os.path.exists(P[key]):
            out[key] = np.load(P[key])
            assert len(out[key]) == len(rows)
            log("%s sample exists" % key)
            continue
        V = np.empty((len(rows), DIM), np.float16)
        names_all = []
        for a in range(0, len(rows), 32768):
            v, nm = encode_rows(EH, enc, c, rows[a:a + 32768], fpos, nn)
            V[a:a + len(v)] = v
            names_all += nm
            log("%s sample %d / %d encoded (%.0fs)" % (key, a + len(v), len(rows), time.time() - t0))
        save_npy(P[key + "_rows"], rows)
        save_npy(P[key], V)
        out[key] = V
        out[key + "_names_sha256"] = names_digest(names_all)
    del enc
    t1 = time.time()
    A, cb = train_codebook(out["train"], int(c["representation"]["m_bytes_per_row"]), nthreads)
    np.savez(P["codebook"] + ".tmp.npz", opq_A=A, pq_centroids=cb)
    os.replace(P["codebook"] + ".tmp.npz", P["codebook"])
    pq = make_pq(cb)
    rc_tr = recon_cosine(A, pq, out["train"][:20000], quantise(A, pq, out["train"][:20000]))
    rc_ho = recon_cosine(A, pq, out["hold"], quantise(A, pq, out["hold"]))
    wj(P["rec"], {"RECORD": "FBX_QWEN_CODEBOOK", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "contract_sha256": csha, "m": int(c["representation"]["m_bytes_per_row"]),
                  "train_rows_sha256": sha_arr(np.load(P["train_rows"])), "train_emb_sha256": sha_file(P["train"]), "holdout_rows_sha256": sha_arr(np.load(P["hold_rows"])), "holdout_emb_sha256": sha_file(P["hold"]),
                  "codebook_file": "data/freebase_scale/enc/codebook_opq_pq.npz", "codebook_sha256": sha_file(P["codebook"]), "opq_A_sha256": sha_arr(A), "pq_centroids_sha256": sha_arr(cb),
                  "reconstruction_cosine_train_first20000": rc_tr, "reconstruction_cosine_holdout": rc_ho, "encode_seconds": round(t1 - t0, 1), "train_seconds": round(time.time() - t1, 1),
                  "code_sha256": sha_file(os.path.abspath(__file__))})
    log("codebook frozen: holdout reconstruction cosine %.4f (p05 %.4f)" % (rc_ho["mean"], rc_ho["p05"]))


def chunk_paths(i):
    d = os.path.join(STATE["work"], "chunks")
    return os.path.join(d, "codes_%06d.npy" % i), os.path.join(d, "chunk_%06d.json" % i)


def chunk_ok(i, csha, cbsha):
    cp, jp = chunk_paths(i)
    if not (os.path.exists(cp) and os.path.exists(jp)):
        return False
    r = json.load(io.open(jp, encoding="utf-8"))
    return r["contract_sha256"] == csha and r["codebook_sha256"] == cbsha and sha_file(cp) == r["codes_sha256"]


def cmd_chunks(first, last, nthreads=6):
    c, csha = load_contract()
    P = train_paths()
    assert os.path.exists(P["rec"]), "no frozen codebook: run TRAIN first"
    import torch  # noqa: F401  (torch before faiss: the import order that works on the host)
    cbsha = json.load(io.open(P["rec"], encoding="utf-8"))["codebook_sha256"]
    assert sha_file(P["codebook"]) == cbsha
    z = np.load(P["codebook"])
    A, cb = z["opq_A"], z["pq_centroids"]
    pq = make_pq(cb)
    import faiss
    faiss.omp_set_num_threads(nthreads)
    fpos = distinct()
    D = int(c["ordering"]["n_distinct_rows"])
    cr = int(c["ordering"]["chunk_rows"])
    nch = int(c["ordering"]["n_chunks"])
    last = min(last, nch)
    todo = [i for i in range(first, last) if not chunk_ok(i, csha, cbsha)]
    log("chunks %d..%d of %d: %d to do" % (first, last, nch, len(todo)))
    if not todo:
        return
    wait_gpu(8)
    EH, enc = load_encoder(c)
    nn = NodeNames()
    t00 = time.time()
    for k, i in enumerate(todo):
        t0 = time.time()
        rows = np.arange(i * cr, min((i + 1) * cr, D))
        V, names = encode_rows(EH, enc, c, rows, fpos, nn)
        t1 = time.time()
        codes = quantise(A, pq, V)
        del V
        cp, jp = chunk_paths(i)
        save_npy(cp, codes)
        wj(jp, {"chunk": int(i), "row_first": int(rows[0]), "row_last_exclusive": int(rows[-1]) + 1, "vectors": int(len(rows)), "m": int(codes.shape[1]), "format": "uint8[n, m] OPQ+PQ codes (contract representation)",
                "names_sha256": names_digest(names), "codes_sha256": sha_file(cp), "model": c["encoding"]["model"], "revision": c["encoding"]["revision"], "contract_sha256": csha, "codebook_sha256": cbsha,
                "max_seq_length": c["encoding"]["max_seq_length"], "batch_size": c["encoding"]["batch_size"], "encode_seconds": round(t1 - t0, 1), "quantise_seconds": round(time.time() - t1, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           overwrite=True)
        el = time.time() - t00
        log("chunk %d (%d rows): encode %.0fs (%.0f names/s), quantise %.1fs; %d/%d done, eta %.1f h" % (i, len(rows), t1 - t0, len(rows) / (t1 - t0), time.time() - t1, k + 1, len(todo), el / (k + 1) * (len(todo) - k - 1) / 3600.0))


def cmd_verify():
    c, csha = load_contract()
    P = train_paths()
    cbsha = json.load(io.open(P["rec"], encoding="utf-8"))["codebook_sha256"]
    D = int(c["ordering"]["n_distinct_rows"])
    nch = int(c["ordering"]["n_chunks"])
    tot, prev = 0, 0
    for i in range(nch):
        assert chunk_ok(i, csha, cbsha), "chunk %d missing or does not verify" % i
        r = json.load(io.open(chunk_paths(i)[1], encoding="utf-8"))
        assert r["row_first"] == prev and r["vectors"] == r["row_last_exclusive"] - r["row_first"]
        prev = r["row_last_exclusive"]
        tot += r["vectors"]
    assert tot == D and prev == D
    log("VERIFY PASS: %d chunks, %d vectors = distinct names" % (nch, tot))


def cmd_test():
    """8 tiny chunks end to end on a scratch work dir (host, GPU): contract -> TRAIN (tiny) -> CHUNKS -> VERIFY -> the stored codes equal a direct re-encode+quantise."""
    STATE.update({"work": os.path.join(REPO, "data", "freebase_scale", "enc_test"), "recs": os.path.join(REPO, "results", "FREEBASE_SCALE", "enc_test"), "chunk_rows": 512, "train_rows": 2048, "holdout_rows": 256})
    import shutil
    for d in (STATE["work"], STATE["recs"]):
        shutil.rmtree(d, ignore_errors=True)
    # the contract records the real distinct count; the test confines itself to the first 4096 distinct names
    cmd_contract(16, 512, 32, "none")
    p = contract_path()
    c = json.load(io.open(p, encoding="utf-8"))
    c["ordering"]["n_distinct_rows"] = 4096
    c["ordering"]["n_chunks"] = 8
    wj(p, c, overwrite=True)
    cmd_train(nthreads=2)
    cmd_chunks(0, 8, nthreads=2)
    cmd_verify()
    # a direct re-encode of chunk 1 equals the stored codes
    c, csha = load_contract()
    P = train_paths()
    z = np.load(P["codebook"])
    pq = make_pq(z["pq_centroids"])
    EH, enc = load_encoder(c)
    nn = NodeNames()
    fpos = distinct()
    rows = np.arange(512, 1024)
    V, _ = encode_rows(EH, enc, c, rows, fpos, nn)
    again = quantise(z["opq_A"], pq, V)
    stored = np.load(chunk_paths(1)[0])
    assert np.array_equal(again, stored), "re-encode differs from the stored chunk"
    log("TEST PASS: contract -> train -> 8 chunks -> verify -> bit-identical re-encode; codes %s" % (stored.shape,))


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "CONTRACT" and len(a) == 5:
        cmd_contract(int(a[1]), int(a[2]), int(a[3]), a[4])
    elif cmd == "TRAIN":
        cmd_train()
    elif cmd == "CHUNKS" and len(a) == 3:
        cmd_chunks(int(a[1]), int(a[2]))
    elif cmd == "VERIFY":
        cmd_verify()
    elif cmd == "TEST":
        cmd_test()
    else:
        raise SystemExit(__doc__)
