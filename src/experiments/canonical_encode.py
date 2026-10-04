"""Phase C5 — reusable CANONICAL encodings (dense gte-Qwen2 + SPLADE), sharded / resumable / atomic.

Job unit = (dataset, kind, model, shard_id). Fixed SHARD_SIZE so shard_id is deterministic and each
job has exactly one owner. Source is PRE-SHARDED into per-shard jsonl slices so a worker uploads only
its slice (not e.g. hotpot's 3.89GB). Workers write ONLY their own shard artifacts + a status file;
they NEVER touch manifest.json (a single coordinator finalizes it). NO BM25 (unused by CRAG).

Layout under data/canonical/<ds>/encodings/:
  _src/<kind>/shard_NNNNN.jsonl                     pre-sharded source (id, text)
  <model>/<kind>/shard_NNNNN.npy | .npz             output (dense float16 / splade CSR)
  <model>/<kind>/ids_NNNNN.json                     row ids for the shard
  <model>/<kind>/status/shard_NNNNN.json            worker completion record (validated)
  <model>/<kind>/manifest.json, index.json          written by coordinator finalize() only

Atomic write: encode -> shard.tmp -> validate (rows, dim, no NaN/Inf) -> os.replace -> shard.npy.
A killed job never leaves a valid-looking shard.

Encoders (reproduced from reencode_ukb / splade_encode):
  dense = Alibaba-NLP/gte-Qwen2-1.5B-instruct (trust_remote_code, FP16, normalize); docs plain,
          queries prefixed with the retrieval instruction; dim 1536, stored float16.
  splade = naver/splade-cocondenser-ensembledistil, log(1+relu(logits)) max-pooled -> CSR (vocab).
"""
import os, json, glob, hashlib, logging, argparse
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
log = logging.getLogger("experiments.canonical_encode")

GTE = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
GTE_QINSTR = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
GTE_DIM = 1536
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
SHARD_SIZE = 40_000                                     # ~10-30 min GPU jobs; deterministic shard_id


def _canon_dir(ds): return os.path.join("data", "canonical", ds)
def _enc_dir(ds):   return os.path.join(_canon_dir(ds), "encodings")
def _src_dir(ds, kind): return os.path.join(_enc_dir(ds), "_src", kind)
def _out_dir(ds, kind, model): return os.path.join(_enc_dir(ds), model, kind)


def _source(ds, kind):
    """(path, id_field, text_field, source_sha) for docs|queries of dataset ds."""
    if kind == "docs":
        man = json.load(open(os.path.join(_canon_dir(ds), "document_manifest.json")))
        return (os.path.join(_canon_dir(ds), "documents.jsonl"), "canonical_doc_id", "text",
                man.get("documents_jsonl_sha256"))
    man = json.load(open(os.path.join(_canon_dir(ds), "query_manifest.json")))
    return (os.path.join(_canon_dir(ds), "queries.jsonl"), "query_id", "question",
            man.get("queries_jsonl_sha256"))


def n_items(ds, kind):
    path, _, _, sha = _source(ds, kind)
    meta_p = os.path.join(_src_dir(ds, kind), "_srcmeta.json")   # fast path: reuse pre-shard count (avoids re-scanning multi-GB jsonl)
    if os.path.exists(meta_p):
        try:
            m = json.load(open(meta_p))
            if m.get("source_sha256") == sha and "n_items" in m:
                return m["n_items"]
        except Exception:
            pass
    return sum(1 for _ in open(path, encoding="utf-8"))

def n_shards(ds, kind, shard_size=SHARD_SIZE):
    return (n_items(ds, kind) + shard_size - 1) // shard_size


# ── pre-shard the source jsonl into per-shard slices (idempotent) ─────────────
def pre_shard(ds, kind, shard_size=SHARD_SIZE):
    path, idf, txf, sha = _source(ds, kind)
    sd = _src_dir(ds, kind); os.makedirs(sd, exist_ok=True)
    meta_p = os.path.join(sd, "_srcmeta.json")
    if os.path.exists(meta_p):
        m = json.load(open(meta_p))
        if m.get("source_sha256") == sha and m.get("shard_size") == shard_size:
            return m["n_shards"]                        # already sharded on current source
    for p in glob.glob(os.path.join(sd, "shard_*.jsonl")): os.remove(p)
    idx = 0; shard = -1; fout = None; ns = 0
    for line in open(path, encoding="utf-8"):
        if idx % shard_size == 0:
            if fout: fout.close()
            shard += 1; ns += 1
            fout = open(os.path.join(sd, f"shard_{shard:05d}.jsonl"), "w", encoding="utf-8")
        r = json.loads(line)
        fout.write(json.dumps({"id": r[idf], "text": r.get(txf) or ""}, ensure_ascii=False) + "\n")
        idx += 1
    if fout: fout.close()
    json.dump({"source_sha256": sha, "shard_size": shard_size, "n_shards": ns, "n_items": idx},
              open(meta_p, "w"))
    log.info("[c5] pre-sharded %s/%s -> %d shards (%d items)", ds, kind, ns, idx)
    return ns


def _read_src_shard(ds, kind, shard_id):
    ids, texts = [], []
    for line in open(os.path.join(_src_dir(ds, kind), f"shard_{shard_id:05d}.jsonl"), encoding="utf-8"):
        r = json.loads(line); ids.append(r["id"]); texts.append(r["text"])
    return ids, texts


# ── shard validation / completion contract ───────────────────────────────────
def shard_complete(ds, kind, model, shard_id, expect_rows=None):
    od = _out_dir(ds, kind, model)
    ext = "npy" if model == "dense" else "npz"
    out = os.path.join(od, f"shard_{shard_id:05d}.{ext}")
    idp = os.path.join(od, f"ids_{shard_id:05d}.json")
    if not (os.path.exists(out) and os.path.exists(idp)):
        return False
    try:
        ids = json.load(open(idp))
        if model == "dense":
            a = np.load(out, mmap_mode="r")
            if a.shape[1] != GTE_DIM: return False
            rows = a.shape[0]
        else:
            import scipy.sparse as sp
            rows = sp.load_npz(out).shape[0]
        if rows != len(ids): return False
        if expect_rows is not None and rows != expect_rows: return False
        return True
    except Exception:
        return False


# ── worker: encode a set of shard ids (atomic) ────────────────────────────────
def run(ds, kind, model, shard_ids, batch=32, max_seq=None, shard_size=SHARD_SIZE):
    pre_shard(ds, kind, shard_size)                     # ensure slices exist (idempotent)
    od = _out_dir(ds, kind, model); os.makedirs(os.path.join(od, "status"), exist_ok=True)
    _, _, _, src_sha = _source(ds, kind)
    todo = [s for s in shard_ids if not shard_complete(ds, kind, model, s)]
    log.info("[c5] %s/%s/%s worker shards=%s todo=%s", ds, model, kind, shard_ids, todo)
    if not todo:
        return
    enc = _dense_encoder(batch, max_seq) if model == "dense" else _splade_encoder(batch, max_seq)
    for s in todo:
        ids, texts = _read_src_shard(ds, kind, s)
        enc.write_atomic(texts, ids, od, s, kind)
        json.dump({"dataset": ds, "kind": kind, "model": model, "shard": s, "rows": len(ids),
                   "source_sha256": src_sha, "ok": True, **enc.meta()},
                  open(os.path.join(od, "status", f"shard_{s:05d}.json"), "w"))
        log.info("[c5]   shard %05d OK (%d rows)", s, len(ids))


# ── coordinator: build manifest.json + index.json from validated shards ───────
def finalize(ds, kind, model, shard_size=SHARD_SIZE):
    od = _out_dir(ds, kind, model)
    ns = n_shards(ds, kind, shard_size); ntot = n_items(ds, kind)
    _, _, _, src_sha = _source(ds, kind)
    present, rows_total, index = [], 0, []
    for s in range(ns):
        a, b = s * shard_size, min((s + 1) * shard_size, ntot)
        if shard_complete(ds, kind, model, s, expect_rows=b - a):
            present.append(s); rows_total += (b - a)
            index.append({"shard": s, "rows": b - a,
                          "file": f"shard_{s:05d}." + ("npy" if model == "dense" else "npz"),
                          "ids": f"ids_{s:05d}.json"})
    meta = ({"encoder": GTE, "dim": GTE_DIM, "dtype": "float16", "query_instruction": GTE_QINSTR}
            if model == "dense" else {"encoder": SPLADE_MODEL})
    complete = present == list(range(ns))
    json.dump({"dataset": ds, "kind": kind, "model": model, "n_items": ntot, "rows_covered": rows_total,
               "shard_size": shard_size, "n_shards": ns, "shards_present": present, "complete": complete,
               "source_sha256": src_sha, **meta}, open(os.path.join(od, "manifest.json"), "w"), indent=2)
    json.dump({"shard_size": shard_size, "n_shards": ns, "shards": index},
              open(os.path.join(od, "index.json"), "w"), indent=2)
    log.info("[c5] finalize %s/%s/%s: %d/%d shards, %d/%d rows, complete=%s",
             ds, model, kind, len(present), ns, rows_total, ntot, complete)
    return complete, present, ns


# ── dense (gte-Qwen2) ─────────────────────────────────────────────────────────
class _dense_encoder:
    def __init__(self, batch, max_seq):
        from sentence_transformers import SentenceTransformer
        self.batch = batch
        self.model = SentenceTransformer(GTE, trust_remote_code=True).half()
        if max_seq: self.model.max_seq_length = int(max_seq)
        self.dim = self.model.get_sentence_embedding_dimension()
        log.info("[c5] dense gte-Qwen2 FP16 dim=%d batch=%d", self.dim, batch)
    def write_atomic(self, texts, ids, od, s, kind):
        pre = GTE_QINSTR if kind == "queries" else ""
        embs = self.model.encode([pre + t for t in texts], batch_size=self.batch,
                                  normalize_embeddings=True, show_progress_bar=True).astype("float16")
        if embs.shape != (len(ids), self.dim) or not np.isfinite(embs).all():
            raise RuntimeError(f"dense shard {s}: bad shape/NaN {embs.shape}")
        tmp = os.path.join(od, f"shard_{s:05d}.tmp.npy")
        np.save(tmp, embs)
        json.dump(ids, open(os.path.join(od, f"ids_{s:05d}.json"), "w"))
        os.replace(tmp, os.path.join(od, f"shard_{s:05d}.npy"))
    def meta(self): return {"encoder": GTE, "dim": self.dim, "dtype": "float16"}


# ── splade ────────────────────────────────────────────────────────────────────
class _splade_encoder:
    def __init__(self, batch, max_seq):
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.batch = batch; self.max_len = int(max_seq or 256)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(SPLADE_MODEL)
        self.mdl = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL).to(self.device).eval()
        self.vocab = self.mdl.config.vocab_size
        log.info("[c5] splade vocab=%d device=%s", self.vocab, self.device)
    def write_atomic(self, texts, ids, od, s, kind):
        import torch, scipy.sparse as sp
        rows, cols, data = [], [], []
        for i in range(0, len(texts), self.batch):
            b = [str(t) for t in texts[i:i + self.batch]]
            enc = self.tok(b, return_tensors="pt", truncation=True, max_length=self.max_len, padding=True).to(self.device)
            with torch.no_grad():
                v = torch.log1p(torch.relu(self.mdl(**enc).logits)).max(dim=1).values.cpu().numpy()
            for j in range(v.shape[0]):
                nz = np.nonzero(v[j])[0]
                rows += [i + j] * len(nz); cols += nz.tolist(); data += v[j][nz].tolist()
        mat = sp.csr_matrix((data, (rows, cols)), shape=(len(texts), self.vocab), dtype=np.float32)
        if mat.shape[0] != len(ids) or not np.isfinite(mat.data).all():
            raise RuntimeError(f"splade shard {s}: bad shape/NaN")
        tmp = os.path.join(od, f"shard_{s:05d}.tmp.npz")
        sp.save_npz(tmp, mat)
        json.dump(ids, open(os.path.join(od, f"ids_{s:05d}.json"), "w"))
        os.replace(tmp, os.path.join(od, f"shard_{s:05d}.npz"))
    def meta(self): return {"encoder": SPLADE_MODEL, "vocab": self.vocab, "max_len": self.max_len}


def _parse_ids(spec, ns):
    if spec in (None, "", "all"):
        return list(range(ns))
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < ns]


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="canonical-encode")
    p.add_argument("--dataset", required=True)
    p.add_argument("--kind", required=True, choices=["docs", "queries"])
    p.add_argument("--model", required=True, choices=["dense", "splade"])
    p.add_argument("--shard-ids", default="all", help="e.g. '0,3,5' or '10-14' or 'all'")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-seq", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--pre-shard-only", action="store_true", help="just build _src slices (local, no GPU)")
    p.add_argument("--finalize", action="store_true", help="coordinator: build manifest.json+index.json")
    a = p.parse_args(argv)
    if a.pre_shard_only:
        pre_shard(a.dataset, a.kind, a.shard_size); return
    if a.finalize:
        finalize(a.dataset, a.kind, a.model, a.shard_size); return
    ns = pre_shard(a.dataset, a.kind, a.shard_size)
    ids = _parse_ids(a.shard_ids, ns)
    run(a.dataset, a.kind, a.model, ids, batch=a.batch, max_seq=a.max_seq, shard_size=a.shard_size)


if __name__ == "__main__":
    main()
