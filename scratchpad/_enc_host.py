"""Stage L3_HOST addendum 6: the canonical encoders on the host GPU.
  REPRO   the placement test of the encoder setup (nothing scientific): re-encode frozen canonical vectors and compare.
  RELENC  the E2 relation-name encodings (REPORT section 40.15); refuses to run unless REPRO passed and a dated authorization
          names this module.

Setup, as the canonical C5 encoder src/experiments/canonical_encode.py (not pushed; its constants are restated here and
checked against the frozen manifests); CPU threads = OPENBLAS_NUM_THREADS (rx sets it to --cpus):
  dense   sentence_transformers.SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", revision a9af15a6...,
          trust_remote_code=True, cache_folder=<addendum-5 cache>, local_files_only=True) on cuda, then .half();
          encode(texts, batch_size, normalize_embeddings=True).astype(float16); documents plain, queries prefixed with GTE_QINSTR
  splade  AutoTokenizer / AutoModelForMaskedLM("naver/splade-cocondenser-ensembledistil", revision 49cf4c7b...) on cuda, fp32,
          eval; consecutive batches in input order; tok(b, truncation=True, max_length=256, padding=True) ->
          log1p(relu(logits)).max(dim=1) over EVERY position, the padded ones included (the canonical encoder pools without the
          attention mask, so a vector depends on the padding of its batch) -> CSR float32
  The canonical MetaQA runs used dense batch 32 and splade batch 64 over whole 40,000-row shards (scratchpad/c5_workers logs).
  Offline only (HF_HUB_OFFLINE=1); no token.

REPRO
  docs     all 43,234 MetaQA node texts (nodes.jsonl "text"; row i == line i), per canonical shard (rows 0..39,999 and
           40,000..43,233) with the canonical batch sizes.  The archived materialisation pointers (_history/materialize/
           metaqa__pointer_index__{dense,splade}.npz, shas asserted against the frozen manifests) say the phase-C stores were in
           node order: dense row == position for every node; splade row == position except 3,087 positions (3,083 case/accent
           variants of an earlier name) that reuse the earlier position's row.  So the re-encoded batches equal the canonical
           ones, and the frozen splade row of position i is compared with the re-encoding at pointer row[i]
  queries  the 1,998 L1_DEV population rows only (the rows of the TYPED v1 npz).  Each row must be a dev row by its query id; the
           question text is read from those lines of queries/dev.jsonl only (never another row: the canonical query shards also
           hold reserved held-out and TEST rows).  Both text fields (question, question_plain), in population order, canonical
           batch sizes.  The canonical query batches (per-hop stores) cannot be rebuilt without reading other rows, so splade
           queries (pooled over padding) are reported, not gated
  metrics  per-row cosine, max |diff|, rows exactly equal, splade support Jaccard; nearest frozen doc (over all 43,234) of each
           re-encoded doc: dense must carry the same text, splade must be the same stored row (pointer)
  gate     as declared in addendum 6, read from it
RELENC
  the relation vocabularies of the frozen graph manifests (MetaQA 9 labels, WebQSP 7,058; manifest shas asserted), verbalised as
  the lane's Typed verbalises them for its tokens (label.replace("_", " ").replace(".", " ")), plus for MetaQA the 72 ordered
  pairs of distinct relations as "<r1>, <r2>"; encoded as DOCUMENTS (plain), each string alone (batch 1: no padding, so a vector
  does not depend on the other strings) -> results/L3_DEV/relenc_<ds>__v1.{json,npz} (write-once)

Usage (host, through rx, env mpr-cu128): python -u scratchpad/_enc_host.py PROBE (no GPU) | REPRO | RELENC (--gpus 1)
"""
import hashlib
import importlib.util
import io
import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_MODULES_CACHE"] = "C:/Users/Student2/rx/projects/crag/models/hf_modules"  # the remote-code copy lands here
for _k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    os.environ.pop(_k, None)

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ADD6 = "results/L3_HOST/HOST_STAGE_DECLARATION__L3_HOST__v1__ADDENDUM_6.json"
DL = "results/L3_HOST/HF_DOWNLOAD__v1.json"
AUTH = "results/L3_HOST/HOST_STAGE_AUTHORIZATION__L3_HOST_ENC__2026-09-29.json"
CACHE = "C:/Users/Student2/rx/projects/crag/models/hf"
GTE = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
GTE_REV = "a9af15a6372d7d6b25e9fb07c2ccb9e1fe645644"
SPL = "naver/splade-cocondenser-ensembledistil"
SPL_REV = "49cf4c7b0db5b870a401ddf5e2669993ef3699c7"
SPL_ST_REV = "83f6dbef1346b4685ff94ab120a5b8e9a7bf7234"
TIES = {"cls.predictions.decoder.weight": "bert.embeddings.word_embeddings.weight",
        "cls.predictions.decoder.bias": "cls.predictions.bias"}
GTE_QINSTR = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
SPLADE_MAXLEN = 256
DEVICE = "cuda"
DENSE_BATCH, SPLADE_BATCH = 32, 64
SHARD = 40000
MQ = "data/final_canonical/metaqa"
TYPED_NPZ = "results/L3_DEV/l3typed_metaqa__v1.npz"
TYPED_NPZ_SHA = "0ee65e76660223d0378708582e8bd19142e58595ff495340705d5112d29e1355"
REPRO_OUT = "results/L3_HOST/ENC_REPRO__v1.json"
PTR = {"dense": "data/final_canonical/_history/materialize/metaqa__pointer_index__dense.npz",
       "splade": "data/final_canonical/_history/materialize/metaqa__pointer_index__splade.npz"}
MANIFEST_SHA = {"metaqa": "df360a84ab18b006f4f57391d02fc63992dc61495d3a7899fca8a0c6f8313bc9",
                "webqsp": "4130ad1fee08dac8c9a84a7cd4c2e56a2cbbd4909c97d932a723d13cd90a7a2c"}
RELENC_OUT = "results/L3_DEV/relenc_%s__v1"


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        return "BELOW_NORMAL"
    except Exception as e:  # noqa: BLE001
        return "unchanged (%s)" % e


def set_threads():
    import torch
    n = int(os.environ.get("OPENBLAS_NUM_THREADS", "8"))
    torch.set_num_threads(n)
    return n


def env_versions():
    import platform
    import torch
    import transformers
    import sentence_transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "processor": platform.processor(), "torch_threads": torch.get_num_threads(),
            "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None}


def snap(repo, rev):
    """the local snapshot directory of repo@rev in the addendum-5 cache.  Models are loaded from it, not by repo id: transformers
    4.57.3's tokenizer loader asks the Hub API about a repo id (is_base_mistral -> model_info) even in offline mode."""
    p = os.path.join(CACHE, "models--" + repo.replace("/", "--"), "snapshots", rev).replace("\\", "/")
    assert os.path.isdir(p), p
    return p


def load_dense(dev=None):
    """the canonical load, then use_cache off: the remote modeling_qwen.py (written for transformers 4.41) builds a KV cache
    with DynamicCache.get_usable_length, which transformers 4.57 no longer has.  One forward pass does not read the cache (it
    stores keys/values for later decoding steps), so the embeddings do not depend on it."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(snap(GTE, GTE_REV), trust_remote_code=True, local_files_only=True, device=dev or DEVICE)
    m[0].auto_model.config.use_cache = False
    if (dev or DEVICE) != "cpu":
        m = m.half()
    return m


def dense_encode(m, texts, batch):
    v = m.encode(list(texts), batch_size=batch, normalize_embeddings=True, show_progress_bar=False).astype("float16")
    assert v.shape == (len(texts), 1536) and np.isfinite(v).all()
    return v


def load_splade(dev=None):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap(SPL, SPL_REV), local_files_only=True)
    mdl = AutoModelForMaskedLM.from_pretrained(snap(SPL, SPL_REV), local_files_only=True).to(dev or DEVICE).eval()
    return tok, mdl


def splade_weights_identical():
    """the .bin of refs/main and the model.safetensors of the conversion commit hold identical tensors (asserted, recorded)."""
    import torch
    from safetensors.torch import load_file
    pb = snap(SPL, SPL_REV) + "/pytorch_model.bin"
    ps = snap(SPL, SPL_ST_REV) + "/model.safetensors"
    a = torch.load(pb, map_location="cpu", weights_only=True)
    b = load_file(ps)
    ka, kb = set(a), set(b)
    same = sorted(ka & kb)
    diff = [k for k in same if not (a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]))]
    # the safetensors conversion drops tied duplicates; a .bin-only key is accepted only as an exact copy of its
    # tie target (BertForMaskedLM: decoder.weight = word embeddings, decoder.bias = predictions.bias), a key both files hold
    ties = {}
    for k in sorted(ka - kb):
        t = TIES.get(k)
        ties[k] = {"tied_to": t, "tie_target_in_both_files": bool(t and t in ka and t in kb),
                   "equal_to_tie_target": bool(t and t in ka and a[k].dtype == a[t].dtype and a[k].shape == a[t].shape
                                               and torch.equal(a[k], a[t]))}
    return {"bin_keys": len(ka), "safetensors_keys": len(kb), "only_bin": sorted(ka - kb), "only_safetensors": sorted(kb - ka),
            "shared": len(same), "shared_differing": diff, "identical_shared": not diff, "only_bin_ties": ties,
            "only_bin_all_verified_ties": all(v["tie_target_in_both_files"] and v["equal_to_tie_target"] for v in ties.values())}


def splade_encode(tok, mdl, texts, batch, dev=None):
    import torch
    rows, cols, data = [], [], []
    for i in range(0, len(texts), batch):
        b = [str(t) for t in texts[i:i + batch]]
        enc = tok(b, return_tensors="pt", truncation=True, max_length=SPLADE_MAXLEN, padding=True).to(dev or DEVICE)
        with torch.no_grad():
            v = torch.log1p(torch.relu(mdl(**enc).logits)).max(dim=1).values.cpu().numpy()
        for j in range(v.shape[0]):
            nz = np.nonzero(v[j])[0]
            rows += [i + j] * len(nz)
            cols += nz.tolist()
            data += v[j][nz].tolist()
    mat = sp.csr_matrix((data, (rows, cols)), shape=(len(texts), mdl.config.vocab_size), dtype=np.float32)
    assert mat.shape[0] == len(texts) and np.isfinite(mat.data).all()
    return mat


def q(x, n=6):
    return None if x is None else float(round(float(x), n))


def dense_cmp(H, C):
    h = H.astype(np.float32)
    c = C.astype(np.float32)
    cos = (h * c).sum(1) / (np.linalg.norm(h, axis=1) * np.linalg.norm(c, axis=1))
    return {"rows": int(len(cos)), "cos_mean": q(cos.mean(), 7), "cos_min": q(cos.min(), 7), "cos_p001": q(np.quantile(cos, 0.001), 7),
            "rows_cos_below_0.999": int((cos < 0.999).sum()), "rows_cos_below_0.99": int((cos < 0.99).sum()),
            "max_abs_diff": q(np.abs(h - c).max(), 7), "rows_exactly_equal": int((H == C).all(1).sum())}, cos


def sparse_cmp(Hs, Cs):
    Hs = Hs.tocsr()
    Cs = Cs.tocsr()
    num = np.asarray(Hs.multiply(Cs).sum(1)).ravel().astype(np.float64)
    nh = np.sqrt(np.asarray(Hs.multiply(Hs).sum(1)).ravel().astype(np.float64))
    nc = np.sqrt(np.asarray(Cs.multiply(Cs).sum(1)).ravel().astype(np.float64))
    cos = num / np.maximum(nh * nc, 1e-30)
    jac = []
    eq = 0
    for i in range(Hs.shape[0]):
        a = set(Hs.indices[Hs.indptr[i]:Hs.indptr[i + 1]].tolist())
        b = set(Cs.indices[Cs.indptr[i]:Cs.indptr[i + 1]].tolist())
        jac.append(len(a & b) / max(1, len(a | b)))
        if a == b:
            ra = dict(zip(Hs.indices[Hs.indptr[i]:Hs.indptr[i + 1]].tolist(), Hs.data[Hs.indptr[i]:Hs.indptr[i + 1]].tolist()))
            rb = dict(zip(Cs.indices[Cs.indptr[i]:Cs.indptr[i + 1]].tolist(), Cs.data[Cs.indptr[i]:Cs.indptr[i + 1]].tolist()))
            eq += int(ra == rb)
    jac = np.asarray(jac)
    d = abs(Hs - Cs)
    return {"rows": int(len(cos)), "cos_mean": q(cos.mean(), 7), "cos_min": q(cos.min(), 7), "cos_p001": q(np.quantile(cos, 0.001), 7),
            "rows_cos_below_0.999": int((cos < 0.999).sum()), "rows_cos_below_0.99": int((cos < 0.99).sum()),
            "jaccard_mean": q(jac.mean(), 7), "jaccard_min": q(jac.min(), 7), "rows_same_support": int((jac == 1.0).sum()),
            "rows_exactly_equal": int(eq), "max_abs_diff": q(d.max() if d.nnz else 0.0, 7),
            "nnz_host": int(Hs.nnz), "nnz_frozen": int(Cs.nnz)}, cos


def nn_same_text_dense(H, C, texts):
    import torch
    Ct = torch.from_numpy(C.astype(np.float32)).to(DEVICE)
    hits = 0
    self_hits = 0
    for i in range(0, H.shape[0], 4096):
        h = torch.from_numpy(H[i:i + 4096].astype(np.float32)).to(DEVICE)
        a = (h @ Ct.T).argmax(1).cpu().numpy()
        for j, k in enumerate(a):
            hits += int(texts[int(k)] == texts[i + j])
            self_hits += int(int(k) == i + j)
    return {"rows": int(H.shape[0]), "nearest_frozen_same_text": int(hits), "nearest_frozen_is_self": int(self_hits)}


def l2rows(M):
    M = M.tocsr().astype(np.float64)
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
    n[n == 0] = 1.0
    return sp.diags(1.0 / n) @ M


def nn_same_row_sparse(Hs, Cs, ptr):
    """Hs row p = the re-encoding at position p; the frozen store holds row ptr[i] at position i.  A hit: the nearest frozen
    position k of re-encoded position p holds the stored row of p (ptr[k] == p's stored row, i.e. ptr[k] == ptr[p]).  Nearest by
    cosine (splade vectors are not unit length; by raw dot product a longer vector can outscore the row itself)."""
    Hs = l2rows(Hs)
    CT = l2rows(Cs).T.tocsc()
    hits = 0
    self_hits = 0
    for i in range(0, Hs.shape[0], 1024):
        blk = (Hs[i:i + 1024] @ CT).toarray()
        a = blk.argmax(1)
        for j, k in enumerate(a):
            hits += int(ptr[int(k)] == ptr[i + j])
            self_hits += int(int(k) == i + j)
    return {"rows": int(Hs.shape[0]), "nearest_frozen_same_stored_row": int(hits), "nearest_frozen_is_self": int(self_hits)}


def load_pointers(man):
    """the archived materialisation pointers of the MetaQA doc stores (shas asserted against the frozen manifests)."""
    out = {}
    for ch in ("dense", "splade"):
        exp = man["%s/docs" % ch]["materialised"]["pointer_index_sha256"]
        got = fsha(PTR[ch])
        assert got == exp, (PTR[ch], got, exp)
        z = np.load(PTR[ch], allow_pickle=False)
        src, row = z["src"], z["row"].astype(np.int64)
        assert len(row) == 43234 and (src == 0).all()
        out[ch] = row
    assert (out["dense"] == np.arange(43234)).all(), "dense pointer is not the identity"
    r = out["splade"]
    red = np.nonzero(r != np.arange(43234))[0]
    assert (r[red] < red).all() and (r[r] == r).all(), "splade pointer: a redirect is not to an earlier self-stored row"
    return out, {"dense": {"sha256": fsha(PTR["dense"]), "identity": True},
                 "splade": {"sha256": fsha(PTR["splade"]), "redirected_positions": int(len(red)),
                            "distinct_stored_rows": int(len(np.unique(r))), "redirects_to_earlier_self_stored_row": True}}


def population_questions(rows):
    """both text fields of the population rows only: each row's query id must be a dev id; only those lines of dev.jsonl are
    parsed (stop after the last one); the parsed query_id is checked against query_ids.json."""
    qids = json.load(io.open(MQ + "/queries/query_ids.json", encoding="utf-8"))
    dev_idx = [j for j, s in enumerate(qids) if ":dev:" in s]
    dev0, dev1 = dev_idx[0], dev_idx[-1] + 1
    assert dev_idx == list(range(dev0, dev1)), "dev ids are not one contiguous block"
    want = {}
    for r in rows:
        r = int(r)
        assert ":test:" not in qids[r], "refused: row %d is a TEST row" % r
        assert dev0 <= r < dev1 and ":dev:" in qids[r], "refused: row %d is not a dev row" % r
        want[r - dev0] = r
    last = max(want)
    out = {}
    with io.open(MQ + "/queries/dev.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i in want:
                d = json.loads(ln)
                r = want[i]
                assert d["query_id"] == qids[r], "query id mismatch at row %d" % r
                assert d.get("split") == "dev"
                out[r] = (d.get("question") or "", d.get("question_plain") or "")
            if i >= last:
                break
    assert len(out) == len(rows) and all(out[int(r)][0] and out[int(r)][1] for r in rows)
    return [out[int(r)][0] for r in rows], [out[int(r)][1] for r in rows], {"dev_block": [dev0, dev1], "lines_parsed": len(out)}


def probe():
    """load both encoders on the CPU and encode two fixed strings (no dataset data): the code path check before the GPU is asked
    for (so another project's job is not made to yield for a run that cannot load)."""
    out = "results/L3_HOST/ENC_PROBE__v1.json"
    k = 1
    while os.path.exists(out):
        k += 1
        out = "results/L3_HOST/ENC_PROBE__v1_attempt%d.json" % k
    t0 = time.time()
    rec = {"what": "PROBE: both encoders loaded on the CPU, two fixed strings encoded; no dataset data", "module_sha256": fsha(__file__),
           "priority": lower_priority(), "threads": set_threads(), "env": env_versions()}
    strings = ["a probe string", "Instruct: a second probe string"]
    m = load_dense("cpu")
    rec["dense_attn_implementation"] = getattr(m[0].auto_model.config, "_attn_implementation", None)
    rec["dense_use_cache"] = m[0].auto_model.config.use_cache
    rec["dense_max_seq_length"] = m.max_seq_length
    rec["dense_tokenizer_padding_side"] = m.tokenizer.padding_side
    v = m.encode(strings, batch_size=2, normalize_embeddings=True, show_progress_bar=False)
    rec["dense"] = {"shape": list(v.shape), "norms": [q(x) for x in np.linalg.norm(v, axis=1)], "finite": bool(np.isfinite(v).all())}
    del m
    tok, mdl = load_splade("cpu")
    S = splade_encode(tok, mdl, strings, 2, dev="cpu")
    rec["splade"] = {"shape": list(S.shape), "nnz": int(S.nnz), "vocab": int(mdl.config.vocab_size)}
    rec["splade_weight_files"] = splade_weights_identical()
    rec["seconds"] = round(time.time() - t0, 1)
    rec["status"] = "OK" if rec["dense"]["finite"] and rec["dense"]["shape"] == [2, 1536] and S.nnz > 0 else "BAD"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(out, json.dumps(rec), flush=True)
    sys.exit(0 if rec["status"] == "OK" else 1)


def repro():
    if os.path.exists(REPRO_OUT):
        sys.exit("refusing: %s exists (write-once)" % REPRO_OUT)
    t_all = time.time()
    prio = lower_priority()
    set_threads()
    add6 = json.load(open(ADD6, encoding="utf-8"))
    gate = add6["reproduction_test"]["gate"]
    dl = json.load(open(DL, encoding="utf-8"))
    assert dl["status"] == "VERIFIED", "the addendum-5 download is not VERIFIED"
    rec = {"stage": "L3_HOST", "addendum": 6, "what": "REPRO: the placement test of the encoder setup",
           "addendum_sha256": fsha(ADD6), "download_record_sha256": fsha(DL), "module_sha256": fsha(__file__),
           "priority": prio, "env": env_versions(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    # ---- frozen inputs: manifests, shard shas
    man = {k: json.load(open(MQ + "/embeddings/%s/manifest.json" % k, encoding="utf-8"))
           for k in ("dense/docs", "dense/queries", "splade/docs", "splade/queries")}
    assert man["dense/docs"]["encoder"]["encoder"] == GTE and man["dense/queries"]["encoder"]["encoder"] == GTE
    assert man["dense/docs"]["encoder"]["query_instruction"] == GTE_QINSTR == man["dense/queries"]["encoder"]["query_instruction"]
    assert man["splade/docs"]["encoder"]["encoder"] == SPL and man["splade/queries"]["encoder"]["encoder"] == SPL
    used = {("dense/docs", 0), ("dense/docs", 1), ("splade/docs", 0), ("splade/docs", 1)}
    rows = np.load(TYPED_NPZ)["rows"].astype(np.int64)
    assert fsha(TYPED_NPZ) == TYPED_NPZ_SHA and len(rows) == 1998
    qshards = sorted(set((rows // SHARD).tolist()))
    assert set(qshards) <= {8, 9}, qshards
    used |= {("dense/queries", s) for s in qshards} | {("splade/queries", s) for s in qshards}
    shas = {}
    for k, s in sorted(used):
        ent = next(e for e in man[k]["shards"] if e["shard"] == s)
        p = MQ + "/embeddings/%s/%s" % (k, ent["file"])
        got = fsha(p)
        assert got == ent["sha256"], (p, got, ent["sha256"])
        shas["%s/%s" % (k, ent["file"])] = got
    rec["frozen_shards_verified"] = shas
    ptr, rec["pointers"] = load_pointers(man)
    # ---- texts
    texts = [json.loads(ln)["text"] for ln in io.open(MQ + "/nodes.jsonl", encoding="utf-8")]
    assert len(texts) == man["dense/docs"]["n_rows"] == 43234
    qa, qb, qinfo = population_questions(rows)
    rec["population"] = {"rows": int(len(rows)), "query_shards": qshards, **qinfo}
    # ---- splade weight files
    rec["splade_weight_files"] = splade_weights_identical()
    # ---- dense
    t0 = time.time()
    dm = load_dense()
    rec["dense_load_seconds"] = round(time.time() - t0, 1)
    rec["dense_attn_implementation"] = getattr(dm[0].auto_model.config, "_attn_implementation", None)
    t0 = time.time()
    Hd = np.concatenate([dense_encode(dm, texts[s * SHARD:(s + 1) * SHARD], DENSE_BATCH) for s in (0, 1)])
    rec["dense_docs_seconds"] = round(time.time() - t0, 1)
    Cd = np.concatenate([np.load(MQ + "/embeddings/dense/docs/shard_%05d.npy" % s) for s in (0, 1)])
    rec["dense_docs"], _ = dense_cmp(Hd, Cd)
    rec["dense_docs"]["nn"] = nn_same_text_dense(Hd, Cd, texts)
    parts = {s: np.load(MQ + "/embeddings/dense/queries/shard_%05d.npy" % s, mmap_mode="r") for s in qshards}
    Cq = np.stack([np.asarray(parts[int(r // SHARD)][int(r % SHARD)]) for r in rows])
    del parts
    t0 = time.time()
    rec["dense_queries"] = {}
    for name, tx in (("question", qa), ("question_plain", qb)):
        Hq = dense_encode(dm, [GTE_QINSTR + t for t in tx], DENSE_BATCH)
        rec["dense_queries"][name], _ = dense_cmp(Hq, Cq)
    rec["dense_queries_seconds"] = round(time.time() - t0, 1)
    del dm
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # ---- splade
    t0 = time.time()
    tok, mdl = load_splade()
    rec["splade_load_seconds"] = round(time.time() - t0, 1)
    t0 = time.time()
    Hs = sp.vstack([splade_encode(tok, mdl, texts[s * SHARD:(s + 1) * SHARD], SPLADE_BATCH) for s in (0, 1)]).tocsr()
    rec["splade_docs_seconds"] = round(time.time() - t0, 1)
    Cs = sp.vstack([sp.load_npz(MQ + "/embeddings/splade/docs/shard_%05d.npz" % s) for s in (0, 1)]).tocsr()
    rec["splade_docs"], _ = sparse_cmp(Hs[ptr["splade"]], Cs)
    rec["splade_docs"]["compared"] = "frozen position i vs the re-encoding at pointer row[i] (the stored row's own batch)"
    rec["splade_docs"]["nn"] = nn_same_row_sparse(Hs, Cs, ptr["splade"])
    red = np.nonzero(ptr["splade"] != np.arange(len(texts)))[0]
    rec["splade_docs_redirected_own_position"], _ = sparse_cmp(Hs[red], Cs[red])
    rec["splade_docs_redirected_own_position"]["what"] = ("report only: the %d redirected positions compared at their OWN "
                                                          "position (another batch; the pooled-over-padding effect)" % len(red))
    sq = {s: sp.load_npz(MQ + "/embeddings/splade/queries/shard_%05d.npz" % s).tocsr() for s in qshards}
    Csq = sp.vstack([sq[int(r // SHARD)][int(r % SHARD)] for r in rows]).tocsr()
    rec["splade_queries"] = {}
    for name, tx in (("question", qa), ("question_plain", qb)):
        Hsq = splade_encode(tok, mdl, tx, SPLADE_BATCH)
        rec["splade_queries"][name], _ = sparse_cmp(Hsq, Csq)
    # ---- the gate (declared in addendum 6)
    ok = {}
    for ch, hit in (("dense_docs", "nearest_frozen_same_text"), ("splade_docs", "nearest_frozen_same_stored_row")):
        r_ = rec[ch]
        ok[ch] = bool(r_["cos_mean"] >= gate["docs_cos_mean_min"] and r_["cos_min"] >= gate["docs_cos_min_min"]
                      and r_["nn"][hit] >= gate["docs_nn_hit_share_min"] * r_["rows"])
    best = max(("question", "question_plain"), key=lambda f: rec["dense_queries"][f]["cos_mean"])
    rq = rec["dense_queries"][best]
    ok["dense_queries(%s)" % best] = bool(rq["cos_mean"] >= gate["queries_dense_cos_mean_min"]
                                          and rq["cos_min"] >= gate["queries_dense_cos_min_min"])
    ok["splade_weight_files_identical"] = bool(rec["splade_weight_files"]["identical_shared"]
                                               and rec["splade_weight_files"]["only_bin_all_verified_ties"]
                                               and not rec["splade_weight_files"]["only_safetensors"])
    rec["query_field_matching_canonical"] = best
    rec["gate"] = gate
    rec["gate_results"] = ok
    rec["status"] = "PASS" if all(ok.values()) else "FAIL"
    rec["seconds"] = round(time.time() - t_all, 1)
    with open(REPRO_OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("status", "gate_results", "query_field_matching_canonical", "seconds")}), flush=True)
    for ch in ("dense_docs", "splade_docs", "splade_docs_redirected_own_position"):
        print(ch, json.dumps(rec[ch]), flush=True)
    for ch in ("dense_queries", "splade_queries"):
        for f_ in ("question", "question_plain"):
            print(ch, f_, json.dumps(rec[ch][f_]), flush=True)
    sys.exit(0 if rec["status"] == "PASS" else 1)


def relenc():
    t_all = time.time()
    prio = lower_priority()
    set_threads()
    auth = json.load(open(AUTH, encoding="utf-8"))
    me = fsha(__file__)
    assert auth["module"]["path"] == "scratchpad/_enc_host.py" and auth["module"]["sha256"] == me, "the authorization does not name this module"
    rp = json.load(open(REPRO_OUT, encoding="utf-8"))
    assert rp["status"] == "PASS" and rp["module_sha256"] == me, "REPRO did not pass with this module"
    assert auth["repro"]["sha256"] == fsha(REPRO_OUT)
    outs = [RELENC_OUT % ds for ds in ("metaqa", "webqsp")]
    for o in outs:
        for ext in (".json", ".npz"):
            if os.path.exists(o + ext):
                sys.exit("refusing: %s exists (write-once)" % (o + ext))
    import torch
    dm = load_dense()
    tok, mdl = load_splade()
    ver = env_versions()
    done = {}
    for ds in ("metaqa", "webqsp"):     # both are encoded before anything is written: a run stopped early leaves no output
        t0 = time.time()
        mp = "data/final_canonical/%s/graph/GRAPH_MANIFEST.json" % ds
        assert fsha(mp) == MANIFEST_SHA[ds], "graph manifest sha differs: " + mp
        voc = json.load(open(mp, encoding="utf-8"))["families"]["structural"]["relation_vocabulary"]
        verb = [lbl.replace("_", " ").replace(".", " ") for lbl in voc]
        strings = list(verb)
        kind = [0] * len(verb)
        r1 = list(range(len(verb)))
        r2 = [-1] * len(verb)
        if ds == "metaqa":
            for i in range(len(verb)):
                for j in range(len(verb)):
                    if i != j:
                        strings.append(verb[i] + ", " + verb[j])
                        kind.append(1)
                        r1.append(i)
                        r2.append(j)
        D_ = dense_encode(dm, strings, 1)
        S_ = splade_encode(tok, mdl, strings, 1)
        done[ds] = (mp, voc, strings, kind, r1, r2, D_, S_, round(time.time() - t0, 1))
    for ds, (mp, voc, strings, kind, r1, r2, D_, S_, secs) in done.items():
        o = RELENC_OUT % ds
        np.savez(o + ".npz", dense=D_, sp_indptr=S_.indptr.astype(np.int64), sp_indices=S_.indices.astype(np.int32),
                 sp_data=S_.data.astype(np.float32), sp_shape=np.asarray(S_.shape, np.int64), kind=np.asarray(kind, np.int8),
                 r1=np.asarray(r1, np.int32), r2=np.asarray(r2, np.int32))
        rec = {"dataset": ds, "what": "E2 relation-name encodings (canonical dense + splade, documents, each string alone)",
               "graph_manifest": {"path": mp, "sha256": MANIFEST_SHA[ds]}, "n_relations": len(voc),
               "n_strings": len(strings), "kinds": {"0": "relation label", "1": "ordered pair of distinct relations (MetaQA only)"},
               "verbalisation": 'label.replace("_", " ").replace(".", " "); pairs "<r1>, <r2>"',
               "relation_vocabulary": voc, "strings": strings,
               "dense": {"model": GTE, "revision": GTE_REV, "dtype": "float16", "normalised": True, "prefix": "", "batch": 1},
               "splade": {"model": SPL, "revision": SPL_REV, "max_length": SPLADE_MAXLEN, "batch": 1,
                          "pooling": "log1p(relu(logits)).max(dim=1) (no padding at batch 1)"},
               "npz": {"path": o + ".npz", "sha256": fsha(o + ".npz"), "bytes": os.path.getsize(o + ".npz")},
               "module_sha256": me, "authorization_sha256": fsha(AUTH), "repro_sha256": fsha(REPRO_OUT), "env": ver,
               "priority": prio, "seconds": secs}
        with open(o + ".json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        print(ds, "strings", len(strings), "dense", D_.shape, "splade nnz", S_.nnz, "%.1f s" % rec["seconds"], flush=True)
    del dm, mdl
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("RELENC done in %.1f s" % (time.time() - t_all), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "PROBE":
        probe()
    elif mode == "REPRO":
        repro()
    elif mode == "RELENC":
        relenc()
    else:
        sys.exit(__doc__)
