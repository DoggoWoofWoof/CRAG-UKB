# -*- coding: utf-8 -*-
"""Modal GPU job: encode the WebQSP canonical corpus under the frozen encoder contract.

WHY THIS RUNS ON A GPU AND NOT HERE
    The work order is 1.79M DISTINCT texts (2.59M nodes deduplicated by the pointer index).
    Measured local throughput is 33.8 GFLOPS effective with no CUDA device (torch 2.8.0+cpu):

        dense   2 * 1.5e9 params * ~34M tokens = 1.0e17 FLOP  ->  ~34 DAYS locally
        splade  2 * 1.1e8 params * ~34M tokens = 7.5e15 FLOP  ->  ~62 hours locally

    On one A10G the dense pass is well under an hour, and it fans out. This is the same reason
    the hotpotqa and 2wiki retrieval caches were left unbuilt: not a technical obstacle, a
    throughput one. Unlike those, WebQSP cannot be skipped -- there is no substrate without it.

THE CONTRACT THIS REPRODUCES, AND THE ONE PLACE IT DELIBERATELY DOES NOT
    dense   Alibaba-NLP/gte-Qwen2-1.5B-instruct, trust_remote_code, fp16,
            normalize_embeddings=True, dim 1536, float16, max_seq_length 32768.
    splade  naver/splade-cocondenser-ensembledistil, max_length 256,
            log(1+relu(logits)) attention-masked and max-pooled over positions, vocab 30522,
            stored CSR.

    THE MASK IS A DELIBERATE DEPARTURE FROM THE OTHER FIVE DATASETS, NOT AN OVERSIGHT.
    canonical_encode.py, which produced 99.23% of the splade rows in this package, pools over
    PADDING positions too. Measured against a batch-of-one encoding that costs up to 0.615
    absolute and drops cosine to 0.9787, and -- because HuggingFace pads to the longest text in
    the batch -- it makes a vector depend on which other texts shared its shard batch. See
    data/final_canonical/SPLADE_POOLING_FINDING.json.

    WebQSP masks because it has no legacy splade rows to stay consistent with, because its
    queries and documents are BOTH built here so they match each other, and because the
    unmasked path is not reproducible for new data in the first place -- "unmasked with batch
    size B" is a different contamination pattern than the one already on disk, so copying the
    defect would not even buy the uniformity it was copied for.

    DOCUMENTS GET NO PREFIX. These are corpus rows, so the prefix is the empty string and is
    passed explicitly rather than defaulted, so a docs channel can never silently inherit the
    query instruction and poison the corpus side against the 11.4M rows it sits beside.

    flash_attn is deliberately NOT installed. The gte-Qwen2 remote modeling file hard-requires
    it at IMPORT time and then falls back to a non-flash attention path at RUN time. Every
    vector already in the package was produced without it, so the non-flash path IS the frozen
    contract -- installing flash_attn here would silently produce a different corpus.

INPUT / OUTPUT
    Input   data/final_canonical/webqsp/encoder_inputs.jsonl, one {"row": int, "text": str}
            per DISTINCT text, in row order. Written by scratchpad/build_webqsp_canonical.py.
    Output  per-part arrays in the volume crag-webqsp-enc, each carrying the row ids it covers
            so reassembly is checked rather than assumed:
                <part>/dense.npy       float16 (n, 1536)
                <part>/splade.npz      CSR
                <part>/rows.json       the encoder_inputs rows in this part, in array order

RUN
    modal run scratchpad/modal_webqsp_encode.py                 # both models
    modal run scratchpad/modal_webqsp_encode.py --models dense  # one of them
  then pull with
    modal volume get crag-webqsp-enc / data/final_canonical/webqsp/_enc
"""
import modal

app = modal.App("crag-webqsp-enc")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.4.0", "transformers==4.44.2", "sentence-transformers==3.0.1",
                    "numpy==1.26.4", "scipy==1.14.1", "einops")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"}))
vol = modal.Volume.from_name("crag-webqsp-enc", create_if_missing=True)
cache = modal.Volume.from_name("crag-hf-cache", create_if_missing=True)

CHUNK = 12000


@app.function(image=img, gpu="A10G", timeout=60 * 60 * 5,
              volumes={"/out": vol, "/root/.cache/huggingface": cache})
def encode_dense(payload: dict):
    import json, os, time
    import numpy as np
    t0 = time.time()
    part = payload["part"]
    pre = payload["prefix"]                  # "" for documents; passed, never defaulted
    rows = payload["rows"]
    ids = [r["row"] for r in rows]
    texts = [pre + (r["text"] or "") for r in rows]
    os.makedirs("/out/%s" % part, exist_ok=True)

    import transformers.dynamic_module_utils as _dmu
    _dmu.check_imports = lambda *a, **k: []
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            trust_remote_code=True).half().cuda()
    m.max_seq_length = 32768

    # batch 32 is canonical_encode.py's default (:244), i.e. what the other five corpora were
    # encoded at. Padding is attention-masked in this model, so batch size is not supposed to
    # change the result -- but "not supposed to" is exactly the assumption that produced the
    # splade pooling finding, so the known value is used rather than a convenient one.
    v = m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False,
                 convert_to_numpy=True).astype(np.float16)
    assert v.shape == (len(rows), 1536), v.shape
    nr = np.linalg.norm(v.astype(np.float32), axis=1)
    assert np.isfinite(nr).all(), "non-finite rows"
    assert float(nr.min()) > 0.99 and float(nr.max()) < 1.01, (float(nr.min()), float(nr.max()))

    np.save("/out/%s/dense.npy" % part, v)
    json.dump(ids, open("/out/%s/rows.json" % part, "w"))
    vol.commit()
    print("%s dense %s l2=[%.6f,%.6f] %.0fs"
          % (part, str(v.shape), nr.min(), nr.max(), time.time() - t0), flush=True)
    return {"part": part, "model": "dense", "n": len(rows),
            "l2_min": round(float(nr.min()), 6), "elapsed_s": round(time.time() - t0, 1)}


@app.function(image=img, gpu="A10G", timeout=60 * 60 * 5,
              volumes={"/out": vol, "/root/.cache/huggingface": cache})
def encode_splade(payload: dict):
    import json, os, time
    import numpy as np
    import scipy.sparse as sp
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    t0 = time.time()
    part = payload["part"]
    rows = payload["rows"]
    ids = [r["row"] for r in rows]
    # splade takes no prefix on either side; the caller must not send one, and sending one
    # would silently produce a query channel the corpus was never scored against.
    if payload.get("prefix"):
        raise RuntimeError("splade takes no prefix, got %r" % payload["prefix"][:40])
    texts = [r["text"] or "" for r in rows]
    os.makedirs("/out/%s" % part, exist_ok=True)

    name = "naver/splade-cocondenser-ensembledistil"
    tok = AutoTokenizer.from_pretrained(name)
    # fp32, NOT half: canonical_encode.py:201 loads this model at full precision
    # (unlike the dense model at :176, which IS .half()). splade is 110M params, so
    # matching the existing dtype costs almost nothing and removes one more axis on
    # which dataset 6 could differ from the other five for no reason.
    mdl = AutoModelForMaskedLM.from_pretrained(name).cuda().eval()

    out = []
    B = 64
    with torch.no_grad():
        for i in range(0, len(texts), B):
            b = tok(texts[i:i + B], padding=True, truncation=True, max_length=256,
                    return_tensors="pt")
            b = {k: v.cuda() for k, v in b.items()}
            logits = mdl(**b).logits                       # (b, L, V)
            w = torch.log1p(torch.relu(logits))
            # attention-masked max-pool over positions: padding must not contribute
            w = w.masked_fill(~b["attention_mask"].bool().unsqueeze(-1), 0.0)
            v = w.max(dim=1).values.cpu().numpy()
            out.append(sp.csr_matrix(v))
    M = sp.vstack(out).tocsr()
    assert M.shape == (len(rows), 30522), M.shape
    assert M.data.min() >= 0.0, "negative splade weight"
    empty = int((np.diff(M.indptr) == 0).sum())

    sp.save_npz("/out/%s/splade.npz" % part, M)
    json.dump(ids, open("/out/%s/rows.json" % part, "w"))
    vol.commit()
    print("%s splade %s nnz=%d empty=%d %.0fs"
          % (part, str(M.shape), M.nnz, empty, time.time() - t0), flush=True)
    return {"part": part, "model": "splade", "n": len(rows), "nnz": int(M.nnz),
            "empty_rows": empty, "elapsed_s": round(time.time() - t0, 1)}


# Verbatim from canonical_encode.py:30. The dense QUERY channel takes this instruction and the
# document channel takes nothing; that asymmetry is what the instruct model is for, and a query
# channel encoded without it is not comparable to the corpus it gets scored against. SPLADE
# takes no prefix on either side.
# chr(10) rather than an escape: this string has been mangled in transit once already,
# and it is verified against canonical_encode.py below rather than eyeballed.
GTE_QINSTR = ("Instruct: Given a web search query, retrieve relevant passages that answer the "
              "query" + chr(10) + "Query: ")


@app.local_entrypoint()
def main(models: str = "dense,splade", limit: int = 0, kind: str = "docs"):
    """kind=docs encodes the corpus; kind=queries encodes the 4,737 canonical queries.

    limit>0 encodes only the first `limit` PARTS -- a smoke test before the full fan-out. The
    parts it produces are real and land in the volume under the same names the full run uses,
    so a smoke test is not thrown away: rerunning without --limit refills the rest.
    """
    import io, json
    if kind not in ("docs", "queries"):
        raise SystemExit("kind must be docs or queries")
    src = ("data/final_canonical/webqsp/encoder_inputs.jsonl" if kind == "docs"
           else "data/final_canonical/webqsp/query_inputs.jsonl")
    rows = [json.loads(l) for l in io.open(src, encoding="utf-8")]
    # row must be exactly 0..N-1 in file order. For docs the pointer index maps a node to a
    # position in THIS list; for queries the position IS the query identity, fixed by
    # queries/pointer_index/query_ids.json. If either drifts, vectors attach to the wrong
    # thing with every shape still agreeing, so it is asserted rather than trusted.
    assert [r["row"] for r in rows] == list(range(len(rows))), "%s rows are not 0..N-1" % src
    print("%s: %d texts" % (kind, len(rows)))
    for model in models.split(","):
        fn = {"dense": encode_dense, "splade": encode_splade}[model]
        # the prefix is chosen HERE and passed explicitly, never defaulted inside the worker.
        prefix = GTE_QINSTR if (kind == "queries" and model == "dense") else ""
        tag = model if kind == "docs" else "%s_q" % model
        jobs = [{"part": "%s__p%04d" % (tag, k // CHUNK), "prefix": prefix,
                 "rows": rows[k:k + CHUNK]}
                for k in range(0, len(rows), CHUNK)]
        if limit:
            jobs = jobs[:limit]
        print("%s/%s: %d parts of <=%d  prefix=%r%s"
              % (kind, model, len(jobs), CHUNK, prefix[:24],
                 "  [SMOKE TEST]" if limit else ""))
        done = 0
        for r in fn.map(jobs):
            done += r["n"]
            print("DONE", r)
        print("%s/%s ENCODED %d" % (kind, model, done))
