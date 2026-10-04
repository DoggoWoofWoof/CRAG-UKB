# -*- coding: utf-8 -*-
"""Modal GPU job: re-encode the dense rows lost to storage corruption in the Phase-C tree.

WHAT THIS IS FOR
    A pointer-index verification found dense rows with L2 norm 0. The zeros are on disk (mmap,
    full read and raw bytes all agree), they come in contiguous runs, and the runs are 4096-byte
    aligned and end on filesystem extent boundaries -- in metaqa_1hop the damaged span runs from
    byte 8,458,240 to exactly 16,777,216, cutting row 2753 at element 448. Encoder or batching
    faults align to row and batch boundaries, never to 4 KiB pages, so this is lost writeback on a
    98%-full volume, not a bad encode. SPLADE scanned clean on every channel; its shards are small
    compressed archives while dense shards are 122 MB flat files.

    No manifest could have caught it. manifest.json records n_items, rows_covered and a hash of the
    SOURCE jsonl, never a hash of the shard contents, and declares "complete": true on row count.

THE CONTRACT THIS MUST REPRODUCE BIT-FOR-BIT
    Alibaba-NLP/gte-Qwen2-1.5B-instruct, trust_remote_code, fp16, normalize_embeddings=True,
    dim 1536, dtype float16, max_seq_length 32768.
    DOCUMENTS GET NO PREFIX. QUERIES GET GTE_QINSTR. The prefix travels in the payload per channel
    so a docs channel can never silently receive the query instruction, which would poison the
    corpus side against the 11.4M rows it has to sit beside.

    flash_attn is deliberately NOT installed. The gte-Qwen2 remote modeling file hard-requires it at
    IMPORT time and then falls back to a non-flash attention path at RUN time. The vectors these
    repairs sit beside were produced without it, so the non-flash path IS the frozen contract.

RUN
    modal run modal_repair_encode.py
  outputs land in the volume crag-dense-repair, pulled with
    modal volume get crag-dense-repair <channel> <local dir>
"""
import modal

app = modal.App("crag-dense-repair")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.4.0", "transformers==4.44.2", "sentence-transformers==3.0.1",
                    "numpy==1.26.4", "scipy==1.14.1", "einops")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"}))
vol = modal.Volume.from_name("crag-dense-repair", create_if_missing=True)
cache = modal.Volume.from_name("crag-hf-cache", create_if_missing=True)


@app.function(image=img, gpu="A10G", timeout=60 * 60 * 5,
              volumes={"/out": vol, "/root/.cache/huggingface": cache})
def encode(payload: dict):
    import json, os, time
    import numpy as np
    t0 = time.time()
    ch = payload["channel"]                  # e.g. "2wiki_universe__docs"
    pre = payload["prefix"]
    rows = payload["rows"]
    ids = [r["id"] for r in rows]
    src_rows = [r["row"] for r in rows]
    texts = [pre + (r["text"] or "") for r in rows]
    os.makedirs("/out/%s" % ch, exist_ok=True)
    print("%s: %d rows, prefix=%r" % (ch, len(rows), pre[:40]), flush=True)

    import transformers.dynamic_module_utils as _dmu
    _dmu.check_imports = lambda *a, **k: []
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            trust_remote_code=True).half().cuda()
    m.max_seq_length = 32768

    v = m.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False,
                 convert_to_numpy=True).astype(np.float16)
    assert v.shape == (len(rows), 1536), v.shape
    # the whole point of the job: none of these may come back zero or unnormalised
    nr = np.linalg.norm(v.astype(np.float32), axis=1)
    assert np.isfinite(nr).all(), "non-finite rows"
    assert float(nr.min()) > 0.99 and float(nr.max()) < 1.01, (float(nr.min()), float(nr.max()))

    np.save("/out/%s/dense.npy" % ch, v)
    json.dump(ids, open("/out/%s/dense_ids.json" % ch, "w"), ensure_ascii=False)
    json.dump(src_rows, open("/out/%s/dense_rows.json" % ch, "w"))
    vol.commit()
    print("%s done %s l2=[%.6f,%.6f] %.0fs"
          % (ch, str(v.shape), nr.min(), nr.max(), time.time() - t0), flush=True)
    return {"channel": ch, "n": len(rows), "l2_min": round(float(nr.min()), 6),
            "l2_max": round(float(nr.max()), 6), "elapsed_s": round(time.time() - t0, 1)}


CHUNK = 12000


@app.local_entrypoint()
def main():
    import io, json
    man = json.load(io.open("scratchpad/REPAIR_WORKLIST.json", encoding="utf-8"))
    jobs = []
    for key, c in sorted(man["channels"].items()):
        rows = [json.loads(l) for l in io.open(c["file"], encoding="utf-8")]
        assert len(rows) == c["n"], (key, len(rows), c["n"])
        # chunked so no single payload carries tens of MB of text, and so the long
        # channels fan out across containers instead of serialising on one GPU
        base = "%s__%s" % (c["tree"], c["kind"])
        for k in range(0, len(rows), CHUNK):
            part = rows[k:k + CHUNK]
            jobs.append({"channel": "%s__p%03d" % (base, k // CHUNK),
                         "prefix": c["prefix"], "rows": part})
        print("queued %-34s rows=%-7d parts=%-3d prefix=%r"
              % (key, len(rows), (len(rows) + CHUNK - 1) // CHUNK, c["prefix"][:28]))
    print("TOTAL rows", sum(len(j["rows"]) for j in jobs), "in", len(jobs), "jobs")
    ok = 0
    for r in encode.map(jobs):
        ok += r["n"]
        print("DONE", r)
    print("ENCODED", ok)
