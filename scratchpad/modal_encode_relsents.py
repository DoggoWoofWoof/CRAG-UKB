"""Modal job: encode unique RELATION connecting sentences with the canonical gte-Qwen2-1.5B-instruct.

CANONICAL semantics (matches nodes.npy / meta.json doc_prefix=""):
  - connecting sentences are PASSAGES -> encoded RAW (NO query instruction, NO prefix).
  - normalize_embeddings=True (cosine-ready), last-token pooling (gte-Qwen2 default via ST).
The query side is NOT encoded here (reuses queries_all.npy = eq(query)).

Run (background):
  modal run scratchpad/modal_encode_relsents.py --inp scratchpad/_qwen_sentences.pkl --outp scratchpad/_qwen_sent_emb.npy
"""
import modal

app = modal.App("crag-relsent-encode")

image = (
    modal.Image.micromamba(python_version="3.11")
    .env({"CONDA_OVERRIDE_CUDA": "12.1", "CUDA_HOME": "/opt/conda", "TORCH_CUDA_ARCH_LIST": "8.6"})
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("torch==2.2.1", "numpy<2.0")
    .pip_install("sentence-transformers<3.0", "transformers==4.44.2")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1%2Bcu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
)


@app.function(image=image, gpu="A10G", timeout=3600)
def encode_sentences(sentences, model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct"):
    import os, numpy as np, time, torch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    # fp16 weights + short max_seq_length (connecting sentences are 1-2 sentences) to avoid the 32k-ctx OOM
    m = SentenceTransformer(model_name, trust_remote_code=True)
    m.max_seq_length = 256
    m = m.half()                                          # fp16 weights (ST<3.0 has no model_kwargs)
    print(f"model loaded fp16 in {time.time()-t0:.1f}s; max_seq_length={m.max_seq_length}; "
          f"encoding {len(sentences)} sentences RAW (no instruction)")
    t1 = time.time()
    emb = m.encode(sentences, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    print(f"encoded {emb.shape} in {time.time()-t1:.1f}s ({1000*(time.time()-t1)/max(1,len(sentences)):.1f} ms/sent)")
    return emb


@app.local_entrypoint()
def main(inp: str, outp: str):
    import pickle, numpy as np
    obj = pickle.load(open(inp, "rb"))
    sents = obj["sentences"]
    print(f"[local] {len(sents)} unique sentences -> Modal gte-Qwen2 (raw/document semantics)")
    emb = encode_sentences.remote(sents)
    np.save(outp, emb)
    print(f"[local] SAVED {outp} shape={emb.shape} dtype={emb.dtype}")
