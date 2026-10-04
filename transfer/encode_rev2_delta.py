# -*- coding: utf-8 -*-
"""Re-encode ONLY the rows whose canonical text changed in TEXTUALIZATION_REV2.

Measured staleness (transfer/TRANSFER_MANIFEST.md):
  2wiki    87,765 rows   hotpotqa 94 rows        total 87,859 of 11,223,176 doc embeddings
Every one of them encoded the EMPTY STRING; REV2 gave them their article title as text. So the
stale vectors are not merely drifted, they are embeddings of "" -- they must not be shipped.

Output is a PATCH, not a rebuild: base shards stay untouched and 99.2% of the 25 GB of existing
embeddings is reused verbatim. Apply by overriding the patched node_ids after loading the base.

  transfer/rev2_patch/<ds>/dense.npy      float16 [n, 1536], row i belongs to ids[i]
  transfer/rev2_patch/<ds>/splade.npz     CSR over the 30,522-dim SPLADE vocab
  transfer/rev2_patch/<ds>/ids.json       canonical_v1 node_ids, patch row order
  transfer/rev2_patch/<ds>/status.json    encoder config actually used

Encoder config is taken from src/experiments/canonical_encode.py verbatim so the patched vectors
live in the same space as the base shards: gte-Qwen2-1.5B-instruct, FP16, normalize, NO prefix on
documents (the instruction prefix is a QUERY-side thing); SPLADE log(1+relu) max-pool, max_len 256.
"""
import os, sys, io, json, argparse
import numpy as np

GTE = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
GTE_DIM = 1536
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
OUT = "transfer/rev2_patch"


def changed_nodes(ds):
    """(node_id, text) for every node REV2 changed -- read from the canonical corpus itself."""
    if ds == "2wiki":
        ids = set(x.strip() for x in
                  io.open("data/final_canonical/2wiki/REV2_CHANGED_NODE_IDS.txt", encoding="utf-8")
                  if x.strip())
    elif ds == "hotpotqa":
        raw = json.load(io.open("data/final_canonical/_work/hotpot_phase_c_empty_encoder_inputs.json",
                                encoding="utf-8"))
        ids = {"hotpotqa:c" + x.split("_", 1)[1] for x in raw}
    else:
        raise SystemExit("no REV2 delta for %s -- its embeddings are already current" % ds)
    got = []
    with io.open("data/final_canonical/%s/nodes.jsonl" % ds, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["node_id"] in ids:
                got.append((r["node_id"], r["text"]))
    assert len(got) == len(ids), "%s: %d of %d changed nodes found" % (ds, len(got), len(ids))
    assert all(t.strip() for _, t in got), "a REV2 node still has empty text -- refusing to encode"
    return got


def dense(texts, batch):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(GTE, trust_remote_code=True, device="cpu")
    m.max_seq_length = 512
    v = m.encode(texts, batch_size=batch, normalize_embeddings=True,
                 show_progress_bar=True).astype("float16")
    assert v.shape[1] == GTE_DIM and np.isfinite(v).all()
    return v, {"encoder": GTE, "dim": GTE_DIM, "dtype": "float16", "normalized": True,
               "document_prefix": None}


def splade(texts, batch):
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from scipy import sparse
    tok = AutoTokenizer.from_pretrained(SPLADE_MODEL)
    mdl = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL).eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = [str(t) for t in texts[i:i + batch]]
            e = tok(b, return_tensors="pt", truncation=True, max_length=256, padding=True)
            lg = mdl(**e).logits
            w = torch.log1p(torch.relu(lg)) * e["attention_mask"].unsqueeze(-1)
            rows.append(sparse.csr_matrix(w.max(dim=1).values.numpy()))
            if i % (batch * 50) == 0:
                print("  splade %d/%d" % (i, len(texts)), flush=True)
    M = sparse.vstack(rows).tocsr()
    return M, {"encoder": SPLADE_MODEL, "vocab": M.shape[1], "max_len": 256}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["2wiki", "hotpotqa"])
    ap.add_argument("--model", required=True, choices=["dense", "splade", "both"])
    ap.add_argument("--batch", type=int, default=8)   # CPU-safe; raise to 64+ on a GPU
    a = ap.parse_args()

    got = changed_nodes(a.dataset)
    ids = [i for i, _ in got]
    txt = [t for _, t in got]
    d = os.path.join(OUT, a.dataset)
    os.makedirs(d, exist_ok=True)
    json.dump(ids, io.open(os.path.join(d, "ids.json"), "w", encoding="utf-8"))
    print("%s: %d rows to re-encode" % (a.dataset, len(txt)), flush=True)

    st = {}
    if a.model in ("dense", "both"):
        v, meta = dense(txt, a.batch)
        np.save(os.path.join(d, "dense.npy"), v)
        st["dense"] = meta
    if a.model in ("splade", "both"):
        from scipy import sparse
        M, meta = splade(txt, a.batch)
        sparse.save_npz(os.path.join(d, "splade.npz"), M)
        st["splade"] = meta
    st.update({"dataset": a.dataset, "rows": len(ids),
               "reason": "TEXTUALIZATION_REV2: these rows had encoded the empty string"})
    p = os.path.join(d, "status.json")
    old = json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else {}
    old.update(st)
    json.dump(old, io.open(p, "w", encoding="utf-8"), indent=1)
    print("wrote", d, flush=True)
