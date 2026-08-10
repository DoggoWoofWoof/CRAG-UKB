"""Pre-encode a dataset's documents with SPLADE (naver/splade-cocondenser-ensembledistil) into
`data/ukb_storage/{dataset}/splade_doc_embs.pkl` = {"matrix": CSR (n_docs, vocab), "id_to_idx":
{node_id: row}}. Rows align with `CoreEngine(dataset).nodes` (the doc order = substrate X order),
so L2 can fuse a SPLADE lexical axis WITHOUT touching the frozen gte vectors/partitions.
Mirrors the encode in evaluation/level2.py (log(1+relu(logits)) max-pooled)."""
import os
import pickle
import logging

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
log = logging.getLogger("experiments.splade_encode")

SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


def encode(dataset, batch=64, max_len=256):
    import torch
    import scipy.sparse
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from src.core.engine import CoreEngine

    eng = CoreEngine(source=dataset)
    docs = eng.nodes                                           # doc-only, aligned with the substrate X
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(SPLADE_MODEL)
    mdl = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL).to(device).eval()
    id_to_idx = {n.node_id: i for i, n in enumerate(docs)}
    rows, cols, data = [], [], []
    log.info(f"[SPLADE] encoding {len(docs)} docs for {dataset} (batch={batch})")
    for s in range(0, len(docs), batch):
        b = docs[s:s + batch]
        inp = tok([n.content[:1024] for n in b], return_tensors="pt",
                  padding=True, truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            lg = mdl(**inp).logits
            sp = torch.max(torch.log(1 + torch.relu(lg)) * inp.attention_mask.unsqueeze(-1), dim=1).values.cpu()
        del lg, inp
        for i, v in enumerate(sp):
            nz = v.nonzero(as_tuple=True)[0]
            rows.extend([s + i] * len(nz)); cols.extend(nz.numpy().tolist()); data.extend(v[nz].numpy().tolist())
        if s % 12800 == 0:
            log.info(f"  {s}/{len(docs)}")
    mat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(docs), mdl.config.vocab_size), dtype=np.float32)
    path = os.path.join("data", "ukb_storage", dataset, "splade_doc_embs.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"matrix": mat, "id_to_idx": id_to_idx}, f)
    log.info(f"[SPLADE] saved {path}: matrix {mat.shape}, nnz={mat.nnz}")
    return path


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="splade-encode")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args(argv)
    for d in a.datasets:
        encode(d, batch=a.batch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
