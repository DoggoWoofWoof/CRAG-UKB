"""
Encoder swap — run L1/L3 experiments on a STRONGER encoder than MiniLM.
=======================================================================
The whole system sits on a frozen MiniLM-L6 (384-d, 2021). The #1 SIGIR reviewer
attack is "your relational offsets just compensate for a weak encoder." To answer it
we must re-run on strong encoders (bge-large / e5-large, via reencode_ukb.py) and show
our gains STACK. This helper returns the doc-embedding matrix + a query-encode fn from
EITHER the default MiniLM UKB index (subdir=None) or a re-encoded subdir, so the same
offset-training / fusion / PPR code runs unchanged on any encoder.
"""
import os
import json

import numpy as np
import faiss

from src.experiments.overlap_retrain import _reconstruct
from src.core.encoders import DenseEncoder

_ST_CACHE = {}


_TRUST = ("Alibaba-NLP/gte-Qwen2-1.5B-instruct",)


def _st(model_name):
    from sentence_transformers import SentenceTransformer
    if model_name not in _ST_CACHE:
        _ST_CACHE[model_name] = SentenceTransformer(model_name, trust_remote_code=(model_name in _TRUST))
    return _ST_CACHE[model_name]


def load_docs_and_encoder(engine, dataset, subdir=None):
    """Return (X [n,d] L2-normalized float32, encode_queries(list[str])->[m,d] normalized, tag).

    subdir=None -> default MiniLM UKB index + DenseEncoder.
    subdir='bge_large' etc. -> nodes.npy + the subdir's model (with its query instruction).
    """
    if not subdir:
        X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
        enc = DenseEncoder()

        def eq(qs):
            q = enc.encode(list(qs)).astype("float32"); faiss.normalize_L2(q); return q
        return X, eq, "minilm-L6"

    d = os.path.join("data", "ukb_storage", dataset, subdir)
    X = np.load(os.path.join(d, "nodes.npy")).astype("float32"); faiss.normalize_L2(X)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    instr = meta.get("query_instruction", ""); _m = {}

    def eq(qs):                                            # lazy: only load the (possibly 1.5B) model if actually called
        if "model" not in _m:
            _m["model"] = _st(meta["model"])
        q = _m["model"].encode([instr + x for x in qs], normalize_embeddings=True,
                               show_progress_bar=False).astype("float32")
        faiss.normalize_L2(q); return q
    return X, eq, meta["model"]


def has_subdir(dataset, subdir):
    return os.path.exists(os.path.join("data", "ukb_storage", dataset, subdir, "nodes.npy"))
