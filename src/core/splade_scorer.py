"""
Lazy SPLADE sparse scorer for Level-1 partition voting.
=======================================================
The selectivity router's "lexical arm" is a swappable slot. BM25 is the default
(zero-cost, always available, the canonical exact-lexical baseline for the
lexical-vs-dense crossover). SPLADE is a *learned* sparse retriever that usually
scores higher (on CRAG's own Level-2 SQuAD: SPLADE R@1 71.9 vs BM25 49.2) but
needs a GPU model + a per-dataset document matrix.

This reuses the exact SPLADE document matrix cached by run_level2_eval.py at
`data/ukb_storage/{dataset}/splade_doc_embs.pkl` (keys: `matrix` = scipy CSR
(n_docs, vocab), `id_to_idx` = {node_id: row}), and the same query encoding
(`naver/splade-cocondenser-ensembledistil`, log(1+relu(logits)) max-pooled).

Availability is gated: `available()` is False when the doc matrix hasn't been
generated for a dataset (currently only squad/metaqa have it; musique/2wiki need
the Level-2 SPLADE pre-encode first). Callers should skip SPLADE methods when
`available()` is False rather than crash. All heavy imports (torch/transformers)
are deferred to first use so importing this module is cheap.
"""
import os
import pickle
import logging

import numpy as np

log = logging.getLogger(__name__)

SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"

_MODEL_CACHE = {}   # (model_name, device) -> (tokenizer, model); load the query encoder ONCE per process


class SpladeScorer:
    def __init__(self, dataset: str, device=None):
        self.dataset = dataset
        self.cache_path = f"data/ukb_storage/{dataset}/splade_doc_embs.pkl"
        self._device = device
        self._matrix = None       # scipy CSR (n_docs, vocab)
        self._idx_to_id = None    # row -> node_id
        self._model = None
        self._tokenizer = None

    def available(self) -> bool:
        """True iff the pre-encoded SPLADE document matrix exists on disk."""
        return os.path.exists(self.cache_path)

    def _ensure_matrix(self):
        if self._matrix is None:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self._matrix = data["matrix"]
            self._idx_to_id = {row: nid for nid, row in data["id_to_idx"].items()}
            log.info(f"[SPLADE] loaded doc matrix {self._matrix.shape} for {self.dataset}")

    def _ensure_model(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            key = (SPLADE_MODEL, str(self._device))
            if key not in _MODEL_CACHE:                        # load ONCE, share across all datasets
                tok = AutoTokenizer.from_pretrained(SPLADE_MODEL)
                mdl = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL).to(self._device)
                mdl.eval()
                _MODEL_CACHE[key] = (tok, mdl)
                log.info(f"[SPLADE] query encoder loaded on {self._device} (cached)")
            self._tokenizer, self._model = _MODEL_CACHE[key]

    def encode_query(self, query: str) -> np.ndarray:
        import torch
        self._ensure_model()
        inputs = self._tokenizer(
            [query], return_tensors="pt", padding=True, truncation=True, max_length=64
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
            relu_log = torch.log(1 + torch.relu(logits))
            mask = inputs["attention_mask"].unsqueeze(-1)
            q = torch.max(relu_log * mask, dim=1).values.cpu().numpy()[0]
        return q  # (vocab,)

    def top_doc_ids(self, query: str, k: int = 100):
        """Return the node_ids of the top-k documents by SPLADE sparse dot product."""
        self._ensure_matrix()
        q = self.encode_query(query)
        scores = self._matrix.dot(q)  # (n_docs,)
        if k >= scores.shape[0]:
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, k)[:k]
            order = top[np.argsort(-scores[top])]
        return [self._idx_to_id[int(i)] for i in order]
