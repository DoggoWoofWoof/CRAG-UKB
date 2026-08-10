"""
Offline query-selectivity signal + sparse/dense routing.
========================================================
Jigsaw's biggest transferable lesson (playbook §3.2): there is no universal
retriever. Lexical/exact retrieval is near-oracle when a query's discriminative
tokens are RARE (high selectivity); dense/semantic retrieval wins when they are
COMMON/paraphrastic (low selectivity). The right move is not to blindly
hybrid-fuse but to ROUTE by an offline statistic computable from the query and
the corpus index alone — never from the gold answer.

Jigsaw's statistic was "median partitions-per-label". The RAG analog is term
rarity: query-term IDF / document-frequency, read straight off the BM25 index.

This module is deliberately model-free and side-effect-free so it can be used
both inside the Level-1 benchmark (as a `selectivity_route` method) and at
runtime. Thresholds default to reasonable values on normalized IDF in [0, 1] but
SHOULD be calibrated per corpus (sweep them against FullCov / worst-gold rank).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


def _corpus_size(bm25) -> Optional[int]:
    n = getattr(bm25, "corpus_size", None)
    return int(n) if n else None


def query_idf_stats(bm25, query: str) -> Dict[str, float]:
    """IDF-based rarity statistics for a query, read off the BM25 index.

    Tokenization is `query.lower().split()` to MATCH the corpus vocabulary,
    which was built with `content.lower().split()` (indexers.build_*). Using a
    case-preserving split would drop every capitalized proper noun as OOV —
    exactly the rare, discriminative terms the router must detect — so the
    lexical arm would never fire. IDF values are normalized so the statistic
    lands (mostly) in [0, 1]: by log(N+1) when corpus_size is known, else by the
    max vocabulary IDF. Out-of-vocabulary tokens are ignored for the average but
    counted in coverage.
    """
    idf_map = getattr(bm25, "idf", None) or {}
    tokens = [t for t in query.lower().split() if t]
    n_terms = len(tokens)

    n = _corpus_size(bm25)
    if n and n > 1:
        norm = math.log(n + 1.0)
    else:
        # Fallback when corpus_size is unavailable: normalize by the largest
        # vocabulary IDF so the rarest term maps to ~1.0 (keeps thresholds sane).
        vocab_idfs = [float(v) for v in idf_map.values()] if idf_map else []
        norm = max(vocab_idfs) if vocab_idfs else 1.0
        norm = norm if norm > 0 else 1.0

    idfs: List[float] = []
    for tok in tokens:
        if tok in idf_map:
            # BM25Okapi IDF can be slightly negative for ultra-common terms;
            # clamp at 0 so "very common" reads as minimal rarity, not negative.
            idfs.append(max(0.0, float(idf_map[tok])) / norm)

    n_covered = len(idfs)
    if not idfs:
        return {
            "mean_idf": 0.0, "median_idf": 0.0, "max_idf": 0.0, "min_idf": 0.0,
            "n_terms": float(n_terms), "coverage": 0.0,
        }

    idfs_sorted = sorted(idfs)
    mid = len(idfs_sorted) // 2
    median = (
        idfs_sorted[mid]
        if len(idfs_sorted) % 2 == 1
        else 0.5 * (idfs_sorted[mid - 1] + idfs_sorted[mid])
    )
    return {
        "mean_idf": sum(idfs) / len(idfs),
        "median_idf": median,
        "max_idf": max(idfs),
        "min_idf": min(idfs),
        "n_terms": float(n_terms),
        "coverage": n_covered / n_terms if n_terms else 0.0,
    }


def route_from_stats(
    stats: Dict[str, float],
    high_threshold: float = 0.55,
    low_threshold: float = 0.35,
    statistic: str = "median_idf",
) -> str:
    """Decide the retriever arm from a selectivity statistic.

    Returns "lexical" (route to BM25/exact), "dense" (route to embedding
    retrieval), or "hybrid" (fuse — the query genuinely ties). Uses the
    normalized `statistic` (median IDF by default): rare terms -> lexical,
    common terms -> dense, in-between -> hybrid.
    """
    score = float(stats.get(statistic, 0.0))
    if score >= high_threshold:
        return "lexical"
    if score <= low_threshold:
        return "dense"
    return "hybrid"


class QuerySelectivityRouter:
    """Convenience wrapper binding a BM25 index + thresholds."""

    def __init__(self, bm25, high_threshold: float = 0.55,
                 low_threshold: float = 0.35, statistic: str = "median_idf"):
        self.bm25 = bm25
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.statistic = statistic

    def stats(self, query: str) -> Dict[str, float]:
        return query_idf_stats(self.bm25, query)

    def selectivity_score(self, query: str) -> float:
        return float(self.stats(query).get(self.statistic, 0.0))

    def route(self, query: str) -> str:
        return route_from_stats(
            self.stats(query),
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
            statistic=self.statistic,
        )
