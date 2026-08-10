from rank_bm25 import BM25Okapi
import numpy as np

from src.experiments.sota_baselines import (
    _InvertedBM25,
    _evaluate,
    _rrf,
)


def test_inverted_bm25_matches_rank_bm25_scores():
    corpus = [
        "alpha beta beta".split(),
        "alpha gamma".split(),
        "delta epsilon".split(),
        "beta gamma".split(),
    ]
    bm25 = BM25Okapi(corpus)
    scorer = _InvertedBM25(bm25, ["alpha beta", "delta"])

    for query in ("alpha beta", "delta"):
        scores = bm25.get_scores(query.split())
        expected = np.lexsort((np.arange(len(corpus)), -scores)).tolist()
        assert scorer.rank(query, k=len(corpus)) == expected


def test_rrf_deduplicates_and_respects_weights():
    dense = [[0, 1, 2]]
    lexical = [[2, 1, 3]]

    equal = _rrf([dense, lexical], [1.0, 1.0], k=4)
    lexical_heavy = _rrf([dense, lexical], [1.0, 4.0], k=4)

    assert len(equal[0]) == len(set(equal[0])) == 4
    assert lexical_heavy[0][0] == 2


def test_full_coverage_requires_every_positive():
    summary, rows = _evaluate([[4, 1, 3, 2]], [[1, 2]])

    assert rows[0]["fullcov"]["2"] == 0
    assert rows[0]["fullcov"]["5"] == 1
    assert summary["recall"]["2"] == 50.0
