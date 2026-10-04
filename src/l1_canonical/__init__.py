"""L1 on the canonical substrates (data/final_canonical/).

    adapter.py           CanonicalDataset -- the one manifest-driven, zero-copy view of a frozen
                         dataset in canonical POSITION space (queries, gold, eval population,
                         node/query embeddings, dense/splade top-k, struct/knn/ner edges, keysets)
    validate_adapter.py  step 3 of the return-to-L1 plan: prove the adapter on all six
    hypergraph.py        H4_SPLIT_PRESERVE over STRUCT+KNN closed neighbourhoods (frozen rule)
    partition.py         Mt-KaHyPar (frozen contract) under WSL -> position -> block id
    replay_cache.py      the frozen L1 traversal/retrieval cache, canonical positions
    l1_eval.py           L1 BASE / SAFE (B6_S4_F6_Ms64_Mr32) replay on a canonical cache
    external_kit.py      pack / import the external >=250 GB partition lane (webqsp, hotpotqa, 2wiki)

Nothing here writes into data/final_canonical/. Derived L1-owned artefacts live under
data/l1_canonical/<ds>/ and every one of them is pinned to the DATASET.json RECORD_SHA256 it was
built from.
"""
