# Pipeline And Substrate Construction

This package maps heterogeneous datasets into the shared `StandardNode` schema,
builds label-free document graphs, and persists per-dataset retrieval views.

## StandardNode

`standardizer.py` defines:

- `node_id`: stable record identifier;
- `content`: text encoded and supplied to readers;
- `metadata`: source, node type, title, and provenance;
- `neighbors`: structural node links.

Question nodes retain links to gold documents/entities as labels for training
and evaluation. They are excluded from FAISS, BM25, graph, partition, and
centroid indexes.

## Loaders

`loaders.py` contains loaders for SQuAD, MuSiQue, 2WikiMultiHopQA, HotpotQA, and
MetaQA. The clean multi-hop loaders do **not** connect co-supporting gold
documents to each other. Earlier gold-bridge behavior was identified as label
leakage and removed.

MetaQA document nodes are entities connected by the source KB triples. MuSiQue
titles are recovered from FlashRAG's stringified decomposition paragraphs so
that legitimate title/entity links can be built.

## Clean Rebuild

`build_clean.py` creates deduplicated, label-free corpora:

- document/entity structural edges come only from corpus content or the source
  KB;
- question-to-gold edges remain only on question nodes;
- no document-to-question backedges are retained;
- no co-gold bridge is constructed from QA labels.

`build_clean_index.py` reuses existing document embeddings when content hashes
match, then builds BM25, graph, partition, and centroid artifacts.

## Edge Types

The graph has two edge families:

1. **Structural:** title mentions, shared-title structure, entity/KB relations,
   or other corpus-derived links.
2. **Synthetic:** dense-kNN edges added during indexing for connectivity.

`CoreEngine._attach_synthetic_neighbor_metadata()` reconstructs the synthetic
edge tags by comparing `graph.pt` with the structural links in the clean master
file. Level 3 and SRW ablations rely on this distinction.

## Current Dataset Caveats

- `2wiki_clean` and `hotpotqa_clean` contain meaningful distractor pools.
- `musique_clean`, `squad_clean`, and `metaqa` are effectively all-gold pools
  in the local source data, so absolute open-corpus recall is optimistic.
- Current experiment helpers reshuffle questions into deterministic internal
  70/20/10 splits. Publication runs must preserve official boundaries.
- The validation suite currently performs substrate leak checks only for
  MuSiQue and SQuAD; extend it to all paper datasets.

Run the existing validation:

```bash
python -m tests.test_validation --substrate
```
