# `src/ingestion/` — Offline Knowledge Base Construction

Responsible for the **one-time, offline preprocessing** pipeline that transforms raw datasets (SQuAD v2, WebQSP, Cora) into the Unified Knowledge Base (UKB) stored in `data/kg_store/`. These scripts are computationally expensive and are **never run at query time**.

> **Rule:** Anything that takes longer than 1 second per query belongs here, not in a retriever.

---

## Files

### `kb_builder.py` — Master KB Constructor

The main entry point for building the UKB. Orchestrates all three data sources and handles entity linking between them.

```bash
python -m src.ingestion.kb_builder \
  --squad  data/raw/squad_v2.json \
  --webqsp data/raw/webqsp.json \
  --cora   data/raw/cora/ \
  --output data/kg_store/
```

**Processing Pipeline:**

```
SQuAD v2 Contexts ──► Chunk (~200 tokens each) ──► Document Nodes
                                                    │
                                                    ▼ NEXT edges (sequential)
                                                    
WebQSP / Wikidata ──► Entity extraction ──────────► Entity Nodes
                                                    │
                                                    ▼ SUBJECT_OF / OBJECT_OF edges (triples)
                                                    
Cora Papers ────────► Abstract + Title ───────────► Paper Nodes + Concept Nodes
                                                    │
                                                    ▼ CITES + CATEGORY edges

NER Pass (spaCy) ──► Entity Linking ─────────────► MENTIONS edges (SQuAD ↔ WebQSP)
```

**Output files:** `data/processed/nodes.json`, `data/processed/edges.json`, `data/kg_store/graph.pt`, `data/kg_store/node_metadata.json`

---

### `cora_adapter.py` — Cora Dataset Adapter

Converts the Cora citation graph (a standard GNN benchmark dataset) into the unified node/edge schema used by `kb_builder.py`.

```bash
python -m src.ingestion.cora_adapter --output data/raw/cora/
```

**Cora in Context:**
Cora contains 2,708 scientific papers classified into 7 categories, with 5,429 citation edges. In the UKB, Cora serves three roles:

| Role | How Cora Is Used |
|---|---|
| **Partition Validation** | Small, well-studied graph used to validate METIS edge-cut ratios before applying to the full UKB |
| **Graph Traversal Testing** | Citation paths are clean multi-hop chains ideal for testing the Graph-RAG BFS logic |
| **InfoNCE Training data** | Paper-to-category labels provide natural positive pairs for contrastive training |

**Schema Mapping:**

| Cora Concept | Unified Schema |
|---|---|
| Paper node | `{node_id, type="paper", text_content="<title> <abstract>"}` |
| Paper label (7 classes) | `{node_id, type="concept", text_content="<class_name>"}` |
| Citation edge | `{src_id, dst_id, relation_type="CITES"}` |
| Label assignment | `{src_id, dst_id, relation_type="CATEGORY"}` |

---

### `partitioner.py` — METIS Graph Partitioner

Applies METIS density-based partitioning to the unified graph topology. Outputs a `partition_map.json` embedding every node to a partition ID, and triggers `embedder.py` to compute partition centroids.

```bash
python -m src.ingestion.partitioner \
  --input data/kg_store/graph.pt \
  --n-partitions 50 \
  --validate        # Runs on Cora first to check edge-cut ratio
```

**Key Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--n-partitions` | `50` | Number of METIS partitions. Each partition targets ≤ 200 nodes |
| `--max-partition-size` | `200` | Hard cap. Nodes exceeding this trigger a recursive split |
| `--validate` | `False` | If True, runs METIS on Cora first and asserts edge-cut < 15% |

**Validation Criteria (Cora):**
- Edge-cut ratio: **< 15%** (METIS typically achieves 8–12% on Cora)
- Partition size standard deviation: **< 10%** of mean partition size

**Output:** `data/kg_store/partition_map.json` — `{node_id: partition_id}` for all ~150k nodes.

---

### `embedder.py` — Node & Centroid Embedder

Generates 768-dimensional dense embeddings for all nodes using `sentence-transformers/all-mpnet-base-v2` (or configurable via `configs/unified.yaml`). Saves two FAISS indices.

```bash
python -m src.ingestion.embedder \
  --nodes data/processed/nodes.json \
  --partition-map data/kg_store/partition_map.json \
  --output data/kg_store/
```

**Process:**
1. Embed each node's `text_content` → 768-dim vector
2. Save all vectors to `data/kg_store/nodes.index` (FAISS Flat L2)
3. For each partition, compute the **mean** of all its node embeddings → centroid vector
4. Save all centroids to `data/kg_store/centroids.index` (FAISS Flat IP)

> **Why two different FAISS index types?**  
> `nodes.index` uses L2 (Euclidean) for symmetric similarity. `centroids.index` uses inner product (IP) to match the cosine similarity objective of the InfoNCE bi-encoder.

---

## Execution Order

These scripts **must** be run in order. Later scripts depend on the outputs of earlier ones.

```
1. cora_adapter.py    → data/raw/cora/
2. kb_builder.py      → data/processed/ + data/kg_store/graph.pt
3. partitioner.py     → data/kg_store/partition_map.json
4. embedder.py        → data/kg_store/nodes.index + centroids.index
```
