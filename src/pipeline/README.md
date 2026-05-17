# Pipeline Standardization Methodology
This directory maps raw heterogeneous datasets (SQuAD, MuSiQue, 2Wiki) into a single homogenous `StandardNode` architecture.

## Director Structure & File Mechanics

### `standardizer.py`
The architectural blueprint for the Unified Knowledge Base schema.
*   **Purpose**: Defines the `StandardNode` python class, guaranteeing that regardless of origin, every record has an identical footprint.
*   **Mechanics**:
    *   `node_id`: (str) Uniquely identifies the record (e.g., `squad_0`, `musique_doc_3`, `2wiki_q_12`).
    *   `content`: (str) The raw text payload. Used by Encoders and LLM Generators identically.
    *   `metadata`: (dict) Holds lineage (e.g., `source: "squad"`, `type: "document"`, `is_impossible: True`).
    *   `neighbors`: (list[str]) Maintains direct topological edge references to other `node_id`s.

### `loaders.py`
The Data Parser & Normalization Utility.
*   **Purpose**: To extract specific records out of arbitrarily nested JSON formats and map them sequentially into `StandardNode` arrays.
*   **Mechanics**:
    *   `load_squad`: Parses SQuAD v2. Iterates over paragraphs, dumping contexts into `document` nodes. Simultaneously extracts all questions (`qas`), converting them into distinct `question` nodes. Vitally, it appends the paragraph `node_id` into the question's `neighbors` list, creating a bi-directional Semantic-to-Source edge natively.
    *   `load_musique`: Parses MuSiQue JSONL multi-hop data. Extracts all paragraphs as `document` nodes, and links the `question` node to any document marked with `is_supporting: True`.
    *   `load_2wiki`: Parses 2WikiMultiHopQA JSONL. Groups sentences by Wikipedia title into `document` nodes. Links the `question` node to article titles listed in `supporting_facts`.
    *   `build_unified_dataset`: Executes the loaders successively, writing the gigantic aggregated array directly to `data/processed/master_nodes.json`.
