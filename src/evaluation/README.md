# Evaluation and Benchmarking Architecture
This directory houses the logic to dynamically test and mathematically prove the efficacy of the 5 RAG Strategies. 

## Directory Structure & File Mechanics

### `ground_truth.py` (Deprecated in v4)
Originally generated synthetic multi-hop benchmarks via random graph walks. Superseded by exact native ground-truth matching from the dataloaders.

### `benchmark_partition_selection.py`
The definitive Evaluation Matrix Compiler for Graph/Vector routing.
*   **Purpose**: To systematically evaluate the retrieval accuracy of GNNs vs MLPs vs dense baselines, strictly respecting identical experimental splits to prevent data leakage.
*   **Mechanics**:
    1.  **Deterministic Splits**: Uses the exact `70/20/10` seed logic (`get_split_pairs()`) from training to separate `Train`, `Val`, and `Test` queries.
    2.  **Dataset Extraction**: For each dataset (SQuAD, MuSiQue, 2Wiki), it fetches the mapped `question_id → [ground_truth_document_ids]`.
    3.  **Metric Computation**:
        *   **Recall@10**: Did any partition containing the ground truth document appear in the top 10 partitions retrieved?
        *   **GT@20**: (Ground Truth Coverage) What percentage of the *required* ground-truth partitions were caught in the top 20 results? Essential for multi-hop 2Wiki/MuSiQue queries.
        *   **MRR**: Mathematically scores the position of the *first* correct partition.
    4.  **Concurrent Execution**: Runs the benchmark across all three splits simultaneously and outputs a comprehensive tabular matrix.

## 📊 K-HOP Multi-Partition Recall (GT@K)
In C-RAG, retrieval isn't just about finding *one* correct document. For multi-hop reasoning (MuSiQue, 2Wiki), the answer is often fragmented across multiple documents. If these documents are split into different partitions by METIS, the model must retrieve **ALL** those partitions. 
- **Recall@K**: Measures if we found at least one correct partition.
- **GT@K**: Measures the fraction of the "total puzzle pieces" found. 
A high `GT@20` score (Target > 0.90) is the prerequisite for a successful Level-3 Agentic Traversal.
