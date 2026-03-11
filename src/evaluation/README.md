# `src/evaluation/` — Benchmarking & Metrics

Computes the final benchmark results comparing all 5 retrieval strategies. Uses strict Information Retrieval (IR) metrics against ground-truth node IDs — **not** LLM-as-a-judge soft metrics — ensuring reproducible, deterministic results.

> Ragas (`faithfulness`, `answer_relevancy`) is also supported if an LLM endpoint is available, but IR metrics are the primary comparison basis.

---

## Files

### `metrics.py` — IR Metric Calculator

```python
def precision_at_1(retrieved: List[str], expected: List[str]) -> float:
    """Was the top-1 returned node in the expected set? (0 or 1)"""

def recall_at_k(retrieved: List[str], expected: List[str], k: int = 10) -> float:
    """What fraction of expected nodes appeared in the top-K results?"""

def mrr(retrieved: List[str], expected: List[str]) -> float:
    """1/rank of the first expected node found in the retrieved list."""

def evaluate_file(results_json: str, ground_truth_json: str) -> DataFrame:
    """
    Loads a results JSON from the router, computes all three metrics
    per query, groups by category, and returns a summary DataFrame.
    """
```

**Usage:**

```bash
python -m src.evaluation.metrics \
  --results      results/ \
  --groundtruth  data/processed/ground_truth.json \
  --output       results/benchmark_comparison.csv
```

**Output format (`benchmark_comparison.csv`):**

| category | strategy | P@1 | R@10 | MRR | avg_latency_s |
|---|---|---|---|---|---|
| Semantic | vector | 0.71 | 0.88 | 0.79 | 0.01 |
| Multi-hop | crag_agent | 0.65 | 0.91 | 0.73 | 2.94 |
| … | … | … | … | … | … |

---

### `benchmark_gen.py` — Synthetic Query Generator (Routing Category Only)

> **Scope:** Used **only** for the Routing/Graph-Selection category (queries 321–400). All other categories use real SQuAD v2 and WebQSP questions with their existing ground truth.

```bash
python -m src.evaluation.benchmark_gen \
  --graph   data/kg_store/graph.pt \
  --n-walks 80 \             # Generate 80 queries (enough for 321-400)
  --hops    3 \              # Walk 3 hops per query
  --llm     ollama \         # Use to synthesize question text from node content
  --output  data/processed/routing_ground_truth.json
```

**How it works:**

1. Start at a random node in the graph
2. Traverse 2–3 hops across diverse edge types
3. Collect the exact node IDs traversed (`expected_node_ids`)
4. Send node text contents to the LLM: *"Write a complex question that requires reading all of these texts."*
5. Save `{question, expected_node_ids}` pairs

**Why only for the Routing category?**  
Using synthetic questions for Semantic or Multi-hop categories would bias the benchmark — the generator trivially knows which nodes are "correct" since it started there. WebQSP and SQuAD have independently annotated ground truth that is free from this circular dependency.

---

## Ground Truth Sources by Category

| Category | Rows | Ground Truth Source |
|---|---|---|
| Semantic | 1–80 | SQuAD v2 (answerable) — passage chunk node IDs |
| Multi-hop | 81–160 | WebQSP — Freebase entity node IDs |
| Fallback/Web | 161–240 | SQuAD v2 (unanswerable) — expected signal: refusal / empty context |
| Exact Lexical | 241–320 | Manual annotation from `benchmark_400.csv` |
| Routing | 321–400 | `benchmark_gen.py` synthesis |
