import json
from typing import List, Dict, Any
from tqdm import tqdm
from src.super_model import SuperModel

class MetricsCalculator:
    @staticmethod
    def calculate_precision_at_k(retrieved: List[str], truth: List[str], k: int = 1) -> float:
        if not retrieved: return 0.0
        top_k = retrieved[:k]
        intersect = set(top_k) & set(truth)
        return len(intersect) / k

    @staticmethod
    def calculate_recall_at_k(retrieved: List[str], truth: List[str], k: int = 10) -> float:
        if not truth: return 0.0
        top_k = retrieved[:k]
        intersect = set(top_k) & set(truth)
        return len(intersect) / len(truth)

    @staticmethod
    def calculate_mrr(retrieved: List[str], truth: List[str]) -> float:
        for i, node_id in enumerate(retrieved):
            if node_id in truth:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def generate_report(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple queries."""
        summary = {}
        for entry in results_list:
            strategy = entry["strategy"]
            if strategy not in summary:
                summary[strategy] = {"p@1": [], "r@10": [], "mrr": [], "latency": []}
            
            summary[strategy]["p@1"].append(entry["p@1"])
            summary[strategy]["r@10"].append(entry["r@10"])
            summary[strategy]["mrr"].append(entry["mrr"])
            summary[strategy]["latency"].append(entry["latency"])
            
        final_report = {}
        for strategy, metrics in summary.items():
            final_report[strategy] = {
                "avg_p@1": sum(metrics["p@1"]) / len(metrics["p@1"]),
                "avg_r@10": sum(metrics["r@10"]) / len(metrics["r@10"]),
                "avg_mrr": sum(metrics["mrr"]) / len(metrics["mrr"]),
                "avg_latency": sum(metrics["latency"]) / len(metrics["latency"])
            }
        return final_report

if __name__ == "__main__":
    print("Initializing C-RAG SuperModel and Loading Unified Knowledge Base...")
    super_model = SuperModel("configs/config.yaml")
    
    benchmark_path = "data/processed/synthetic_benchmark.json"
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmarks = json.load(f)
        
    print(f"Loaded {len(benchmarks)} synthetic ground-truth queries.")
    
    results = []
    strategies_to_test = [
        "vector_rag",
        "graph_rag",
        "crag_faiss_faiss",
        "crag_mlp_faiss",
        "crag_colbert_faiss",
    ]
    
    for i, benchmark in enumerate(tqdm(benchmarks, desc="Evaluating Benchmarks")):
        query = benchmark["query"]
        truth = benchmark["truth_nodes"]
        
        source = benchmark.get("source", "squad")
        try:
            benchmark_results = super_model.run_benchmark(
                query, truth, strategies=strategies_to_test, source=source
            )
        except Exception as e:
            print(f"\n[Warning] Benchmark failed on Query {i}: {e}")
            benchmark_results = {}

        for strategy_name in strategies_to_test:
            strategy_metrics = benchmark_results.get(strategy_name, {})
            results.append({
                "strategy": strategy_name,
                "p@1": strategy_metrics.get("precision", 0.0),
                "r@10": strategy_metrics.get("recall", 0.0),
                "mrr": strategy_metrics.get("mrr", 0.0),
                "latency": strategy_metrics.get("latency_s", 0.0),
            })
            
    # Compile execution results into the final comprehensive report
    print("\n--- Generating Scientific Benchmark Report ---")
    report = MetricsCalculator.generate_report(results)
    print(json.dumps(report, indent=4))
    
    # Save the output
    output_path = "data/processed/evaluation_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
    print(f"\nSaved scientific benchmark matrix to {output_path}")
