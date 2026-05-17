import json
import time
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
    strategies_to_test = ["vector", "graph", "crag_standard", "crag_colbert", "crag_v4_agent"]
    
    for i, benchmark in enumerate(tqdm(benchmarks, desc="Evaluating Benchmarks")):
        query = benchmark["query"]
        truth = benchmark["truth_nodes"]
        
        for strategy_name in strategies_to_test:
            start_time = time.time()
            
            # Execute the specific RAG strategy via the framework Singleton
            try:
                # The retrieve API returns a RetrievalResult object with a .nodes list
                response = super_model.strategies[strategy_name].retrieve(query)
                retrieved_ids = [n.node_id for n in response.nodes]
            except Exception as e:
                print(f"\n[Warning] Strategy '{strategy_name}' failed on Query {i}: {e}")
                retrieved_ids = []
                
            latency = time.time() - start_time
            
            p_1 = MetricsCalculator.calculate_precision_at_k(retrieved_ids, truth, k=1)
            r_10 = MetricsCalculator.calculate_recall_at_k(retrieved_ids, truth, k=10)
            mrr = MetricsCalculator.calculate_mrr(retrieved_ids, truth)
            
            results.append({
                "strategy": strategy_name,
                "p@1": p_1,
                "r@10": r_10,
                "mrr": mrr,
                "latency": latency
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
