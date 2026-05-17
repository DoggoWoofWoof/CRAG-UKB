"""
SuperModel — Strategy Router and Benchmark Runner
==================================================
Entry point that initializes all RAG strategies and runs benchmarks.
Each dataset gets its own CoreEngine and independent set of strategies.

Registered strategies:
    Baselines:
        vector_rag              — Dense + BM25 RRF fusion
        graph_rag               — Seed-node BFS multi-hop
    
    CRAG (6 combinations for Level 1+2 benchmark):
        crag_faiss_faiss   — FAISS centroid → FAISS re-rank
        crag_faiss_colbert — FAISS centroid → ColBERT re-rank
        crag_mlp_faiss     — MLP → FAISS re-rank
        crag_mlp_colbert   — MLP → ColBERT re-rank
        crag_colbert_faiss — ColBERT centroid → FAISS re-rank
        crag_colbert_colbert — ColBERT centroid → ColBERT re-rank
    
    Experimental:
        query_graph_gnn         — Text → entity graph → GIN → partition match
"""

import os
import logging
import time
import numpy as np
import torch
from typing import List, Dict, Any, Union, Optional

import yaml

from src.core.engine import CoreEngine
from src.core.llm_manager import LLMManager, MockLLMManager
from src.core.encoders import DenseEncoder
from src.strategies.vector_rag import VectorRAG
from src.strategies.graph_rag import GraphRAG
from src.strategies.crag import CRAG
from src.strategies.query_graph_gnn import QueryGraphGNN

log = logging.getLogger(__name__)


class SuperModel:
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load Configuration
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}

        # Shared settings
        self._storage_dir = self.config.get("storage", {}).get("ukb_dir", "data/ukb_storage")
        self._master_nodes = self.config.get("storage", {}).get(
            "master_nodes", "data/processed/master_nodes.json")
        llm_model = self.config.get("retrieval", {}).get(
            "models", {}).get("generator", "gpt-3.5-turbo")
        encoder_model = self.config.get("retrieval", {}).get(
            "models", {}).get("encoder", "all-MiniLM-L6-v2")
        context_limit = self.config.get("retrieval", {}).get("max_context_tokens", 3000)

        self.llm = MockLLMManager(model_name=llm_model, context_limit=context_limit)
        self.encoder = DenseEncoder(model_name=encoder_model)

        # Per-dataset engine + strategy cache
        self._engines: Dict[str, CoreEngine] = {}
        self._strategies_cache: Dict[str, Dict[str, Any]] = {}
        self._current_source: str = ""

    def _ensure_engine(self, source: str) -> CoreEngine:
        """Load (or retrieve cached) CoreEngine for a given dataset source."""
        if source not in self._engines:
            log.info(f"Loading CoreEngine for source='{source}'...")
            self._engines[source] = CoreEngine(
                storage_dir=self._storage_dir,
                master_nodes_path=self._master_nodes,
                source=source,
            )
        return self._engines[source]

    def _get_strategies(self, source: str) -> Dict[str, Any]:
        """Build (or retrieve cached) strategy set bound to a specific engine."""
        if source not in self._strategies_cache:
            engine = self._ensure_engine(source)
            mlp_encoder = self._load_mlp_encoder(source)
            self._strategies_cache[source] = {
                # Baselines
                "vector_rag": VectorRAG(engine, self.llm, self.encoder),
                "graph_rag":  GraphRAG(engine, self.llm, self.encoder),

                # CRAG — 6 combos (3 selectors × 2 rerankers)
                "crag_faiss_faiss": CRAG(
                    engine, self.llm, self.encoder,
                    mode="faiss_centroid", reranker="faiss"),
                "crag_faiss_colbert": CRAG(
                    engine, self.llm, self.encoder,
                    mode="faiss_centroid", reranker="colbert"),
                "crag_mlp_faiss": CRAG(
                    engine, self.llm, self.encoder,
                    mode="mlp", reranker="faiss", mlp_encoder=mlp_encoder),
                "crag_mlp_colbert": CRAG(
                    engine, self.llm, self.encoder,
                    mode="mlp", reranker="colbert", mlp_encoder=mlp_encoder),
                "crag_colbert_faiss": CRAG(
                    engine, self.llm, self.encoder,
                    mode="colbert_centroid", reranker="faiss"),
                "crag_colbert_colbert": CRAG(
                    engine, self.llm, self.encoder,
                    mode="colbert_centroid", reranker="colbert"),

                # Experimental
                "query_graph_gnn": QueryGraphGNN(engine, self.llm, self.encoder),
            }
        return self._strategies_cache[source]

    def _load_mlp_encoder(self, source: str = ""):
        """Try to load trained TextPartitionMLP for a specific dataset."""
        # Try source-specific checkpoint first, then generic
        checkpoint_path = self.config.get("alignment", {}).get(
            "checkpoint", "checkpoints/text_mlp_encoder.pt")
        if source:
            source_ckpt = f"checkpoints/alignment_mlp_{source}.pth"
            if os.path.exists(source_ckpt):
                checkpoint_path = source_ckpt

        if os.path.exists(checkpoint_path):
            try:
                import torch
                from src.alignment.mlp_encoder import TextPartitionMLP
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                model = TextPartitionMLP(
                    input_dim=ckpt['input_dim'],
                    hidden_dim=ckpt['hidden_dim'],
                    output_dim=ckpt['output_dim'])
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                log.info(f"Loaded MLP encoder from {checkpoint_path}")
                return model
            except Exception as e:
                log.warning(f"Failed to load MLP encoder: {e}")
        return None

    def run_benchmark(self, query: str, truth_nodes: List[str],
                      strategies: List[str] = None,
                      source: str = "2wiki") -> Dict:
        """
        Run one or more strategies on a query and return metrics.
        
        Args:
            query: Question text
            truth_nodes: List of ground-truth node IDs
            strategies: Optional list of strategy names to run (default: all)
            source: Dataset source to use for index loading
        """
        strat_map = self._get_strategies(source)
        strats = strategies or list(strat_map.keys())
        results = {}

        for name in strats:
            if name not in strat_map:
                log.warning(f"Unknown strategy: {name}")
                continue

            strategy = strat_map[name]
            log.info(f"Running strategy: {name}")
            t0 = time.time()
            res = strategy.retrieve(query)
            latency = time.time() - t0

            retrieved_ids = [n.node_id for n in res.nodes]
            precision = self._precision(retrieved_ids, truth_nodes)
            recall = self._recall(retrieved_ids, truth_nodes)
            mrr = self._mrr(retrieved_ids, truth_nodes)

            results[name] = {
                "precision": float(round(precision, 4)),
                "recall": float(round(recall, 4)),
                "mrr": float(round(mrr, 4)),
                "latency_s": float(round(latency, 4)),
                "answer": res.answer if hasattr(res, 'answer') else "",
                "nodes_retrieved": len(res.nodes),
                "metadata": res.metadata if hasattr(res, 'metadata') else {},
            }
        return results

    def _precision(self, retrieved: List[str], truth: List[str]) -> float:
        if not retrieved:
            return 0.0
        return len(set(retrieved) & set(truth)) / len(retrieved)

    def _recall(self, retrieved: List[str], truth: List[str]) -> float:
        if not truth:
            return 0.0
        return len(set(retrieved) & set(truth)) / len(truth)

    def _mrr(self, retrieved: List[str], truth: List[str]) -> float:
        for i, node_id in enumerate(retrieved):
            if node_id in truth:
                return 1.0 / (i + 1)
        return 0.0

    def run_full_benchmark(self, dataset: str = "2wiki", n_queries: int = 100) -> Dict:
        import numpy as np
        log.info(f"Generating full benchmark for dataset: {dataset} (n={n_queries})")
        
        # Load the engine for this dataset
        engine = self._ensure_engine(dataset)

        # 1. Collect queries and their ground truth from this dataset's nodes
        queries = []
        for node in engine.all_nodes:
            if node.metadata.get("type") == "question":
                gt_nodes = node.neighbors
                if gt_nodes:
                    queries.append((node.content, gt_nodes))
            if len(queries) >= n_queries:
                break
                
        if not queries:
            log.error(f"No queries found for dataset {dataset}.")
            return {}
            
        log.info(f"Collected {len(queries)} evaluation queries.")
        
        # 2. Iterate and aggregate
        strat_map = self._get_strategies(dataset)
        strats = list(strat_map.keys())
        agg = {s: {"precision": [], "recall": [], "mrr": [], "latency_s": [], "accuracy": []} for s in strats}
        
        for q_idx, (q_text, gt_nodes) in enumerate(queries):
            res_dict = self.run_benchmark(q_text, gt_nodes, strategies=strats, source=dataset)
            for s_name, s_res in res_dict.items():
                agg[s_name]["precision"].append(s_res["precision"])
                agg[s_name]["recall"].append(s_res["recall"])
                agg[s_name]["mrr"].append(s_res.get("mrr", 0.0))
                agg[s_name]["latency_s"].append(s_res["latency_s"])
                acc = 1.0 if s_res.get("mrr", 0.0) == 1.0 else 0.0
                agg[s_name]["accuracy"].append(acc)
                
        # 3. Final summary
        final_summary = {}
        print(f"\n{'='*75}")
        print(f" END-TO-END RAG BENCHMARK RESULTS ({dataset.upper()} | n={len(queries)})")
        print(f"{'='*75}")
        print(f"{'Strategy':<22} | {'Recall':<6} | {'Prec':<6} | {'Accuracy':<8} | {'MRR':<5} | {'Lat(s)':<6}")
        print("-" * 75)
        
        for s_name, metrics in agg.items():
            if not metrics["recall"]:
                continue
                
            final_summary[s_name] = {
                "avg_precision": round(float(np.mean(metrics["precision"])), 4),
                "avg_recall": round(float(np.mean(metrics["recall"])), 4),
                "avg_mrr": round(float(np.mean(metrics["mrr"])), 4),
                "avg_accuracy": round(float(np.mean(metrics["accuracy"])), 4),
                "avg_latency_s": round(float(np.mean(metrics["latency_s"])), 4),
                "total_queries_run": len(queries)
            }
            res = final_summary[s_name]
            print(f"{s_name:<22} | {res['avg_recall']:>6.4f} | {res['avg_precision']:>6.4f} | {res['avg_accuracy']:>8.4f} | {res['avg_mrr']:>5.4f} | {res['avg_latency_s']:>6.4f}")
            
        print(f"{'='*75}\n")
        return final_summary


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    model = SuperModel()
    query = "What is the relationship between SQuAD and knowledge graphs?"
    truth = ["squad_0"]

    print("Running SuperModel Benchmark...")
    report = model.run_benchmark(query, truth, source="squad")
    print(json.dumps(report, indent=2))


