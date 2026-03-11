import pandas as pd
import json
import argparse
import time
from tqdm import tqdm

from crag.llm.interface import MockLLMClient
from crag.retrieval.vector_store import FaissVectorStore
from crag.graph.wikidata import WikidataKG
from crag.retrieval.hybrid import HybridRetrievalModule
from crag.model.colbert import ColBERTReranker
from crag.agent.cra import CognitiveRetrievalAgent
from crag.baselines.vector import VectorBaseline
from crag.baselines.static_graph import StaticGraphBaseline
from crag.agent.state import AgentConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["vector", "graph", "crag_base"])
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Initialize components (similar to main branch's run_exp.py)
    llm = MockLLMClient()
    vs = FaissVectorStore()
    kg = WikidataKG()
    hrm = HybridRetrievalModule(kg, vs)
    reranker = ColBERTReranker()

    # Instantiate model
    if args.mode == "vector":
        model = VectorBaseline(vs, llm, k=5)
        def solve_fn(q):
            # Assuming return is similar to string or dictionary
            ans = model.solve(q)
            # Typically returns string or dict, let's just make it return standard format
            return {"answer": ans if isinstance(ans, str) else ans.get("answer", ""), "contexts": []}
    
    elif args.mode == "graph":
        model = StaticGraphBaseline(hrm, llm)
        def solve_fn(q):
            ans = model.solve(q)
            return {"answer": ans if isinstance(ans, str) else ans.get("answer", ""), "contexts": []}
    
    elif args.mode == "crag_base":
        agent_config = AgentConfig(max_steps=3, max_expansions=3, use_reranker=False) # Base CRAG might not use colbert
        agent = CognitiveRetrievalAgent(hrm, reranker, llm_client=llm, config=agent_config)
        def solve_fn(q):
            state = agent.solve(q)
            return {
                "answer": state.final_answer if state.final_answer else "No answer found",
                "contexts": state.path if hasattr(state, "path") else []
            }
    
    df = pd.read_csv("benchmark_400.csv")
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = row['query']
        start_time = time.time()
        
        try:
            res = solve_fn(query)
            answer = res.get("answer", "Error")
            contexts = res.get("contexts", [])
        except Exception as e:
            answer = f"Error: {str(e)}"
            contexts = []
            
        latency = time.time() - start_time
        
        results.append({
            "id": row['id'],
            "query": query,
            "category": row['category'],
            "target_system_advantaged": row['target_system_advantaged'],
            "generated_answer": str(answer),
            "retrieved_contexts": [str(c) for c in contexts] if isinstance(contexts, list) else [],
            "latency_seconds": latency
        })
        
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
