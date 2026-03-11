import sys
import os

def patch_run_exp():
    path = 'src/crag/run_exp.py'
    if not os.path.exists(path):
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    code = code.replace('from crag.graph.partitioning import GraphPartitioner', 'from crag.graph.partitioning import SemanticPartitioner')
    
    old_partitioner_init = "        partitioner = GraphPartitioner(\n            method=partition_config.get('method', 'metis'),\n            n_partitions=partition_config.get('n_partitions', 10)\n        )"
    new_partitioner_init = "        partitioner = SemanticPartitioner(\n            resolution=1.0\n        )"
    code = code.replace(old_partitioner_init, new_partitioner_init)
    
    old_call = "partitioner.partition(graph_engine)"
    new_call = "graph_engine.data.part_id = partitioner.partition(graph_engine.data)"
    code = code.replace(old_call, new_call)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(code)

def patch_interface():
    path = 'src/crag/llm/interface.py'
    if not os.path.exists(path):
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    code = code.replace('def __init__(self):', 'def __init__(self, **kwargs):')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(code)

patch_run_exp()
patch_interface()
print("Patched repo files.")
