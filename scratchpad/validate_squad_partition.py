import json, os, collections, numpy as np, faiss, torch
src='squad_clean'
pm=json.load(open(f'data/ukb_storage/{src}/partition_map.json'))
parts=list(pm.values())
n_parts=len(set(parts))
print(f'{src}: n_nodes {len(pm)} n_parts {n_parts}')
cnt=collections.Counter(parts)
print(f'  min {min(cnt.values())} max {max(cnt.values())} median {np.median(list(cnt.values())):.1f} mean {np.mean(list(cnt.values())):.1f}')
cent_path=f'data/ukb_storage/{src}/centroids.index'
if os.path.exists(cent_path):
    idx=faiss.read_index(cent_path)
    print(f'  centroids {idx.ntotal} dim {idx.d}')
pid_path=f'data/ukb_storage/{src}/centroid_pids.json'
if os.path.exists(pid_path):
    pids=json.load(open(pid_path))
    print(f'  centroid_pids {len(pids)} sorted {pids[:5]}')
gpath=f'data/ukb_storage/{src}/graph.pt'
if os.path.exists(gpath):
    import torch.serialization
    torch.serialization.add_safe_globals([__import__('torch_geometric.data.data', fromlist=['Data']).Data])
    data=torch.load(gpath, map_location='cpu', weights_only=False)
    print(f'  graph.pt nodes {data.num_nodes} edges {data.edge_index.shape[1]//2} undirected')
dm=json.load(open('data/canonical/squad/document_manifest.json'))
print(f'canonical squad n_docs {dm["n_docs"]} vs ukb {len(pm)}')
# also check 2wiki_clean for comparison
src='2wiki_clean'
pm=json.load(open(f'data/ukb_storage/{src}/partition_map.json'))
parts=list(pm.values())
print(f'{src}: n_nodes {len(pm)} n_parts {len(set(parts))} min {min(collections.Counter(parts).values())}')
