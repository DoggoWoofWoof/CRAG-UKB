import json
import os

def check_json_metrics(base_dir, dataset_lower, method_key, split="test"):
    filename = ""
    # find the matching json file
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".json") and dataset_lower in f.lower():
                filename = os.path.join(root, f)
                break
    
    if not filename:
        return f"[ERROR] No json found for {dataset_lower} in {base_dir}"

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if method_key not in data:
        return f"[ERROR] Key {method_key} not in {filename}"
        
    metrics = data[method_key][split]
    
    # We dump specific metrics to verify against markdown
    r1 = metrics.get('recall@1', 'N/A')
    r20 = metrics.get('recall@20', 'N/A')
    f1_5 = metrics.get('f1@5', 'N/A')
    ndcg_10 = metrics.get('ndcg@10', 'N/A')
    latency = metrics.get('avg_latency_ms', 'N/A')
    
    return f"{base_dir} | {dataset_lower} | {method_key} -> R@1: {r1} | R@20: {r20} | F1@5: {f1_5} | NDCG@10: {ndcg_10} | Latency: {latency}ms"

suites = [
    ("results/level_1", "squad", "mlp"),
    ("results/level_1", "2wiki", "mlp"),
    ("results/level_1", "metaqa", "mlp"),
    ("results/level_1", "musique", "mlp"),
    
    ("results/loss_ablation", "2wiki", "mlp_info_nce_multi"),
    ("results/loss_ablation", "2wiki", "mlp_kl_div"),
    ("results/loss_ablation", "squad", "mlp_info_nce_multi"),
    
    ("results/temp_ablation", "2wiki", "mlp_kl_div_tau_0.07"),
    ("results/temp_ablation", "metaqa", "mlp_info_nce_multi_tau_0.01"),
    ("results/temp_ablation", "squad", "mlp_kl_div_tau_0.1")
]

with open("audit_legacy.txt", "w") as f:
    for s in suites:
        f.write(check_json_metrics(s[0], s[1], s[2]) + "\n")
