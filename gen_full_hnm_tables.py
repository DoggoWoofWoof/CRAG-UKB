import json
import os
import glob

def generate_full_tables(results_dir="results/hnm_ablation"):
    json_files = glob.glob(os.path.join(results_dir, "comparison_*_hnm.json"))
    
    output = []
    
    for json_file in sorted(json_files):
        dataset_name = os.path.basename(json_file).split("_")[1].upper()
        output.append(f"\n### {dataset_name} Exhaustive Metrics\n")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Group by loss type
        for loss_type in ["info_nce_multi", "kl_div"]:
            header = f"**Method: MLP + {loss_type.upper()}**\n"
            table_header = "| HNM_k | R@1 | P@1 | F1@1 | NDCG@1 | R@20 | P@20 | F1@20 | NDCG@20 | MRR |\n"
            table_sep = "|---|---|---|---|---|---|---|---|---|---|\n"
            
            rows = []
            # Filter keys for this loss type
            method_keys = [k for k in data.keys() if f"mlp_{loss_type}" in k]
            # Sort by hnm_k
            method_keys.sort(key=lambda x: int(x.split("_")[-1]))
            
            for k in method_keys:
                m = data[k].get("test", {})
                hnm_k = k.split("_")[-1]
                row = f"| {hnm_k} | {m.get('recall@1'):.2f} | {m.get('precision@1'):.2f} | {m.get('f1@1'):.2f} | {m.get('ndcg@1'):.2f} | {m.get('recall@20'):.2f} | {m.get('precision@20'):.2f} | {m.get('f1@20'):.2f} | {m.get('ndcg@20'):.2f} | {m.get('mrr'):.2f} |\n"
                rows.append(row)
            
            if rows:
                output.append(header)
                output.append(table_header)
                output.append(table_sep)
                output.extend(rows)
                output.append("\n")
                
    return "".join(output)

if __name__ == "__main__":
    print(generate_full_tables())
