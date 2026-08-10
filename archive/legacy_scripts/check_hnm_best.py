import json
import os
import glob
import pandas as pd

def analyze_hnm_results(results_dir="results/hnm_ablation"):
    json_files = glob.glob(os.path.join(results_dir, "comparison_*_hnm.json"))
    
    summary_data = []
    
    for json_file in json_files:
        dataset_name = os.path.basename(json_file).split("_")[1]
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        for method_key, splits in data.items():
            if "test" not in splits:
                continue
                
            test_metrics = splits["test"]
            
            # Extract method type and hnm_k
            # Format: mlp_info_nce_multi_hnm_X or mlp_kl_div_hnm_X
            parts = method_key.split("_")
            if "info_nce_multi" in method_key:
                loss_type = "info_nce_multi"
            elif "kl_div" in method_key:
                loss_type = "kl_div"
            else:
                continue
                
            hnm_k = int(parts[-1])
            
            summary_data.append({
                "dataset": dataset_name,
                "loss": loss_type,
                "hnm_k": hnm_k,
                "recall@1": test_metrics.get("recall@1"),
                "recall@20": test_metrics.get("recall@20"),
                "mrr": test_metrics.get("mrr"),
                "full_coverage": test_metrics.get("full_coverage@20")
            })
            
    df = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print(f"{'HNM ANALYSIS SUMMARY':^80}")
    print("="*80)
    
    # Group by dataset and loss to find best hnm_k based on recall@20
    for (dataset, loss), group in df.groupby(["dataset", "loss"]):
        print(f"\n>>> Dataset: {dataset.upper()} | Loss: {loss}")
        
        # Sort by recall@20 descending, then mrr descending
        sorted_group = group.sort_values(by=["recall@20", "recall@1", "mrr"], ascending=False)
        
        best = sorted_group.iloc[0]
        baseline = group[group["hnm_k"] == 0].iloc[0] if 0 in group["hnm_k"].values else None
        
        print(f"  Best HNM Type: hnm_{best['hnm_k']}")
        print(f"  Recall@20: {best['recall@20']:.2f}% (Baseline: {baseline['recall@20']:.2f}% if available)")
        print(f"  Recall@1:  {best['recall@1']:.2f}%")
        print(f"  MRR:       {best['mrr']:.2f}")
        
        if baseline is not None and best["hnm_k"] > 0:
            improvement = best["recall@20"] - baseline["recall@20"]
            print(f"  Delta R@20: {'+' if improvement >= 0 else ''}{improvement:.2f}%")
            
        # List all configurations for this group
        print("\n  Full sweep for this category:")
        print(group.sort_values(by="hnm_k")[["hnm_k", "recall@1", "recall@20", "mrr"]].to_string(index=False))

if __name__ == "__main__":
    analyze_hnm_results()
