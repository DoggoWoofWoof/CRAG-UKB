import json
import os
import pandas as pd

def generate_total_comparison():
    datasets = ["metaqa", "musique", "2wiki", "squad"]
    temp_configs = {
        "metaqa": 0.01,
        "musique": 0.05,
        "2wiki": 0.07,
        "squad": 0.1
    }
    
    results = []
    
    for ds in datasets:
        tau = temp_configs[ds]
        
        # 1. Load Temp Ablation (Original Baselines)
        temp_file = f"results/temp_ablation/comparison_{ds}_temp.json"
        with open(temp_file, 'r') as f:
            temp_data = json.load(f)
            
        orig_nce = temp_data.get(f"mlp_info_nce_multi_tau_{tau:g}", {}).get("test", {})
        orig_kl = temp_data.get(f"mlp_kl_div_tau_{tau:g}", {}).get("test", {})
        
        # 2. Load HNM Ablation
        hnm_file = f"results/hnm_ablation/comparison_{ds}_hnm.json"
        with open(hnm_file, 'r') as f:
            hnm_data = json.load(f)
            
        # HNM 0 (Current Baselines)
        hnm0_nce = hnm_data.get("mlp_info_nce_multi_hnm_0", {}).get("test", {})
        hnm0_kl = hnm_data.get("mlp_kl_div_hnm_0", {}).get("test", {})
        
        # Find Bests in HNM Sweep
        best_hnm_nce = None
        best_nce_type = "N/A"
        max_r1_nce = -1
        
        best_hnm_kl = None
        best_kl_type = "N/A"
        max_r1_kl = -1
        
        for k, metrics in hnm_data.items():
            r1 = metrics.get("test", {}).get("recall@1", 0)
            if "info_nce_multi" in k and r1 > max_r1_nce:
                max_r1_nce = r1
                best_hnm_nce = metrics.get("test", {})
                best_nce_type = k.split("_")[-1]
            elif "kl_div" in k and r1 > max_r1_kl:
                max_r1_kl = r1
                best_hnm_kl = metrics.get("test", {})
                best_kl_type = k.split("_")[-1]

        # Aggregate Row Data
        results.append({
            "Dataset": ds.upper(),
            "Loss": "InfoNCE",
            "Temp_Baseline_R1": orig_nce.get("recall@1"),
            "HNM_Best_R1": best_hnm_nce.get("recall@1"),
            "R1_Gain": best_hnm_nce.get("recall@1", 0) - orig_nce.get("recall@1", 0),
            "Temp_Baseline_R20": orig_nce.get("recall@20"),
            "HNM_Best_R20": best_hnm_nce.get("recall@20"),
            "R20_Gain": best_hnm_nce.get("recall@20", 0) - orig_nce.get("recall@20", 0),
            "HNM_Best_Type": f"hnm_{best_nce_type}"
        })
        
        results.append({
            "Dataset": ds.upper(),
            "Loss": "KL_Div",
            "Temp_Baseline_R1": orig_kl.get("recall@1"),
            "HNM_Best_R1": best_hnm_kl.get("recall@1"),
            "R1_Gain": best_hnm_kl.get("recall@1", 0) - orig_kl.get("recall@1", 0),
            "Temp_Baseline_R20": orig_kl.get("recall@20"),
            "HNM_Best_R20": best_hnm_kl.get("recall@20"),
            "R20_Gain": best_hnm_kl.get("recall@20", 0) - orig_kl.get("recall@20", 0),
            "HNM_Best_Type": f"hnm_{best_kl_type}"
        })

    df = pd.DataFrame(results)
    
    print("\n" + "="*140)
    print(f"{'ULTIMATE RETRIEVAL BENCHMARK: TEMP ABLATION VS HNM PROGRESSION (R@1 & R@20)':^140}")
    print("="*140)
    
    # Sort for clarity
    df = df.sort_values(by=["Dataset", "Loss"])
    
    cols = ["Dataset", "Loss", "Temp_Baseline_R1", "HNM_Best_R1", "R1_Gain", "Temp_Baseline_R20", "HNM_Best_R20", "R20_Gain", "HNM_Best_Type"]
    print(df[cols].to_string(index=False))
    
    print("\n" + "="*140)
    print(f"{'HEAD-TO-HEAD WINNERS BY DATASET (Optimized for R@1)':^140}")
    print("="*140)
    
    for ds in datasets:
        ds_group = df[df["Dataset"] == ds.upper()]
        winner_row = ds_group.loc[ds_group["HNM_Best_R1"].idxmax()]
        print(f"  {ds.upper():<10} | Winner: {winner_row['Loss']} @ {winner_row['HNM_Best_Type']} | Best R@1: {winner_row['HNM_Best_R1']:.2f}% | Best R@20: {winner_row['HNM_Best_R20']:.2f}%")

if __name__ == "__main__":
    generate_total_comparison()
