import json
import os
import pandas as pd

def compare_results():
    datasets = ["metaqa", "musique", "2wiki", "squad"]
    temp_configs = {
        "metaqa": 0.01,
        "musique": 0.05,
        "2wiki": 0.07,
        "squad": 0.1
    }
    
    records = []
    
    for ds in datasets:
        tau = temp_configs[ds]
        
        # Load Temp Ablation
        temp_file = f"results/temp_ablation/comparison_{ds}_temp.json"
        with open(temp_file, 'r') as f:
            temp_data = json.load(f)
            
        # Locate InfoNCE at target tau
        # Key format: mlp_info_nce_multi_tau_0.01
        temp_method_key = f"mlp_info_nce_multi_tau_{tau:g}"
        temp_metrics = temp_data.get(temp_method_key, {}).get("test", {})
        
        # Load HNM Ablation
        hnm_file = f"results/hnm_ablation/comparison_{ds}_hnm.json"
        with open(hnm_file, 'r') as f:
            hnm_data = json.load(f)
            
        # HNM Baseline (hnm_0)
        hnm0_metrics = hnm_data.get("mlp_info_nce_multi_hnm_0", {}).get("test", {})
        
        # Find Best HNM for InfoNCE in this dataset
        best_hnm_key = None
        best_r1 = -1
        for k in hnm_data:
            if "mlp_info_nce_multi" in k:
                r1 = hnm_data[k].get("test", {}).get("recall@1", 0)
                if r1 > best_r1:
                    best_r1 = r1
                    best_hnm_key = k
        
        best_hnm_metrics = hnm_data.get(best_hnm_key, {}).get("test", {})
        
        records.append({
            "Dataset": ds.upper(),
            "Opt_Tau": tau,
            "Temp_Ablation_R1": temp_metrics.get("recall@1"),
            "HNM_Baseline_R1": hnm0_metrics.get("recall@1"),
            "Best_HNM_R1": best_hnm_metrics.get("recall@1"),
            "Best_HNM_Type": best_hnm_key.split("_")[-1] if best_hnm_key else "N/A",
            "Temp_Ablation_R20": temp_metrics.get("recall@20"),
            "HNM_Baseline_R20": hnm0_metrics.get("recall@20"),
            "Best_HNM_R20": best_hnm_metrics.get("recall@20"),
        })
        
    df = pd.DataFrame(records)
    
    print("\n" + "="*100)
    print(f"{'INFONCE: TEMP ABLATION (BASELINE) VS HNM ABLATION':^100}")
    print("="*100)
    
    print("\n>>> RECALL @ 1 COMPARISON")
    print(df[["Dataset", "Opt_Tau", "Temp_Ablation_R1", "HNM_Baseline_R1", "Best_HNM_R1", "Best_HNM_Type"]].to_string(index=False))
    
    print("\n>>> RECALL @ 20 COMPARISON")
    print(df[["Dataset", "Opt_Tau", "Temp_Ablation_R20", "HNM_Baseline_R20", "Best_HNM_R20", "Best_HNM_Type"]].to_string(index=False))

    # Calculate Deltas
    print("\n>>> SUMMARY OF GAINS (Best HNM over Temp Baseline)")
    df["R1_Gain"] = df["Best_HNM_R1"] - df["Temp_Ablation_R1"]
    df["R20_Gain"] = df["Best_HNM_R20"] - df["Temp_Ablation_R20"]
    print(df[["Dataset", "R1_Gain", "R20_Gain"]].to_string(index=False))

if __name__ == "__main__":
    compare_results()
