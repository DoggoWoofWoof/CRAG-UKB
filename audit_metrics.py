import json
import os

def exhaustive_audit():
    output = ["Initiating EXHAUSTIVE Audit of all Core Metrics...\n"]
    
    # Expected metrics matching the "Detailed HNM Sweep Metrics" tables exactly
    expected = {
        "SQUAD": {
            "key": "mlp_kl_div_hnm_18",
            "metrics": {"recall@1": 69.42, "recall@20": 100.00, "mrr": 78.89}
        },
        "METAQA": {
            "key": "mlp_kl_div_hnm_0",
            "metrics": {"recall@1": 48.07, "recall@20": 95.80, "mrr": 61.24}
        },
        "2WIKI": {
            "key": "mlp_kl_div_hnm_149",
            "metrics": {"recall@1": 25.07, "recall@20": 84.27, "mrr": 38.59}
        },
        "MUSIQUE": {
            "key": "mlp_kl_div_hnm_33",
            "metrics": {"recall@1": 80.30, "recall@20": 99.30, "mrr": 86.29}
        }
    }
    
    results_dir = "results/hnm_ablation"
    files = [f for f in os.listdir(results_dir) if f.startswith("comparison_") and f.endswith(".json")]
    
    errors = 0
    for file in files:
        dataset = file.split("_")[1].upper()
        if dataset not in expected:
            continue
            
        with open(os.path.join(results_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        target_key = expected[dataset]["key"]
        if target_key not in data:
            output.append(f"[ERROR] Expected key {target_key} missing in {file}")
            errors += 1
            continue
            
        test_metrics = data[target_key]["test"]
        
        for metric, exp_val in expected[dataset]["metrics"].items():
            actual_val = test_metrics[metric]
            if abs(actual_val - exp_val) > 0.01:
                output.append(f"[MISMATCH] in {dataset} [{target_key}] -> {metric}: Expected {exp_val}, Got {actual_val}")
                errors += 1
            else:
                output.append(f"[PASS] {dataset} [{target_key}] {metric}: {actual_val} matches.")
                
    if errors == 0:
        output.append("\n[AUDIT PASSED] All deep metrics in Markdown precisely match JSON bounds.")
    else:
        output.append(f"\n[AUDIT FAILED] {errors} discrepancies detected.")

    with open("audit_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    exhaustive_audit()
