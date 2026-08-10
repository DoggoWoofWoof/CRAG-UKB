import json
import os

results_dir = r"c:\Users\Swastik\Desktop\CRAG\results"
datasets = ["2wiki", "metaqa", "musique", "squad"]

for ds in datasets:
    file_path = os.path.join(results_dir, f"comparison_{ds}.json")
    if not os.path.exists(file_path):
        continue
    
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"=== Dataset: {ds} ===")
    
    for method, splits in data.items():
        if "test" in splits:
            test = splits["test"]
            print(f"[{method}] R@1: {test.get('recall@1')}, R@5: {test.get('recall@5')}, f1@5: {test.get('f1@5')}, nDCG@5: {test.get('ndcg@5')}, lat: {test.get('avg_latency_ms')}ms")
    print("\n")
