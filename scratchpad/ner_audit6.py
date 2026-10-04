import os, sys
sys.path.insert(0, os.path.abspath("."))
import hashlib, glob, json, subprocess
from src.experiments import credentials

datasets=["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"]
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()

for ds in datasets:
    local=None
    for cand in [f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl", f"data/ukb_storage/{ds}/gte_qwen/ner_edges_w_df25.pkl"]:
        if os.path.exists(cand):
            local=cand
            break
    local_exists=local is not None
    local_sha=sha(local)[:16] if local_exists else "MISSING"
    local_size=os.path.getsize(local) if local_exists else 0
    print(f"{ds} LOCAL {local or 'MISSING'} {local_size} {local_sha}")

    # Remote account1
    cred=credentials.load_pool('modal')[1]
    cred.activate()
    found=False
    for rp in [f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl", f"data/ukb_storage/{ds}/gte_qwen/ner_edges_w_df25.pkl"]:
        r=subprocess.run([sys.executable,"-m","modal","volume","ls","crag-data-volume",rp,"--json"], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
        if r.returncode==0 and r.stdout.strip() and r.stdout.strip()!="[]":
            try:
                j=json.loads(r.stdout)
                if j:
                    print(f"  REMOTE {rp} EXISTS size {j[0].get('Size','?')}")
                    # Try to get hash by pulling file? For now just note exists
                    # Pull if local missing
                    if not local_exists:
                        print(f"  PULL remote-only {rp}")
                        # Use modal volume get
                        env=dict(os.environ)
                        env["PYTHONUTF8"]="1"
                        pr=subprocess.run([sys.executable,"-m","modal","volume","get","crag-data-volume",rp,".","--force"], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60, env=env)
                        print(f"    pull rc {pr.returncode}")
                        # After pull, file will be at ./<rp>?? Actually gets to ./data/...
                        # Check
                        if os.path.exists(rp):
                            print(f"    pulled {rp} size {os.path.getsize(rp)} sha {sha(rp)[:16]}")
                    else:
                        print(f"  MATCH check pending (need remote sha)")
                    found=True
                    break
            except Exception as e:
                print(f"  remote json err {e} {r.stdout[:200]}")
    if not found:
        print(f"  REMOTE MISSING_BOTH {ds}")
