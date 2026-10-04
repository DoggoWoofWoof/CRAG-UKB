import subprocess, os, sys, time, json
env=dict(os.environ)
env["PYTHONUTF8"]="1"
env["PYTHONIOENCODING"]="utf-8"
cmd=[sys.executable,"experiments.py","run","e2e-ner","--backend","modal","--account","4","--",
     "--datasets","metaqa",
     "--te-cap","2000",
     "--eval-ids-file","data/canonical/metaqa/splits/historical_2000_ids.json",
     "--hard-head-path","data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt",
     "--mix-head-path","data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt",
     "--no-head-train",
     "--scope-topk","0",
     "--shard-tag","metaqa_2000_repro"]
print(" ".join(cmd))
proc=subprocess.Popen(cmd, stdout=open("scratchpad/metaqa_2000_repro.log","w",encoding="utf-8"), stderr=subprocess.STDOUT, env=env, cwd=".")
print(f"PID {proc.pid}")
open("scratchpad/metaqa_2000_repro.pid","w").write(str(proc.pid))
