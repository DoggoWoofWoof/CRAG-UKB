#!/usr/bin/env python
import os, sys, subprocess, time, json
ACCOUNT=1
DATASETS=["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"]
HEAD=["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"]
SUBDIR="gte_qwen"
TAG="D_full6_universal_v2"
LOGDIR="scratchpad/D_shards"
os.makedirs(LOGDIR, exist_ok=True)
logp=os.path.join(LOGDIR, f"full6_a{ACCOUNT}_{TAG}.log")
cmd=[sys.executable,"experiments.py","run","e2e-ner","--backend","modal","--account",str(ACCOUNT),"--",
     "--datasets",*DATASETS,
     "--head-datasets",*HEAD,
     "--subdir",SUBDIR,
     "--scope-topk","0",
     "--epochs","15",
     "--te-cap","0",
     "--shard-tag",TAG]
print("[LAUNCH SINGLE v2]", " ".join(cmd))
kwargs={}
if os.name=="nt":
    kwargs["creationflags"]=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
lf=open(logp,"w",encoding="utf-8")
lf.write(f"# Launch full6 D universal v2 tag {TAG} account {ACCOUNT}\n")
lf.write(f"# Cmd: {' '.join(cmd)}\n")
lf.write(f"# Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); lf.flush()
proc=subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=".", **kwargs)
print(f"-> PID {proc.pid} log {logp}")
lf.write(f"# PID {proc.pid}\n"); lf.flush()
json.dump({"account":ACCOUNT,"pid":proc.pid,"log":logp,"datasets":DATASETS,"tag":TAG,"start":time.time()}, open(os.path.join(LOGDIR,"single_manifest_v2.json"),"w"), indent=2)
print("manifest written")
