#!/usr/bin/env python
import os, sys, subprocess, time, json, pathlib
# Launch full partition curves and C6 dense/splade/offset in parallel, D untouched
# Healthy accounts: 0,1,4,6,8,10 (1 occupied by D)
# Use 0,4,6,8,10 for GPU C6, CPU for partition metrics
LOGDIR="scratchpad/C6_and_partition"
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs("results/L1", exist_ok=True)
os.makedirs("scratchpad/full_partition", exist_ok=True)

def launch(name, cmd, account=None, logname=None, backend="modal"):
    # if account provided, use modal backend
    if account is not None:
        full_cmd=[sys.executable,"experiments.py","run",name,"--backend",backend,"--account",str(account),"--"]+cmd
    else:
        full_cmd=[sys.executable,"-m",name.split(":")[0]]+cmd if ":" in name else [sys.executable,"experiments.py","run",name,"--backend","local","--"]+cmd
    # Simplified: if name is script path, launch directly
    if name.endswith(".py"):
        full_cmd=[sys.executable, name] + cmd
    logp=os.path.join(LOGDIR, logname or f"{name.replace('/','_')}.log")
    print(f"[LAUNCH] {logname} -> {' '.join(full_cmd[:8])}...")
    kwargs={}
    if os.name=="nt":
        kwargs["creationflags"]=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    lf=open(logp,"w",encoding="utf-8")
    lf.write(f"# {logname} start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lf.write(f"# cmd {' '.join(full_cmd)}\n")
    lf.flush()
    proc=subprocess.Popen(full_cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=".", **kwargs)
    print(f"  PID {proc.pid} log {logp}")
    lf.write(f"# PID {proc.pid}\n"); lf.flush()
    return {"name":logname,"pid":proc.pid,"log":logp,"cmd":full_cmd,"start":time.time(),"account":account}

launches=[]

# 1) FULL partition curves per dataset via canonical reader (CPU, local, batched FAISS)
# Launch one script that will handle all 6, but we launch it as single local job
launches.append(launch("scratchpad/full_partition_canonical.py", ["--datasets","squad","webqsp","metaqa","2wiki","musique","hotpotqa","--batch","500"], logname="partition_full_canonical", account=None))

# 2) C6-A dense/splade/offset via canonical - for now launch 3 representative GPU jobs
# Use existing tasks: splade-encode, l1-universal-head (offset)
# For dense, C5 already complete, but we mark C6_dense as reuse
# Launch splade-encode for webqsp (small) as example
launches.append(launch("splade-encode", ["--datasets","webqsp"], logname="C6_splade_webqsp", account="0"))

# Launch l1-universal-head for offset generation (graph-independent)
launches.append(launch("l1-universal-head", ["--datasets","squad_clean","webqsp","metaqa","musique_clean","2wiki_clean","hotpotqa_clean","--subdir","gte_qwen","--limit","8000","--tr-cap","3000","--te-cap","2000","--epochs","15"], logname="C6_offset_universal", account="4"))

# Launch crag-gates as example of C6 pipeline (will be provisional)
launches.append(launch("crag-gates", ["--datasets","squad_clean","webqsp"], logname="C6_crag_gates", account="6"))

json.dump(launches, open(os.path.join(LOGDIR,"manifest.json"),"w"), indent=2)
print(f"Launched {len(launches)} jobs, manifest {LOGDIR}/manifest.json")
for l in launches:
    print(f" {l['name']} PID {l['pid']} account {l.get('account')} log {l['log']}")
