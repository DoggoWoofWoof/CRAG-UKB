#!/usr/bin/env python
"""
Launch D full-validation shards on Modal across healthy accounts.
Each shard = one dataset (full validation, te_cap 0 uncapped) with scope_topk 0.
We shard at dataset granularity first; if timeout/OOM, will subshard 0:500 etc.
"""
import os, sys, subprocess, time, json, pathlib

# Healthy accounts from billing check: 0-10 ACTIVE, 11-14 AUTH_FAILED
HEALTHY = [0,1,2,3,4,5,6,7,8,9,10]  # 11 accounts, prefer 0,1,4,6,8,10 as before but now all active
# Pick 6 datasets mapping
DATASETS = ["squad_clean", "2wiki_clean", "musique_clean", "hotpotqa_clean", "webqsp", "metaqa"]
# Head datasets = full-6 universal (matches e2e_full6_universal_gte_qwen.json)
HEAD_DATASETS = ["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"]
# Subdir
SUBDIR = "gte_qwen"
# Account assignment round-robin
ASSIGN = {
    "squad_clean": 1,
    "2wiki_clean": 4,
    "musique_clean": 6,
    "hotpotqa_clean": 8,
    "webqsp": 0,
    "metaqa": 10,
}
# For now dataset-level shards (full validation per dataset, te_cap 0)
# To respect 3/account limit, each account gets 1 job here, so ok.
LOGDIR = "scratchpad/D_shards"
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs("results/L2", exist_ok=True)

def launch_one(dataset, account):
    shard_tag = f"D_{dataset}_full"  # uncapped full validation, scope 0
    logp = os.path.join(LOGDIR, f"{dataset}_a{account}_{shard_tag}.log")
    # Build argv for experiments.py run e2e-ner
    # Use te_cap 0 = uncapped, tr_cap 3000, limit 8000 but te_cap 0 overrides to full
    cmd = [
        sys.executable, "experiments.py", "run", "e2e-ner",
        "--backend", "modal", "--account", str(account), "--",
        "--datasets", dataset,
        "--head-datasets", *HEAD_DATASETS,
        "--subdir", SUBDIR,
        "--scope-topk", "0",
        "--epochs", "15",
        "--te-cap", "0",  # 0 = uncapped full validation
        "--shard-tag", shard_tag
    ]
    # For sharding at 500 granularity, we would add --shard-start/end
    # For dataset-level, omit shard slicing
    print(f"[LAUNCH] {dataset} -> account {account} tag {shard_tag}")
    print(" ", " ".join(cmd))
    # Launch detached
    # Use Popen with creationflags on Windows
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    # Open log file
    lf = open(logp, "w", encoding="utf-8")
    lf.write(f"# Launch {dataset} account {account} tag {shard_tag}\n")
    lf.write(f"# Cmd: {' '.join(cmd)}\n")
    lf.write(f"# Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lf.flush()
    # Use Popen to run and log
    # We need to capture output to log file
    # For detached, we cannot easily capture live, but we can start process that writes to log
    # Instead use python -u and redirect via shell? Use Popen with stdout=lf
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=".", **kwargs)
    # On Windows detached, Popen returns immediately but proc may not be trackable; we still get PID
    print(f"  -> PID {proc.pid} log {logp}")
    lf.write(f"# PID {proc.pid}\n")
    lf.flush()
    # Don't wait; return info
    return {"dataset": dataset, "account": account, "pid": proc.pid, "log": logp, "tag": shard_tag, "start": time.time(), "cmd": cmd}

def main():
    launches = []
    for ds in DATASETS:
        acct = ASSIGN[ds]
        info = launch_one(ds, acct)
        launches.append(info)
        time.sleep(5)  # stagger to avoid simultaneous volume push thrash
    # Write manifest
    manifest = {
        "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "healthy_pool": HEALTHY,
        "assign": ASSIGN,
        "head_datasets": HEAD_DATASETS,
        "scope_topk": 0,
        "K": 8,
        "MAXK": 500,
        "te_cap": 0,
        "shards": launches
    }
    json.dump(manifest, open(os.path.join(LOGDIR, "manifest.json"), "w"), indent=2)
    print(f"\n[MANIFEST] written to {LOGDIR}/manifest.json")
    for l in launches:
        print(f"  {l['dataset']:16s} account {l['account']} PID {l['pid']} log {l['log']}")
    print("\n[NOTE] Jobs are detached Modal launches; monitor via 'modal app list' or log tail. Use supervisor polling.")

if __name__ == "__main__":
    main()
