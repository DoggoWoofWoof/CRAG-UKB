import subprocess, os, sys, time
env=dict(os.environ)
env["PYTHONUTF8"]="1"
env["PYTHONIOENCODING"]="utf-8"
cmd=[sys.executable,"-m","modal","run","scratchpad/metaqa_2000_modal.py"]
print("launch", " ".join(cmd))
proc=subprocess.Popen(cmd, stdout=open("scratchpad/metaqa_2000_modal.log","w",encoding="utf-8"), stderr=open("scratchpad/metaqa_2000_modal.err","w",encoding="utf-8"), env=env, cwd=".")
print(f"PID {proc.pid}")
open("scratchpad/metaqa_2000_modal.pid","w").write(str(proc.pid))
