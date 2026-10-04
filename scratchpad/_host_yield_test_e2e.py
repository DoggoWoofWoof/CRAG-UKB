"""End-to-end checks of _host_yield.py run / wait / guard against a fake rx home and a fake agent."""
import json
import os
import shutil
import subprocess
import sys
import time

SP = "C:/Users/Swastik/AppData/Local/Temp/claude/C--Users-Swastik-Desktop-CRAG/348e519a-c28c-4463-a7fc-3fb109605d7e/scratchpad/"
MOD = "C:/Users/Swastik/Desktop/CRAG/scratchpad/_host_yield.py"
HOME = SP + "fake_rx_home2"
WS = SP + "fake_ws"
shutil.rmtree(HOME, ignore_errors=True)
shutil.rmtree(WS, ignore_errors=True)
for d in ("active", "queue", "agent", "jobs"):
    os.makedirs(os.path.join(HOME, d))
os.makedirs(WS + "/scratchpad")
shutil.copy(MOD, WS + "/scratchpad/_host_yield.py")
open(WS + "/scratchpad/child.py", "w").write(
    "import subprocess, sys, time\n"
    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
    "open('grandchild.pid', 'w').write(str(p.pid))\n"
    "time.sleep(600)\n")
CAP = {"cpus": 32, "mem_gb": 127.7, "gpus": [{"index": 0}], "reserve_cpus": 2, "reserve_mem_gb": 8,
       "priority": {"mpr": 10.0}, "hold_after_s": 1800}
json.dump(CAP, open(HOME + "/host.json", "w"))
# the fake agent: records each request; 'cancel' removes the job's active entry (as rx would once it stops)
open(HOME + "/agent/rx_agent-000000000000.py", "w").write(
    "import json, os, sys\n"
    "req = json.loads(sys.stdin.readline())\n"
    "home = os.environ['RX_HOME']\n"
    "with open(os.path.join(home, 'agent_calls.jsonl'), 'a') as f: f.write(json.dumps(req) + '\\n')\n"
    "if req['op'] == 'cancel':\n"
    "    p = os.path.join(home, 'active', '%s--%s.json' % (req['project'], req['id']))\n"
    "    if os.path.exists(p): os.remove(p)\n"
    "print(json.dumps({'ok': True, 'job': {'state': 'launching'}}))\n")


def jobdir(project, jid, spec):
    jd = os.path.join(HOME, "jobs", project + "--" + jid)
    os.makedirs(jd, exist_ok=True)
    json.dump(spec, open(jd + "/spec.json", "w"))
    return jd


def beat(jd):
    json.dump({"t": time.time()}, open(jd + "/heartbeat.json", "w"))


def entry(kind, project, jid, req, jd, admitted=None):
    e = {"key": "%s--%s" % (project, jid), "project": project, "id": jid, "jd": jd, "req": req, "enq": time.time() - 30}
    if admitted is not None:
        e["admitted"] = admitted
        e["assign"] = {"cpus": req["cpus"], "mem_gb": req["mem_gb"], "gpus": [0] if req["gpus"] else [],
                       "gpu_shares": {"0": 1.0} if req["gpus"] else {}}
    json.dump(e, open(os.path.join(HOME, kind, e["key"] + ".json"), "w"))
    beat(jd)
    return e


def env_for(jid, jd):
    e = {k: v for k, v in os.environ.items() if not k.startswith("RX_")}
    e.update({"RX_HOME": HOME, "RX_JOB_DIR": jd, "RX_PROJECT": "crag", "RX_JOB_ID": jid, "PYTHONUTF8": "1"})
    return e


fails = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name, detail)
    if not cond:
        fails.append(name)


# ---- A. run: a GPU job yields to a queued mpr GPU job, stops its whole tree, requeues a waiter
spec = {"id": "j1", "project": "crag", "name": "crag-encrepro", "argv": ["python", "-u", "scratchpad/_host_yield.py", "run", "--",
        "python", "-u", "scratchpad/child.py"], "cpus": 4.0, "mem_gb": 16.0, "gpus": 1.0, "env": "mpr-cu128",
        "outputs": ["results/L1_HOST/**"], "created": 1.0, "agent": "abc", "_fetch": True, "ws": "ws", "cwd": ".", "shell": "exec",
        "timeout_s": 7200.0, "mem_hard_gb": 32.0, "display": "python -u scratchpad/_host_yield.py run -- ..."}
jd1 = jobdir("crag", "j1", spec)
entry("active", "crag", "j1", {"cpus": 4.0, "mem_gb": 16.0, "gpus": 1.0}, jd1, admitted=time.time() - 60)
t0 = time.time()
p = subprocess.Popen([sys.executable, "-u", "scratchpad/_host_yield.py", "run", "--confirm-s", "2", "--poll-s", "1", "--",
                      "python", "-u", "scratchpad/child.py"], cwd=WS, env=env_for("j1", jd1),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(4)
check("A child running before contention", p.poll() is None)
jdm = jobdir("mpr", "mq", {"id": "mq", "project": "mpr"})
entry("queue", "mpr", "mq", {"cpus": 8.0, "mem_gb": 16.0, "gpus": 1.0}, jdm)
try:
    out, _ = p.communicate(timeout=60)
except subprocess.TimeoutExpired:
    p.kill()
    out, _ = p.communicate()
out = out.decode("utf-8", "replace")
check("A exit 75", p.returncode == 75, "rc %s after %.1fs" % (p.returncode, time.time() - t0))
gpid = int(open(WS + "/grandchild.pid").read())
import psutil
check("A grandchild stopped", not psutil.pid_exists(gpid) or psutil.Process(gpid).status() == psutil.STATUS_ZOMBIE)
calls = [json.loads(l) for l in open(HOME + "/agent_calls.jsonl")]
w = calls[-1]["spec"] if calls and calls[-1]["op"] == "launch" else {}
check("A waiter launched", bool(w) and w["argv"][-1] == "wait" and w["cpus"] == 1e-11 and w["gpus"] == 0.0)
t = w.get("crag_yield_target") or {}
check("A target keeps command/request/env", t.get("argv") == spec["argv"] and t.get("cpus") == 4.0 and t.get("gpus") == 1.0
      and t.get("env") == "mpr-cu128" and t.get("timeout_s") == 7200.0 and t.get("mem_hard_gb") == 32.0)
check("A target has no id/launch fields", all(k not in t for k in ("id", "created", "agent")) and t.get("_fetch") is False)
check("A meta gen 1 + code", (t.get("crag_yield") or {}).get("gen") == 1 and "scratchpad/child.py" in (t["crag_yield"].get("code") or {}))
log = [json.loads(l) for l in open(WS + "/results/HOST_YIELD/crag-encrepro.jsonl")]
check("A log events", [r["event"] for r in log] == ["start", "child", "yielded", "requeued"], [r["event"] for r in log])

# ---- B. wait: blocked while the mpr job is queued, relaunches after it leaves the queue
os.remove(os.path.join(HOME, "active", "crag--j1.json"))
w["crag_yield"]["policy"].update(settle_s=1.0, poll_s=1.0)
jdw = jobdir("crag", w["id"], w)
entry("active", "crag", w["id"], {"cpus": 1e-11, "mem_gb": 1e-11, "gpus": 0.0}, jdw, admitted=time.time())
pw = subprocess.Popen([sys.executable, "-u", "scratchpad/_host_yield.py", "wait"], cwd=WS, env=env_for(w["id"], jdw),
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(5)
n_before = len(open(HOME + "/agent_calls.jsonl").readlines())
check("B waiter holds while mpr queued", pw.poll() is None and n_before == len(calls))
# the mpr job is admitted (moves to active and takes the GPU), then finishes
os.remove(os.path.join(HOME, "queue", "mpr--mq.json"))
entry("active", "mpr", "mq", {"cpus": 8.0, "mem_gb": 16.0, "gpus": 1.0}, jdm, admitted=time.time())
time.sleep(5)
check("B waiter holds while mpr holds the GPU", pw.poll() is None)
os.remove(os.path.join(HOME, "active", "mpr--mq.json"))
beat(jdw)
try:
    outw, _ = pw.communicate(timeout=60)
except subprocess.TimeoutExpired:
    pw.kill()
    outw, _ = pw.communicate()
calls = [json.loads(l) for l in open(HOME + "/agent_calls.jsonl")]
r = calls[-1]["spec"] if calls[-1]["op"] == "launch" else {}
check("B relaunched", pw.returncode == 0 and r.get("argv") == spec["argv"] and r.get("id", "").endswith(r.get("id", "")[-4:])
      and r.get("id") != "j1" and r.get("crag_yield", {}).get("gen") == 1, outw.decode()[-300:])

# ---- C. run at gen 1 with changed code stops the chain (exit 3)
jd2 = jobdir("crag", r["id"], r)
entry("active", "crag", r["id"], {"cpus": 4.0, "mem_gb": 16.0, "gpus": 1.0}, jd2, admitted=time.time())
open(WS + "/scratchpad/child.py", "a").write("# changed\n")
p3 = subprocess.run([sys.executable, "-u", "scratchpad/_host_yield.py", "run", "--confirm-s", "2", "--poll-s", "1", "--",
                     "python", "-u", "scratchpad/child.py"], cwd=WS, env=env_for(r["id"], jd2), capture_output=True, timeout=60)
check("C code change -> exit 3", p3.returncode == 3, p3.stdout.decode()[-200:])
os.remove(os.path.join(HOME, "active", "crag--%s.json" % r["id"]))

# ---- D. guard: an unwrapped CRAG job that must yield is cancelled; a wrapped one is left alone; guard exits after
specU = {"id": "k59", "project": "crag", "name": "2wiki-k59898", "argv": ["python", "-u", "scratchpad/_l1h_host.py", "CELL", "2wiki", "59898"]}
jdU = jobdir("crag", "k59", specU)
entry("active", "crag", "k59", {"cpus": 4.0, "mem_gb": 20.0, "gpus": 0.0}, jdU, admitted=time.time() - 9000)
specG = {"id": "g1", "project": "crag", "name": "crag-guard", "argv": ["python", "-u", "scratchpad/_host_yield.py", "guard"]}
jdG = jobdir("crag", "g1", specG)
entry("active", "crag", "g1", {"cpus": 1e-11, "mem_gb": 1e-11, "gpus": 0.0}, jdG, admitted=time.time())
# host full of jigsaw except what k59 holds; an mpr job needing 28 cpu: 30 total - k59's 4 -> CRAG must yield
jdJ = jobdir("jigsaw", "jj", {"id": "jj", "project": "jigsaw"})
entry("active", "jigsaw", "jj", {"cpus": 26.0, "mem_gb": 99.0, "gpus": 1.0}, jdJ, admitted=time.time() - 100)
pg = subprocess.Popen([sys.executable, "-u", "scratchpad/_host_yield.py", "guard", "--confirm-s", "2", "--poll-s", "1"], cwd=WS,
                      env=env_for("g1", jdG), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(4)
check("D guard quiet without contention", pg.poll() is None and not any(json.loads(l)["op"] == "cancel" for l in open(HOME + "/agent_calls.jsonl")))
entry("queue", "mpr", "mbig", {"cpus": 28.0, "mem_gb": 60.0, "gpus": 0.0}, jdm)
for jd in (jdU, jdG, jdJ, jdm):
    beat(jd)
try:
    outg, _ = pg.communicate(timeout=60)
except subprocess.TimeoutExpired:
    pg.kill()
    outg, _ = pg.communicate()
calls = [json.loads(l) for l in open(HOME + "/agent_calls.jsonl")]
cancels = [c for c in calls if c["op"] == "cancel"]
check("D cancelled k59 only, once", [c["id"] for c in cancels] == ["k59"], cancels)
check("D guard exited 0", pg.returncode == 0, outg.decode()[-400:])

print("fails", len(fails), fails)
