import os
import subprocess
import sys
import time

sys.path.insert(0, "C:/Users/Swastik/Desktop/CRAG/scratchpad")
import _host_yield as Y

CAP = {"cpus": 32, "mem_gb": 127.7, "gpus": [{"index": 0}], "reserve_cpus": 2, "reserve_mem_gb": 8,
       "priority": {"mpr": 10.0}, "hold_after_s": 1800}
NOW = 1_000_000.0


def act(project, jid, cpus, mem, gpus=0, admitted=0.0):
    req = {"cpus": float(cpus), "mem_gb": float(mem), "gpus": float(gpus)}
    return {"key": "%s--%s" % (project, jid), "project": project, "id": jid, "req": req, "enq": admitted - 1,
            "admitted": admitted, "assign": Y._fit(req, 1e9, 1e9, {0: 1.0})}


def que(project, jid, cpus, mem, gpus=0, enq=NOW - 20):
    return {"key": "%s--%s" % (project, jid), "project": project, "id": jid,
            "req": {"cpus": float(cpus), "mem_gb": float(mem), "gpus": float(gpus)}, "enq": enq}


def view(active, queue, now=NOW):
    return Y.View("-", now, CAP, active, queue)


def my(v, jid):
    return Y.must_yield(v, "crag--" + jid, ["mpr"], ["jigsaw"], 120.0)


ok = 0
fails = []


def check(name, got, want):
    global ok
    if got == want:
        ok += 1
    else:
        fails.append((name, got, want))


K59 = act("crag", "k59898", 4, 20, 0, admitted=100)
REPRO = act("crag", "repro", 4, 16, 1, admitted=900)
JMAG = act("jigsaw", "dev-mag", 3, 40, 0, admitted=500)          # (holds no GPU while REPRO runs)
JCORA = act("jigsaw", "dev-cora", 6, 16, 0, admitted=600)
JWSL = act("jigsaw", "wsl", 8, 12, 0, admitted=700)
base = [K59, REPRO, JMAG, JCORA, JWSL]                             # 25 cpu, 104 GB, gpu held by REPRO

# 1. empty queue: nobody yields
v = view(base, [])
check("1 k59", my(v, "k59898"), None)
check("1 repro", my(v, "repro"), None)

# 2. mpr GPU job queued: jigsaw holds no GPU -> REPRO (the only GPU holder) yields, K59898 stays
v = view(base, [que("mpr", "gpu1", 8, 16, 1)])
r = my(v, "repro")
check("2 repro yields", bool(r) and r["job"] == "gpu1" and r["crag_jobs_released"] == ["repro"], True)
check("2 k59 stays", my(v, "k59898"), None)

# 3. mpr CPU job that fits the free pool now (5 cpu, 15.7 GB free) -> nobody yields
v = view(base, [que("mpr", "small", 4, 8)])
check("3 repro", my(v, "repro"), None)
check("3 k59", my(v, "k59898"), None)

# 4. mpr job jigsaw alone can make room for (needs 20 cpu; free 5 + jigsaw 17 = 22) -> CRAG stays
v = view(base, [que("mpr", "mid", 20, 60)])
check("4 repro", my(v, "repro"), None)
check("4 k59", my(v, "k59898"), None)

# 5. the same job after 200 s (> below_wait_s): jigsaw not counted; crag alone frees 8 cpu -> 13 < 20 -> no yield
v = view(base, [que("mpr", "mid", 20, 60, enq=NOW - 200)])
check("5 repro", my(v, "repro"), None)
check("5 k59", my(v, "k59898"), None)

# 5b. a job CRAG alone can make room for after 200 s: needs 8 cpu 30 GB; free 5/15.7; REPRO gives 4/16 -> 9/31.7 -> REPRO only
v = view(base, [que("mpr", "c8", 8, 30, enq=NOW - 200)])
r = my(v, "repro")
check("5b repro yields", bool(r) and r["crag_jobs_released"] == ["repro"] and r["below_counted"] is False, True)
check("5b k59 stays", my(v, "k59898"), None)
# ... and before 120 s jigsaw is counted: jigsaw's 17 cpu suffice -> CRAG stays
v = view(base, [que("mpr", "c8", 8, 30, enq=NOW - 20)])
check("5c repro stays", my(v, "repro"), None)

# 6. mpr job needing 28 cpu: free 5 + jigsaw 17 = 22 < 28; + REPRO 4 = 26 < 28; + K59 4 = 30 -> both yield
v = view(base, [que("mpr", "big", 28, 60)])
r1, r2 = my(v, "repro"), my(v, "k59898")
check("6 both yield", bool(r1) and bool(r2) and sorted(r1["crag_jobs_released"]) == ["k59898", "repro"], True)

# 6b. 26 cpu: jigsaw 17 + free 5 = 22; + REPRO (youngest) 4 = 26 -> only REPRO
v = view(base, [que("mpr", "big26", 26, 60)])
check("6b repro yields", bool(my(v, "repro")), True)
check("6b k59 stays", my(v, "k59898"), None)

# 6c. prune: 24 cpu 100 GB; youngest-first picks REPRO (4/16) then K59 (4/20); memory needs K59's 20 but not REPRO's?
#     free mem 15.7 + jigsaw 68 = 83.7; +REPRO 16 = 99.7 < 100; +K59 20 = 119.7 -> both chosen; prune REPRO: 83.7+20 = 103.7 >= 100 and cpu 5+17+4 = 26 >= 24 -> REPRO pruned
v = view(base, [que("mpr", "mem100", 24, 100)])
check("6c repro pruned", my(v, "repro"), None)
r = my(v, "k59898")
check("6c k59 yields", bool(r) and r["crag_jobs_released"] == ["k59898"], True)

# 7. impossible mpr job (40 cpu) -> nobody yields
v = view(base, [que("mpr", "huge", 40, 10)])
check("7 repro", my(v, "repro"), None)
check("7 k59", my(v, "k59898"), None)

# 8. mpr job blocked by mpr's own running jobs (needs the GPU mpr holds) -> nobody yields
M1 = act("mpr", "m1", 8, 16, 1, admitted=950)
base8 = [K59, act("crag", "cpuonly", 4, 16, 0, admitted=900), JMAG, JCORA, JWSL, M1]
v = view(base8, [que("mpr", "gpu2", 8, 16, 1)])
check("8 cpuonly", my(v, "cpuonly"), None)
check("8 k59", my(v, "k59898"), None)

# 9. jigsaw queued (below crag): never makes CRAG yield
v = view(base, [que("jigsaw", "jq", 30, 100, 1, enq=NOW - 5000)])
check("9 repro", my(v, "repro"), None)

# 10. launch_blocker: REPRO-sized relaunch while an mpr GPU job waits -> blocked; with an empty queue -> free?
act10 = [K59, JMAG, JCORA, JWSL]            # 21 cpu 88 GB used; free 9 / 31.7, GPU free
v = view(act10, [que("mpr", "gpu3", 8, 16, 1)])
check("10 blocked", Y.launch_blocker(v, {"cpus": 4, "mem_gb": 16, "gpus": 1}, ["mpr"], ["jigsaw"], 120.0) is not None, True)
v = view(act10, [])
check("10 free", Y.launch_blocker(v, {"cpus": 4, "mem_gb": 16, "gpus": 1}, ["mpr"], ["jigsaw"], 120.0), None)
# a CPU-only relaunch beside a waiting mpr GPU job that fits now is fine
v = view(act10 + [act("mpr", "m2", 4, 8, 1, admitted=990)], [])
check("10 cpu relaunch", Y.launch_blocker(v, {"cpus": 4, "mem_gb": 8, "gpus": 0}, ["mpr"], ["jigsaw"], 120.0), None)

# 11. a job seen in both queue and active (mid-admission) counts as active only
m3 = act("mpr", "m3", 8, 16, 1, admitted=999)
v = view([K59, JMAG, JCORA, JWSL, m3], [que("mpr", "m3", 8, 16, 1)])
check("11 dedupe", [q["id"] for q in v.queue], [])

# 12. rx's own decide on the same inputs agrees with the mirror (import the local agent)
sys.path.insert(0, "C:/Users/Swastik/Desktop/message-passing-retrieval/tools/rx")
import rx_agent as RA
cases = [(que("mpr", "a", 8, 16, 1), base, []), (que("mpr", "b", 20, 60), base, []), (que("crag", "c", 4, 16, 1), act10, [que("mpr", "d", 8, 16, 1, enq=NOW - 50)]),
         (que("crag", "e", 4, 16, 0), act10, [que("crag", "f", 30, 10, enq=NOW - 4000)]), (que("jigsaw", "g", 2, 2), base, [])]
for i, (me, A, Q) in enumerate(cases):
    a = RA.decide(me, CAP, A, Q, NOW)
    b = Y.decide(me, CAP, A, Q, NOW)
    check("12 decide %d" % i, (a[0], a[1]), (b[0], b[1]))

# 13. the command-line surface: probe on a fake home, and run with no rx context runs the child and returns its rc
home = "C:/Users/Swastik/AppData/Local/Temp/claude/C--Users-Swastik-Desktop-CRAG/348e519a-c28c-4463-a7fc-3fb109605d7e/scratchpad/fake_rx_home"
os.makedirs(home + "/active", exist_ok=True)
os.makedirs(home + "/queue", exist_ok=True)
import json
json.dump(CAP, open(home + "/host.json", "w"))
p = subprocess.run([sys.executable, "C:/Users/Swastik/Desktop/CRAG/scratchpad/_host_yield.py", "probe", "--home", home],
                   capture_output=True, text=True, env={k: v for k, v in os.environ.items() if not k.startswith("RX_")})
check("13 probe rc", p.returncode, 0)
env = {k: v for k, v in os.environ.items() if not k.startswith("RX_")}
p = subprocess.run([sys.executable, "C:/Users/Swastik/Desktop/CRAG/scratchpad/_host_yield.py", "run", "--no-requeue", "--",
                    "python", "-c", "import sys; sys.exit(7)"], capture_output=True, text=True, env=env,
                   cwd="C:/Users/Swastik/AppData/Local/Temp/claude/C--Users-Swastik-Desktop-CRAG/348e519a-c28c-4463-a7fc-3fb109605d7e/scratchpad")
check("13 run rc passthrough", p.returncode, 7)
print(p.stdout[-600:], p.stderr[-600:])

print("ok", ok, "fails", len(fails))
for f in fails:
    print("FAIL", f)
