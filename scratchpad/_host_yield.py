"""CRAG's host jobs step aside for mpr's: on the shared rx host the order is mpr > crag > jigsaw.

    rx run -n NAME [--env ENV] --cpus C --mem M [--gpus G] ... -- python -u scratchpad/_host_yield.py run -- python -u scratchpad/X.py ARGS
    rx run -n crag-guard --env mpr-cpu --cpus 1e-11 --mem 1e-11 -- python -u scratchpad/_host_yield.py guard
    rx exec --env mpr-cpu -- python scratchpad/_host_yield.py probe

rx (tools/rx of the message-passing-retrieval checkout) admits a queued job by project priority (host.json
`priority`: mpr 10, any other project 0), then first in, first out with backfill, and never stops a running job.
Jigsaw's jobs step aside for mpr's and crag's through Jigsaw's own launcher (scripts/canonical/launch/lowprio.py,
--yield-to mpr,crag). This module is CRAG's side of the same order:

1. `run` starts the command and reads rx's state (host.json, the live entries of queue/ and active/) every
   --poll-s. The job yields when a live queued job of a project above crag (--above, default mpr, and any project
   rx's priority map ranks above crag) is refused by rx now and would be admitted once
     (a) every active job of the projects below crag (--below, default jigsaw) and
     (b) the fewest CRAG jobs, taken youngest first,
   gave back what they hold, and this job is one of those CRAG jobs. When (a) alone is enough, CRAG keeps running
   and the projects below make room, until the queued job has waited --below-wait-s; from then on (a) is not
   counted, so a project below that does not step aside never keeps mpr waiting on CRAG. A queued job that
   nothing CRAG holds would get admitted (it waits for its own project's jobs, or can never run on this host)
   stops nothing.
2. The contention must persist --confirm-s. The child's process tree is then stopped and `run` exits 75
   (EX_TEMPFAIL). A job that meets it before its child starts yields at once.
3. With --requeue (the default) the yielding job launches a waiter through the host's rx agent (op launch): a job
   asking for 1e-11 CPU and GB and no GPU, below rx's fit tolerance (1e-9), so it never keeps anyone out. The
   waiter relaunches the job unchanged (the same spec under a new id) once rx would admit it and it would not
   have to yield at once, for --settle-s in a row.
4. Every hop checks that the command's .py files and this module hash as at the first hop; a changed file stops
   the chain (exit 3), so no chain mixes code.
5. `guard` watches the CRAG jobs that run without this wrapper (launched before it existed) by the same rule and
   cancels one that must yield (op cancel on the host's agent). It relaunches nothing: such a job may run
   processes inside WSL that a cancel does not reach, so its relaunch is a manual step. It exits once no
   unwrapped CRAG job is active.

Events go to results/HOST_YIELD/<campaign>.jsonl (workspace-relative) and to the job log. The scheduler arithmetic
mirrors rx_agent.py (decide, _fit, _take, _hold_back, project_priority, heartbeat_alive, live_entries) as Jigsaw's
launcher does; rx's state files are only read (a launch or a cancel goes through the agent's own rpc).
"""
import argparse
import glob
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time

EXIT_YIELD = 75                   # EX_TEMPFAIL: the job stepped aside and did not finish
EXIT_CODE_CHANGED = 3
EXIT_GAVE_UP = 4
LOST_AFTER_S = 120.0              # rx_agent.LOST_AFTER_S
WAITER = {"cpus": 1e-11, "mem_gb": 1e-11, "gpus": 0.0}
LAUNCH_FIELDS = ("created", "agent", "_fetch")   # set by rx at launch; a relaunch gets its own
IS_WIN = os.name == "nt"
CREATE_NO_WINDOW = 0x08000000
PROJECT = "crag"
ME = "scratchpad/_host_yield.py"
LOGDIR = "results/HOST_YIELD"


# ---------------------------------------------------------------- rx's scheduler (rx_agent.py), read only

def job_request(spec):
    return {"cpus": float(spec.get("cpus") or 1), "mem_gb": float(spec.get("mem_gb") or 1),
            "gpus": float(spec.get("gpus") or 0)}


def project_priority(cap, project):
    try:
        return float((cap.get("priority") or {}).get(project) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _hold_back(pool, req):
    cpu, mem, g = pool
    g = dict(g)
    need = float(req.get("gpus") or 0)
    for i in sorted(g, key=lambda i: (-g[i], i)):
        if need <= 1e-9:
            break
        take = min(max(g[i], 0.0), need)
        g[i] -= take
        need -= take
    return max(0.0, cpu - req["cpus"]), max(0.0, mem - req["mem_gb"]), g


def _fit(req, cpu, mem, gfree):
    if req["cpus"] > cpu + 1e-9 or req["mem_gb"] > mem + 1e-9:
        return None
    g = req.get("gpus") or 0
    shares = {}
    if g > 0:
        if g < 1:
            cands = sorted((v, i) for i, v in gfree.items() if v >= g - 1e-9)
            if not cands:
                return None
            shares = {cands[0][1]: g}
        else:
            n = int(round(g))
            full = sorted(i for i, v in gfree.items() if v >= 1 - 1e-9)
            if len(full) < n:
                return None
            shares = {i: 1.0 for i in full[:n]}
    return {"cpus": req["cpus"], "mem_gb": req["mem_gb"], "gpus": sorted(shares),
            "gpu_shares": {str(i): s for i, s in shares.items()}}


def _take(pool, assign):
    cpu, mem, g = pool
    g = dict(g)
    for i, s in (assign.get("gpu_shares") or {}).items():
        g[int(i)] = g.get(int(i), 0) - s
    return cpu - assign["cpus"], mem - assign["mem_gb"], g


def total_pool(cap):
    return (cap["cpus"] - cap.get("reserve_cpus", 0), cap["mem_gb"] - cap.get("reserve_mem_gb", 0),
            {int(g["index"]): 1.0 for g in cap.get("gpus") or []})


def queue_order(cap, q):
    return (-project_priority(cap, q.get("project")), float(q.get("enq") or 0), str(q.get("key") or ""))


def decide(me, cap, active, queue, now):
    """rx_agent.decide: (admit, assignment, reason) for queue entry `me`."""
    req = me["req"]
    if _fit(req, *total_pool(cap)) is None:
        return False, None, "impossible on this host"
    pool = total_pool(cap)
    for a in active:
        pool = _take(pool, a.get("assign") or {"cpus": a["req"]["cpus"], "mem_gb": a["req"]["mem_gb"]})
    if _fit(req, *pool) is None:
        return False, None, "waiting for resources"
    hold = cap.get("hold_after_s", 1800)
    mine_p = project_priority(cap, me.get("project"))
    mine_key = queue_order(cap, me)
    higher = None
    for q in sorted(queue, key=lambda q: queue_order(cap, q)):
        if queue_order(cap, q) >= mine_key:
            continue
        above = project_priority(cap, q.get("project")) > mine_p
        higher = higher or (q if above else None)
        qa = _fit(q["req"], *pool)
        if qa is not None:
            pool = _take(pool, qa)
        elif above:
            pool = _hold_back(pool, q["req"])
        elif now - float(q.get("enq") or now) > hold:
            return False, None, "holding for older job %s" % q.get("id")
    mine = _fit(req, *pool)
    if mine is None:
        if higher is not None:
            return False, None, "waiting for higher-priority job %s (project %s)" % (higher.get("id"), higher.get("project"))
        return False, None, "waiting behind older queued jobs"
    return True, mine, None


# ---------------------------------------------------------------- rx's state files

def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def boot_time():
    try:
        import psutil
        return float(psutil.boot_time())
    except Exception:
        return 0.0


def heartbeat_alive(jd, now, boot):
    hb = read_json(os.path.join(jd, "heartbeat.json")) if jd else None
    if not hb:
        return False
    t = float(hb.get("t") or 0)
    return now - t <= LOST_AFTER_S and not (boot and t < boot - 5)


def live_entries(home, kind, now, boot):
    """rx's active/ or queue/ entries whose supervisor still beats (stale files are left to rx)."""
    out = []
    d = os.path.join(home, kind)
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    for n in names:
        if n.endswith(".json"):
            e = read_json(os.path.join(d, n))
            if isinstance(e, dict) and "req" in e and heartbeat_alive(e.get("jd", ""), now, boot):
                out.append(e)
    return out


class View:
    """One snapshot of host.json, active/ and queue/ (a job seen in both, mid-admission, counts as active)."""

    def __init__(self, home, now=None, cap=None, active=None, queue=None):
        self.home = home
        self.now = time.time() if now is None else now
        if cap is None:
            boot = boot_time()
            cap = read_json(os.path.join(home, "host.json")) or {}
            active = live_entries(home, "active", self.now, boot)
            queue = live_entries(home, "queue", self.now, boot)
        keys = {a.get("key") for a in active}
        self.cap, self.active, self.queue = cap, list(active), [q for q in queue if q.get("key") not in keys]

    @property
    def usable(self):
        return bool(self.cap.get("cpus")) and bool(self.cap.get("mem_gb"))


# ---------------------------------------------------------------- the rule

def outranks(cap, other, above):
    return other != PROJECT and (other in above or project_priority(cap, other) > project_priority(cap, PROJECT))


def must_yield(view, key, above, below, below_wait_s):
    """The queued job the active CRAG job `key` must make room for, or None (rule 1 of the module docstring)."""
    cap, active, queue, now = view.cap, view.active, view.queue, view.now
    ours = sorted((a for a in active if a.get("project") == PROJECT),
                  key=lambda a: (a.get("admitted") or 0, a.get("key") or ""), reverse=True)   # youngest first
    if not any(a.get("key") == key for a in ours):
        return None
    lower = [a for a in active if a.get("project") in below]

    def admitted(q, released):
        gone = {a.get("key") for a in released}
        return decide(q, cap, [a for a in active if a.get("key") not in gone], queue, now)[0]

    total = total_pool(cap)
    for q in sorted(queue, key=lambda q: queue_order(cap, q)):
        if not outranks(cap, q.get("project"), above):
            continue
        if _fit(q["req"], *total) is None or admitted(q, []):
            continue
        waited = now - float(q.get("enq") or now)
        room = lower if waited < below_wait_s else []
        if room and admitted(q, room):
            continue                                   # the projects below crag make room for it
        chosen = []
        for a in ours:
            chosen.append(a)
            if admitted(q, room + chosen):
                break
        else:
            continue                                   # nothing CRAG holds gets it admitted
        for a in list(chosen):
            rest = [c for c in chosen if c is not a]
            if admitted(q, room + rest):
                chosen = rest
        if any(a.get("key") == key for a in chosen):
            return {"job": q.get("id"), "project": q.get("project"), "req": dict(q["req"]), "queued_s": round(waited, 1),
                    "crag_jobs_released": [a.get("id") for a in chosen], "below_counted": bool(room)}
    return None


def launch_blocker(view, req, above, below, below_wait_s):
    """Why a CRAG job asking `req` should not be launched now, or None: rx must admit it as a job queued now, and
    once running it must not be one that must_yield would stop."""
    me = {"key": PROJECT + "--next-hop", "project": PROJECT, "id": "next-hop", "req": dict(req), "enq": view.now}
    ok, assign, why = decide(me, view.cap, view.active, view.queue, view.now)
    if not ok:
        return why
    trial = View(view.home, view.now, view.cap, list(view.active) + [dict(me, assign=assign, admitted=view.now)], view.queue)
    c = must_yield(trial, me["key"], above, below, below_wait_s)
    return "it would have to yield to %s job %s" % (c["project"], c["job"]) if c else None


# ---------------------------------------------------------------- the host's rx agent

def rx_python(home):
    found = sorted(glob.glob(os.path.join(home, "python", "*", "python.exe" if IS_WIN else os.path.join("bin", "python3"))))
    return found[-1] if found else sys.executable


def agent_path(home):
    found = sorted(glob.glob(os.path.join(home, "agent", "rx_agent-*.py")), key=os.path.getmtime)
    if not found:
        raise RuntimeError("no rx agent under %s" % os.path.join(home, "agent"))
    return found[-1]


def agent_call(home, request, timeout=300.0):
    """One request to the host's rx agent (`rx_agent.py rpc`), as rx.py sends it over ssh."""
    env = dict(os.environ)
    env["RX_HOME"] = home
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "VIRTUAL_ENV", "RX_LAUNCH"):
        env.pop(k, None)
    p = subprocess.run([rx_python(home), agent_path(home), "rpc"],
                       input=(json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8"),
                       capture_output=True, timeout=timeout, env=env,
                       **({"creationflags": CREATE_NO_WINDOW} if IS_WIN else {}))
    lines = p.stdout.decode("utf-8", "replace").splitlines()
    try:
        res = json.loads(lines[0]) if lines else None
    except ValueError:
        res = None
    if not isinstance(res, dict) or not res.get("ok"):
        raise RuntimeError("rx agent %s: rc %s, %s %s" % (request.get("op"), p.returncode,
                                                         (res or {}).get("error") if isinstance(res, dict) else None,
                                                         p.stderr.decode("utf-8", "replace")[-300:]))
    return res


def make_id(name):
    """rx.make_id: yymmdd-HHMMSS-<name>-<4 hex>."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "job").strip("-._")[:40] or "job"
    return "%s-%s-%s" % (time.strftime("%y%m%d-%H%M%S"), name, secrets.token_hex(2))


# ---------------------------------------------------------------- this job

class Ctx:
    """What rx tells a job about itself (RX_* variables and its job directory)."""

    def __init__(self):
        e = os.environ
        self.home, self.jd, self.project, self.jid = e.get("RX_HOME"), e.get("RX_JOB_DIR"), e.get("RX_PROJECT"), e.get("RX_JOB_ID")
        self.spec = read_json(os.path.join(self.jd, "spec.json"), {}) if self.jd else {}

    @property
    def under_rx(self):
        return bool(self.home and self.jd and self.project == PROJECT and self.jid and self.spec)

    @property
    def key(self):
        return "%s--%s" % (self.project, self.jid)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_of(cmd):
    """sha256 of this module and of every .py file the command names (workspace-relative)."""
    out = {ME: sha_file(os.path.abspath(__file__))}
    out.update({c.replace("\\", "/"): sha_file(c) for c in cmd if c.endswith(".py") and os.path.isfile(c)})
    return out


def log_event(campaign, event, **fields):
    rec = {"t": round(time.time(), 3), "local": time.strftime("%Y-%m-%dT%H:%M:%S"), "campaign": campaign, "event": event}
    rec.update(fields)
    print("[yield %s] %s %s" % (time.strftime("%H:%M:%S"), event, json.dumps(fields, sort_keys=True, default=str)), flush=True)
    try:
        os.makedirs(LOGDIR, exist_ok=True)
        with open(os.path.join(LOGDIR, "%s.jsonl" % campaign), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    except OSError as exc:
        print("[yield] cannot write the campaign log: %s" % exc, flush=True)


def start_child(cmd):
    cmd = list(cmd)
    if cmd and cmd[0] in ("python", "python3", "py"):
        cmd[0] = sys.executable                        # as rx maps a bare python to the job's environment
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL)


def stop_tree(proc, timeout=60.0):
    """Stop the child and its descendants; returns how many processes were asked to stop."""
    try:
        import psutil
    except ImportError:
        if IS_WIN:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        return -1
    try:
        parent = psutil.Process(proc.pid)
        procs = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return 0
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=timeout)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    return len(procs)


def policy_of(args):
    return {"above": list(args.above), "below": list(args.below), "below_wait_s": float(args.below_wait_s),
            "confirm_s": float(args.confirm_s), "poll_s": float(args.poll_s), "settle_s": float(args.settle_s)}


def target_spec(spec, meta):
    """The next hop of this job: the same spec (command, request, workspace, environment) without its id."""
    t = {k: v for k, v in spec.items() if k not in LAUNCH_FIELDS and k not in ("id", "crag_yield_target")}
    t.update({"crag_yield": dict(meta, role="job"), "rerun_of": meta["origin"], "_fetch": False})
    return t


def waiter_spec(spec, target, meta):
    name = "%s-wait" % (spec.get("name") or "job")
    w = {k: v for k, v in spec.items() if k not in LAUNCH_FIELDS and k not in ("crag_yield_target",)}
    w.update({"id": make_id(name), "name": name, "shell": "exec", "command": None, "argv": ["python", "-u", ME, "wait"],
              "cpus": WAITER["cpus"], "mem_gb": WAITER["mem_gb"], "gpus": WAITER["gpus"], "mem_hard_gb": None,
              "timeout_s": None, "outputs": [LOGDIR + "/**"], "outputs_exclude": [],
              "display": "yield wait -> %s" % (target.get("display") or target.get("name")),
              "rerun_of": meta["origin"], "crag_yield": dict(meta, role="waiter"), "crag_yield_target": dict(target),
              "_fetch": False})
    return w


def requeue(ctx, meta, reason, args):
    campaign = meta["campaign"]
    if not args.requeue:
        log_event(campaign, "no_requeue", job=ctx.jid, why="--no-requeue")
        return None
    if not ctx.under_rx:
        log_event(campaign, "no_requeue", job=ctx.jid, why="not a crag rx job")
        return None
    gen = int(meta.get("gen", 0)) + 1
    if gen > args.max_requeues:
        log_event(campaign, "no_requeue", job=ctx.jid, why="requeue %d exceeds --max-requeues %d" % (gen, args.max_requeues))
        return None
    nmeta = dict(meta, gen=gen, last_yield={"t": round(time.time(), 3), "reason": reason})
    target = target_spec(ctx.spec, nmeta)
    w = waiter_spec(ctx.spec, target, nmeta)
    for attempt in range(3):
        try:
            res = agent_call(ctx.home, {"op": "launch", "spec": w})
            log_event(campaign, "requeued", job=ctx.jid, waiter=w["id"], gen=gen, waiter_state=(res.get("job") or {}).get("state"))
            return w["id"]
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            err = str(exc)
            time.sleep(5.0 * (attempt + 1))
    log_event(campaign, "requeue_failed", job=ctx.jid, error=err)
    return None


# ---------------------------------------------------------------- commands

def cmd_run(args):
    ctx = Ctx()
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("_host_yield run: nothing to run (run [options] -- COMMAND ...)")
    meta = dict(ctx.spec.get("crag_yield") or {})
    if not meta:
        meta = {"campaign": args.campaign or ctx.spec.get("name") or "local", "gen": 0, "origin": ctx.jid or "local",
                "code": code_of(cmd), "policy": policy_of(args)}
    campaign = meta["campaign"]
    log_event(campaign, "start", job=ctx.jid, gen=meta["gen"], under_rx=ctx.under_rx, cmd=cmd, policy=policy_of(args))
    if meta["gen"] > 0:
        now_code = code_of(cmd)
        if now_code != meta["code"]:
            log_event(campaign, "code_changed", job=ctx.jid, first_hop=meta["code"], now=now_code)
            return EXIT_CODE_CHANGED
    above, below = list(args.above), list(args.below)

    def sample():
        if not ctx.under_rx:
            return None
        v = View(ctx.home)
        return must_yield(v, ctx.key, above, below, args.below_wait_s) if v.usable else None

    reason = sample()
    if reason:
        log_event(campaign, "yielded", job=ctx.jid, phase="before start", reason=reason)
        requeue(ctx, meta, reason, args)
        return EXIT_YIELD
    child = start_child(cmd)
    log_event(campaign, "child", job=ctx.jid, pid=child.pid)
    since = None
    while True:
        rc = child.poll()
        if rc is not None:
            log_event(campaign, "done" if rc == 0 else "child_failed", job=ctx.jid, rc=rc)
            return rc
        now = time.time()
        try:
            reason = sample()
        except Exception as exc:                       # a torn read of rx's state is retried at the next poll
            print("[yield] sample failed: %s: %s" % (type(exc).__name__, exc), flush=True)
            reason = None
        if reason:
            since = since or now
            if now - since >= args.confirm_s:
                rc = child.poll()
                if rc is not None:
                    log_event(campaign, "done" if rc == 0 else "child_failed", job=ctx.jid, rc=rc, note="ended while yielding")
                    return rc
                stopped = stop_tree(child)
                log_event(campaign, "yielded", job=ctx.jid, phase="running", stopped=stopped,
                          confirmed_s=round(time.time() - since, 1), reason=reason, child_rc=child.returncode)
                requeue(ctx, meta, reason, args)
                return EXIT_YIELD
        else:
            since = None
        try:
            child.wait(timeout=args.poll_s)
        except subprocess.TimeoutExpired:
            pass


def cmd_wait(args):
    ctx = Ctx()
    if not ctx.under_rx or "crag_yield_target" not in ctx.spec:
        raise SystemExit("_host_yield wait runs only as the waiter a yielding `run` launches")
    meta = ctx.spec["crag_yield"]
    target = dict(ctx.spec["crag_yield_target"])
    pol = meta["policy"]
    campaign = meta["campaign"]
    req = job_request(target)
    log_event(campaign, "waiting", job=ctx.jid, gen=meta["gen"], target_req=req)
    t0, ready, last = time.time(), None, None
    while True:
        now = time.time()
        if now - t0 > args.give_up_s:
            log_event(campaign, "gave_up", job=ctx.jid, waited_s=round(now - t0, 1))
            return EXIT_GAVE_UP
        try:
            v = View(ctx.home)
            why = launch_blocker(v, req, pol["above"], pol["below"], pol["below_wait_s"]) if v.usable else "host.json unreadable"
        except Exception as exc:
            why = "sample failed: %s: %s" % (type(exc).__name__, exc)
        if why:
            ready = None
            if why != last:
                print("[yield %s] waiting: %s" % (time.strftime("%H:%M:%S"), why), flush=True)
                last = why
            time.sleep(pol["poll_s"])
            continue
        ready = ready or now
        if now - ready < pol["settle_s"]:
            time.sleep(pol["poll_s"])
            continue
        spec = dict(target, id=make_id(str(target.get("name") or "job")))
        try:
            agent_call(ctx.home, {"op": "launch", "spec": spec})
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            log_event(campaign, "relaunch_failed", job=ctx.jid, error=str(exc))
            ready = None
            time.sleep(30.0)
            continue
        log_event(campaign, "relaunched", job=ctx.jid, relaunch=spec["id"], gen=meta["gen"])
        return 0


def wrapped(entry):
    spec = read_json(os.path.join(entry.get("jd") or "", "spec.json"), {}) or {}
    argv = [str(a).replace("\\", "/") for a in (spec.get("argv") or [])]
    return any(a.endswith(ME) for a in argv), spec


def cmd_guard(args):
    ctx = Ctx()
    if not ctx.under_rx:
        raise SystemExit("_host_yield guard runs as a crag rx job")
    campaign = args.campaign or ctx.spec.get("name") or "guard"
    above, below = list(args.above), list(args.below)
    log_event(campaign, "guard_start", job=ctx.jid, policy=policy_of(args))
    since = {}
    while True:
        now = time.time()
        try:
            v = View(ctx.home)
        except Exception as exc:
            print("[yield] sample failed: %s: %s" % (type(exc).__name__, exc), flush=True)
            time.sleep(args.poll_s)
            continue
        watched = []
        for a in v.active:
            if a.get("project") != PROJECT or a.get("key") == ctx.key:
                continue
            is_wrapped, spec = wrapped(a)
            if not is_wrapped:
                watched.append((a, spec))
        if not watched:
            log_event(campaign, "guard_done", job=ctx.jid, why="no unwrapped crag job is active")
            return 0
        for a, spec in watched:
            reason = must_yield(v, a.get("key"), above, below, args.below_wait_s) if v.usable else None
            if not reason:
                since.pop(a.get("key"), None)
                continue
            since.setdefault(a.get("key"), now)
            if now - since[a.get("key")] < args.confirm_s:
                continue
            try:
                res = agent_call(ctx.home, {"op": "cancel", "project": PROJECT, "id": a.get("id")})
                log_event(campaign, "guard_cancelled", job=a.get("id"), argv=spec.get("argv"), reason=reason,
                          state=(res.get("job") or {}).get("state"), relaunch="manual (see the module docstring, rule 5)")
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                log_event(campaign, "guard_cancel_failed", job=a.get("id"), error=str(exc))
            since.pop(a.get("key"), None)
        time.sleep(args.poll_s)


def cmd_probe(args):
    home = os.environ.get("RX_HOME") or args.home
    v = View(home)
    out = {"now": v.now, "capacity": {k: v.cap.get(k) for k in ("cpus", "mem_gb", "reserve_cpus", "reserve_mem_gb", "priority", "hold_after_s")},
           "active": [{k: a.get(k) for k in ("project", "id", "req", "assign", "admitted")} for a in v.active],
           "queue": [{k: q.get(k) for k in ("project", "id", "req", "enq")} for q in v.queue],
           "must_yield": {a.get("id"): must_yield(v, a.get("key"), list(args.above), list(args.below), args.below_wait_s)
                          for a in v.active if a.get("project") == PROJECT},
           "wrapped": {a.get("id"): wrapped(a)[0] for a in v.active if a.get("project") == PROJECT}}
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


def projects(text):
    names = [p.strip() for p in str(text).split(",") if p.strip()]
    if not all(re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$", n) for n in names):
        raise argparse.ArgumentTypeError("%r: expected PROJECT[,PROJECT...]" % text)
    return names


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    def policy(s):
        s.add_argument("--above", type=projects, default=["mpr"], help="projects CRAG yields to (beside any rx ranks above crag)")
        s.add_argument("--below", type=projects, default=["jigsaw"], help="projects that yield to CRAG and make room first")
        s.add_argument("--below-wait-s", type=float, default=120.0,
                       help="how long a queued job may wait for the projects below before CRAG's jobs are counted alone")
        s.add_argument("--confirm-s", type=float, default=10.0, help="how long contention must persist")
        s.add_argument("--poll-s", type=float, default=5.0)
        s.add_argument("--settle-s", type=float, default=20.0, help="waiter: how long the host must have room")

    r = sub.add_parser("run", help="run a command, yielding to the projects above crag")
    r.add_argument("--campaign", default=None, help="name of the chain of hops (default: the rx job name)")
    r.add_argument("--no-requeue", dest="requeue", action="store_false", help="after a yield, relaunch nothing")
    r.add_argument("--max-requeues", type=int, default=50)
    policy(r)
    r.add_argument("cmd", nargs=argparse.REMAINDER)

    w = sub.add_parser("wait", help="the waiter job (launched by a yielding run)")
    w.add_argument("--give-up-s", type=float, default=172800.0)

    g = sub.add_parser("guard", help="cancel an unwrapped crag job that must yield")
    g.add_argument("--campaign", default=None)
    policy(g)

    q = sub.add_parser("probe", help="print the host as the rule sees it")
    q.add_argument("--home", default=os.path.join(os.path.expanduser("~"), "rx"))
    policy(q)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"run": cmd_run, "wait": cmd_wait, "guard": cmd_guard, "probe": cmd_probe}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
