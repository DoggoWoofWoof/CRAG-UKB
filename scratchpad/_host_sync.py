"""Laptop-side periodic sync of everything the lab host produces, so that losing the host loses nothing:

  records   small result / log / checkpoint files  ->  the laptop working tree        (rx fetch of an incremental zip made by _host_pack_records.py; never overwrites a laptop file that differs from the
                                                                                         last host copy -- those go to work/HOST_HOUSEKEEPING/conflicts/<stamp>/)
  big data  new / grown files (encode chunks, new run outputs, V-cycle levels in the WSL tree)  ->  the private HF dataset repo  (token stays on the laptop; the host uploads through presigned urls, see _hfx.py)

  python -u scratchpad/_host_sync.py [--once] [--interval-min 180] [--repo Swastik9895/crag-host-backup] [--max-repo-gb 80] [--no-records] [--no-big]

Each cycle is idempotent and resumable (state: work/HOST_HOUSEKEEPING/sync_state.json, log: sync_log.jsonl, summary: SYNC_STATUS.json).  Host jobs run under scratchpad/_host_yield.py with small measured reservations.
Files younger than 15 min are never backed up (a writer may still hold them).  Restore on another host: scratchpad/_hfx.py GETURLS / FETCH (see results/HOST_HOUSEKEEPING/RESTORE_RUNBOOK__v1.md)."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "work", "HOST_HOUSEKEEPING")
RX = [sys.executable, "C:/Users/Swastik/Desktop/message-passing-retrieval/tools/rx/rx.py"]
STATE_P = os.path.join(OUT, "sync_state.json")
LOG_P = os.path.join(OUT, "sync_log.jsonl")
STATUS_P = os.path.join(OUT, "SYNC_STATUS.json")
KNOWN_P = os.path.join("data", "_cache", "hfx_known.json")
WS_GROUPS = ["fbs_enc", "fbs_enc_meta", "recs_big", "work_big", "l1parts", "fbs_names", "fbs_pq", "fbs_ivft"]      # fbs_ner / calib are final on HF and retired from the host (_host_rm.py PHASE4)
WSL_GROUPS = ["wsl_fb_l1", "wsl_fb_work"]                                 # + wsl_fb (the 17 GB CSR) only with --with-csr
ENV = dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONIOENCODING="utf-8")


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def rx(*args, timeout=900):
    r = subprocess.run(RX + list(args), cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return r.returncode, r.stdout


def job(name, inputs, cmd, mem=1, cpus=1, timeout_s=3600):
    """submit a host job under the yield wrapper, wait for it, return (job id, rc)"""
    a = ["run", "-n", name, "--cpus", str(cpus), "--mem", str(mem), "--timeout", str(timeout_s)]
    for i in inputs:
        a += ["--inputs", i]
    a += ["--", "python", "-u", "scratchpad/_host_yield.py", "run", "--"] + cmd
    rc, out = rx(*a, timeout=300)
    ids = re.findall(r"^\d{6}-\d{6}-\S+$", out, re.M)
    if rc != 0 or not ids:
        raise RuntimeError("rx run failed: %s" % out[-400:])
    jid = ids[-1]
    for _ in range(int(timeout_s / 300) + 3):
        try:
            rc, out = rx("wait", jid, timeout=1200)
        except subprocess.TimeoutExpired:
            continue
        m = re.search(r"done rc=(-?\d+)", out)
        if m:
            return jid, int(m.group(1))
        if re.search(r"(failed|cancelled|timeout|killed)", out.splitlines()[-1] if out.strip() else ""):
            return jid, 99
        time.sleep(20)
    return jid, 98


def fetch(jid, glob):
    for k in range(4):
        rc, out = rx("fetch", jid, "--glob", glob, timeout=1800)
        if rc == 0:
            return out
        time.sleep(30 * (k + 1))
    raise RuntimeError("rx fetch failed: %s" % out[-300:])


def py(*args, env_extra=None, timeout=3600):
    r = subprocess.run([sys.executable, os.path.join(HERE, "_hfx.py")] + list(args), cwd=ROOT, env=dict(ENV, **(env_extra or {})), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("_hfx.py %s failed: %s" % (args[0], r.stdout[-600:]))
    return r.stdout


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_state():
    return json.load(open(STATE_P)) if os.path.exists(STATE_P) else {"host_epoch": 0, "synced": {}, "cycles": 0, "backed_up_gb": 0.0}


def save_state(st):
    json.dump(st, open(STATE_P, "w"))


# ------------------------------------------------------------------ records
def step_records(st):
    stem = "sync_%s" % time.strftime("%y%m%d_%H%M%S")
    since = int(st["host_epoch"] - 3600) if st["host_epoch"] else int(os.path.getmtime(os.path.join(OUT, "host_records.zip")) - 3600)
    jid, rc = job("sync-rec", ["scratchpad/_host_pack_records.py"], ["python", "-u", "scratchpad/_host_pack_records.py", "since=%d" % since, "out=" + stem], timeout_s=1800)
    assert rc == 0, "pack job rc %s" % rc
    fetch(jid, "work/HOST_HOUSEKEEPING/%s.*" % stem)
    man = json.load(open(os.path.join(OUT, stem + ".manifest.json")))
    z = zipfile.ZipFile(os.path.join(OUT, stem + ".zip"))
    new = same = upd = conf = 0
    cdir = os.path.join(OUT, "conflicts", stem)
    for f in man["files"]:
        rel = f["rel"]
        dst = os.path.join(ROOT, rel.replace("/", os.sep))
        data = z.read(rel)
        assert hashlib.sha256(data).hexdigest() == f["sha"], "zip member %s differs from the host manifest" % rel
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(data)
            new += 1
        elif sha(dst) == f["sha"]:
            same += 1
        elif st["synced"].get(rel) == sha(dst) or rel.startswith("results/HOST_YIELD/"):   # the laptop file is still the untouched previous host copy (or a host-side append log) -> the host version is newer
            open(dst, "wb").write(data)
            upd += 1
        else:                                                           # both sides changed (or the laptop made it): keep the laptop file, keep the host one aside
            cp = os.path.join(cdir, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            open(cp, "wb").write(data)
            conf += 1
        st["synced"][rel] = f["sha"]
    z.close()
    os.remove(os.path.join(OUT, stem + ".zip"))
    st["host_epoch"] = man["host_now"]
    return {"files": len(man["files"]), "new": new, "same": same, "updated": upd, "conflicts": conf}


# ------------------------------------------------------------------ big data -> HF
def repo_gb(repo):
    from huggingface_hub import HfApi
    tok = json.load(open(os.path.join(os.path.expanduser("~"), "hf_tokens.json")))["accounts"][0]["token"]
    info = HfApi(token=tok).dataset_info(repo, files_metadata=True)
    return sum((s.size or 0) for s in info.siblings) / 1e9


def backup(tag, groups, wsl, repo, max_gb):
    """plan new files of `groups` on the host, upload them via presigned urls, commit + verify; returns planned GB (0 when nothing is new)"""
    py("KNOWN", KNOWN_P, repo)
    inputs = [KNOWN_P, "scratchpad/_hfx.py", "scratchpad/_hfx_wsl.py"]
    runner = "scratchpad/_hfx_wsl.py" if wsl else "scratchpad/_hfx.py"
    jid, rc = job("hfx-plan-" + tag, inputs, ["python", "-u", runner, "PLAN", tag, "--settle", "900", "--known", KNOWN_P] + groups, mem=2, cpus=2, timeout_s=7200)
    assert rc == 0, "plan job rc %s" % rc
    fetch(jid, "work/HOST_HOUSEKEEPING/hfx_plan_%s.json" % tag)
    plan = json.load(open(os.path.join(OUT, "hfx_plan_%s.json" % tag)))
    gb = plan["bytes"] / 1e9
    if not plan["objects"]:
        os.remove(os.path.join(OUT, "hfx_plan_%s.json" % tag))
        return 0.0
    used = repo_gb(repo)
    if used + gb > max_gb:
        raise RuntimeError("QUOTA GUARD: repo %.1f GB + planned %.1f GB > %.0f GB; a second HF account (or a ruling on what to drop) is needed" % (used, gb, max_gb))
    py("BATCH", tag, repo)
    jid, rc = job("hfx-upload-" + tag, ["data/_cache/hfx_urls_%s.json" % tag, "scratchpad/_hfx.py", "scratchpad/_hfx_wsl.py"], ["python", "-u", runner, "UPLOAD", tag, "8"], mem=2, cpus=2, timeout_s=28800)
    assert rc == 0, "upload job rc %s" % rc
    fetch(jid, "work/HOST_HOUSEKEEPING/hfx_state_%s.json" % tag)
    py("COMMIT", tag, repo)
    return gb


def step_big(st, repo, max_gb, with_csr):
    stamp = time.strftime("%y%m%d%H%M")
    out = {}
    out["ws_gb"] = backup("s" + stamp, WS_GROUPS, False, repo, max_gb)
    out["wsl_gb"] = backup("w" + stamp, WSL_GROUPS + (["wsl_fb"] if with_csr else []), True, repo, max_gb)
    st["backed_up_gb"] = round(st.get("backed_up_gb", 0.0) + out["ws_gb"] + out["wsl_gb"], 3)
    return out


def cycle(a):
    st = load_state()
    rec = {"t": time.strftime("%F %T"), "cycle": st["cycles"] + 1}
    if not a.no_records:
        try:
            rec["records"] = step_records(st)
        except Exception as e:
            rec["records_error"] = str(e)[:500]
        save_state(st)
    if not a.no_big:
        try:
            rec["big"] = step_big(st, a.repo, a.max_repo_gb, a.with_csr)
        except Exception as e:
            rec["big_error"] = str(e)[:500]
    try:                                                       # CODE snapshot on HF every cycle (keeps the 3 newest), then the periodic disk housekeeping (laptop + host)
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "_code_snapshot.py")], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
        rec["code_snapshot"] = (r.stdout.strip().splitlines() or [""])[-1][:160]
    except Exception as e:
        rec["code_snapshot_error"] = str(e)[:300]
    try:
        import _housekeep
        hk = _housekeep.step(apply=True)
        rec["housekeep"] = {"host_free_gb": hk.get("host_free_gb_after"), "laptop_free_gb": hk.get("laptop_free_gb_after"), "wsl": hk.get("wsl")}
    except Exception as e:
        rec["housekeep_error"] = str(e)[:300]
    st["cycles"] += 1
    save_state(st)
    with open(LOG_P, "a") as f:
        f.write(json.dumps(rec) + "\n")
    json.dump(dict(rec, backed_up_gb_total=st.get("backed_up_gb"), host_epoch=st["host_epoch"]), open(STATUS_P, "w"), indent=1)
    log("cycle", rec)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval-min", type=float, default=180)
    p.add_argument("--repo", default="Swastik9895/crag-host-backup")
    p.add_argument("--max-repo-gb", type=float, default=80)
    p.add_argument("--no-records", action="store_true")
    p.add_argument("--no-big", action="store_true")
    p.add_argument("--with-csr", action="store_true")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    while True:
        try:
            cycle(a)
        except Exception as e:
            log("cycle failed:", str(e)[:500])
        if a.once:
            break
        time.sleep(a.interval_min * 60)


if __name__ == "__main__":
    main()
