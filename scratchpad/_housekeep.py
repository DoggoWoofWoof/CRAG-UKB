"""Periodic disk housekeeping for the laptop AND the lab host (user order 2026-10-04: remove what is not required, periodically, from both; never lose the ability to restart where we stopped).
Called at the end of every _host_sync.py cycle (3 h) and runnable by hand:
  python -u scratchpad/_housekeep.py [--apply] [--no-wsl]

What it does (dry run unless --apply):
  host    scratchpad/_host_rm.py PHASE1 (scratch) + PHASE3/PHASE4 (completed experiment outputs, ner, calibration bundle): every PHASE3/4 file must be in the verified HF restore index
          (data/_cache/hfx_known.json), otherwise the directory is BLOCKED.  Directories listed in data/_cache/HOUSEKEEP_PINS.json (a job needs them again, e.g. after a restore) are skipped.
  WSL     the host's WSL ext4.vhdx never shrinks by itself: when it exceeds the used ext4 bytes by > 20 GB, WSL is idle (`wsl --list --running` empty) and no other WSL user is active, run
          fstrim + wsl --shutdown + diskpart compact vdisk (non-destructive; measured 2026-10-04: 67.9 -> 31.9 GB).
  laptop  scratch only: data/ooc_tmp, data/vc_tmp, leftover %TEMP%/crag_code_*.tar.gz older than a day.
  always  work/HOST_HOUSEKEEPING/DISK_STATUS.json (laptop + host free GB, vhdx vs used, actions) -- read it first when a disk question comes up.
NEVER deleted here: the Freebase tree, the Hotpot/2Wiki substrates, models, the encode chunks/names/pq (a running or planned job reads them), anything not in the HF restore index, other users' files.
Not done by design (system settings / other people's data): hiberfil.sys / pagefile.sys, other accounts' folders and recycle bins."""
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "work", "HOST_HOUSEKEEPING")
RX = [sys.executable, "C:/Users/Swastik/Desktop/message-passing-retrieval/tools/rx/rx.py"]
ENV = dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONIOENCODING="utf-8")
PINS_P = os.path.join(ROOT, "data", "_cache", "HOUSEKEEP_PINS.json")
VHDX = r"C:\Users\Student2\rx\wsl\ubuntu-24.04\ext4.vhdx"
DISTRO = "Ubuntu-24.04"
SLACK_GB = 20.0
HOST_PHASES = ["PHASE1", "PHASE3", "PHASE4"]


def log(*a):
    print(time.strftime("%F %T"), "[housekeep]", *a, flush=True)


def rx(*args, timeout=900):
    r = subprocess.run(RX + list(args), cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return r.returncode, r.stdout.replace("\x00", "")


def ps(cmd, timeout=900):
    return rx("exec", "--shell", "powershell", "--no-ws", "--timeout", str(timeout), "--", cmd, timeout=timeout + 60)


def host_free_gb():
    rc, out = ps("'{0:N1}' -f ((Get-PSDrive C).Free/1GB)", 120)
    m = re.findall(r"^\s*([\d.]+)\s*$", out, re.M)
    return float(m[-1]) if rc == 0 and m else None


def pins():
    try:
        return json.load(open(PINS_P))
    except Exception:  # noqa: BLE001
        return []


def host_clean(apply):
    import _host_sync as hs
    cmd = ["python", "-u", "scratchpad/_host_rm.py"] + HOST_PHASES + ["--skip=%s" % p for p in pins()] + (["--apply"] if apply else [])
    jid, rc = hs.job("hostrm-" + ("apply" if apply else "dry"), [hs.KNOWN_P, "scratchpad/_host_rm.py"], cmd, mem=1, cpus=1, timeout_s=1800)
    _, out = rx("logs", jid)
    lines = [l for l in out.splitlines() if re.match(r"^(DELETE|would|BLOCKED|absent|pinned|PHASE)", l)]
    return {"job": jid, "rc": rc, "lines": [l[:200] for l in lines]}


def wsl_state():
    """(vhdx GB, ext4 used GB or None, idle?) -- reading df starts the distro for a second, only done when no distribution is running"""
    rc, out = ps("$f='%s'; 'VHDX ' + [math]::Round((Get-Item $f).Length/1GB,2); 'RUN ' + ((wsl.exe --list --running | Out-String).Replace([char]0,'').Trim() -replace '\\s+',' ')" % VHDX, 120)
    vh = re.search(r"VHDX ([\d.]+)", out)
    run = re.search(r"RUN (.*)", out)
    idle = bool(run and "no running distributions" in run.group(1).lower())
    used = None
    if idle:
        rc2, o2 = ps("wsl.exe -d %s --exec bash -c \"df -B1 / | tail -1\"" % DISTRO, 120)
        m = re.search(r"\s(\d{9,})\s+(\d{9,})\s+\d{9,}\s+\d+%", o2)
        used = int(m.group(2)) / 1e9 if m else None
    return (float(vh.group(1)) if vh else None), used, idle


def wsl_compact():
    body = ("$f='%s'; wsl.exe -d %s -u root --exec fstrim -v / | Out-String; wsl.exe --shutdown; Start-Sleep 10; "
            "$s = \"select vdisk file=`\"$f`\"`r`nattach vdisk readonly`r`ncompact vdisk`r`ndetach vdisk`r`nexit`r`n\"; $p = Join-Path $env:TEMP 'crag_compact.txt'; Set-Content -Path $p -Value $s -Encoding ASCII; "
            "diskpart /s $p | Select-String 'successfully|error' | Out-String; Remove-Item $p; 'after ' + [math]::Round((Get-Item $f).Length/1GB,2)" % (VHDX, DISTRO))
    rc, out = ps(body, 3000)
    m = re.search(r"after ([\d.]+)", out)
    return {"rc": rc, "after_gb": float(m.group(1)) if m else None, "tail": out.strip()[-300:]}


def laptop_clean(apply):
    acts = []
    for d in ("data/ooc_tmp", "data/vc_tmp"):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            n = sum(len(f) for _, _, f in os.walk(p))
            acts.append({"path": d, "files": n})
            if apply:
                shutil.rmtree(p, ignore_errors=True)
    tmp = os.environ.get("TEMP") or os.environ.get("TMP") or ""
    if tmp and os.path.isdir(tmp):
        for f in os.listdir(tmp):
            if re.match(r"^crag_code_\d+_\d+\.tar\.gz$", f):
                fp = os.path.join(tmp, f)
                if time.time() - os.path.getmtime(fp) > 86400:
                    acts.append({"path": fp, "bytes": os.path.getsize(fp)})
                    if apply:
                        os.remove(fp)
    return acts


def step(apply=True, wsl=True):
    st = {"t": time.strftime("%F %T"), "apply": apply}
    st["laptop_free_gb"] = round(shutil.disk_usage(ROOT).free / 1e9, 1)
    st["host_free_gb_before"] = host_free_gb()
    try:
        st["laptop"] = laptop_clean(apply)
    except Exception as e:  # noqa: BLE001
        st["laptop_error"] = str(e)[:300]
    try:
        st["host"] = host_clean(apply)
    except Exception as e:  # noqa: BLE001
        st["host_error"] = str(e)[:300]
    if wsl:
        try:
            vh, used, idle = wsl_state()
            st["wsl"] = {"vhdx_gb": vh, "ext4_used_gb": None if used is None else round(used, 1), "idle": idle}
            if apply and idle and vh and used is not None and vh - used > SLACK_GB:
                st["wsl"]["compact"] = wsl_compact()
        except Exception as e:  # noqa: BLE001
            st["wsl_error"] = str(e)[:300]
    st["host_free_gb_after"] = host_free_gb()
    st["laptop_free_gb_after"] = round(shutil.disk_usage(ROOT).free / 1e9, 1)
    os.makedirs(OUT, exist_ok=True)
    json.dump(st, open(os.path.join(OUT, "DISK_STATUS.json"), "w"), indent=1)
    log("host free %s -> %s GB; laptop free %s -> %s GB; wsl %s" % (st["host_free_gb_before"], st["host_free_gb_after"], st["laptop_free_gb"], st["laptop_free_gb_after"], st.get("wsl")))
    return st


if __name__ == "__main__":
    step(apply="--apply" in sys.argv, wsl="--no-wsl" not in sys.argv)
