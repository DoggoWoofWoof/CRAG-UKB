"""Parallel per-account Modal spend + headroom estimate.

Modal exposes NO grant/balance endpoint -- only usage. Headroom is therefore an
ESTIMATE: assumed_grant - measured_spend_this_month. Grants are assumptions, not
measurements, and are labelled as such in the output.

Parallel because each account is an isolated subprocess with its own token env.
"""
import os, sys, json, subprocess, re
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath("."))
from src.experiments import credentials

# Grants are ASSUMPTIONS (Modal Starter default $30/mo). Known exceptions only.
GRANT = {"pes1ug23cs623": 5.0}
DEFAULT_GRANT = 30.0
# acct0 and acct5 are the SAME workspace/credit pool -- count the pair once.
SHARED = {"audit_kk": "kuttakamina9895"}


def spend_for(cred):
    env = {**os.environ, **cred.env, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run(
            [sys.executable, "-m", "modal", "billing", "report", "--for", "this month", "--json"],
            capture_output=True, text=True, env=env,
            encoding="utf-8", errors="replace", timeout=120)
    except Exception as e:
        return None, "EXC %s" % str(e)[:60]
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip().replace("\n", " ")
        low = msg.lower()
        if "token not found" in low or "unauthenticated" in low or "invalid" in low:
            return None, "AUTH_DEAD"
        if "spend limit" in low or "disabled" in low:
            return None, "CAPPED/DISABLED"
        return None, "ERR %s" % msg[:60]
    m = re.search(r"[\{\[].*[\}\]]", r.stdout.strip(), re.S)
    if not m:
        return None, "NOJSON"
    try:
        data = json.loads(m.group(0))
    except Exception as e:
        return None, "BADJSON %s" % str(e)[:40]

    total = 0.0
    found = False

    def as_num(v):
        try:
            return float(str(v).replace("$", "").replace(",", ""))
        except (TypeError, ValueError):
            return None

    def walk(x):
        nonlocal total, found
        if isinstance(x, dict):
            for k, v in x.items():
                n = as_num(v) if any(t in k.lower() for t in ("cost", "amount", "charge", "usd")) else None
                if n is not None:
                    total += n
                    found = True
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return (total if found else 0.0), ("ok" if found else "zero")


def main():
    pool = credentials.load_pool("modal")
    with ThreadPoolExecutor(max_workers=15) as ex:
        results = list(ex.map(spend_for, pool))

    print("%3s %-24s %>10s %>10s %>10s  %s" .replace(">", "") % (
        "idx", "name", "spend", "grant*", "headroom*", "state"))
    print("-" * 78)

    live = 0.0
    dead = []
    counted = set()
    for i, (cred, (val, note)) in enumerate(zip(pool, results)):
        name = cred.name
        grant = GRANT.get(name, DEFAULT_GRANT)
        if val is None:
            print("%3d %-24s %10s %10s %10s  %s" % (i, name, "n/a", "%.2f" % grant, "n/a", note))
            dead.append((i, name, note))
            continue
        head = max(0.0, grant - val)
        dup = SHARED.get(name)
        tag = note
        if dup and dup in counted:
            tag = note + " (SHARED pool with %s -- not added)" % dup
        else:
            live += head
            counted.add(name)
        print("%3d %-24s %10.2f %10.2f %10.2f  %s" % (i, name, val, grant, head, tag))

    print("-" * 78)
    print("ESTIMATED TOTAL HEADROOM (reachable accounts only): $%.2f" % live)
    if dead:
        print("UNREACHABLE (%d): %s" % (len(dead), ", ".join("%s[%d]=%s" % (n, i, t) for i, n, t in dead)))
        unknown = sum(GRANT.get(n, DEFAULT_GRANT) for _, n, _ in dead)
        print("  -> up to $%.2f more IF those tokens were re-issued (NOT counted above)" % unknown)
    print()
    print("* grant is an ASSUMPTION (Modal Starter $30/mo default; pes1ug23cs623 known $5).")
    print("  Modal exposes no balance endpoint. Spend is MEASURED; headroom is derived.")


if __name__ == "__main__":
    main()
