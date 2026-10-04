"""Per-account Modal spend THIS MONTH via `modal billing report --for 'this month' --json`.
Fresh subprocess per account with that account's token env (process-isolated, no client caching).
Prints index, name, this-month spend, and a grant inference from spend-limit state."""
import os, sys, json, subprocess, re
sys.path.insert(0, os.path.abspath("."))
from src.experiments import credentials

def spend_for(cred):
    env = {**os.environ, **cred.env, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run([sys.executable, "-m", "modal", "billing", "report", "--for", "this month", "--json"],
                           capture_output=True, text=True, env=env, encoding="utf-8", errors="replace", timeout=90)
    except Exception as e:
        return None, f"EXC {e}"
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip().replace("\n", " ")
        return None, f"ERR rc={r.returncode} {msg[:80]}"
    out = r.stdout.strip()
    # find the JSON payload (skip any banner lines)
    m = re.search(r"[\{\[].*[\}\]]", out, re.S)
    if not m:
        return None, f"NOJSON {out[:80]}"
    try:
        data = json.loads(m.group(0))
    except Exception as e:
        return None, f"BADJSON {e}"
    # sum any 'cost'/'amount'/'total' numeric fields we can find
    total = 0.0; found = False
    def as_num(v):
        try: return float(v)
        except (TypeError, ValueError): return None
    def walk(x):
        nonlocal total, found
        if isinstance(x, dict):
            for k, v in x.items():
                if any(t in k.lower() for t in ("cost", "amount", "charge", "usd")) and as_num(v) is not None:
                    total += as_num(v); found = True
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(data)
    return (total if found else 0.0), ("ok" if found else "zero/nofield")

def main():
    pool = credentials.load_pool("modal")
    print(f"{'idx':>3} {'name':24s} {'spend_this_month':>16s}  note")
    for i, cred in enumerate(pool):
        val, note = spend_for(cred)
        s = f"${val:6.2f}" if val is not None else "   n/a"
        print(f"{i:3d} {cred.name:24s} {s:>16s}  {note}", flush=True)

if __name__ == "__main__":
    main()
