"""Cheap health probe: per Modal account, run a trivial CPU no-op and classify the outcome.
Spend-limit blocks ALL compute (CPU included), so this surfaces capped accounts for ~nothing.
Prints ACTIVE / SPEND_LIMIT / AUTH_FAILED / ERROR per account and writes scratchpad/modal_health.json."""
import os, json, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, os.path.abspath("."))
from src.experiments import credentials

def probe(cred):
    import importlib
    # fresh modal per account so the activated token env is picked up
    for m in [k for k in list(sys.modules) if k == "modal" or k.startswith("modal.")]:
        del sys.modules[m]
    import modal
    app = modal.App("crag-health")
    img = modal.Image.debian_slim()
    @app.function(image=img, timeout=120)
    def noop(): return "ok"
    with app.run():
        return noop.remote()

def classify(exc):
    s = str(exc).lower()
    if any(k in s for k in ("spend limit","exceeded its","resourceexhausted","quota","no capacity")): return "SPEND_LIMIT"
    if any(k in s for k in ("unauthor","auth","token","forbidden","401","403")): return "AUTH_FAILED"
    return "ERROR"

def main():
    pool = credentials.load_pool("modal")
    out = {}
    for i, cred in enumerate(pool):
        cred.activate()
        t0 = time.time()
        try:
            r = probe(cred); out[i] = {"name": cred.name, "status": "ACTIVE", "r": r}
            print(f"{i:2d} {cred.name:24s} ACTIVE  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            st = classify(e); out[i] = {"name": cred.name, "status": st, "err": str(e)[:120]}
            print(f"{i:2d} {cred.name:24s} {st}", flush=True)
    json.dump(out, open("scratchpad/modal_health.json","w"), indent=2)
    healthy = [i for i,v in out.items() if v["status"]=="ACTIVE"]
    print(f"\nHEALTHY: {healthy}  ({len(healthy)}/{len(pool)})")

if __name__ == "__main__":
    import logging; logging.basicConfig(level=logging.WARNING)
    main()
