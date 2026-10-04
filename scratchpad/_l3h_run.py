"""Host runner for the L3 development lane (stage L3_HOST): runs an L3 harness UNCHANGED, with two differences only.

1. Retrieval-cache rows come from the population bundle. CanonicalDataset.cache(model) of a bundled dataset is served from
   results/L3_HOST/bundle_<ds>__L1DEV.npz (_l3h_bundle.py: the population rows of the frozen top-1000 caches, exactly as stored,
   sha-checked here). Indexing it with any row outside the bundle, or in any way other than an integer row array, raises; an audit
   hook raises on any open of data/final_canonical/<ds>/retrieval_cache/ (the full caches are never read on this path).
2. The process runs below normal CPU priority (Windows BELOW_NORMAL_PRIORITY_CLASS, POSIX nice 10), so on the shared host
   mpr's (and anyone's) normal-priority work always gets the CPU first ("mpr has the priority wherever possible").

Nothing else is patched: the harness computes FLAT, IR_L1, routing and L3 exactly as on the laptop, and its own records (code shas,
identities, write-once outputs) are unchanged. A sidecar results/L3_HOST/runs/<harness>__<args>.run.json records the runner, the
bundle and the priority used.

usage: python scratchpad/_l3h_run.py <harness.py> <harness args...>
"""
import datetime
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import _l1d_lib as D  # noqa: E402

AD = D.AD
BUNDLES = {"metaqa": "results/L3_HOST/bundle_metaqa__L1DEV.npz"}
RUNS = os.path.join(D.REPO, "results", "L3_HOST", "runs")


class PopRows(object):
    """one per-query member of a cache, restricted to the bundled rows"""

    def __init__(self, rows, arr, what):
        self.arr = arr
        self.pos = {int(r): i for i, r in enumerate(rows)}
        self.what = what
        self.dtype = arr.dtype

    def __getitem__(self, key):
        if isinstance(key, np.ndarray) and key.ndim == 1 and key.dtype.kind in "iu":
            try:
                idx = np.fromiter((self.pos[int(r)] for r in key), np.int64, len(key))
            except KeyError as e:
                raise KeyError("population bundle %s: row %s is not in the bundle" % (self.what, e))
            return self.arr[idx]
        raise TypeError("population bundle %s: only an integer row array may index it (got %r)" % (self.what, type(key)))


class BundleView(object):
    def __init__(self, z, rows, model, what):
        pre = model + "__"
        self.files = [k[len(pre):] for k in z.files if k.startswith(pre)]
        self._m = {}
        for k in self.files:
            a = z[pre + k]
            self._m[k] = PopRows(rows, a, "%s %s %s" % (what, model, k)) if a.ndim >= 1 and a.shape[0] == len(rows) else a

    def __contains__(self, k):
        return k in self._m

    def __getitem__(self, k):
        return self._m[k]


STATE = {"bundles": {}, "served": {}}


def install():
    orig = AD.CanonicalDataset.cache
    loaded = {}

    def cache(self, model):
        if self.name not in BUNDLES:
            return orig(self, model)
        if self.name not in loaded:
            fz = os.path.join(D.REPO, BUNDLES[self.name])
            fj = fz[:-4] + ".json"
            rec = json.load(open(fj, encoding="utf-8"))
            sha = D.sha_file(fz)
            assert sha == rec["npz"]["sha256"], "the bundle of %s differs from its record" % self.name
            for mdl, s in rec["sources"].items():
                man = self.manifest.get("retrieval_cache", {}).get(mdl, {}).get("sha256")
                assert man is None or man == s["sha256"], "the bundle's %s source is not this dataset's frozen cache" % mdl
            z = np.load(fz)
            rows = z["rows"]
            loaded[self.name] = (z, rows)
            STATE["bundles"][self.name] = {"npz": BUNDLES[self.name], "sha256": sha, "record_sha256": D.sha_file(fj), "n_rows": int(len(rows)),
                                           "sources": {m: s["sha256"] for m, s in rec["sources"].items()}}
        z, rows = loaded[self.name]
        key = (self.name, model)
        if key not in STATE["served"]:
            STATE["served"][key] = BundleView(z, rows, model, self.name)
        return STATE["served"][key]

    AD.CanonicalDataset.cache = cache
    forbidden = [os.path.normcase(os.path.abspath(os.path.join(D.REPO, "data", "final_canonical", ds, "retrieval_cache"))) for ds in BUNDLES]

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, os.PathLike)):
            p = os.path.normcase(os.path.abspath(os.fspath(args[0])))
            for f in forbidden:
                if p.startswith(f):
                    raise PermissionError("L3_HOST: the full retrieval cache must not be opened on this path: %s" % p)

    sys.addaudithook(hook)


def lower_priority():
    try:
        import psutil
        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return "BELOW_NORMAL_PRIORITY_CLASS (%s)" % p.nice()
        p.nice(10)
        return "nice %s" % p.nice()
    except Exception as e:  # the run proceeds at the inherited priority; the sidecar says so
        return "unchanged (%s: %s)" % (type(e).__name__, e)


def main():
    assert len(sys.argv) >= 2, __doc__
    harness = os.path.abspath(sys.argv[1])
    args = sys.argv[2:]
    prio = lower_priority()
    install()
    t0 = time.time()
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    D.log("L3_HOST runner: priority %s; bundles %s; harness %s %s" % (prio, json.dumps(BUNDLES), D.rel(harness), " ".join(args)))
    sys.argv = [harness] + args
    rc = 0
    err = None
    try:
        runpy.run_path(harness, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException as e:
        rc, err = 1, "%s: %s" % (type(e).__name__, e)
        raise
    finally:
        os.makedirs(RUNS, exist_ok=True)
        stem = "__".join([os.path.splitext(os.path.basename(harness))[0]] + [a.replace("=", "-") for a in args if not a.startswith("--out=")])
        side = os.path.join(RUNS, stem + ".run.json")
        k = 1
        while os.path.exists(side):
            k += 1
            side = os.path.join(RUNS, "%s.%d.run.json" % (stem, k))
        rec = {"stage": "L3_HOST", "runner": {"path": "scratchpad/_l3h_run.py", "sha256": D.sha_file(os.path.abspath(__file__))},
               "harness": {"path": D.rel(harness), "sha256": D.sha_file(harness)}, "args": args, "priority": prio,
               "bundles_served": STATE["bundles"], "cache_views_served": sorted("%s/%s" % k_ for k_ in STATE["served"]),
               "started": started, "seconds": round(time.time() - t0, 1), "rc": rc, "error": err,
               "platform": D.platform_record(), "host": D.host_state()}
        with open(side, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        D.log("L3_HOST runner: rc %s -> %s" % (rc, D.rel(side)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
