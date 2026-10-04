"""FBX_SCALE addendum 17 -- the Freebase drivers of stages E3 / E4 (the ladder and the surrogate), after E1 (_ooc_e1.py FB_BUILD / FB_E1) has written the symmetric adjacency and the level-1 twin map.

  python -u scratchpad/_ooc_fb.py LADDER <csr_dir> <l1_dir> [threads]   # level-1 explicit quotient, then the SPAN passes L2..L4 (each: agg -> compose -> explicit quotient); census per level
                                                                       # (files in <l1_dir>: cl<l>.i32 cumulative maps, q<l>.{eptr.i64,eidx.i32,ew.i64,vw.i32}; resumable per step)
No gold label is read anywhere in this file.
"""
import base64
import io
import json
import os
import sys
import time

import _ooc_lib as L

REPO = L.REPO
STATS = os.path.join(REPO, "results", "FREEBASE_SCALE", "FBX_GRAPH_STATS__v1.json")
LMAX, WMAX = 200, 512
LOG = L.log


def sh(args):
    return L._wsl(args)


def exists(p):
    try:
        sh(["test", "-e", p])
        return True
    except RuntimeError:
        return False


def rm(*paths):
    sh(["rm", "-f"] + list(paths))


def put_json(path, obj):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1))


def step(d, name, fn):
    """run fn() once; its JSON result is stored in <d>/<name>.done.json (resume = skip)"""
    mark = "%s/%s.done.json" % (d, name)
    if exists(mark):
        return json.loads(sh(["cat", mark]))
    r = fn()
    tmp = "%s/%s.tmp" % (d, name)
    b64 = base64.b64encode(json.dumps(r).encode()).decode()
    sh(["bash", "-c", "echo %s | base64 -d > %s && mv %s %s" % (b64, tmp, tmp, mark)])
    return r


def ladder(csr, d, threads, N=None, V1=None):
    N = N if N is not None else int(json.load(io.open(STATS, encoding="utf-8"))["stats"]["nodes"])
    fam = csr + "/xa.i64," + csr + "/aa.i32"
    cl = {1: d + "/cl1.i32"}
    assert exists(cl[1]), "run FB_E1 first (the level-1 twin map)"
    if V1 is None:
        V1 = int(json.load(io.open(os.path.join(REPO, "results", "FREEBASE_SCALE", "E1_CENSUS__fbx.json"), encoding="utf-8"))["twins"]["clusters"])     # the level-1 vertex count (census record)
    rec = {"stage": "FBX_SCALE addendum 17 / E3 ladder on Freebase (STRUCT-only, unsplit nets, exactly as the WebQSP ladder was built)", "N": N, "Lmax": LMAX, "Wmax": WMAX, "levels": {}}
    t0 = time.time()
    q = step(d, "q1", lambda: L.run("quot", {"src": "csr", "N": N, "fam": fam, "map": cl[1], "vout": V1, "out": d + "/q1", "threads": threads}))
    rec["levels"][1] = q
    LOG("level 1: V %d, M %d, P %d" % (q["V"], q["M"], q["P"]))
    Vp = q["V"]
    for l in (2, 3, 4):
        pre = "%s/q%d" % (d, l - 1)
        ag = step(d, "agg%d" % l, lambda: L.run("agg", {"eptr": pre + ".eptr.i64", "eidx": pre + ".eidx.i32", "ew": pre + ".ew.i64", "vw": pre + ".vw.i32", "vin": Vp, "lmax": LMAX, "wmax": WMAX, "mode": 1,
                                                       "out": "%s/inc%d.i32" % (d, l)}))
        LOG("level %d: SPAN pass %d -> %d clusters (%.0fs)" % (l, Vp, ag["clusters"], ag["seconds"]))
        cm = step(d, "cl%d" % l, lambda: L.run("compose", {"a": cl[l - 1], "b": "%s/inc%d.i32" % (d, l), "out": "%s/cl%d.i32" % (d, l)}))
        cl[l] = "%s/cl%d.i32" % (d, l)
        qq = step(d, "q%d" % l, lambda: L.run("quot", {"src": "expl", "eptr": pre + ".eptr.i64", "eidx": pre + ".eidx.i32", "ew": pre + ".ew.i64", "vin": Vp, "map": "%s/inc%d.i32" % (d, l), "vout": ag["clusters"],
                                                       "vwin": pre + ".vw.i32", "out": "%s/q%d" % (d, l), "threads": threads}))
        assert qq["V"] == ag["clusters"]
        LOG("level %d: V %d, M %d, P %d (cumulative map max id %d)" % (l, qq["V"], qq["M"], qq["P"], cm["max"]))
        rec["levels"][l] = dict(qq, agg=ag, compose=cm)
        if l - 1 >= 1 and l - 1 != 4:                              # the previous explicit level is consumed; its maps stay
            rm(pre + ".eptr.i64", pre + ".eidx.i32", pre + ".ew.i64", pre + ".vw.i32")
        Vp = qq["V"]
    rec["seconds"] = round(time.time() - t0, 1)
    if N == int(json.load(io.open(STATS, encoding="utf-8"))["stats"]["nodes"]):
        put_json(os.path.join(REPO, "results", "FREEBASE_SCALE", "E3_LADDER__fbx.json"), rec)
    return rec


def main():
    a = sys.argv[1:]
    if a[:1] == ["LADDER"] and len(a) >= 3:
        print(json.dumps(ladder(a[1], a[2], int(a[3]) if len(a) > 3 else 8)))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
