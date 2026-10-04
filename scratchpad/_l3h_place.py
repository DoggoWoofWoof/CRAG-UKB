"""Placement comparison for the host L3 lane (stage L3_HOST): is a run of an L3 harness on one machine bit-identical to its run on another?

Two write-once outputs <stem>.{json,npz} are compared:
- npz: the same member names, and every array equal in dtype, shape and bytes (BIT_IDENTICAL);
- json: every value equal after dropping the volatile keys (VOLATILE, and any key starting with "seconds" or "latency": timing,
  memory, host and platform descriptions, the run tag and the npz block); any other difference fails.

usage: python scratchpad/_l3h_place.py <a_stem> <b_stem> [--record=<out.json> --label=<text>]
(a stem is the output path without .json/.npz)
"""
import datetime
import json
import os
import sys

import numpy as np

import _l1d_lib as D

VOLATILE = {"seconds", "seconds_loop", "process_peak_rss_mb", "host_at_start", "platform", "latency_ms_flat", "latency_ms_per_row", "tag", "npz"}


def walk(a, b, path, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE or k.startswith("seconds") or k.startswith("latency"):
                continue
            if k not in a or k not in b:
                diffs.append("%s/%s: present in one side only" % (path, k))
                continue
            walk(a[k], b[k], path + "/" + k, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append("%s: list lengths %d != %d" % (path, len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, "%s[%d]" % (path, i), diffs)
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)):
        diffs.append("%s: %r != %r" % (path, a if len(repr(a)) < 120 else repr(a)[:120], b if len(repr(b)) < 120 else repr(b)[:120]))


def compare(sa, sb):
    za, zb = np.load(sa + ".npz"), np.load(sb + ".npz")
    ka, kb = sorted(za.files), sorted(zb.files)
    arr_diffs = []
    if ka != kb:
        arr_diffs.append("members differ: only a %s, only b %s" % (sorted(set(ka) - set(kb)), sorted(set(kb) - set(ka))))
    n_bytes = 0
    for k in sorted(set(ka) & set(kb)):
        x, y = za[k], zb[k]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            arr_diffs.append("%s: dtype %s/%s shape %s/%s" % (k, x.dtype, y.dtype, x.shape, y.shape))
        n_bytes += x.nbytes
    ja = json.load(open(sa + ".json", encoding="utf-8"))
    jb = json.load(open(sb + ".json", encoding="utf-8"))
    js_diffs = []
    walk(ja, jb, "", js_diffs)
    verdict = "BIT_IDENTICAL" if not arr_diffs and not js_diffs else "DIFFERENT"
    return {"verdict": verdict, "npz_members": len(set(ka) & set(kb)), "npz_array_bytes_compared": n_bytes, "npz_differences": arr_diffs,
            "json_differences": js_diffs[:200], "json_differences_total": len(js_diffs), "json_volatile_keys_ignored": sorted(VOLATILE),
            "a": {"stem": D.rel(sa) if os.path.abspath(sa).lower().startswith(D.REPO.lower()) else sa, "json_sha256": D.sha_file(sa + ".json"),
                  "npz_sha256": D.sha_file(sa + ".npz"), "platform": ja.get("platform"), "host": (ja.get("host_at_start") or {}).get("machine")},
            "b": {"stem": D.rel(sb) if os.path.abspath(sb).lower().startswith(D.REPO.lower()) else sb, "json_sha256": D.sha_file(sb + ".json"),
                  "npz_sha256": D.sha_file(sb + ".npz"), "platform": jb.get("platform"), "host": (jb.get("host_at_start") or {}).get("machine")},
            "npz_file_bytes_identical": D.sha_file(sa + ".npz") == D.sha_file(sb + ".npz")}


def main():
    a, b = sys.argv[1], sys.argv[2]
    out = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--record=")), None)
    label = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--label=")), None)
    r = compare(a, b)
    print(json.dumps({k: v for k, v in r.items() if k not in ("a", "b")}, indent=1)[:3000])
    if out:
        assert not os.path.exists(out), "write-once: %s exists" % out
        rec = {"stage": "L3_HOST", "test": "placement", "label": label, "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "code": {"path": "scratchpad/_l3h_place.py", "sha256": D.sha_file(os.path.abspath(__file__))}, **r}
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        print("record ->", out, D.sha_file(out)[:16])
    return 0 if r["verdict"] == "BIT_IDENTICAL" else 2


if __name__ == "__main__":
    sys.exit(main())
