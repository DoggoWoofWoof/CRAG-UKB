"""End-to-end check of the HotpotQA / 2Wiki substrates moved to the host through the Hugging Face relay (scratchpad/_txt_relay.py).

  python scratchpad/_txt_verify_host.py MANIFEST     laptop: size + sha256 of every staged file (the stage = hard links of the frozen files) -> data/_cache/txt_sha_manifest.json
  python scratchpad/_txt_verify_host.py VERIFY       host (run under _host_yield.py): re-hash every file under data/final_canonical/<ds>/<path> and compare with the manifest;
                                                     also compare the graph npz / retrieval-cache shas with the pins of CANONICAL_FREEZE.json (data/final_canonical/CANONICAL_FREEZE.json)
                                                     -> results/L1_X/TXT_HOST_VERIFY__v1.json  (write-once; fetch it back with rx fetch)
Read-only on the data; every file is read once.  Nothing is repaired: a mismatch is reported and the file is to be pulled again."""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STAGE = os.path.join(ROOT, "data", "_cache", "txt_stage")
MAN = os.path.join(ROOT, "data", "_cache", "txt_sha_manifest.json")
FREEZE = os.path.join(ROOT, "data", "final_canonical", "CANONICAL_FREEZE.json")
OUT = os.path.join(ROOT, "results", "L1_X", "TXT_HOST_VERIFY__v1.json")
BASE = os.path.join(ROOT, "data", "final_canonical")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(base):
    out = []
    for r, ds, fs in os.walk(base):
        if ".cache" in r.replace("\\", "/").split("/"):
            continue
        for f in fs:
            p = os.path.join(r, f)
            out.append(os.path.relpath(p, base).replace("\\", "/"))
    return sorted(out)


def hash_all(base, rels, workers=4):
    t0 = time.time()
    res = {}

    def one(rp):
        p = os.path.join(base, rp)
        return rp, os.path.getsize(p), sha(p)
    with ThreadPoolExecutor(workers) as ex:
        for i, (rp, n, h) in enumerate(ex.map(one, rels)):
            res[rp] = {"bytes": n, "sha256": h}
            if i % 50 == 0:
                print("[%6.0fs] %d / %d hashed" % (time.time() - t0, i + 1, len(rels)), flush=True)
    return res


def manifest():
    rels = rows(STAGE)
    res = hash_all(STAGE, rels, workers=2)
    with open(MAN, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"what": "size + sha256 of the files staged for the HF relay (laptop, frozen files)", "n": len(res), "bytes": sum(v["bytes"] for v in res.values()), "files": res}, f)
    print("manifest", len(res), "files,", sum(v["bytes"] for v in res.values()), "bytes ->", MAN)


def verify():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    man = json.load(open(MAN, encoding="utf-8"))["files"]
    missing = [rp for rp in man if not os.path.isfile(os.path.join(BASE, rp))]
    have = {rp: man[rp] for rp in man if rp not in missing}
    t0 = time.time()
    got = hash_all(BASE, sorted(have))
    bad = [rp for rp in got if got[rp] != man[rp]]
    fz = json.load(open(FREEZE, encoding="utf-8"))["DATASETS"]
    pins = []
    for ds in ("hotpotqa", "2wiki"):
        d = fz[ds]
        for fam, h in d["graph"]["npz_sha256"].items():
            rp = "%s/graph/%s.npz" % (ds, fam)
            pins.append({"file": rp, "pinned": h, "got": got.get(rp, {}).get("sha256"), "ok": got.get(rp, {}).get("sha256") == h})
        for kind in ("dense", "splade"):
            rp = "%s/retrieval_cache/%s_top1000.npz" % (ds, kind)
            h = d["retrieval_cache"][kind]["sha256"]
            pins.append({"file": rp, "pinned": h, "got": got.get(rp, {}).get("sha256"), "ok": got.get(rp, {}).get("sha256") == h})
        for ch in ("dense", "splade"):
            for part in ("docs", "queries"):
                rp = "%s/embeddings/%s/%s/manifest.json" % (ds, ch, part)
                h = d["embeddings"][ch][part]["manifest_sha256"]
                pins.append({"file": rp, "pinned": h, "got": got.get(rp, {}).get("sha256"), "ok": got.get(rp, {}).get("sha256") == h, "note": "manifest_sha256 may use a different hashing convention; informational if the manifest-vs-laptop check above passes"})
    ok = not missing and not bad
    rec = {"RECORD": "TXT_HOST_VERIFY", "version": 1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "where": "host (rx), files pulled from a private Hugging Face repo by signed links",
           "manifest": {"n_files": len(man), "bytes": sum(v["bytes"] for v in man.values())}, "files_missing": missing, "files_mismatching_the_laptop": bad,
           "freeze_pins": pins, "seconds": round(time.time() - t0, 1), "VERDICT": "PASS" if ok else "FAIL"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    os.replace(OUT + ".tmp", OUT)
    print("VERDICT", rec["VERDICT"], "| files", len(man), "| missing", len(missing), "| mismatching", len(bad), "| freeze pins ok %d / %d" % (sum(p["ok"] for p in pins), len(pins)), "|", OUT)


if __name__ == "__main__":
    {"MANIFEST": manifest, "VERIFY": verify}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
