"""Host side of the laptop relay for _host_sync.py.

The host normally PUTs its shards to the presigned HF urls itself.  When the lab network starts inspecting TLS with a root the host does not trust
(2026-10-06: Sophos SSL CA, every https host fails verification) those PUTs cannot succeed and must not be forced by turning verification off.
Instead the host packs the members of the not-yet-uploaded objects into a stored (uncompressed) zip, the laptop fetches it through rx (ssh, unaffected),
rebuilds the SAME deterministic shards (sha256 must equal the plan) and PUTs them from its own network.

  host   python -u scratchpad/_hfx_relay.py PACK <tag>      ->  work/HOST_HOUSEKEEPING/relay_<tag>.zip   (older relay_*.zip removed)
"""
import json
import os
import sys
import zipfile

OUT = os.path.join(os.getcwd(), "work", "HOST_HOUSEKEEPING")
CACHE = os.path.join(os.getcwd(), "data", "_cache")


def pack(tag):
    plan = json.load(open(os.path.join(OUT, "hfx_plan_%s.json" % tag)))
    urls = json.load(open(os.path.join(CACHE, "hfx_urls_%s.json" % tag)))["urls"]
    sp = os.path.join(OUT, "hfx_state_%s.json" % tag)
    done = set(json.load(open(sp))["done"]) if os.path.exists(sp) else set()
    rels = []
    for o in plan["objects"]:
        if o.get("root", "ws") != "ws":
            continue
        if urls[o["sha256"]].get("done") or o["sha256"] in done:
            continue
        rels += o["rels"]
    for f in os.listdir(OUT):
        if f.startswith("relay_") and f.endswith(".zip"):
            os.remove(os.path.join(OUT, f))
    dst = os.path.join(OUT, "relay_%s.zip" % tag)
    n = b = 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for r in rels:
            p = os.path.join(os.getcwd(), r.replace("/", os.sep))
            z.write(p, r)
            n += 1
            b += os.path.getsize(p)
    print("PACK %s: %d files, %.3f GB -> %s" % (tag, n, b / 1e9, dst), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "PACK":
        pack(sys.argv[2])
    else:
        raise SystemExit("usage: _hfx_relay.py PACK <tag>")
