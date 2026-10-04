"""Acquire the NSM route-A artifacts and record provenance BEFORE any extraction.

    python scratchpad/final_canonical_build/_acquisition/fetch_nsm.py

Authorised by the user on 2026-09-06 for EXACTLY these three files (~1.03 GB declared).
Pre-registration: data/final_canonical/webqsp/NSM_ACQUISITION_PREREGISTRATION.json

Payloads land under data/final_canonical/webqsp/_acquisition/nsm/ (gitignored, bulky).
The provenance record is small and lives beside the other webqsp reports so it reaches the
tracked mirror.  Nothing is extracted here: extraction is a separate step that must run
AFTER provenance is on disk, so the hashes describe the bytes we actually received.

Downloads go to <name>.part and are renamed only after the full read succeeds, so an
interrupted run can never leave a truncated file wearing the final name.
"""
import hashlib, json, os, time

import gdown

DST = "data/final_canonical/webqsp/_acquisition/nsm"
OUT = "data/final_canonical/webqsp/NSM_ACQUISITION_PROVENANCE.json"

# declared_* are the pre-download figures from WEBQSP_SOURCE_MATRIX.json.  For the two tarballs
# they are Google Drive interstitial strings with no published hash; only entities_names.json has
# a byte-exact figure (Content-Range probe).  Measured bytes are the authority -- a mismatch is
# reported, never silently accepted.
FILES = [
    {"name": "webqsp.tgz", "drive_id": "1KcIVAi4nf2uyflMOz5OSr54FOL2s2tAi",
     "declared_size": "136M", "declared_bytes": None},
    {"name": "CWQ.tgz", "drive_id": "1ua7h88kJ6dECih6uumLeOIV9a3QNdP-g",
     "declared_size": "872M", "declared_bytes": None},
    {"name": "entities_names.json", "drive_id": "1H7BqbVsQXr0bGuBsLUphb_Cb0Jf0rSoN",
     "declared_size": "23377816 B", "declared_bytes": 23377816},
]


def sha256_and_bytes(path):
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def main():
    os.makedirs(DST, exist_ok=True)
    recs = []
    for f in FILES:
        final = f"{DST}/{f['name']}"
        url = f"https://drive.google.com/uc?id={f['drive_id']}"
        t0 = time.time()
        if os.path.exists(final):
            print(f"[skip] {f['name']} already present", flush=True)
            downloaded = False
        else:
            part = final + ".part"
            print(f"[get ] {f['name']} <- {url}", flush=True)
            got = gdown.download(url, part, quiet=False)
            if not got or not os.path.exists(part):
                recs.append({**f, "source_url": url, "status": "DOWNLOAD_FAILED",
                             "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                print(f"[FAIL] {f['name']}", flush=True)
                continue
            os.replace(part, final)          # atomic: the final name only ever wears a complete file
            downloaded = True
        sha, n = sha256_and_bytes(final)
        rec = {
            "name": f["name"],
            "source_url": url,
            "drive_id": f["drive_id"],
            "drive_folder": "https://drive.google.com/drive/folders/1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv",
            "upstream_repo": "https://github.com/RichardHGL/WSDM2021_NSM",
            "upstream_commit_pinned_by_prior_audit": "2e20915956a69b6f832143ded9d54442ccbe698e",
            "local_path": final,
            "measured_bytes": n,
            "sha256": sha,
            "declared_size": f["declared_size"],
            "declared_bytes": f["declared_bytes"],
            "declared_vs_measured": (
                "N/A - no byte-exact figure was published pre-download"
                if f["declared_bytes"] is None else
                ("EXACT_MATCH" if f["declared_bytes"] == n else f"MISMATCH: declared {f['declared_bytes']}, measured {n}")),
            "hash_published_upstream": None,
            "hash_is_ours_not_theirs": "No checksum is published for any of these artifacts. This sha256 "
                                       "pins the bytes THIS project received on this date; it is not a "
                                       "verification against a publisher-stated value.",
            "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - t0, 1),
            "newly_downloaded": downloaded,
            "status": "OK",
        }
        recs.append(rec)
        print(f"[ok  ] {f['name']} {n} B sha256={sha[:16]}...", flush=True)

    doc = {
        "schema": "NSM_ACQUISITION_PROVENANCE/v1",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recorded_before_extraction": True,
        "authorised_by": "user, 2026-09-06, for exactly these three files",
        "preregistration": "data/final_canonical/webqsp/NSM_ACQUISITION_PREREGISTRATION.json",
        "purpose": "Route A falsifiable test: does the union of NSM's released per-question subgraphs "
                   "reproduce 8,309,195 triples / 7,058 relations, and is it MID-keyed at all?",
        "files": recs,
        "total_measured_bytes": sum(r.get("measured_bytes", 0) for r in recs),
        "all_ok": all(r["status"] == "OK" for r in recs),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({"total_measured_bytes": doc["total_measured_bytes"],
                      "all_ok": doc["all_ok"],
                      "files": [{k: r.get(k) for k in ("name", "measured_bytes", "sha256", "status")} for r in recs]},
                     indent=1))


if __name__ == "__main__":
    main()
