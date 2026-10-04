"""Cascade step 5, acquisition: the processed FACC1 mention maps.

    python .../facc1_fetch.py

WHAT AND FROM WHERE
  mentions.zip, 1,411,580,332 bytes, modified 2021-03-07, from the OneDrive share that GrailQA,
  ChatKBQA, GMT-KBQA, RouterKGQA and Interactive-KBQA all point to as the single host:
      https://1drv.ms/u/s!AuJiG47gLqTznjl7VbnOESK6qPW2?e=HDy2Ye
  It contains entity_list_file_freebase_complete_all_mention and
  surface_map_file_freebase_complete_all_mention -- FACC1 mentions plus Freebase aliases, already
  aggregated. This is the reason the raw ~TB ClueWeb/FACC1 corpora are not being acquired.

  There is no mirror. GitHub code search returns 74 repositories referencing these filenames and
  every one resolves to this same share; GrailQA's own entity_linker/data holds only a README
  pointing here; HuggingFace has nothing. So the acquisition is written to be resumable rather than
  assumed to succeed in one attempt.

WHY THE COOKIE DANCE
  The anonymous share API (api.onedrive.com/v1.0/shares) now answers 401, and download.aspx on its
  own redirects to a login page. Redeeming the 1drv.ms link first in the SAME http client sets a
  FedAuth cookie, after which download.aspx serves the bytes. That is exactly what a browser does;
  no credentials are involved and nothing is signed in to.

Writes only under _acquisition/facc1/. Touches neither the frozen graph nor the frozen overlay.
"""
import sys, io, os, time, json, hashlib, urllib.request, http.cookiejar

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

V3 = "data/final_canonical/freebase_v3"
OUT = f"{V3}/_acquisition/facc1"
DEST = f"{OUT}/mentions.zip"
SHARE = "https://1drv.ms/u/s!AuJiG47gLqTznjl7VbnOESK6qPW2?e=HDy2Ye"
DL = ("https://onedrive.live.com/personal/f3a42ee08e1b62e2/_layouts/15/download.aspx"
      "?SourceUrl=%2Fpersonal%2Ff3a42ee08e1b62e2%2FDocuments%2Fmentions%2Ezip")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
EXPECT = 1_411_580_332
BLK = 1 << 20
t0 = time.time()
os.makedirs(OUT, exist_ok=True)


def opener():
    """A fresh session: redeem the share, keep the FedAuth cookie, then fetch."""
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", UA), ("Accept", "*/*")]
    op.open(SHARE, timeout=120).read(1)
    if not any(c.name == "FedAuth" for c in cj):
        raise RuntimeError("redeem did not yield a FedAuth cookie")
    return op


done = os.path.getsize(DEST) if os.path.exists(DEST) else 0
if done and done >= EXPECT:
    print(f"already complete: {done:,} bytes")
else:
    for attempt in range(12):
        try:
            op = opener()
            req = urllib.request.Request(DL, headers={"Range": f"bytes={done}-"})
            with op.open(req, timeout=300) as r:
                total = done + int(r.headers.get("Content-Length") or 0)
                print(f"attempt {attempt+1}: HTTP {r.status}, resuming at {done:,} of {total:,}",
                      flush=True)
                last = time.time()
                with open(DEST, "ab" if done else "wb") as f:
                    while True:
                        b = r.read(BLK)
                        if not b:
                            break
                        f.write(b)
                        done += len(b)
                        if time.time() - last > 20:
                            last = time.time()
                            el = time.time() - t0
                            print(f"  {done:,}/{EXPECT:,} ({100*done/EXPECT:5.1f}%) "
                                  f"{done/el/1e6:.1f} MB/s ({el:.0f}s)", flush=True)
            if done >= EXPECT:
                break
        except Exception as e:
            print(f"  attempt {attempt+1} interrupted at {done:,}: {type(e).__name__} {e}",
                  flush=True)
            time.sleep(min(60, 2 ** attempt))
            done = os.path.getsize(DEST) if os.path.exists(DEST) else 0

size = os.path.getsize(DEST) if os.path.exists(DEST) else 0
h = hashlib.sha256()
with open(DEST, "rb") as f:
    for blk in iter(lambda: f.read(1 << 20), b""):
        h.update(blk)
ok = size == EXPECT
rec = {"schema": "FACC1_ACQUISITION/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "writes only under _acquisition/facc1/; frozen artifacts untouched.",
       "artifact": "mentions.zip",
       "bytes": size, "expected_bytes": EXPECT, "SIZE_OK": ok,
       "sha256": h.hexdigest(),
       "source": {"share": SHARE, "download": DL,
                  "modified_shown_by_host": "2021-03-07",
                  "MIRRORS": ("none found. GitHub code search returns 74 repositories referencing "
                              "these filenames and all resolve to this share; GrailQA's own "
                              "entity_linker/data holds only a README pointing here; HuggingFace "
                              "has no copy. Recorded because a single-host dependency is a "
                              "reproducibility risk worth stating."),
                  "cited_by": ["dki-lab/GrailQA", "LHRLAB/ChatKBQA", "HXX97/GMT-KBQA",
                               "Oldcircle/RouterKGQA", "JimXiongGM/Interactive-KBQA"]},
       "CONTAINS_EXPECTED": ["entity_list_file_freebase_complete_all_mention",
                             "surface_map_file_freebase_complete_all_mention"],
       "NOT_YET_VERIFIED": ("this record covers acquisition only. What the archive actually holds "
                            "is listed and counted in a separate step; nothing about its contents "
                            "is asserted here."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FACC1_ACQUISITION.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\n{DEST}\n  {size:,} bytes  SIZE_OK={ok}\n  sha256 {h.hexdigest()}")
print(f"{time.time()-t0:.0f}s")
