"""FBQ v2 step A: approved downloads of Freebase-MID-keyed question sources (train+dev only) + provenance record.

  python scratchpad/_fbq_download_v2.py --phase fetch            # HEAD-verify, download the approved files into data/original/<name>/ (never overwrites)
  python scratchpad/_fbq_download_v2.py --phase list-grailqa     # list the GrailQA zip members (no extraction)
  python scratchpad/_fbq_download_v2.py --phase extract-grailqa  # extract ONLY the train/dev JSON members (rule below)
  python scratchpad/_fbq_download_v2.py --phase record           # write results/FREEBASE_SCALE/FBQ_DOWNLOAD_PROVENANCE__v1.json (write-once)

APPROVED (user ruling 2026-10-04) and nothing else:
  (1) FreebaseQA train + dev   https://raw.githubusercontent.com/kelvin-jiang/FreebaseQA/master/FreebaseQA-{train,dev}.json  (NOT eval, NOT partial)
  (2) GrailQA archive          https://dl.orangedox.com/WyaCpL/  (download endpoint ?dl=1); train + dev members only, test member never extracted/parsed
  (3) ComplexWebQuestions train + dev  (Dropbox links named in the HF loader script) ONLY IF an HTTP HEAD shows each size <= 200 MB; test URL is not in this code
Data is untrusted bytes: only json is ever parsed from it; nothing is executed; archive members are validated (no absolute path, no '..') before extraction.
Free disk is checked before each download (stop if < 3 GB would remain)."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG = os.path.join(ROOT, "data", "original")
OUTDIR = os.path.join(ROOT, "results", "FREEBASE_SCALE")
STATE = os.path.join(os.environ.get("FBQ_STATE_DIR", os.environ.get("TEMP", ".")), "fbq_dl_state_v2.json")
UA = "crag-fbq-provenance/1.0"
MAX_CWQ_BYTES = 200 * 1024 * 1024
MIN_FREE_AFTER = 3 * 1024 ** 3

FQA_BASE = "https://raw.githubusercontent.com/kelvin-jiang/FreebaseQA/master/"
GRAIL_URL = "https://dl.orangedox.com/WyaCpL/"
GRAIL_DL = "https://dl.orangedox.com/WyaCpL?dl=1"
CWQ = {
    "ComplexWebQuestions_train.json": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AAAIHeWX0cPpbpwK6w06BCxva/ComplexWebQuestions_train.json?dl=1",
    "ComplexWebQuestions_dev.json": "https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AADH8beLbOUWxwvY_K38E3ADa/ComplexWebQuestions_dev.json?dl=1",
}
PLAN = [
    # name, dest dir, file name, url, declared size (bytes or None), declared-size text, license text, expected exact bytes or None
    ("freebaseqa", "freebaseqa", "FreebaseQA-train.json", FQA_BASE + "FreebaseQA-train.json", 23888089, "23,888,089 B (GitHub API, FBQ_EXTERNAL_SOURCES__v1)", "CC-BY-4.0 (GitHub README, per FBQ_EXTERNAL_SOURCES__v1)", 23888089),
    ("freebaseqa", "freebaseqa", "FreebaseQA-dev.json", FQA_BASE + "FreebaseQA-dev.json", 4656349, "4,656,349 B (GitHub API, FBQ_EXTERNAL_SOURCES__v1)", "CC-BY-4.0 (GitHub README, per FBQ_EXTERNAL_SOURCES__v1)", 4656349),
    ("grailqa", "grailqa", "GrailQA_v1.0.zip", GRAIL_DL, None, "'~120 MB' (GrailQA homepage text, exact bytes unverified, FBQ_EXTERNAL_SOURCES__v1)", "CC BY-SA 4.0 (GrailQA homepage per FBQ_EXTERNAL_SOURCES__v1); Apache-2.0 for the code repo", None),
    ("cwq_original", "cwq_original", "ComplexWebQuestions_train.json", CWQ["ComplexWebQuestions_train.json"], None, "unverified in the survey; gate = HEAD size <= 200 MB", "Apache-2.0 on the HF loader card; data license 'Not specified' on the HF card (per FBQ_EXTERNAL_SOURCES__v1)", None),
    ("cwq_original", "cwq_original", "ComplexWebQuestions_dev.json", CWQ["ComplexWebQuestions_dev.json"], None, "unverified in the survey; gate = HEAD size <= 200 MB", "Apache-2.0 on the HF loader card; data license 'Not specified' on the HF card (per FBQ_EXTERNAL_SOURCES__v1)", None),
]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def head_chain(url, maxhops=8):
    """HEAD, following redirects manually; returns the hop list (headers only) and the final content-length. The token path of dl.dropboxusercontent.com is redacted in the record."""
    chain, opener = [], urllib.request.build_opener(NoRedirect)
    for _ in range(maxhops):
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        try:
            r = opener.open(req, timeout=30)
            code, h = r.status, r.headers
        except urllib.error.HTTPError as e:
            code, h = e.code, e.headers
        hh = {k.lower(): v for k, v in h.items() if k.lower() in ("content-length", "content-type", "location", "content-disposition", "etag")}
        host = urllib.parse.urlparse(url).netloc
        chain.append({"host": host, "path_prefix": urllib.parse.urlparse(url).path[:60], "status": code, "headers": {k: (v if k != "location" else urllib.parse.urlparse(v).netloc + urllib.parse.urlparse(v).path[:40] + "...") for k, v in hh.items()}})
        if code in (301, 302, 303, 307, 308) and "location" in hh:
            url = urllib.parse.urljoin(url, hh["location"])
            continue
        cl = hh.get("content-length")
        return chain, (int(cl) if (cl and code == 200) else None), code
    return chain, None, None


def free_bytes(path):
    return shutil.disk_usage(path).free


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_state():
    return json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}


def save_state(st):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)


def phase_fetch(only):
    st = load_state()
    for name, d, fn, url, declared, declared_txt, lic, exact in PLAN:
        if only and name not in only:
            continue
        key = name + "/" + fn
        dest_dir = os.path.join(ORIG, d)
        dest = os.path.join(dest_dir, fn)
        chain, clen, code = head_chain(url)
        print(key, "HEAD ->", code, "content-length", clen, flush=True)
        entry = {"name": name, "file": fn, "source_url": GRAIL_URL if name == "grailqa" else url, "download_endpoint": url, "local_path": os.path.relpath(dest, ROOT).replace("\\", "/"),
                 "declared_size": declared_txt, "declared_bytes": declared, "head_content_length": clen, "head_chain": chain, "license_as_stated": lic,
                 "publisher_hash": "none published (no checksum on the host page / repo); sha256 below is our own", "status": None}
        if clen is None:
            entry["status"] = "STOPPED_head_failed_or_no_content_length"
            st[key] = entry
            save_state(st)
            print("STOP", key, flush=True)
            continue
        if exact is not None and clen != exact:
            entry["status"] = "STOPPED_size_differs_from_survey(%d != %d)" % (clen, exact)
            st[key] = entry
            save_state(st)
            print("STOP", key, entry["status"], flush=True)
            continue
        if name == "cwq_original" and clen > MAX_CWQ_BYTES:
            entry["status"] = "NOT_DOWNLOADED_size_gt_200MB(%d)" % clen
            st[key] = entry
            save_state(st)
            print("STOP", key, entry["status"], flush=True)
            continue
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(dest):
            raise SystemExit("never overwrite: " + dest + " exists")
        fr = free_bytes(dest_dir)
        if fr - clen < MIN_FREE_AFTER:
            entry["status"] = "STOPPED_disk(free %d - %d < 3 GB)" % (fr, clen)
            st[key] = entry
            save_state(st)
            print("STOP", key, entry["status"], flush=True)
            continue
        part = dest + ".part"
        if os.path.exists(part):
            raise SystemExit("stale part file: " + part)
        t0 = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        n = 0
        h = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "xb") as f:
            for b in iter(lambda: r.read(1 << 20), b""):
                f.write(b)
                h.update(b)
                n += len(b)
        utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if n != clen:
            entry["status"] = "STOPPED_bytes_mismatch(got %d, HEAD %d)" % (n, clen)
            entry["part_file_left_for_inspection"] = os.path.relpath(part, ROOT).replace("\\", "/")
            st[key] = entry
            save_state(st)
            print("STOP", key, entry["status"], flush=True)
            continue
        os.rename(part, dest)
        entry.update(status="DOWNLOADED", measured_bytes=n, sha256=h.hexdigest(), downloaded_utc=utc, seconds=round(time.time() - t0, 2), free_bytes_after=free_bytes(dest_dir))
        st[key] = entry
        save_state(st)
        print("OK", key, n, h.hexdigest(), flush=True)


def safe_member(name):
    return not (name.startswith("/") or name.startswith("\\") or ".." in name.replace("\\", "/").split("/") or ":" in name)


def phase_list():
    zp = os.path.join(ORIG, "grailqa", "GrailQA_v1.0.zip")
    with zipfile.ZipFile(zp) as zf:
        tot = 0
        for i in zf.infolist():
            print("%-60s comp=%10d unc=%11d dir=%s safe=%s" % (i.filename, i.compress_size, i.file_size, i.is_dir(), safe_member(i.filename)))
            tot += i.file_size
        print("total uncompressed", tot)


def member_allowed(name):
    """a-priori rule: a safe, non-directory .json member whose basename contains 'train' or 'dev' and does NOT contain 'test'."""
    base = os.path.basename(name).lower()
    if name.startswith("__MACOSX") or base.startswith("._"):  # AppleDouble resource-fork shadows, never data
        return False
    return safe_member(name) and base.endswith(".json") and ("test" not in base) and (("train" in base) or ("dev" in base))


def phase_extract():
    zp = os.path.join(ORIG, "grailqa", "GrailQA_v1.0.zip")
    st = load_state()
    ex = {}
    with zipfile.ZipFile(zp) as zf:
        members = zf.infolist()
        names = [i.filename for i in members]
        unsafe = [n for n in names if not safe_member(n)]
        if unsafe:
            raise SystemExit("unsafe member names, refusing: " + repr(unsafe[:5]))
        for i in members:
            if i.is_dir() or not member_allowed(i.filename):
                ex[i.filename] = {"extracted": False, "bytes_listed": i.file_size, "reason": "not matching the train/dev rule" if not i.is_dir() else "directory"}
                continue
            dest = os.path.join(ORIG, "grailqa", os.path.basename(i.filename))
            if os.path.exists(dest):
                raise SystemExit("never overwrite: " + dest)
            fr = free_bytes(os.path.dirname(dest))
            if fr - i.file_size < MIN_FREE_AFTER:
                raise SystemExit("disk stop")
            h = hashlib.sha256()
            n = 0
            with zf.open(i) as src, open(dest, "xb") as out:
                for b in iter(lambda: src.read(1 << 20), b""):
                    out.write(b)
                    h.update(b)
                    n += len(b)
            ex[i.filename] = {"extracted": True, "local_path": os.path.relpath(dest, ROOT).replace("\\", "/"), "bytes": n, "bytes_listed": i.file_size, "crc32_listed": i.CRC, "sha256": h.hexdigest()}
            print("extracted", i.filename, n, flush=True)
    st["grailqa_members"] = ex
    save_state(st)


def phase_record():
    st = load_state()
    out = os.path.join(OUTDIR, "FBQ_DOWNLOAD_PROVENANCE__v1.json")
    assert not os.path.exists(out), "write-once: " + out
    rec = {"RECORD": "FBQ_DOWNLOAD_PROVENANCE", "version": "v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "approval": "user ruling 2026-10-04: only FreebaseQA train+dev, GrailQA archive (train+dev members), original ComplexWebQuestions train+dev if HEAD size <= 200 MB; nothing else; no test/eval split",
           "not_downloaded_by_rule": ["FreebaseQA-eval.json", "FreebaseQA-partial.json", "ComplexWebQuestions_test.json", "GraphQuestions (any)", "SimpleQuestions", "anything else"],
           "handling": "untrusted bytes: only json is parsed, nothing executed; zip members validated (no absolute path / '..' / drive colon) and only the train/dev JSON members extracted; test member bytes exist only inside the archive file and were not read or parsed",
           "publisher_hashes": "none published for any of these files; sha256 values are our own",
           "grailqa_extract_rule": "safe, non-directory .json member whose basename contains 'train' or 'dev' and not 'test'; __MACOSX/ and '._*' AppleDouble shadows excluded",
           "grailqa_size_note": "the survey said '~120 MB' (homepage text); the orangedox page and HEAD report GrailQA_v1.0.zip = 17,636,773 B (17.64 MB); the zip members total 126,183,720 B uncompressed, so the survey's 120 MB is the unzipped size",
           "download_hosts_contacted": ["raw.githubusercontent.com", "dl.orangedox.com -> www.dropbox.com -> dl.dropboxusercontent.com (redirect chain of the approved GrailQA link)", "www.dropbox.com -> dl.dropboxusercontent.com (the Dropbox links named in the HF loader script for CWQ train/dev)", "huggingface.co (read the small text loader script complex_web_questions.py once, to read the two Dropbox URLs; no data downloaded from it)"],
           "files": {k: v for k, v in st.items() if k != "grailqa_members"}, "grailqa_archive_members": st.get("grailqa_members"),
           "code": {"scratchpad/_fbq_download_v2.py": sha_file(os.path.abspath(__file__))}}
    with open(out, "x", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["fetch", "list-grailqa", "extract-grailqa", "record"])
    ap.add_argument("--only", default="", help="comma list of freebaseqa,grailqa,cwq_original")
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    {"fetch": lambda: phase_fetch(only), "list-grailqa": phase_list, "extract-grailqa": phase_extract, "record": phase_record}[a.phase]()
