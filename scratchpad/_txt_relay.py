"""Hugging Face relay for the HotpotQA / 2Wiki substrates -> the GPU host (user 2026-10-02: "push the data use hf as relay, cleanup space if required").

Same lane as the Freebase relay (_fbx_stage.py / _fbx_hf_upload.py / _fbx_hf_urls.py / _fbx_hf_pull.py), generalised to several datasets.  Differences:
  * the token is the one the user stored under the NAME 'freebase' (account Swastik9895, the CRAG relay token) and is passed explicitly in-process; the globally ACTIVE stored token ('mpr', another
    account) is never used and never switched.  The token is never printed or written to a file, and Authorization is only ever sent to huggingface.co.
  * the repo is PRIVATE (the script refuses a non-private repo).
  * only what the host harnesses read is staged: embeddings/ graph/ retrieval_cache/ queries/ and the three small manifests; nodes.jsonl, legacy maps and the audits stay home.

  python scratchpad/_txt_relay.py STAGE            hard-link stage of the files under data/_cache/txt_stage/<ds>/  (refuses a non-empty stage)
  python scratchpad/_txt_relay.py DROP             remove the stage (links only; the dataset files are untouched)
  python scratchpad/_txt_relay.py UPLOAD           upload_large_folder of the stage to the private repo (resumable: re-run after any interruption)
  python scratchpad/_txt_relay.py STATUS           files in the repo vs the stage (name + size)
  python scratchpad/_txt_relay.py URLS             signed, ~1 h download links -> data/_cache/txt_urls.json (git-ignored); the host pull needs no token
"""
import base64
import configparser
import json
import os
import sys
import time
import urllib.parse

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
from huggingface_hub import HfApi, constants, hf_hub_url  # noqa: E402
from huggingface_hub.utils import get_session  # noqa: E402

ROOT = "C:/Users/Swastik/Desktop/CRAG"
SRC = ROOT + "/data/final_canonical"
STAGE = ROOT + "/data/_cache/txt_stage"
URLS = ROOT + "/data/_cache/txt_urls.json"
REPO = "Swastik9895/crag-text-substrates"
DATASETS = ("hotpotqa", "2wiki")
DIRS = ("embeddings", "graph", "retrieval_cache", "queries")
FILES = ("dataset_manifest.json", "build_info.json", "integrity_report.json")
INLINE_MAX = 20 * (1 << 20)


def token():
    cp = configparser.ConfigParser()
    cp.read(constants.HF_STORED_TOKENS_PATH)
    return cp["freebase"]["hf_token"]


def api():
    return HfApi(token=token())


def selected():
    """[(abs path, repo path)]"""
    out = []
    for ds in DATASETS:
        base = SRC + "/" + ds
        for f in FILES:
            p = base + "/" + f
            if os.path.isfile(p):
                out.append((p, ds + "/" + f))
        for d in DIRS:
            for r, dn, fs in os.walk(base + "/" + d):
                dn.sort()
                for f in sorted(fs):
                    p = os.path.join(r, f).replace("\\", "/")
                    out.append((p, ds + "/" + os.path.relpath(p, base).replace("\\", "/")))
    return out


def stage():
    assert not (os.path.isdir(STAGE) and os.listdir(STAGE)), "stage exists and is not empty: " + STAGE
    n = b = 0
    for p, rp in selected():
        d = os.path.dirname(STAGE + "/" + rp)
        os.makedirs(d, exist_ok=True)
        os.link(p, STAGE + "/" + rp)
        n += 1
        b += os.path.getsize(p)
    print("staged %d files, %.3f GB (%d bytes) at %s" % (n, b / 1e9, b, STAGE))


def drop():
    for r, ds, fs in os.walk(STAGE, topdown=False):
        for f in fs:
            os.remove(os.path.join(r, f))
        os.rmdir(r)
    print("stage removed")


def private_repo(a):
    a.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    assert a.repo_info(REPO, repo_type="dataset").private is True, "REFUSING: %s is not private" % REPO


def upload():
    a = api()
    print("logged in as:", a.whoami()["name"], flush=True)
    private_repo(a)
    assert os.path.isdir(STAGE) and os.listdir(STAGE), "run STAGE first"
    t0 = time.time()
    a.upload_large_folder(repo_id=REPO, repo_type="dataset", folder_path=STAGE, num_workers=8, print_report=True, print_report_every=60)
    print("upload finished in %.0f s" % (time.time() - t0), flush=True)


def status():
    a = api()
    have = {e.path: e.size for e in a.list_repo_tree(REPO, repo_type="dataset", recursive=True) if hasattr(e, "size")}
    want = {}
    for r, ds, fs in os.walk(STAGE):
        if ".cache" in r.replace("\\", "/").split("/"):
            continue
        for f in fs:
            p = os.path.join(r, f)
            want[os.path.relpath(p, STAGE).replace("\\", "/")] = os.path.getsize(p)
    ok = sum(1 for k, v in want.items() if have.get(k) == v)
    print("repo has %d of %d staged files with the right size (%.3f GB of %.3f GB)" % (ok, len(want), sum(v for k, v in want.items() if have.get(k) == v) / 1e9, sum(want.values()) / 1e9))


def resolve(url, hdr_auth):
    s = get_session()
    for _ in range(5):
        host = urllib.parse.urlparse(url).hostname or ""
        hdr = hdr_auth if host.endswith("huggingface.co") else {}
        r = s.get(url, headers=hdr, allow_redirects=False, timeout=60, stream=True)
        if r.status_code in (301, 302, 303, 307, 308):
            url = urllib.parse.urljoin(url, r.headers["Location"])
            r.close()
            if not (urllib.parse.urlparse(url).hostname or "").endswith("huggingface.co"):
                return "url", url
            continue
        if r.status_code == 200:
            n = int(r.headers.get("Content-Length", "0"))
            assert n <= INLINE_MAX, "no redirect for a %d-byte file: %s" % (n, url.split("?")[0])
            b = r.content
            r.close()
            return "inline", b
        r.raise_for_status()
    raise RuntimeError("too many redirects")


def urls():
    a = api()
    hdr = {"Authorization": "Bearer " + token(), "User-Agent": "crag-relay"}
    files = sorted((e.path, e.size) for e in a.list_repo_tree(REPO, repo_type="dataset", recursive=True) if hasattr(e, "size") and not e.path.startswith("_probe/") and not e.path.endswith(".gitattributes"))
    out, exp = [], None
    for p, size in files:
        kind, v = resolve(hf_hub_url(REPO, p, repo_type="dataset"), hdr)
        if kind == "url":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(v).query)
            if "X-Amz-Expires" in q:
                exp = int(q["X-Amz-Expires"][0])
            out.append({"path": p, "size": size, "url": v})
        else:
            assert len(v) == size, "inline size differs: " + p
            out.append({"path": p, "size": size, "inline_b64": base64.b64encode(v).decode("ascii")})
    os.makedirs(os.path.dirname(URLS), exist_ok=True)
    with open(URLS, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"repo": REPO, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generated_epoch": int(time.time()), "url_lifetime_s": exp, "files": out}, f)
    print("%d files (%d by link, %d inline), %.3f GB, link lifetime %s s -> %s" % (len(out), sum(1 for o in out if "url" in o), sum(1 for o in out if "inline_b64" in o), sum(o["size"] for o in out) / 1e9, exp, URLS))


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else ""
    {"STAGE": stage, "DROP": drop, "UPLOAD": upload, "STATUS": status, "URLS": urls}.get(m, lambda: sys.exit(__doc__))()
