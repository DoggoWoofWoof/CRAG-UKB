"""Time-limited download links for the private Freebase repo (FREEBASE_SCALE lane transfer, user 2026-09-30).

The host holds NO token (same rule as _enc_hfdl.py).  This runs on the LAPTOP with the token the user stored via `hf auth login` (never read
into a file, never printed, sent only to huggingface.co): for every file of the repo it follows the resolve redirect by hand and keeps the
signed CDN URL (no Authorization is ever sent to a non-huggingface.co host).  Small git-stored files answer 200 with their bytes instead of a
redirect; those are embedded (base64, <= 20 MB each).  Output data/_cache/fbx_urls.json (git-ignored; contains short-lived signed URLs).

Usage: python scratchpad/_fbx_hf_urls.py Swastik9895/crag-freebase-tree [--only SUBSTR ...]
"""
import base64
import json
import os
import sys
import time
import urllib.parse

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import HfApi, hf_hub_url  # noqa: E402
from huggingface_hub.utils import build_hf_headers, get_session  # noqa: E402

OUT = "C:/Users/Swastik/Desktop/CRAG/data/_cache/fbx_urls.json"
INLINE_MAX = 20 * (1 << 20)


def resolve(url):
    """(kind, value): ('url', signed cdn url) or ('inline', bytes)."""
    s = get_session()
    for _ in range(5):
        host = urllib.parse.urlparse(url).hostname or ""
        hdr = build_hf_headers() if host.endswith("huggingface.co") else {}
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


def main():
    repo = sys.argv[1]
    only = [a for a in sys.argv[3:]] if len(sys.argv) > 2 and sys.argv[2] == "--only" else []
    api = HfApi()
    files = sorted((e.path, e.size) for e in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if hasattr(e, "size") and not e.path.startswith("_probe/"))
    if only:
        files = [f for f in files if any(o in f[0] for o in only)]
    out = []
    exp = None
    for p, size in files:
        kind, v = resolve(hf_hub_url(repo, p, repo_type="dataset"))
        if kind == "url":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(v).query)
            if "X-Amz-Expires" in q:
                exp = int(q["X-Amz-Expires"][0])
            out.append({"path": p, "size": size, "url": v})
        else:
            assert len(v) == size, "inline size differs: " + p
            out.append({"path": p, "size": size, "inline_b64": base64.b64encode(v).decode("ascii")})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"repo": repo, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generated_epoch": int(time.time()),
                   "url_lifetime_s": exp, "files": out}, f)
    print("%d files (%d by link, %d inline), %.3f GB, link lifetime %s s -> %s" % (
        len(out), sum(1 for o in out if "url" in o), sum(1 for o in out if "inline_b64" in o), sum(o["size"] for o in out) / 1e9, exp, OUT))


if __name__ == "__main__":
    main()
