"""Third-copy backup of the big laptop-master data to the SECOND HF account (user 2026-10-05: new token 'crag' in DarKnight9895, "put and cleanup").

Why: the laptop is the only copy of final_canonical/{metaqa,musique,squad,webqsp}; hotpotqa/2wiki and the Freebase tree exist on laptop + host only (their relay repos are deleted).
How:  every file is hashed (sha256 of the ORIGINAL bytes) and, if it compresses (sampled zstd ratio < RAW_ABOVE), stored as <path>.zst; incompressible files are stored as-is.  Streaming: at most
      two ~BATCH_GB batches are staged on the laptop disk; one HF commit per batch; resumable (state file) -- re-run the same command after any interruption.  Private repo only (asserts).
      Restore verifies the sha256 of the decompressed bytes against the index, so a restored tree is bit-identical to the one CANONICAL_FREEZE pins.

  python scratchpad/_hf_big.py PLAN                 sample compressibility -> data/_cache/hfbig_plan.json, print the estimated stored size per group
  python scratchpad/_hf_big.py UPLOAD [groups,csv]  stream compress+hash+commit (default: all groups in priority order, stops before the quota guard)
  python scratchpad/_hf_big.py CHECK                repo tree (size + LFS sha256) vs the state file
  python scratchpad/_hf_big.py RESTORE <dest> [groups,csv]   download + decompress + verify into <dest>/final_canonical/...   (token from env HF_TOKEN or ~/hf_tokens.json account DarKnight9895)
The token is never printed or written anywhere inside the repo."""
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
import time

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
import zstandard as zstd  # noqa: E402
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download  # noqa: E402

ROOT = "C:/Users/Swastik/Desktop/CRAG"
SRC = ROOT + "/data/final_canonical"
CACHE = ROOT + "/data/_cache"
STAGE = CACHE + "/hfbig_stage"
PLAN_P = CACHE + "/hfbig_plan.json"
STATE_P = CACHE + "/hfbig_state.json"
REPO = "DarKnight9895/crag-large-backup"
ACCOUNT = "DarKnight9895"
QUOTA_GB = 95.0                    # free private quota is ~100 GB; keep a margin
BATCH_GB = 8.0
RAW_ABOVE = 0.88                   # sampled zstd ratio above this -> store raw
LEVEL = 3
# priority order = single-copy first (laptop is the only copy), then the two-copy data, then the tree
GROUPS = [
    ("g0_meta", ["__top__", "_history"]),
    ("g1_small", ["metaqa", "musique", "squad", "webqsp"]),
    ("g2_text", ["hotpotqa", "2wiki"]),
    ("g3_tree", ["freebase"]),
]
SKIP_NAMES = {"__pycache__", "freebase_v3"}


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    for a in json.load(open(os.path.join(os.path.expanduser("~"), "hf_tokens.json")))["accounts"]:
        if a.get("account") == ACCOUNT:
            return a["token"]
    raise SystemExit("no token for %s (set HF_TOKEN or add it to ~/hf_tokens.json)" % ACCOUNT)


def enumerate_objects():
    objs = []
    for g, members in GROUPS:
        for m in members:
            if m == "__top__":
                for f in sorted(os.listdir(SRC)):
                    fp = SRC + "/" + f
                    if os.path.isfile(fp):
                        objs.append({"group": g, "rel": f, "size": os.path.getsize(fp)})
                continue
            base = SRC + "/" + m
            if not os.path.isdir(base) or m in SKIP_NAMES:
                continue
            for d, dirs, fs in os.walk(base):
                dirs[:] = sorted(x for x in dirs if x not in SKIP_NAMES)
                for f in sorted(fs):
                    fp = os.path.join(d, f).replace("\\", "/")
                    objs.append({"group": g, "rel": fp[len(SRC) + 1:], "size": os.path.getsize(fp)})
    return objs


def sample_ratio(fp, size):
    c = zstd.ZstdCompressor(level=LEVEL)
    if size == 0:
        return 1.0
    if size <= (48 << 20):
        b = open(fp, "rb").read()
        return len(c.compress(b)) / len(b)
    n = m = 0
    with open(fp, "rb") as f:
        for frac in (0.15, 0.5, 0.85):
            f.seek(int(size * frac))
            b = f.read(4 << 20)
            n += len(b)
            m += len(c.compress(b))
    return m / max(1, n)


def cmd_plan():
    objs = enumerate_objects()
    t0 = time.time()
    for i, o in enumerate(objs):
        r = sample_ratio(SRC + "/" + o["rel"], o["size"])
        o["ratio"] = round(r, 3)
        o["mode"] = "zst" if r < RAW_ABOVE else "raw"
        o["est_stored"] = int(o["size"] * (r if o["mode"] == "zst" else 1.0))
        if i % 100 == 0:
            log("sampled", i, "/", len(objs))
    json.dump({"objects": objs}, open(PLAN_P, "w"))
    cum = 0.0
    print("group           files    raw GB   est stored GB   cumulative GB")
    for g, _ in GROUPS:
        sel = [o for o in objs if o["group"] == g]
        raw = sum(o["size"] for o in sel) / 1e9
        est = sum(o["est_stored"] for o in sel) / 1e9
        cum += est
        print("%-14s %6d %9.2f %14.2f %15.2f" % (g, len(sel), raw, est, cum))
    log("plan done in %.0fs -> %s" % (time.time() - t0, PLAN_P))


class HashReader:
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        return b


class HashWriter:
    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        return self.f.write(b)

    def flush(self):
        self.f.flush()


def prepare(o):
    """returns the entry {rel, mode, orig_size, orig_sha, stored_path(local), stored_rel(repo), stored_size, stored_sha}"""
    src = SRC + "/" + o["rel"]
    if o["mode"] == "raw":
        h = hashlib.sha256()
        with open(src, "rb") as f:
            while True:
                b = f.read(16 << 20)
                if not b:
                    break
                h.update(b)
        d = h.hexdigest()
        return {"rel": o["rel"], "mode": "raw", "orig_size": o["size"], "orig_sha": d, "stored_local": src, "stored_rel": "final_canonical/" + o["rel"],
                "stored_size": o["size"], "stored_sha": d, "staged": False}
    dst = STAGE + "/" + o["rel"] + ".zst"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    c = zstd.ZstdCompressor(level=LEVEL, threads=-1)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        hr, hw = HashReader(fi), HashWriter(fo)
        c.copy_stream(hr, hw, size=o["size"])
    return {"rel": o["rel"], "mode": "zst", "orig_size": o["size"], "orig_sha": hr.h.hexdigest(), "stored_local": dst, "stored_rel": "final_canonical/" + o["rel"] + ".zst",
            "stored_size": hw.n, "stored_sha": hw.h.hexdigest(), "staged": True}


def load_state():
    try:
        return json.load(open(STATE_P))
    except Exception:  # noqa: BLE001
        return {"done": {}}


def save_state(st):
    tmp = STATE_P + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, STATE_P)


def private_repo(api):
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    assert api.repo_info(REPO, repo_type="dataset").private is True, "REFUSING: %s is not private" % REPO


def cmd_upload(groups):
    objs = json.load(open(PLAN_P))["objects"]
    st = load_state()
    todo = [o for o in objs if o["group"] in groups and o["rel"] not in st["done"]]
    log("todo %d objects (%d already done), est %.2f GB stored" % (len(todo), len(st["done"]), sum(o["est_stored"] for o in todo) / 1e9))
    api = HfApi(token=token())
    private_repo(api)
    batches, cur, acc = [], [], 0
    for o in todo:
        if cur and acc + o["est_stored"] > BATCH_GB * 1e9:
            batches.append(cur)
            cur, acc = [], 0
        cur.append(o)
        acc += o["est_stored"]
    if cur:
        batches.append(cur)
    q = queue.Queue(maxsize=1)
    stop = threading.Event()
    done_gb = sum(v["stored_size"] for v in st["done"].values()) / 1e9

    def producer():
        nonlocal done_gb
        committed = done_gb
        for bi, b in enumerate(batches):
            if stop.is_set():
                break
            est = sum(o["est_stored"] for o in b) / 1e9
            if committed + est > QUOTA_GB:
                log("QUOTA GUARD: %.1f + %.1f GB > %.0f GB -> stop before batch %d" % (committed, est, QUOTA_GB, bi))
                break
            t0 = time.time()
            ents = [prepare(o) for o in b]
            committed += sum(e["stored_size"] for e in ents) / 1e9
            log("batch %d/%d prepared: %d files, %.2f GB stored (%.0fs)" % (bi + 1, len(batches), len(ents), sum(e["stored_size"] for e in ents) / 1e9, time.time() - t0))
            q.put((bi, ents))
        q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is None:
            break
        bi, ents = item
        ops = [CommitOperationAdd(path_in_repo=e["stored_rel"], path_or_fileobj=e["stored_local"]) for e in ents]
        t0 = time.time()
        for attempt in range(4):
            try:
                api.create_commit(REPO, repo_type="dataset", operations=ops, commit_message="batch %d: %d files" % (bi + 1, len(ents)))
                break
            except Exception as e:  # noqa: BLE001
                log("commit attempt %d failed: %s" % (attempt + 1, str(e)[:200]))
                if attempt == 3:
                    stop.set()
                    raise
                time.sleep(30 * (attempt + 1))
        gb = sum(e["stored_size"] for e in ents) / 1e9
        log("batch %d committed: %.2f GB in %.0fs (%.1f MB/s)" % (bi + 1, gb, time.time() - t0, gb * 1e3 / max(1, time.time() - t0)))
        for e in ents:
            st["done"][e["rel"]] = {k: v for k, v in e.items() if k not in ("stored_local", "staged")}
            if e["staged"]:
                try:
                    os.remove(e["stored_local"])
                except OSError:
                    pass
        save_state(st)
        write_index(api, st)
    th.join()
    shutil.rmtree(STAGE, ignore_errors=True)
    log("UPLOAD finished: %d objects, %.2f GB stored in the repo" % (len(st["done"]), sum(v["stored_size"] for v in st["done"].values()) / 1e9))


def write_index(api, st):
    idx = {"repo": REPO, "written": time.strftime("%F %T"), "objects": st["done"], "format": "stored_rel = repo path; mode zst => zstd -> original; orig_sha = sha256 of the original bytes"}
    api.upload_file(path_or_fileobj=json.dumps(idx).encode(), path_in_repo="MANIFEST/HFBIG_INDEX.json", repo_id=REPO, repo_type="dataset", commit_message="index (%d objects)" % len(st["done"]))


def cmd_check():
    st = load_state()
    api = HfApi(token=token())
    have = {}
    for e in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True):
        if hasattr(e, "size") and e.path.startswith("final_canonical/"):
            have[e.path] = (e.size, getattr(getattr(e, "lfs", None), "sha256", None) if getattr(e, "lfs", None) else None)
    bad = miss = 0
    for rel, v in st["done"].items():
        h = have.get(v["stored_rel"])
        if h is None:
            miss += 1
            continue
        if h[0] != v["stored_size"] or (h[1] and h[1] != v["stored_sha"]):
            bad += 1
    log("CHECK: state %d objects, repo %d objects, missing %d, size/sha mismatch %d, stored %.2f GB" % (len(st["done"]), len(have), miss, bad, sum(v["stored_size"] for v in st["done"].values()) / 1e9))
    return miss == 0 and bad == 0


def cmd_restore(dest, groups):
    api = HfApi(token=token())
    p = hf_hub_download(REPO, "MANIFEST/HFBIG_INDEX.json", repo_type="dataset", token=token())
    idx = json.load(open(p))["objects"]
    plan = {o["rel"]: o["group"] for o in json.load(open(PLAN_P))["objects"]} if os.path.exists(PLAN_P) else {}
    n = bad = 0
    for rel, v in idx.items():
        if groups and plan.get(rel) not in groups:
            continue
        lp = hf_hub_download(REPO, v["stored_rel"], repo_type="dataset", token=token())
        out = dest.rstrip("/") + "/final_canonical/" + rel
        os.makedirs(os.path.dirname(out), exist_ok=True)
        h = hashlib.sha256()
        with open(out, "wb") as fo:
            if v["mode"] == "zst":
                with open(lp, "rb") as fi:
                    for chunk in zstd.ZstdDecompressor().read_to_iter(fi, read_size=16 << 20):
                        h.update(chunk)
                        fo.write(chunk)
            else:
                with open(lp, "rb") as fi:
                    while True:
                        b = fi.read(16 << 20)
                        if not b:
                            break
                        h.update(b)
                        fo.write(b)
        if h.hexdigest() != v["orig_sha"]:
            bad += 1
            log("SHA MISMATCH", rel)
        n += 1
    log("RESTORE: %d files, %d sha mismatches" % (n, bad))


def cmd_urls(groups, out):
    """signed ~1 h CDN links of the stored objects of <groups> (+ the index) so a host with NO token can pull them with scratchpad/_fbx_hf_pull.py and unpack with _hf_big_unpack.py"""
    import base64
    import urllib.parse
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import get_session
    st = load_state()
    plan = {o["rel"]: o["group"] for o in json.load(open(PLAN_P))["objects"]}
    tk = token()
    hdr = {"Authorization": "Bearer " + tk, "User-Agent": "crag-hfbig"}
    s = get_session()

    def resolve(path):
        url = hf_hub_url(REPO, path, repo_type="dataset")
        for _ in range(6):
            h = hdr if (urllib.parse.urlparse(url).hostname or "").endswith("huggingface.co") else {}
            r = s.get(url, headers=h, allow_redirects=False, timeout=60, stream=True)
            if r.status_code in (301, 302, 303, 307, 308):
                url = urllib.parse.urljoin(url, r.headers["Location"])
                r.close()
                if not (urllib.parse.urlparse(url).hostname or "").endswith("huggingface.co"):
                    return {"url": url}
                continue
            if r.status_code == 200:
                b = r.content
                r.close()
                return {"inline_b64": base64.b64encode(b).decode("ascii")}
            r.raise_for_status()
        raise RuntimeError("too many redirects: " + path)

    files, exp = [], None
    for rel, v in sorted(st["done"].items()):
        if plan.get(rel) in groups or "all" in groups:
            r = resolve(v["stored_rel"])
            files.append(dict(path=v["stored_rel"], size=v["stored_size"], **r))
            if "url" in r:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(r["url"]).query)
                exp = int(q["X-Amz-Expires"][0]) if "X-Amz-Expires" in q else exp
    r = resolve("MANIFEST/HFBIG_INDEX.json")
    idx = base64.b64decode(r["inline_b64"]) if "inline_b64" in r else None
    files.append(dict(path="MANIFEST/HFBIG_INDEX.json", size=len(idx), **r) if idx is not None else dict(path="MANIFEST/HFBIG_INDEX.json", size=0, **r))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"repo": REPO, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "generated_epoch": int(time.time()), "url_lifetime_s": exp, "files": files}, open(out, "w"))
    log("URLS: %d files, %.2f GB, lifetime %s s -> %s" % (len(files), sum(f["size"] for f in files) / 1e9, exp, out))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "URLS":
        cmd_urls(sys.argv[2].split(","), sys.argv[3] if len(sys.argv) > 3 else CACHE + "/hfbig_urls.json")
    elif c == "PLAN":
        cmd_plan()
    elif c == "UPLOAD":
        cmd_upload(sys.argv[2].split(",") if len(sys.argv) > 2 else [g for g, _ in GROUPS])
    elif c == "CHECK":
        sys.exit(0 if cmd_check() else 1)
    elif c == "RESTORE":
        cmd_restore(sys.argv[2], sys.argv[3].split(",") if len(sys.argv) > 3 else None)
    else:
        raise SystemExit(__doc__)
