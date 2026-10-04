"""Host housekeeping -- back up host-only crag data to a Hugging Face DATASET repo WITHOUT putting any HF token on the shared lab host.

HF's LFS protocol lets the token holder (the laptop) ask for presigned S3 multipart URLs; the host only PUTs bytes to those URLs and posts the part etags to the `complete_multipart` href (no auth header); the laptop then makes
the one commit and verifies the repo's size + sha256.  Five steps (the host steps are `rx run` jobs under the yield wrapper, the laptop steps use the token from C:\\Users\\Swastik\\hf_tokens.json or $HF_TOKEN):

  host    PLAN    python -u scratchpad/_hfx.py PLAN <tag> <group>[=<rel dir>] ...     deterministic tar shards (+ big single files) -> work/HOST_HOUSEKEEPING/hfx_plan_<tag>.json (size + sha256 of every object)
  laptop  BATCH   python scratchpad/_hfx.py BATCH <tag> <repo>                         presigned URLs for every object -> data/_cache/hfx_urls_<tag>.json (push to the host with `rx run --inputs`)
  host    UPLOAD  python -u scratchpad/_hfx.py UPLOAD <tag> [threads]                 re-creates each shard (sha256 must equal the plan), PUTs the parts, completes the multipart; resumable via hfx_state_<tag>.json
  laptop  COMMIT  python scratchpad/_hfx.py COMMIT <tag> <repo>                       (verify action for small objects) + ONE commit of lfsFile ops + manifest, then size + sha256 verification against the plan
  laptop  CHECK   python scratchpad/_hfx.py CHECK <tag> <repo>                        repo vs plan only

Groups (workspace-relative source dirs; `name=rel` adds/overrides): see GROUPS.  Objects: files > BIG_GB are uploaded as themselves, the rest are packed (uncompressed GNU tar, fixed mtime/uid/gid/mode, member path = workspace-relative,
files sorted) into shards of ~SHARD_GB.  A shard is written at most once at a time (<= SHARD_GB of scratch under work/HOST_HOUSEKEEPING/pack/, removed after its upload).
"""
import base64
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LFS = {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"}
GROUPS = {
    "fbs_names": "data/freebase_scale/names", "fbs_ner": "data/freebase_scale/ner", "fbs_pq": "data/freebase_scale/pq", "fbs_ivft": "data/freebase_scale/ivf_transfer", "fbs_enc": "data/freebase_scale/enc",
    "calib": "work/FBX_CALIB/bundle_v1", "l1parts": "results/L1_HOST/parts", "r_ml2": "results/FREEBASE_SCALE/ml2", "r_h2l": "results/FREEBASE_SCALE/h2l", "r_ivf": "results/FREEBASE_SCALE/ivf",
    "r_partsS": "results/FREEBASE_SCALE/parts_S",
    "fbs_enc_meta": "data/freebase_scale/enc",                  # top-level files only (codebook, train/holdout embeddings, probe sample); the chunks are group fbs_enc
    "fbs_enc": "data/freebase_scale/enc/chunks",
    "recs_big": "results",                                      # result files larger than RECS_MAX_MB (the smaller ones are synced to the laptop by _host_sync.py)
    "work_big": "work",                                         # run outputs > RECS_MAX_MB (minus EXCLUDE)
    # root "wsl" groups (run through scratchpad/_hfx_wsl.py): rels are relative to /home/student2/crag_ooc
    "wsl_fb": "fb", "wsl_fb_l1": "fb_l1", "wsl_fb_work": "fb_work",
}
ROOT_OF = {"wsl_fb": "wsl", "wsl_fb_l1": "wsl", "wsl_fb_work": "wsl"}
SHARD_GB = float(os.environ.get("HFX_SHARD_GB", "2.0"))
BIG_GB = float(os.environ.get("HFX_BIG_GB", "3.0"))
WS = os.environ.get("HFX_WS") or os.getcwd()                    # root every member path (rel) is relative to; the WSL ext4 tree for the wsl_* groups
STATE = os.environ.get("HFX_STATE") or os.getcwd()               # where plans / state / url files live (the rx workspace)
ROOT = os.environ.get("HFX_ROOT", "ws")                          # restore root tag of this process's rels: "ws" (rx workspace) or "wsl" (/home/student2/crag_ooc)
OUT = os.path.join(STATE, "work", "HOST_HOUSEKEEPING")
CACHE = os.path.join(STATE, "data", "_cache")
PACK = os.environ.get("HFX_PACK") or os.path.join(OUT, "pack")
SETTLE_S = float(os.environ.get("HFX_SETTLE_S", "0"))            # skip files modified in the last N seconds (a writer may still hold them)
KNOWN = os.environ.get("HFX_KNOWN")                              # json list of "rel|size" already in the repo -> only new files are planned (names carry the tag)
RECS_MAX_B = int(float(os.environ.get("RECS_MAX_MB", "8")) * 1e6)
EXCLUDE = ("work/L1_HOST/phg_data", "work/HOST_HOUSEKEEPING", "work/FBX_SCALE")   # completed-lane scratch / our own transfer scratch / regression scratch: never backed up
SKIP_SUFFIX = (".part", ".rxpart", ".tmp")
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Sink:
    """a write-only file object that hashes what it receives and optionally keeps it"""

    def __init__(self, path=None):
        self.h = hashlib.sha256()
        self.n = 0
        self.f = open(path, "wb") if path else None

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        if self.f:
            self.f.write(b)
        return len(b)

    def close(self):
        if self.f:
            self.f.close()


def _det(ti):
    ti.mtime = 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mode = 0o644
    return ti


def write_tar(sink, rels):
    with tarfile.open(fileobj=sink, mode="w|", format=tarfile.GNU_FORMAT) as tf:
        for r in rels:
            tf.add(os.path.join(WS, r.replace("/", os.sep)), arcname=r, recursive=False, filter=_det)


def list_group(srcdir, top_only=False):
    root = os.path.join(WS, srcdir.replace("/", os.sep))
    known = set(json.load(open(KNOWN))) if KNOWN else set()
    out = []
    now = time.time()
    for dp, dn, fn in os.walk(root):
        if top_only:
            dn[:] = []
        for f in fn:
            if f.endswith(SKIP_SUFFIX):
                continue
            p = os.path.join(dp, f)
            st = os.stat(p)
            if SETTLE_S and now - st.st_mtime < SETTLE_S:
                continue
            rel = os.path.relpath(p, WS).replace("\\", "/")
            if "%s|%d" % (rel, st.st_size) in known:
                continue
            if srcdir in ("results", "work") and (st.st_size <= RECS_MAX_B or rel.startswith(EXCLUDE)):
                continue
            out.append((rel, st.st_size))
    return sorted(out)


def objects_for(group, srcdir, tag=""):
    files = list_group(srcdir, top_only=(group == "fbs_enc_meta"))
    big = int(BIG_GB * 1e9)
    shard = int(SHARD_GB * 1e9)
    nm = "%s_%s" % (group, tag) if KNOWN else group                    # incremental plans carry the tag so a later batch never collides with an earlier object
    objs = [{"group": group, "kind": "file", "root": ROOT_OF.get(group, ROOT), "remote": "%s/%s" % (group, ("%s_" % tag if KNOWN else "") + os.path.basename(r)), "rels": [r], "size": s} for r, s in files if s > big]
    small = [(r, s) for r, s in files if s <= big]
    cur, cb, k = [], 0, 0
    for r, s in small + [(None, 0)]:
        if r is None or (cur and cb + s > shard):
            if cur:
                objs.append({"group": group, "kind": "tar", "root": ROOT_OF.get(group, ROOT), "remote": "%s/%s_%04d.tar" % (group, nm, k), "rels": [x for x, _ in cur], "size": None, "members": [{"rel": x, "size": y} for x, y in cur]})
                k += 1
            cur, cb = [], 0
        if r is not None:
            cur.append((r, s))
            cb += s
    return objs


def plan_one(o):
    if o["kind"] == "file":
        o["size"], o["sha256"] = os.path.getsize(os.path.join(WS, o["rels"][0].replace("/", os.sep))), sha_file(os.path.join(WS, o["rels"][0].replace("/", os.sep)))
    else:
        s = Sink()
        write_tar(s, o["rels"])
        o["size"], o["sha256"] = s.n, s.h.hexdigest()
    log("planned %s %.3f GB sha %s" % (o["remote"], o["size"] / 1e9, o["sha256"][:12]))
    return o


def cmd_plan(tag, specs):
    objs = []
    for sp in specs:
        g, _, d = sp.partition("=")
        objs += objects_for(g, d or GROUPS[g], tag)
    log("%d objects, %d files" % (len(objs), sum(len(o["rels"]) for o in objs)))
    os.makedirs(PACK, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        objs = list(ex.map(plan_one, objs))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "hfx_plan_%s.json" % tag)
    json.dump({"tag": tag, "shard_gb": SHARD_GB, "big_gb": BIG_GB, "objects": objs, "bytes": sum(o["size"] for o in objs)}, open(p, "w"))
    log("PLAN %s: %d objects, %.2f GB -> %s" % (tag, len(objs), sum(o["size"] for o in objs) / 1e9, p))


# ---------------------------------------------------------------- laptop steps
def token():
    t = os.environ.get("HF_TOKEN")
    if not t:
        t = json.load(open(os.path.join(os.path.expanduser("~"), "hf_tokens.json")))["accounts"][0]["token"]
    return t


def cmd_batch(tag, repo):
    from huggingface_hub import HfApi
    from huggingface_hub._commit_api import UploadInfo
    from huggingface_hub.lfs import post_lfs_batch_info
    HfApi(token=token()).create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    plan = json.load(open(os.path.join(OUT, "hfx_plan_%s.json" % tag)))
    objs = plan["objects"]
    urls = {}
    for i in range(0, len(objs), 50):
        part = objs[i:i + 50]
        infos = [UploadInfo(sha256=bytes.fromhex(o["sha256"]), size=o["size"], sample=b"") for o in part]
        res, errs, _ = post_lfs_batch_info(infos, token=token(), repo_type="dataset", repo_id=repo, revision="main")
        assert not errs, errs
        for r in res:
            a = r.get("actions") or {}
            if not a:
                urls[r["oid"]] = {"done": True}
                continue
            up = a["upload"]
            hd = up.get("header", {})
            if "chunk_size" in hd:
                parts = [u for _, u in sorted(((int(k), v) for k, v in hd.items() if k.isdigit()))]
                urls[r["oid"]] = {"mode": "multipart", "href": up["href"], "chunk_size": int(hd["chunk_size"]), "parts": parts}
            else:
                urls[r["oid"]] = {"mode": "basic", "href": up["href"], "verify": a.get("verify")}
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "hfx_urls_%s.json" % tag)
    json.dump({"repo": repo, "urls": urls}, open(p, "w"))
    log("BATCH %s: %d objects, %d already on the server, urls -> %s (%.1f MB)" % (tag, len(objs), sum(1 for v in urls.values() if v.get("done")), p, os.path.getsize(p) / 1e6))


def _post_json(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.status, r.read()


def cmd_commit(tag, repo):
    import requests
    from huggingface_hub import HfApi
    tok = token()
    plan = json.load(open(os.path.join(OUT, "hfx_plan_%s.json" % tag)))
    urls = json.load(open(os.path.join(CACHE, "hfx_urls_%s.json" % tag)))["urls"]
    state = json.load(open(os.path.join(OUT, "hfx_state_%s.json" % tag)))
    ops = []
    for o in plan["objects"]:
        u = urls[o["sha256"]]
        if not u.get("done") and o["sha256"] not in state["done"]:
            raise SystemExit("object %s was not uploaded (state has %d done)" % (o["remote"], len(state["done"])))
        if u.get("mode") == "basic" and u.get("verify"):                      # small objects: the laptop does the authenticated verify call
            v = u["verify"]
            st, _ = _post_json(v["href"], {"oid": o["sha256"], "size": o["size"]}, {**LFS, **v.get("header", {})})
            assert st == 200, (o["remote"], st)
        ops.append({"key": "lfsFile", "value": {"path": o["remote"], "algo": "sha256", "oid": o["sha256"], "size": o["size"]}})
    man = json.dumps({"tag": tag, "repo": repo, "committed": time.strftime("%F %T"), "objects": [{k: v for k, v in o.items() if k != "rels"} | {"n_members": len(o["rels"])} | ({"rel": o["rels"][0]} if o["kind"] == "file" else {}) for o in plan["objects"]]}).encode()
    ops.append({"key": "file", "value": {"content": base64.b64encode(man).decode(), "path": "MANIFEST/hfx_%s.json" % tag, "encoding": "base64"}})
    for i in range(0, len(ops), 400):
        body = "\n".join(json.dumps(x) for x in [{"key": "header", "value": {"summary": "crag host backup %s (%d/%d)" % (tag, i // 400 + 1, (len(ops) + 399) // 400), "description": ""}}] + ops[i:i + 400]).encode()
        r = requests.post("https://huggingface.co/api/datasets/%s/commit/main" % repo, headers={"Authorization": "Bearer " + tok, "Content-Type": "application/x-ndjson"}, data=body, timeout=600)
        assert r.status_code == 200, (r.status_code, r.text[:500])
    log("COMMIT %s: %d lfs objects committed" % (tag, len(plan["objects"])))
    cmd_check(tag, repo)
    cmd_index(repo)


def cmd_check(tag, repo):
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    plan = json.load(open(os.path.join(OUT, "hfx_plan_%s.json" % tag)))
    bad = 0
    paths = [o["remote"] for o in plan["objects"]]
    got = {}
    for i in range(0, len(paths), 100):
        for x in api.get_paths_info(repo, paths[i:i + 100], repo_type="dataset", expand=True):
            got[x.path] = x
    for o in plan["objects"]:
        x = got.get(o["remote"])
        ok = x is not None and x.size == o["size"] and x.lfs is not None and x.lfs.sha256 == o["sha256"]
        bad += 0 if ok else 1
        if not ok:
            log("MISMATCH", o["remote"], getattr(x, "size", None), o["size"])
    log("CHECK %s: %d/%d objects present with the planned size + sha256 (%.2f GB)" % (tag, len(paths) - bad, len(paths), sum(o["size"] for o in plan["objects"]) / 1e9))
    assert bad == 0


# ---------------------------------------------------------------- host upload
def put(url, data):
    for k in range(7):
        try:
            req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Length": str(len(data)), "Content-Type": "application/octet-stream"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.headers.get("ETag") or r.headers.get("etag")
        except Exception as e:
            log("  put retry %d: %s" % (k + 1, str(e)[:120]))
            time.sleep(min(60, 2 ** k))
    raise RuntimeError("part upload failed")


def upload_file(path, u, threads):
    size = os.path.getsize(path)
    if u["mode"] == "basic":
        with open(path, "rb") as f:
            put(u["href"], f.read())
        return
    cs = u["chunk_size"]
    n = len(u["parts"])
    assert n == -(-size // cs), (n, size, cs)

    def part(i):
        with open(path, "rb") as f:
            f.seek(i * cs)
            return put(u["parts"][i], f.read(cs))
    with ThreadPoolExecutor(threads) as ex:
        etags = list(ex.map(part, range(n)))
    assert all(etags), "missing etag"
    st, body = _post_json(u["href"], {"oid": u["oid"], "parts": [{"partNumber": i + 1, "etag": e} for i, e in enumerate(etags)]}, LFS)
    assert st == 200, (st, body[:300])


def cmd_upload(tag, threads=8):
    plan = json.load(open(os.path.join(OUT, "hfx_plan_%s.json" % tag)))
    urls = json.load(open(os.path.join(CACHE, "hfx_urls_%s.json" % tag)))["urls"]
    sp = os.path.join(OUT, "hfx_state_%s.json" % tag)
    state = json.load(open(sp)) if os.path.exists(sp) else {"done": [], "failed": {}}
    pack = PACK
    os.makedirs(pack, exist_ok=True)
    tot = sum(o["size"] for o in plan["objects"])
    sent = 0
    for i, o in enumerate(plan["objects"]):
        u = dict(urls[o["sha256"]], oid=o["sha256"])
        if u.get("done") or o["sha256"] in state["done"]:
            sent += o["size"]
            continue
        t = time.time()
        try:
            if o["kind"] == "file":
                path = os.path.join(WS, o["rels"][0].replace("/", os.sep))
                assert os.path.getsize(path) == o["size"] and sha_file(path) == o["sha256"], "source changed since the plan"
            else:
                path = os.path.join(pack, "hfx_%s_%04d.tar" % (tag, i))
                s = Sink(path)
                write_tar(s, o["rels"])
                s.close()
                assert s.n == o["size"] and s.h.hexdigest() == o["sha256"], "shard differs from the plan (a source file changed)"
            upload_file(path, u, threads)
            state["done"].append(o["sha256"])
            sent += o["size"]
            log("[%3d/%d] %s %.2f GB uploaded in %.0fs (%.1f MB/s); %.1f / %.1f GB total" % (i + 1, len(plan["objects"]), o["remote"], o["size"] / 1e9, time.time() - t, o["size"] / 1e6 / max(1e-3, time.time() - t), sent / 1e9, tot / 1e9))
        except Exception as e:
            state["failed"][o["sha256"]] = "%s: %s" % (o["remote"], str(e)[:300])
            log("[%3d/%d] FAILED %s: %s" % (i + 1, len(plan["objects"]), o["remote"], str(e)[:300]))
        finally:
            if o["kind"] == "tar" and os.path.exists(path):
                os.remove(path)                                                  # our own scratch shard
            json.dump(state, open(sp, "w"))
    log("UPLOAD %s: %d/%d objects done, %d failed" % (tag, len(state["done"]), len(plan["objects"]), len(state["failed"])))


# ---------------------------------------------------------------- restore index + restore on another host
def _all_plans():
    out = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith("hfx_plan_") and f.endswith(".json"):
            pl = json.load(open(os.path.join(OUT, f)))
            if os.path.exists(os.path.join(OUT, "hfx_state_%s.json" % pl["tag"])):   # only plans that were uploaded; cmd_index keeps only objects really in the repo
                out.append(pl)
    return out


def known_list(repo):
    """'rel|size' of every file already in the repo (uploaded plans whose object really exists there) -> json list for --known"""
    from huggingface_hub import HfApi
    have = {x.path for x in HfApi(token=token()).list_repo_tree(repo, repo_type="dataset", recursive=True) if hasattr(x, "size")}
    k = set()
    for pl in _all_plans():
        for o in pl["objects"]:
            if o["remote"] not in have:
                continue
            msz = {} if o["kind"] == "file" else dict((m["rel"], m["size"]) for m in o["members"])
            for r in o["rels"]:
                k.add("%s|%d" % (r, o["size"] if o["kind"] == "file" else msz[r]))
    return sorted(k)


def cmd_index(repo):
    """one cumulative restore index in the repo: every committed object (remote, size, sha256, root, rel / members) -- what FETCH reads"""
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    have = {x.path for x in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if hasattr(x, "size")}
    objs = []
    for pl in _all_plans():
        for o in pl["objects"]:
            if o["remote"] not in have:
                continue
            e = {"remote": o["remote"], "group": o["group"], "kind": o["kind"], "root": o.get("root", "ws"), "size": o["size"], "sha256": o["sha256"], "tag": pl["tag"]}
            if o["kind"] == "file":
                e["rel"] = o["rels"][0]
            else:
                e["members"] = [[m["rel"], m["size"]] for m in o["members"]]
            objs.append(e)
    idx = {"repo": repo, "made": time.strftime("%F %T"), "n_objects": len(objs), "bytes": sum(o["size"] for o in objs),
           "roots": {"ws": "the rx workspace root (data/, results/, work/ ...)", "wsl": "/home/student2/crag_ooc on the host WSL ext4"}, "objects": objs}
    api.upload_file(path_or_fileobj=json.dumps(idx).encode(), path_in_repo="MANIFEST/RESTORE_INDEX.json", repo_id=repo, repo_type="dataset", commit_message="restore index (%d objects)" % len(objs))
    log("INDEX: %d objects, %.2f GB -> %s/MANIFEST/RESTORE_INDEX.json" % (len(objs), idx["bytes"] / 1e9, repo))


def cmd_known(out, repo):
    json.dump(known_list(repo), open(out, "w"))
    log("KNOWN: %d files already backed up -> %s" % (len(json.load(open(out))), out))


def cmd_geturls(repo, out, groups=""):
    """laptop: short-lived signed GET urls for the repo objects (so a shared host needs no token to restore)"""
    import requests
    from huggingface_hub import hf_hub_download
    tok = token()
    idx = json.load(open(hf_hub_download(repo, "MANIFEST/RESTORE_INDEX.json", repo_type="dataset", token=tok, force_download=True)))
    want = set(g for g in groups.split(",") if g)
    res = []
    for o in idx["objects"]:
        if want and o["group"] not in want:
            continue
        r = requests.get("https://huggingface.co/datasets/%s/resolve/main/%s" % (repo, o["remote"]), headers={"Authorization": "Bearer " + tok}, allow_redirects=False, timeout=60)
        assert r.status_code in (301, 302, 307), (o["remote"], r.status_code)
        res.append(dict(o, url=r.headers["Location"]))
    json.dump({"repo": repo, "made": time.time(), "objects": res}, open(out, "w"))
    log("GETURLS: %d objects, %.2f GB -> %s (signed urls expire: fetch soon after)" % (len(res), sum(o["size"] for o in res) / 1e9, out))


def _download(url, dest, size, headers=None):
    """resumable ranged download (stdlib)"""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    while have < size:
        req = urllib.request.Request(url, headers=dict(headers or {}, Range="bytes=%d-" % have))
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
        except Exception as e:
            log("  download retry at %.2f GB: %s" % (have / 1e9, str(e)[:120]))
            time.sleep(5)
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
    return have


def cmd_fetch(src, ws_root, wsl_root, groups=""):
    """new host: download (urls file from GETURLS, or `src` = repo id + $HF_TOKEN on a host you own), verify sha256, extract into the roots, verify member sizes; resumable"""
    hdr = None
    if os.path.exists(src):
        d = json.load(open(src))
    else:
        tok = os.environ["HF_TOKEN"]
        hdr = {"Authorization": "Bearer " + tok}
        idx = json.load(urllib.request.urlopen(urllib.request.Request("https://huggingface.co/datasets/%s/resolve/main/MANIFEST/RESTORE_INDEX.json" % src, headers=hdr)))
        d = {"repo": src, "objects": [dict(o, url="https://huggingface.co/datasets/%s/resolve/main/%s" % (src, o["remote"])) for o in idx["objects"]]}
    roots = {"ws": ws_root, "wsl": wsl_root}
    want = set(g for g in groups.split(",") if g)
    dl = os.path.join(ws_root, "work", "HOST_HOUSEKEEPING", "restore_dl")
    os.makedirs(dl, exist_ok=True)
    stp = os.path.join(dl, "restore_state.json")
    done = set(json.load(open(stp))) if os.path.exists(stp) else set()
    for i, o in enumerate(d["objects"]):
        if (want and o["group"] not in want) or o["sha256"] in done:
            continue
        root = roots[o["root"]]
        path = os.path.join(dl, o["sha256"] + ".dl")
        t = time.time()
        _download(o["url"], path, o["size"], hdr)
        assert os.path.getsize(path) == o["size"], (o["remote"], "size")
        assert sha_file(path) == o["sha256"], (o["remote"], "sha256 mismatch")
        if o["kind"] == "file":
            dst = os.path.join(root, o["rel"].replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(path, dst)
            assert os.path.getsize(dst) == o["size"]
        else:
            with tarfile.open(path, "r:") as tf:
                for m in tf:
                    assert not os.path.isabs(m.name) and ".." not in m.name.split("/"), m.name
                    tf.extract(m, root)
            for r, s in o["members"]:
                assert os.path.getsize(os.path.join(root, r.replace("/", os.sep))) == s, (r, "member size")
            os.remove(path)
        done.add(o["sha256"])
        json.dump(sorted(done), open(stp, "w"))
        log("[%d/%d] %s %.2f GB restored in %.0fs" % (i + 1, len(d["objects"]), o["remote"], o["size"] / 1e9, time.time() - t))
    log("FETCH done: %d objects restored" % len(done))


if __name__ == "__main__":
    for _o, _name in (("--settle", "SETTLE_S"), ("--known", "KNOWN")):          # options usable on any command line (rx jobs pass no environment): --settle <s>  --known <json>
        if _o in sys.argv:
            _i = sys.argv.index(_o)
            globals()[_name] = float(sys.argv[_i + 1]) if _name == "SETTLE_S" else sys.argv[_i + 1]
            del sys.argv[_i:_i + 2]
    c = sys.argv[1]
    if c == "PLAN":
        cmd_plan(sys.argv[2], sys.argv[3:])
    elif c == "BATCH":
        cmd_batch(sys.argv[2], sys.argv[3])
    elif c == "UPLOAD":
        cmd_upload(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    elif c == "COMMIT":
        cmd_commit(sys.argv[2], sys.argv[3])
    elif c == "CHECK":
        cmd_check(sys.argv[2], sys.argv[3])
    elif c == "INDEX":
        cmd_index(sys.argv[2])
    elif c == "KNOWN":
        cmd_known(sys.argv[2], sys.argv[3])
    elif c == "GETURLS":
        cmd_geturls(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    elif c == "FETCH":                                                       # FETCH <urls.json | repo_id> <ws_root> <wsl_root> [groups,csv]
        cmd_fetch(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "")
