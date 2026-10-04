"""Host housekeeping -- back up host-only crag data to a Hugging Face DATASET repo from the lab host, resumable and verified.  Read-only on the sources; its only writes are one scratch tar at a time under
work/HOST_HOUSEKEEPING/pack/ (removed after its upload verifies) and a record under results/HOST_HOUSEKEEPING/.

  HF_TOKEN=<write token> python -u scratchpad/_host_hf_backup.py <GROUP> <repo_id> [--plan] [--dry] [--shard-gb 2]

GROUP -> source directory (relative to the rx workspace); files larger than --big-gb are uploaded as themselves, everything else is packed (uncompressed tar, member paths = workspace-relative) into shards of ~shard-gb:
  fbs_names  data/freebase_scale/names          fbs_ner  data/freebase_scale/ner          fbs_pq  data/freebase_scale/pq        fbs_ivft data/freebase_scale/ivf_transfer
  fbs_enc    data/freebase_scale/enc (only after the encode job ended)                      calib   work/FBX_CALIB/bundle_v1
  l1parts    results/L1_HOST/parts              r_ml2  results/FREEBASE_SCALE/ml2          r_h2l  results/FREEBASE_SCALE/h2l    r_ivf results/FREEBASE_SCALE/ivf   r_partsS results/FREEBASE_SCALE/parts_S
--plan prints the shard plan and exits; --dry packs + hashes but uploads nothing.  A shard already in the repo with the same size + LFS sha256 is skipped (resume).  The token is read from the environment only and never
printed or written.  Verification: the repo's LFS sha256 + size for every uploaded path must equal the sha256 computed while packing; a record with the member manifest is written and also uploaded.
"""
import hashlib
import io
import json
import os
import sys
import tarfile
import time

WS = os.getcwd()
GROUPS = {
    "fbs_names": "data/freebase_scale/names", "fbs_ner": "data/freebase_scale/ner", "fbs_pq": "data/freebase_scale/pq", "fbs_ivft": "data/freebase_scale/ivf_transfer", "fbs_enc": "data/freebase_scale/enc",
    "calib": "work/FBX_CALIB/bundle_v1", "l1parts": "results/L1_HOST/parts", "r_ml2": "results/FREEBASE_SCALE/ml2", "r_h2l": "results/FREEBASE_SCALE/h2l", "r_ivf": "results/FREEBASE_SCALE/ivf",
    "r_partsS": "results/FREEBASE_SCALE/parts_S",
}
PACK = os.path.join(WS, "work", "HOST_HOUSEKEEPING", "pack")
RECD = os.path.join(WS, "results", "HOST_HOUSEKEEPING")


def opt(name, default):
    a = sys.argv
    return type(default)(a[a.index(name) + 1]) if name in a else default


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def plan(group, shard_b, big_b):
    root = os.path.join(WS, GROUPS[group].replace("/", os.sep))
    files = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            if f.endswith((".part", ".rxpart")):
                continue
            files.append((os.path.relpath(p, WS).replace("\\", "/"), os.path.getsize(p)))
    files.sort()
    direct = [(r, s) for r, s in files if s > big_b]
    small = [(r, s) for r, s in files if s <= big_b]
    shards, cur, cb = [], [], 0
    for r, s in small:
        if cur and cb + s > shard_b:
            shards.append(cur)
            cur, cb = [], 0
        cur.append((r, s))
        cb += s
    if cur:
        shards.append(cur)
    return direct, shards, len(files), sum(s for _, s in files)


def main():
    group, repo = sys.argv[1], sys.argv[2]
    shard_b = int(opt("--shard-gb", 2.0) * 1e9)
    big_b = int(opt("--big-gb", 3.0) * 1e9)
    direct, shards, nf, nb = plan(group, shard_b, big_b)
    print("group %s: %d files, %.2f GB -> %d direct files + %d tar shards of ~%.1f GB" % (group, nf, nb / 1e9, len(direct), len(shards), shard_b / 1e9), flush=True)
    if "--plan" in sys.argv:
        return
    dry = "--dry" in sys.argv
    api = None
    have = {}
    if not dry:
        from huggingface_hub import HfApi
        tok = os.environ["HF_TOKEN"]
        api = HfApi(token=tok)
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        for it in api.list_repo_tree(repo, repo_type="dataset", path_in_repo=group, recursive=True, expand=True):
            if getattr(it, "lfs", None):
                have[it.path] = (it.size, it.lfs.sha256)
    os.makedirs(PACK, exist_ok=True)
    os.makedirs(RECD, exist_ok=True)
    rec = {"group": group, "repo": repo, "source": GROUPS[group], "files": nf, "bytes": nb, "items": [], "t0": time.strftime("%F %T")}
    items = [("direct", [r]) for r, _ in direct] + [("tar", [r for r, _ in sh]) for sh in shards]
    for i, (kind, rels) in enumerate(items):
        t = time.time()
        if kind == "direct":
            src = os.path.join(WS, rels[0].replace("/", os.sep))
            remote = "%s/%s" % (group, os.path.basename(src))
            sha = sha_file(src)
            size = os.path.getsize(src)
            members = [{"rel": rels[0], "size": size, "sha": sha}]
            tmp = None
        else:
            remote = "%s/%s_%04d.tar" % (group, group, i)
            tmp = os.path.join(PACK, "%s_%04d.tar" % (group, i))
            members = []
            with tarfile.open(tmp, "w") as tf:
                for r in rels:
                    p = os.path.join(WS, r.replace("/", os.sep))
                    members.append({"rel": r, "size": os.path.getsize(p), "sha": sha_file(p)})
                    tf.add(p, arcname=r, recursive=False)
            src, sha, size = tmp, sha_file(tmp), os.path.getsize(tmp)
        if not dry and remote in have and have[remote] == (size, sha):
            print("[%3d/%d] %s already in the repo (size+sha match), skipped" % (i + 1, len(items), remote), flush=True)
        elif not dry:
            api.upload_file(path_or_fileobj=src, path_in_repo=remote, repo_id=repo, repo_type="dataset", commit_message="crag host backup %s" % remote)
            got = list(api.get_paths_info(repo, [remote], repo_type="dataset", expand=True))[0]
            ok = got.lfs is not None and got.lfs.sha256 == sha and got.size == size
            assert ok, "verification failed for %s: repo %s/%s vs local %s/%s" % (remote, got.size, getattr(got.lfs, "sha256", None), size, sha)
            print("[%3d/%d] %s %.2f GB uploaded + sha256 verified in %.0fs" % (i + 1, len(items), remote, size / 1e9, time.time() - t), flush=True)
        else:
            print("[%3d/%d] %s %.2f GB packed + hashed (dry)" % (i + 1, len(items), remote, size / 1e9), flush=True)
        rec["items"].append({"remote": remote, "size": size, "sha256": sha, "members": members})
        if tmp:
            os.remove(tmp)                                            # our own scratch shard
    rec["t1"] = time.strftime("%F %T")
    rec["verified"] = not dry
    rp = os.path.join(RECD, "HF_BACKUP__%s.json" % group)
    json.dump(rec, open(rp, "w"))
    if not dry:
        api.upload_file(path_or_fileobj=rp, path_in_repo="MANIFEST/HF_BACKUP__%s.json" % group, repo_id=repo, repo_type="dataset", commit_message="manifest %s" % group)
    print("DONE %s: %d items, record %s" % (group, len(items), rp), flush=True)


main()
