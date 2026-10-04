"""Back up the CRAG working tree's CODE + RECORDS (every non-ignored text/record file <= 3 MB: src/, scratchpad/, docs/, results/ records, rx.toml, ...) as one tar.gz into the private HF backup repo, from the laptop (token never
leaves it), so a new host / a new laptop can be set up from HF alone.  Keeps the 3 newest snapshots under CODE/.
   python -u scratchpad/_code_snapshot.py [--repo Swastik9895/crag-host-backup] [--dry]"""
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = (".py", ".cpp", ".c", ".h", ".sh", ".toml", ".md", ".json", ".txt", ".csv", ".yaml", ".yml", ".jsonl", ".cfg")
MAXB = 3 << 20
KEEP = 3


def main():
    repo = sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else "Swastik9895/crag-host-backup"
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, stdout=subprocess.PIPE, text=True).stdout.splitlines()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tmp = os.path.join(tempfile.gettempdir(), "crag_code_%s.tar.gz" % stamp)
    n = tot = 0
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tf:
        for f in files:
            p = os.path.join(ROOT, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if not os.path.isfile(p) or st.st_size > MAXB or not f.lower().endswith(EXT):
                continue
            tf.add(p, arcname=f, recursive=False)
            n += 1
            tot += st.st_size
        info = {"made": stamp, "files": n, "raw_bytes": tot, "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, stdout=subprocess.PIPE, text=True).stdout.strip()}
        b = json.dumps(info).encode()
        ti = tarfile.TarInfo("CODE_SNAPSHOT_INFO.json")
        ti.size = len(b)
        tf.addfile(ti, io.BytesIO(b))
    print("snapshot %d files, %.1f MB raw -> %.1f MB (%s)" % (n, tot / 1e6, os.path.getsize(tmp) / 1e6, tmp), flush=True)
    if "--dry" in sys.argv:
        return
    from huggingface_hub import HfApi
    tok = json.load(open(os.path.join(os.path.expanduser("~"), "hf_tokens.json")))["accounts"][0]["token"]
    api = HfApi(token=tok)
    api.upload_file(path_or_fileobj=tmp, path_in_repo="CODE/crag_code_%s.tar.gz" % stamp, repo_id=repo, repo_type="dataset", commit_message="code snapshot %s" % stamp)
    got = list(api.get_paths_info(repo, ["CODE/crag_code_%s.tar.gz" % stamp], repo_type="dataset", expand=True))[0]
    assert got.size == os.path.getsize(tmp), (got.size, os.path.getsize(tmp))
    print("uploaded CODE/crag_code_%s.tar.gz (%d bytes, verified size)" % (stamp, got.size), flush=True)
    old = sorted(x.path for x in api.list_repo_tree(repo, repo_type="dataset", path_in_repo="CODE") if x.path.endswith(".tar.gz"))
    for p in old[:-KEEP]:
        api.delete_file(p, repo_id=repo, repo_type="dataset", commit_message="prune old code snapshot")
        print("pruned", p, flush=True)
    os.remove(tmp)


main()
