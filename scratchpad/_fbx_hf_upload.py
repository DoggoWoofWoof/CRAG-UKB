"""Upload the staged Freebase tree to a PRIVATE Hugging Face dataset repo (FREEBASE_SCALE lane transfer, user 2026-09-30).

Uses the token the user stored with `hf auth login` (this script never reads or prints it).  Refuses to touch a repo that is not private.
Modes:
  probe  <user/repo>   create the private repo if missing, upload a 100 MB random file, print MB/s, delete it   (measures the real uplink)
  upload <user/repo>   upload_large_folder of data/_cache/fbx_stage (resumable: re-run the same command after any interruption)
  status <user/repo>   files in the repo vs the stage (name + size)
Run scratchpad/_fbx_stage.py first.
"""
import os
import sys
import time

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")          # use all cores / the full uplink
from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.errors import HfHubHTTPError  # noqa: E402

STAGE = "C:/Users/Swastik/Desktop/CRAG/data/_cache/fbx_stage"
TMP = "C:/Users/Swastik/Desktop/CRAG/data/_cache/fbx_probe_100MB.bin"


def private_repo(api, repo):
    try:
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    except HfHubHTTPError as e:                                   # a token scoped to one existing repo cannot create repos: the repo must pre-exist
        if e.response is None or e.response.status_code not in (401, 403):
            raise
        print("token cannot create repos; using the existing repo", flush=True)
    assert api.repo_info(repo, repo_type="dataset").private is True, "REFUSING: %s is not private" % repo


def main():
    mode, repo = sys.argv[1], sys.argv[2]
    api = HfApi()
    try:
        print("logged in as:", api.whoami()["name"], flush=True)
    except HfHubHTTPError:
        print("logged in (whoami not permitted for this token)", flush=True)
    private_repo(api, repo)
    if mode == "probe":
        os.makedirs(os.path.dirname(TMP), exist_ok=True)
        with open(TMP, "wb") as f:
            for _ in range(100):
                f.write(os.urandom(1 << 20))
        t0 = time.time()
        api.upload_file(path_or_fileobj=TMP, path_in_repo="_probe/probe_100MB.bin", repo_id=repo, repo_type="dataset")
        dt = time.time() - t0
        print("uplink probe: 100 MB in %.1f s = %.2f MB/s  -> 79.2 GB would take about %.1f h" % (dt, 100 / dt, 79200 / (100 / dt) / 3600), flush=True)
        api.delete_file("_probe/probe_100MB.bin", repo_id=repo, repo_type="dataset")
        os.remove(TMP)
    elif mode == "upload":
        assert os.path.isdir(STAGE) and os.listdir(STAGE), "run scratchpad/_fbx_stage.py first"
        t0 = time.time()
        api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=STAGE, num_workers=8, print_report=True, print_report_every=60)
        print("upload finished in %.0f s" % (time.time() - t0), flush=True)
    elif mode == "status":
        have = {e.path: e.size for e in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if hasattr(e, "size")}
        want = {}
        for r, ds, fs in os.walk(STAGE):
            if ".cache" in r.replace("\\", "/").split("/"):
                continue
            for f in fs:
                p = os.path.join(r, f)
                want[os.path.relpath(p, STAGE).replace("\\", "/")] = os.path.getsize(p)
        ok = sum(1 for k, v in want.items() if have.get(k) == v)
        print("repo has %d of %d staged files with the right size (%.3f GB of %.3f GB)" % (
            ok, len(want), sum(v for k, v in want.items() if have.get(k) == v) / 1e9, sum(want.values()) / 1e9))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
