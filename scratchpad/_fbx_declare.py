"""Write the FREEBASE_SCALE host-stage declaration (write-once), pinning the transfer code by sha256.

The Freebase served tree (79.2 GB, 120 files) moves to the host through a private Hugging Face dataset repo because the laptop->host
path is a DERP relay (~0.87 MB/s).  This records what moves, how, and what is verified; the tree itself is verified against the frozen
CANONICAL_FREEZE.json by verify_freebase --full on the host (every pinned byte hashed), so no transfer-side hash is trusted.

Usage: python scratchpad/_fbx_declare.py            (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
OUT = "results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_HOST__v1.json"
CODE = ["scratchpad/_fbx_stage.py", "scratchpad/_fbx_hf_upload.py", "scratchpad/_fbx_hf_urls.py", "scratchpad/_fbx_hf_pull.py",
        "scratchpad/_fbx_verify_host.py", "scratchpad/_host_yield.py",
        "src/dataset_canonical/freeze_canonical.py", "src/dataset_canonical/verify_canonical.py", "src/dataset_canonical/freebase/verify_freebase.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    fr = json.load(io.open("data/final_canonical/CANONICAL_FREEZE.json", encoding="utf-8"))
    fb = fr.get("FREEBASE", {})
    rec = {
        "stage": "FBX_HOST",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S+05:30", time.localtime()),
        "what": "the Freebase served tree (data/final_canonical/freebase, 120 files, 79,223,455,868 bytes) is copied to the shared host 'gpu' so the FREEBASE_SCALE lane can run there; the frozen tree on the laptop is not modified",
        "why": [
            "the laptop->host path is a DERP relay (tailscale ping: no direct connection; rx doctor --speed 50 measured 868 KB/s up), about 25 hours for 79.2 GB",
            "an outbound Hugging Face upload from the laptop ran at 4.8-5.2 MB/s and the host downloads from the HF CDN, so the same bytes move in about 5 hours and never cross the relay"
        ],
        "user_direction (verbatim, chat, 2026-09-30)": [
            "can we somehow do a fast transfer of freebase and do everything on the host by uploading to some cloud platform or even modal volume or something and use that as a intermediary and download? so that we arent bottlenecked by tailscale ssh?",
            "i give you permission to login and do whatever you need to",
            "gpu should be free if you need (2026-09-29)",
            "you have the second highest priority for cpu and gpu you should never wait unless mpr and finish everything as fast as possible (2026-09-29)"
        ],
        "user_choices (AskUserQuestion)": {"intermediary": "Hugging Face private dataset", "scope": "Full served tree, 79.2 GB"},
        "intermediary": {
            "repo": "Swastik9895/crag-freebase-tree (dataset, PRIVATE; the upload script refuses a repo that is not private)",
            "laptop_credential": "the token the user stored with `hf auth login`; scripts never read it into a file or print it; a token pasted in chat was NOT used",
            "host_credential": "NONE. The host holds no token: the laptop resolves per-file time-limited signed CDN links (Authorization is sent only to huggingface.co, never to the CDN) and the host pulls them with the standard library",
            "after": "the user decides whether the private repo is deleted; the token pasted in chat should be revoked"
        },
        "host": {"name": "gpu", "workspace": "C:/Users/Student2/rx/projects/crag/ws",
                 "pull_destination": "C:/Users/Student2/rx/projects/crag/data/freebase (outside the rx workspace, like the encoder cache under .../crag/models/hf; C: had 482 GB free)",
                 "workspace_link": "directory junction ws/data/final_canonical/freebase -> the pull destination, so the frozen relative paths resolve unchanged",
                 "env": "mpr-cpu (read-only by name): Python 3.13.5, numpy 2.3.2, pyarrow 23.0.1 (checked 2026-09-30 01:29, job 260930-012832-fbx-envcheck-e2cd)"},
        "priority_policy": {"wrapper": "every host job runs as `python -u scratchpad/_host_yield.py run -- <cmd>` (mpr > crag > jigsaw)",
                            "reservation": "the scheduler's 2 CPU / 8 GB reserve is untouched",
                            "never": ["cancel, rerun or gc another project's job", "touch another project's processes", "edit mpr's [envs.*]", "rx gc",
                                      "put a credential or token on the host (no --var secrets)"]},
        "integrity": {"rule": "no transfer-side checksum is trusted: scratchpad/_fbx_verify_host.py runs verify_freebase(freeze, full=True) on the host, which hashes every pinned byte against CANONICAL_FREEZE.json/DATASET.json and fails on any file the tree does not pin",
                      "record": "results/FREEBASE_SCALE/FBX_HOST_VERIFY__v1.json (write-once; the frozen tree's VERIFICATION.json is not touched)",
                      "freeze_freebase_block": {k: fb.get(k) for k in ("n_nodes", "n_edges", "n_relations", "tree_bytes") if k in fb}},
        "byte_accounting": "moved 79,223,455,868 bytes = the freeze's tree_bytes 79,223,377,787 + DATASET.json 62,278 + VERIFICATION.json 15,803 (the two records the freeze does not count)",
        "upload_measured": "the laptop upload finished 2026-09-30 in 10,134 s (about 7.8 MB/s average) and the repo lists 120 of 120 staged files with the exact sizes",
        "repo_extra_file": "the Hugging Face repo also carries .gitattributes (2,504 bytes, created by the hub, not part of the tree): the manifest lists 121 entries; the pull writes it, and it is deleted on the host before verification because verify_freebase fails on any unpinned file",
        "pull_test": "scratchpad/_fbx_hf_pull.py was tried on the laptop against relations/relations.parquet and metadata/label_residue.parquet: both came back byte-identical (sha256) to the frozen copies",
        "not_moved": ["freebase_v3 and any raw dump", "any other dataset", "any token or credential"],
        "refusals_kept": ["TEST is never read", "split B stays sealed; held-out rows not read", "nothing under data/final_canonical is written on the laptop",
                          "on the host the pulled tree is only read after the pull", "no encoder training, no LLM"],
        "gate": "no FREEBASE_SCALE compute on the host until FBX_HOST_VERIFY__v1.json reports PASS with 0 failed checks; a FAIL re-pulls only the files named in it",
        "code": {p: sha(p) for p in CODE},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
