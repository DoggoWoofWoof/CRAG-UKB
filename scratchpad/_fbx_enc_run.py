"""Orchestration only: run one _fbx_encode.py command and retry it when it dies of a transient GPU fault.

The GPU is shared with other projects' jobs; a device-wide CUDA fault (seen 2026-10-01 05:18: 'illegal memory access', three unrelated jobs died within seconds) kills the encode process.  CHUNKS is
resumable (a chunk whose record + checksum verify is skipped), so a retry costs at most the chunk in flight; TRAIN restarts its sample encode.  A deterministic failure is not retried forever.

  python -u scratchpad/_fbx_enc_run.py <MAX_TRIES> <_fbx_encode.py args ...>
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENC = os.path.join(HERE, "_fbx_encode.py")


def main():
    tries, args = int(sys.argv[1]), sys.argv[2:]
    rc = 1
    for i in range(1, tries + 1):
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-u", ENC] + args)
        print("[enc_run] try %d/%d rc %d after %.0f s" % (i, tries, rc, time.time() - t0), flush=True)
        if rc == 0:
            return 0
        time.sleep(60)
    return rc


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.exit(main())
