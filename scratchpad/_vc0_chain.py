"""Level-0 V-cycle stage job (FBX_SCALE addendum 19): ONE host job that (1) re-runs the WebQSP end-to-end regression of the patched engine (E4 surrogate + Zoltan top + E6 V-cycle against the stored records, byte for byte) and ONLY IF
it passes (2) runs the Freebase level-0 refinement under the measured anonymous-memory cap.  A regression failure therefore blocks the heavy stage (addendum 17 stop rule); nothing is written to the Freebase work dir before it passes.
  python -u scratchpad/_vc0_chain.py [anon_gb=50]"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FB = "/home/student2/crag_ooc"


def run(cmd):
    print(time.strftime("[chain %H:%M:%S]"), " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        print(time.strftime("[chain %H:%M:%S]"), "FAILED rc=%d: %s" % (r.returncode, cmd[2]), flush=True)
        sys.exit(r.returncode)


def main():
    anon = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("anon_gb=")), "50")
    run([sys.executable, "-u", "scratchpad/_ooc_wq_chain.py", "VC"])
    run([sys.executable, "-u", "scratchpad/_ooc_vc.py", "VCYCLE", FB + "/fb", FB + "/fb_l1", FB + "/fb_work", "4", "levels=0-0", "anon_gb=" + anon])
    print(time.strftime("[chain %H:%M:%S]"), "VC0_CHAIN PASS", flush=True)


if __name__ == "__main__":
    main()
