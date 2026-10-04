"""FBX_SCALE Track A: run the post-extract stages of the Freebase NER family back to back (GROUP -> AGG -> FINALIZE) in ONE host job, so no stage waits for a human to submit the next.
Every stage is resumable on its own records (a killed bucket / range is redone, finished ones are skipped) and FINALIZE writes the write-once family record LAST.  No recipe code lives here: it only calls
scratchpad/_fbx_ner_build.py's own cmd_group / cmd_agg / cmd_finalize.

  python -u scratchpad/_fbx_ner_chain.py [GROUP] [AGG] [FINALIZE]       # default: all three, in this order
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fbx_ner_build as B

if __name__ == "__main__":
    stages = [a for a in sys.argv[1:] if a in ("GROUP", "AGG", "FINALIZE")] or ["GROUP", "AGG", "FINALIZE"]
    t0 = time.time()
    for s in stages:
        t = time.time()
        print("== %s start" % s, flush=True)
        {"GROUP": B.cmd_group, "AGG": B.cmd_agg, "FINALIZE": B.cmd_finalize}[s]()
        print("== %s done %.0fs (total %.0fs)" % (s, time.time() - t, time.time() - t0), flush=True)
