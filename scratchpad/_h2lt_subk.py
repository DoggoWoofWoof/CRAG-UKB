"""Orchestration only (no science): the nested K-way stage of the addendum-9 arms for ONE K, arm after arm, in ONE process chain.

_h2l.cmd_sub names its Zoltan run directory and shard directory 'H2L<S>_k<K>_b<block>' WITHOUT the arm, so two arms must never run the K-way stage for the same K at the same time (the run
directory is deleted and recreated per block).  One job per K runs its arms in sequence; jobs of different K do not collide.

The job never waits: it runs the arms whose top partition record (RUN or FAILED) already exists and exits, listing the arms it skipped.  A job that waited would hold its CPU and memory
reservation while idle and could block the very top job it is waiting for.  Submit it again when more tops are finished; a finished arm record is skipped by _h2lt.py itself.

  python -u scratchpad/_h2lt_subk.py <K> <S> <spec> [<spec> ...]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
T = os.path.join(HERE, "_h2lt.py")
PARTS = os.path.join(REPO, "results", "FREEBASE_SCALE", "h2l", "parts")


def top_ready(spec, S):
    base = os.path.join(PARTS, "webqsp__H4_H2LT_%s_top_k%d__PHG_con" % (spec, S))
    return os.path.exists(base + ".RUN.json") or os.path.exists(base + ".FAILED.json")


def main():
    K, S, specs = int(sys.argv[1]), int(sys.argv[2]), list(sys.argv[3:])
    ready = [s for s in specs if top_ready(s, S)]
    skipped = [s for s in specs if s not in ready]
    print("[subk K %d] ready %s; top not finished yet (skipped, resubmit later) %s" % (K, ready, skipped), flush=True)
    for spec in ready:
        print("[subk K %d] arm %s: running the K-way stage" % (K, spec), flush=True)
        subprocess.check_call([sys.executable, "-u", T, "SUB", spec, str(S), str(K)])
    print("[subk K %d] done: ran %s, skipped %s" % (K, ready, skipped), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main()
