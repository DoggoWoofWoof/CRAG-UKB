"""Orchestration only (no science): the nested K-way stage of the addendum-14 V-cycle arms for ONE K, arm after arm, in ONE process chain.

_h2l.cmd_sub names its Zoltan run directory and shard directory 'H2L<S>_k<K>_b<block>' WITHOUT the arm, so two arms must never run the K-way stage for the same K at the same time.  One job per K runs its
arms in sequence; jobs of different K do not collide.  The job never waits: it runs the arms whose V-cycle top record (RUN or FAILED) already exists and exits, listing the arms it skipped (resubmit later).

  python -u scratchpad/_vc_subk.py <K> <S> <arm> [<arm> ...]          # arm = VC<spec><REFINER>, e.g. VCQ512L6N4FM2
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
T = os.path.join(HERE, "_vc_sub.py")
PARTS = os.path.join(REPO, "results", "FREEBASE_SCALE", "h2l", "parts")


def top_ready(arm, S):
    base = os.path.join(PARTS, "webqsp__H4_H2LT_%s_top_k%d__PHG_con" % (arm, S))
    return os.path.exists(base + ".RUN.json") or os.path.exists(base + ".FAILED.json")


def main():
    K, S, arms = int(sys.argv[1]), int(sys.argv[2]), list(sys.argv[3:])
    ready = [s for s in arms if top_ready(s, S)]
    skipped = [s for s in arms if s not in ready]
    print("[vc-subk K %d] ready %s; top not finished yet (skipped, resubmit later) %s" % (K, ready, skipped), flush=True)
    for arm in ready:
        print("[vc-subk K %d] arm %s: running the K-way stage" % (K, arm), flush=True)
        subprocess.check_call([sys.executable, "-u", T, "SUB", arm, str(S), str(K)])
    print("[vc-subk K %d] done: ran %s, skipped %s" % (K, ready, skipped), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main()
