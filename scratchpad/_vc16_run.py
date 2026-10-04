"""Orchestration only (no science): the TOP stage of the addendum-16 arms, one spec after the other in ONE process chain, each step resumable (a step whose record exists is skipped by the step itself).

For every spec:  _h2lt.py TOP <spec> 25   (Zoltan on the surrogate hypergraph; label-free)
                 _vc.py D1  <spec> fm2     (the V-cycle development record; KM1 only)
                 _vc.py TOP <spec> fm2     (the V-cycle top as a TOP record of the nested pipeline; refuses a map whose KM1 differs from the D1 record)
It refuses to start unless addendum 16 exists (the declaration precedes the first top).  No gold label is read.

  python -u scratchpad/_vc16_run.py TOPS <spec> [<spec> ...]          # spec = Q512L<l>N<c>, e.g. Q512L5N4
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ADD16 = os.path.join(REPO, "results", "FREEBASE_SCALE", "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_16.json")


def run(*a):
    print("[vc16-run] " + " ".join(a), flush=True)
    subprocess.check_call([sys.executable, "-u"] + list(a), cwd=REPO)


def main():
    assert os.path.exists(ADD16), "addendum 16 must be declared before any top is partitioned"
    for spec in sys.argv[2:]:
        run(os.path.join("scratchpad", "_h2lt.py"), "TOP", spec, "25")
        run(os.path.join("scratchpad", "_vc.py"), "D1", spec, "fm2")
        run(os.path.join("scratchpad", "_vc.py"), "TOP", spec, "fm2")
    print("[vc16-run] done: %s" % sys.argv[2:], flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "TOPS":
        raise SystemExit(__doc__)
    main()
