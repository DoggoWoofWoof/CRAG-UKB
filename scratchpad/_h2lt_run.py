"""Orchestration only (no science): one addendum-9 arm end to end = the surrogate top partition, then the nested K-way maps.  Every step is resumable (a finished record is skipped).

  python -u scratchpad/_h2lt_run.py <spec> <S> <K> [<K> ...]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "_h2lt.py")

if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 3:
        raise SystemExit(__doc__)
    spec, S, Ks = a[0], a[1], a[2:]
    subprocess.check_call([sys.executable, "-u", T, "TOP", spec, S])
    subprocess.check_call([sys.executable, "-u", T, "SUB", spec, S] + Ks)
