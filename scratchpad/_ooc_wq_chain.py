"""FBX_SCALE addendum 17 -- the WebQSP regression chain of the out-of-core engine (E1..E6) in ONE host job (one binary compile, sequential): twins / quotients / SPAN passes (re-run on the current binary),
the split-preserve top and its level quotients + the cap-4 surrogate, then the surrogate shards, the Zoltan top and the V-cycle against the stored records.  Every test asserts bit identity.

  python -u scratchpad/_ooc_wq_chain.py [E1 | TOP | VC]...      # default: all three
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _ooc_e1 as E  # noqa: E402
import _ooc_vc as V  # noqa: E402


def main():
    a = sys.argv[1:] or ["E1", "TOP", "VC"]
    if "E1" in a:
        E.test_webqsp()
    if "TOP" in a:
        E.test_webqsp_top()
    if "VC" in a:
        V.test_webqsp_vc()
    print("WQ_CHAIN PASS: " + " ".join(a), flush=True)


if __name__ == "__main__":
    main()
