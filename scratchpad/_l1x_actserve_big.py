"""L1X step 11b: the same routed ACT/POOL study as _l1x_actserve.py (imported unchanged, so its sha256 pins the logic) with the larger hit budgets that step 11 left open.
ACT 200 / POOL 5000 stays first (the identity run); then ACT 6000 and 10000.   python -u scratchpad/_l1x_actserve_big.py RUN metaqa v2"""
import sys
import _l1x_actserve as AS

AS.CONFIGS = [(200, 5000), (3000, 5000), (6000, 5000), (6000, 10000), (10000, 5000), (10000, 20000)]
if __name__ == "__main__":
    assert sys.argv[1] == "RUN", __doc__
    AS.run()
