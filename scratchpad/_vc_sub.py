"""FBX_SCALE Track B, addendum 14 -- the nested K-way maps whose S-way TOP level is a V-cycle top (scratchpad/_vc.py TOP).  The sub-stage is addendum 7's, unchanged: this module only sets the arm name that
scratchpad/_h2lt.py would parse from its command line (its spec grammar is Q<W>L<l>N<c>, pinned by addendum 9) and calls the same _h2l.cmd_sub.

  python -u scratchpad/_vc_sub.py SUB <arm> <S> <K> [<K> ...]      # arm = VC<spec><REFINER>, e.g. VCQ512L6N4FM2
"""
import json
import os
import sys
import time

import _h2lt as T

H, L = T.H, T.L

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) >= 4 and a[0] == "SUB":
        name, S = a[1], int(a[2])
        T.CUR["spec"] = {"name": name, "W": None, "level": None, "cap": None}
        L.MODE["top"] = True
        tfail = H.phg_npy(T.DS, S)[:-4] + ".FAILED.json"
        L.MODE["top"] = False
        for K in a[3:]:
            if os.path.exists(tfail):                 # the top was refused by the frozen validity rule: no map at any K (recorded, never relaxed)
                _, runp, failp = T.arm_files(S, int(K))
                if not (os.path.exists(runp) or os.path.exists(failp)):
                    H.wj(failp, {"RECORD": "H2LT_ARM", "STATUS": "PARTITION_INVALID", "spec": T.CUR["spec"], "S": S, "K": int(K), "reason": "the V-cycle top is PARTITION_INVALID",
                                 "top_record": H.rel(tfail), "top_record_sha256": H.sha_file(tfail), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                continue
            L.cmd_sub(S, int(K))
    else:
        raise SystemExit(__doc__)
