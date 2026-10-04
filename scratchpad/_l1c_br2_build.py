"""BOUNDED_VERTEXCUT_R2 step 1 (twelfth ruling): write the OWNER hypergraph of a dataset -- the primal STRUCT graph (objects = nodes of
degree >= 1, nets = STRUCT edges with 2 pins, unit weight, k = R * k_f) -- for the validated PHG driver.  Nothing else; the partition
itself is made by the pinned driver:

    python -u scratchpad/_l1c_br2_build.py <ds>                       -> parts/<ds>__BR2_OWNERS_k<k>.npz + .json
    python -u scratchpad/_l1c_phg.py <ds> BR2_OWNERS_k<k>              -> parts/<ds>__BR2_OWNERS_k<k>__PHG_con.npy + .RUN.json   (NP = 4, unchanged)
    python -u scratchpad/_l1c_br2_assign.py <ds>                      -> parts/<ds>__BR2_k<k>__R2.npz + .json   (degree-0 rule + alternates)
    python -u scratchpad/_l1c_br2_replay.py <ds>                      -> results/L1_COVPART/br2_replay_A_<ds>.json

The pre-registration (PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json) must exist and pin this file and the library before anything is built.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1c_br2_lib as L  # noqa: E402
import _l1c_vcut_lib as V  # noqa: E402

X = L.X
OUT = L.OUT


def main(ds):
    pre = json.load(open(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json"), encoding="utf-8"))
    for mod, pn in pre["code"]["new_modules"].items():
        if mod in ("_l1c_br2_lib.py", "_l1c_br2_build.py"):
            assert L.sha_file(os.path.join(HERE, mod)) == pn["sha256"], "%s changed since the pre-registration" % mod
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert L.sha_file(os.path.join(X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    Gr = V.Graph(ds)
    assert Gr.keys_sha == pre["graphs_frozen"][ds]["struct_keys_sha256"] and Gr.C == pre["constants"]["C"] and Gr.k_f == pre["graphs_frozen"][ds]["k_f"]
    fp, meta = L.build_owner_hypergraph(ds, Gr)
    L.log("next: python -u scratchpad/_l1c_phg.py %s %s" % (ds, meta["tag"]))


if __name__ == "__main__":
    main(sys.argv[1])
