"""Local re-run of the Modal apisig discriminating test -- same checks, WSL Linux instead of a
Modal container.  If this disagrees with hmeta/MTKAHYPAR_API.json the local build is not trusted.
"""
import json
import mtkahypar

out = {"version": getattr(mtkahypar, "__version__", "installed")}
init = mtkahypar.initialize(2)


def ctx_of(k, eps):
    c = init.context_from_preset(mtkahypar.PresetType.DETERMINISTIC_QUALITY)
    c.set_partitioning_parameters(k, eps, mtkahypar.Objective.KM1)
    try:
        c.logging = False
    except Exception:
        pass
    return c


he = [[0, 1, 2], [2, 3]]
for label, args in (("unweighted", (ctx_of(2, 0.03), 4, 2, he)),
                    ("node+edge_weights", (ctx_of(2, 0.03), 4, 2, he, [1, 1, 1, 1], [7, 3]))):
    hg = init.create_hypergraph(*args)
    p = hg.partition(args[0])
    out["ok_" + label] = [int(p.block_id(i)) for i in range(4)]

path = [[0, 1], [1, 2], [2, 3]]
for label, ew in (("equal_1_1_1", [1, 1, 1]), ("heavy_middle_1_100_1", [1, 100, 1])):
    cc = ctx_of(2, 0.001)
    mtkahypar.set_seed(0)
    hg = init.create_hypergraph(cc, 4, 3, path, [1, 1, 1, 1], ew)
    p = hg.partition(cc)
    blk = [int(p.block_id(i)) for i in range(4)]
    out["discriminate_" + label] = {"blocks": blk, "km1": float(p.km1()),
                                    "same_block_1_2": blk[1] == blk[2]}
d1, d2 = out["discriminate_equal_1_1_1"], out["discriminate_heavy_middle_1_100_1"]
out["EDGE_WEIGHTS_AFFECT_OBJECTIVE"] = bool(d1["same_block_1_2"] is False
                                            and d2["same_block_1_2"] is True)
print(json.dumps(out, indent=1))
