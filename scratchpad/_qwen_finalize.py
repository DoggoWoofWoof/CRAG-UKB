"""Run AFTER _qwen_sent_emb.npy lands. stage=materialize|valeval|matrix."""
import sys, json, numpy as np
sys.path.insert(0, 'scratchpad'); sys.path.insert(0, '.')
import l2_relation_qwen as Q

stage = sys.argv[1]
if stage == "materialize":
    out = {}
    for ds in Q.PILOTS:
        for sp in Q.SPLITS:
            out[f"{ds}/{sp}"] = Q.materialize_relation(ds, sp)
    # mask parity on VAL
    par = {ds: Q.mask_parity_with_minilm(ds, "val") for ds in Q.PILOTS}
    json.dump({"materialize": out, "mask_parity": par}, open("scratchpad/_qwen_materialize.json", "w"), indent=1, default=str)
    print(json.dumps({"materialize": out, "mask_parity": par}, indent=1, default=str))
elif stage == "valeval":
    out = {ds: Q.qwen_eval(ds, "val") for ds in Q.PILOTS}
    json.dump(out, open("scratchpad/_qwen_valeval.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
elif stage == "matrix":
    out = {}
    for ds in Q.PILOTS:
        for sp in Q.SPLITS:
            out[f"{ds}/{sp}"] = Q.build_expert_matrix(ds, sp)
            print("matrix done", ds, sp, flush=True)
    json.dump(out, open("scratchpad/_qwen_matrix.json", "w"), indent=1, default=str)
    print("MATRIX_ALL_DONE")
