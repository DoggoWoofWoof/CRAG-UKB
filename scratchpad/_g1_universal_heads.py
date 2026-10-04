"""G1 step 0 — build the frozen dataset-agnostic UNIVERSAL offset + mixture heads on the SOURCE mix
(2wiki_clean + musique_clean TRAIN), in the gte_qwen frozen space. These close the Mixture gap the same
frozen way as the universal offset head, and are applied WITHOUT refit to every target to produce the
offset_score / mixture_score experts in the C11a interface.

Trains on TRAIN only (test is never evaluated here). base = OffsetHead (offset expert), mix = MixtureHead
K=8 (mlpT / mixture expert). In-batch negatives (CPU-cheap; no hard mining). Saves stable checkpoints +
records the L1 partition FullCov@20 of each head on the SOURCE datasets' own train-derived probe so we can
report whether ONE universal head generalizes across the two source distributions (the user's 'try the
universal head, see if it generalizes' check) — no target data, no test."""
import os, sys, json, time, logging
sys.path.insert(0, os.path.abspath("."))
import numpy as np, torch
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
import src.experiments.l1_universal_head as U

SRC = ["2wiki_clean", "musique_clean"]
SUBDIR = "gte_qwen"
OUTDIR = "results/L2/_heads"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # ---- load source datasets (cached queries; train used for training, tiny test just to satisfy _load) ----
    per_ds, data = {}, {}
    import faiss
    for d in SRC:
        data[d] = U._load(d, SUBDIR, limit=8000, tr_cap=4000, te_cap=50)
        idx = faiss.IndexFlatIP(data[d]["X"].shape[1]); idx.add(data[d]["X"])
        per_ds[d] = {"train": data[d]["train"], "Xt": torch.tensor(data[d]["X"], device=DEV), "index": idx}
        log(f"loaded {d}: X{data[d]['X'].shape} npart={data[d]['npart']} ntrain={len(data[d]['train'][0])}")

    manifest = {"source": SRC, "subdir": SUBDIR, "device": "cpu", "K": 8,
                "trained_on": "SOURCE TRAIN only (2wiki+musique); NO target data; NO test eval",
                "heads": {}}
    HEADS = (("hard", "offset"), ("mix_hard", "mixture")) if DEV.type == "cuda" else (("base", "offset"), ("mix", "mixture"))
    for kind, name in HEADS:
        log(f"=== training universal '{name}' (kind={kind}) ===")
        t = time.time()
        head = U._train_universal(kind, per_ds, DEV, epochs=15, K=8)
        dt = time.time() - t
        out = f"{OUTDIR}/universal_{name}_src_gteqwen.pt"
        torch.save(head.state_dict(), out)
        nparam = sum(p.numel() for p in head.parameters())
        # source-only generalization probe: partition FullCov@20 head-vs-dense on each source's TRAIN tail
        probe = {}
        for d in SRC:
            X = data[d]["X"]; mem_idx = data[d]["mem_idx"]; npart = data[d]["npart"]
            qtr, str_, gtr = data[d]["train"]
            # hold out a deterministic tail of TRAIN as an internal probe (never test)
            nprobe = min(1500, len(qtr) // 3)
            qp, sp, gp = qtr[-nprobe:], str_[-nprobe:], gtr[-nprobe:]
            gpl = [[mem_idx[g] for g in gg] for gg in gp]
            ho = U._head_order(head, qp, X[sp], X, DEV)
            vh = U._votes(ho, mem_idx, npart)
            do = U._dense_order(qp, X, DEV); vd = U._votes(do, mem_idx, npart)
            probe[d] = {"dense@20": U._fullcov(vd, gpl, npart)[20],
                        "head@20": U._fullcov(vh, gpl, npart)[20],
                        "equal@20": U._fullcov(vd + vh, gpl, npart)[20],
                        "n_probe": nprobe}
            log(f"  probe[{d}] dense={probe[d]['dense@20']} head={probe[d]['head@20']} equal={probe[d]['equal@20']}")
        manifest["heads"][name] = {"kind": kind, "ckpt": out, "params": nparam,
                                   "train_sec": round(dt, 1), "source_train_probe_fullcov20": probe}
        log(f"saved {name} -> {out} ({nparam} params, {dt:.0f}s)")
        del head
    json.dump(manifest, open(f"{OUTDIR}/_universal_heads_src_manifest.json", "w"), indent=1)
    log("DONE universal heads -> " + f"{OUTDIR}/_universal_heads_src_manifest.json")


if __name__ == "__main__":
    main()
