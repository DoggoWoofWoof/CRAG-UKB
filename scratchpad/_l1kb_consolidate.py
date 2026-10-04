"""CONSOLIDATED SCOREBOARD -- every config from every round, paired against BOTH baselines.

The per-round scoreboards pair each config against BASE (the un-routed canonical P50).  The
promotion decision needs the other pairing too: against the SAFE UNIVERSAL BASELINE
B6_S4_F6_Ms64_Mr32, which is what would actually be given up by promoting something else.
Selection criterion 1 ("no statistically significant ALL regression") is therefore enforced
against BOTH -- a rule that beats the un-routed baseline while significantly regressing the
router we already trust is not a safe promotion.

  python scratchpad/_l1kb_consolidate.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT
import _l1kb_core as KB
import _l1kb_router as JR
import _l1kb_round3 as R3
import _l1kb_round4 as R4

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
TEXT = ["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]


def all_specs():
    """union of the four round grids, normalised to one spec format."""
    out = []
    seen = set()
    for spec in JR.GRIDS["round1"]:
        fam, gate, B, sch = spec
        nm = fam + ("_" + gate if gate else "") + ("_" + sch if sch != "S4" else "") + "_B%d" % B
        out.append(dict(name=nm, kind={"BASE": "BASE", "F6": "F6"}.get(fam, "SET"),
                        fam=fam, gate=gate, schan=sch, B=B))
    out += [dict(s) for s in JR.GRIDS["round2"]]
    out += [dict(s) for s in R3.GRID]
    out += [dict(s) for s in R4.GRID]
    uniq = []
    for s in out:
        if s["name"] in seen:
            continue
        seen.add(s["name"]); uniq.append(s)
    return uniq


def build(spec, GRP):
    k = spec["kind"]
    if k == "BASE":
        return KB.sel_base
    if k == "F6":
        return KB.sel_f6
    if k == "SET":
        return JR.make_selector(spec["fam"], spec.get("gate"), GRP, spec.get("schan", "S4"))
    if k == "COMPLETE":
        return JR.build_sel(spec, GRP)
    if k == "D":
        return R3.make_dsel(spec["lex"], spec["when"], spec["gate"], spec["prio"], GRP)
    if k == "ALT":
        return R4.make_alt(spec["sched"], spec["lead"], spec["disc"], spec["when"],
                           spec["gate"], GRP)
    raise ValueError(k)


def main():
    SPECS = all_specs()
    log("%d unique configs" % len(SPECS))
    SB = {}
    for ds in KB.DSETS:
        z, meta = KB.load(ds)
        hops = z["hops"]; goldp = KB.goldparts(z, meta)
        C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
        ind_base = C["ind_base"]; fold = JR.folds(ds, z)
        ctxs = {B: KB.contexts(z, meta, C, B) for B in sorted({s["B"] for s in SPECS})}
        ref = RT.evaluate(ds, z, meta, dict(KB.BASE_CFG), C); ref.pop("_ind")
        f6 = KB.run_selector(ctxs[6], KB.sel_f6, goldp, ind_base)
        assert round(float(f6["ind"].mean()), 4) == ref["ALL"], f"{ds}: F6 replay drift"
        res = {}
        for spec in SPECS:
            r = KB.run_selector(ctxs[spec["B"]], build(spec, GRP), goldp, ind_base)
            e = JR.score_run(r, ind_base, hops, fold, r["churn"])
            e["vs_F6"] = RT.mcnemar(r["ind"], f6["ind"])
            e["dALL_vs_F6"] = round(float(r["ind"].mean() - f6["ind"].mean()), 4)
            e["abstain_frac"] = round(float((r["churn"] == 0).mean()), 4)
            if (hops >= 0).any():
                for h in e["per_hop"]:
                    m = hops == int(h)
                    e["per_hop"][h]["vs_F6"] = RT.mcnemar(r["ind"][m], f6["ind"][m])
                    e["per_hop"][h]["dALL_vs_F6"] = round(
                        float(r["ind"][m].mean() - f6["ind"][m].mean()), 4)
            res[spec["name"]] = e
        SB[ds] = {"meta": {"n": meta["n_dev_queries"],
                           "BASE_ALL": round(float(ind_base.mean()), 4),
                           "F6_ALL": ref["ALL"]}, "configs": res}
        log("%-15s done (%d configs)" % (ds, len(res)))
    fp = f"{KB.KBD}/scoreboard_consolidated.json"
    json.dump({"SPECS": SPECS, "SCOREBOARD": SB}, open(fp, "w"), indent=1)
    log("wrote", fp)

    names = sorted(set.intersection(*[set(SB[d]["configs"]) for d in SB]))
    rows = []
    for n in names:
        d = {ds: SB[ds]["configs"][n] for ds in SB}
        mh = d["metaqa"]["per_hop"]
        rows.append(dict(
            n=n,
            reg_base=sum(1 for v in d.values() if v["sig"] and v["dALL"] < 0),
            reg_f6=sum(1 for v in d.values() if v["vs_F6"]["sig"] and v["dALL_vs_F6"] < 0),
            win_f6=sum(1 for v in d.values() if v["vs_F6"]["sig"] and v["dALL_vs_F6"] > 0),
            h3=mh["3"]["dALL"], h2=mh["2"]["dALL"],
            h3f=mh["3"]["dALL_vs_F6"], h2f=mh["2"]["dALL_vs_F6"],
            wq=d["webqsp"]["dALL"],
            keep=min(d[ds]["dALL_vs_F6"] for ds in TEXT),
            worst=min(v["dALL"] for v in d.values()),
            macro=float(np.mean([v["dALL"] for v in d.values()])),
            chn=float(np.mean([v["churn_mean"] for v in d.values()])),
            ev=sum(v["gold_evicted"] for v in d.values())))
    rows.sort(key=lambda r: (r["reg_base"] + r["reg_f6"], -r["h3"], -r["h2"], -r["wq"],
                             -r["keep"], -r["worst"], -r["macro"], r["ev"], r["chn"]))
    hdr = "%-26s %4s %4s %4s %8s %8s %8s %8s %8s" % (
        "config", "rgB", "rgF6", "winF", "mq_h3", "mq_h2", "webqsp", "keepTXT", "macro")
    print("\n=== ALL CONFIGS, criterion 1 enforced against BOTH baselines ===")
    print(hdr)
    for r in rows[:26]:
        print("%-26s %4d %4d %4d %+8.4f %+8.4f %+8.4f %+8.4f %+8.4f" % (
            r["n"], r["reg_base"], r["reg_f6"], r["win_f6"], r["h3"], r["h2"], r["wq"],
            r["keep"], r["macro"]))
    safe = [r for r in rows if r["reg_base"] == 0 and r["reg_f6"] == 0]
    print("\n=== SAFE SET: zero significant regression vs BASE *and* vs F6 (%d configs) ===" % len(safe))
    print(hdr)
    for r in sorted(safe, key=lambda r: (-r["h3"], -r["h2"], -r["wq"]))[:18]:
        print("%-26s %4d %4d %4d %+8.4f %+8.4f %+8.4f %+8.4f %+8.4f" % (
            r["n"], r["reg_base"], r["reg_f6"], r["win_f6"], r["h3"], r["h2"], r["wq"],
            r["keep"], r["macro"]))
    json.dump(rows, open(f"{KB.KBD}/consolidated_ranking.json", "w"), indent=1)


if __name__ == "__main__":
    main()
