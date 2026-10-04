import io, sys
p = "scratchpad/_l1ep_verdict.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)

# 1. compute the class split once, thread it through
rep('    metaqa_conv = [x for x in conv if x[0] == "metaqa"]\n'
    '    p4_regr = [x for x in regr if x[1].startswith("P4_")]\n',
    '    metaqa_conv = [x for x in conv if x[0] == "metaqa"]\n'
    '    p4_regr = [x for x in regr if x[1].startswith("P4_")]\n'
    '    CS = class_split(AN, "P4_CE_LOCAL_ONLY")\n')

# 2. label text: two KBs, not one corpus
rep('            "the alternative partitioning is a genuinely better partitioner on KB-style "\n'
    '            "multi-hop AND the router converts it (+0.0966 on MetaQA, hop3 nearly doubled). It "\n'
    '            "is withheld only because the same universal rule significantly regresses a text "\n'
    '            "corpus. This is explicitly NOT `PARTITION_QUALITY_IMPROVED_ROUTER_CANNOT_CONVERT`: "\n'
    '            "the router had no trouble converting it.")',
    '            f"the alternative partitioning is a genuinely better partitioner on KB-style "\n'
    '            f"multi-hop AND the router converts it, on BOTH knowledge-base corpora "\n'
    '            f"({CS[\'kb_sig_positive\']}/{CS[\'kb_n\']} significantly positive: MetaQA +0.0966 with "\n'
    '            f"hop3 nearly doubled, WebQSP +0.0423) and on neither text corpus "\n'
    '            f"({CS[\'text_sig_negative\']}/{CS[\'text_n\']} significantly negative). It is withheld "\n'
    '            f"only because the same universal rule significantly regresses a text corpus. This "\n'
    '            f"is explicitly NOT `PARTITION_QUALITY_IMPROVED_ROUTER_CANNOT_CONVERT`: the router "\n'
    '            f"had no trouble converting it -- twice, on the corpus class where the headroom is.")')

# 3. combined basis: state the split, not a single corpus
rep('                f"The edge axis moves exact-P50 by at most {max(abs(best), abs(worst)):.4f} "\n'
    '                f"anywhere. The partition axis moves it by +0.0966 on MetaQA, and by -0.07 to "\n'
    '                f"-0.25 when degraded to a balanced random blocking at identical exposure. The "\n'
    '                f"two axes are not comparable in magnitude.",',
    '                f"The edge axis moves exact-P50 by at most {max(abs(best), abs(worst)):.4f} "\n'
    '                f"anywhere. The partition axis moves it by +0.0966 on MetaQA and +0.0423 on "\n'
    '                f"WebQSP, and by -0.07 to -0.25 when degraded to a balanced random blocking at "\n'
    '                f"identical exposure. The two axes are not comparable in magnitude.",\n'
    '                f"The partition gain is separated by CORPUS CLASS, not by corpus: both "\n'
    '                f"knowledge-base corpora convert significantly and no free-text corpus does "\n'
    '                f"(clean split = {CS[\'CLEAN_KB_VS_TEXT_SPLIT\']}). One KB could be a quirk; two, "\n'
    '                f"under one rule with no per-corpus tuning, is a class effect.",')

# 4. thread CS into questions()
rep('            "QUESTIONS": questions(R, AN, universal)}',
    '            "CLASS_SPLIT_P4_CE_LOCAL_ONLY": CS,\n'
    '            "QUESTIONS": questions(R, AN, universal, CS)}')
rep('def questions(R, AN, universal):\n    E = R["EDGE_SUBSTRATE"]',
    'def questions(R, AN, universal, CS=None):\n'
    '    E = R["EDGE_SUBSTRATE"]\n'
    '    CS = CS or class_split(AN, "P4_CE_LOCAL_ONLY")\n'
    '    tbl = "; ".join(f"{d} ({c}) {x:+.4f}{\'*\' if sg else \'\'}"\n'
    '                    for d, c, _f, x, _g, _l, _p, sg, _n in\n'
    '                    sorted(CS["rows"], key=lambda z: -z[3]))')
io.open(p, "w", encoding="utf-8").write(s)
print("OK")
