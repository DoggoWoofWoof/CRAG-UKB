"""One-off patcher: scratchpad/_l1x_blockrel_route.py -> scratchpad/_l1x_blockrel_quot.py (step 15).  Plain string replacement; every anchor must match exactly once."""
import io

src = io.open("scratchpad/_l1x_blockrel_route.py", encoding="utf-8").read()


def sub(old, new, cnt=1):
    global src
    assert src.count(old) == cnt, (src.count(old), old[:80])
    src = src.replace(old, new)


head = '''"""L1X -- SELECT lane, step 15: BLOCK-QUOTIENT ROUTER, END TO END (development only; dataset-agnostic; a TWO-PROPAGATION-STEP candidate: NOT an L1 adoption -- measured so the contract ruling has a number).
Step 14 (_l1x_coloc_quot.py) found the relation-weighted block-quotient step applied to the block relation mass raises the contact ceiling (MetaQA K1000 .67 -> .78).  Here the SERVED order (ALL-gold at B_N) is measured with the same harness as step 12; only the block ORDER is varied (count B_P(q) shipped, asserted per row):
  ES     shipped            ES_m   BM desc (step 12)
  QMr    blocks by bm @ Pw  (Pw = row-normalised symmetric relation-weighted block quotient; ties -> ES)
  ES_qm  equal-weight RRF (K0 = 60) of the block ranks ES, BM, QMr (ties -> ES)
Everything else is as _l1x_blockrel_route.py (step 12), whose docstring follows.

'''
sub('"""L1X -- SELECT lane, step 12:', head + 'STEP 12 (parent):', 1)
sub('ROUTERS = ["ES", "ES_m", "ES_mr"]', 'ROUTERS = ["ES", "ES_m", "QMr", "ES_qm"]')
sub('"blockrelroute_%s__%s"', '"blockrelquot_%s__%s"')
sub('"blockrelroute_%s__%s.json"', '"blockrelquot_%s__%s.json"')
sub("_l1x_blockrel_route.py RUN", "_l1x_blockrel_quot.py RUN")
sub("_l1x_blockrel_route.py SHOW", "_l1x_blockrel_quot.py SHOW")
sub("import _l1x_rrt as R\n", "import _l1x_rrt as R\nfrom _l1x_coloc_quot import quotient_tables, sym_norm\n")
sub("    LIFT = {K: block_tables(maps[K], s, d, r, K, nrel) for K in cells}\n",
    "    LIFT = {K: block_tables(maps[K], s, d, r, K, nrel) for K in cells}\n    QT = {K: quotient_tables(maps[K], s, d, r, K, nrel) for K in cells}\n")
sub('''                sc = 1.0 / (K0 + rk_es) + 1.0 / (K0 + rk_m) + 1.0 / (K0 + rk_b)
                RO = {"ES": ro_es, "ES_m": o_m, "ES_mr": np.lexsort((rk_es, -sc))}
''', '''                pair, rel, cq = QT[K]
                Pw = sym_norm(np.bincount(pair, weights=cq * w[rel], minlength=K * K).reshape(K, K))
                o_q = np.lexsort((rk_es, -(bm @ Pw)))
                rk_q = np.empty(K, np.int64)
                rk_q[o_q] = np.arange(K)
                sc = 1.0 / (K0 + rk_es) + 1.0 / (K0 + rk_m) + 1.0 / (K0 + rk_q)
                RO = {"ES": ro_es, "ES_m": o_m, "QMr": o_q, "ES_qm": np.lexsort((rk_es, -sc))}
''')
io.open("scratchpad/_l1x_blockrel_quot.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
