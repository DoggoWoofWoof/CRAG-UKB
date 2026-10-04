import json, numpy as np
from scipy.stats import binomtest
rec = json.load(open('results/L1_X/blockrel_metaqa__v1.json'))
z = np.load('results/L1_X/blockrel_metaqa__v1.npz')
names = rec['rules']; cells = rec['cells']; NM = 6
rows = z['rows']; B = (rows % 2) == 1; A = ~B
obj = lambda v: float(np.mean(v[:4]))
def oA(n, h): return float(np.mean([obj(rec['table'][c][n][h]) for c in cells]))
print("marginal contribution (objA / objB):")
for n in ['SHIPPED','Sro+Sri+Tin','Sro+Sri+Tin+BRin','Sro+Sri+Tin+BRout','Sro+Sri+Tin+BM','Sro+Sri+BRin','Sro+Sri+BRout','Sro+Sri+BM','BRin','BRout','BM','Sro+BM','Sro+BRin','Sro+Sri+BRin+BM','Sro+Sri+BRin+BRout']:
    if n in names: print("  %-24s %.4f / %.4f" % (n, oA(n,'A'), oA(n,'B')))
best = max(names, key=lambda n: oA(n,'A')); print('best by A:', best)
def all_(c, n):
    return np.unpackbits(z['ALL__%s' % c], axis=2)[:, :, :NM].astype(bool)[names.index(n)]
for rule in [best, 'Sro+Sri+BM']:
    print('\nRULE', rule, '(half B, paired gained/lost vs typed T3 | vs shipped; p)')
    for c in cells:
        a = all_(c, rule)[B]; t = all_(c, 'Sro+Sri+Tin')[B]; s = all_(c, 'SHIPPED')[B]
        out = []
        for mi, M in enumerate(rec['budgets'][:4]):
            x, y = int((a[:, mi] & ~t[:, mi]).sum()), int((~a[:, mi] & t[:, mi]).sum())
            p = binomtest(x, x+y, .5).pvalue if x+y else 1
            x2, y2 = int((a[:, mi] & ~s[:, mi]).sum()), int((~a[:, mi] & s[:, mi]).sum())
            p2 = binomtest(x2, x2+y2, .5).pvalue if x2+y2 else 1
            out.append("B_N%d +%d/-%d p%.0e | +%d/-%d p%.0e" % (M, x, y, p, x2, y2, p2))
        print(' K%-5s' % c, ' ; '.join(out[:4]))
print('\nhop split @K432 B_N100 (half A / half B): rules', 'SHIPPED','Sro+Sri+Tin', best)
for n in ['SHIPPED','Sro+Sri+Tin',best]:
    e = rec['table']['432'][n]; print('  %-20s hopA %s hopB %s' % (n, e.get('hopA100'), e.get('hopB100')))
print('gold blocks all contacted', rec['gold_blocks_all_contacted'])
