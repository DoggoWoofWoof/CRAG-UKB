"""Generate scratchpad/_l1x_colloc17.py from scratchpad/_l1x_colloc16.py (string patches with occurrence asserts).
Step 17 = the user's 2026-10-04 'confidence-gated co-location + node-level within-block scoring' round, plus the contact (router) lever the step-16 ceiling analysis points at:
  * routers   ES (shipped), ES_m (BM order), ES_md (BM * Dpr order), ES_cov (greedy capacity-capped coverage of every seed's one-hop block mass; first B_P picks, then the BM order) -- ALL one propagation step, parameter-free;
  * channels  BM, CRD, RN (node relation compatibility, no block mass), RB (RN normalised by the block maximum = within-block node score);
  * gate signals per (K, row), stored; the GATES THEMSELVES are pre-declared in work/_colloc17_gate.py (no tuning): G1 diversity, G2 relation concentration, G3 rank agreement, G4 majority consensus."""
import re

SRC, DST = "scratchpad/_l1x_colloc16.py", "scratchpad/_l1x_colloc17.py"
s = open(SRC, encoding="utf-8").read()


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, "expected %d occurrence(s) of %r, found %d" % (n, old[:70], s.count(old))
    s = s.replace(old, new)


sub("colloc16", "colloc17", s.count("colloc16"))
sub('ROUTERS = ["ES", "ES_m"]', 'ROUTERS = ["ES", "ES_m", "ES_md", "ES_cov"]')
# views / channels / rules
m = re.search(r'^VIEWS = \[.*\]$', s, re.M)
assert m and '"CRD"]' in m.group(0)
s = s.replace(m.group(0), m.group(0)[:-1] + ', "RN", "RB"]')
m = re.search(r'^CH = \[.*$', s, re.M)
s = s.replace(m.group(0), 'CH = ["BM", "CRD", "RN", "RB"]                                         # channels: COLLOC-B, -RD, node relation compatibility, within-block-normalised relation compatibility')
m = re.search(r'^RULES = \[.*$', s, re.M)
s = s.replace(m.group(0), 'RULES = [(), ("Sro", "Sri", "Tin"), ("Sro", "Sri", "BRin", "BM")] + [("Sro", "Sri", "Tin", c) for c in CH] + [("T", "CRD"), ("T", "RB")]')
m = re.search(r'^PROT_C = .*$', s, re.M)
s = s.replace(m.group(0), "PROT_C = []")
# coverage router helper
sub("def rinv(v):", '''def cov_order(A_, b, base, rkm):
    """greedy capacity-capped coverage of every seed's one-hop block mass: pick the block with the largest covered mass sum_h min(A[h, blk], rem_h) (ties -> the better BM rank), rem_h -= covered;
    the first b picks, then the remaining blocks in the base (BM) order.  One propagation step; no parameters."""
    nz = np.flatnonzero(A_.sum(0) > 0)
    Z = A_[:, nz]
    rem = A_.sum(1).copy()
    taken = np.zeros(len(nz), bool)
    out = []
    rz = rkm[nz]
    for _ in range(min(b, len(nz))):
        g = np.minimum(Z, rem[:, None]).sum(0)
        g[taken] = -1.0
        mx = g.max()
        if mx <= 0:
            break
        cand = np.flatnonzero(g >= mx * (1.0 - 1e-12))
        i = int(cand[np.argmin(rz[cand])])
        taken[i] = True
        out.append(int(nz[i]))
        rem = np.maximum(rem - Z[:, i], 0.0)
    out = np.asarray(out, np.int64)
    rest = base[~np.isin(base, out)]
    return np.concatenate([out, rest])


def rinv(v):''')
# routers + signals
sub('''                RO = {"ES": ro_es, "ES_m": o_m, "ES_mr": np.lexsort((rk_es, -sc))}''',
    '''                o_md = np.lexsort((rk_es, -(bm * Dpr)))
                o_cov = cov_order(A_, b, o_m, rk_m)
                RO = {"ES": ro_es, "ES_m": o_m, "ES_mr": np.lexsort((rk_es, -sc)), "ES_md": o_md, "ES_cov": o_cov}
                tb = o_m[:b]
                bmt = bm[tb]
                pb = bm / max(bm.sum(), 1e-300)
                SIG[K]["dpr1"][j] = Dpr[o_m[0]]
                SIG[K]["dprb"][j] = float((bmt * Dpr[tb]).sum() / max(bmt.sum(), 1e-300))
                SIG[K]["marg"][j] = (1.0 - bm[o_m[1]] / bm[o_m[0]]) if K > 1 and bm[o_m[0]] > 0 else 0.0
                SIG[K]["ent"][j] = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum() / np.log(K)) if K > 1 else 0.0
                SIG[K]["agree1"][j] = float(o_m[0] == o_md[0])
                SIG[K]["agreeb"][j] = float(len(np.intersect1d(o_m[:b], o_md[:b])) / b)
                SIG[K]["agreec"][j] = float(len(np.intersect1d(o_m[:b], o_cov[:b])) / b)
                SIG[K]["relconc"][j] = relconc''')
# relation concentration of the seed-edge mass: computed once per row after the seed edges exist
sub('''            # ---- routing (shipped, not varied)''', '''            _rm = np.bincount(np.concatenate([rO[pO], rI[pI]]).astype(np.int64), weights=np.concatenate([xh[hO] * w[rO[pO].astype(np.int64)], xhi[hI] * w[rI[pI].astype(np.int64)]]), minlength=nrel)
            relconc = float(_rm.max() / max(_rm.sum(), 1e-300))
            # ---- routing (shipped, not varied)''')
# node-level channels
sub('''                    raw["CD"], raw["CRD"] = bm[hb] * Dpr[hb], bm[hb] * Dpr[hb] * Rn''',
    '''                    raw["CD"], raw["CRD"] = bm[hb] * Dpr[hb], bm[hb] * Dpr[hb] * Rn
                    bmx = np.zeros(K)
                    np.maximum.at(bmx, hb, Rn)
                    raw["RN"], raw["RB"] = Rn, Rn / np.maximum(bmx[hb], 1e-12)''')
sub('''    BPQ = {K: np.zeros(nq, np.int32) for K in cells}''', '''    BPQ = {K: np.zeros(nq, np.int32) for K in cells}
    SIG = {K: {n_: np.zeros(nq, np.float32) for n_ in ("dpr1", "dprb", "marg", "ent", "agree1", "agreeb", "agreec", "relconc")} for K in cells}''')
sub('''    for K in cells:
        arrays["BPQ__%d" % K] = BPQ[K]''', '''    for K in cells:
        arrays["BPQ__%d" % K] = BPQ[K]
        for n_, v_ in SIG[K].items():
            arrays["SIG__%d__%s" % (K, n_)] = v_''')
# the step-12 identity check covers the two routers that record has
sub('''    for c in CK:
        pa = np.unpackbits(prev["ALL__%d|%s" % c], axis=2)[:, :, :NM].astype(bool)''', '''    for c in CK:
        if c[1] not in ("ES", "ES_m"):
            continue
        pa = np.unpackbits(prev["ALL__%d|%s" % c], axis=2)[:, :, :NM].astype(bool)''')
open(DST, "w", encoding="utf-8", newline="\n").write(s)
print("wrote", DST, len(s))
