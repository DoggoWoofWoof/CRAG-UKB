"""One-off patcher: scratchpad/_l1x_blockrel_route.py -> scratchpad/_l1x_colloc16.py (step 16, COLLOC ladder).  Plain string replacement; every anchor must match exactly once."""
import io

src = io.open("scratchpad/_l1x_blockrel_route.py", encoding="utf-8").read()


def sub(old, new, cnt=1):
    global src
    assert src.count(old) == cnt, (src.count(old), old[:90])
    src = src.replace(old, new)


head = '''"""L1X -- SELECT lane, step 16: COLLOC LADDER = block evidence -> node CO-LOCATION prior (development only; dataset-agnostic typed-graph switch; ONE propagation step, NO traversal).
User direction 2026-10-04: 94-100 % of MetaQA questions fit every gold in the B_P contacted blocks, but the one-hop router contacts the hop-2 gold block only ~40 %: L1 headroom is the relevance already inside the
contacted blocks.  A NON-recursive promotion channel projects the block evidence back to nodes,  C(u|q) = B(P(u)|q) * R(u|q),  and is fused with the node order S0 (S0 is never replaced):
  B(P)   = BM(P), the relation-conditioned one-hop mass the block receives from the FLAT hits (step 11)               [COLLOC-B == the frozen candidate's BM view: an identity check]
  R(u)   = sum over the DISTINCT relations r incident to u (in or out) of w(q, r)  (w = 1 / (1 + rank of r by dense cosine): the step-5 query relation mass; nothing fitted)
  Rs(u)  = the same with each term divided by log(2 + deg_r(u))  (specificity-normalised)
  D(P)   = independent-source diversity of the block's mass = participation ratio (sum_h a_h)^2 / sum_h a_h^2 over the FLAT seeds h, a_h = the mass seed h sends into P  (1 = one hub seed, 3 = three equal seeds)
  mechanisms (the ladder; 5 interpretable arms, no search):  B = B      R = B*R      Rs = B*Rs      D = B*D      RD = B*D*R
  fusion (each mechanism x each fusion):
    F4    equal-weight RRF (K0 = 60) of Sro, Sri, Tin and the channel                                   (B in F4 == 'Sro+Sri+Tin+BM', the candidate)
    F2    RRF of the typed-rule order T = RRF(Sro, Sri, Tin) and the channel                              (the channel is half the vote)
    PROT  conservative: the top ceil(B_N/2) of T are PROTECTED, the remaining B_N - ceil(B_N/2) slots are filled in the F2 order
  routers: ES (shipped) and ES_m (the candidate router); count B_P(q) is the shipped one (asserted per row); the served candidate set is every node of the contacted blocks.
Identity (always on): the shipped, 'Sro+Sri+Tin', 'Sro+Sri+Tin+BM' and 'Sro+Sri+BRin+BM' arms must equal the step-12 record blockrelroute_<ds>__v1 cell for cell (rows are the same).
Text datasets (one relation) are not run: the typed switch is off, so they are byte-identical to the shipped L1 by construction.
  python -u scratchpad/_l1x_colloc16.py RUN <metaqa|webqsp> <tag> [--rows=K]      -> results/L1_X/colloc16_<ds>__<tag>.{json,npz}  (write-once)

'''
sub('"""L1X -- SELECT lane, step 12:', head + 'STEP 12 (parent):', 1)
sub('VIEWS = ["ship", "Sro", "Sri", "Tin", "Tout", "BRin", "BRout", "BM"]', 'VIEWS = ["ship", "Sro", "Sri", "Tin", "Tout", "BRin", "BRout", "BM", "T", "CR", "CRs", "CD", "CRD"]')
sub('''ROUTERS = ["ES", "ES_m", "ES_mr"]
RULES = [(), ("Sro", "Sri", "Tin"), ("Sro", "BM"), ("Sro", "Sri", "BM"), ("Sro", "Sri", "Tin", "BM"), ("Sro", "Sri", "BRin", "BM")]
''', '''ROUTERS = ["ES", "ES_m"]
CH = ["BM", "CR", "CRs", "CD", "CRD"]                                   # the channels: COLLOC-B, -R, -Rs, -D, -RD
RULES = [(), ("Sro", "Sri", "Tin"), ("Sro", "Sri", "BRin", "BM")] + [("Sro", "Sri", "Tin", c) for c in CH] + [("T", c) for c in CH]
PROT_C = list(CH)


def rinv(v):
    o = np.argsort(-v, kind="stable")
    rk = np.empty(len(v), np.int64)
    rk[o] = np.arange(len(v))
    return 1.0 / (K0 + rk)


def profc(key, rel, N, nrel):
    """CSR of the DISTINCT relations of every node with the edge count of each (node, relation): (ptr, rels, cnt)"""
    u, c = np.unique(key * nrel + rel, return_counts=True)
    node, r = u // nrel, u % nrel
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(node, minlength=N))
    return ptr, r.astype(np.int32), c.astype(np.int32)
''')
sub('''    pIn, relIn = RS.profile(d, r, N, nrel)
    pOut, relOut = RS.profile(s, r, N, nrel)
''', '''    pIn, relIn, cIn = profc(d, r, N, nrel)
    pOut, relOut, cOut = profc(s, r, N, nrel)
''')
sub('''    names = ["SHIPPED" if not r_ else "+".join(r_) for r_ in rules]
    nr = len(rules)
    RULEM = np.zeros((nr, len(VIEWS)), np.float64)
''', '''    names = ["SHIPPED" if not r_ else "+".join(r_) for r_ in rules] + ["PROT:T+" + c for c in PROT_C]
    nr, nrules = len(names), len(rules)
    RULEM = np.zeros((nrules, len(VIEWS)), np.float64)
''')
sub('''            sm = sro + sri
''', '''            sm = sro + sri
            nO_, vO_ = aO[pO].astype(np.int64), xh[hO] * w[rO[pO].astype(np.int64)]
            nI_, vI_ = aI[pI].astype(np.int64), xhi[hI] * w[rI[pI].astype(np.int64)]
''')
sub('''                bm = np.bincount(hard, weights=sm, minlength=K)
''', '''                bm = np.bincount(hard, weights=sm, minlength=K)
                key_ = np.concatenate([hO * K + hard[nO_], hI * K + hard[nI_]])
                A_ = np.bincount(key_, weights=np.concatenate([vO_, vI_]), minlength=nh * K).reshape(nh, K)
                bmA = A_.sum(0)
                assert np.allclose(bmA, bm, rtol=1e-6, atol=1e-9), "block mass from the (seed, block) table differs from BM"
                Dpr = np.divide(bmA ** 2, (A_ ** 2).sum(0), out=np.zeros(K), where=bmA > 0)
''')
sub('''                    raw = {"ship": -np.arange(m, dtype=np.float64), "Sro": sro[C], "Sri": sri[C], "Tin": tmax(pIn, relIn, C), "Tout": tmax(pOut, relOut, C),
                           "BRin": brin[hb], "BRout": (lout @ w)[hb], "BM": bm[hb]}
''', '''                    raw = {"ship": -np.arange(m, dtype=np.float64), "Sro": sro[C], "Sri": sri[C], "Tin": tmax(pIn, relIn, C), "Tout": tmax(pOut, relOut, C),
                           "BRin": brin[hb], "BRout": (lout @ w)[hb], "BM": bm[hb]}
                    raw["T"] = rinv(raw["Sro"]) + rinv(raw["Sri"]) + rinv(raw["Tin"])
                    Rn = rsum(pIn, relIn, cIn, C, False) + rsum(pOut, relOut, cOut, C, False)
                    Rs = rsum(pIn, relIn, cIn, C, True) + rsum(pOut, relOut, cOut, C, True)
                    raw["CR"], raw["CRs"] = bm[hb] * Rn, bm[hb] * Rs
                    raw["CD"], raw["CRD"] = bm[hb] * Dpr[hb], bm[hb] * Dpr[hb] * Rn
''')
sub('''                    O = np.argsort(-S, axis=1, kind="stable")
                    NI = np.empty((nr, m), np.int32)
                    NI[np.arange(nr)[:, None], O] = np.arange(m, dtype=np.int32)[None, :]
                    mx = NI[:, gi].max(axis=1)
                    ALL[(K, rn)][:, j] = mx[:, None] < MCa[None, :]
''', '''                    O = np.argsort(-S, axis=1, kind="stable")
                    NI = np.empty((nrules, m), np.int32)
                    NI[np.arange(nrules)[:, None], O] = np.arange(m, dtype=np.int32)[None, :]
                    mx = NI[:, gi].max(axis=1)
                    ALL[(K, rn)][:nrules, j] = mx[:, None] < MCa[None, :]
                    rkT = np.empty(m, np.int64)
                    rkT[np.argsort(-raw["T"], kind="stable")] = np.arange(m)
                    iT = INV[VIEWS.index("T")]
                    for pi, cn in enumerate(PROT_C):
                        O2 = np.argsort(-(iT + INV[VIEWS.index(cn)]), kind="stable")
                        P2 = np.empty(m, np.int64)
                        P2[O2] = np.arange(m)
                        for mi, mb in enumerate(MC):
                            h_ = (mb + 1) // 2
                            prot = rkT < h_
                            posnp = np.cumsum(~prot[O2]) - 1
                            ALL[(K, rn)][nrules + pi, j, mi] = bool((prot[gi] | (posnp[P2[gi]] < mb - h_)).all())
''')
sub('''def block_tables(hard, s, d, r, K, nrel):''', '''def rsum(ptr, rels, cnts, nodes, spec):
    """sum over the DISTINCT relations incident to every node of the weight w(q, r) -- w is set per row in the caller's scope through the module global _W"""
    s2 = ptr[nodes]
    ln = ptr[nodes + 1] - s2
    out = np.zeros(len(nodes))
    if ln.sum() > 0:
        h_, p_ = E.entries(s2, ln)
        ww = _W[0][rels[p_]]
        if spec:
            ww = ww / np.log(2.0 + cnts[p_])
        out = np.bincount(h_, weights=ww, minlength=len(nodes))
    return out


_W = [None]


def block_tables(hard, s, d, r, K, nrel):''')
sub('''            w = 1.0 / (1.0 + rr)
''', '''            w = 1.0 / (1.0 + rr)
            _W[0] = w
''')
sub('''    log("identities: " + json.dumps(ident))
''', '''    log("identities: " + json.dumps(ident))
    prev = np.load(os.path.join(OUT, "blockrelroute_%s__v1.npz" % ds))
    pj = json.load(open(os.path.join(OUT, "blockrelroute_%s__v1.json" % ds), encoding="utf-8"))
    assert (prev["rows"][:nq] == pop.rows).all()
    nbad = 0
    for c in CK:
        pa = np.unpackbits(prev["ALL__%d|%s" % c], axis=2)[:, :, :NM].astype(bool)
        for nm in ("SHIPPED", "Sro+Sri+Tin", "Sro+Sri+BRin+BM"):
            nbad += int((pa[pj["rules"].index(nm)][:nq] != ALL[(c[0], c[1])][names.index(nm)]).sum())
        nbad += int((pa[pj["rules"].index("Sro+Sri+Tin+BM")][:nq] != ALL[(c[0], c[1])][names.index("Sro+Sri+Tin+BM")]).sum())
    log("identity vs the step-12 record: %d mismatching cells" % nbad)
    assert nbad == 0, "the re-implemented step-12 arms differ from blockrelroute_%s__v1" % ds
''')
sub('''            if len(Hs) > 1:
                e["hopA100"] = [round(float(ALL[c][ri, A & (hops == h), 0].mean()), 4) for h in Hs]
                e["hopB100"] = [round(float(ALL[c][ri, B & (hops == h), 0].mean()), 4) for h in Hs]
''', '''            if len(Hs) > 1:
                e["hopA"] = {str(h): [round(float(ALL[c][ri, A & (hops == h), mi].mean()), 4) for mi in range(NM)] for h in Hs}
                e["hopB"] = {str(h): [round(float(ALL[c][ri, B & (hops == h), mi].mean()), 4) for mi in range(NM)] for h in Hs}
''')
sub('"blockrelroute_%s__%s"', '"colloc16_%s__%s"')
sub('"blockrelroute_%s__%s.json"', '"colloc16_%s__%s.json"')
sub('"L1X_BLOCK_RELATION_SUPPORT (development; descriptive; nothing chosen)"', '"L1X_COLLOC_LADDER (development; descriptive; nothing chosen)"')
sub("_l1x_blockrel_route.py RUN", "_l1x_colloc16.py RUN")
sub("_l1x_blockrel_route.py SHOW", "_l1x_colloc16.py SHOW")
io.open("scratchpad/_l1x_colloc16.py", "w", encoding="utf-8", newline="\n").write(src)
print("written", len(src))
