# L1 CAPACITY + TRANSFORMATION EVICTION -- TABLES

Every number below is read straight out of `diag/`; nothing is retyped by hand.

## PART A1 -- candidate oracle vs candidate depth (append-only, nothing evicted)

```
        corpus/slice  K(q)    K0+0    K0+4    K0+8   K0+16   K0+32    delta
---------------------------------------------------------------------------
         metaqa/hop1  49.2  0.9970  0.9970  0.9970  0.9970  0.9970  +0.0000
         metaqa/hop2  49.2  0.8033  0.8183  0.8273  0.8498  0.8739  +0.0706
         metaqa/hop3  49.2  0.3634  0.3664  0.3679  0.3889  0.4204  +0.0570
          metaqa/ALL  49.2  0.7212  0.7272  0.7307  0.7452  0.7638  +0.0426
          webqsp/ALL  34.4  0.7886  0.7942  0.7963  0.8034  0.8104  +0.0218
     2wiki_clean/ALL  41.8  0.9515  0.9520  0.9520  0.9535  0.9565  +0.0050
   musique_clean/ALL  18.6  0.9750  0.9765  0.9780  0.9805  0.9860  +0.0110
  hotpotqa_clean/ALL  55.4  0.9645  0.9650  0.9650  0.9680  0.9685  +0.0040
     squad_clean/ALL  15.8  0.9945  0.9945  0.9945  0.9945  0.9955  +0.0010
```

## PART A2 -- do the ACTUAL selectors convert it?  exact P50, net vs frozen (K0+0 / F6)

```
        corpus/slice  sel  K0+0  K0+4  K0+8  K0+16  K0+32  capacity-only
------------------------------------------------------------------------
         metaqa/hop1   F6    +0    +0    +0     +0     +0             +0
         metaqa/hop1   G4    +0    +0    +0     +0     +0             +0
         metaqa/hop2   F6    +0    +0    +0     +0     +0             +0
         metaqa/hop2   G4    -5    -5    -6     -6     -6             -1
         metaqa/hop3   F6    +0    +0    +0     +0     +0             +0
         metaqa/hop3   G4  +13*  +13*  +13*   +13*   +13*             +0
          metaqa/ALL   F6    +0    +0    +0     +0     +0             +0
          metaqa/ALL   G4    +8    +8    +7     +7     +7             -1
          webqsp/ALL   F6    +0    +0    +0     +0     +0             +0
          webqsp/ALL   G4    -2    -2    -4     -5     -5             -3
     2wiki_clean/ALL   F6    +0    +0    +0     +0     +0             +0
     2wiki_clean/ALL   G4    +0    -1    -1     -1     -1             -1
   musique_clean/ALL   F6    +0    +0    +0     +0     +0             +0
   musique_clean/ALL   G4   -7*   -9*  -11*   -11*   -11*             -4
  hotpotqa_clean/ALL   F6    +0    +0    +0     +0     +0             +0
  hotpotqa_clean/ALL   G4  -11*  -24*  -28*   -31*   -31*            -20
     squad_clean/ALL   F6    +0    +0    +0     +0     +0             +0
     squad_clean/ALL   G4    -1    -3    -3     -3     -3             -2
```

`capacity-only` holds the SELECTOR fixed and moves only the depth, so it isolates what the
extra candidates are worth.  `*` = significant (McNemar, vs the frozen system).

## PART A -- why F6 converts exactly zero (algebraic closure, all six corpora)

```
          corpus  appended  with spos|rpos  could enter top6  max appended score  min 6th pool score
----------------------------------------------------------------------------------------------------
          metaqa     62776               0                 0            0.009091            0.009346
          webqsp     42121               0                 0            0.009091            0.009259
     2wiki_clean     42824               0                 0            0.009091            0.009524
   musique_clean     59892               0                 0            0.009091            0.009174
  hotpotqa_clean     63214               0                 0            0.009091            0.009615
     squad_clean     53545               0                 0            0.009091            0.009174
```

The SAFE pool is `bnd ++ chal` and `chal` is DEFINED as everything carrying struct or ret
evidence outside base50, so an appendable partition has neither and scores at most
`1/(K0+cpos) <= 1/110 = 0.009091`, while all six `bnd` members score at least
`1/(60+49) = 0.009174`.  The bound is tight and holds exactly on every corpus.

## STEP 1 / 2 / 6 -- operator gates on real replayed edges

```
              family  max|T(u)-v|  ||Tz||/||z|| on random unit z  chain==manual  scalar==vector
-----------------------------------------------------------------------------------------------
      F0_TRANSLATION     1.49e-08           1.570656 +- 0.070539       0.00e+00        1.19e-06
      F1_HOUSEHOLDER     2.20e-07           1.000000 +- 0.000000       0.00e+00        3.23e-07
     F2_MIN_ROTATION     7.45e-08           1.000000 +- 0.000000       0.00e+00        3.25e-07
  F3_RANK1_TRANSPORT     1.49e-08           1.000004 +- 0.000619       0.00e+00        3.76e-07
```

F2 identity on span(u,v)-perp 5.59e-09; F1 identity on (u-v)-perp 5.59e-09; F1 involution 7.45e-09; 18511 real edges.

## STEP 3 + STEP 5 -- edge-level AUC (missing-required target vs non-required target)

```
          corpus  state   NO_TF  F0_TRANS 1hop  F1_HOUSE 1hop  F2_ROT 1hop  F3_RANK1 1hop  F0_TRANS chain  F1_HOUSE chain  F2_ROT chain  F3_RANK1 chain
-------------------------------------------------------------------------------------------------------------------------------------------------------
          metaqa  q_raw  0.6663         0.6051         0.6245       0.6245         0.6602          0.6880          0.6917        0.6917          0.6954
          metaqa    r_q  0.5866         0.5204         0.5200       0.5200         0.5740          0.5023          0.4938        0.5086          0.5728
          webqsp  q_raw  0.5367         0.5887         0.6102       0.6102         0.5867          0.5546          0.4729        0.4729          0.6271
          webqsp    r_q  0.6390         0.6467         0.7140       0.7140         0.6833          0.6264          0.4663        0.4542          0.7150
     2wiki_clean  q_raw  0.6172         0.5390         0.4766       0.4767         0.5756          0.5320          0.5069        0.5068          0.5489
     2wiki_clean    r_q  0.5546         0.4893         0.3838       0.3838         0.4950          0.4544          0.4186        0.4033          0.5115
   musique_clean  q_raw  0.4914         0.5120         0.4917       0.4917         0.4883          0.4414          0.3927        0.3927          0.4781
   musique_clean    r_q  0.5813         0.5925         0.6254       0.6254         0.6150          0.5638          0.4308        0.4700          0.6257
  hotpotqa_clean  q_raw  0.5696         0.3565         0.5277       0.5277         0.4962          0.3686          0.4342        0.4340          0.3891
  hotpotqa_clean    r_q  0.4351         0.3329         0.4949       0.4949         0.4451          0.3793          0.5978        0.5600          0.4651
     squad_clean  q_raw  0.5759         0.5802         0.4279       0.4279         0.4995          0.4364          0.1725        0.1726          0.3579
     squad_clean    r_q  0.7340         0.6598         0.4889       0.4889         0.6320          0.7453          0.4171        0.6199          0.5961
```

`NO_TF` is the identity control -- `cos(z0, x_v)` on the SAME edge population.  Any cell that
does not beat it is a transformation that destroys information rather than adding it.

## STEP 3 -- marginal target-partition rank of missing-required (percentile, lower better)

```
          corpus  state   NO_TF  F0_TRANS 1hop  F1_HOUSE 1hop  F2_ROT 1hop  F3_RANK1 1hop  F0_TRANS chain  F1_HOUSE chain  F2_ROT chain  F3_RANK1 chain
-------------------------------------------------------------------------------------------------------------------------------------------------------
          metaqa  q_raw  0.4125         0.4464         0.3867       0.3858         0.4177          0.3810          0.3734        0.3677          0.3985
          metaqa    r_q                 0.4641         0.4751       0.4706         0.4466          0.4232          0.4454        0.4374          0.4480
          webqsp  q_raw  0.3348         0.2319         0.2485       0.2554         0.2370          0.4151          0.4101        0.4302          0.3388
          webqsp    r_q                 0.3387         0.3393       0.3403         0.3470          0.4734          0.5019        0.5007          0.3489
     2wiki_clean  q_raw  0.3764         0.4095         0.3860       0.3915         0.3412          0.4923          0.5782        0.6353          0.3936
     2wiki_clean    r_q                 0.5227         0.5377       0.5721         0.4595          0.6431          0.4756        0.6356          0.4416
   musique_clean  q_raw  0.5093         0.5971         0.5924       0.6127         0.5685          0.5658          0.5537        0.6600          0.5862
   musique_clean    r_q                 0.5985         0.5721       0.5852         0.5904          0.5570          0.6451        0.6091          0.5980
  hotpotqa_clean  q_raw  0.1842         0.4225         0.3986       0.3987         0.3987          0.4755          0.4443        0.4881          0.4284
  hotpotqa_clean    r_q                 0.5307         0.3332       0.3425         0.3274          0.4887          0.3897        0.4019          0.3121
     squad_clean  q_raw  0.5757         0.5689         0.6469       0.6443         0.5486          0.4070          0.4267        0.4465          0.4784
     squad_clean    r_q                 0.4331         0.5644       0.5483         0.3971          0.3868          0.3565        0.5265          0.4190
```

## STEP 6 -- composition stability (chain, q_raw)

```
          corpus              family  |z1|/|z0|  |z2|/|z0|  |z3|/|z0|  cos(z1,z0)  cos(z2,z0)  cos(z3,z0)
---------------------------------------------------------------------------------------------------------
          metaqa      F0_TRANSLATION      1.240      1.262      1.335       0.692       0.625       0.570
          metaqa      F1_HOUSEHOLDER      1.000      1.000      1.000       0.925       0.890       0.865
          metaqa     F2_MIN_ROTATION      1.000      1.000      1.000       0.855       0.802       0.769
          metaqa  F3_RANK1_TRANSPORT      1.048      1.068      1.109       0.873       0.812       0.769
          webqsp      F0_TRANSLATION      1.187      1.198      1.263       0.734       0.650       0.538
          webqsp      F1_HOUSEHOLDER      1.000      1.000      1.000       0.923       0.863       0.800
          webqsp     F2_MIN_ROTATION      1.000      1.000      0.999       0.861       0.808       0.742
          webqsp  F3_RANK1_TRANSPORT      1.045      1.041      1.059       0.877       0.816       0.742
     2wiki_clean      F0_TRANSLATION      1.295      1.294      1.377       0.640       0.628       0.528
     2wiki_clean      F1_HOUSEHOLDER      1.000      1.000      1.000       0.922       0.898       0.861
     2wiki_clean     F2_MIN_ROTATION      1.000      1.000      1.000       0.842       0.824       0.767
     2wiki_clean  F3_RANK1_TRANSPORT      1.049      1.070      1.109       0.861       0.830       0.761
   musique_clean      F0_TRANSLATION      1.381      1.394      1.426       0.572       0.548       0.504
   musique_clean      F1_HOUSEHOLDER      1.000      1.000      1.000       0.919       0.902       0.872
   musique_clean     F2_MIN_ROTATION      1.000      1.000      1.000       0.824       0.792       0.757
   musique_clean  F3_RANK1_TRANSPORT      1.053      1.091      1.132       0.848       0.803       0.748
  hotpotqa_clean      F0_TRANSLATION      1.269      1.299      1.321       0.625       0.551       0.496
  hotpotqa_clean      F1_HOUSEHOLDER      1.000      1.000      1.000       0.890       0.836       0.783
  hotpotqa_clean     F2_MIN_ROTATION      1.000      1.000      1.000       0.803       0.759       0.716
  hotpotqa_clean  F3_RANK1_TRANSPORT      1.055      1.068      1.093       0.827       0.767       0.708
     squad_clean      F0_TRANSLATION      1.368      1.373      1.401       0.542       0.520       0.477
     squad_clean      F1_HOUSEHOLDER      1.000      0.980      0.980       0.889       0.864       0.834
     squad_clean     F2_MIN_ROTATION      1.000      0.980      0.980       0.806       0.758       0.721
     squad_clean  F3_RANK1_TRANSPORT      1.046      1.063      1.101       0.826       0.769       0.713
```

## STEP 7 -- MISSING_REQUIRED_ADMISSION_RECALL inside FULL_VISITED

**metaqa** (467 queries with a missing-required partition)

```
                    ranker over FULL_VISITED      @6     @12     @20     @50
----------------------------------------------------------------------------
                                 FROZEN_SAFE  0.0000  0.0677  0.1313  0.3222
                                  SRC_ASSIGN  0.0853  0.1389  0.1931  0.4158
                                  OFFSET_RAW  0.0629  0.1110  0.1619  0.3191
                              V0_NOTRANSFORM  0.0311  0.0754  0.1272  0.2844
  BEST TF: q_raw/F2_MIN_ROTATION/V4_SRC_EXIT  0.0801  0.1317  0.1884  0.3545
  TRANSLATION: q_raw/F0_TRANSLATION/V2_CHAIN  0.0748  0.1299  0.1753  0.3101
```

**webqsp** (132 queries with a missing-required partition)

```
                    ranker over FULL_VISITED      @6     @12     @20     @50
----------------------------------------------------------------------------
                                 FROZEN_SAFE  0.0000  0.0674  0.1381  0.3077
                                  SRC_ASSIGN  0.0695  0.1199  0.1887  0.3097
                                  OFFSET_RAW  0.0188  0.0323  0.0775  0.2288
                              V0_NOTRANSFORM  0.0642  0.0931  0.1330  0.2596
     BEST TF: q_raw/F0_TRANSLATION/V1_SINGLE  0.1676  0.2185  0.2751  0.3865
  TRANSLATION: q_raw/F0_TRANSLATION/V2_CHAIN  0.0509  0.0577  0.0933  0.1991
```

**2wiki_clean** (22 queries with a missing-required partition)

```
                    ranker over FULL_VISITED      @6     @12     @20     @50
----------------------------------------------------------------------------
                                 FROZEN_SAFE  0.0000  0.1818  0.2045  0.4318
                                  SRC_ASSIGN  0.0455  0.0909  0.1591  0.3864
                                  OFFSET_RAW  0.0000  0.0000  0.1136  0.1591
                              V0_NOTRANSFORM  0.0909  0.1136  0.1591  0.2955
     BEST TF: r_q/F1_HOUSEHOLDER/V4_SRC_EXIT  0.1818  0.2273  0.3182  0.3864
  TRANSLATION: q_raw/F0_TRANSLATION/V2_CHAIN  0.0455  0.0455  0.0909  0.1364
```

**musique_clean** (26 queries with a missing-required partition)

```
                    ranker over FULL_VISITED      @6     @12     @20     @50
----------------------------------------------------------------------------
                                 FROZEN_SAFE  0.0000  0.2308  0.6154  0.8462
                                  SRC_ASSIGN  0.0769  0.1538  0.2692  0.8077
                                  OFFSET_RAW  0.0769  0.1154  0.3846  0.8462
                              V0_NOTRANSFORM  0.0385  0.1538  0.2308  0.8846
   BEST TF: q_raw/F1_HOUSEHOLDER/V4_SRC_EXIT  0.1154  0.1538  0.2692  0.5769
  TRANSLATION: q_raw/F0_TRANSLATION/V2_CHAIN  0.0769  0.1154  0.2308  0.6538
```

**hotpotqa_clean** (30 queries with a missing-required partition)

```
                    ranker over FULL_VISITED      @6     @12     @20     @50
----------------------------------------------------------------------------
                                 FROZEN_SAFE  0.0000  0.3500  0.4833  0.5500
                                  SRC_ASSIGN  0.0333  0.0333  0.0833  0.3167
                                  OFFSET_RAW  0.0000  0.0000  0.0000  0.0333
                              V0_NOTRANSFORM  0.1333  0.1333  0.1500  0.3167
     BEST TF: q_raw/F0_TRANSLATION/V1_SINGLE  0.1500  0.1500  0.2167  0.2167
  TRANSLATION: q_raw/F0_TRANSLATION/V2_CHAIN  0.0333  0.0333  0.0667  0.1000
```

**squad_clean** (10 queries with a missing-required partition)

```
                    ranker over FULL_VISITED      @6     @12     @20     @50
----------------------------------------------------------------------------
                                 FROZEN_SAFE  0.0000  0.5000  1.0000  1.0000
                                  SRC_ASSIGN  0.0000  0.0000  0.0000  0.1000
                                  OFFSET_RAW  0.0000  0.0000  0.0000  0.2000
                              V0_NOTRANSFORM  0.1000  0.1000  0.1000  0.1000
       BEST TF: r_q/F0_TRANSLATION/V1_SINGLE  0.1000  0.1000  0.1000  0.1000
  TRANSLATION: q_raw/F0_TRANSLATION/V2_CHAIN  0.0000  0.0000  0.0000  0.1000
```

## STEP 8 -- multi-state coverage: the four state views, not collapsed to one scalar

`T_SINGLE` = V1_SINGLE, one transformation applied to q0 with no composition.  `T_ASSIGN` =
V2_CHAIN, every partition taking its OWN best transported state.  `T_MAX_VIEW` = V3_MAXVIEW, the
union view that also contains the untransformed q0.  `V4_SRC_EXIT` restricts the view to edges
leaving the protected core.  Cells are STEP-7 recall@6 / @50 at each corpus's own best (state,
family), so this is the most favourable reading available to the transformation side.

```
          corpus      best state/family    T_SINGLE (V1)    T_ASSIGN (V2)  T_MAX_VIEW (V3)      V4_SRC_EXIT  SRC_ASSIGN (no transform)
--------------------------------------------------------------------------------------------------------------------------------------
          metaqa  q_raw/F2_MIN_ROTATION  0.0737 / 0.3608  0.0743 / 0.3489  0.0743 / 0.3489  0.0801 / 0.3545            0.0853 / 0.4158
          webqsp   q_raw/F0_TRANSLATION  0.1676 / 0.3865  0.0509 / 0.1991  0.0509 / 0.1991  0.0597 / 0.2566            0.0695 / 0.3097
     2wiki_clean     r_q/F1_HOUSEHOLDER  0.0909 / 0.1818  0.1364 / 0.3409  0.0909 / 0.2955  0.1818 / 0.3864            0.0455 / 0.3864
   musique_clean   q_raw/F1_HOUSEHOLDER  0.0385 / 0.5385  0.0769 / 0.5385  0.0769 / 0.5385  0.1154 / 0.5769            0.0769 / 0.8077
  hotpotqa_clean   q_raw/F0_TRANSLATION  0.1500 / 0.2167  0.0333 / 0.1000  0.0333 / 0.1000  0.0333 / 0.1333            0.0333 / 0.3167
     squad_clean     r_q/F0_TRANSLATION  0.1000 / 0.1000  0.0000 / 0.1000  0.0000 / 0.1000  0.0000 / 0.2000            0.0000 / 0.1000
```

## PART B4 -- forced-swap margin  score(admitted SRC candidate) - score(evicted SAFE candidate)

```
          corpus  GOOD_SWAP  BAD_SWAP  unlabelled  all forced pairs
-------------------------------------------------------------------
          metaqa      27120     21940     1741882           1790942
          webqsp       2826      2597      768499            773922
     2wiki_clean        268       560     1003572           1004400
   musique_clean         65       417      185464            185946
  hotpotqa_clean        384      2116     2911384           2913884
     squad_clean          0       300      183518            183818
```

A corpus with an empty class has nothing for any score to separate.

```
                                 score  metaqa  webqsp  2wiki_clean  musique_clean  hotpotqa_clean  squad_clean  sign consistent
--------------------------------------------------------------------------------------------------------------------------------
           r_q/F0_TRANSLATION/V2_CHAIN  0.5763  0.3498       0.2424         0.4516          0.2639          n/a               no
         r_q/F0_TRANSLATION/V3_MAXVIEW  0.5860  0.3673       0.2407         0.4516          0.3862          n/a               no
          r_q/F0_TRANSLATION/V1_SINGLE  0.5374  0.3997       0.2365         0.3947          0.3740          n/a               no
      r_q/F3_RANK1_TRANSPORT/V1_SINGLE  0.5558  0.4299       0.3043         0.3267          0.4276          n/a               no
         q_raw/F0_TRANSLATION/V2_CHAIN  0.6785  0.4248       0.4057         0.4573          0.3363          n/a               no
       q_raw/F0_TRANSLATION/V3_MAXVIEW  0.6785  0.4248       0.4057         0.4573          0.3363          n/a               no
          r_q/F2_MIN_ROTATION/V2_CHAIN  0.5669  0.4179       0.3188         0.3757          0.4037          n/a               no
      q_raw/F0_TRANSLATION/V4_SRC_EXIT  0.6223  0.6085       0.6248         0.5872          0.4433          n/a               no
        r_q/F0_TRANSLATION/V4_SRC_EXIT  0.5268  0.5644       0.3446         0.5766          0.3246          n/a               no
       r_q/F3_RANK1_TRANSPORT/V2_CHAIN  0.5492  0.4497       0.3392         0.3460          0.4389          n/a               no
      q_raw/F1_HOUSEHOLDER/V4_SRC_EXIT  0.6429  0.6056       0.6620         0.5175          0.4882          n/a               no
     q_raw/F2_MIN_ROTATION/V4_SRC_EXIT  0.6441  0.6040       0.6618         0.5072          0.4883          n/a               no
        q_raw/F2_MIN_ROTATION/V2_CHAIN  0.6893  0.4571       0.4762         0.4683          0.3646          n/a               no
  q_raw/F3_RANK1_TRANSPORT/V4_SRC_EXIT  0.6207  0.6467       0.6269         0.4913          0.4822          n/a               no
      q_raw/F2_MIN_ROTATION/V3_MAXVIEW  0.6892  0.4594       0.4762         0.4683          0.3658          n/a               no
    r_q/F3_RANK1_TRANSPORT/V4_SRC_EXIT  0.5329  0.6102       0.2841         0.4812          0.4633          n/a               no
       r_q/F2_MIN_ROTATION/V4_SRC_EXIT  0.5612  0.5618       0.3524         0.4333          0.4451          n/a               no
         q_raw/F1_HOUSEHOLDER/V2_CHAIN  0.6900  0.4599       0.5197         0.5032          0.3675          n/a               no
     q_raw/F3_RANK1_TRANSPORT/V2_CHAIN  0.6622  0.5059       0.4857         0.3945          0.4027          n/a               no
   q_raw/F3_RANK1_TRANSPORT/V3_MAXVIEW  0.6622  0.5059       0.4857         0.3945          0.4028          n/a               no
       q_raw/F1_HOUSEHOLDER/V3_MAXVIEW  0.6899  0.4632       0.5197         0.5032          0.3688          n/a               no
        r_q/F1_HOUSEHOLDER/V4_SRC_EXIT  0.5683  0.5475       0.6608         0.5763          0.4925          n/a               no
        q_raw/F0_TRANSLATION/V1_SINGLE  0.5862  0.5092       0.3356         0.5393          0.5045          n/a               no
           r_q/F1_HOUSEHOLDER/V2_CHAIN  0.5756  0.4023       0.5277         0.5053          0.4050          n/a               no
       q_raw/F2_MIN_ROTATION/V1_SINGLE  0.6420  0.5423       0.4837         0.4805          0.4629          n/a               no
        q_raw/F1_HOUSEHOLDER/V1_SINGLE  0.6420  0.5423       0.4837         0.4816          0.4628          n/a               no
          r_q/F1_HOUSEHOLDER/V1_SINGLE  0.5357  0.4925       0.4299         0.4334          0.4341          n/a               no
     r_q/F3_RANK1_TRANSPORT/V3_MAXVIEW  0.5898  0.4517       0.4759         0.4749          0.5552          n/a               no
         r_q/F2_MIN_ROTATION/V1_SINGLE  0.5355  0.4924       0.4304         0.4399          0.4342          n/a               no
                        V0_NOTRANSFORM  0.5898  0.4504       0.4755         0.5161          0.5560          n/a               no
         r_q/F1_HOUSEHOLDER/V3_MAXVIEW  0.5898  0.4504       0.4755         0.5161          0.5560          n/a               no
        r_q/F2_MIN_ROTATION/V3_MAXVIEW  0.5898  0.4504       0.4755         0.5161          0.5560          n/a               no
    q_raw/F3_RANK1_TRANSPORT/V1_SINGLE  0.6226  0.5287       0.4792         0.5102          0.5128          n/a               no
```

`sign consistent` = the score points the SAME way (all AUC above 0.5, or all below) on every
corpus where both classes exist.  A score that inverts is not a usable eviction signal.

```
      score (BOTH-evidence pairs only)  metaqa  webqsp  2wiki_clean  musique_clean  hotpotqa_clean  squad_clean  sign consistent
--------------------------------------------------------------------------------------------------------------------------------
                        V0_NOTRANSFORM  0.5821  0.6983       0.6448         0.5081          0.8638          n/a              YES
    q_raw/F3_RANK1_TRANSPORT/V1_SINGLE  0.6199  0.8652       0.6791         0.5025          0.8032          n/a              YES
         r_q/F1_HOUSEHOLDER/V3_MAXVIEW  0.5821  0.6983       0.6448         0.5081          0.8638          n/a              YES
        r_q/F1_HOUSEHOLDER/V4_SRC_EXIT  0.5552  0.5465       0.8049         0.5168          0.5342          n/a              YES
        r_q/F2_MIN_ROTATION/V3_MAXVIEW  0.5821  0.6983       0.6448         0.5081          0.8638          n/a              YES
           r_q/F1_HOUSEHOLDER/V2_CHAIN  0.5659  0.6119       0.7132         0.4930          0.5376          n/a               no
        q_raw/F1_HOUSEHOLDER/V1_SINGLE  0.6419  0.8977       0.7064         0.4694          0.7247          n/a               no
       q_raw/F2_MIN_ROTATION/V1_SINGLE  0.6418  0.8976       0.7065         0.4681          0.7249          n/a               no
       q_raw/F1_HOUSEHOLDER/V3_MAXVIEW  0.6961  0.7347       0.7150         0.4944          0.4663          n/a               no
         q_raw/F1_HOUSEHOLDER/V2_CHAIN  0.6962  0.7285       0.7150         0.4944          0.4631          n/a               no
     r_q/F3_RANK1_TRANSPORT/V3_MAXVIEW  0.5821  0.7008       0.6453         0.4606          0.8614          n/a               no
        q_raw/F2_MIN_ROTATION/V2_CHAIN  0.6954  0.7221       0.6105         0.4530          0.4551          n/a               no
      q_raw/F2_MIN_ROTATION/V3_MAXVIEW  0.6953  0.7262       0.6105         0.4530          0.4582          n/a               no
      q_raw/F1_HOUSEHOLDER/V4_SRC_EXIT  0.6749  0.6759       0.6307         0.4485          0.4910          n/a               no
     q_raw/F2_MIN_ROTATION/V4_SRC_EXIT  0.6770  0.6506       0.6266         0.4340          0.4928          n/a               no
         r_q/F2_MIN_ROTATION/V1_SINGLE  0.5204  0.7948       0.5913         0.4206          0.6431          n/a               no
  q_raw/F3_RANK1_TRANSPORT/V4_SRC_EXIT  0.6433  0.6807       0.5118         0.4136          0.5878          n/a               no
          r_q/F1_HOUSEHOLDER/V1_SINGLE  0.5207  0.7949       0.5903         0.4128          0.6428          n/a               no
      q_raw/F0_TRANSLATION/V4_SRC_EXIT  0.6425  0.5258       0.5475         0.5458          0.4065          n/a               no
         q_raw/F0_TRANSLATION/V2_CHAIN  0.6832  0.6439       0.4632         0.4395          0.3898          n/a               no
       q_raw/F0_TRANSLATION/V3_MAXVIEW  0.6832  0.6439       0.4632         0.4395          0.3898          n/a               no
        q_raw/F0_TRANSLATION/V1_SINGLE  0.5783  0.8223       0.3880         0.5367          0.7653          n/a               no
     q_raw/F3_RANK1_TRANSPORT/V2_CHAIN  0.6648  0.8163       0.6701         0.3657          0.5549          n/a               no
   q_raw/F3_RANK1_TRANSPORT/V3_MAXVIEW  0.6648  0.8163       0.6701         0.3657          0.5550          n/a               no
       r_q/F2_MIN_ROTATION/V4_SRC_EXIT  0.5395  0.5568       0.4757         0.3184          0.5108          n/a               no
       r_q/F3_RANK1_TRANSPORT/V2_CHAIN  0.5360  0.6958       0.4043         0.3077          0.6743          n/a               no
          r_q/F2_MIN_ROTATION/V2_CHAIN  0.5559  0.6412       0.2921         0.3404          0.5490          n/a               no
      r_q/F3_RANK1_TRANSPORT/V1_SINGLE  0.5435  0.6542       0.3203         0.2853          0.6432          n/a               no
        r_q/F0_TRANSLATION/V4_SRC_EXIT  0.4875  0.4612       0.2602         0.5238          0.2339          n/a               no
    r_q/F3_RANK1_TRANSPORT/V4_SRC_EXIT  0.5031  0.6041       0.2301         0.3967          0.6242          n/a               no
          r_q/F0_TRANSLATION/V1_SINGLE  0.5225  0.5904       0.1735         0.3658          0.4655          n/a               no
           r_q/F0_TRANSLATION/V2_CHAIN  0.5667  0.4837       0.1689         0.4319          0.2752          n/a               no
         r_q/F0_TRANSLATION/V3_MAXVIEW  0.5778  0.5143       0.1654         0.4319          0.5759          n/a               no
```
```
          corpus  GOOD/BAD pairs with evidence on both sides
------------------------------------------------------------
          metaqa                                 24861/20905
          webqsp                                   2361/1398
     2wiki_clean                                     200/312
   musique_clean                                      59/387
  hotpotqa_clean                                    284/1047
     squad_clean                                       0/166
```

## STEPS 9-10 / B5-B6 -- fixed-K transformation-driven REPLACEMENT at exact P50

```
        corpus/slice                      score / selector  frozen     acc  net  gain/loss        p
---------------------------------------------------------------------------------------------------
         metaqa/hop1  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/F6  0.9955  0.9955   +0      +0/-0   1.0000
         metaqa/hop1  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/G4  0.9955  0.9955   +0      +0/-0   1.0000
         metaqa/hop1      q_raw/F1_HOUSEHOLDER/V2_CHAIN/F6  0.9955  0.9955   +0      +0/-0   1.0000
         metaqa/hop1      q_raw/F1_HOUSEHOLDER/V2_CHAIN/G4  0.9955  0.9955   +0      +0/-0   1.0000
         metaqa/hop1      q_raw/F0_TRANSLATION/V2_CHAIN/F6  0.9955  0.9955   +0      +0/-0   1.0000
         metaqa/hop1      q_raw/F0_TRANSLATION/V2_CHAIN/G4  0.9955  0.9955   +0      +0/-0   1.0000
         metaqa/hop1                    V6_EXIT_SRC_SIM/F6  0.9955  0.9955   +0      +0/-0   1.0000
         metaqa/hop1                    V6_EXIT_SRC_SIM/G4  0.9955  0.9955   +0      +0/-0   1.0000
         metaqa/hop1                     V0_NOTRANSFORM/F6  0.9955  0.9955   +0      +0/-0   1.0000
         metaqa/hop1                     V0_NOTRANSFORM/G4  0.9955  0.9955   +0      +0/-0   1.0000
         metaqa/hop2  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/F6  0.7297  0.7417   +8      +8/-0  0.0078*
         metaqa/hop2  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/G4  0.7297  0.7267   -2     +8/-10   0.8145
         metaqa/hop2      q_raw/F1_HOUSEHOLDER/V2_CHAIN/F6  0.7297  0.7387   +6      +7/-1   0.0703
         metaqa/hop2      q_raw/F1_HOUSEHOLDER/V2_CHAIN/G4  0.7297  0.7297   +0      +8/-8   1.0000
         metaqa/hop2      q_raw/F0_TRANSLATION/V2_CHAIN/F6  0.7297  0.7357   +4      +7/-3   0.3438
         metaqa/hop2      q_raw/F0_TRANSLATION/V2_CHAIN/G4  0.7297  0.7252   -3     +7/-10   0.6291
         metaqa/hop2                    V6_EXIT_SRC_SIM/F6  0.7297  0.7387   +6      +9/-3   0.1460
         metaqa/hop2                    V6_EXIT_SRC_SIM/G4  0.7297  0.7237   -4     +7/-11   0.4807
         metaqa/hop2                     V0_NOTRANSFORM/F6  0.7297  0.7387   +6      +8/-2   0.1094
         metaqa/hop2                     V0_NOTRANSFORM/G4  0.7297  0.7222   -5     +5/-10   0.3018
         metaqa/hop3  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/F6  0.2583  0.2643   +4      +5/-1   0.2188
         metaqa/hop3  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/G4  0.2583  0.2778  +13     +16/-3  0.0044*
         metaqa/hop3      q_raw/F1_HOUSEHOLDER/V2_CHAIN/F6  0.2583  0.2628   +3      +4/-1   0.3750
         metaqa/hop3      q_raw/F1_HOUSEHOLDER/V2_CHAIN/G4  0.2583  0.2763  +12     +15/-3  0.0075*
         metaqa/hop3      q_raw/F0_TRANSLATION/V2_CHAIN/F6  0.2583  0.2628   +3      +4/-1   0.3750
         metaqa/hop3      q_raw/F0_TRANSLATION/V2_CHAIN/G4  0.2583  0.2748  +11     +14/-3  0.0127*
         metaqa/hop3                    V6_EXIT_SRC_SIM/F6  0.2583  0.2643   +4      +4/-0   0.1250
         metaqa/hop3                    V6_EXIT_SRC_SIM/G4  0.2583  0.2823  +16     +19/-3  0.0009*
         metaqa/hop3                     V0_NOTRANSFORM/F6  0.2583  0.2628   +3      +5/-2   0.4531
         metaqa/hop3                     V0_NOTRANSFORM/G4  0.2583  0.2748  +11     +15/-4  0.0192*
          metaqa/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/F6  0.6612  0.6672  +12     +13/-1  0.0018*
          metaqa/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/G4  0.6612  0.6667  +11    +24/-13   0.0989
          metaqa/ALL      q_raw/F1_HOUSEHOLDER/V2_CHAIN/F6  0.6612  0.6657   +9     +11/-2  0.0225*
          metaqa/ALL      q_raw/F1_HOUSEHOLDER/V2_CHAIN/G4  0.6612  0.6672  +12    +23/-11   0.0576
          metaqa/ALL      q_raw/F0_TRANSLATION/V2_CHAIN/F6  0.6612  0.6647   +7     +11/-4   0.1185
          metaqa/ALL      q_raw/F0_TRANSLATION/V2_CHAIN/G4  0.6612  0.6652   +8    +21/-13   0.2295
          metaqa/ALL                    V6_EXIT_SRC_SIM/F6  0.6612  0.6662  +10     +13/-3  0.0213*
          metaqa/ALL                    V6_EXIT_SRC_SIM/G4  0.6612  0.6672  +12    +26/-14   0.0807
          metaqa/ALL                     V0_NOTRANSFORM/F6  0.6612  0.6657   +9     +13/-4  0.0490*
          metaqa/ALL                     V0_NOTRANSFORM/G4  0.6612  0.6642   +6    +20/-14   0.3915
          webqsp/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/F6  0.7646  0.7653   +1      +5/-4   1.0000
          webqsp/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/G4  0.7646  0.7625   -3      +3/-6   0.5078
          webqsp/ALL                    V6_EXIT_SRC_SIM/F6  0.7646  0.7660   +2      +5/-3   0.7266
          webqsp/ALL                    V6_EXIT_SRC_SIM/G4  0.7646  0.7639   -1      +4/-5   1.0000
          webqsp/ALL                     V0_NOTRANSFORM/F6  0.7646  0.7639   -1      +3/-4   1.0000
          webqsp/ALL                     V0_NOTRANSFORM/G4  0.7646  0.7632   -2      +3/-5   0.7266
     2wiki_clean/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/F6  0.9435  0.9410   -5      +1/-6   0.1250
     2wiki_clean/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/G4  0.9435  0.9400   -7     +3/-10   0.0923
     2wiki_clean/ALL                    V6_EXIT_SRC_SIM/F6  0.9435  0.9410   -5      +1/-6   0.1250
     2wiki_clean/ALL                    V6_EXIT_SRC_SIM/G4  0.9435  0.9400   -7     +3/-10   0.0923
     2wiki_clean/ALL                     V0_NOTRANSFORM/F6  0.9435  0.9405   -6      +2/-8   0.1094
     2wiki_clean/ALL                     V0_NOTRANSFORM/G4  0.9435  0.9410   -5      +4/-9   0.2668
   musique_clean/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/F6  0.9635  0.9560  -15     +4/-19  0.0026*
   musique_clean/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/G4  0.9635  0.9545  -18     +3/-21  0.0003*
   musique_clean/ALL                    V6_EXIT_SRC_SIM/F6  0.9635  0.9585  -10     +1/-11  0.0063*
   musique_clean/ALL                    V6_EXIT_SRC_SIM/G4  0.9635  0.9565  -14     +2/-16  0.0013*
   musique_clean/ALL                     V0_NOTRANSFORM/F6  0.9635  0.9605   -6      +3/-9   0.1460
   musique_clean/ALL                     V0_NOTRANSFORM/G4  0.9635  0.9590   -9     +3/-12  0.0352*
  hotpotqa_clean/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/F6  0.9505  0.9410  -19     +7/-26  0.0013*
  hotpotqa_clean/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/G4  0.9505  0.9345  -32     +5/-37  0.0000*
  hotpotqa_clean/ALL                    V6_EXIT_SRC_SIM/F6  0.9505  0.9405  -20     +4/-24  0.0002*
  hotpotqa_clean/ALL                    V6_EXIT_SRC_SIM/G4  0.9505  0.9345  -32     +4/-36  0.0000*
  hotpotqa_clean/ALL                     V0_NOTRANSFORM/F6  0.9505  0.9420  -17     +2/-19  0.0002*
  hotpotqa_clean/ALL                     V0_NOTRANSFORM/G4  0.9505  0.9340  -33     +4/-37  0.0000*
     squad_clean/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/F6  0.9875  0.9810  -13     +1/-14  0.0010*
     squad_clean/ALL  q_raw/F2_MIN_ROTATION/V4_SRC_EXIT/G4  0.9875  0.9815  -12     +2/-14  0.0042*
     squad_clean/ALL                     V0_NOTRANSFORM/F6  0.9875  0.9835   -8      +0/-8  0.0078*
     squad_clean/ALL                     V0_NOTRANSFORM/G4  0.9875  0.9840   -7      +1/-8  0.0391*
```

```
          corpus                                                                                                                                         queries whose pool actually changed  transform ms/q
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
          metaqa  {'q_raw/F2_MIN_ROTATION/V4_SRC_EXIT': 1986, 'q_raw/F1_HOUSEHOLDER/V2_CHAIN': 1986, 'q_raw/F0_TRANSLATION/V2_CHAIN': 1986, 'V6_EXIT_SRC_SIM': 1986, 'V0_NOTRANSFORM': 1986}          116.59
          webqsp                                                                                {'q_raw/F2_MIN_ROTATION/V4_SRC_EXIT': 1352, 'V6_EXIT_SRC_SIM': 1351, 'V0_NOTRANSFORM': 1352}           72.86
     2wiki_clean                                                                                {'q_raw/F2_MIN_ROTATION/V4_SRC_EXIT': 1591, 'V6_EXIT_SRC_SIM': 1591, 'V0_NOTRANSFORM': 1591}           27.68
   musique_clean                                                                                {'q_raw/F2_MIN_ROTATION/V4_SRC_EXIT': 1953, 'V6_EXIT_SRC_SIM': 1952, 'V0_NOTRANSFORM': 1953}          118.92
  hotpotqa_clean                                                                                {'q_raw/F2_MIN_ROTATION/V4_SRC_EXIT': 1995, 'V6_EXIT_SRC_SIM': 1995, 'V0_NOTRANSFORM': 1995}           66.11
     squad_clean                                                                                                         {'q_raw/F2_MIN_ROTATION/V4_SRC_EXIT': 1703, 'V0_NOTRANSFORM': 1702}          352.88
```

## STEP 12 / A3 -- cost

```
          corpus  admit K0+0  admit K0+4  admit K0+8  admit K0+16  admit K0+32  1 family x 1 state ms/q  edges/q
----------------------------------------------------------------------------------------------------------------
          metaqa      0.0201      0.0209      0.0233       0.0281       0.0436                    7.974     1386
          webqsp      0.0151      0.0150      0.0150       0.0198       0.0281                    9.089     1506
     2wiki_clean      0.0156      0.0159      0.0166       0.0222       0.0287                    2.899      547
   musique_clean      0.0110      0.0120      0.0131       0.0166       0.0214                   15.089     3251
  hotpotqa_clean      0.0154      0.0161      0.0174       0.0223       0.0305                    8.622     1389
     squad_clean      0.0045      0.0051      0.0053       0.0065       0.0093                   67.439    14123
```

