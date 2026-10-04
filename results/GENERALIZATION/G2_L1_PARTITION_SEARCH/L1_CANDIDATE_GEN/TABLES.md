# L1 CANDIDATE GENERATION -- TABLES

Selector frozen at R0 / B6_S4_F6_Ms64_Mr32 throughout. Every proposal family contributes candidate MEMBERSHIP only; no family's score enters the selector.

## T1 -- corpora and the pool-limited target set (STEP 1)

`pool-limited` = the query is P50-feasible and at least one required gold partition lies outside the current candidate pool (boundary incumbents + S4/ret challengers).

| corpus | npart | nq | pool-limited q | frac | SAFE | current-pool oracle | slice |
|---|---|---|---|---|---|---|---|
| metaqa | 401 | 1998 | 529 | 0.265 | 0.2583 | 0.3634 | hop3 |
| webqsp | 7814 | 1419 | 293 | 0.206 | 0.7646 | 0.7886 | ALL |
| 2wiki_clean | 658 | 2000 | 97 | 0.049 | 0.9435 | 0.9515 | ALL |
| musique_clean | 136 | 2000 | 50 | 0.025 | 0.9635 | 0.9750 | ALL |
| hotpotqa_clean | 5074 | 2000 | 71 | 0.035 | 0.9505 | 0.9645 | ALL |
| squad_clean | 190 | 2000 | 11 | 0.005 | 0.9875 | 0.9945 | ALL |

## T2 -- proposal-set size, and how much of the universe each family can see (STEP 1)

`frac universe @256` = mean proposals actually available at M=256, divided by the number of non-base50 partitions. A family near 1.00 is close to enumerating the corpus and its recall is NOT comparable with a selective family's.

**metaqa** (npart = 401, non-base50 universe = 351)

| family | mean proposals available | frac universe @256 |
|---|---|---|
| A dense-part | 351.0 | 0.729 |
| B splade-part | 351.0 | 0.729 |
| C node-dense | 94.1 | 0.268 |
| D node-splade | 102.7 | 0.293 |
| E S4-struct | 61.1 | 0.174 |
| F ppr-reach | 351.0 | 0.729 |
| G nbr-raw | 65.7 | 0.187 |
| G nbr-norm | 80.3 | 0.229 |
| H frontier | 84.7 | 0.241 |
| I canon-cont | 351.0 | 0.729 |

**webqsp** (npart = 7814, non-base50 universe = 7764)

| family | mean proposals available | frac universe @256 |
|---|---|---|
| A dense-part | 7764.0 | 0.033 |
| B splade-part | 7764.0 | 0.033 |
| C node-dense | 71.0 | 0.009 |
| D node-splade | 55.2 | 0.007 |
| E S4-struct | 50.2 | 0.006 |
| F ppr-reach | 7764.0 | 0.033 |
| G nbr-raw | 63.3 | 0.008 |
| G nbr-norm | 69.4 | 0.009 |
| H frontier | 115.3 | 0.015 |
| I canon-cont | 7764.0 | 0.033 |

## T3 -- ALL_REQUIRED_PARTITIONS_PROPOSED@M, MetaQA by hop (STEP 2, primary metric)

Fraction of pool-limited queries whose ENTIRE missing gold-partition set is proposed. A query is not solved because one partition of a chain was found.

**MetaQA hop1** (pool-limited queries = 2)

| family | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|---|
| A dense-part | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| B splade-part | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| C node-dense | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| D node-splade | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| E S4-struct | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 |
| F ppr-reach | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 1.0000 |
| G nbr-raw | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| G nbr-norm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| H frontier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| I canon-cont | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

**MetaQA hop2** (pool-limited queries = 129)

| family | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|---|
| A dense-part | 0.0078 | 0.0233 | 0.0698 | 0.1008 | 0.2558 | 0.4806 |
| B splade-part | 0.2403 | 0.3411 | 0.4574 | 0.5581 | 0.6977 | 0.7907 |
| C node-dense | 0.0078 | 0.0155 | 0.0388 | 0.0853 | 0.1163 | 0.1163 |
| D node-splade | 0.0000 | 0.0155 | 0.0465 | 0.0775 | 0.1318 | 0.1318 |
| E S4-struct | 0.0000 | 0.0078 | 0.0775 | 0.2171 | 0.2171 | 0.2171 |
| F ppr-reach | 0.0310 | 0.0853 | 0.1240 | 0.1705 | 0.2868 | 0.7054 |
| G nbr-raw | 0.0233 | 0.0853 | 0.1163 | 0.2093 | 0.2171 | 0.2171 |
| G nbr-norm | 0.0310 | 0.0388 | 0.0698 | 0.1550 | 0.2326 | 0.2326 |
| H frontier | 0.0233 | 0.0853 | 0.1240 | 0.2326 | 0.2558 | 0.2558 |
| I canon-cont | 0.0930 | 0.2248 | 0.3643 | 0.4884 | 0.6589 | 0.8140 |

**MetaQA hop3** (pool-limited queries = 398)

| family | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|---|
| A dense-part | 0.0025 | 0.0101 | 0.0226 | 0.0553 | 0.1583 | 0.3643 |
| B splade-part | 0.0427 | 0.0553 | 0.1080 | 0.1859 | 0.3040 | 0.5327 |
| C node-dense | 0.0025 | 0.0050 | 0.0201 | 0.0477 | 0.0779 | 0.0779 |
| D node-splade | 0.0000 | 0.0050 | 0.0126 | 0.0452 | 0.1106 | 0.1106 |
| E S4-struct | 0.0000 | 0.0000 | 0.0025 | 0.0427 | 0.0503 | 0.0503 |
| F ppr-reach | 0.0477 | 0.1106 | 0.1884 | 0.2739 | 0.3970 | 0.6432 |
| G nbr-raw | 0.0528 | 0.0804 | 0.1181 | 0.1834 | 0.2060 | 0.2060 |
| G nbr-norm | 0.0201 | 0.0402 | 0.0678 | 0.1281 | 0.1658 | 0.1658 |
| H frontier | 0.0528 | 0.0804 | 0.1231 | 0.2111 | 0.2789 | 0.2789 |
| I canon-cont | 0.0327 | 0.0528 | 0.0879 | 0.1357 | 0.2638 | 0.5377 |

## T4 -- ALL_REQUIRED_PARTITIONS_PROPOSED@M, WebQSP and the text controls (STEP 2 / STEP 10)

**webqsp** (pool-limited queries = 293)

| family | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|---|
| A dense-part | 0.0341 | 0.0751 | 0.1468 | 0.3106 | 0.4164 | 0.4744 |
| B splade-part | 0.0683 | 0.1263 | 0.2253 | 0.3447 | 0.4949 | 0.5939 |
| C node-dense | 0.0137 | 0.0239 | 0.0444 | 0.0819 | 0.1092 | 0.1126 |
| D node-splade | 0.0307 | 0.0580 | 0.0853 | 0.1195 | 0.1195 | 0.1195 |
| E S4-struct | 0.0068 | 0.0102 | 0.0171 | 0.0444 | 0.0512 | 0.0512 |
| F ppr-reach | 0.0102 | 0.0341 | 0.0887 | 0.1365 | 0.2082 | 0.3208 |
| G nbr-raw | 0.0273 | 0.0512 | 0.0580 | 0.0887 | 0.1058 | 0.1058 |
| G nbr-norm | 0.0102 | 0.0205 | 0.0375 | 0.0717 | 0.0853 | 0.0853 |
| H frontier | 0.0273 | 0.0512 | 0.0580 | 0.0956 | 0.1433 | 0.1570 |
| I canon-cont | 0.0375 | 0.1160 | 0.2116 | 0.3754 | 0.5392 | 0.6485 |

**2wiki_clean** (pool-limited queries = 97)

| family | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|---|
| A dense-part | 0.0206 | 0.1237 | 0.1546 | 0.2887 | 0.3608 | 0.6701 |
| B splade-part | 0.0722 | 0.0928 | 0.1237 | 0.2165 | 0.4433 | 0.7113 |
| C node-dense | 0.0825 | 0.1546 | 0.2165 | 0.3093 | 0.3814 | 0.3814 |
| D node-splade | 0.0928 | 0.1031 | 0.2062 | 0.3711 | 0.4742 | 0.4742 |
| E S4-struct | 0.0000 | 0.0000 | 0.0103 | 0.0515 | 0.0619 | 0.0619 |
| F ppr-reach | 0.0515 | 0.1031 | 0.1856 | 0.2268 | 0.3814 | 0.6289 |
| G nbr-raw | 0.0412 | 0.0722 | 0.1443 | 0.1959 | 0.2165 | 0.2165 |
| G nbr-norm | 0.0722 | 0.1031 | 0.1753 | 0.2474 | 0.3196 | 0.3196 |
| H frontier | 0.0412 | 0.0722 | 0.1443 | 0.2062 | 0.2680 | 0.2680 |
| I canon-cont | 0.0825 | 0.1237 | 0.1856 | 0.2887 | 0.4742 | 0.6907 |

**musique_clean** (pool-limited queries = 50)

| family | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|---|
| A dense-part | 0.1800 | 0.2800 | 0.4600 | 0.7800 | 1.0000 | 1.0000 |
| B splade-part | 0.1800 | 0.2800 | 0.5000 | 0.7800 | 1.0000 | 1.0000 |
| C node-dense | 0.3000 | 0.4400 | 0.5600 | 0.6400 | 0.6400 | 0.6400 |
| D node-splade | 0.1800 | 0.3600 | 0.6200 | 0.7000 | 0.7000 | 0.7000 |
| E S4-struct | 0.0200 | 0.0800 | 0.1400 | 0.1800 | 0.1800 | 0.1800 |
| F ppr-reach | 0.0600 | 0.2000 | 0.4400 | 0.9000 | 1.0000 | 1.0000 |
| G nbr-raw | 0.1600 | 0.2400 | 0.3200 | 0.5000 | 0.5000 | 0.5000 |
| G nbr-norm | 0.1200 | 0.2400 | 0.3400 | 0.5600 | 0.5600 | 0.5600 |
| H frontier | 0.1600 | 0.2400 | 0.3400 | 0.6000 | 0.6000 | 0.6000 |
| I canon-cont | 0.1600 | 0.3800 | 0.5200 | 0.8400 | 1.0000 | 1.0000 |

**hotpotqa_clean** (pool-limited queries = 71)

| family | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|---|
| A dense-part | 0.0282 | 0.0563 | 0.0845 | 0.1268 | 0.3099 | 0.3662 |
| B splade-part | 0.0141 | 0.0141 | 0.0423 | 0.0986 | 0.1549 | 0.2958 |
| C node-dense | 0.0563 | 0.1268 | 0.2254 | 0.3380 | 0.4648 | 0.4648 |
| D node-splade | 0.0000 | 0.0282 | 0.0845 | 0.1972 | 0.2958 | 0.3239 |
| E S4-struct | 0.0000 | 0.0000 | 0.0000 | 0.0563 | 0.0986 | 0.0986 |
| F ppr-reach | 0.0000 | 0.0000 | 0.0423 | 0.1972 | 0.3099 | 0.3239 |
| G nbr-raw | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0141 | 0.0141 |
| G nbr-norm | 0.0282 | 0.0282 | 0.0423 | 0.0704 | 0.1127 | 0.1127 |
| H frontier | 0.0000 | 0.0000 | 0.0000 | 0.0282 | 0.0845 | 0.0845 |
| I canon-cont | 0.0141 | 0.0282 | 0.0845 | 0.1549 | 0.2676 | 0.3944 |

**squad_clean** (pool-limited queries = 11)

| family | M=8 | M=16 | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|---|
| A dense-part | 0.0909 | 0.2727 | 0.3636 | 0.6364 | 1.0000 | 1.0000 |
| B splade-part | 0.0909 | 0.1818 | 0.1818 | 0.3636 | 1.0000 | 1.0000 |
| C node-dense | 0.2727 | 0.3636 | 0.6364 | 0.7273 | 0.7273 | 0.7273 |
| D node-splade | 0.0909 | 0.2727 | 0.7273 | 0.7273 | 0.7273 | 0.7273 |
| E S4-struct | 0.0000 | 0.0909 | 0.0909 | 0.0909 | 0.0909 | 0.0909 |
| F ppr-reach | 0.1818 | 0.2727 | 0.2727 | 0.6364 | 0.8182 | 1.0000 |
| G nbr-raw | 0.0909 | 0.3636 | 0.5455 | 0.6364 | 0.6364 | 0.6364 |
| G nbr-norm | 0.1818 | 0.4545 | 0.4545 | 0.6364 | 0.6364 | 0.6364 |
| H frontier | 0.0909 | 0.3636 | 0.5455 | 0.6364 | 0.6364 | 0.6364 |
| I canon-cont | 0.0000 | 0.1818 | 0.2727 | 0.5455 | 1.0000 | 1.0000 |

## T5 -- ANY vs ALL at M=256 (STEP 2, secondary)

The gap between ANY and ALL is the multi-hop penalty: how often a family finds part of the chain but not all of it.

| corpus | slice | family | ANY | ALL | micro partition recall |
|---|---|---|---|---|---|
| metaqa | hop3 | B splade-part | 0.9749 | 0.5327 | 0.8155 |
| metaqa | hop3 | F ppr-reach | 0.9749 | 0.6432 | 0.8790 |
| metaqa | hop3 | I canon-cont | 0.9598 | 0.5377 | 0.8247 |
| metaqa | hop3 | G nbr-raw | 0.7965 | 0.2060 | 0.3790 |
| webqsp | ALL | B splade-part | 0.8089 | 0.5939 | 0.6528 |
| webqsp | ALL | F ppr-reach | 0.6485 | 0.3208 | 0.4238 |
| webqsp | ALL | I canon-cont | 0.8771 | 0.6485 | 0.6723 |
| webqsp | ALL | G nbr-raw | 0.3481 | 0.1058 | 0.1396 |
| 2wiki_clean | ALL | B splade-part | 0.7113 | 0.7113 | 0.7143 |
| 2wiki_clean | ALL | F ppr-reach | 0.6392 | 0.6289 | 0.6327 |
| 2wiki_clean | ALL | I canon-cont | 0.6907 | 0.6907 | 0.6939 |
| 2wiki_clean | ALL | G nbr-raw | 0.2268 | 0.2165 | 0.2245 |
| musique_clean | ALL | B splade-part | 1.0000 | 1.0000 | 1.0000 |
| musique_clean | ALL | F ppr-reach | 1.0000 | 1.0000 | 1.0000 |
| musique_clean | ALL | I canon-cont | 1.0000 | 1.0000 | 1.0000 |
| musique_clean | ALL | G nbr-raw | 0.5200 | 0.5000 | 0.5098 |
| hotpotqa_clean | ALL | B splade-part | 0.2958 | 0.2958 | 0.2838 |
| hotpotqa_clean | ALL | F ppr-reach | 0.3239 | 0.3239 | 0.3108 |
| hotpotqa_clean | ALL | I canon-cont | 0.3944 | 0.3944 | 0.3784 |
| hotpotqa_clean | ALL | G nbr-raw | 0.0141 | 0.0141 | 0.0135 |
| squad_clean | ALL | B splade-part | 1.0000 | 1.0000 | 1.0000 |
| squad_clean | ALL | F ppr-reach | 1.0000 | 1.0000 | 1.0000 |
| squad_clean | ALL | I canon-cont | 1.0000 | 1.0000 | 1.0000 |
| squad_clean | ALL | G nbr-raw | 0.6364 | 0.6364 | 0.6364 |

## T6 -- marginal complementarity at M=64 (STEP 3)

Judged only on DISCOVERY. `unique` = recovered by this family and no other. `greedy` adds families in order of how many additional pool-limited queries become fully covered.

**metaqa** (pool-limited queries = 529)

| family | missing partitions recovered | unique partitions | queries fully solved | unique queries |
|---|---|---|---|---|
| A dense-part | 814 | 55 | 36 | 0 |
| B splade-part | 1082 | 77 | 147 | 19 |
| C node-dense | 653 | 73 | 30 | 3 |
| D node-splade | 592 | 86 | 29 | 3 |
| E S4-struct | 522 | 86 | 46 | 5 |
| F ppr-reach | 1045 | 44 | 132 | 21 |
| G nbr-raw | 1054 | 0 | 100 | 0 |
| G nbr-norm | 939 | 52 | 71 | 3 |
| H frontier | 1133 | 2 | 114 | 1 |
| I canon-cont | 1078 | 1 | 118 | 1 |

| step | family added | marginal queries solved | cumulative | cumulative frac |
|---|---|---|---|---|
| 1 | B splade-part | 147 | 147 | 0.2779 |
| 2 | F ppr-reach | 61 | 208 | 0.3932 |
| 3 | G nbr-norm | 14 | 222 | 0.4197 |
| 4 | E S4-struct | 9 | 231 | 0.4367 |
| 5 | C node-dense | 4 | 235 | 0.4442 |
| 6 | A dense-part | 3 | 238 | 0.4499 |
| 7 | D node-splade | 3 | 241 | 0.4556 |
| 8 | H frontier | 1 | 242 | 0.4575 |
| 9 | I canon-cont | 1 | 243 | 0.4594 |
| 10 | G nbr-raw | 0 | 243 | 0.4594 |

**webqsp** (pool-limited queries = 293)

| family | missing partitions recovered | unique partitions | queries fully solved | unique queries |
|---|---|---|---|---|
| A dense-part | 371 | 22 | 91 | 8 |
| B splade-part | 484 | 47 | 101 | 18 |
| C node-dense | 128 | 21 | 24 | 1 |
| D node-splade | 252 | 34 | 35 | 5 |
| E S4-struct | 66 | 9 | 13 | 3 |
| F ppr-reach | 242 | 15 | 40 | 2 |
| G nbr-raw | 137 | 0 | 26 | 0 |
| G nbr-norm | 107 | 5 | 21 | 1 |
| H frontier | 156 | 0 | 28 | 0 |
| I canon-cont | 509 | 4 | 110 | 2 |

| step | family added | marginal queries solved | cumulative | cumulative frac |
|---|---|---|---|---|
| 1 | I canon-cont | 110 | 110 | 0.3754 |
| 2 | B splade-part | 23 | 133 | 0.4539 |
| 3 | D node-splade | 13 | 146 | 0.4983 |
| 4 | A dense-part | 10 | 156 | 0.5324 |
| 5 | F ppr-reach | 4 | 160 | 0.5461 |
| 6 | E S4-struct | 3 | 163 | 0.5563 |
| 7 | G nbr-raw | 2 | 165 | 0.5631 |
| 8 | C node-dense | 1 | 166 | 0.5666 |
| 9 | G nbr-norm | 1 | 167 | 0.5700 |
| 10 | H frontier | 0 | 167 | 0.5700 |

**2wiki_clean** (pool-limited queries = 97)

| family | missing partitions recovered | unique partitions | queries fully solved | unique queries |
|---|---|---|---|---|
| A dense-part | 28 | 2 | 28 | 2 |
| B splade-part | 22 | 1 | 21 | 1 |
| C node-dense | 30 | 4 | 30 | 4 |
| D node-splade | 37 | 5 | 36 | 5 |
| E S4-struct | 5 | 1 | 5 | 1 |
| F ppr-reach | 22 | 0 | 22 | 0 |
| G nbr-raw | 20 | 0 | 19 | 0 |
| G nbr-norm | 25 | 2 | 24 | 2 |
| H frontier | 21 | 0 | 20 | 0 |
| I canon-cont | 28 | 0 | 28 | 0 |

| step | family added | marginal queries solved | cumulative | cumulative frac |
|---|---|---|---|---|
| 1 | D node-splade | 36 | 36 | 0.3711 |
| 2 | C node-dense | 15 | 51 | 0.5258 |
| 3 | G nbr-norm | 8 | 59 | 0.6082 |
| 4 | A dense-part | 3 | 62 | 0.6392 |
| 5 | B splade-part | 3 | 65 | 0.6701 |
| 6 | E S4-struct | 2 | 67 | 0.6907 |
| 7 | F ppr-reach | 0 | 67 | 0.6907 |
| 8 | G nbr-raw | 0 | 67 | 0.6907 |
| 9 | H frontier | 0 | 67 | 0.6907 |
| 10 | I canon-cont | 0 | 67 | 0.6907 |

**musique_clean** (pool-limited queries = 50)

| family | missing partitions recovered | unique partitions | queries fully solved | unique queries |
|---|---|---|---|---|
| A dense-part | 40 | 0 | 39 | 0 |
| B splade-part | 40 | 0 | 39 | 0 |
| C node-dense | 33 | 0 | 32 | 0 |
| D node-splade | 36 | 0 | 35 | 0 |
| E S4-struct | 9 | 0 | 9 | 0 |
| F ppr-reach | 46 | 0 | 45 | 0 |
| G nbr-raw | 26 | 0 | 25 | 0 |
| G nbr-norm | 29 | 0 | 28 | 0 |
| H frontier | 31 | 0 | 30 | 0 |
| I canon-cont | 43 | 0 | 42 | 0 |

| step | family added | marginal queries solved | cumulative | cumulative frac |
|---|---|---|---|---|
| 1 | F ppr-reach | 45 | 45 | 0.9000 |
| 2 | B splade-part | 4 | 49 | 0.9800 |
| 3 | A dense-part | 1 | 50 | 1.0000 |
| 4 | C node-dense | 0 | 50 | 1.0000 |
| 5 | D node-splade | 0 | 50 | 1.0000 |
| 6 | E S4-struct | 0 | 50 | 1.0000 |
| 7 | G nbr-raw | 0 | 50 | 1.0000 |
| 8 | G nbr-norm | 0 | 50 | 1.0000 |
| 9 | H frontier | 0 | 50 | 1.0000 |
| 10 | I canon-cont | 0 | 50 | 1.0000 |

**hotpotqa_clean** (pool-limited queries = 71)

| family | missing partitions recovered | unique partitions | queries fully solved | unique queries |
|---|---|---|---|---|
| A dense-part | 9 | 0 | 9 | 0 |
| B splade-part | 7 | 1 | 7 | 1 |
| C node-dense | 24 | 10 | 24 | 10 |
| D node-splade | 15 | 3 | 14 | 2 |
| E S4-struct | 4 | 2 | 4 | 2 |
| F ppr-reach | 14 | 0 | 14 | 0 |
| G nbr-raw | 0 | 0 | 0 | 0 |
| G nbr-norm | 5 | 1 | 5 | 1 |
| H frontier | 2 | 0 | 2 | 0 |
| I canon-cont | 11 | 0 | 11 | 0 |

| step | family added | marginal queries solved | cumulative | cumulative frac |
|---|---|---|---|---|
| 1 | C node-dense | 24 | 24 | 0.3380 |
| 2 | D node-splade | 8 | 32 | 0.4507 |
| 3 | B splade-part | 3 | 35 | 0.4930 |
| 4 | E S4-struct | 2 | 37 | 0.5211 |
| 5 | A dense-part | 1 | 38 | 0.5352 |
| 6 | G nbr-norm | 1 | 39 | 0.5493 |
| 7 | F ppr-reach | 0 | 39 | 0.5493 |
| 8 | G nbr-raw | 0 | 39 | 0.5493 |
| 9 | H frontier | 0 | 39 | 0.5493 |
| 10 | I canon-cont | 0 | 39 | 0.5493 |

**squad_clean** (pool-limited queries = 11)

| family | missing partitions recovered | unique partitions | queries fully solved | unique queries |
|---|---|---|---|---|
| A dense-part | 7 | 0 | 7 | 0 |
| B splade-part | 4 | 0 | 4 | 0 |
| C node-dense | 8 | 0 | 8 | 0 |
| D node-splade | 8 | 0 | 8 | 0 |
| E S4-struct | 1 | 0 | 1 | 0 |
| F ppr-reach | 7 | 0 | 7 | 0 |
| G nbr-raw | 7 | 0 | 7 | 0 |
| G nbr-norm | 7 | 0 | 7 | 0 |
| H frontier | 7 | 0 | 7 | 0 |
| I canon-cont | 6 | 0 | 6 | 0 |

| step | family added | marginal queries solved | cumulative | cumulative frac |
|---|---|---|---|---|
| 1 | C node-dense | 8 | 8 | 0.7273 |
| 2 | B splade-part | 2 | 10 | 0.9091 |
| 3 | A dense-part | 0 | 10 | 0.9091 |
| 4 | D node-splade | 0 | 10 | 0.9091 |
| 5 | E S4-struct | 0 | 10 | 0.9091 |
| 6 | F ppr-reach | 0 | 10 | 0.9091 |
| 7 | G nbr-raw | 0 | 10 | 0.9091 |
| 8 | G nbr-norm | 0 | 10 | 0.9091 |
| 9 | H frontier | 0 | 10 | 0.9091 |
| 10 | I canon-cont | 0 | 10 | 0.9091 |

## T7 -- overlap between proposal families, MetaQA at M=64 (STEP 3)

Jaccard over the set of (query, recovered missing partition) pairs.

|  | A | B | C | D | E | F | G | G | H | I |
|---|---|---|---|---|---|---|---|---|---|---|
| A dense-part | 1.00 | 0.19 | 0.26 | 0.12 | 0.15 | 0.27 | 0.22 | 0.20 | 0.23 | 0.43 |
| B splade-part | 0.19 | 1.00 | 0.15 | 0.19 | 0.14 | 0.34 | 0.26 | 0.20 | 0.28 | 0.60 |
| C node-dense | 0.26 | 0.15 | 1.00 | 0.16 | 0.12 | 0.24 | 0.16 | 0.14 | 0.17 | 0.20 |
| D node-splade | 0.12 | 0.19 | 0.16 | 1.00 | 0.11 | 0.21 | 0.12 | 0.11 | 0.13 | 0.17 |
| E S4-struct | 0.15 | 0.14 | 0.12 | 0.11 | 1.00 | 0.12 | 0.14 | 0.14 | 0.15 | 0.16 |
| F ppr-reach | 0.27 | 0.34 | 0.24 | 0.21 | 0.12 | 1.00 | 0.40 | 0.26 | 0.44 | 0.34 |
| G nbr-raw | 0.22 | 0.26 | 0.16 | 0.12 | 0.14 | 0.40 | 1.00 | 0.69 | 0.93 | 0.30 |
| G nbr-norm | 0.20 | 0.20 | 0.14 | 0.11 | 0.14 | 0.26 | 0.69 | 1.00 | 0.65 | 0.24 |
| H frontier | 0.23 | 0.28 | 0.17 | 0.13 | 0.15 | 0.44 | 0.93 | 0.65 | 1.00 | 0.32 |
| I canon-cont | 0.43 | 0.60 | 0.20 | 0.17 | 0.16 | 0.34 | 0.30 | 0.24 | 0.32 | 1.00 |

## T8 -- anatomy of the MetaQA hop3 CURRENT_POOL_LIMIT failures (STEP 4)

398 pool-limited hop3 queries, 2596 required gold partitions they cannot see. `frac defined` = fraction of those partitions the channel ranks at all (a rank of -1 means the channel never lists it).

| attribute | n defined | frac defined | p10 | median | p90 | max |
|---|---|---|---|---|---|---|
| dense_rank | 2596 | 1.0000 | 67.0 | 189.0 | 369.0 | 400 |
| splade_rank | 2596 | 1.0000 | 52.0 | 167.0 | 361.0 | 400 |
| node_dense_rank | 784 | 0.3020 | 35.0 | 105.0 | 179.7 | 199 |
| node_splade_rank | 865 | 0.3332 | 38.0 | 105.0 | 179.0 | 199 |
| s4_rank | 422 | 0.1626 | 42.0 | 67.0 | 87.0 | 109 |
| ppr_rank | 2596 | 1.0000 | 32.5 | 150.0 | 319.0 | 400 |
| graph_dist_min | 2596 | 1.0000 | 1.0 | 1.0 | 1.0 | 2 |
| part_degree | 2596 | 1.0000 | 109.0 | 172.0 | 312.0 | 394 |
| part_size | 2596 | 1.0000 | 97.0 | 100.0 | 103.0 | 103 |

## T9 -- how those failures group, derived from T8 (STEP 4)

| group (partition level) | n | frac |
|---|---|---|
| 2_GRAPH_REACHABLE_BUT_NO_FAMILY_PROPOSES_AT_256 | 32 | 0.0123 |
| 3_INSIDE_CANONICAL_TOP200_BUT_OUTSIDE_POOL | 2001 | 0.7708 |
| 4_PROPOSED_ONLY_OUTSIDE_CANONICAL_TOP200 | 563 | 0.2169 |

| group (query level, worst partition) | n | frac |
|---|---|---|
| 3_INSIDE_CANONICAL_TOP200_BUT_OUTSIDE_POOL | 189 | 0.4749 |
| 4_PROPOSED_ONLY_OUTSIDE_CANONICAL_TOP200 | 209 | 0.5251 |

Graph distance from the nearest retrieval-seed partition, and depth in the precomputed top-32 neighbour tables (-1 = absent from the tables):

| min graph distance | n |
|---|---|
| 1 | 2415 |
| 2 | 181 |

| precomputed table depth | n |
|---|---|
| -1 | 1384 |
| 1 | 984 |
| 2 | 205 |
| 3 | 23 |

| family | frac of the 2596 missing partitions proposed at M=256 |
|---|---|
| A dense-part | 0.7396 |
| B splade-part | 0.8155 |
| C node-dense | 0.3020 |
| D node-splade | 0.3332 |
| E S4-struct | 0.1626 |
| F ppr-reach | 0.8790 |
| G nbr-raw | 0.3790 |
| G nbr-norm | 0.3902 |
| H frontier | 0.4669 |

## T10 -- PPR as a PROPOSER only (STEP 5)

PPR contributes candidate membership only; no PPR score reaches F6/R0. Both bookends are reported for the same pool, as STEP 8 requires.

| corpus | slice | M_ppr | SAFE | current-pool oracle | PPR-pool oracle | ACTUAL frozen | delta | gained/lost |
|---|---|---|---|---|---|---|---|---|
| metaqa | hop3 | 16 | 0.2583 | 0.3634 | 0.4279 | 0.2583 | +0.0000 | +0/-0 |
| metaqa | hop3 | 32 | 0.2583 | 0.3634 | 0.4745 | 0.2583 | +0.0000 | +0/-0 |
| metaqa | hop3 | 64 | 0.2583 | 0.3634 | 0.5090 | 0.2583 | +0.0000 | +0/-0 |
| metaqa | hop3 | 128 | 0.2583 | 0.3634 | 0.5601 | 0.2583 | +0.0000 | +0/-0 |
| webqsp | ALL | 16 | 0.7646 | 0.7886 | 0.7956 | 0.7646 | +0.0000 | +0/-0 |
| webqsp | ALL | 32 | 0.7646 | 0.7886 | 0.8069 | 0.7646 | +0.0000 | +0/-0 |
| webqsp | ALL | 64 | 0.7646 | 0.7886 | 0.8168 | 0.7646 | +0.0000 | +0/-0 |
| webqsp | ALL | 128 | 0.7646 | 0.7886 | 0.8302 | 0.7646 | +0.0000 | +0/-0 |
| 2wiki_clean | ALL | 16 | 0.9435 | 0.9515 | 0.9565 | 0.9435 | +0.0000 | +0/-0 |
| 2wiki_clean | ALL | 32 | 0.9435 | 0.9515 | 0.9605 | 0.9435 | +0.0000 | +0/-0 |
| 2wiki_clean | ALL | 64 | 0.9435 | 0.9515 | 0.9625 | 0.9435 | +0.0000 | +0/-0 |
| 2wiki_clean | ALL | 128 | 0.9435 | 0.9515 | 0.9700 | 0.9435 | +0.0000 | +0/-0 |
| musique_clean | ALL | 16 | 0.9635 | 0.9750 | 0.9800 | 0.9635 | +0.0000 | +0/-0 |
| musique_clean | ALL | 32 | 0.9635 | 0.9750 | 0.9860 | 0.9635 | +0.0000 | +0/-0 |
| musique_clean | ALL | 64 | 0.9635 | 0.9750 | 0.9975 | 0.9635 | +0.0000 | +0/-0 |
| musique_clean | ALL | 128 | 0.9635 | 0.9750 | 1.0000 | 0.9635 | +0.0000 | +0/-0 |
| hotpotqa_clean | ALL | 16 | 0.9505 | 0.9645 | 0.9645 | 0.9505 | +0.0000 | +0/-0 |
| hotpotqa_clean | ALL | 32 | 0.9505 | 0.9645 | 0.9660 | 0.9505 | +0.0000 | +0/-0 |
| hotpotqa_clean | ALL | 64 | 0.9505 | 0.9645 | 0.9715 | 0.9505 | +0.0000 | +0/-0 |
| hotpotqa_clean | ALL | 128 | 0.9505 | 0.9645 | 0.9755 | 0.9505 | +0.0000 | +0/-0 |
| squad_clean | ALL | 16 | 0.9875 | 0.9945 | 0.9960 | 0.9875 | +0.0000 | +0/-0 |
| squad_clean | ALL | 32 | 0.9875 | 0.9945 | 0.9960 | 0.9875 | +0.0000 | +0/-0 |
| squad_clean | ALL | 64 | 0.9875 | 0.9945 | 0.9980 | 0.9875 | +0.0000 | +0/-0 |
| squad_clean | ALL | 128 | 0.9875 | 0.9945 | 0.9990 | 0.9875 | +0.0000 | +0/-0 |

## T11 -- online structural discovery vs the precomputed tables (STEP 6)

Same target set as T8. `edges` = ONLINE_GRAPH_EDGES_TOUCHED per query. The precomputed tables are corpus-side, so K is a storage knob, never a query-time cost.

| corpus | npart | mean partition degree | source | missing-partition recall | mean proposals | frac universe | edges touched | table MB |
|---|---|---|---|---|---|---|---|---|
| metaqa_hop3 | 401 | 158.1 | ONLINE depth1 | 0.9303 | 284.7 | 0.811 | 698 | - |
| metaqa_hop3 | 401 | 158.1 | ONLINE depth2 | 1.0000 | 351.0 | 1.000 | 55933 | - |
| metaqa_hop3 | 401 | 158.1 | TABLE K32_depth1 | 0.3790 | 66.8 | - | 0 | 0.05 |
| metaqa_hop3 | 401 | 158.1 | TABLE K32_depth2 | 0.9653 | 323.6 | - | 0 | 0.10 |
| metaqa_hop3 | 401 | 158.1 | TABLE K64_depth1 | 0.6098 | 133.7 | - | 0 | 0.10 |
| metaqa_hop3 | 401 | 158.1 | TABLE K64_depth2 | 0.9988 | 350.5 | - | 0 | 0.21 |
| metaqa_hop3 | 401 | 158.1 | TABLE K128_depth1 | 0.8459 | 230.2 | - | 0 | 0.21 |
| metaqa_hop3 | 401 | 158.1 | TABLE K128_depth2 | 1.0000 | 351.0 | - | 0 | 0.41 |
| webqsp | 7814 | 129.1 | ONLINE depth1 | 0.5591 | 408.2 | 0.053 | 496 | - |
| webqsp | 7814 | 129.1 | ONLINE depth2 | 1.0000 | 7511.3 | 0.968 | 110095 | - |
| webqsp | 7814 | 129.1 | TABLE K32_depth1 | 0.1396 | 71.0 | - | 0 | 1.00 |
| webqsp | 7814 | 129.1 | TABLE K32_depth2 | 0.6204 | 1459.5 | - | 0 | 2.00 |
| webqsp | 7814 | 129.1 | TABLE K64_depth1 | 0.2621 | 151.2 | - | 0 | 2.00 |
| webqsp | 7814 | 129.1 | TABLE K64_depth2 | 0.8868 | 3857.0 | - | 0 | 4.00 |
| webqsp | 7814 | 129.1 | TABLE K128_depth1 | 0.4238 | 273.3 | - | 0 | 4.00 |
| webqsp | 7814 | 129.1 | TABLE K128_depth2 | 0.9915 | 6282.1 | - | 0 | 8.00 |
| 2wiki_clean | 658 | 85.3 | ONLINE depth1 | 0.5408 | 210.9 | 0.347 | 387 | - |
| 2wiki_clean | 658 | 85.3 | ONLINE depth2 | 1.0000 | 608.0 | 1.000 | 30733 | - |
| 2wiki_clean | 658 | 85.3 | TABLE K32_depth1 | 0.2245 | 70.8 | - | 0 | 0.08 |
| 2wiki_clean | 658 | 85.3 | TABLE K32_depth2 | 0.8571 | 468.2 | - | 0 | 0.17 |
| 2wiki_clean | 658 | 85.3 | TABLE K64_depth1 | 0.3980 | 138.7 | - | 0 | 0.17 |
| 2wiki_clean | 658 | 85.3 | TABLE K64_depth2 | 0.9796 | 583.5 | - | 0 | 0.34 |
| 2wiki_clean | 658 | 85.3 | TABLE K128_depth1 | 0.5408 | 192.1 | - | 0 | 0.34 |
| 2wiki_clean | 658 | 85.3 | TABLE K128_depth2 | 1.0000 | 604.7 | - | 0 | 0.67 |
| hotpotqa_clean | 5074 | 471.0 | ONLINE depth1 | 0.5676 | 1590.9 | 0.317 | 2327 | - |
| hotpotqa_clean | 5074 | 471.0 | ONLINE depth2 | 1.0000 | 5024.0 | 1.000 | 1079499 | - |
| hotpotqa_clean | 5074 | 471.0 | TABLE K32_depth1 | 0.0135 | 74.7 | - | 0 | 0.65 |
| hotpotqa_clean | 5074 | 471.0 | TABLE K32_depth2 | 0.4324 | 1473.2 | - | 0 | 1.30 |
| hotpotqa_clean | 5074 | 471.0 | TABLE K64_depth1 | 0.1351 | 164.1 | - | 0 | 1.30 |
| hotpotqa_clean | 5074 | 471.0 | TABLE K64_depth2 | 0.8649 | 3383.0 | - | 0 | 2.60 |
| hotpotqa_clean | 5074 | 471.0 | TABLE K128_depth1 | 0.2838 | 362.9 | - | 0 | 2.60 |
| hotpotqa_clean | 5074 | 471.0 | TABLE K128_depth2 | 1.0000 | 4855.0 | - | 0 | 5.20 |

## T12 -- union proposers, candidate-pool oracle at a single global budget (STEP 7)

Round-robin interleave of the member families, deduped, cut at M_TOTAL. One budget for every corpus; no per-corpus tuning; no learned component. `gf` = touches no graph at query time; `pc` = fully served by corpus-side precomputed tables.

**metaqa** slice `hop3` -- SAFE 0.2583, current-pool oracle 0.3634

| union | gf | pc | oracle M=32 | oracle M=64 | oracle M=128 | oracle M=256 | ACTUAL M=256 |
|---|---|---|---|---|---|---|---|
| U_RET2 | yes | yes | 0.4189 | 0.4595 | 0.5180 | 0.6186 | 0.2583 |
| U_RET3 | yes | yes | 0.4084 | 0.4505 | 0.5030 | 0.6186 | 0.2583 |
| U_RET2_G | no | yes | 0.4309 | 0.4820 | 0.5360 | 0.6336 | 0.2583 |
| U_RET2_P | no | no | 0.4429 | 0.5000 | 0.5541 | 0.6306 | 0.2583 |
| U_RET2_NODE | yes | yes | 0.4009 | 0.4339 | 0.5030 | 0.6231 | 0.2583 |
| U_BROAD | no | no | 0.4324 | 0.4880 | 0.5435 | 0.6306 | 0.2583 |
| U_PC4 | no | yes | 0.4249 | 0.4610 | 0.5405 | 0.6321 | 0.2583 |
| U_PC5 | no | yes | 0.4189 | 0.4535 | 0.5270 | 0.6351 | 0.2583 |
| U_ALL | no | no | 0.4039 | 0.4505 | 0.5195 | 0.6351 | 0.2583 |

**webqsp** slice `ALL` -- SAFE 0.7646, current-pool oracle 0.7886

| union | gf | pc | oracle M=32 | oracle M=64 | oracle M=128 | oracle M=256 | ACTUAL M=256 |
|---|---|---|---|---|---|---|---|
| U_RET2 | yes | yes | 0.8309 | 0.8598 | 0.8837 | 0.9070 | 0.7646 |
| U_RET3 | yes | yes | 0.8295 | 0.8562 | 0.8837 | 0.9077 | 0.7646 |
| U_RET2_G | no | yes | 0.8252 | 0.8471 | 0.8760 | 0.9049 | 0.7646 |
| U_RET2_P | no | no | 0.8231 | 0.8464 | 0.8802 | 0.8999 | 0.7646 |
| U_RET2_NODE | yes | yes | 0.8203 | 0.8471 | 0.8816 | 0.9098 | 0.7646 |
| U_BROAD | no | no | 0.8175 | 0.8443 | 0.8717 | 0.8992 | 0.7646 |
| U_PC4 | no | yes | 0.8217 | 0.8436 | 0.8774 | 0.9049 | 0.7646 |
| U_PC5 | no | yes | 0.8168 | 0.8414 | 0.8724 | 0.9027 | 0.7646 |
| U_ALL | no | no | 0.8132 | 0.8309 | 0.8619 | 0.8978 | 0.7646 |

**2wiki_clean** slice `ALL` -- SAFE 0.9435, current-pool oracle 0.9515

| union | gf | pc | oracle M=32 | oracle M=64 | oracle M=128 | oracle M=256 | ACTUAL M=256 |
|---|---|---|---|---|---|---|---|
| U_RET2 | yes | yes | 0.9605 | 0.9650 | 0.9755 | 0.9860 | 0.9435 |
| U_RET3 | yes | yes | 0.9615 | 0.9655 | 0.9745 | 0.9855 | 0.9435 |
| U_RET2_G | no | yes | 0.9625 | 0.9675 | 0.9730 | 0.9860 | 0.9435 |
| U_RET2_P | no | no | 0.9615 | 0.9655 | 0.9735 | 0.9875 | 0.9435 |
| U_RET2_NODE | yes | yes | 0.9625 | 0.9675 | 0.9770 | 0.9890 | 0.9435 |
| U_BROAD | no | no | 0.9615 | 0.9680 | 0.9750 | 0.9855 | 0.9435 |
| U_PC4 | no | yes | 0.9635 | 0.9670 | 0.9750 | 0.9870 | 0.9435 |
| U_PC5 | no | yes | 0.9640 | 0.9690 | 0.9765 | 0.9890 | 0.9435 |
| U_ALL | no | no | 0.9595 | 0.9675 | 0.9775 | 0.9860 | 0.9435 |

**musique_clean** slice `ALL` -- SAFE 0.9635, current-pool oracle 0.9750

| union | gf | pc | oracle M=32 | oracle M=64 | oracle M=128 | oracle M=256 | ACTUAL M=256 |
|---|---|---|---|---|---|---|---|
| U_RET2 | yes | yes | 0.9890 | 0.9960 | 1.0000 | 1.0000 | 0.9635 |
| U_RET3 | yes | yes | 0.9880 | 0.9955 | 1.0000 | 1.0000 | 0.9635 |
| U_RET2_G | no | yes | 0.9890 | 0.9965 | 1.0000 | 1.0000 | 0.9635 |
| U_RET2_P | no | no | 0.9875 | 0.9970 | 1.0000 | 1.0000 | 0.9635 |
| U_RET2_NODE | yes | yes | 0.9915 | 0.9970 | 1.0000 | 1.0000 | 0.9635 |
| U_BROAD | no | no | 0.9885 | 0.9965 | 1.0000 | 1.0000 | 0.9635 |
| U_PC4 | no | yes | 0.9905 | 0.9975 | 1.0000 | 1.0000 | 0.9635 |
| U_PC5 | no | yes | 0.9920 | 0.9970 | 1.0000 | 1.0000 | 0.9635 |
| U_ALL | no | no | 0.9890 | 0.9985 | 1.0000 | 1.0000 | 0.9635 |

**hotpotqa_clean** slice `ALL` -- SAFE 0.9505, current-pool oracle 0.9645

| union | gf | pc | oracle M=32 | oracle M=64 | oracle M=128 | oracle M=256 | ACTUAL M=256 |
|---|---|---|---|---|---|---|---|
| U_RET2 | yes | yes | 0.9655 | 0.9685 | 0.9735 | 0.9765 | 0.9505 |
| U_RET3 | yes | yes | 0.9665 | 0.9700 | 0.9725 | 0.9800 | 0.9505 |
| U_RET2_G | no | yes | 0.9650 | 0.9670 | 0.9710 | 0.9765 | 0.9505 |
| U_RET2_P | no | no | 0.9655 | 0.9685 | 0.9730 | 0.9785 | 0.9505 |
| U_RET2_NODE | yes | yes | 0.9685 | 0.9720 | 0.9790 | 0.9845 | 0.9505 |
| U_BROAD | no | no | 0.9660 | 0.9675 | 0.9710 | 0.9785 | 0.9505 |
| U_PC4 | no | yes | 0.9660 | 0.9665 | 0.9715 | 0.9780 | 0.9505 |
| U_PC5 | no | yes | 0.9665 | 0.9710 | 0.9755 | 0.9825 | 0.9505 |
| U_ALL | no | no | 0.9670 | 0.9700 | 0.9740 | 0.9805 | 0.9505 |

**squad_clean** slice `ALL` -- SAFE 0.9875, current-pool oracle 0.9945

| union | gf | pc | oracle M=32 | oracle M=64 | oracle M=128 | oracle M=256 | ACTUAL M=256 |
|---|---|---|---|---|---|---|---|
| U_RET2 | yes | yes | 0.9960 | 0.9980 | 1.0000 | 1.0000 | 0.9875 |
| U_RET3 | yes | yes | 0.9970 | 0.9980 | 1.0000 | 1.0000 | 0.9875 |
| U_RET2_G | no | yes | 0.9970 | 0.9980 | 1.0000 | 1.0000 | 0.9875 |
| U_RET2_P | no | no | 0.9965 | 0.9970 | 1.0000 | 1.0000 | 0.9875 |
| U_RET2_NODE | yes | yes | 0.9975 | 0.9990 | 1.0000 | 1.0000 | 0.9875 |
| U_BROAD | no | no | 0.9965 | 0.9985 | 1.0000 | 1.0000 | 0.9875 |
| U_PC4 | no | yes | 0.9965 | 0.9985 | 1.0000 | 1.0000 | 0.9875 |
| U_PC5 | no | yes | 0.9975 | 0.9985 | 1.0000 | 1.0000 | 0.9875 |
| U_ALL | no | no | 0.9965 | 0.9980 | 0.9995 | 1.0000 | 0.9875 |

## T13 -- MANDATORY MetaQA hop table (STEP 9)

| configuration | hop1 | hop2 | hop3 | ALL |
|---|---|---|---|---|
| SAFE current (frozen selector, current pool) | 0.9955 | 0.7297 | 0.2583 | 0.6612 |
| current-pool oracle | 0.9970 | 0.8033 | 0.3634 | 0.7212 |
| PPR-proposal pool oracle (M_ppr=128) | 0.9985 | 0.8574 | 0.5601 | 0.8053 |
| best single proposer pool oracle (F ppr-reach, M=128) | 0.9985 | 0.8574 | 0.5601 | 0.8053 |
| best union proposer pool oracle (U_PC5, M=256) | 0.9985 | 0.9474 | 0.6351 | 0.8604 |
| best union + FROZEN selector (U_PC5, M=256) | 0.9955 | 0.7297 | 0.2583 | 0.6612 |
| unlimited-pool oracle at B=6 | 1.0000 | 0.9505 | 0.6802 | 0.8769 |
| full-universe P50 oracle | 1.0000 | 0.9970 | 0.9640 | 0.9870 |

The pool-oracle-maximising union and the union recommended under the STEP 12 criteria are the same object here (U_PC5), so no separate row is needed.


Pool headroom closed on hop3 by U_PC5 at M=256: (0.6351 - 0.3634) / (0.6802 - 0.3634) = 0.858.

## T14 -- the recommended generator on every corpus (STEP 10)

`U_PC5` = round-robin over B splade-part, I canon-cont, C node-dense, D node-splade, G nbr-raw. Identical on every corpus.

| corpus | slice | SAFE | current-pool oracle | oracle M=32 | oracle M=64 | oracle M=128 | oracle M=256 | ACTUAL M=256 | mean proposals |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | hop3 | 0.2583 | 0.3634 | 0.4189 | 0.4535 | 0.5270 | 0.6351 | 0.2583 | 256 |
| webqsp | ALL | 0.7646 | 0.7886 | 0.8168 | 0.8414 | 0.8724 | 0.9027 | 0.7646 | 256 |
| 2wiki_clean | ALL | 0.9435 | 0.9515 | 0.9640 | 0.9690 | 0.9765 | 0.9890 | 0.9435 | 256 |
| musique_clean | ALL | 0.9635 | 0.9750 | 0.9920 | 0.9970 | 1.0000 | 1.0000 | 0.9635 | 86 |
| hotpotqa_clean | ALL | 0.9505 | 0.9645 | 0.9665 | 0.9710 | 0.9755 | 0.9825 | 0.9505 | 256 |
| squad_clean | ALL | 0.9875 | 0.9945 | 0.9975 | 0.9985 | 1.0000 | 1.0000 | 0.9875 | 140 |

## T15 -- B = {6, 8, 12} re-run against the EXPANDED pool (STEP 11)

The ceiling audit measured B as non-binding against the OLD pool. Re-measured here once the generator has moved the pool, as the directive requires.

| corpus | slice | B | SAFE | current-pool oracle | expanded-pool oracle | ACTUAL frozen |
|---|---|---|---|---|---|---|
| metaqa | hop3 | 6 | 0.2583 | 0.3634 | 0.6351 | 0.2583 |
| metaqa | hop3 | 8 | 0.2628 | 0.3634 | 0.6562 | 0.2628 |
| metaqa | hop3 | 12 | 0.2658 | 0.3664 | 0.6937 | 0.2658 |
| webqsp | ALL | 6 | 0.7646 | 0.7886 | 0.9027 | 0.7646 |
| webqsp | ALL | 8 | 0.7618 | 0.7886 | 0.9084 | 0.7618 |
| webqsp | ALL | 12 | 0.7555 | 0.7886 | 0.9147 | 0.7555 |
| 2wiki_clean | ALL | 6 | 0.9435 | 0.9515 | 0.9890 | 0.9435 |
| 2wiki_clean | ALL | 8 | 0.9430 | 0.9515 | 0.9890 | 0.9430 |
| 2wiki_clean | ALL | 12 | 0.9420 | 0.9515 | 0.9890 | 0.9420 |
| musique_clean | ALL | 6 | 0.9635 | 0.9750 | 1.0000 | 0.9635 |
| musique_clean | ALL | 8 | 0.9655 | 0.9750 | 1.0000 | 0.9655 |
| musique_clean | ALL | 12 | 0.9650 | 0.9750 | 1.0000 | 0.9650 |
| hotpotqa_clean | ALL | 6 | 0.9505 | 0.9645 | 0.9825 | 0.9505 |
| hotpotqa_clean | ALL | 8 | 0.9520 | 0.9645 | 0.9825 | 0.9520 |
| hotpotqa_clean | ALL | 12 | 0.9555 | 0.9645 | 0.9825 | 0.9555 |
| squad_clean | ALL | 6 | 0.9875 | 0.9945 | 1.0000 | 0.9875 |
| squad_clean | ALL | 8 | 0.9885 | 0.9945 | 1.0000 | 0.9885 |
| squad_clean | ALL | 12 | 0.9900 | 0.9945 | 1.0000 | 0.9900 |

## T16 -- why the frozen selector converts none of it (verification)

Union of A, B, F, I, G at M=256. A `truly new` candidate is one outside base50 and outside the frozen challenger set, so it carries no structural and no retrieval rank.

| corpus | nq | queries offered a new candidate | new candidates offered | new candidates SELECTED | selections identical to SAFE | min score margin |
|---|---|---|---|---|---|---|
| metaqa | 1998 | 1998 | 436201 | 0 | 1998 | 2.55e-04 |
| 2wiki_clean | 2000 | 2000 | 464795 | 0 | 2000 | 8.34e-05 |
| webqsp | 1419 | 1419 | 349486 | 0 | 1419 | 8.34e-05 |
| musique_clean | 2000 | 2000 | 146863 | 0 | 2000 | 8.34e-05 |
| hotpotqa_clean | 2000 | 2000 | 471487 | 0 | 2000 | 5.24e-04 |
| squad_clean | 2000 | 2000 | 260453 | 0 | 2000 | 8.34e-05 |

Bound: any candidate outside base50 scores at most 1/(K0+P) = 0.00909091; every boundary incumbent scores at least 1/(K0+P-1) = 0.00917431. There are exactly B incumbents for B slots, so the new candidate is strictly dominated for every B.

## T17 -- generator aggregation depth, the one CONVERTIBLE knob

M_struct / M_ret set how deep the generator aggregates. Raising them extends the spos / rpos rank lists themselves, so new candidates arrive WITH channel evidence. The scoring rule is untouched and ranks of already-listed partitions are unchanged. Whether this counts as inside the frozen contract is the user's call; it is reported separately for that reason.

| corpus | slice | config | mean pool | pool oracle | ACTUAL | delta vs Ms64/Mr32 | gained/lost | p | sig |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | hop3 | Ms64_Mr32 | 43.2 | 0.3634 | 0.2583 | +0.0000 | +0/-0 | 1 | no |
| metaqa | hop3 | Ms128_Mr32 | 65.2 | 0.3874 | 0.2568 | -0.0015 | +1/-2 | 1 | no |
| metaqa | hop3 | Ms256_Mr32 | 68.7 | 0.3904 | 0.2553 | -0.0030 | +1/-3 | 0.625 | no |
| metaqa | hop3 | Ms64_Mr64 | 58.0 | 0.3679 | 0.2553 | -0.0030 | +2/-4 | 0.688 | no |
| metaqa | hop3 | Ms64_Mr128 | 87.8 | 0.3859 | 0.2568 | -0.0015 | +3/-4 | 1 | no |
| metaqa | hop3 | Ms256_Mr128 | 108.8 | 0.4144 | 0.2583 | +0.0000 | +4/-4 | 1 | no |
| webqsp | ALL | Ms64_Mr32 | 28.4 | 0.7886 | 0.7646 | +0.0000 | +0/-0 | 1 | no |
| webqsp | ALL | Ms128_Mr32 | 48.4 | 0.7970 | 0.7639 | -0.0007 | +3/-4 | 1 | no |
| webqsp | ALL | Ms256_Mr32 | 52.2 | 0.7992 | 0.7646 | +0.0000 | +4/-4 | 1 | no |
| webqsp | ALL | Ms64_Mr64 | 35.5 | 0.7928 | 0.7618 | -0.0028 | +2/-6 | 0.289 | no |
| webqsp | ALL | Ms64_Mr128 | 56.5 | 0.8034 | 0.7604 | -0.0042 | +4/-10 | 0.18 | no |
| webqsp | ALL | Ms256_Mr128 | 79.2 | 0.8125 | 0.7597 | -0.0049 | +8/-15 | 0.21 | no |
| 2wiki_clean | ALL | Ms64_Mr32 | 35.8 | 0.9515 | 0.9435 | +0.0000 | +0/-0 | 1 | no |
| 2wiki_clean | ALL | Ms128_Mr32 | 51.8 | 0.9545 | 0.9430 | -0.0005 | +0/-1 | 1 | no |
| 2wiki_clean | ALL | Ms256_Mr32 | 53.1 | 0.9545 | 0.9430 | -0.0005 | +0/-1 | 1 | no |
| 2wiki_clean | ALL | Ms64_Mr64 | 48.0 | 0.9590 | 0.9440 | +0.0005 | +3/-2 | 1 | no |
| 2wiki_clean | ALL | Ms64_Mr128 | 73.6 | 0.9665 | 0.9420 | -0.0015 | +4/-7 | 0.549 | no |
| 2wiki_clean | ALL | Ms256_Mr128 | 89.2 | 0.9690 | 0.9410 | -0.0025 | +4/-9 | 0.267 | no |

