# UNIVERSAL BALANCED PARTITION + OVERLAP SEARCH -- FINAL REPORT

Question: is there ONE dataset-agnostic retrieval-block substrate that improves L1 on
some corpus without significantly regressing any other?  The previous program closed the
hard-partition recipe search at PARTITIONING_VERDICT=D (strong partition effect, no
universal rule).  This program reopened it two ways: real balanced partitioners on Linux
CPU, and retrieval blocks that are allowed to OVERLAP -- so a node participating in the
structural, entity-NER and semantic-kNN topologies at once no longer has to be assigned to
exactly one block, with the rest of its locality cut away.

Everything below is measured under the FROZEN L1 contract (MASTER_TOPOLOGY=C, P=50, B=6,
S4 aggregation, F6 selector, K0=60, M_struct=64, M_ret=32).  No gold or query information
is used during BUILD; gold appears only in evaluation.  No GPU was allocated anywhere.


## THE ANSWER

**BEST_UNIVERSAL_METHOD = `O4_FULL_C_b0.5`**

A balanced METIS core, unchanged, plus a halo of 1-hop boundary nodes drawn from the
union of all three edge families and capped at beta = 0.5 of each core's
size.  Halo nodes are ranked parameter-free by
boundary mass `s(v, C_j) = sum_{u in C_j, (u,v) in E} w(u,v) / deg(u)` -- no query, no
gold, no learning, one identical recipe on all six corpora.

It is the first substrate in this program that improves EVERY corpus:

| corpus   | hard core | core+halo | delta   | p (exact McNemar) | sig | P50 exposure | queries rescued |
|----------|-----------|-----------|---------|-------------------|-----|--------------|-----------------|
| MetaQA   | 0.6612    | 0.7432    | +0.0820 | 8.5e-50           | yes | x1.358       | 164             |
| WebQSP   | 0.7646    | 0.8118    | +0.0472 | 1.4e-20           | yes | x1.394       | 67              |
| 2Wiki    | 0.9435    | 0.9540    | +0.0105 | 9.5e-07           | yes | x1.349       | 21              |
| MuSiQue  | 0.9635    | 0.9775    | +0.0140 | 7.4e-09           | yes | x1.204       | 28              |
| SQuAD    | 0.9875    | 0.9910    | +0.0035 | 1.6e-02           | yes | x1.229       | 7               |
| HotpotQA | 0.9505    | 0.9575    | +0.0070 | 1.2e-04           | yes | x1.289       | 14              |

MetaQA by hop -- the gain concentrates exactly in the multi-hop tail that every
previous phase of this program failed to move:

| hop  | hard core | core+halo | delta   |
|------|-----------|-----------|---------|
| hop1 | 0.9955    | 0.9970    | +0.0015 |
| hop2 | 0.7297    | 0.8063    | +0.0766 |
| hop3 | 0.2583    | 0.4264    | +0.1681 |

Offline replication factor R = 1.497; online mean node exposure x1.304, worst x1.394.
A hard partition is R = 1.0 by definition, so this stores 1.50 memberships per node and reads about x1.304 the nodes the incumbent reads.

The runner-up is worth naming, because it is cheaper.  `O4_FULL_C_b0.25` also
has zero significant regressions, gains significantly on 5 of 6
corpora, and runs at x1.145 mean exposure (R = 1.245) -- essentially the 1.25x budget --
for MetaQA hop3 0.3378 instead of 0.4264.  It also wins
the matched-exposure control on 2 corpora against 1.  Which of the two ships is a budget decision, not
an evidence one: both are universal, and beta is the dial.


## WHY THIS IS NOT JUST READING MORE NODES (P15)

A halo reads more nodes, so the only honest comparison holds the unique-node budget fixed
and asks whether overlap beats simply selecting MORE hard core partitions from the same
frozen BASE ranking.  One confound had to be removed first: `O0_CORE` IS hard P=50, yet it
scores above the P'=50 control, because the control ranks by BASE while every overlap cell
is selected by F6.  That selector gap (+0.0025 to +0.0160) is subtracted from
every cell, so the reported quantity is a difference-in-differences, tested by paired
bootstrap over queries.  `--` marks cells whose matched hard depth saturated the 200-block
ranking and therefore read FEWER nodes than the overlap cell, where no claim is possible.

| cell            | MetaQA      | WebQSP      | 2Wiki       | MuSiQue  | SQuAD    | HotpotQA    | passes P15 | sig win/loss |
|-----------------|-------------|-------------|-------------|----------|----------|-------------|------------|--------------|
| O4_FULL_C       | (unmatched) | (unmatched) | (unmatched) | -0.0035  | -0.0070* | (unmatched) | no         | 0/1          |
| O1_STRUCT       | +0.1271*    | (unmatched) | (unmatched) | -0.0080* | -0.0075* | (unmatched) | no         | 1/2          |
| O1_STRUCT_b1.0  | +0.1246*    | +0.0416*    | -0.0085*    | -0.0080* | -0.0055* | -0.0075*    | no         | 2/4          |
| O4_FULL_C_b1.0  | +0.0801*    | +0.0056     | -0.0020     | -0.0015  | -0.0045  | -0.0075*    | no         | 1/1          |
| O2_NERX         | +0.0385*    | -0.0176*    | -0.0015     | -0.0025  | -0.0070* | (unmatched) | no         | 1/2          |
| O1_STRUCT_b0.5  | +0.1166*    | +0.0303*    | -0.0045     | -0.0025  | -0.0030  | -0.0040*    | no         | 2/1          |
| O3_KNN          | -0.0075     | -0.0458*    | -0.0015     | +0.0020  | -0.0050  | -0.0105*    | no         | 0/2          |
| O4_FULL_C_b0.5  | +0.0480*    | +0.0113     | -0.0005     | +0.0015  | -0.0035  | -0.0020     | yes        | 1/0          |
| O1_STRUCT_b0.25 | +0.0916*    | +0.0275*    | -0.0020     | -0.0030  | -0.0020  | -0.0020*    | no         | 2/1          |
| O2_NERX_b1.0    | +0.0100     | -0.0141*    | -0.0010     | +0.0010  | -0.0025  | -0.0135*    | no         | 0/2          |
| O3_KNN_b1.0     | -0.0195*    | -0.0374*    | +0.0000     | +0.0015  | -0.0030  | -0.0075     | no         | 0/2          |
| O4_FULL_C_b0.25 | +0.0230*    | +0.0226*    | +0.0000     | +0.0035  | -0.0020  | +0.0000     | yes        | 2/0          |
| O2_NERX_b0.5    | +0.0080     | -0.0092     | -0.0020     | +0.0025  | -0.0040  | -0.0050     | no         | 0/0          |
| O2_NERX_b0.25   | +0.0085     | -0.0007     | +0.0015     | +0.0075* | -0.0030  | -0.0025     | yes        | 1/0          |
| O3_KNN_b0.5     | -0.0275*    | -0.0205*    | +0.0015     | -0.0010  | -0.0040  | -0.0030     | no         | 0/2          |
| O3_KNN_b0.25    | -0.0165*    | -0.0091     | +0.0015     | +0.0035  | -0.0010  | +0.0005     | no         | 0/1          |

The families separate cleanly.  STRUCT halos buy the largest raw knowledge-base gains,
but significantly LOSE to reading deeper on HotpotQA at EVERY bounded budget;
kNN halos do the same on MetaQA.
The cells that survive the control are `O4_FULL_C_b0.5`, `O4_FULL_C_b0.25`, `O2_NERX_b0.25` -- the FULL_C union at beta <= 0.5, plus the
tightest NERX halo.  The union is the one that wins on the most corpora while losing
on none, which is why the selection rule lands on it rather than on a family.


## IS THE HALO SCORE DOING THE WORK, OR JUST THE 1-HOP RELATION?

P15 rules out `it only reads more nodes`.  This rules out the other easy story.
Hold the candidate set (the same FULL_C 1-hop pairs) and the same per-block quota
beta*|C_j| fixed, and replace the boundary-mass score with a seeded RANDOM one, so
both arms pay out exactly the same NUMBER of halo nodes from exactly the same
pool.  Three seeds, frozen F6 core selection in both arms.

| corpus   | beta | core only | random-ranked halo | boundary-mass halo | score gain | sig (3/3 seeds) | Jaccard vs real |
|----------|------|-----------|--------------------|--------------------|------------|-----------------|-----------------|
| 2Wiki    | 0.5  | 0.9435    | 0.9493             | 0.9540             | +0.0047    | no              | 0.043           |
| 2Wiki    | 0.25 | 0.9435    | 0.9463             | 0.9500             | +0.0037    | no              | 0.021           |
| HotpotQA | 0.5  | 0.9505    | 0.9542             | 0.9575             | +0.0033    | no              | 0.022           |
| HotpotQA | 0.25 | 0.9505    | 0.9522             | 0.9550             | +0.0028    | no              | 0.011           |
| MetaQA   | 0.5  | 0.6612    | 0.7265             | 0.7432             | +0.0167    | yes             | 0.044           |
| MetaQA   | 0.25 | 0.6612    | 0.6999             | 0.7032             | +0.0033    | no              | 0.020           |
| MuSiQue  | 0.5  | 0.9635    | 0.9738             | 0.9775             | +0.0037    | no              | 0.049           |
| MuSiQue  | 0.25 | 0.9635    | 0.9693             | 0.9730             | +0.0037    | no              | 0.022           |
| SQuAD    | 0.5  | 0.9875    | 0.9895             | 0.9910             | +0.0015    | no              | 0.020           |
| SQuAD    | 0.25 | 0.9875    | 0.9885             | 0.9895             | +0.0010    | no              | 0.009           |
| WebQSP   | 0.5  | 0.7646    | 0.8118             | 0.8118             | -0.0000    | no              | 0.135           |
| WebQSP   | 0.25 | 0.7646    | 0.7895             | 0.8013             | +0.0118    | no              | 0.064           |

The honest reading is a split one, and it refines the mechanism rather than
threatening it.  MOST of the halo`s value comes from the CANDIDATE RELATION: a
randomly scored subset of the core`s own 1-hop neighbours already recovers
64% of the halo gain on average (range 37-100% over the 12 cells), while
overlapping the real halo by only 4% of its nodes.
The parameter-free score is the smaller remaining increment: positive in 35 of 36 corpus-seed arms and negative in 1,
but it reaches per-corpus significance only on MetaQA.  So it is a real but modest
refinement, not the source of the effect -- and nothing here is tuned per corpus,
since the same boundary-mass rule and the same beta are used everywhere.

The one cell where the score buys nothing at all is WebQSP at beta=0.5.
That is also the cell with by far the highest overlap between the two arms
(Jaccard 0.135 against a 0.009-0.049 range elsewhere), i.e. the quota
is large enough there that the two rankings are choosing from the same
saturated pool and end up picking much the same nodes.  The candidate relation
still delivers its full gain in that cell; only the ordering stops mattering.

This does NOT reduce to `reading more nodes helps`: the random arm is still a
GRAPH object -- 1-hop neighbours of the selected cores, at an identical node
budget -- and P15 already shows that spending the same budget on more hard
partitions does worse.  What it says is that the overlap FORMULATION is the
load-bearing part and the ranking rule is a refinement.  That is worth knowing
before anyone tries to tune the score: the headroom there is small, and the
headroom in the candidate relation is where the effect lives.

## P8  THE EXPLOSION IS REAL, SO BOUNDED HALOS ARE THE OPERATIVE FORM

Measured before any retrieval evaluation, as the program requires.  Unbounded 1-hop
overlap is unusable at membership level on every corpus:

| corpus   | STRUCT | NERX | KNN  | FULL_C | FULL_C max multiplicity | multi-family pair frac |
|----------|--------|------|------|--------|-------------------------|------------------------|
| 2Wiki    | 3.02   | 6.29 | 3.25 | 9.97   | 658                     | 0.062                  |
| HotpotQA | 9.43   | 6.61 | 3.68 | 16.93  | 5074                    | 0.0445                 |
| MetaQA   | 4.07   | 5.32 | 2.68 | 9.18   | 397                     | 0.1026                 |
| MuSiQue  | 2.74   | 4.56 | 2.68 | 7.29   | 131                     | 0.099                  |
| SQuAD    | 9.08   | 7.39 | 2.68 | 15.86  | 190                     | 0.0809                 |
| WebQSP   | 3.18   | 1.32 | 2.06 | 4.42   | 4964                    | 0.0376                 |

On HotpotQA the unbounded FULL_C halo does reach ALL_REQUIRED 0.998 -- at x50 node
exposure, with some nodes in EVERY block's halo (max multiplicity = the block count).
Coverage bought that way is worthless, which is what sends the search to bounded halos.
The three families are largely disjoint (3.8-10.3% of halo pairs carry more than one
provenance bit), so the union is genuinely additive rather than three names for one edge
set.


## P1-P5  REAL PARTITIONERS: THE ALGORITHM IS NOT THE LEVER, THE REPRESENTATION IS

KaHIP and Mt-KaHyPar both built and ran on Modal CPU.  The previous program's inability to
run them was environmental, and the two causes are worth recording: KaHIP's CMakeLists
calls `find_package(MPI)` unconditionally, so without MPI no Makefile is generated at all
and `deploy/` ends up holding only a header; and its default `-march=native` produces a
binary that dies with SIGILL because Modal builds and runs on different CPUs.

Balance tolerance is matched exactly, which required care: production `pymetis ufactor=30`
means 1 + 30/1000 = 3% in METIS, not 30%.  Measured production max/mean is 1.025-1.030,
confirming 3%, so KaHIP `--imbalance=3` and Mt-KaHyPar `epsilon=0.03` are the matched
settings.

| partitioner x representation                     | MetaQA  | WebQSP  | 2Wiki   | MuSiQue | SQuAD   | HotpotQA | worst   | macro   | sig reg | sig gains |
|--------------------------------------------------|---------|---------|---------|---------|---------|----------|---------|---------|---------|-----------|
| H2_KAHIP_CONNECTED__G0_TOPOLOGY_C                | --      | --      | --      | -0.0005 | -0.0015 | --       | -0.0015 | -0.0010 | 0       | 0         |
| H1_KAHIP_STRONG__G0_TOPOLOGY_C                   | -0.0020 | --      | --      | +0.0005 | +0.0010 | --       | -0.0020 | -0.0002 | 0       | 0         |
| H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | +0.0030 | +0.0078 | +0.0015 | -0.0050 | +0.0005 | +0.0005  | -0.0050 | +0.0014 | 0       | 0         |
| H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | +0.1201 | +0.0571 | -0.0045 | -0.0005 | -0.0120 | +0.0140  | -0.0120 | +0.0290 | 1       | 3         |
| H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | +0.1286 | +0.0599 | -0.0020 | -0.0010 | -0.0130 | +0.0150  | -0.0130 | +0.0312 | 1       | 3         |

Delta of F6 ALL@50 against the production METIS partition.  Significant means exact
McNemar p < 0.05 AND above that corpus's measured METIS reseed noise floor.

7 of the 30 partitioner-by-corpus cells above are blank. Every one of them is a KaHIP cell, and every one is a spend decision rather than a failure:

- `H2_KAHIP_CONNECTED__G0_TOPOLOGY_C`: no result on MetaQA, WebQSP, 2Wiki, HotpotQA.
- `H1_KAHIP_STRONG__G0_TOPOLOGY_C`: no result on WebQSP, 2Wiki, HotpotQA.

**Time, and that one IS a decision.**  KaHIP `strong` is single-threaded.
Its slowest completed cell is MetaQA at 5384 s on 0.78M directed edges.  Scaling linearly in edges -- optimistic, since `strong` runs several V-cycles and its local search also grows with nodes -- puts the largest corpora at 2Wiki ~3 h, HotpotQA ~33 h, WebQSP ~16 h for ONE of the two KaHIP cells.
That cost was spent on every corpus where it is affordable -- MetaQA, MuSiQue, SQuAD.
What was never launched is HotpotQA, WebQSP -- the largest corpora, where the estimate
above exceeds a working day per cell.
The remaining KaHIP cells (MetaQA H2_KAHIP_CONNECTED, 2Wiki H1_KAHIP_STRONG, 2Wiki H2_KAHIP_CONNECTED) were LAUNCHED and then
STOPPED ON THE USER`S INSTRUCTION once the H3/H4 representation rows were
complete on all six corpora.  That is a decision about spend, and it is
recorded rather than hidden: those cells have no result and never will in
this program.
Neither omission is load-bearing, because a strictly STRONGER version of the
same test already exists on every corpus.
KaHIP would answer `does a better hard
partitioner help at the SAME representation?`  `H3_MTKAHYPAR_GRAPH x
G0_TOPOLOGY_C` answers exactly that, with a partitioner that reaches a LOWER
cut than production METIS on the identical graph -- and it moves no corpus
significantly.  KaHIP is null wherever it did run.  For these blanks to change
a verdict they would have to behave unlike the partitioner they replicate AND
unlike their own results elsewhere.

The cells that actually move corpora -- the REPRESENTATION cells H3/H4 -- are not
rationed by either constraint on the corpora where the effect lives.

One runner note, because it was first misdiagnosed and the wrong lesson is expensive.
HotpotQA`s Mt-KaHyPar cells were killed three times by Modal`s runner heartbeat, which
looked like an out-of-memory kill.  It was not: the largest peak RSS measured anywhere
is 43.8 GB (WebQSP) inside a 64 GB container, and HotpotQA H3 x G1 completed there at
833 s using 41.4 GB.  The cause is the GIL.  mtkahypar`s Python binding holds it for
the whole of `hg.partition(ctx)`, and the pure-Python graph construction above it holds
it too, so on a call longer than 900 s the heartbeat thread never runs and the task is
killed as unresponsive.  KaHIP never failed despite a 2389 s partition because it goes
through `subprocess.run`, which releases the GIL.  Running Mt-KaHyPar in a child
process fixed it -- HotpotQA H3 x G0 then completed at 939 s -- and it is a plumbing
change that cannot alter a partition.

The decisive control is `H3_MTKAHYPAR_GRAPH x G0_TOPOLOGY_C`: a different, higher-quality
partitioner on EXACTLY the graph production METIS saw.  It has no significant gain and no
significant regression anywhere -- swapping the algorithm alone changes nothing, even
though it achieves a lower edge cut.  Cut quality does not predict retrieval utility.

What DOES move a corpus is changing the representation.  Feeding the same local
neighbourhoods to the partitioner as GROUPS -- clique-expanded (G1) or as genuine
hyperedges (G2) -- rather than as the flat topology (G0) produces the only significant
gains in the whole matrix.  They are NOT a knowledge-base effect: HotpotQA is free text
and converts significantly under G1, which is the same refutation Part 2 of this line
already recorded, reproduced here with a real hypergraph partitioner.


- clique-expanded local groups (G1): significant gains on MetaQA +0.1201, WebQSP +0.0571, HotpotQA +0.0140; significant regressions on SQuAD -0.0120.
- true hyperedges (G2): significant gains on MetaQA +0.1286, WebQSP +0.0599, HotpotQA +0.0150; significant regressions on SQuAD -0.0130.

The user's hypothesis -- that a real hypergraph partitioner should beat approximating a
group with pairwise clique edges -- holds where it was supposed to.  G2 beats its own
clique expansion on MetaQA and WebQSP and 2Wiki and HotpotQA -- including the corpus the group structure was
meant to help: MetaQA +0.1286 against +0.1201, hop3 0.5916.
But both representations regress SQuAD, and neither passes the universal gate.  This is
the same wall the previous program hit with P4_CE_LOCAL_ONLY -- larger in both
directions, and reached this time with the actual algorithms rather than an
approximation of them.


### P4  containment is associated with utility here, and still cannot be the objective

The program asks for the minimum number of blocks the required nodes occupy to be
REPORTED, not optimised.  Reporting it turns up something worth stating plainly,
because it partly revises the previous program: across the 22 cells the change in mean min-blocks and the change in F6 ALL@50 ARE monotonically associated, Spearman rho = -0.490, p = 0.0205.
The two big representation wins are exactly the two partitions that pack a query's
required nodes into fewer blocks -- on MetaQA H4 moves the mean from 5.78 to 4.57
and single-block queries from 46.2% to 61.0%, on WebQSP 3.66 to 3.08 and 52.1% to
56.3% -- while the cells that lose (MuSiQue G1/G2) scatter them further.

That association is real, and it is still not a usable objective, for a reason
visible in the same table: on SQuAD every query has exactly ONE required node, so
min-blocks is 1 under EVERY partition and cannot move at all -- yet SQuAD's utility
spans 0.0140 across these cells, and SQuAD is the corpus that blocks the universal
gate.  A containment objective is blind precisely where the decision is made.  The
previous program's HotpotQA counter-example (containment worse on every statistic,
utility significantly better) still stands alongside this.  So: report it, do not
rank on it -- which is what the program instructed, and now for a measured reason.


### P3  the universal hyperedge cap is what breaks SQuAD

The program requires ONE identical cap on every corpus, and the frozen value is 25.
Reporting the size distribution BEFORE that cap, as the program asks, shows the cap
is not neutral -- a closed neighbourhood over the cap is dropped whole, and how much
that costs varies by two orders of magnitude across corpora:

| corpus   | mean size | p99 | max    | hyperedges over cap | PINS over cap | G1 edges / G0 edges |
|----------|-----------|-----|--------|---------------------|---------------|---------------------|
| MetaQA   | 6.5       | 28  | 4,181  | 1.2%                | 19.7%         | x1.95               |
| WebQSP   | 5.2       | 36  | 41,606 | 1.6%                | 36.0%         | x2.55               |
| 2Wiki    | 5.6       | 21  | 38,593 | 0.7%                | 28.0%         | x1.20               |
| MuSiQue  | 11.4      | 121 | 1,487  | 8.9%                | 52.8%         | x1.19               |
| SQuAD    | 108.3     | 514 | 2,335  | 95.6%               | 99.1%         | x0.13               |
| HotpotQA | 14.6      | 62  | 46,892 | 3.2%                | 43.5%         | x1.68               |

SQuAD is not a little worse off, it is categorically different: its structural neighbourhoods
average 108 nodes, the cap discards 95.6% of them carrying 99.1% of the pins, and it
is the ONLY corpus where the group representation ends up SMALLER than the flat one
(0.13x the edges; every other corpus grows 1.19-2.55x).  The partitioner is therefore
handed an almost structureless SQuAD and balances it close to arbitrarily.

That matters for how the P1-P5 result should be read.  The single significant
regression blocking the hard-partition gate is SQuAD, on exactly the two capped
group representations, while the UNCAPPED G0 control is null there (+0.0005).  So the
honest statement is not that grouped representations do not suit SQuAD -- it is that
a single universal cap cannot serve a corpus whose neighbourhoods are 20x larger than
everyone else's.  Fixing that would mean a per-corpus or size-adaptive cap, which this
program forbids, so the gate result stands as measured and this is flagged as the
obvious next experiment rather than quietly worked around.

### P16  best hard core x halo

A halo can only ADD coverage, so the obvious question is whether one repairs the H4
core's regressions.  The same halo recipe was run on the H4 core and on the production
METIS core, on every corpus where an H4 partition exists:

| corpus  | halo           | METIS core | H4 core | H4 - METIS | METIS R | H4 R  |
|---------|----------------|------------|---------|------------|---------|-------|
| MetaQA  | O0_CORE        | 0.6612     | 0.7898  | +0.1286    | 1.00    | 1.00  |
| MetaQA  | O1_STRUCT_b1.0 | 0.8293     | 0.8323  | +0.0030    | 1.99    | 2.00  |
| MetaQA  | O4_FULL_C_b0.5 | 0.7432     | 0.8093  | +0.0661    | 1.50    | 1.50  |
| MetaQA  | O4_FULL_C_b1.0 | 0.7968     | 0.8233  | +0.0265    | 2.00    | 2.00  |
| MetaQA  | O4_FULL_C      | 0.9384     | 0.9324  | -0.0060    | 9.18    | 10.98 |
| 2Wiki   | O0_CORE        | 0.9435     | 0.9415  | -0.0020    | 1.00    | 1.00  |
| 2Wiki   | O1_STRUCT_b1.0 | 0.9510     | 0.9435  | -0.0075    | 1.77    | 1.64  |
| 2Wiki   | O4_FULL_C_b0.5 | 0.9540     | 0.9510  | -0.0030    | 1.50    | 1.50  |
| 2Wiki   | O4_FULL_C_b1.0 | 0.9600     | 0.9595  | -0.0005    | 2.00    | 2.00  |
| 2Wiki   | O4_FULL_C      | 0.9970     | 0.9985  | +0.0015    | 9.97    | 11.53 |
| MuSiQue | O0_CORE        | 0.9635     | 0.9625  | -0.0010    | 1.00    | 1.00  |
| MuSiQue | O1_STRUCT_b1.0 | 0.9680     | 0.9705  | +0.0025    | 1.91    | 1.91  |
| MuSiQue | O4_FULL_C_b0.5 | 0.9775     | 0.9775  | +0.0000    | 1.50    | 1.50  |
| MuSiQue | O4_FULL_C_b1.0 | 0.9820     | 0.9860  | +0.0040    | 2.00    | 2.00  |
| MuSiQue | O4_FULL_C      | 0.9980     | 0.9985  | +0.0005    | 7.29    | 8.80  |
| SQuAD   | O0_CORE        | 0.9875     | 0.9745  | -0.0130    | 1.00    | 1.00  |
| SQuAD   | O1_STRUCT_b1.0 | 0.9890     | 0.9810  | -0.0080    | 2.00    | 2.00  |
| SQuAD   | O4_FULL_C_b0.5 | 0.9910     | 0.9760  | -0.0150    | 1.50    | 1.50  |
| SQuAD   | O4_FULL_C_b1.0 | 0.9915     | 0.9800  | -0.0115    | 2.00    | 2.00  |
| SQuAD   | O4_FULL_C      | 1.0000     | 0.9980  | -0.0020    | 15.86   | 36.69 |

On SQuAD the answer is no: the best BOUNDED halo (O1_STRUCT_b1.0) lifts the H4
core only to 0.9810,
still below the METIS core's 0.9875 with no halo at all.  The H4 partition
is also markedly more expensive to overlap -- its unbounded replication is 2.3-3.5x
METIS's on the same corpus and the same edge family, because optimising
connectivity (km1) leaves a boundary that costs more to cover.

The pairing is not worthless.  Under the promoted halo the two levers stack on MetaQA: the H4 core plus the
winning beta=0.5 halo reaches 0.8093 on MetaQA against 0.7432 for
the METIS core with the identical halo.  It loses on 2Wiki, SQuAD, so the pair inherits H4's SQuAD regression: the combination is not
universal, and the promotable substrate remains the METIS core plus the halo.

## P9  THE COVERAGE GAIN IS NOT A RANKING EFFECT

Core selection is computed once per corpus, before any halo object exists, and every cell
reuses the identical BASE and F6 block lists -- halo nodes cannot reach partition ranking
by construction, not merely by test.  The measured check: with an empty halo, 'all required
NODES fetched' is the same event as 'all gold PARTITIONS selected', so O0_CORE must equal
replay(CURRENT) exactly.  Any leak would break that identity.

| corpus   | O0_CORE ALL_REQUIRED | replay F6 ALL@50 | F6    | BASE  |
|----------|----------------------|------------------|-------|-------|
| MetaQA   | 0.6612               | 0.6612           | EXACT | EXACT |
| WebQSP   | 0.7646               | 0.7646           | EXACT | EXACT |
| 2Wiki    | 0.9435               | 0.9435           | EXACT | EXACT |
| MuSiQue  | 0.9635               | 0.9635           | EXACT | EXACT |
| SQuAD    | 0.9875               | 0.9875           | EXACT | EXACT |
| HotpotQA | 0.9505               | 0.9505           | EXACT | EXACT |

## P17  SHOULD THE HALO ALSO VOTE?

Phase 9 passing opens the optional question: the halo currently only changes what a
selected block pays out, so should halo nodes also be allowed to vote for a block
during ranking?  Only the membership map the router votes through changes; the fetch
rule is identical in both arms, so any difference is ranking and nothing else.  The
swapped path is guarded: with an EMPTY halo it must reproduce the frozen selection
exactly, and it does on every corpus.

### O4_FULL_C_b0.5

| corpus   | core-scored | halo-aware | delta   | gained/lost | p       | sig | exposure delta | identical top-50 |
|----------|-------------|------------|---------|-------------|---------|-----|----------------|------------------|
| MetaQA   | 0.7432      | 0.7437     | +0.0005 | 8/7         | 1.0e+00 | no  | +0.0001        | 5.3%             |
| WebQSP   | 0.8118      | 0.8111     | -0.0007 | 9/10        | 1.0e+00 | no  | -0.0008        | 4.1%             |
| 2Wiki    | 0.9540      | 0.9600     | +0.0060 | 16/4        | 1.2e-02 | yes | -0.0037        | 0.2%             |
| MuSiQue  | 0.9775      | 0.9800     | +0.0025 | 8/3         | 2.3e-01 | no  | -0.0003        | 1.1%             |
| SQuAD    | 0.9910      | 0.9910     | +0.0000 | 0/0         | 1.0e+00 | no  | +0.0012        | 14.9%            |
| HotpotQA | 0.9575      | 0.9570     | -0.0005 | 1/2         | 1.0e+00 | no  | -0.0003        | 15.8%            |

### O4_FULL_C_b0.25

| corpus   | core-scored | halo-aware | delta   | gained/lost | p       | sig | exposure delta | identical top-50 |
|----------|-------------|------------|---------|-------------|---------|-----|----------------|------------------|
| MetaQA   | 0.7032      | 0.7037     | +0.0005 | 6/5         | 1.0e+00 | no  | +0.0000        | 17.9%            |
| WebQSP   | 0.8013      | 0.8013     | +0.0000 | 7/7         | 1.0e+00 | no  | -0.0002        | 14.4%            |
| 2Wiki    | 0.9500      | 0.9530     | +0.0030 | 9/3         | 1.5e-01 | no  | -0.0009        | 3.5%             |
| MuSiQue  | 0.9730      | 0.9750     | +0.0020 | 7/3         | 3.4e-01 | no  | -0.0002        | 6.8%             |
| SQuAD    | 0.9895      | 0.9895     | +0.0000 | 0/0         | 1.0e+00 | no  | +0.0002        | 29.9%            |
| HotpotQA | 0.9550      | 0.9555     | +0.0005 | 2/1         | 1.0e+00 | no  | -0.0001        | 27.0%            |

Halo-aware ranking is significantly better on 2Wiki and significantly
worse on no corpus.

That is not universal, so following the program the ranking stays CORE-ONLY.
The mechanism is worth stating: halo-aware ranking reshuffles the selection
heavily -- the top-50 is identical on only a few percent of queries -- and
still lands in essentially the same place.  The halo's value is what it
FETCHES, not what it says about which block is worth reading.  Keeping ranking
core-only also keeps the Phase-9 guarantee, which is worth more than a gain on
one corpus: coverage cannot be a ranking artefact if ranking never sees the
halo.


### Cross-implementation check

Phase 17 needed its own coverage function, written separately from the Phase-10
one.  Run on the promoted cell with the frozen selection it reproduces the headline
number on every corpus, so the result does not rest on a single implementation:

| corpus   | Phase 10 | Phase 17 (independent) | agreement |
|----------|----------|------------------------|-----------|
| MetaQA   | 0.7432   | 0.7432                 | EXACT     |
| WebQSP   | 0.8118   | 0.8118                 | EXACT     |
| 2Wiki    | 0.9540   | 0.9540                 | EXACT     |
| MuSiQue  | 0.9775   | 0.9775                 | EXACT     |
| SQuAD    | 0.9910   | 0.9910                 | EXACT     |
| HotpotQA | 0.9575   | 0.9575                 | EXACT     |

## P14  WHICH FAMILY THE HALO SHOULD COME FROM

The best single family varies by corpus, which is precisely why the union is the universal
answer: it is the only choice that does not require picking per corpus.

| corpus (beta=0.5) | NERX - STRUCT | KNN - STRUCT | best single family | FULL_C - best single |
|-------------------|---------------|--------------|--------------------|----------------------|
| 2Wiki             | +0.0065       | +0.0100      | O3_KNN             | -0.0040              |
| HotpotQA          | +0.0065       | +0.0085      | O3_KNN             | -0.0030              |
| MetaQA            | -0.0981       | -0.1326      | O1_STRUCT          | -0.0596              |
| MuSiQue           | +0.0125       | +0.0100      | O2_NERX            | -0.0020              |
| SQuAD             | +0.0025       | +0.0025      | O2_NERX            | +0.0000              |
| WebQSP            | -0.0500       | -0.0458      | O1_STRUCT          | -0.0162              |

## P19  UNIVERSAL LEADERBOARD

Every cell, ranked by the universal selection rule applied verbatim and
lexicographically: zero significant cross-corpus regressions first, then max worst-corpus
delta, then max macro, then max MetaQA hop3, then min exposure.  `P15` is the
matched-exposure gate -- a cell may not significantly lose to simply reading more hard
partitions at the same unique-node budget.

| cell            | worst   | macro   | sig reg | P15 win/loss | P15  | MetaQA h2 | MetaQA h3 | mean exposure | max R |
|-----------------|---------|---------|---------|--------------|------|-----------|-----------|---------------|-------|
| O4_FULL_C_b0.5  | +0.0035 | +0.0274 | 0       | 1/0          | PASS | 0.8063    | 0.4264    | x1.304        | 1.50  |
| O4_FULL_C_b0.25 | +0.0020 | +0.0169 | 0       | 2/0          | PASS | 0.7763    | 0.3378    | x1.145        | 1.25  |
| O2_NERX_b0.25   | +0.0015 | +0.0114 | 0       | 1/0          | PASS | 0.7808    | 0.2973    | x1.172        | 1.25  |
| O4_FULL_C       | +0.0125 | +0.1070 | 0       | 0/1          | --   | 0.9970    | 0.8183    | x13.744       | 16.93 |
| O2_NERX         | +0.0105 | +0.0436 | 0       | 1/2          | --   | 0.8964    | 0.5435    | x3.059        | 7.39  |
| O1_STRUCT       | +0.0080 | +0.0999 | 0       | 1/2          | --   | 0.9940    | 0.7763    | x12.585       | 9.43  |
| O3_KNN          | +0.0080 | +0.0342 | 0       | 0/2          | --   | 0.8093    | 0.4039    | x2.363        | 3.68  |
| O2_NERX_b1.0    | +0.0060 | +0.0229 | 0       | 0/2          | --   | 0.8258    | 0.3574    | x1.594        | 2.00  |
| O3_KNN_b1.0     | +0.0060 | +0.0223 | 0       | 0/2          | --   | 0.7643    | 0.3378    | x1.708        | 2.00  |
| O4_FULL_C_b1.0  | +0.0040 | +0.0439 | 0       | 1/1          | --   | 0.8408    | 0.5526    | x1.607        | 2.00  |
| O2_NERX_b0.5    | +0.0035 | +0.0159 | 0       | 0/0          | --   | 0.7958    | 0.3213    | x1.327        | 1.50  |
| O3_KNN_b0.5     | +0.0035 | +0.0114 | 0       | 0/2          | --   | 0.7402    | 0.2748    | x1.369        | 1.50  |
| O3_KNN_b0.25    | +0.0030 | +0.0075 | 0       | 0/1          | --   | 0.7357    | 0.2673    | x1.183        | 1.25  |
| O1_STRUCT_b1.0  | +0.0015 | +0.0482 | 0       | 2/4          | --   | 0.8363    | 0.6562    | x1.429        | 2.00  |
| O1_STRUCT_b0.5  | +0.0010 | +0.0359 | 0       | 2/1          | --   | 0.8138    | 0.5991    | x1.21         | 1.50  |
| O1_STRUCT_b0.25 | +0.0000 | +0.0250 | 0       | 2/1          | --   | 0.7943    | 0.5105    | x1.102        | 1.25  |

3 cells pass every gate.  The ordering above is the rule, not a preference:
nothing is chosen per corpus at any point.


## P18  MODAL COST

All partitioning ran on Modal CPU containers (16 CPU, 64 GB).  No GPU was requested anywhere: the string
`gpu=` appears 0 times in the runner source, which is
asserted rather than claimed.  KaHIP is compiled once into a cached image layer and
never rebuilt per corpus.

| partitioner                  | cells | total wall s | slowest cell s | peak RSS MB |
|------------------------------|-------|--------------|----------------|-------------|
| H1_KAHIP_STRONG              | 3     | 7056         | 5384           | 581         |
| H2_KAHIP_CONNECTED           | 2     | 2902         | 2389           | 1212        |
| H3_MTKAHYPAR_GRAPH           | 12    | 2716         | 940            | 41399       |
| H4_MTKAHYPAR_TRUE_HYPERGRAPH | 6     | 2704         | 1767           | 43786       |

23 partitioning cells, 15378 s wall, 68.35 CPU-hours billed.  The cost shape is worth recording, because it is what bounds the matrix:
KaHIP `strong` is the slow, small one -- up to 5384 s per cell in 1212 MB -- and Mt-KaHyPar is the fast, enormous one: 1767 s but up to 43 GB.  Both scale with edges, which is what bounds this matrix rather than a footnote: see the coverage note under P1-P5.
Neither cost bought a universal partition.


## VERDICTS

- `OVERLAP_VERDICT` = **B. BOUNDED_OVERLAP_RECOVERED**.  Unbounded overlap is unusable
  (P8: R up to 16.9, exposure up to x50), but a boundary-mass-bounded halo at beta <= 0.5
  improves all 6 corpora at x1.304 exposure and beats reading deeper where it matters.
- `PARTITIONER_VERDICT` = **E. HARD_PARTITIONER_NOT_THE_MAIN_ISSUE**.  At matched
  representation a state-of-the-art partitioner is indistinguishable from production
  METIS; the group REPRESENTATION is what moves corpora, and the true hypergraph
  beats its own clique expansion where the group structure matters (verdict C's
  mechanism is real).  It still fails cross-corpus, so the hard-partition verdict
  stands -- but with a specific, measured caveat: the single blocking regression is
  SQuAD, and P3 traces it to the mandated universal hyperedge cap discarding 99% of
  SQuAD's structural pins.  A size-adaptive cap is the obvious next experiment and
  is outside this program's rules.
- `FINAL VERDICT` = **B. UNIVERSAL_OVERLAP_SUBSTRATE_FOUND**.  No hard partition passed the
  universal gate; the bounded overlap substrate did.

The user's framing is the right one: the earlier result never showed that a universally
good partitioning is impossible, only that the best hard-partition recipe was not
universal.  Removing the hard constraint phi(v) = P_i is what unlocked it.


## KNOWN RISK, STATED NOT HIDDEN

This program measures L1 coverage only.  The previous program established
`L1_GAIN_SURVIVES_FROZEN_L2 = NO` for a different L1 change: pool composition shifts can
invert downstream even when coverage improves, and 54-78% of that loss fell on queries
whose coverage was UNCHANGED.  Testing it here would require touching L2, which this
program forbids, so L2 survival of this gain is UNTESTED and must be the first thing
checked before any promotion.  L1 is NOT frozen and nothing here is promoted.
