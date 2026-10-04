import io
p = "scratchpad/_l1ep_verdict.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)

rep('''        ("Q11. If partition construction matters, WHAT property of it matters?",
         "Second-order locality with degree-normalised strength -- and neither ingredient alone. "
         "On MetaQA: topology C unweighted is the baseline; adding the NER artifact's own stored "
         "1/df weights to it does nothing (-0.0050); closing local neighbourhoods into cliques "
         "with UNIT weights does nothing (-0.0005); NER hyperedges alone do nothing (-0.0010). "
         "Closing local neighbourhoods into cliques WITH the canonical 1/(|e|-1) weight gains "
         "+0.0966, hop3 0.2583 -> 0.4940 and hop2 0.7297 -> 0.7748 -- about 46 standard "
         "deviations of the METIS reseed noise floor. It does not generalise: the same rule is "
         "ns-negative on 2Wiki and MuSiQue and significantly negative on SQuAD "
         "(-0.0095, p = 0.0066)."),''',
    '''        ("Q11. If partition construction matters, WHAT property of it matters?",
         "Second-order locality with degree-normalised strength -- and neither ingredient alone. "
         "On MetaQA: topology C unweighted is the baseline; adding the NER artifact's own stored "
         "1/df weights to it does nothing (-0.0050); closing local neighbourhoods into cliques "
         "with UNIT weights does nothing (-0.0005); NER hyperedges alone do nothing (-0.0010). "
         "Closing local neighbourhoods into cliques WITH the canonical 1/(|e|-1) weight gains "
         "+0.0966, hop3 0.2583 -> 0.4940 and hop2 0.7297 -> 0.7748. The isolation replicates on "
         "WebQSP, where the unweighted clique closure lands inside that corpus's own reseed noise "
         "floor (+0.0064 against sd 0.0077) and the weighted one converts +0.0423.\n\n"
         "It generalises across CORPUS CLASS, not across corpora. Under one rule, no per-corpus "
         f"tuning: {tbl} (* = significant). Both knowledge-base corpora gain significantly; no "
         "free-text corpus gains at all and SQuAD loses significantly (-0.0095, p = 0.0066). "
         "State the two KB gains at their own scales rather than as equals: MetaQA's +0.0966 is "
         "about 46 sd of its 0.0021 reseed noise floor, WebQSP's +0.0423 about 5.5 sd of its much "
         "larger 0.0077 floor (its PM3 reseed replicates span 0.7512-0.7710). Both clear the "
         "floor decisively and both are p < 1e-5 by exact McNemar, but only MetaQA is "
         "overwhelming relative to the variance of its own partitioner."),''')

rep('''         "**Partition construction on KB-style multi-hop; ranking everywhere else; not "
         "traversal.** The edge substrate -- what the router walks -- is closed: its entire "
         "dynamic range at exact P = 50 is about +/-0.004, for up to 4.3x the cost. Partition "
         "construction has roughly 25x that dynamic range on MetaQA (+0.0966) and is the only "
         "intervention in the whole L1 program that has ever moved MetaQA hop3 off 0.2583. On "
         "the text corpora the partitioning is already at its useful limit (every alternative is "
         "<= production and the P=50 oracle is 1.0000), so what remains there is ranking, not "
         "information. Genuinely missing information is the smallest term: only 2%-9% of missed "
         "needed partitions are unreachable by any family."),''',
    '''         "**Partition construction on the KB corpora; ranking on the text corpora; not "
         "traversal, and not missing information.** The answer is different for the two corpus "
         "classes, and that is the finding rather than a hedge.\n\n"
         "The edge substrate -- what the router actually walks -- is closed on all six: its "
         "entire dynamic range at exact P = 50 is about +/-0.004, for up to 4.3x the traversal "
         "cost. Partition construction has roughly 25x that dynamic range on MetaQA (+0.0966) "
         "and 10x on WebQSP (+0.0423), and is the only intervention in the whole L1 program that "
         "has ever moved MetaQA hop3 off 0.2583 (-> 0.4940). On the four text corpora the "
         "partitioning is already at its useful limit -- every alternative is <= production and "
         "the P = 50 oracle is 1.0000 -- so the residual there is ranking: the evidence is inside "
         "the budget and the selector does not pick it. Genuinely missing information is the "
         "smallest term everywhere: only 2%-9% of missed needed partitions are unreachable by "
         "any edge family."),''')
io.open(p, "w", encoding="utf-8").write(s)
print("OK")
