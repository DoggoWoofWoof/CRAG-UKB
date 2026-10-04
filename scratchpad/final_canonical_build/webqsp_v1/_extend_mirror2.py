import io, os
P = "scratchpad/final_canonical_build/mirror_audit.py"
src = io.open(P, encoding="utf-8").read()
A = '            "ROG_UNION_REBUILD.json", "V1_TABLES_REPORT.json", "V1_RELATION_LABEL_ANALYSIS.json",'
N = A + '''
            # section 7 adaptive labels + section 3 classification: the freeze document, the run,
            # the audit and the node_kind freeze marker.  The freeze doc in particular must be in
            # the tracked tree -- it is the record that the thresholds predate the results.
            "V1_RELATION_LABELS.json", "V1_SECTION3_CLASSIFIER_FREEZE.json",
            "V1_CLASSIFICATION_REPORT.json", "V1_CLASSIFICATION_AUDIT.json",
            "V1_NODE_KIND_FROZEN.json",'''
assert src.count(A) == 1
io.open(P + ".tmp", "w", encoding="utf-8", newline="\n").write(src.replace(A, N))
os.replace(P + ".tmp", P)
print("extended")
