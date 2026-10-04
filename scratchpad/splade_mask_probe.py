# -*- coding: utf-8 -*-
"""Does the PHASE_C splade path (no attention mask) differ from the REV2 path (masked)?

canonical_encode.py pools log1p(relu(logits)) over ALL positions; encode_rev2_delta.py zeroes
padding first. Both wrote into the same corpora, so if the two disagree then some rows of one
dataset live in a different space from the rest of the SAME dataset -- and, worse, which space
a row lands in would depend on the lengths of the OTHER texts that happened to share its batch.

This does not argue about it. It runs both paths on one real batch of mixed-length canonical
texts and reports the actual difference.
"""
import io, json, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

NAME = "naver/splade-cocondenser-ensembledistil"
DS = sys.argv[1] if len(sys.argv) > 1 else "musique"
N = 24

texts = []
with io.open("data/final_canonical/%s/nodes.jsonl" % DS, encoding="utf-8") as f:
    for ln in f:
        o = json.loads(ln)
        t = o.get("text") or ""
        if t.strip():
            texts.append(t)
        if len(texts) >= N:
            break
lens = [len(t) for t in texts]
print("%s: %d texts, char len min=%d max=%d" % (DS, len(texts), min(lens), max(lens)))

tok = AutoTokenizer.from_pretrained(NAME)
mdl = AutoModelForMaskedLM.from_pretrained(NAME).eval()

enc = tok(texts, return_tensors="pt", truncation=True, max_length=256, padding=True)
with torch.no_grad():
    lg = mdl(**enc).logits
w = torch.log1p(torch.relu(lg))
unmasked = w.max(dim=1).values.numpy()                                   # canonical_encode.py
masked = (w * enc["attention_mask"].unsqueeze(-1)).max(dim=1).values.numpy()   # rev2

tl = enc["attention_mask"].sum(1).numpy()
print("token len min=%d max=%d  padded rows=%d/%d"
      % (tl.min(), tl.max(), int((tl < tl.max()).sum()), len(texts)))

d = np.abs(unmasked - masked)
nz_u = (unmasked > 0).sum(1)
nz_m = (masked > 0).sum(1)
print("max abs diff        %.6e" % d.max())
print("rows that differ    %d/%d" % (int((d.max(1) > 0).sum()), len(texts)))
print("nnz unmasked/masked %d / %d  (delta %d)" % (nz_u.sum(), nz_m.sum(), nz_u.sum()-nz_m.sum()))

cu = unmasked / np.maximum(np.linalg.norm(unmasked, axis=1, keepdims=True), 1e-9)
cm = masked / np.maximum(np.linalg.norm(masked, axis=1, keepdims=True), 1e-9)
cos = (cu * cm).sum(1)
print("cosine(unmasked,masked) min=%.8f mean=%.8f" % (cos.min(), cos.mean()))

# The batch-dependence question: encode the SHORTEST text alone (no padding at all) and
# compare with its in-batch unmasked vector. If those differ, the PHASE_C store is not a
# function of the text -- it is a function of the text AND its shard neighbours.
i = int(np.argmin(tl))
e1 = tok([texts[i]], return_tensors="pt", truncation=True, max_length=256, padding=True)
with torch.no_grad():
    l1 = mdl(**e1).logits
solo = torch.log1p(torch.relu(l1)).max(dim=1).values.numpy()[0]
d1 = np.abs(solo - unmasked[i])
c1 = float(np.dot(solo, unmasked[i]) /
           max(np.linalg.norm(solo) * np.linalg.norm(unmasked[i]), 1e-9))
print("\nSHORTEST row (%d tokens, batch max %d):" % (tl[i], tl.max()))
print("  solo-vs-inbatch max abs diff %.6e   cosine %.8f" % (d1.max(), c1))
print("  nnz solo=%d  in-batch unmasked=%d  masked=%d"
      % (int((solo > 0).sum()), int(nz_u[i]), int(nz_m[i])))

# The claim "masking IS what encoding this text alone means" is the whole basis for calling one
# path correct and the other contaminated. nnz agreeing is suggestive, not proof, so compare the
# values: encode every text solo (batch of 1, no padding anywhere) and diff against the masked
# in-batch vectors.
solos = []
for t in texts:
    e = tok([t], return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        solos.append(torch.log1p(torch.relu(mdl(**e).logits)).max(dim=1).values.numpy()[0])
S = np.stack(solos)
print("\nSOLO vs MASKED   max abs diff %.6e   rows differing %d/%d"
      % (np.abs(S - masked).max(), int((np.abs(S - masked).max(1) > 1e-6).sum()), len(texts)))
print("SOLO vs UNMASKED max abs diff %.6e   rows differing %d/%d"
      % (np.abs(S - unmasked).max(), int((np.abs(S - unmasked).max(1) > 1e-6).sum()), len(texts)))
