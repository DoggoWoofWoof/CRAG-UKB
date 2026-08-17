"""Fixed open-reader QA harness for CONSISTENT end-to-end EM/F1 across ALL systems (CRAG + every baseline).
One frozen small open LLM reader (default Qwen2.5-1.5B-Instruct, no proprietary API) generates an answer from
each system's top-k retrieved passages; we score EM/F1 vs gold. The SAME reader + prompt is used for every
system so the only variable is retrieval quality. Runnable locally (slow) or as a Modal task (fast)."""
import re
import string
import logging
from collections import Counter

log = logging.getLogger(__name__)
_MODEL = _TOK = None
DEFAULT_READER = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT = ("Answer the question using ONLY the passages below. Reply with the short answer span only — no "
          "explanation.\n\nPassages:\n{ctx}\n\nQuestion: {q}\nAnswer:")


def _load_reader(model_name=DEFAULT_READER, device=None):
    global _MODEL, _TOK
    if _MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _TOK = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=(torch.float16 if device == "cuda" else torch.float32)).to(device).eval()
        log.info("[qa-reader] loaded %s on %s", model_name, device)
    return _MODEL, _TOK


def read(query, passages, max_ctx=5, max_new=32):
    """Generate a short answer from the top passages (fixed reader)."""
    import torch
    model, tok = _load_reader()
    ctx = "\n".join(f"[{i+1}] {p[:400]}" for i, p in enumerate(passages[:max_ctx]))
    msgs = [{"role": "user", "content": PROMPT.format(ctx=ctx, q=query)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---- SQuAD-style EM / F1 (standard normalization) ----
def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, golds):
    p = _norm(pred)
    return float(any(p == _norm(g) for g in golds))


def f1(pred, golds):
    best = 0.0
    pt = _norm(pred).split()
    for g in golds:
        gt = _norm(g).split()
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0 or not pt or not gt:
            best = max(best, 0.0); continue
        prec, rec = ns / len(pt), ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def score_system(queries, passages_per_q, gold_answers):
    """queries: list[str]; passages_per_q: list[list[str]] (a system's top-k doc TEXTS); gold_answers: list[list[str]].
    Returns dict(EM, F1, n). Same reader for every system -> consistent."""
    ems, f1s = [], []
    for q, ps, golds in zip(queries, passages_per_q, gold_answers):
        pred = read(q, ps)
        ems.append(em(pred, golds)); f1s.append(f1(pred, golds))
    import numpy as np
    return {"EM": round(100 * np.mean(ems), 1), "F1": round(100 * np.mean(f1s), 1), "n": len(ems)}
