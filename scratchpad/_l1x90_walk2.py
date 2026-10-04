"""Per-layer block masses of the scheduled frontier walk, so that layer weightings can be evaluated
without re-walking.  Scratch module of the L1_P90_EXPLOIT lane (no contract file touched)."""
import itertools

import numpy as np

import _l1x90_typed as TY


def layered_walk(C, T, questions, seed_rows, H=3, variants=("hard", "mix", "hardperm"), log_every=0):
    """returns (L, info): L[variant] = float32 array (nq, H+1, npart); layer 0 = seeds.  info[i] has
    'order' (relation labels in schedule order), 'k' = len(order), 'anchored'."""
    B = C.blockmat().T.tocsr()
    nq = seed_rows.shape[0]
    N = C.N
    L = {v: np.zeros((nq, H + 1, C.npart), np.float32) for v in variants}
    cache = {}

    def P_of(allowed, mix):
        key = (tuple(sorted(allowed)), mix)
        if key not in cache:
            if not allowed:
                cache[key] = T.transition(set()).T.tocsr()
            elif mix:
                cache[key] = T.transition(set(allowed)).T.tocsr()
            else:
                cache[key] = T.transition_hard(set(allowed)).T.tocsr()
            if len(cache) > 512:
                cache.pop(next(iter(cache)))
        return cache[key]

    def walk(r, order, mix):
        layers = [np.asarray(B @ r).ravel()]
        x = r.copy()
        seen = r > 0
        for k in range(1, H + 1):
            allowed = set(order[:k]) if order else set()
            x = P_of(allowed, mix) @ x
            x[seen] = 0.0
            s = float(x.sum())
            if s > 0:
                x /= s
            seen |= x > 0
            layers.append(np.asarray(B @ x).ravel())
        return np.stack(layers, 0)

    info = []
    for i in range(nq):
        sd = seed_rows[i]
        ok = sd >= 0
        order, anchored = TY.schedule(T, questions[i], int(sd[0]) if ok.any() else None) if T.has_rel else ([], False)
        info.append({"order": [T.vocab[r] for r in order], "k": len(order), "anchored": bool(anchored)})
        r = np.zeros(N, np.float32)
        if ok.any():
            r[sd[ok].astype(np.int64)] = 1.0 / float(ok.sum())
        if "hard" in variants:
            L["hard"][i] = walk(r, order, False)
        if "mix" in variants:
            L["mix"][i] = walk(r, order, True)
        if "hardperm" in variants:
            if len(order) > 1:
                perms = list(itertools.permutations(order))
                L["hardperm"][i] = sum(walk(r, list(od), False) for od in perms) / len(perms)
            else:
                L["hardperm"][i] = L["hard"][i] if "hard" in variants else walk(r, order, False)
        if log_every and (i + 1) % log_every == 0:
            print("  walked %d/%d" % (i + 1, nq), flush=True)
    return L, info


def weight_layers(L, info, mode="equal", beta=0.5, H=3):
    """mode 'equal': sum of all layers; 'answer': layer k=len(order) gets 1, seed layer 1, other layers beta;
    'answer_only': layer k and seed only; 'geometric': layer j gets beta**(H-j)."""
    nq = L.shape[0]
    W = np.ones((nq, H + 1), np.float32)
    for i, inf in enumerate(info):
        k = min(max(inf["k"], 1), H)
        if mode == "answer":
            W[i, 1:] = beta
            W[i, k] = 1.0
        elif mode == "answer_only":
            W[i, 1:] = 0.0
            W[i, k] = 1.0
        elif mode == "answer_tail":
            W[i, 1:] = beta
            W[i, k:] = 1.0
        elif mode == "geometric":
            W[i] = beta ** (H - np.arange(H + 1))
    return np.einsum("qh,qhp->qp", W, L)
