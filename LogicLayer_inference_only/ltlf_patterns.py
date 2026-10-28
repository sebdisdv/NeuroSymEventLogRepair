"""
Richer LTLf patterns + usefulness-driven rule mining.

Everything downstream (features, semantic loss, hard decoder) only needs a list of DFAs over the compact
activity alphabet 0..A-1, so any pattern that can be written as a DFA plugs in unchanged.

Pattern families (all DECLARE-style semantics on finite traces):
  basic    existence, init, end, response, precedence, chain_response, chain_precedence,
           responded_existence, not_succession                         (the original nine)
  declare  alternate_response/precedence/succession, succession, chain_succession, not_chain_succession,
           co_existence, not_coexistence, choice, exclusive_choice
  counts   at_least(a,k), at_most(a,k)  (k = 2,3 / 1,2,3)   -> cardinality constraints
  bounded  bounded_response(a,b,k)  "every a is followed by b within k events"
  ngram    factor_next_in(w,S)  "after the exact sequence w, the next event is in S"   (|w| = 2,3)
           forbidden_factor(w)  "the sequence w never occurs"                          (|w| = 2,3)
and boolean combinations via product()/negate().

mine_rich_rules() does not rank by support alone (that keeps trivially true rules such as
"existence of an activity present in every trace" that rule out nothing). It simulates masking on
training traces and greedily picks the rules that eliminate the most WRONG activities at masked
positions, under a budget on how often the TRUE activity would be eliminated.
"""
import itertools
from collections import Counter
import numpy as np
from ltlf_features import DFA, _table, build_dfa


# ---------------------------------------------------------------------------------------------
# DFA algebra
# ---------------------------------------------------------------------------------------------
def canon(delta, acc, name=""):
    """Reachable part, Moore-minimised, states renumbered by BFS (initial = 0): equal language <=> equal tables."""
    delta, acc = np.asarray(delta), np.asarray(acc, bool)
    A = delta.shape[1]
    order, seen = [0], {0}
    for q in order:
        for a in range(A):
            n = int(delta[q, a])
            if n not in seen: seen.add(n); order.append(n)
    idx = {q: i for i, q in enumerate(order)}
    d = np.array([[idx[int(delta[q, a])] for a in range(A)] for q in order]); ac = acc[order]
    blk = ac.astype(int)
    while True:
        ids = {}
        nb = np.array([ids.setdefault((int(blk[q]),) + tuple(blk[d[q]].tolist()), len(ids)) for q in range(len(d))])
        if len(ids) == len(set(blk.tolist())): break
        blk = nb
    rep = {}
    for q in range(len(d)): rep.setdefault(int(nb[q]), q)
    start = int(nb[0]); bfs, bseen = [start], {start}
    for b in bfs:
        for a in range(A):
            n = int(nb[d[rep[b], a]])
            if n not in bseen: bseen.add(n); bfs.append(n)
    bi = {b: i for i, b in enumerate(bfs)}
    nd = np.array([[bi[int(nb[d[rep[b], a]])] for a in range(A)] for b in bfs])
    return DFA(nd, np.array([ac[rep[b]] for b in bfs]), name)


def product(d1, d2, op, name=""):
    A = d1.delta.shape[1]
    states, order = {(0, 0): 0}, [(0, 0)]
    for (p, q) in order:
        for a in range(A):
            s = (int(d1.delta[p, a]), int(d2.delta[q, a]))
            if s not in states: states[s] = len(order); order.append(s)
    delta = np.array([[states[(int(d1.delta[p, a]), int(d2.delta[q, a]))] for a in range(A)] for p, q in order])
    f = (lambda x, y: x and y) if op == "and" else (lambda x, y: x or y)
    acc = [f(bool(d1.acc[p]), bool(d2.acc[q])) for p, q in order]
    return canon(delta, acc, name)


def negate(d, name=""):
    return canon(d.delta, ~d.acc, name or f"not({d.name})")


def signature(d):
    return (d.delta.shape, d.delta.tobytes(), d.acc.tobytes())


# ---------------------------------------------------------------------------------------------
# New templates
# ---------------------------------------------------------------------------------------------
def alternate_response(A, a, b):      # every a is followed by b, with no other a in between
    def f(q, x):
        if q == 0: return 1 if x == a else 0
        if q == 1: return 0 if x == b else (2 if x == a else 1)
        return 2
    return canon(_table(3, A, f), [1, 0, 0], f"alternate_response({a},{b})")


def alternate_precedence(A, a, b):    # every b is preceded by a, with no other b in between
    def f(q, x):
        if q == 0: return 1 if x == a else (2 if x == b else 0)
        if q == 1: return 0 if x == b else 1
        return 2
    return canon(_table(3, A, f), [1, 1, 0], f"alternate_precedence({a},{b})")


def not_chain_succession(A, a, b):    # a is never immediately followed by b
    def f(q, x):
        if q == 0: return 1 if x == a else 0
        if q == 1: return 2 if x == b else (1 if x == a else 0)
        return 2
    return canon(_table(3, A, f), [1, 1, 0], f"not_chain_succession({a},{b})")


def occurrences(A, a, lo, hi, name=""):   # number of a's in [lo, hi]  (hi=None: at least lo)
    cap = lo if hi is None else hi + 1
    f = lambda q, x: min(q + 1, cap) if x == a else q
    acc = [(q >= lo) if hi is None else (lo <= q <= hi) for q in range(cap + 1)]
    return canon(_table(cap + 1, A, f), acc, name or f"count({a},{lo},{hi})")


def bounded_response(A, a, b, k):     # every a is followed by b within k events
    dead = k + 1
    def f(q, x):
        if q == dead: return dead
        if x == b: return 0
        if q == 0: return k if x == a else 0
        return dead if q == 1 else q - 1
    return canon(_table(k + 2, A, f), [1] + [0] * (k + 1), f"bounded_response({a},{b},{k})")


def _kmp(A, w):
    m = len(w); fail = [0] * (m + 1)
    for i in range(1, m):
        k = fail[i]
        while k and w[i] != w[k]: k = fail[k]
        fail[i + 1] = k + (w[i] == w[k])
    d = np.zeros((m + 1, A), np.int64)
    for s in range(m + 1):
        for x in range(A):
            if s < m and w[s] == x: d[s, x] = s + 1
            else: d[s, x] = d[fail[s], x] if s > 0 else 0
    return d


def forbidden_factor(A, w):           # the exact sequence w never occurs
    m = len(w); d = _kmp(A, w); d[m, :] = m
    return canon(d, [True] * m + [False], f"forbidden_factor({tuple(w)})")


def factor_next_in(A, w, S):          # after every occurrence of w, if any event follows it is in S
    m = len(w); k = _kmp(A, w); dead = m + 1
    d = np.zeros((m + 2, A), np.int64); d[:m + 1] = k; d[dead, :] = dead
    for x in range(A):
        d[m, x] = k[m, x] if x in S else dead
    return canon(d, [True] * (m + 1) + [False], f"factor_next_in({tuple(w)},{sorted(S)})")


def existence(A, a):                  return build_dfa(A, "existence", a)
def _c(d, name):                      return canon(d.delta, d.acc, name)

def succession(A, a, b):              return product(build_dfa(A, "response", a, b), build_dfa(A, "precedence", a, b), "and", f"succession({a},{b})")
def chain_succession(A, a, b):        return product(build_dfa(A, "chain_response", a, b), build_dfa(A, "chain_precedence", a, b), "and", f"chain_succession({a},{b})")
def alternate_succession(A, a, b):    return product(alternate_response(A, a, b), alternate_precedence(A, a, b), "and", f"alternate_succession({a},{b})")
def co_existence(A, a, b):            return product(build_dfa(A, "responded_existence", a, b), build_dfa(A, "responded_existence", b, a), "and", f"co_existence({a},{b})")
def not_coexistence(A, a, b):         return negate(product(existence(A, a), existence(A, b), "and"), f"not_coexistence({a},{b})")
def choice(A, a, b):                  return product(existence(A, a), existence(A, b), "or", f"choice({a},{b})")
def exclusive_choice(A, a, b):        return product(choice(A, a, b), not_coexistence(A, a, b), "and", f"exclusive_choice({a},{b})")


# ---------------------------------------------------------------------------------------------
# Vectorised helpers (many traces at once)
# ---------------------------------------------------------------------------------------------
def pad_traces(seqs, A, L=None):
    L = L or max(len(s) for s in seqs)
    X = np.full((len(seqs), L), A, np.int64)
    for i, s in enumerate(seqs): X[i, :len(s)] = s
    return X


def accept_batch(d, X, A):
    delta = np.concatenate([d.delta, np.arange(d.delta.shape[0])[:, None]], 1)   # pad symbol = identity
    q = np.zeros(len(X), np.int64)
    for t in range(X.shape[1]): q = delta[q, X[:, t]]
    return d.acc[q]


def exclusion(d, X, miss, A):
    """X [N,L] true symbols (pad=A), miss [N,L] bool. Returns excluded [N,L,A] bool: activity a is ruled out
    at missing position t by this rule (others missing = wildcard; rule violated by the visible part -> no info)."""
    Q = d.delta.shape[0]
    T = np.zeros((A + 1, Q, Q), np.int64)
    for a in range(A):
        T[a, np.arange(Q), d.delta[:, a]] = 1
    T[A] = np.eye(Q, dtype=np.int64)
    Tany = (T[:A].sum(0) > 0).astype(np.int64)
    N, L = X.shape
    F = np.zeros((N, L + 1, Q), np.int64); F[:, 0, 0] = 1
    for t in range(L):
        Tt = np.where(miss[:, t, None, None], Tany[None], T[X[:, t]])
        F[:, t + 1] = (np.einsum("nq,nqk->nk", F[:, t], Tt) > 0)
    Bk = np.zeros((N, L + 1, Q), np.int64); Bk[:, L] = d.acc
    for t in range(L - 1, -1, -1):
        Tt = np.where(miss[:, t, None, None], Tany[None], T[X[:, t]])
        Bk[:, t] = (np.einsum("nqk,nk->nq", Tt, Bk[:, t + 1]) > 0)
    sat = (F[:, L] * d.acc[None]).sum(1) > 0
    feas = np.zeros((N, L, A), bool)
    for a in range(A):
        nxt = np.einsum("nlq,qk->nlk", F[:, :L], T[a]) > 0
        feas[:, :, a] = (nxt & (Bk[:, 1:] > 0)).any(2)
    return (~feas) & sat[:, None, None] & miss[:, :, None]


# ---------------------------------------------------------------------------------------------
# Candidate generation and mining
# ---------------------------------------------------------------------------------------------
FAMILIES = ("basic", "declare", "counts", "bounded", "ngram")


def candidates(full_seqs, A, families=FAMILIES, min_count=30, max_ngram=250, succ_mass=0.995):
    out = []
    pairs = [(a, b) for a in range(A) for b in range(A) if a != b]
    if "basic" in families:
        out += [(f"{t}({a})", build_dfa(A, t, a)) for t in ("existence", "init", "end") for a in range(A)]
        out += [(f"{t}({a},{b})", build_dfa(A, t, a, b)) for t in
                ("response", "precedence", "chain_response", "chain_precedence", "responded_existence", "not_succession")
                for a, b in pairs]
    if "declare" in families:
        for fn in (alternate_response, alternate_precedence, not_chain_succession, succession, chain_succession,
                   alternate_succession):
            out += [(fn.__name__ + f"({a},{b})", fn(A, a, b)) for a, b in pairs]
        for fn in (co_existence, not_coexistence, choice, exclusive_choice):
            out += [(fn.__name__ + f"({a},{b})", fn(A, a, b)) for a, b in pairs if a < b]
    if "counts" in families:
        for a in range(A):
            out += [(f"at_least({a},{k})", occurrences(A, a, k, None)) for k in (2, 3)]
            out += [(f"at_most({a},{k})", occurrences(A, a, 0, k)) for k in (1, 2, 3)]
    if "bounded" in families:
        out += [(f"bounded_response({a},{b},{k})", bounded_response(A, a, b, k)) for a, b in pairs for k in (2, 3, 5)]
    if "ngram" in families:
        cnt = Counter(tuple(s) for s in full_seqs)
        ng = {n: Counter() for n in (2, 3)}
        nxt = {n: {} for n in (2, 3)}
        for s, c in cnt.items():
            for n in (2, 3):
                for i in range(len(s) - n + 1): ng[n][s[i:i + n]] += c
                for i in range(len(s) - n):
                    nxt[n].setdefault(s[i:i + n], Counter())[s[i + n]] += c
        for n in (2, 3):
            seen_pref = sorted(nxt[n].items(), key=lambda kv: -sum(kv[1].values()))[:max_ngram]
            for w, cs in seen_pref:
                tot = sum(cs.values())
                if tot < min_count: continue
                S, acc = [], 0
                for x, c in cs.most_common():          # smallest successor set covering succ_mass of the cases
                    S.append(x); acc += c
                    if acc / tot >= succ_mass: break
                if len(S) < A:
                    out.append((f"factor_next_in({w},{sorted(S)})", factor_next_in(A, w, set(S))))
            frequent = [w for w, c in ng[2].items() if c >= min_count]
            if n == 2:
                out += [(f"forbidden_factor({(a, b)})", forbidden_factor(A, (a, b)))
                        for a, b in pairs if (a, b) not in ng[2]]
            else:
                fs = set(frequent)
                tri = [(a, b, c) for a in range(A) for b in range(A) for c in range(A)
                       if (a, b) in fs and (b, c) in fs and (a, b, c) not in ng[3]]
                out += [(f"forbidden_factor({w})", forbidden_factor(A, w)) for w in tri[:max_ngram]]
    seen, uniq = set(), []                              # drop language-equivalent duplicates
    for n_, d in out:
        s = signature(d)
        if s not in seen: seen.add(s); uniq.append((n_, d))
    return uniq


def mine_rich_rules(full_seqs, A, families=FAMILIES, min_support=0.98, top_k=60, n_sim=1500, p_mask=0.3,
                    harm_budget=0.03, harm_weight=3.0, min_gain=0.0005, max_len=60, seed=0, verbose=True):
    """Returns [(label, dfa, weight)]. Selection: greedy on (wrong activities eliminated - harm_weight * true
    activities eliminated) over simulated masking of training traces, until top_k or the gain is negligible.
    harm_budget caps the share of simulated missing nodes whose TRUE activity may be eliminated by the union."""
    rng = np.random.default_rng(seed)
    cnt = Counter(tuple(s) for s in full_seqs); variants = list(cnt)
    V = pad_traces(variants, A); w = np.array([cnt[v] for v in variants], float); w /= w.sum()
    cands = candidates(full_seqs, A, families)
    sup = [float((accept_batch(d, V, A) * w).sum()) for _, d in cands]
    keep = [i for i, s in enumerate(sup) if s >= min_support]
    if verbose: print(f"{len(cands)} candidate patterns, {len(keep)} with support >= {min_support}")
    pool = [s for s in full_seqs if 2 <= len(s) <= max_len]
    sel = rng.choice(len(pool), size=min(n_sim, len(pool)), replace=False)
    X = pad_traces([pool[i] for i in sel], A)
    miss = (rng.random(X.shape) < p_mask) & (X < A)
    for n in range(len(X)):                              # at least one missing node per trace
        if not miss[n].any(): miss[n, rng.integers(0, int((X[n] < A).sum()))] = True
    truth = np.eye(A + 1, dtype=bool)[X][:, :, :A]
    M = int(miss.sum()); wrong_cells = M * (A - 1)
    E = {}
    for i in keep:
        e = exclusion(cands[i][1], X, miss, A)
        if e.any(): E[i] = e[miss]                       # [M, A]
    T = truth[miss]                                      # [M, A]
    covered = np.zeros((M, A), bool); harmed = np.zeros((M, A), bool)
    chosen = []
    for _ in range(top_k):
        best, best_gain = None, min_gain * M
        for i, e in E.items():
            if i in chosen: continue
            new_w = (e & ~covered & ~T).sum(); new_h = (e & T & ~harmed).sum()
            if (harmed.sum() + new_h) > harm_budget * M: continue
            g = new_w - harm_weight * new_h
            if g > best_gain: best, best_gain = i, g
        if best is None: break
        chosen.append(best); covered |= E[best]; harmed |= E[best] & T
    if verbose:
        print(f"selected {len(chosen)} rules | eliminate {covered[~T].sum() / wrong_cells:.1%} of wrong (node, activity) "
              f"pairs, true activity eliminated at {harmed.any(1).sum() / M:.2%} of masked nodes "
              f"(simulation: {M} masked nodes)")
    return [(cands[i][0], cands[i][1], sup[i]) for i in chosen]
