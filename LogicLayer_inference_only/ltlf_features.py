"""
LTLf/DECLARE -> DFA -> per-node "feasibility" features for the HGNN.

For every rule (a DFA) and every MISSING position t of a trace we compute which
activities `a` can be put at t such that the rule can still be satisfied, given
the known events before and after t. Other missing positions are treated
nondeterministically (any activity), so the mask is a sound over-approximation:
it never excludes an activity that some completion of the trace would accept.

Feature layout per rule (A = number of real activities):
    [feasible_a0, ..., feasible_a(A-1), rule_unsatisfiable_flag]
If the known events already make the rule unsatisfiable (mined rules are rarely
100% true) the mask is set to all ones (no information) and the flag to 1.
Known positions get zeros. Final shape: [n_events, R * (A + 1)].

Nothing here reads labels (y), so there is no leakage at train time.
"""
from collections import Counter

import numpy as np
import torch


# ----------------------------------------------------------------------------
# DFA construction (DECLARE templates, LTLf finite-trace semantics)
# delta[q, a] -> q'; state 0 is the initial state.
# ----------------------------------------------------------------------------
class DFA:
    def __init__(self, delta, acc, name=""):
        self.delta = np.asarray(delta, dtype=np.int64)  # [Q, A]
        self.acc = np.asarray(acc, dtype=bool)          # [Q]
        self.name = name

    def accepts(self, seq):
        q = 0
        for a in seq:
            q = self.delta[q, a]
        return bool(self.acc[q])


def _table(Q, A, f):
    d = np.zeros((Q, A), dtype=np.int64)
    for q in range(Q):
        for a in range(A):
            d[q, a] = f(q, a)
    return d


def existence(A, a):
    return DFA(_table(2, A, lambda q, x: 1 if (q == 1 or x == a) else 0), [0, 1], f"existence({a})")


def init(A, a):  # trace starts with a
    return DFA(_table(3, A, lambda q, x: 2 if (q == 2 or (q == 0 and x != a)) else 1), [0, 1, 0],
               f"init({a})")


def end(A, a):  # trace ends with a
    return DFA(_table(2, A, lambda q, x: 1 if x == a else 0), [0, 1], f"end({a})")


def response(A, a, b):  # every a is eventually followed by b
    def f(q, x):
        if q == 0:
            return 1 if x == a else 0
        return 0 if x == b else 1
    return DFA(_table(2, A, f), [1, 0], f"response({a},{b})")


def precedence(A, a, b):  # b only if a occurred before
    def f(q, x):
        if q == 0:
            return 1 if x == a else (2 if x == b else 0)
        return q
    return DFA(_table(3, A, f), [1, 1, 0], f"precedence({a},{b})")


def chain_response(A, a, b):  # every a is immediately followed by b
    def f(q, x):
        if q == 0:
            return 1 if x == a else 0
        if q == 1:
            return 0 if x == b else 2
        return 2
    return DFA(_table(3, A, f), [1, 0, 0], f"chain_response({a},{b})")


def chain_precedence(A, a, b):  # every b is immediately preceded by a
    def f(q, x):
        if q == 0:
            return 1 if x == a else (2 if x == b else 0)
        if q == 1:
            return 1 if x == a else 0
        return 2
    return DFA(_table(3, A, f), [1, 1, 0], f"chain_precedence({a},{b})")


def responded_existence(A, a, b):  # if a occurs, b occurs somewhere
    def f(q, x):
        if q == 0:
            return 1 if x == a else (2 if x == b else 0)
        if q == 1:
            return 2 if x == b else 1
        return 2
    return DFA(_table(3, A, f), [1, 0, 1], f"responded_existence({a},{b})")


def not_succession(A, a, b):  # b never occurs after a
    def f(q, x):
        if q == 0:
            return 1 if x == a else 0
        if q == 1:
            return 2 if x == b else 1
        return 2
    return DFA(_table(3, A, f), [1, 1, 0], f"not_succession({a},{b})")


UNARY = {"existence": existence, "init": init, "end": end}
BINARY = {
    "response": response, "precedence": precedence,
    "chain_response": chain_response, "chain_precedence": chain_precedence,
    "responded_existence": responded_existence, "not_succession": not_succession,
}


def build_dfa(A, template, *args):
    fn = UNARY.get(template) or BINARY[template]
    return fn(A, *args)


# ----------------------------------------------------------------------------
# Rule mining on COMPLETE training traces (use y['Activity'] of train graphs)
# ----------------------------------------------------------------------------
def mine_rules(full_seqs, A, min_support=0.98, min_activation=0.05, top_k=60):
    """full_seqs: list of lists of activity ids. Returns list of (template, args, support).
    Support is the fraction of traces accepted; binary rules must be activated (a present)
    in >= min_activation of the traces so vacuous rules are dropped."""
    counts = Counter(tuple(s) for s in full_seqs)  # few unique variants -> fast
    total = sum(counts.values())
    present = np.zeros(A)
    for s, c in counts.items():
        for a in set(s):
            present[a] += c
    present /= total

    cands = [(t, (a,)) for t in UNARY for a in range(A)]
    cands += [(t, (a, b)) for t in BINARY for a in range(A) for b in range(A) if a != b]

    out = []
    for t, args in cands:
        if t in BINARY and present[args[0]] < min_activation:
            continue
        dfa = build_dfa(A, t, *args)
        sup = sum(c for s, c in counts.items() if dfa.accepts(s)) / total
        if sup >= min_support and sup < 1.0 + 1e-9:
            out.append((t, args, sup * (present[args[0]] if t in BINARY else 1.0)))
    out.sort(key=lambda r: -r[2])
    return out[:top_k]


# ----------------------------------------------------------------------------
# Feasibility features
# ----------------------------------------------------------------------------
def _dfa_feasibility(dfa, acts, missing_id):
    delta, acc = dfa.delta, dfa.acc
    Q, A = delta.shape
    n = len(acts)
    miss = [a == missing_id for a in acts]

    F = np.zeros((n + 1, Q), dtype=bool)   # states reachable after t events
    F[0, 0] = True
    for t in range(n):
        if miss[t]:
            nxt = delta[F[t]].ravel()
        else:
            nxt = delta[F[t], acts[t]]
        F[t + 1, nxt] = True

    B = np.zeros((n + 1, Q), dtype=bool)   # states from which acceptance is reachable
    B[n] = acc
    for t in range(n - 1, -1, -1):
        B[t] = B[t + 1][delta].any(1) if miss[t] else B[t + 1][delta[:, acts[t]]]

    unsat = not (F[n] & acc).any()
    feat = np.zeros((n, A + 1), dtype=np.float32)
    for t in range(n):
        if not miss[t]:
            continue
        if unsat:
            feat[t, :A] = 1.0
            feat[t, A] = 1.0
        else:
            feat[t, :A] = (F[t][:, None] & B[t + 1][delta]).any(0)
    return feat


def rule_features(dfas, acts, missing_id):
    return np.concatenate([_dfa_feasibility(d, acts, missing_id) for d in dfas], axis=1)


# ----------------------------------------------------------------------------
# Graph integration
# ----------------------------------------------------------------------------
def _activity_ids(data, node_type):
    x = data[node_type].x
    return x.argmax(1).tolist() if x.dim() == 2 else x.long().tolist()


def add_rule_features(graphs, dfas, missing_id, node_type="Activity", act_ids_fn=None):
    """In-place: x[node_type] <- cat([x, rule features]). Run once per graph list.
    act_ids_fn(data) -> list of activity ids in temporal order (MISSING = missing_id).
    Default: argmax of the one-hot x, nodes assumed ordered by time."""
    act_ids_fn = act_ids_fn or (lambda d: _activity_ids(d, node_type))
    cache = {}
    for g in graphs:
        acts = tuple(act_ids_fn(g))
        if acts not in cache:
            cache[acts] = torch.from_numpy(rule_features(dfas, list(acts), missing_id))
        x = g[node_type].x
        g[node_type].x = torch.cat([x.float(), cache[acts]], dim=1)
    return graphs
