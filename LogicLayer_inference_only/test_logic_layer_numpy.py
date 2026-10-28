import sys, types, itertools, random, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '/home/claude')
t = types.ModuleType('torch'); t.no_grad = lambda: (lambda f: f); nn = types.ModuleType('torch.nn'); nn.Module = object
fn = types.ModuleType('torch.nn.functional'); nn.functional = fn; t.nn = nn
sys.modules.update({'torch': t, 'torch.nn': nn, 'torch.nn.functional': fn})
import ltlf_features as L, ltlf_logic_layer as G, ltlf_decode as D
A = 4; random.seed(3); np.random.seed(3)
dfas = [L.build_dfa(A, tp, *a) for tp in L.UNARY for a in [(0,), (2,)]] + \
       [L.build_dfa(A, tp, 0, 1) for tp in L.BINARY] + [L.build_dfa(A, tp, 3, 2) for tp in L.BINARY]
R, Q = len(dfas), max(d.delta.shape[0] for d in dfas)
T4 = np.zeros((A, R, Q, Q), np.float32); acc = np.zeros((R, Q), np.float32)
for r, d in enumerate(dfas):
    nq = d.delta.shape[0]
    for q in range(nq):
        for a in range(A): T4[a, r, q, d.delta[q, a]] = 1
    for q in range(nq, Q): T4[:, r, q, q] = 1
    acc[r, :nq] = d.acc
Tflat = T4.reshape(A, -1)
B, Tm = 8, 6
lens = [random.randint(2, Tm) for _ in range(B)]
P = np.zeros((B, Tm, A), np.float32); valid = np.zeros((B, Tm), bool); miss = np.zeros((B, Tm), bool)
for b, n in enumerate(lens):
    for t in range(n):
        valid[b, t] = True
        if random.random() < .5: P[b, t] = np.random.dirichlet(np.ones(A)); miss[b, t] = True
        else: P[b, t] = np.eye(A)[random.randrange(A)]
s0 = np.zeros((B, R, Q), np.float32); s0[:, :, 0] = 1
logq, active = G.condition(P, valid, miss, Tflat, T4, acc, s0, R, Q)
# brute force: messages m_r,t(a) = P(r accepts | a_t=a), others independent; also active via enumeration
err1 = err_m = 0; nact_bad = 0; n_single = 0
for b, n in enumerate(lens):
    mpos = [t for t in range(n) if miss[b, t]]
    # enumerate assignments of missing positions (known one-hot fixed)
    known = [int(P[b, t].argmax()) for t in range(n)]
    msg = np.ones((n, A, R)); sat_any = np.zeros(R, bool)
    joint = {}
    for assign in itertools.product(range(A), repeat=len(mpos)):
        seq = list(known)
        for t, a in zip(mpos, assign): seq[t] = a
        pr = np.prod([P[b, t, a] for t, a in zip(mpos, assign)])
        sat = np.array([d.accepts(seq) for d in dfas])
        sat_any |= sat
        joint[assign] = (pr, sat)
    for r in range(R): nact_bad += int(sat_any[r] != active[b, r])
    # per-rule message by enumeration
    M = np.zeros((n, A, R))
    for j, t in enumerate(mpos):
        for a in range(A):
            num = np.zeros(R)
            for assign, (pr, sat) in joint.items():
                if assign[j] != a: continue
                w = pr / P[b, t, a] if P[b, t, a] > 0 else 0
                num += w * sat
            M[t, a] = num
    for t in mpos:
        z = np.log(np.maximum(P[b, t], 1e-12)) + (np.log(np.maximum(M[t], 1e-8)) * active[b][None]).sum(-1)
        z -= np.logaddexp.reduce(z)
        err_m = max(err_m, np.abs(z - logq[b, t]).max())
    # exact posterior check for ONE active rule
    for r in range(R):
        if not active[b, r] or not mpos: continue
        lq1, _ = G.condition(P, valid, miss, Tflat, T4, acc, s0, R, Q, active=np.arange(R)[None, :].repeat(B, 0) == r)
        for j, t in enumerate(mpos):
            post = np.zeros(A)
            for assign, (pr, sat) in joint.items():
                if sat[r]: post[assign[j]] += pr
            post /= post.sum()
            err1 = max(err1, np.abs(np.exp(lq1[b, t]) - post).max()); n_single += 1
print("active-rule mismatches vs enumeration:", nact_bad)
print("max |layer - brute-force messages (all active rules)| =", err_m)
print(f"single-rule exact posterior: max err {err1:.2e} over {n_single} checks")
# active vs decoder backward viability
mx = 0
for b, n in enumerate(lens):
    acts = [int(P[b, t].argmax()) if not miss[b, t] else A for t in range(n)]
    dec = D.HardDecoder(dfas, A); Bk = dec._backward(acts)
    mx += int((Bk[0][:, 0] != active[b]).sum())
print("active vs HardDecoder viability mismatches:", mx)
lq3, _ = G.condition(P, valid, miss, Tflat, T4, acc, s0, R, Q, n_iter=3)
print("n_iter=3 finite & normalised:", np.isfinite(lq3[miss]).all(), np.abs(np.exp(lq3[miss]).sum(-1) - 1).max())
