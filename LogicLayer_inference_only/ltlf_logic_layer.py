"""
Differentiable LTLf conditioning layer ("neural logic layer") for the HGNN.

Input : per-position activity distributions p_t (one-hot for known events, network softmax for missing ones).
Output: for every MISSING position, the distribution after conditioning on the active rules,

        q_t(a)  ∝  p_t(a) * prod_{r active} m_{r,t}(a),
        m_{r,t}(a) = P( rule r accepts the trace | a_t = a ),   other missing positions ~ their marginals.

m_{r,t}(a) comes from one forward and one backward pass of the rule's DFA over the distributions:
        m_{r,t}(a) = sum_{q,q'} alpha_t[q] * [delta(q,a)=q'] * beta_{t+1}[q']
(alpha_t = state distribution before step t, beta_{t+1}[q'] = P(accept | state q' after t+1 events)).

Exactness
  * ONE active rule: q_t is exactly the posterior p(a_t | rule accepted) when positions are independent
    (checked against brute-force enumeration in the tests).
  * SEVERAL rules: the product of the per-rule messages is a naive-Bayes style combination (exact conditioning
    on the conjunction would need the product automaton, exponential in the number of rules). With
    n_iter > 1 the messages are recomputed from the conditioned marginals (mean-field refinement).

"Active" rules = rules that can still be satisfied given the VISIBLE events (some completion of the gaps
makes the DFA accept). Needs no labels, so it is identical at training and at test time. Inactive rules
(already violated by the visible part; mined rules are ~98% true) are ignored for that trace.

Everything is differentiable w.r.t. the network output; rules are fixed. Core is framework-agnostic
(einsum / where / stack / log), so it is tested against brute force in numpy and runs unchanged on torch.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ltlf_semantic_loss import LTLfSemanticLoss


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
class _Ops:
    def __init__(self, einsum, where, stack, log, exp, log_softmax, clamp_min):
        self.einsum, self.where, self.stack, self.log, self.exp = einsum, where, stack, log, exp
        self.log_softmax, self.clamp_min = log_softmax, clamp_min


def _np_log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


NP_OPS = _Ops(np.einsum, np.where, np.stack, np.log, np.exp, _np_log_softmax, np.maximum)


def _backward(Pd, valid, Tflat, acc, R, Q, ops):
    B, Tn, _ = Pd.shape
    beta, out = acc[None] + 0 * Pd[:, 0, :1, None], [None] * (Tn + 1)    # [B,R,Q]
    out[Tn] = beta
    for t in range(Tn - 1, -1, -1):
        M = (Pd[:, t] @ Tflat).reshape(B, R, Q, Q)
        nxt = ops.einsum("brqk,brk->brq", M, beta)
        beta = ops.where(valid[:, t, None, None], nxt, beta)
        out[t] = beta
    return out                                                              # out[t] = beta_t


def active_rules(P, valid, miss, Tflat, acc, R, Q, ops):
    """[B,R] bool: some completion of the missing positions makes the rule accept."""
    W = ops.where(miss[..., None], 0 * P + 1.0, P)                          # wildcard = weight 1 on every activity
    B, Tn, _ = W.shape
    beta = acc[None] + 0 * P[:, 0, :1, None]
    for t in range(Tn - 1, -1, -1):
        M = (W[:, t] @ Tflat).reshape(B, R, Q, Q)
        nxt = ops.einsum("brqk,brk->brq", M, beta)
        nxt = ops.where(nxt > 1.0, 0 * nxt + 1.0, nxt)                      # keep it a 0/1 reachability indicator
        beta = ops.where(valid[:, t, None, None], nxt, beta)
    return beta[:, :, 0] > 0.5


def condition(P, valid, miss, Tflat, T4, acc, s0, R, Q, ops=NP_OPS, n_iter=1, eps=1e-8, active=None):
    """P [B,T,A] marginals, valid/miss [B,T] bool, Tflat [A,R*Q*Q], T4 [A,R,Q,Q], s0 [B,R,Q] initial state.
    Returns (log q [B,T,A] (meaningful at missing positions), active [B,R])."""
    B, Tn, A = P.shape
    if active is None:
        active = active_rules(P, valid, miss, Tflat, acc, R, Q, ops)
    act = active * 1.0
    logP0 = ops.log(ops.clamp_min(P, 1e-12))
    Pc = P
    for _ in range(n_iter):
        beta = _backward(Pc, valid, Tflat, acc, R, Q, ops)
        alpha, s = [], s0
        for t in range(Tn):
            alpha.append(s)
            M = (Pc[:, t] @ Tflat).reshape(B, R, Q, Q)
            s = ops.where(valid[:, t, None, None], ops.einsum("brq,brqk->brk", s, M), s)
        al, be = ops.stack(alpha, 1), ops.stack(beta[1:], 1)               # alpha_t, beta_{t+1}: [B,T,R,Q]
        m = ops.einsum("btrq,arqk,btrk->btar", al, T4, be)                  # [B,T,A,R]
        logm = ops.log(ops.clamp_min(m, eps)) * act[:, None, None, :]
        logq = ops.log_softmax(logP0 + logm.sum(-1))
        Pc = ops.where(miss[..., None], ops.exp(logq), P)
    return logq, active


# ---------------------------------------------------------------------------
# Layer on a PyG hetero batch
# ---------------------------------------------------------------------------
class LTLfLogicLayer(LTLfSemanticLoss):
    """Inherits the batch plumbing of LTLfSemanticLoss (class->compact-id map, padding, one-hot of known events)."""

    def __init__(self, dfas, class_ids, n_classes, act_key="Activity", n_iter=1, eps=1e-8):
        super().__init__(dfas, class_ids, n_classes, act_key=act_key)
        self.n_iter, self.eps_m = n_iter, eps
        R, Q, A = self.bank.R, self.bank.Q, self.A
        self.register_buffer("T4", self.bank.Tflat.reshape(A, R, Q, Q))
        s0 = torch.zeros(1, 1, Q); s0[..., 0] = 1.0
        self.register_buffer("s0", s0)
        self._ops = _Ops(torch.einsum, torch.where, torch.stack, torch.log, torch.exp,
                         lambda z: z.log_softmax(-1), lambda x, v: x.clamp_min(v))

    def _positions(self, batch):
        node = batch[self.key]
        counts = torch.bincount(node.batch, minlength=batch.num_graphs)
        ptr = torch.cumsum(counts, 0) - counts
        return torch.arange(node.batch.numel(), device=node.batch.device) - ptr[node.batch]

    def condition(self, batch, idx, logits):
        """logits [n_missing, n_classes] (raw network output at the missing Activity nodes).
        Returns (log q [n_missing, A] over the compact real activities, active [B, R])."""
        with torch.autocast(device_type=logits.device.type, enabled=False):
            miss_idx, node = idx[self.key], batch[self.key]
            P = self._distributions(batch, miss_idx, logits)
            Pp, valid = self._pad(batch, P)
            pos = self._positions(batch)
            missflat = torch.zeros(node.batch.numel(), dtype=torch.bool, device=P.device)
            missflat[miss_idx] = True
            miss = torch.zeros_like(valid).index_put((node.batch, pos), missflat)
            B, Tn, A = Pp.shape
            R, Q = self.bank.R, self.bank.Q
            bank = self.bank
            active = active_rules(Pp, valid, miss, bank.Tflat, bank.acc, bank.R, bank.Q, self._ops).detach()
            logq, _ = condition(Pp, valid, miss, bank.Tflat, self.T4, bank.acc, self.s0.expand(B, bank.R, bank.Q),
                                bank.R, bank.Q, self._ops, self.n_iter, self.eps_m, active)
            return logq[node.batch[miss_idx], pos[miss_idx]], active

    # -- helpers ----------------------------------------------------------------
    def nll(self, logq, batch, idx):
        """CE of the conditioned distribution against the true activities of the missing nodes."""
        y = self.lut[batch.y[self.key][idx[self.key]]].clamp(min=0)
        return F.nll_loss(logq, y)

    def conditioned_logits(self, logq):
        """[n_missing, n_classes] logits with the conditioned log-probs in the real-activity columns
        (MISSING column pushed to -1e4): drop-in replacement for the network output, e.g. in evaluate_hgnn."""
        out = logq.new_full((logq.size(0), self.n_classes), -1e4)
        return out.index_copy(1, self.class_ids, logq)

    def forward(self, batch, idx, logits):                       # training term: CE of the conditioned output
        logq, _ = self.condition(batch, idx, logits)
        return self.nll(logq, batch, idx)


# ---------------------------------------------------------------------------
# Evaluation: raw vs conditioned (argmax) vs hard decoding on raw / conditioned scores
# ---------------------------------------------------------------------------
def evaluate_layer(net, graphs, layer, decoder, class_ids, missing_id, n_classes, keys, device,
                   mask_to_idx, DataLoader, beam=16, batch_size=128, act_key="Activity"):
    """Accuracy on missing Activity nodes for
         raw          argmax of the network output
         cond         argmax of the conditioned output (logic layer only)
         hard         hard-constrained beam decoding on the raw log-probs
         cond_hard    hard-constrained beam decoding on the conditioned log-probs
       plus rule satisfaction (only rules true on the ground truth) for each, and the share of
       rules active per trace."""
    A = len(class_ids)
    lut = torch.full((n_classes,), -1, dtype=torch.long); lut[torch.as_tensor(class_ids)] = torch.arange(A)
    cid = torch.as_tensor(class_ids, device=device)
    names = ("raw", "cond", "hard", "cond_hard")
    ok = dict.fromkeys(names, 0); sat = dict.fromkeys(names, 0)
    n_miss = pairs = traces = act_sum = fb_hard = fb_cond = 0
    net.eval()
    with torch.no_grad():
        for x in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            idx = mask_to_idx(x, keys)
            xd = x.to(device); idxd = {k: v.to(device) for k, v in idx.items()}
            out = net(xd, idxd)[act_key]
            lp_raw = out.float().index_select(1, cid).log_softmax(-1).cpu().numpy()
            logq, active = layer.condition(xd, idxd, out)
            lp_cond = logq.cpu().numpy(); active = active.cpu().numpy()
            node = x[act_key]; B = x.num_graphs                          # x is on `device` now: bookkeeping on CPU
            cls = node.x[:, :n_classes].argmax(1).cpu(); nb = node.batch.cpu()
            acts_all = lut[cls].clone(); acts_all[cls == missing_id] = A
            y = lut[x.y[act_key].cpu()]
            cnt = torch.bincount(nb, minlength=B); ptr = torch.cumsum(cnt, 0) - cnt
            mg = nb[idx[act_key].cpu()]
            mc = torch.bincount(mg, minlength=B); mp = torch.cumsum(mc, 0) - mc
            for g in range(B):
                if mc[g] == 0:
                    continue
                sl = slice(int(ptr[g]), int(ptr[g] + cnt[g]))
                acts = acts_all[sl].tolist(); a0, a1 = int(mp[g]), int(mp[g] + mc[g])
                pos = [i for i, a in enumerate(acts) if a == A]
                truth = y[sl].tolist()
                pred = {"raw": lp_raw[a0:a1].argmax(1).tolist(), "cond": lp_cond[a0:a1].argmax(1).tolist()}
                h = decoder.decode(lp_raw[a0:a1], acts, beam); fb_hard += h is None
                pred["hard"] = pred["raw"] if h is None else h
                hc = decoder.decode(lp_cond[a0:a1], acts, beam); fb_cond += hc is None
                pred["cond_hard"] = pred["cond"] if hc is None else hc
                gt_ok = decoder.satisfied(truth)
                traces += 1; n_miss += len(pos); pairs += int(gt_ok.sum()); act_sum += active[g].mean()
                for nme in names:
                    ok[nme] += sum(pred[nme][k] == truth[p] for k, p in enumerate(pos))
                    seq = list(acts)
                    for k, p in enumerate(pos):
                        seq[p] = pred[nme][k]
                    sat[nme] += int((decoder.satisfied(seq) & gt_ok).sum())
    res = {f"acc_{n}": ok[n] / max(n_miss, 1) for n in names}
    res.update({f"sat_{n}": sat[n] / max(pairs, 1) for n in names})
    res.update(active_rules=act_sum / max(traces, 1), fallback_hard=fb_hard / max(traces, 1),
               fallback_cond_hard=fb_cond / max(traces, 1), n_missing=n_miss)
    return res
