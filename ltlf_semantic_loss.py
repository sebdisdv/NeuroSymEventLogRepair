"""
Semantic loss from DECLARE/LTLf rules for the HGNN (Xu et al. 2018, with a DFA run
in the spirit of Umili et al. 2023 / 2026).

Idea. For every trace the model gives a distribution over activities at each MISSING position;
known positions are one-hot. Running the DFA of a rule on this sequence of distributions
(state distribution s_{t+1}[q'] = sum_a p_t[a] * sum_q s_t[q] * [delta(q,a)=q'])
gives P(rule accepts the trace), exactly, if positions are independent.
Loss = -log P(rule accepts), averaged over (trace, rule) pairs.

Which pairs count:
  * only rules that the GROUND-TRUTH trace satisfies (mined rules are ~98% true, so the
    2% violators must not push the model towards wrong answers; this also guarantees
    P > 0, since the visible events cannot already violate the rule);
  * only traces with >= 1 missing Activity node (otherwise P is constant: no gradient).
Uses y only to decide which rules apply, as a training-time-only signal; nothing is used at test time.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core (framework-agnostic algebra: only einsum / where / matmul)
# ---------------------------------------------------------------------------
def _run_dfas(P, valid, Tflat, acc, s0, R, Q, einsum, where):
    """P [B,T,A] activity distributions, valid [B,T] bool, Tflat [A, R*Q*Q],
    acc [R,Q], s0 [B,R,Q]. Returns P(accept) [B,R]."""
    B, Tn, A = P.shape
    s = s0
    for t in range(Tn):
        M = (P[:, t] @ Tflat).reshape(B, R, Q, Q)         # expected transition matrix at step t
        nxt = einsum("brq,brqk->brk", s, M)
        s = where(valid[:, t, None, None], nxt, s)         # padded steps leave the state alone
    return (s * acc[None]).sum(-1)


class DFABank(nn.Module):
    """All rule DFAs stacked into dense tensors (padded to the largest state count)."""

    def __init__(self, dfas, A):
        super().__init__()
        R, Q = len(dfas), max(d.delta.shape[0] for d in dfas)
        T = np.zeros((A, R, Q, Q), np.float32)
        acc = np.zeros((R, Q), np.float32)
        for r, d in enumerate(dfas):
            n_q = d.delta.shape[0]
            for q in range(n_q):
                for a in range(A):
                    T[a, r, q, d.delta[q, a]] = 1.0
            for q in range(n_q, Q):                         # unused padding states: self-loop
                T[:, r, q, q] = 1.0
            acc[r, :n_q] = d.acc
        self.A, self.R, self.Q = A, R, Q
        self.register_buffer("Tflat", torch.from_numpy(T.reshape(A, R * Q * Q)))
        self.register_buffer("acc", torch.from_numpy(acc))

    def sat_prob(self, P, valid):
        s0 = P.new_zeros(P.size(0), self.R, self.Q)
        s0[:, :, 0] = 1.0
        return _run_dfas(P, valid, self.Tflat, self.acc, s0, self.R, self.Q,
                         torch.einsum, torch.where)


# ---------------------------------------------------------------------------
# Loss on a PyG hetero batch
# ---------------------------------------------------------------------------
class LTLfSemanticLoss(nn.Module):
    def __init__(self, dfas, class_ids, n_classes, act_key="Activity", rule_weights=None, eps=1e-6):
        """class_ids: model/graph class indices of the REAL activities (everything except MISSING_VALUE),
        in the compact order the DFAs use: DFA symbol k <-> class_ids[k].
        n_classes: width of the one-hot Activity encoding (= model output size), MISSING class included."""
        super().__init__()
        A = len(class_ids)
        self.bank, self.A, self.key, self.eps, self.n_classes = DFABank(dfas, A), A, act_key, eps, n_classes
        lut = torch.full((n_classes,), -1, dtype=torch.long)       # class index -> compact id (-1 = MISSING)
        lut[torch.as_tensor(class_ids)] = torch.arange(A)
        self.register_buffer("lut", lut)
        self.register_buffer("class_ids", torch.as_tensor(class_ids, dtype=torch.long))
        w = torch.ones(len(dfas)) if rule_weights is None else torch.as_tensor(rule_weights, dtype=torch.float32)
        self.register_buffer("w", w / w.mean())

    def _pad(self, batch, P_flat):
        node = batch[self.key]
        B = batch.num_graphs
        counts = torch.bincount(node.batch, minlength=B)
        ptr = torch.cumsum(counts, 0) - counts
        pos = torch.arange(node.batch.numel(), device=P_flat.device) - ptr[node.batch]
        Tmax = int(counts.max())
        Pp = P_flat.new_zeros(B, Tmax, self.A).index_put((node.batch, pos), P_flat)
        valid = torch.arange(Tmax, device=P_flat.device)[None] < counts[:, None]
        return Pp, valid

    def _distributions(self, batch, miss_idx, logits=None, hard=False):
        """[N, A] distributions: one-hot for known nodes, model output (or its argmax) for missing."""
        node, A = batch[self.key], self.A
        known = self.lut[node.x[:, : self.n_classes].argmax(1)].clamp(min=0)  # extra feature columns ignored;
        P = F.one_hot(known, A).float()                                       # masked rows are overwritten below
        if logits is not None:
            p = logits.float().index_select(1, self.class_ids).softmax(-1)   # real activities only, compact order
            if hard:
                p = F.one_hot(p.argmax(-1), A).float()
            P = P.index_copy(0, miss_idx, p)
        return P

    def _ground_truth_ok(self, batch):
        y = self.lut[batch.y[self.key]].clamp(min=0)                # labels -> compact ids
        Pp, valid = self._pad(batch, F.one_hot(y, self.A).float())
        return self.bank.sat_prob(Pp, valid) > 0.5               # [B, R]

    def _active(self, batch, miss_idx, gt_ok):
        has_miss = torch.bincount(batch[self.key].batch[miss_idx], minlength=batch.num_graphs) > 0
        return gt_ok & has_miss[:, None]

    def forward(self, batch, idx, logits):
        """logits: model output for the missing Activity nodes [n_missing, n_classes]; idx = dict used in forward."""
        with torch.autocast(device_type=logits.device.type, enabled=False):
            miss_idx = idx[self.key]
            with torch.no_grad():
                active = self._active(batch, miss_idx, self._ground_truth_ok(batch))
            if not active.any():
                return logits.sum() * 0.0
            P = self._distributions(batch, miss_idx, logits)
            Pp, valid = self._pad(batch, P)
            p_sat = self.bank.sat_prob(Pp, valid).clamp_min(self.eps)
            nll = -(p_sat.log() * self.w[None]) * active
            return nll.sum() / active.sum()

    @torch.no_grad()
    def hard_satisfaction(self, batch, idx, logits):
        """(#satisfied, #applicable) (trace, rule) pairs when the argmax predictions fill the gaps."""
        miss_idx = idx[self.key]
        active = self._active(batch, miss_idx, self._ground_truth_ok(batch))
        P = self._distributions(batch, miss_idx, logits, hard=True)
        Pp, valid = self._pad(batch, P)
        sat = self.bank.sat_prob(Pp, valid) > 0.5
        return int((sat & active).sum()), int(active.sum())