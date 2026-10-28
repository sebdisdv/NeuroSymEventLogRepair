"""
Hard-constrained decoding with DECLARE/LTLf rules (inference time, numpy only except evaluate_hard).

For one trace: choose activities for ALL missing positions jointly, maximising
    sum_j log p_j(a_j)
subject to every enforced rule's DFA accepting the completed trace.

Beam search over time with one DFA state per rule. A hypothesis is kept only if, for every rule, its
state can still reach acceptance given the known suffix (missing positions in the suffix = wildcards),
so the search never walks into a dead end for any single rule. With beam >= #candidates it is exact.

Rules the visible events already make unsatisfiable (mined rules are ~98% true) are dropped for that
trace, so a violated rule can never force a wrong answer. If no hypothesis survives (rules jointly
unsatisfiable) decode() returns None and the caller falls back to the plain argmax.
"""
import numpy as np


class HardDecoder:
    def __init__(self, dfas, A, rule_idx=None):
        dfas = [dfas[i] for i in rule_idx] if rule_idx is not None else list(dfas)
        self.dfas, self.A, self.R = dfas, A, len(dfas)
        Q = max(d.delta.shape[0] for d in dfas)
        self.Q = Q
        self.delta = np.zeros((self.R, Q, A), np.int64)
        self.acc = np.zeros((self.R, Q), bool)
        for r, d in enumerate(dfas):
            q = d.delta.shape[0]
            self.delta[r, :q], self.acc[r, :q] = d.delta, d.acc
            for p in range(q, Q):
                self.delta[r, p] = p                       # padding states: self-loop, non-accepting

    def _backward(self, acts):
        """B[t][r, q]: from state q after t events, acceptance is reachable (missing = any activity)."""
        n, A, R, Rr = len(acts), self.A, self.R, np.arange(self.R)
        B = np.zeros((n + 1, R, self.Q), bool)
        B[n] = self.acc
        for t in range(n - 1, -1, -1):
            if acts[t] == A:
                B[t] = B[t + 1][Rr[:, None, None], self.delta].any(2)
            else:
                B[t] = B[t + 1][Rr[:, None], self.delta[:, :, acts[t]]]
        return B

    def decode(self, logp, acts, beam=16):
        """logp [m, A] log-probs of the m missing positions (in order); acts: compact ids, A = missing.
        Returns the m chosen compact activities, or None if no assignment satisfies the rules."""
        A, R, Rr = self.A, self.R, np.arange(self.R)
        B = self._backward(acts)
        active = B[0][:, 0]                                # rule satisfiable given the visible events
        S, score, hist, j = np.zeros((1, R), np.int64), np.zeros(1), [], 0
        for t, a in enumerate(acts):
            ok_next = B[t + 1] | ~active[:, None]          # inactive rules never prune
            if a == A:
                newq = self.delta[Rr[None, None, :], S[:, None, :], np.arange(A)[None, :, None]]  # [K,A,R]
                ok = ok_next[Rr[None, None, :], newq].all(2)                                      # [K,A]
                sc = np.where(ok, score[:, None] + logp[j][None, :], -np.inf).ravel()
                k = min(beam, int(np.isfinite(sc).sum()))
                if k == 0:
                    return None
                top = np.argpartition(-sc, k - 1)[:k]
                par, ch = np.divmod(top, A)
                S, score = newq[par, ch], sc[top]
                hist.append((par, ch)); j += 1
            else:
                S = self.delta[Rr[None, :], S, a]
        i, out = int(score.argmax()), []
        for par, ch in reversed(hist):
            out.append(int(ch[i])); i = int(par[i])
        return out[::-1]

    def satisfied(self, seq):
        return np.array([d.accepts(seq) for d in self.dfas])


# ---------------------------------------------------------------------------
# Evaluation on PyG batches: plain argmax vs hard-constrained decoding
# ---------------------------------------------------------------------------
def evaluate_hard(net, graphs, decoder, class_ids, missing_id, n_classes, keys, device,
                  mask_to_idx, DataLoader, beam=16, batch_size=128, act_key="Activity"):
    import torch
    A = len(class_ids)
    lut = torch.full((n_classes,), -1, dtype=torch.long); lut[torch.as_tensor(class_ids)] = torch.arange(A)
    cid = torch.as_tensor(class_ids, device=device)
    st = dict(missing=0, ok_plain=0, ok_hard=0, traces=0, fallback=0, changed=0,
              pairs=0, sat_plain=0, sat_hard=0)
    net.eval()
    with torch.no_grad():
        for x in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            idx = mask_to_idx(x, keys)                                   # CPU
            xd = x.to(device); idxd = {k: v.to(device) for k, v in idx.items()}
            out = net(xd, idxd)[act_key]
            logp = out.float().index_select(1, cid).log_softmax(-1).cpu().numpy()
            # NB: PyG's .to(device) moves the batch IN PLACE, so x is on the GPU here: do the bookkeeping on CPU
            node = x[act_key]; B = x.num_graphs
            cls = node.x[:, :n_classes].argmax(1).cpu()
            nb = node.batch.cpu()
            acts_all = lut[cls].clone(); acts_all[cls == missing_id] = A
            y = lut[x.y[act_key].cpu()]
            cnt = torch.bincount(nb, minlength=B); ptr = torch.cumsum(cnt, 0) - cnt
            mg = nb[idx[act_key].cpu()]
            mc = torch.bincount(mg, minlength=B); mp = torch.cumsum(mc, 0) - mc
            for g in range(B):
                if mc[g] == 0:
                    continue
                sl = slice(int(ptr[g]), int(ptr[g] + cnt[g]))
                acts = acts_all[sl].tolist(); lp = logp[int(mp[g]): int(mp[g] + mc[g])]
                pos = [i for i, a in enumerate(acts) if a == A]
                plain = lp.argmax(1).tolist()
                hard = decoder.decode(lp, acts, beam)
                if hard is None:
                    st["fallback"] += 1; hard = plain
                truth = y[sl].tolist()
                st["traces"] += 1; st["missing"] += len(pos); st["changed"] += int(hard != plain)
                st["ok_plain"] += sum(plain[k] == truth[p] for k, p in enumerate(pos))
                st["ok_hard"] += sum(hard[k] == truth[p] for k, p in enumerate(pos))
                gt_ok = decoder.satisfied(truth)                           # only rules true on the real trace
                for name, pred in (("sat_plain", plain), ("sat_hard", hard)):
                    seq = list(acts)
                    for k, p in enumerate(pos): seq[p] = pred[k]
                    st[name] += int((decoder.satisfied(seq) & gt_ok).sum())
                st["pairs"] += int(gt_ok.sum())
    return {"acc_plain": st["ok_plain"] / max(st["missing"], 1), "acc_hard": st["ok_hard"] / max(st["missing"], 1),
            "sat_plain": st["sat_plain"] / max(st["pairs"], 1), "sat_hard": st["sat_hard"] / max(st["pairs"], 1),
            "traces_changed": st["changed"] / max(st["traces"], 1), "fallback": st["fallback"] / max(st["traces"], 1),
            "n_missing": st["missing"]}
