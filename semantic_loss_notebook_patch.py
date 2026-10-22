# ===== CELL 1 — after the mining / DFA cell (Cell A of the previous step), before training =====
import ltlf_features as LF
import ltlf_semantic_loss as SL

ACT_KEY, MISSING_ID = "Activity", len(list_unique["Activity"]) - 1
A_REAL = MISSING_ID
if "rules" not in globals():                         # reuse the mined rules if Cell A already ran
    rules = LF.mine_rules([g.y[ACT_KEY].tolist() for g in X_train], A_REAL,
                          min_support=0.98, top_k=60)
dfas = [LF.build_dfa(A_REAL, t, *args) for t, args, _ in rules]

# rule_weights: mined support*frequency (normalised to mean 1 inside the module). Use None for uniform.
sem_loss = SL.LTLfSemanticLoss(dfas, A_REAL, act_key=ACT_KEY,
                               rule_weights=[s for _, _, s in rules]).to(device)


# ===== CELL 2 — edits inside train_hgnn (cell 20): 4 small changes =====
# (a) task_losses also returns the model output:
#
#     def task_losses(x, idx, labels, return_out=False):
#         with torch.autocast(**amp_kwargs):
#             out = net(x, idx)
#         l = torch.stack([criteria[k](out[k].float(), labels[k]) for k in task_keys])
#         return (l, out) if return_out else l
#
# (b) before the epoch loop:
#
#     lam = config.get("lambda_sem", 0.0)          # 0.0 = baseline, no semantic loss
#
# (c) in the training step, replace the first line and the backward line:
#
#     losses_values, out = task_losses(x, idx, labels, return_out=True)
#     ...
#     l_sem = sem_loss(x, idx, out["Activity"]) if lam > 0 else 0.0
#     (weights.detach() @ losses_values + lam * l_sem).backward()
#
# (d) validation stays `task_losses(x, idx, labels).sum()` -> early stopping / Optuna still compare
#     the plain cross-entropy, so runs with different lambda_sem stay comparable.
#
# The semantic term is kept OUTSIDE GradNorm (GradNorm balances the CE tasks only); lambda_sem is
# its single weight. Optional Optuna entry in sample_config():
#     "lambda_sem": trial.suggest_float("lambda_sem", 1e-2, 3.0, log=True),


# ===== CELL 3 — satisfaction-rate metric (use next to evaluate_hgnn) =====
@torch.no_grad()
def satisfaction_rate(net, graphs, outputcat, outputreal, batch_size=128):
    """Share of (trace, rule) pairs satisfied when argmax predictions fill the gaps.
    Only pairs whose rule holds on the ground-truth trace are counted, so it is comparable across models."""
    keys = list(outputcat) + list(outputreal)
    net.eval(); sat = tot = 0
    for x in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        idx = mask_to_idx(x, keys)
        x = x.to(device); idx = {k: v.to(device) for k, v in idx.items()}
        out = net(x, idx)
        s, t = sem_loss.hard_satisfaction(x, idx, out[ACT_KEY])
        sat += s; tot += t
    return sat / max(tot, 1)

# Ablation (same seeds): lambda_sem in {0, 0.1, 0.3, 1.0}; report missing-node accuracy AND
#   satisfaction_rate(net, X_test, outputcat, outputreal)
# Combine with the feature variant by running Cell A as well (USE_LTLF=True): the two are independent.
