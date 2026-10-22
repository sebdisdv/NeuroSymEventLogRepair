# Neurosymbolic activity restoration: semantic loss and hard constraints at inference

This document explains, step by step, two ways of using mined DECLARE/LTLf rules to help a heterogeneous
GNN restore missing `Activity` values in event-log traces:

* **Semantic loss (training time).** A differentiable run of each rule's automaton on the model's predicted
  distributions; the loss penalises low probability that the restored trace satisfies the rules.
* **Hard constraints (inference time).** Joint decoding of all missing activities of a trace so that the
  restored trace is accepted by every enforced rule.

The two are independent: the semantic loss changes how the network is trained, hard constraints only change how
its outputs are turned into predictions. Hard constraints work on any trained network, with or without the loss.

---

## 0. TL;DR

| | Semantic loss | Hard constraints |
|---|---|---|
| When | Training | Inference only |
| What it does | Adds `lambda * L_sem` to the loss | Beam search over missing positions under all rule DFAs |
| Guarantee | None (soft) | Restored trace satisfies every enforced rule that the visible events still allow, unless no assignment exists (then fallback to argmax) |
| Needs labels | Only to choose which rules apply to a training trace | No |
| Code | `ltlf_semantic_loss.py` | `ltlf_decode.py` |
| Observed effect (simple patterns, 1,985 masked nodes) | Little | Accuracy 64.7% -> 72.2% |

Both need the same ingredients: the class encoding of the graphs (section 2), the rules compiled to DFAs
(section 3) and a rule miner (section 4).

---

## 1. Problem setting

An event log is a set of traces; a trace is a time-ordered sequence of events; an event has attributes
(`Activity`, `org:resource`, timestamp, ...). Some `Activity` values are missing. The base model is a
heterogeneous GNN: every attribute value of every event is a node, known values pass information to masked nodes
through convolutions, and a classifier head predicts the missing activity.

Notation used below:

* `n` – number of events of a trace, `A` – number of real activities (7 in `bpi_2013_CZ`).
* `a_t` – activity at position `t`; known for visible events, unknown for missing ones.
* `p_t` – the model's probability distribution over the `A` activities at a missing position `t`.
* A **rule** `r` is an LTLf formula; its DFA is `(Q_r, delta_r, q0, F_r)` with `delta_r(q, a) = q'`.

---

## 2. Data encoding (what the code actually assumes)

The graphs store `Activity` as a one-hot matrix `x` of width `N_CLASSES` (8 here). The class order is the
graph's own encoding, **not** the order of `list_unique`, and the `MISSING_VALUE` class sits in the middle of the
range (index 5 in the example graph):

```
x.argmax(1)   = [6, 5, 0, 5, 4]      # masked nodes (positions 1, 3) carry the MISSING class 5
masks         = [F, T, F, T, F]
y             = [6, 1, 0, 1, 4]      # labels: the true activity, never the MISSING class
```

Consequences handled by the setup cell:

1. `MISSING_ID` is **read from the data** (`infer_missing_id`: the single class used by all masked nodes), not assumed.
2. The DFAs work on a **compact alphabet** `0..A-1` that excludes the MISSING class. `CLASS_IDS` lists the real
   class indices in graph order; `lut` maps a graph class to its compact id. Without this, mined rules would
   contain a bogus symbol (e.g. "class 5 never occurs") and the loss would drop the wrong logit.
3. In compact ids a missing node is encoded as `A` (an out-of-range value that the feasibility code only compares).
4. Sanity assertions on 200 training graphs: masked nodes use the MISSING class and visible ones do not; `x.argmax`
   equals `y` on visible nodes; labels never equal MISSING; `followed_by` edges connect consecutive node indices
   (so nodes are in temporal order, which the DFAs rely on).

The model's output has `N_CLASSES` columns; the MISSING column is never a label, so the network learns to ignore it.
All code that turns logits into a distribution over activities selects only the `CLASS_IDS` columns
(`logits.index_select(1, class_ids).softmax(-1)`).

---

## 3. Rules as automata

### 3.1 LTLf and DFAs

DECLARE constraints are formulas of LTLf, linear temporal logic over **finite** traces (De Giacomo & Vardi, 2013),
the logic suited to process instances that start and end. For example:

* `existence(a)`: `F a`
* `response(a, b)`: `G(a -> F b)` ("every a is eventually followed by b")
* `chain_response(a, b)`: `G(a -> X b)` ("every a is immediately followed by b")

Every LTLf formula has an equivalent **deterministic finite automaton** that accepts exactly the finite traces
satisfying it. Checking a rule on a trace therefore reduces to running the automaton and testing whether the final
state is accepting. This is the only place logic enters; everything else is table lookups.

### 3.2 Representation

A DFA is stored as `delta[q, a] -> q'` (integer table, state 0 initial) plus a boolean vector `acc[q]`
(`DFA` class in `ltlf_features.py`). Example, `response(a, b)` with two states:

```
state 0 (nothing pending): on a -> 1,   otherwise -> 0
state 1 (a pending):       on b -> 0,   otherwise -> 1
accepting: {0}             (a trace that ends with a pending a violates the rule)
```

### 3.3 Template library

`ltlf_features.py` has nine templates: existence, init, end, response, precedence, chain_response,
chain_precedence, responded_existence, not_succession. `ltlf_patterns.py` adds (all verified against plain
definitions on 12,791 random traces, no mismatches):

* alternate_response / alternate_precedence / alternate_succession, succession, chain_succession,
  not_chain_succession;
* co_existence, not_coexistence, choice, exclusive_choice;
* `at_least(a, k)`, `at_most(a, k)` (cardinality);
* `bounded_response(a, b, k)` (b within k events);
* `factor_next_in(w, S)` (after the exact sequence `w`, if an event follows it is in `S`) and
  `forbidden_factor(w)` (the sequence `w` never occurs), for `|w| = 2, 3`;
* boolean combinations through `product(d1, d2, "and"/"or")` and `negate(d)`; every DFA is passed through
  `canon()` (reachable part, Moore minimisation, canonical numbering), so language-equivalent rules are detected
  and duplicates dropped.

---

## 4. Mining the rules

Rules are discovered on the **complete training traces** (the labels `y["Activity"]` of the training graphs,
converted to compact ids). Valid and test graphs never contribute rules.

### 4.1 Simple miner (`mine_rules`, used for the results in section 8)

1. Enumerate every instantiation of the nine templates over the `A` activities.
2. Support of a candidate = fraction of training traces its DFA accepts.
3. Keep support >= 0.98 (mined rules are rarely 100% true on real logs).
4. Drop vacuous binary rules whose trigger activity occurs in fewer than 5% of traces.
5. Rank by support x trigger frequency, keep the top 60.

Weakness: this keeps trivially true rules (for instance "activity X occurs" when X is in every trace) that rule
nothing out at a masked position.

### 4.2 Usefulness-driven miner (`mine_rich_rules`, `ltlf_patterns.py`)

1. Generate candidates from the chosen families (`basic`, `declare`, `counts`, `bounded`, `ngram`); n-gram
   candidates come from observed bigrams/trigrams with at least 30 occurrences, with `S` the smallest successor
   set covering 99.5% of cases. Remove language-equivalent duplicates.
2. Keep candidates with support >= 0.98 (vectorised acceptance over trace variants).
3. **Simulate masking**: sample 1,500 training traces (length 2 to 60), mask each event with probability 0.3
   (at least one per trace). For each candidate compute, for every masked node, which activities the rule rules
   out (the feasibility computation of section 6.2, vectorised across traces).
4. **Greedy selection** of up to 60 rules: at each step take the rule with the largest
   `(newly eliminated wrong (node, activity) pairs) - 3 * (newly eliminated true pairs)`; stop when the gain
   falls below 0.05% of the masked nodes or when the union would eliminate the true activity at more than 3%
   of masked nodes.
5. Print the share of wrong pairs eliminated and the share of masked nodes whose true activity would be
   eliminated. This is an **upper bound on what constraints can do** before any training, and it makes the
   original nine templates and the richer set directly comparable (the notebook prints both).

---

## 5. Semantic loss (training time)

Idea (Xu et al., 2018): add a term that measures how likely the model's outputs are to satisfy a logical
constraint. Here the constraint is "the restored trace is accepted by rule `r`", and its probability is
computed by running the DFA on distributions instead of symbols (in the spirit of the DFA-as-a-layer works of
Umili et al.).

### Step 1 - Distribution per position

For every Activity node of the batch:

* visible node: one-hot of its (compact) activity;
* missing node: `p_t = softmax(logits[CLASS_IDS])` (float32, MISSING class excluded).

### Step 2 - Run the DFA on distributions

Keep a distribution `s_t` over DFA states, `s_0 = onehot(q0)`. At every position take the *expected* transition:

```
s_{t+1}[q'] = sum_a p_t[a] * sum_q s_t[q] * 1[ delta(q, a) = q' ]
P_sat(r)    = sum_{q in F} s_n[q]
```

If the positions are independent, `P_sat(r)` is exactly the probability that a trace sampled from the
per-position distributions satisfies `r`. For a one-hot input it reduces to the ordinary DFA run (0 or 1).
Implementation: all rules are stacked into dense tensors `T[a, r, q, q']` (padded to the largest state count,
padding states self-loop and are non-accepting), so one batch matmul per step handles every trace and every rule.
Traces of different lengths are padded; a padded step leaves the state unchanged. Verified against brute-force
enumeration of all completions (maximum error 3e-8).

### Step 3 - Which (trace, rule) pairs count

A pair contributes only if:

1. **the ground-truth trace satisfies the rule.** Mined rules are about 98% true; for the other 2% the rule must
   not push the model toward wrong answers. This also guarantees `P_sat > 0` (the visible events cannot already
   violate the rule), so `-log P_sat` is finite;
2. **the trace has at least one missing Activity node**, otherwise `P_sat` is constant and gives no gradient.

The labels are used only to decide which pairs count, and only during training.

### Step 4 - Loss

```
L_sem = (1 / #pairs) * sum_{(trace, r) counted} w_r * ( - log max(P_sat(trace, r), 1e-6) )
```

`w_r` is the rule weight (its support, normalised to mean 1; pass `None` for uniform weights). The DFA part runs
in float32 with autocast turned off, because the network itself runs in bf16.

### Step 5 - Add to training

```
loss = sum_k  gradnorm_weight_k * CE_k   +   lambda_sem * L_sem
```

* the cross-entropy tasks keep their GradNorm weights; the semantic term is **outside GradNorm** and has one
  weight, `lambda_sem` (default 0.3, `0.0` = baseline);
* validation loss and early stopping still use plain cross-entropy, so runs with different `lambda_sem` are
  comparable and Optuna is not biased toward `lambda_sem = 0`; `lambda_sem` is fixed, not tuned.

Changes to `train_hgnn`: `task_losses` also returns the model output; the training step calls
`sem_loss(x, idx, out["Activity"])` when `lambda_sem > 0`; the backward pass uses the combined loss.

### Worked example

Rule `response(a, c)`; trace `[a, ?, d]`; model output at the missing node `p = (a: 0.1, c: 0.3, d: 0.6)`.
The DFA is in state "pending" after `a`, and only a `c` at the missing position clears it, so
`P_sat = p[c] = 0.3` and the term is `-log 0.3 = 1.20`. If training moves `p[c]` to 0.9 the term falls to 0.10.
The gradient raises the probability of `c` even though the label might also be explained by other signals.

### Hard variant of the satisfaction metric

`hard_satisfaction` fills the gaps with the argmax predictions and counts, among pairs whose rule holds on the
ground truth, how many are satisfied. It lets you report "share of rules satisfied" for any model.

### Known weakness

The loss uses independent marginals. It can be lowered by spreading probability instead of making one consistent
choice (two adjacent missing events with `p(c) = 0.5` each). It improves the model *on average*; it does not
guarantee consistency for an individual trace. Hard constraints address exactly this.

---

## 6. Hard constraints at inference time

Goal: for one trace, choose activities for **all** missing positions jointly, maximising the model's
log-probability, subject to every enforced rule accepting the completed trace. The network is unchanged.

### 6.1 Why joint decoding

With two missing positions, `p1 = p2 = (c: 0.4, d: 0.6)` and the rule "c occurs at least once": independent argmax
gives `(d, d)` which violates the rule; the probability that sampling satisfies it is `1 - 0.6*0.6 = 0.64`. The best
satisfying joint choice is `(c, d)` or `(d, c)` with score `log 0.4 + log 0.6`. Independent decisions cannot see
the rule; joint decoding can.

### 6.2 Backward viability sets

For each rule `r` and trace compute `B_t[q]`: from state `q` after `t` events, can acceptance still be reached by
the remaining events, where the remaining **missing** positions are wildcards (any activity) and known ones are
fixed:

```
B_n[q]   = [q in F]
B_t[q]   = B_{t+1}[ delta(q, a_{t+1}) ]                     if position t+1 is known
B_t[q]   = OR over a of B_{t+1}[ delta(q, a) ]              if position t+1 is missing
```

A rule is **active** for the trace iff `B_0[q0]` is true, i.e. some completion satisfies it. A rule the visible
events already violate is **ignored for that trace**, so a rule that is wrong on this trace can never force a
wrong answer.

### 6.3 Beam search over time

State of a hypothesis: the vector of current DFA states (one per rule), its accumulated score and a back-pointer.

```
hypotheses = { all rules in their initial state, score 0 }
for t in 1..n:
    if position t is known:
        advance every rule state with delta(q, a_t)
    else:
        for each hypothesis h and each activity a:
            q' = delta(q_h, a)                           # for every rule
            keep (h, a) only if for every active rule  q'_r in B_t   # viability prune
            score = score_h + log p_t[a]
        keep the best `beam` (h, a) pairs
return the best final hypothesis; recover choices through the back-pointers
```

Properties:

* the viability test is a necessary condition for each rule separately, so the search never walks into a dead end
  for any single rule; known events cannot cause pruning (viability is invariant under them);
* with `beam >=` the number of viable hypotheses the result is the exact optimum; with a small beam it is still
  rule-satisfying but possibly sub-optimal in score;
* cost per trace: `O(n * beam * A * R)` (R = number of enforced rules), done in numpy;
* **fallback**: if no hypothesis survives (the active rules are jointly unsatisfiable although each one alone
  is satisfiable) `decode` returns `None` and the plain argmax is used; the notebook reports the fallback rate
  (0.3% in the first experiment).

Tested against brute-force enumeration on 545 random traces with rule sets satisfied by the true trace: the beam
result equals the exact optimum and satisfies all rules in every case; with a small beam the result still
satisfies all rules.

### 6.4 Choosing which rules to enforce

Hard constraints are only as good as the rules. `sup_mins` selects rules by their **training support**:
enforce only rules with support >= `sup_min` (0.98, 0.99, 0.999 ...). Stricter thresholds enforce fewer, almost
always-true rules; looser thresholds enforce more rules but risk forcing errors where a rule does not hold.
`train_sup` is recomputed from the complete training traces for the current list of DFAs.

### 6.5 Evaluation (`evaluate_hard`, `test_multi`)

For every batch of test graphs:

1. forward pass; take the logits of the missing Activity nodes and turn them into log-probabilities over the real
   activities;
2. per trace: `plain = argmax`, `hard = decode(...)` (fallback to `plain` if `None`);
3. accuracy of both against the labels at the masked nodes;
4. rule satisfaction of both completions (only rules true on the ground-truth trace are counted, so plain and
   hard are compared on the same pairs);
5. `traces_changed` (share of traces where `hard != plain`) and `fallback` rate.

`test_multi` repeats this for each training run and each test type (`even`, `odd`, `window`, `random`,
`attr_level`) and writes, per test type, the full per-run table, a mean/std table and one `summary_hard.csv` with
plain accuracy, hard accuracy, the gain (mean and standard deviation over runs), satisfaction and changed traces,
for every `sup_min`. Only the `Activity` attribute is decoded under constraints; other attributes are evaluated
as before.

---

## 7. How the pieces fit (files)

| File | Role |
|---|---|
| `ltlf_features.py` | DFA class, nine templates, simple miner, per-node feasibility features (variant A) |
| `ltlf_patterns.py` | DFA algebra, richer templates, vectorised acceptance/exclusion, usefulness-driven miner |
| `ltlf_semantic_loss.py` | `DFABank`, `LTLfSemanticLoss`, `hard_satisfaction` |
| `ltlf_decode.py` | `HardDecoder` (backward sets + beam search), `evaluate_hard` |
| notebook "Variant B setup" cell | class-id inference, compact ids, mining, `dfas`, `sem_loss` |
| `train_hgnn` cell | semantic term added to the loss (`lambda_sem`) |
| `test_multi` cell | plain vs hard evaluation on every test type |

Typical use:

1. Run the setup cell (it asserts the encoding assumptions and prints the mined rules).
2. Train with `LAMBDA_SEM = 0.0` (baseline) and `> 0` (semantic loss).
3. Evaluate with `test_multi`; compare `Activity_acc` (plain) with `hard<sup_min>_acc_hard`.

---

## 8. Results so far

Setup: simple (nine-template) patterns, one trained network, `X_test`, 1,985 masked nodes, beam 16:

| Enforced rules | Rules | Accuracy | Rule satisfaction | Traces changed | Fallback |
|---|---|---|---|---|---|
| none (argmax) | - | 64.74% | 97.92% | - | - |
| support >= 0.999 | 17 | 69.52% | 100% | 14.0% | 0.0% |
| support >= 0.99 | 38 | 71.44% | 99.97% | 24.1% | 0.3% |
| support >= 0.98 | 51 | 72.19% | 99.97% | 28.1% | 0.3% |

Reading:

* With 1,985 masked nodes one standard error of an accuracy is about 1 point, so a 7.5-point gain is not noise.
* More rules gave more accuracy, so the rules seldom fail on test traces; a lower threshold or richer patterns may
  add more.
* The soft variants (features, semantic loss) showed little effect: the network's own predictions largely ignore
  the rules, whereas enforcing them at decoding time moves 28% of the traces.
* `sup_min` was compared on the test set, which slightly inflates the best number (see section 9).

---

## 9. Limitations and pitfalls

* **Rules are mined, not given.** A rule true on 98% of training traces is wrong on about 2%. Hard constraints
  can force an error on such a trace; the rule-selection step and the "active rule" check reduce but do not remove
  this. The `sup_min` sweep shows the trade-off.
* **Select hyperparameters on validation.** `sup_min`, `lambda_sem` and the beam should be chosen on `X_valid`,
  and the test set used once.
* **Independence.** The semantic loss assumes independent positions; the hard decoder removes this for the
  rules it enforces, but the network's own per-position probabilities remain independent marginals.
* **Joint feasibility.** The viability pruning is exact per rule, not across rules; the beam handles
  interactions heuristically, hence the fallback.
* **Scope.** Only control-flow (the activity sequence) is constrained; data and timing conditions are not.
* **Temporal order.** Nodes must be stored in time order; the setup cell asserts that `followed_by` edges join
  consecutive indices.
* **Statistical power.** Report mean and standard deviation over the 10 runs and all test types; use a paired
  comparison (same nodes, same networks) for the gain.
* **Untested code paths.** The numpy cores (DFA run, decoder, templates, mining) were tested against brute force.
  The PyTorch Geometric glue was exercised in your notebook, not in an automated test.

---

## 10. Possible extensions

* Richer rule families with the usefulness-driven miner (section 4.2), compared with the nine templates.
* Combining the semantic loss, the feasibility features and hard decoding (they are independent).
* Rule nodes in the heterogeneous graph (message passing through the rules).
* Learning the rule weights, or a soft penalty during decoding instead of a hard constraint.
* Data-aware rules (conditions on resource, organisation, timing).

---

## 11. References

Checked against their source pages:

* Dissegna, S., Di Francescomarino, C., Ronzani, M. (2025). *Graph-based Event Log Repair*. arXiv:2508.05145.
* De Giacomo, G., Vardi, M. Y. (2013). *Linear Temporal Logic and Linear Dynamic Logic on Finite Traces*. IJCAI.
* Di Ciccio, C., Mecella, M. (2015). *On the Discovery of Declarative Control Flows for Artful Processes*. ACM TMIS 5(4).
* Xu, J., Zhang, Z., Friedman, T., Liang, Y., Van den Broeck, G. (2018). *A Semantic Loss Function for Deep Learning
  with Symbolic Knowledge*. ICML, PMLR 80:5502-5511.
* Umili, E., Capobianco, R., De Giacomo, G. (2023). *Grounding LTLf Specifications in Image Sequences*. KR, 668-678.
* Umili, E., Capobianco, R. (2024). *DeepDFA: Automata Learning through Neural Probabilistic Relaxations*. arXiv:2408.08622.
* Umili, E., Argenziano, F., Capobianco, R. (2026). *DeepDFA: Injecting Temporal Logic in Deep Learning for
  Sequential Subsymbolic Applications*. arXiv:2602.03486.

Cited from memory, not re-checked (verify before use):

* Pesic, M., van der Aalst, W. M. P. (2006). *A declarative approach for flexible business process management*. BPM Workshops.
* Hamilton, W., Ying, Z., Leskovec, J. (2017). *Inductive representation learning on large graphs*. NeurIPS.
