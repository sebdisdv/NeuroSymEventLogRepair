# ===== CELL A — put right after the ToUndirected cell (14) and after list_unique (11) =====
import ltlf_features as LF          # ltlf_features.py next to the notebook

USE_LTLF   = True                   # flip to False for the ablation baseline
LTLF_TOP_K = 60                     # number of mined rules -> 60 * (A+1) extra features per Activity node
LTLF_MIN_SUPPORT = 0.98

ACT_KEY    = "Activity"
MISSING_ID = len(list_unique[ACT_KEY]) - 1          # MISSING_VALUE is appended last in list_unique
A_REAL     = MISSING_ID                              # number of real activities

def act_ids(g):
    """Activity id of every Activity node (MISSING_ID where masked). No labels used."""
    x = g[ACT_KEY].x
    return x.argmax(1).tolist() if x.dim() == 2 else x.long().tolist()

# --- sanity checks on a few training graphs (decoding / ordering assumptions) ---
for g in X_train[:50]:
    ids  = torch.tensor(act_ids(g))
    m    = g.masks[ACT_KEY].bool()
    assert (ids[m] == MISSING_ID).all(),        "masked nodes are not encoded as MISSING_VALUE"
    assert (ids[~m] == g.y[ACT_KEY][~m].cpu()).all(), "x.argmax != y on visible nodes (x has extra columns?)"
    # if nodes are NOT stored in temporal order the DFA features are meaningless:
    # check against the timestamp column if you have it, e.g.
    # assert (g[ACT_KEY].time.diff() >= 0).all()

if USE_LTLF:
    # --- mine rules on COMPLETE train traces only (labels of train graphs, never valid/test) ---
    full_seqs = [g.y[ACT_KEY].tolist() for g in X_train]
    rules = LF.mine_rules(full_seqs, A_REAL, min_support=LTLF_MIN_SUPPORT, top_k=LTLF_TOP_K)
    acts_names = list_unique[ACT_KEY][:A_REAL]
    for t, args, s in rules[:10]:
        print(t, [acts_names[a] for a in args], round(s, 3))
    print(len(rules), "rules")
    dfas = [LF.build_dfa(A_REAL, t, *args) for t, args, _ in rules]

    # --- append [feasible_a0..a(A-1), unsat_flag] per rule to Activity-node features, IN PLACE ---
    # (run once! re-running would concatenate twice -> reload the pickles if you need to redo it)
    for graphs in (X_train, X_valid, X_test):
        LF.add_rule_features(graphs, dfas, MISSING_ID, node_type=ACT_KEY, act_ids_fn=act_ids)
    print("Activity x:", X_train[0][ACT_KEY].x.shape)

# HGNN needs no change: SAGEConv((-1,-1), ...) infers the new input width on the dry run.


# ===== CELL B — inside test_multi(), right after `X[i] = transform(X[i])` loop =====
#     if USE_LTLF:
#         LF.add_rule_features(X, dfas, MISSING_ID, node_type=ACT_KEY, act_ids_fn=act_ids)
# (the other test sets — even/odd/window/random/attr_level — have different masks, so they
#  need their own features; without this line they'd get a different input width and crash.)
