# %%
import torch
import pickle
import numpy as np
import pandas as pd
import os
import json 
from os.path import dirname



root_path = dirname(os.getcwd()) + "/NeuroSymEventLogRepair"

pd.set_option("display.max_columns", None)
# data_dir = root_path + "/data/datasets/original/"
data_dir_processed = root_path + "/data/datasets/_processed/"
data_dir_graphs = root_path + "/data/datasets/graphs_repair/"

print(root_path, data_dir_processed, data_dir_graphs, sep="\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# %%
ACT_TIME_ONLY = True

# %%
#ACT_TIME_ONLY = False

# %%
with open("data/dataset_features.json", 'r') as file:
    datasets_info = json.load(file)

# %%
list(datasets_info.keys())

# %%
dataset = "bpi_2012_CZ"

# %%
with open("data/dataset_features.json", 'r') as file:
    dataset_info = json.load(file)[dataset]

# %%
tab_all = pd.read_csv(f"{data_dir_processed}/{dataset}/{dataset}_processed_all.csv") 
tab_all.head()

# %%
print(f"number of events {len(tab_all)}")
print(f"number of traces {len(tab_all['CaseID'].unique())}")
print(f"number of attributes {len(tab_all.columns) - 1}") 

# %%
MISSING_VALUE = "MISSING_VALUE"

# %%
if ACT_TIME_ONLY:
    categorical_columns = ["Activity"]
    real_value_columns = ["time:timestamp"]
    dataset = f"{dataset}_AT_only"
else:
    categorical_columns = dataset_info["categorical"]
    real_value_columns = dataset_info["numerical"]

# %%
list_unique = {k : list(tab_all[k].unique()) + [MISSING_VALUE] for k in categorical_columns}
#list_unique

# %%
import random

torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)

# %%
with open(data_dir_graphs + dataset + "_TRAIN_V2_repair.pkl", "rb") as f:
    X_train = pickle.load(f)
with open(data_dir_graphs + dataset + "_VALID_V2_repair.pkl", "rb") as f:
    X_valid = pickle.load(f)
with open(data_dir_graphs + dataset + "_TEST_V2_repair.pkl", "rb") as f:
    X_test = pickle.load(f)

# %%
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

transform = ToUndirected()


with torch.no_grad():
        for i in range(len(X_train)):
                X_train[i] = transform(X_train[i])
        for i in range(len(X_valid)):
                X_valid[i] = transform(X_valid[i])
        for i in range(len(X_test)):
                X_test[i] = transform(X_test[i])

# %%
edge_types = set()
node_types = set()
for i in range(len(X_train)):
    n, edge_type = X_train[i].metadata()
    for x in n:
        node_types.add(x)
    for x in edge_type:
        edge_types.add(x)
for i in range(len(X_valid)):
    n, edge_type = X_valid[i].metadata()
    for x in n:
        node_types.add(x)
    for x in edge_type:
        edge_types.add(x)
for i in range(len(X_test)):
    n, edge_type = X_test[i].metadata()
    for x in n:
        node_types.add(x)
    for x in edge_type:
        edge_types.add(x)
        
node_types = list(node_types)
edge_types = list(edge_types)

# %% [markdown]
# ## Hyperopt

# %%
from ax.service.managed_loop import optimize
from torch_geometric.nn import (
    HeteroConv,
    SAGEConv,
    GATv2Conv
)
from torch.nn import (
    ModuleList,
    Module,
    Linear,
    ModuleDict
  )
from typing_extensions import Self

# %%
import torch
import torch.nn as nn
from torch.nn import Module, ModuleList, ModuleDict, Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from tqdm.notebook import tqdm


# %%
# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class HGNN(Module):
    def __init__(self, output_cat, output_real, nodes_relations, parameters):
        super().__init__()
        hid = parameters["hid"]
        layers = parameters["layers"]
        aggregation = parameters["aggregation"]
 
        self.output_cat = output_cat
        self.output_real = output_real
 
        self.convs = ModuleList()
        for _ in range(layers):
            self.convs.append(
                HeteroConv(
                    {
                        rel: SAGEConv((-1, -1), out_channels=hid, aggr=aggregation, normalize=True)
                        #rel: GATv2Conv((-1, -1), out_channels=hid, aggr=aggregation, residual=True, add_self_loops=False)
                        
                        for rel in nodes_relations
                    },
                    aggr=aggregation,
                )
            )
 
        self.FC = ModuleDict()
        for k in output_cat:
            self.FC[k] = Linear(hid, output_cat[k])
        for k in output_real:
            self.FC[k] = Linear(hid, 1)
 
    def forward(self, batch, idx):
        """idx: dict {task: LongTensor of node indices to predict}.
        Pass precomputed indices (from masks, ~masks, or `full`)."""
        x_dict = batch.x_dict  # local copy: batch is not mutated
        for conv in self.convs:
            x_dict = {k: v.relu() for k, v in conv(x_dict, batch.edge_index_dict).items()}
 
        out = {}
        for k in self.output_cat:
            out[k] = self.FC[k](x_dict[k].index_select(0, idx[k]))
        for k in self.output_real:
            out[k] = self.FC[k](x_dict[k].index_select(0, idx[k])).squeeze(-1)
        return out

 

 


# %%
import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import multiclass_accuracy
 
# Expected globals from your notebook: X_test, device
 
 
def _indices(x, keys, mode):
    """Node indices to evaluate per task. Runs on the CPU batch (no GPU sync)."""
    if mode == "full":      # every node of that type
        return {k: torch.arange(x.y[k].shape[0]) for k in keys}
    if mode == "correct":   # nodes NOT masked
        return {k: (~x.masks[k]).nonzero().squeeze(1) for k in keys}
    if mode == "missing":   # masked nodes (same as training/validation)
        return {k: x.masks[k].nonzero().squeeze(1) for k in keys}
    raise ValueError(f"unknown mode {mode!r}")
 
 
@torch.no_grad()
def evaluate_hgnn(net, output_cat, output_real, test_graphs=None,
                  mode="missing", batch_size=128, num_workers=4):
    if test_graphs is None:
        test_graphs = X_test
 
    keys = list(output_cat) + list(output_real)
    criteria = {k: nn.CrossEntropyLoss() for k in output_cat}
    criteria.update({k: nn.L1Loss() for k in output_real})
 
    loader = DataLoader(
        test_graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
 
    pred_cat = {k: [] for k in output_cat}
    tgt_cat = {k: [] for k in output_cat}
    pred_real = {k: [] for k in output_real}
    tgt_real = {k: [] for k in output_real}
    batch_losses = []  # one [n_tasks] tensor per batch, kept on GPU
 
    net.eval()
    for x in loader:
        idx = _indices(x, keys, mode)                     # CPU
        x = x.to(device, non_blocking=True)
        idx = {k: v.to(device, non_blocking=True) for k, v in idx.items()}
 
        labels = {k: x.y[k].index_select(0, idx[k]) for k in keys}
        out = net(x, idx)
 
        batch_losses.append(torch.stack([criteria[k](out[k], labels[k]) for k in keys]))
 
        for k in output_cat:
            pred_cat[k].append(out[k].argmax(dim=1))      # softmax is monotonic: not needed
            tgt_cat[k].append(labels[k])
        for k in output_real:
            pred_real[k].append(out[k])
            tgt_real[k].append(labels[k])
 
    # single GPU -> CPU sync for all per-batch losses
    L = torch.stack(batch_losses).cpu()                   # [n_batches, n_tasks]
    per_task = dict(zip(keys, L.mean(0).tolist()))        # mean of per-batch losses
    avg_total = L.sum(1).mean().item()
 
    accuracy = {
        k: multiclass_accuracy(
            torch.cat(pred_cat[k]), torch.cat(tgt_cat[k]), num_classes=output_cat[k]
        ).item()
        for k in output_cat
    }
    mae_global = {
        k: l1_loss(torch.cat(pred_real[k]), torch.cat(tgt_real[k])).item()
        for k in output_real
    }
 
    res = (
        {f"{k}_acc": accuracy[k] for k in output_cat}
        | {f"{k}_mae": per_task[k] for k in output_real}
        | {"AVG_total_loss": avg_total}
        | {f"MAE_{k}": mae_global[k] for k in output_real}
    )
    print(res)
    return res
 

# Drop-in replacements for your three functions
def test_hgnn_full(net, output_cat, output_real, test_graphs=None, **kw):
    return evaluate_hgnn(net, output_cat, output_real, test_graphs, mode="full", **kw)
 
 
def test_correct_hgnn(net, output_cat, output_real, test_graphs=None, **kw):
    return evaluate_hgnn(net, output_cat, output_real, test_graphs, mode="correct", **kw)
 
 
def test_misssing_hgnn(net, output_cat, output_real, test_graphs=None, **kw):
    return evaluate_hgnn(net, output_cat, output_real, test_graphs, mode="missing", **kw)


# %%

outputcat = {k : len(list_unique[k]) for k in list_unique}
outputreal = real_value_columns

if ACT_TIME_ONLY:
    outputcat= {"Activity" : outputcat["Activity"]}
    outputreal = ['time:timestamp']

print(outputcat)
print(outputreal)

# %% [markdown]
# ## LTLf semantic loss (variant B)
# Mines DECLARE rules on the complete training traces, compiles them to DFAs and builds `sem_loss`. `LAMBDA_SEM` weights the term added to the training loss; set it to `0.0` for the baseline.

# %%
# Variant B (semantic loss) -- needs ltlf_features.py and ltlf_semantic_loss.py next to this notebook.
from collections import Counter
import ltlf_features as LF

LAMBDA_SEM = 0.3     # weight of the semantic loss; 0.0 = baseline (no rules used)

ACT_KEY   = "Activity"
N_CLASSES = outputcat[ACT_KEY]              # 8 here: width of the Activity one-hot = model output size

def infer_missing_id(graphs, n=2000):
    """Class index used to encode masked Activity nodes (read from the data, not assumed)."""
    c = Counter()
    for g in graphs[:n]:
        m = g.masks[ACT_KEY].bool()
        c.update(g[ACT_KEY].x[m][:, :N_CLASSES].argmax(1).tolist())
    assert len(c) == 1, f"masked Activity nodes use several encodings: {c}"
    return next(iter(c))

MISSING_ID = infer_missing_id(X_train)                      # in your example graph: 5
CLASS_IDS  = [c for c in range(N_CLASSES) if c != MISSING_ID]   # real activities, in graph-class order
A_REAL     = len(CLASS_IDS)                                  # 7
_lut = torch.full((N_CLASSES,), -1, dtype=torch.long); _lut[CLASS_IDS] = torch.arange(A_REAL)
print("MISSING class id:", MISSING_ID, "| real activities:", A_REAL)

# class index -> compact DFA symbol 0..A_REAL-1 (the missing class is NOT a symbol of any rule)
def compact(ids):                     return _lut[torch.as_tensor(ids)].tolist()

def act_ids(g):
    """Compact activity id per Activity node; masked nodes -> A_REAL (what ltlf_features calls missing_id).
    No labels used."""
    cls = g[ACT_KEY].x[:, :N_CLASSES].argmax(1)
    out = _lut[cls]
    out[cls == MISSING_ID] = A_REAL
    return out.tolist()

# --- sanity checks on training graphs ---
for g in X_train[:200]:
    m, cls, y = g.masks[ACT_KEY].bool(), g[ACT_KEY].x[:, :N_CLASSES].argmax(1), g.y[ACT_KEY]
    assert (cls[m] == MISSING_ID).all() and (cls[~m] != MISSING_ID).all(), "masked <-> MISSING class mismatch"
    assert (cls[~m] == y[~m]).all(),            "x.argmax != y on visible nodes"
    assert (y != MISSING_ID).all(),             "labels contain the MISSING class"
    # temporal order: followed_by edges must connect consecutive node indices
    ei = g[(ACT_KEY, "followed_by", ACT_KEY)].edge_index
    assert ((ei[0] - ei[1]).abs() == 1).all(),  "nodes are not stored in consecutive (temporal) order"

# --- mine rules on COMPLETE train traces (labels), in compact ids ---
full_seqs = [compact(g.y[ACT_KEY]) for g in X_train]
rules = LF.mine_rules(full_seqs, A_REAL, min_support=0.98, top_k=60)
dfas  = [LF.build_dfa(A_REAL, t, *args) for t, args, _ in rules]
# readable names only if the graph encoding is the alphabetical one (check on a known trace!)
names = sorted(list_unique[ACT_KEY])
if names[MISSING_ID] == "MISSING_VALUE":
    names = [names[c] for c in CLASS_IDS]
    for t, args, s in rules[:10]: print(t, [names[a] for a in args], round(s, 3))
else:
    print("class order differs from sorted(list_unique); printing ids"); print(rules[:10])

import ltlf_semantic_loss as SL
sem_loss = SL.LTLfSemanticLoss(dfas, CLASS_IDS, N_CLASSES, act_key=ACT_KEY,
                               rule_weights=[s for _, _, s in rules]).to(device)


# %%
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
import torch.nn as nn
from copy import deepcopy
from tqdm.notebook import tqdm

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def mask_to_idx(batch, keys):
    """Run on CPU batch BEFORE .to(device): nonzero() causes no GPU sync there."""
    return {k: batch.masks[k].nonzero().squeeze(1) for k in keys}
 
 
def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = [t.pow(2).sum() for t in g if t is not None]
    return torch.stack(sq).sum().sqrt() if sq else loss.new_zeros(())
 
 
# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_hgnn(config, output_cat, output_real, epochs=20, callback=None):
    print(config)
 
    use_amp = config.get("amp", True)                 # bf16 autocast
    gn_every = config.get("gn_every", 1)              # GradNorm update every k steps
    num_workers = config.get("num_workers", 4)
    lam = config.get("lambda_sem", 0.0)               # weight of the LTLf semantic loss (0 = baseline)
    amp_kwargs = dict(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)
 
    net = HGNN(
        parameters=config,
        output_cat=output_cat,
        output_real=output_real,
        nodes_relations=edge_types,
    ).to(device)
 
    task_keys = list(output_cat) + list(output_real)
    n_tasks = len(task_keys)
    criteria = {k: nn.CrossEntropyLoss() for k in output_cat}
    criteria.update({k: nn.L1Loss() for k in output_real})
 
    loader_kw = dict(
        batch_size=config["batch_size"],
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(X_train, shuffle=True, **loader_kw)
    valid_loader = DataLoader(X_valid, shuffle=False, **loader_kw)
 
    def prepare(x):
        idx = mask_to_idx(x, task_keys)                       # CPU
        x = x.to(device, non_blocking=True)
        idx = {k: v.to(device, non_blocking=True) for k, v in idx.items()}
        labels = {k: x.y[k].index_select(0, idx[k]) for k in task_keys}
        return x, idx, labels
 
    def task_losses(x, idx, labels, return_out=False):
        with torch.autocast(**amp_kwargs):
            out = net(x, idx)
        l = torch.stack([criteria[k](out[k].float(), labels[k]) for k in task_keys])
        return (l, out) if return_out else l
 
    # Dry run so lazy (-1,-1) layers are initialised before optimizer / param lists
    with torch.no_grad():
        x0, i0, _ = prepare(next(iter(train_loader)))
        net(x0, i0)
        del x0, i0
 
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    shared = list(net.convs[-1].parameters())  # last shared layer for GradNorm
 
    # GradNorm state: created once, never rebuilt
    weights = torch.ones(n_tasks, device=device, requires_grad=True)
    w_opt = torch.optim.Adam([weights], lr=config["lr_gradNorm"])
    L0 = None
    step = 0
 
    best_state, best_loss, pat_count, patience = None, float("inf"), 0, 10
 
    for epoch in tqdm(range(epochs)):
        net.train()
        for x in train_loader:
            x, idx, labels = prepare(x)
            losses_values, out = task_losses(x, idx, labels, return_out=True)
 
            if L0 is None:
                L0 = losses_values.detach()
 
            do_gn = step % gn_every == 0
            if do_gn:
                # per-task gradient norms on last shared layer (no double backward)
                norms = torch.stack([grad_norm(losses_values[i], shared) for i in range(n_tasks)])
                gw = weights * norms                      # differentiable wrt weights only
                loss_ratio = losses_values.detach() / L0
                rt = loss_ratio / loss_ratio.mean()
                target = (gw.mean() * rt ** config["alpha"]).detach()
                gn_loss = (gw - target).abs().sum()
 
                w_opt.zero_grad(set_to_none=True)
                gn_loss.backward()                        # touches only `weights`
 
            optimizer.zero_grad(set_to_none=True)
            l_sem = sem_loss(x, idx, out["Activity"]) if lam > 0 else 0.0   # semantic loss, outside GradNorm
            (weights.detach() @ losses_values + lam * l_sem).backward()      # frees the graph
            optimizer.step()
 
            if do_gn:
                w_opt.step()
                with torch.no_grad():
                    weights.mul_(n_tasks / weights.sum()) # renormalise in place
            step += 1
 
        # ---- validation (single sync per epoch) ----
        net.eval()
        total = torch.zeros((), device=device)
        n_batches = 0
        with torch.no_grad():
            for x in valid_loader:
                x, idx, labels = prepare(x)
                total += task_losses(x, idx, labels).sum()
                n_batches += 1
        val_loss = (total / n_batches).item()
 
        print(f"Epoch {epoch}: val loss [{val_loss:.5f}] best loss [{best_loss:.5f}]")
 
        if callback is not None:
            callback(epoch, val_loss)   # may raise optuna.TrialPruned
 
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            pat_count = 0
        else:
            pat_count += 1
            if pat_count >= patience:
                break
 
    net.load_state_dict(best_state)
    net.best_val_loss = best_loss
    return net


# %%
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


# %%
import gc
import json
import math
 
import optuna
import torch
 
# Expected in the notebook: train_hgnn, evaluate_hgnn / test_* functions,
# X_train, X_valid, X_test, edge_types, device, output_cat, output_real
 
 
# ----------------------------------------------------------------------------
# Search space
# ----------------------------------------------------------------------------
def sample_config(trial):
    return {
        "hid": trial.suggest_categorical("hid", [32, 64, 128, 256]),
        "layers": trial.suggest_int("layers", 1, 4),
        "aggregation": trial.suggest_categorical("aggregation", ["mean", "sum", "max"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        #"batch_size": trial.suggest_categorical("batch_size", [256, 512,1024]),
        # GradNorm
        "lr_gradNorm": trial.suggest_float("lr_gradNorm", 1e-3, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 0.0, 2.0),
        #"gn_every": trial.suggest_categorical("gn_every", [1, 5, 10]),
        "gn_every": trial.suggest_categorical("gn_every", [1,2]),
        # fixed
        "amp": True,
        "num_workers": 4,
        "lambda_sem": LAMBDA_SEM,      # fixed (not tuned): set LAMBDA_SEM=0.0 for the baseline run
    }
 
 
# ----------------------------------------------------------------------------
# Objective: train on X_train, early-stop + score on X_valid (inside train_hgnn)
# ----------------------------------------------------------------------------
def make_objective(output_cat, output_real, epochs):
    def objective(trial):
        config = sample_config(trial)
        torch.manual_seed(trial.number)
 
        def callback(epoch, val_loss):
            if not math.isfinite(val_loss):
                raise optuna.TrialPruned()
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
 
        net = None
        try:
            net = train_hgnn(config, output_cat, output_real, epochs=epochs, callback=callback)
            return net.best_val_loss
        except torch.cuda.OutOfMemoryError:
            raise optuna.TrialPruned()      # config too big for the GPU
        finally:
            del net
            gc.collect()
            torch.cuda.empty_cache()
 
    return objective
 
 
def run_hpo(output_cat, output_real, n_trials=50, epochs=30, timeout=None,
            study_name="hgnn", storage=None, seed=0):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,                # re-run the cell to resume
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        make_objective(output_cat, output_real, epochs),
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
    )
 
    print("Best val loss:", study.best_value)
    print("Best params:", json.dumps(study.best_params, indent=2))
    try:
        print("Importances:", optuna.importance.get_param_importances(study))
    except Exception:
        pass
 
    #with open("best_params.json", "w") as f:
    #    json.dump(study.best_params, f, indent=2)
    return study


# %%
study = run_hpo(outputcat, outputreal, n_trials=20, epochs=120)
# Retrain best config (more epochs), still validated on X_valid for early stopping
best_config = {**study.best_params, "amp": True, "num_workers": 4, "lambda_sem": LAMBDA_SEM}
net = train_hgnn(best_config, outputcat, outputreal, epochs=120)
#torch.save(net.state_dict(), "best_hgnn.pt")
# Touch the test set only now, once
print("Test on missing")
test_misssing_hgnn(net, outputcat, outputreal)

print("Test on correct")
test_correct_hgnn(net, outputcat, outputreal)

print("Test on full")
test_hgnn_full(net, outputcat, outputreal)


# %% [markdown]
# ## Test on different test sets

# %%
test_types = ["even", "odd", "window", "random", "attr_level"]

# %%
import pandas


def create_df(results):
    res = {}
    
    for k in results[0]:
        res[k] = [x[k] for x in results]
    
    res = pandas.DataFrame(data=res)
    
    return res, res.mean(), res.std()
        
    

# %%
def test_multi(config, outputcat, outputreal, num_runs=10, num_epochs=120):
    
    res = {}
    
    save_path = f"results/results_Caise_GRADNORM/{dataset}/"
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    for i in range(num_runs):
        
        print(f"Run {i}")
        
        net = train_hgnn(
                config, 
                outputcat,
                outputreal,
                num_epochs
            )
        
        
        
        for test_type in test_types:
            
            print(f"Test type {test_type}")
            
            with open(data_dir_graphs + dataset + f"_TEST_V2_repair_{test_type}.pkl", "rb") as f:
                X = pickle.load(f)    
                
            for i in range(len(X)):
                X[i] = transform(X[i])

            if test_type not in res:
                res[test_type] = []
            
           
                
            
                
            r = evaluate_hgnn(
                net,
                outputcat,
                outputreal,
                test_graphs=X,
                mode="missing"
            )
            r["rule_satisfaction"] = satisfaction_rate(net, X, outputcat, outputreal)
            res[test_type].append(r)
    
    for test_type in test_types:    
        results_table, means, stds = create_df(res[test_type])
                
        results_table.to_csv(f"{save_path}{test_type}_V2_RESULTS_END.csv", sep=",", index=False)
        
        pd.DataFrame(data={"mean" : means, "std" : stds}).to_csv(f"{save_path}{test_type}_V2_MEAN_STD_END.csv", sep=",")
        
        print(test_type)
        print(pd.DataFrame(data={"mean" : means, "std" : stds}))
        
        
    
    return res

# %%
import ltlf_decode as LD
from collections import Counter
 
 
def test_multi(config, outputcat, outputreal, num_runs=10, num_epochs=120, sup_mins=(0.98, 0.999), beam=16):
    """Train num_runs models; on every test type compare plain argmax with hard-constrained decoding.
    sup_mins: enforce only rules whose TRAIN support >= sup_min (one decoder per value)."""
    res = {}
    save_path = f"results/results_Caise_GRADNORM/{dataset}/"
    model_path = f"trained_model/{dataset}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    keys = list(outputcat) + list(outputreal)
 
    # --- hard-constraint decoders (rules/DFAs come from the Variant-B setup cell: dfas, full_seqs, ...) ---
    cnt = Counter(tuple(s) for s in full_seqs)
    train_sup = np.array([sum(c for s, c in cnt.items() if d.accepts(s)) / len(full_seqs) for d in dfas])
    decoders = {}
    for sm in sup_mins:
        idx = [k for k, s in enumerate(train_sup) if s >= sm]
        if not idx:
            print(f"sup_min={sm}: no rule, skipped"); continue
        decoders[sm] = LD.HardDecoder(dfas, A_REAL, rule_idx=idx)
        print(f"sup_min={sm}: enforcing {len(idx)} rules")
 
    # --- load the test sets once (not once per run) ---
    tests = {}
    for test_type in test_types:
        with open(data_dir_graphs + dataset + f"_TEST_V2_repair_{test_type}.pkl", "rb") as f:
            X = pickle.load(f)
        for j in range(len(X)):
            X[j] = transform(X[j])
        tests[test_type] = X
 
    for run in range(num_runs):
        print(f"Run {run}")
        net = train_hgnn(config, outputcat, outputreal, num_epochs)
        torch.save(net.state_dict(), f"{model_path}run{run}.pt")          # best-validation weights of this run
        with open(f"{model_path}run{run}_config.json", "w") as f:
            json.dump({**config, "best_val_loss": float(net.best_val_loss)}, f, indent=2)
 
        for test_type, X in tests.items():
            print(f"Test type {test_type}")
            r = evaluate_hgnn(net, outputcat, outputreal, test_graphs=X, mode="missing")   # plain, all attributes
            for sm, dec in decoders.items():
                h = LD.evaluate_hard(net, X, dec, CLASS_IDS, MISSING_ID, N_CLASSES, keys, device,
                                     mask_to_idx, DataLoader, beam=beam)
                if abs(h["acc_plain"] - r.get("Activity_acc", h["acc_plain"])) > 1e-3:
                    print(f"  warning: plain accuracy differs ({h['acc_plain']:.4f} vs {r['Activity_acc']:.4f})")
                r.update({f"hard{sm}_{k}": v for k, v in h.items()})
            res.setdefault(test_type, []).append(r)
 
    # --- per test type: full tables + one summary comparing plain vs hard ---
    summary = []
    for test_type in test_types:
        results_table, means, stds = create_df(res[test_type])
        results_table.to_csv(f"{save_path}{test_type}_V2_RESULTS_END_HARD.csv", sep=",", index=False)
        pd.DataFrame(data={"mean": means, "std": stds}).to_csv(
            f"{save_path}{test_type}_V2_MEAN_STD_END_HARD.csv", sep=",")
        row = {"test_type": test_type, "acc_plain": results_table["Activity_acc"].mean(),
               "n_missing": results_table[f"hard{next(iter(decoders))}_n_missing"].iloc[0]}
        for sm in decoders:
            gain = results_table[f"hard{sm}_acc_hard"] - results_table["Activity_acc"]
            row.update({f"acc_hard_{sm}": results_table[f"hard{sm}_acc_hard"].mean(),
                        f"gain_{sm}": gain.mean(), f"gain_{sm}_std": gain.std(),
                        f"sat_plain_{sm}": results_table[f"hard{sm}_sat_plain"].mean(),
                        f"sat_hard_{sm}": results_table[f"hard{sm}_sat_hard"].mean(),
                        f"changed_{sm}": results_table[f"hard{sm}_traces_changed"].mean()})
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{save_path}{dataset}_summary_hard.csv", index=False)
    print(summary.round(4).to_string(index=False))
 
    return res


# %%
dataset

# %%
best_config

# %%
res = test_multi(best_config, outputcat, outputreal, 10, 120)

# %% [markdown]
# ## Hard constraints at inference
# `diagnose` shows how often the rules can exclude anything at all. `LD.HardDecoder` then decodes all missing activities of a trace jointly, keeping only assignments accepted by every enforced rule (beam search, exact for large beams). Rules the visible events already violate are ignored for that trace. Needs the trained `net` from the HPO/retrain cell.

# %%
# Do the rules carry information on this log? Per-position check on the test graphs.
# mask[i] = activities allowed at missing node i by ALL rules (a rule already violated by visible events is ignored).
def diagnose(graphs):
    A = A_REAL
    n_miss = n_excl_any = n_excl_true = 0; allowed = []
    for g in graphs:
        acts = act_ids(g); m = [i for i, a in enumerate(acts) if a == A]
        if not m: continue
        f = LF.rule_features(dfas, acts, A).reshape(len(acts), len(dfas), A + 1)
        feas = (f[:, :, :A] > 0) | (f[:, :, A:] > 0)          # unsatisfiable rule -> no restriction
        mask = feas.all(1); y = compact(g.y[ACT_KEY])
        for i in m:
            n_miss += 1; allowed.append(int(mask[i].sum()))
            n_excl_any  += mask[i].sum() < A
            n_excl_true += not mask[i, y[i]]
    print(f"missing nodes: {n_miss} | mean #activities left (of {A}): {np.mean(allowed):.2f}")
    print(f"rules exclude at least one activity at {100*n_excl_any/n_miss:.1f}% of missing nodes")
    print(f"rules exclude the TRUE activity at   {100*n_excl_true/n_miss:.2f}% of missing nodes")
diagnose(X_test)


# %%
import numpy as np
from collections import Counter
import ltlf_decode as LD

# enforce only rules that are (almost) always true on the training traces
cnt = Counter(tuple(s) for s in full_seqs)
train_sup = np.array([sum(c for s, c in cnt.items() if d.accepts(s)) / len(full_seqs) for d in dfas])
print("rules by train support:", np.round(np.sort(train_sup)[::-1][:60:6], 4))

keys = list(outputcat) + list(outputreal)
for SUP_MIN in (0.98, 0.99, 0.999):
    rule_idx = [i for i, s in enumerate(train_sup) if s >= SUP_MIN]
    decoder = LD.HardDecoder(dfas, A_REAL, rule_idx=rule_idx)
    r = LD.evaluate_hard(net, X_test, decoder, CLASS_IDS, MISSING_ID, N_CLASSES, keys, device,
                         mask_to_idx, DataLoader, beam=16)
    print(f"SUP_MIN={SUP_MIN} ({len(rule_idx)} rules):", {k: round(v, 4) for k, v in r.items()})



