import ltlf_decode as LD
import ltlf_logic_layer as LL
from collections import Counter


def test_multi(config, outputcat, outputreal, num_runs=10, num_epochs=120, sup_min=0.98, beam=16, tag=None):
    """Train num_runs models (config["lambda_cond"] > 0 trains with the logic layer) and, on every test type, compare
         raw        argmax of the network output
         cond       argmax after the logic layer (conditioning on the active rules)
         hard       hard-constrained beam decoding on the raw output
         cond_hard  hard-constrained beam decoding on the conditioned output
    The layer and the decoder use the rules with train support >= sup_min (also used in training).
    tag: name of this configuration in the files (default: 'cond{lambda_cond}')."""
    global logic_layer
    tag = tag or f"cond{config.get('lambda_cond', 0.0)}"
    save_path = f"results/results_Caise_GRADNORM/{dataset}/"
    model_path = f"trained_model/{dataset}/{tag}/"
    os.makedirs(save_path, exist_ok=True); os.makedirs(model_path, exist_ok=True)
    keys = list(outputcat) + list(outputreal)

    cnt = Counter(tuple(s) for s in full_seqs)
    train_sup = np.array([sum(c for s, c in cnt.items() if d.accepts(s)) / len(full_seqs) for d in dfas])
    ridx = [k for k, s in enumerate(train_sup) if s >= sup_min]
    sub = [dfas[k] for k in ridx]
    logic_layer = LL.LTLfLogicLayer(sub, CLASS_IDS, N_CLASSES, act_key=ACT_KEY, n_iter=config.get("layer_iters", 1)).to(device)
    decoder = LD.HardDecoder(dfas, A_REAL, rule_idx=ridx)
    print(f"{tag}: sup_min={sup_min}, {len(ridx)} rules")

    tests = {}
    for test_type in test_types:
        with open(data_dir_graphs + dataset + f"_TEST_V2_repair_{test_type}.pkl", "rb") as f:
            X = pickle.load(f)
        for j in range(len(X)):
            X[j] = transform(X[j])
        tests[test_type] = X

    res = {}
    for run in range(num_runs):
        print(f"Run {run}")
        net = train_hgnn(config, outputcat, outputreal, num_epochs)
        torch.save(net.state_dict(), f"{model_path}run{run}.pt")
        with open(f"{model_path}run{run}_config.json", "w") as f:
            json.dump({**config, "best_val_loss": float(net.best_val_loss), "sup_min": sup_min}, f, indent=2)
        for test_type, X in tests.items():
            print(f"Test type {test_type}")
            r = evaluate_hgnn(net, outputcat, outputreal, test_graphs=X, mode="missing")
            h = LL.evaluate_layer(net, X, logic_layer, decoder, CLASS_IDS, MISSING_ID, N_CLASSES, keys, device,
                                  mask_to_idx, DataLoader, beam=beam)
            if abs(h["acc_raw"] - r.get("Activity_acc", h["acc_raw"])) > 1e-3:
                print(f"  warning: raw accuracy differs ({h['acc_raw']:.4f} vs {r['Activity_acc']:.4f})")
            r.update({f"layer_{k}": v for k, v in h.items()})
            res.setdefault(test_type, []).append(r)

    summary = []
    for test_type in test_types:
        tab, means, stds = create_df(res[test_type])
        tab.to_csv(f"{save_path}{test_type}_V2_RESULTS_END_{tag}.csv", index=False)
        pd.DataFrame({"mean": means, "std": stds}).to_csv(f"{save_path}{test_type}_V2_MEAN_STD_END_{tag}.csv")
        row = {"tag": tag, "test_type": test_type, "n_missing": tab["layer_n_missing"].iloc[0],
               "active_rules": tab["layer_active_rules"].mean()}
        for n in ("raw", "cond", "hard", "cond_hard"):
            row[f"acc_{n}"] = tab[f"layer_acc_{n}"].mean(); row[f"acc_{n}_std"] = tab[f"layer_acc_{n}"].std()
            row[f"sat_{n}"] = tab[f"layer_sat_{n}"].mean()
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{save_path}{dataset}_summary_{tag}.csv", index=False)
    print(summary.round(4).to_string(index=False))
    return res
