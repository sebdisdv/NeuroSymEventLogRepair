import ltlf_decode as LD
from collections import Counter


def test_multi(config, outputcat, outputreal, num_runs=10, num_epochs=120, sup_mins=(0.98, 0.999), beam=16):
    """Train num_runs models; on every test type compare plain argmax with hard-constrained decoding.
    sup_mins: enforce only rules whose TRAIN support >= sup_min (one decoder per value)."""
    res = {}
    save_path = f"results/results_Caise_GRADNORM/{dataset}/"
    os.makedirs(save_path, exist_ok=True)
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
    summary.to_csv(f"{save_path}summary_hard.csv", index=False)
    print(summary.round(4).to_string(index=False))

    return res
