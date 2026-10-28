#!/usr/bin/env python3
"""
One figure per dataset: accuracy on the masked activities for every test type (even, odd, window, ...),
plain argmax vs hard-constrained decoding (one bar series per `sup_min` that test_multi evaluated).

Reads what test_multi writes:
    <results_dir>/<dataset>/<test_type>_V2_RESULTS_END_HARD.csv   (one row per run)   -> mean +- std over runs
    <results_dir>/<dataset>/<dataset>_summary_hard.csv                                -> fallback, no error bars

Usage:
    python plot_results.py                                  # all datasets under results/results_Caise_GRADNORM
    python plot_results.py --datasets bpi_2013_CZ --dark
    python plot_results.py --results-dir path/to/results --out-dir plots
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEST_ORDER = ["even", "odd", "window", "random", "attr_level"]

# categorical slots 1-3 of the validated palette (fixed order), light and dark steps
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e4e3df", axis="#9a9892",
                  series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", grid="#34342f", axis="#6f6e68",
                 series=["#3987e5", "#d95926", "#199e70"]),
}


def load_dataset(ds_dir: Path):
    """Returns (test_types, series, runs): series = {label: (mean[list], std[list or None])}, in plotting order."""
    name = ds_dir.name
    files = {re.sub(r"_V2_RESULTS_END_HARD\.csv$", "", f.name): f for f in ds_dir.glob("*_V2_RESULTS_END_HARD.csv")}
    if files:
        tts = sorted(files, key=lambda t: (TEST_ORDER.index(t) if t in TEST_ORDER else len(TEST_ORDER), t))
        tables = {t: pd.read_csv(files[t]) for t in tts}
        sms = sorted({m.group(1) for t in tts for c in tables[t].columns
                      if (m := re.fullmatch(r"hard([0-9.]+)_acc_hard", c))}, key=float)
        def ms(col):
            v = [tables[t][col] for t in tts]
            return [x.mean() for x in v], [x.std(ddof=1) if len(x) > 1 else 0.0 for x in v]
        series = {"Plain argmax": ms("Activity_acc")}
        for sm in sms:
            series[f"Hard constraints (rule support ≥ {sm})"] = ms(f"hard{sm}_acc_hard")
        return tts, series, int(min(len(tables[t]) for t in tts))
    summ = ds_dir / f"{name}_summary_hard.csv"
    if not summ.exists():
        return None
    df = pd.read_csv(summ)
    tts = sorted(df["test_type"], key=lambda t: (TEST_ORDER.index(t) if t in TEST_ORDER else len(TEST_ORDER), t))
    df = df.set_index("test_type").loc[tts]
    series = {"Plain argmax": (df["acc_plain"].tolist(), None)}
    for c in df.columns:
        if m := re.fullmatch(r"acc_hard_([0-9.]+)", c):
            series[f"Hard constraints (rule support ≥ {m.group(1)})"] = (df[c].tolist(), None)
    return tts, series, None


def plot_dataset(name, tts, series, runs, out_dir: Path, theme: str, formats):
    th = THEMES[theme]
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(tts) + 2.5), 4.9), facecolor=th["surface"])
    ax.set_facecolor(th["surface"])
    n = len(series); width = 0.8 / n; x = np.arange(len(tts))
    base_mean = np.array(list(series.values())[0][0])
    labels = list(series)
    for k, lab in enumerate(labels):
        mean, std = series[lab]; mean = np.array(mean)
        xs = x - 0.4 + width * (k + 0.5)
        ax.bar(xs, mean * 100, width * 0.92, color=th["series"][k % 3], label=lab, zorder=3, linewidth=0)
        if std is not None:
            ax.errorbar(xs, mean * 100, yerr=np.array(std) * 100, fmt="none", ecolor=th["text2"], elinewidth=1,
                        capsize=3, capthick=1, zorder=4)
        if k > 0:                                                    # gain over plain, in percentage points
            top = mean * 100 + (np.array(std) * 100 if std is not None else 0)
            for i, (xi, m, t) in enumerate(zip(xs, mean, top)):
                g = (m - base_mean[i]) * 100
                ax.text(xi, t + 1.2, f"{g:+.1f}", ha="center", va="bottom", fontsize=8.5, color=th["text"])
    ax.set_xticks(x, [t.replace("_", " ") for t in tts], color=th["text"], fontsize=10)
    ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], color=th["text2"], fontsize=9)
    ax.set_xlabel("Test set (type of missing data)", color=th["text2"], fontsize=10, labelpad=8)
    ax.yaxis.grid(True, color=th["grid"], linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(th["axis"]); ax.tick_params(axis="both", length=0)
    fig.suptitle(name, x=0.07, y=0.985, ha="left", fontsize=14, fontweight="bold", color=th["text"])
    sub = "Accuracy on masked activities" + (f", mean ± std over {runs} runs" if runs else "") + \
          "; numbers above the hard bars are the gain over plain argmax (percentage points)"
    fig.text(0.07, 0.915, sub, ha="left", fontsize=9, color=th["text2"])
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=n, frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(th["text"])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.22)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_dir / f"{name}_hard_constraints{'_dark' if theme == 'dark' else ''}.{fmt}"
        fig.savefig(p, dpi=200, facecolor=th["surface"]); paths.append(p)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/results_Caise_GRADNORM")
    ap.add_argument("--out-dir", default="plots")
    ap.add_argument("--datasets", nargs="*", help="dataset folder names (default: all found)")
    ap.add_argument("--formats", nargs="*", default=["png", "pdf"])
    ap.add_argument("--dark", action="store_true", help="dark theme")
    a = ap.parse_args()
    root = Path(a.results_dir)
    dirs = [root / d for d in a.datasets] if a.datasets else sorted(p for p in root.iterdir() if p.is_dir())
    done = 0
    for d in dirs:
        data = load_dataset(d) if d.is_dir() else None
        if data is None:
            print(f"skip {d.name}: no *_V2_RESULTS_END_HARD.csv or *_summary_hard.csv"); continue
        for p in plot_dataset(d.name, *data, Path(a.out_dir), "dark" if a.dark else "light", a.formats):
            print("wrote", p)
        done += 1
    print(f"{done} dataset plot(s) in {a.out_dir}/")


if __name__ == "__main__":
    main()
