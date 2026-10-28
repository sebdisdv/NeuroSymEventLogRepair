
#!/usr/bin/env python3
"""
One figure per dataset: accuracy on masked activities for each test type,
with plain argmax, hard-constrained decoding, and AE/SANA (AT) baselines.

Reads:
    <results_dir>/<dataset>/<test_type>_V2_RESULTS_END_HARD.csv
    <results_dir>/<dataset>/<dataset>_summary_hard.csv
    activity_label_results.json

Usage:
    python plot_results.py
    python plot_results.py --datasets bpi_2013_CZ --dark
    python plot_results.py --results-dir path/to/results --out-dir plots
    python plot_results.py --baseline-json path/to/activity_label_results.json
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEST_ORDER = ["even", "odd", "window", "random", "attr_level"]

BASELINE_MASK = {
    "even": "EVEN",
    "odd": "ODD",
    "window": "WIND",
    "random": "RAND",
    "attr_level": "ATTR",
}

BASELINE_METHODS = ["AE", "SANA (AT)"]

# Optional aliases: normalized folder-name fragment -> baseline dataset key.
# Folder names are normalized by removing punctuation and converting to lowercase.
BASELINE_DATASET_ALIASES = {
    "small_log": "Small Log",
    "large_log": "Large Log",
    "bpi12": "BPI12",
    "bpi_2012": "BPI12",
    "bpi13": "BPI13",
    "bpi_2013": "BPI13",
    "sp2020": "SP2020",
    "sp_2020": "SP2020",
    "bpi20rfp": "BPI20RFP",
    "bpi_2020_rfp": "BPI20RFP",
    "BPI20_RequestForPayment_CZ" : "BPI20RFP"
}

# Colors: existing result series followed by AE and SANA (AT).
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        text="#0b0b0b",
        text2="#52514e",
        grid="#e4e3df",
        axis="#9a9892",
        series=[
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#8b6fc0",
            "#c49a00",
        ],
    ),
    "dark": dict(
        surface="#1a1a19",
        text="#f0efec",
        text2="#c3c2b7",
        grid="#34342f",
        axis="#6f6e68",
        series=[
            "#3987e5",
            "#d95926",
            "#199e70",
            "#a58bd4",
            "#e0bd4f",
        ],
    ),
}


def normalize_name(value):
    """Lowercase a name and remove non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def load_baselines(json_path):
    """Load baseline results from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        baselines = json.load(f)

    required_masks = set(BASELINE_MASK.values())
    required_methods = set(BASELINE_METHODS)

    for mask in required_masks:
        if mask not in baselines:
            raise ValueError(f"Baseline JSON is missing mask {mask!r}")
        for method in required_methods:
            if method not in baselines[mask]:
                raise ValueError(
                    f"Baseline JSON is missing method {method!r} under {mask!r}"
                )

    return baselines


def match_baseline_dataset(dataset_name, baselines):
    """Match a results folder name to a dataset key in the baseline JSON."""
    available = set(baselines["ATTR"]["AE"])
    normalized_available = {
        normalize_name(key): key for key in available
    }
    name = normalize_name(dataset_name)

    # Direct match, allowing punctuation differences.
    if name in normalized_available:
        return normalized_available[name]

    # Match configured aliases, preferring the longest matching alias.
    aliases = sorted(
        BASELINE_DATASET_ALIASES.items(),
        key=lambda item: len(normalize_name(item[0])),
        reverse=True,
    )

    for alias, baseline_key in aliases:
        if (
            normalize_name(alias) in name
            and baseline_key in available
        ):
            return baseline_key

    # Common dataset folder-name variations.
    if "bpi20rfp" in name or ("bpi2020" in name and "rfp" in name):
        return "BPI20RFP" if "BPI20RFP" in available else None

    if "bpi2013" in name or "bpi13" in name:
        return "BPI13" if "BPI13" in available else None

    if "bpi2012" in name or "bpi12" in name:
        return "BPI12" if "BPI12" in available else None

    if "sp2020" in name:
        return "SP2020" if "SP2020" in available else None

    if "smalllog" in name:
        return "Small Log" if "Small Log" in available else None

    if "largelog" in name:
        return "Large Log" if "Large Log" in available else None

    return None


def add_baseline_series(dataset_name, tts, series, baselines):
    """
    Append AE and SANA (AT) baseline bars.

    Baseline values in JSON are percentages (0-100); plotting data is
    represented as fractions (0-1), so divide by 100 here.
    """
    dataset_key = match_baseline_dataset(dataset_name, baselines)

    if dataset_key is None:
        print(
            f"warning: no baseline dataset mapping for {dataset_name!r}; "
            "plotting without baseline bars"
        )
        return series

    for method in BASELINE_METHODS:
        values = []

        for test_type in tts:
            mask = BASELINE_MASK.get(test_type)

            if mask is None:
                values.append(float("nan"))
                print(
                    f"warning: no baseline mask mapping for test type "
                    f"{test_type!r}; leaving its {method} baseline empty"
                )
                continue

            try:
                value = baselines[mask][method][dataset_key]
            except KeyError:
                value = float("nan")
                print(
                    f"warning: missing baseline value for "
                    f"{mask}/{method}/{dataset_key}"
                )

            values.append(value / 100.0 if np.isfinite(value) else float("nan"))

        series[f"Baseline: {method}"] = (values, None)

    return series


def load_dataset(ds_dir: Path):
    """
    Return (test_types, series, runs).

    series = {label: (mean[list], std[list or None])}
    """
    name = ds_dir.name

    files = {
        re.sub(r"_V2_RESULTS_END_HARD\.csv$", "", f.name): f
        for f in ds_dir.glob("*_V2_RESULTS_END_HARD.csv")
    }

    if files:
        tts = sorted(
            files,
            key=lambda t: (
                TEST_ORDER.index(t) if t in TEST_ORDER else len(TEST_ORDER),
                t,
            ),
        )
        tables = {t: pd.read_csv(files[t]) for t in tts}

        sms = sorted(
            {
                m.group(1)
                for t in tts
                for c in tables[t].columns
                if (m := re.fullmatch(r"hard([0-9.]+)_acc_hard", c))
            },
            key=float,
        )

        def ms(col):
            means = []
            stds = []

            for t in tts:
                if col not in tables[t].columns:
                    raise KeyError(
                        f"Column {col!r} missing in {files[t]}"
                    )

                vals = pd.to_numeric(tables[t][col], errors="coerce").dropna()
                means.append(vals.mean() if len(vals) else float("nan"))
                stds.append(
                    vals.std(ddof=1) if len(vals) > 1 else
                    (0.0 if len(vals) == 1 else float("nan"))
                )

            return means, stds

        series = {"Plain argmax": ms("Activity_acc")}

        for sm in sms:
            series[f"Hard constraints (rule support ≥ {sm})"] = ms(
                f"hard{sm}_acc_hard"
            )

        return tts, series, int(min(len(tables[t]) for t in tts))

    summ = ds_dir / f"{name}_summary_hard.csv"

    if not summ.exists():
        return None

    df = pd.read_csv(summ)

    if "test_type" not in df.columns:
        raise KeyError(f"'test_type' column missing in {summ}")

    tts = sorted(
        df["test_type"].astype(str).tolist(),
        key=lambda t: (
            TEST_ORDER.index(t) if t in TEST_ORDER else len(TEST_ORDER),
            t,
        ),
    )

    df = df.set_index("test_type").loc[tts]

    series = {
        "Plain argmax": (df["acc_plain"].tolist(), None)
    }

    for c in df.columns:
        if m := re.fullmatch(r"acc_hard_([0-9.]+)", c):
            series[f"Hard constraints (rule support ≥ {m.group(1)})"] = (
                df[c].tolist(),
                None,
            )

    return tts, series, None


def plot_dataset(
    name,
    tts,
    series,
    runs,
    out_dir: Path,
    theme: str,
    formats,
):
    th = THEMES[theme]

    fig, ax = plt.subplots(
        figsize=(max(7.5, 1.9 * len(tts) + 2.5), 4.9),
        facecolor=th["surface"],
    )
    ax.set_facecolor(th["surface"])

    labels = list(series)
    n = len(labels)
    width = 0.8 / n
    x = np.arange(len(tts))

    # Only actual model series participate in gain calculations.
    model_labels = [
        lab for lab in labels if not lab.startswith("Baseline:")
    ]
    base_mean = np.array(series["Plain argmax"][0], dtype=float)

    for k, lab in enumerate(labels):
        mean, std = series[lab]
        mean = np.asarray(mean, dtype=float)

        xs = x - 0.4 + width * (k + 0.5)

        ax.bar(
            xs,
            mean * 100,
            width * 0.92,
            color=th["series"][k % len(th["series"])],
            label=lab,
            zorder=3,
            linewidth=0,
        )

        if std is not None:
            std = np.asarray(std, dtype=float)
            ax.errorbar(
                xs,
                mean * 100,
                yerr=std * 100,
                fmt="none",
                ecolor=th["text2"],
                elinewidth=1,
                capsize=3,
                capthick=1,
                zorder=4,
            )

        # Annotate hard-constraint gains only, not baseline values.
        if (
            k > 0
            and lab in model_labels
            and not lab.startswith("Baseline:")
        ):
            top = mean * 100 + (
                np.asarray(std) * 100 if std is not None else 0
            )

            for i, (xi, m, t) in enumerate(zip(xs, mean, top)):
                if not np.isfinite(m) or not np.isfinite(base_mean[i]):
                    continue

                gain = (m - base_mean[i]) * 100
                ax.text(
                    xi,
                    t + 1.2,
                    f"{gain:+.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=th["text"],
                )

    ax.set_xticks(
        x,
        [t.replace("_", " ") for t in tts],
        color=th["text"],
        fontsize=10,
    )

    from matplotlib.ticker import MultipleLocator

    # More granular accuracy scale
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 5))
    ax.set_yticklabels(
        [f"{v}%" for v in range(0, 101, 5)],
        color=th["text2"],
        fontsize=8.5,
    )

    # Fine gridlines for easier value estimation
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.grid(
        True,
        which="major",
        axis="y",
        color=th["grid"],
        linewidth=0.8,
    )

    ax.grid(
        True,
        which="minor",
        axis="y",
        color=th["grid"],
        linewidth=0.35,
        alpha=0.65,
    )

    ax.set_xlabel(
        "Test set (type of missing data)",
        color=th["text2"],
        fontsize=10,
        labelpad=8,
    )

    ax.yaxis.grid(
        True,
        color=th["grid"],
        linewidth=0.8,
        zorder=0,
    )
    ax.set_axisbelow(True)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax.spines["bottom"].set_color(th["axis"])
    ax.tick_params(axis="both", length=0)

    fig.suptitle(
        name,
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=th["text"],
    )

    subtitle = "Accuracy on masked activities"
    if runs:
        subtitle += f", mean ± std over {runs} runs"
    subtitle += (
        "; numbers above hard-constraint bars show gain over plain argmax "
        "(percentage points)"
    )

    fig.text(
        0.07,
        0.915,
        subtitle,
        ha="left",
        fontsize=9,
        color=th["text2"],
    )

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(n, 3),
        frameon=False,
        fontsize=8.5,
    )

    for text in leg.get_texts():
        text.set_color(th["text"])

    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.86,
        bottom=0.29,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for fmt in formats:
        suffix = "_dark" if theme == "dark" else ""
        p = out_dir / f"{name}_hard_constraints{suffix}.{fmt}"
        fig.savefig(p, dpi=200, facecolor=th["surface"], bbox_inches="tight")
        paths.append(p)

    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--results-dir",
        default="results/results_Caise_GRADNORM",
    )
    ap.add_argument("--out-dir", default="plots")
    ap.add_argument(
        "--datasets",
        nargs="*",
        help="Dataset folder names (default: all found)",
    )
    ap.add_argument(
        "--formats",
        nargs="*",
        default=["png", "pdf"],
    )
    ap.add_argument(
        "--dark",
        action="store_true",
        help="Use dark theme",
    )
    ap.add_argument(
        "--baseline-json",
        default="activity_baselines.json",
        help="JSON file containing activity-label baseline results",
    )

    a = ap.parse_args()

    root = Path(a.results_dir)
    baseline_path = Path(a.baseline_json)

    if not baseline_path.exists():
        ap.error(
            f"Baseline JSON not found: {baseline_path}. "
            "Pass its location using --baseline-json."
        )

    baselines = load_baselines(baseline_path)

    if a.datasets:
        dirs = [root / d for d in a.datasets]
    else:
        if not root.exists():
            ap.error(f"Results directory not found: {root}")
        dirs = sorted(p for p in root.iterdir() if p.is_dir())

    done = 0

    for d in dirs:
        data = load_dataset(d) if d.is_dir() else None

        if data is None:
            print(
                f"skip {d.name}: no *_V2_RESULTS_END_HARD.csv "
                "or *_summary_hard.csv"
            )
            continue

        tts, series, runs = data
        series = add_baseline_series(d.name, tts, series, baselines)

        for p in plot_dataset(
            d.name,
            tts,
            series,
            runs,
            Path(a.out_dir),
            "dark" if a.dark else "light",
            a.formats,
        ):
            print("wrote", p)

        done += 1

    print(f"{done} dataset plot(s) in {a.out_dir}/")


if __name__ == "__main__":
    main()
