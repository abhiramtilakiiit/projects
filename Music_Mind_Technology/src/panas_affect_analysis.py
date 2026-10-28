"""
PANAS (Positive and Negative Affect Schedule) Affective Shift Analysis.

Analyzes explicit subjective emotional state changes by comparing baseline (Set 1)
to post-music listening (Set 2) ratings on the 20 PANAS mood adjectives.
Ratings are standardized across all items per participant to yield normalized z-scores.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, wilcoxon

from src.config import (
    CONDITION_SHORT_MAP,
    IP_COLUMN_NAME,
    LIKERT_5PT_MAP,
    NEGATIVE_ADJECTIVES,
    POSITIVE_ADJECTIVES,
    extract_group_port,
)


def load_and_preprocess_panas(data_path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load survey data, map 5-pt Likert ratings, and compute within-subject z-scores."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df = pd.read_csv(data_path)

    set1_cols = POSITIVE_ADJECTIVES + NEGATIVE_ADJECTIVES
    set2_cols = [f"{adj}2" if adj != "upset" else "upset27" for adj in set1_cols]
    all_40_cols = set1_cols + set2_cols

    for col in all_40_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(LIKERT_5PT_MAP)

    df["group"] = df[IP_COLUMN_NAME].apply(extract_group_port)
    filtered_df = df[df["group"] != "Other"].copy()

    # Calculate within-subject z-scores across all 40 items
    def calculate_zscores(row: pd.Series) -> pd.Series:
        existing = [c for c in all_40_cols if c in row.index and pd.notna(row[c])]
        if not existing or len(existing) < 2:
            return pd.Series([np.nan] * len(all_40_cols), index=[f"{c}_zscore" for c in all_40_cols])
        ratings = row[existing].dropna().astype(float)
        mean = ratings.mean()
        std = ratings.std()
        if std == 0:
            return pd.Series(0.0, index=[f"{c}_zscore" for c in all_40_cols])
        return pd.Series({f"{c}_zscore": (row[c] - mean) / std if c in existing else np.nan for c in all_40_cols})

    z_df = filtered_df.apply(calculate_zscores, axis=1)
    analysis_df = pd.concat([filtered_df[["ID", "group"]], z_df], axis=1)

    # Calculate mean positive and negative z-scores for Set 1 and Set 2
    pos_s1 = [f"{adj}_zscore" for adj in POSITIVE_ADJECTIVES]
    neg_s1 = [f"{adj}_zscore" for adj in NEGATIVE_ADJECTIVES]
    pos_s2 = [f"{adj}2_zscore" for adj in POSITIVE_ADJECTIVES]
    neg_s2 = [f"{adj}2_zscore" if adj != "upset" else "upset27_zscore" for adj in NEGATIVE_ADJECTIVES]

    analysis_df["mean_pos_zscore_set1"] = analysis_df[pos_s1].mean(axis=1)
    analysis_df["mean_neg_zscore_set1"] = analysis_df[neg_s1].mean(axis=1)
    analysis_df["mean_pos_zscore_set2"] = analysis_df[pos_s2].mean(axis=1)
    analysis_df["mean_neg_zscore_set2"] = analysis_df[neg_s2].mean(axis=1)

    analysis_df["pos_shift"] = (
        analysis_df["mean_pos_zscore_set2"] - analysis_df["mean_pos_zscore_set1"]
    )
    analysis_df["neg_shift"] = (
        analysis_df["mean_neg_zscore_set2"] - analysis_df["mean_neg_zscore_set1"]
    )

    analysis_df = analysis_df.dropna(
        subset=[
            "mean_pos_zscore_set1",
            "mean_neg_zscore_set1",
            "mean_pos_zscore_set2",
            "mean_neg_zscore_set2",
        ]
    )

    return analysis_df, set1_cols, set2_cols


def run_paired_affect_tests(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Test for significant pre-to-post changes in Positive and Negative Affect per group."""
    results = {}
    alpha = 0.05

    for group_name, grp in df.groupby("group"):
        if len(grp) < 3:
            continue

        # Positive affect change
        diff_pos = (grp["mean_pos_zscore_set2"] - grp["mean_pos_zscore_set1"]).dropna()
        stat_pos, p_norm_pos = shapiro(diff_pos) if len(diff_pos) >= 3 else (None, 1.0)
        norm_pos = p_norm_pos is not None and p_norm_pos > alpha

        s1_pos = grp["mean_pos_zscore_set1"].loc[diff_pos.index]
        s2_pos = grp["mean_pos_zscore_set2"].loc[diff_pos.index]

        if norm_pos:
            t_p, p_val_pos = stats.ttest_rel(s1_pos, s2_pos)
            pos_test = "Paired t-test"
            pos_stat_str = f"t={t_p:.4f}"
        else:
            w_p, p_val_pos = wilcoxon(s1_pos, s2_pos)
            pos_test = "Wilcoxon"
            pos_stat_str = f"W={w_p:.4f}"

        # Negative affect change
        diff_neg = (grp["mean_neg_zscore_set2"] - grp["mean_neg_zscore_set1"]).dropna()
        stat_neg, p_norm_neg = shapiro(diff_neg) if len(diff_neg) >= 3 else (None, 1.0)
        norm_neg = p_norm_neg is not None and p_norm_neg > alpha

        s1_neg = grp["mean_neg_zscore_set1"].loc[diff_neg.index]
        s2_neg = grp["mean_neg_zscore_set2"].loc[diff_neg.index]

        if norm_neg:
            t_n, p_val_neg = stats.ttest_rel(s1_neg, s2_neg)
            neg_test = "Paired t-test"
            neg_stat_str = f"t={t_n:.4f}"
        else:
            w_n, p_val_neg = wilcoxon(s1_neg, s2_neg)
            neg_test = "Wilcoxon"
            neg_stat_str = f"W={w_n:.4f}"

        results[group_name] = {
            "size": len(grp),
            "pos_diff_mean": float(diff_pos.mean()),
            "pos_diff_sem": float(stats.sem(diff_pos)) if len(diff_pos) > 1 else 0.0,
            "pos_test": pos_test,
            "pos_stat": pos_stat_str,
            "pos_p": float(p_val_pos),
            "pos_sig": p_val_pos < alpha,
            "neg_diff_mean": float(diff_neg.mean()),
            "neg_diff_sem": float(stats.sem(diff_neg)) if len(diff_neg) > 1 else 0.0,
            "neg_test": neg_test,
            "neg_stat": neg_stat_str,
            "neg_p": float(p_val_neg),
            "neg_sig": p_val_neg < alpha,
        }

    return results


def plot_affect_shifts(results: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """Plot mean affective shift (Post - Pre) for Positive and Negative affect."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_groups = sorted(results.keys())
    x = np.arange(len(sorted_groups))
    width = 0.35

    pos_shifts = [results[g]["pos_diff_mean"] for g in sorted_groups]
    pos_sems = [results[g]["pos_diff_sem"] for g in sorted_groups]
    neg_shifts = [results[g]["neg_diff_mean"] for g in sorted_groups]
    neg_sems = [results[g]["neg_diff_sem"] for g in sorted_groups]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    bars_pos = ax.bar(
        x - width / 2,
        pos_shifts,
        width,
        yerr=pos_sems,
        label="Positive Affect Shift ($\\Delta Z$)",
        color="#2b83ba",
        alpha=0.85,
        capsize=4,
        edgecolor="#1c557a",
    )
    bars_neg = ax.bar(
        x + width / 2,
        neg_shifts,
        width,
        yerr=neg_sems,
        label="Negative Affect Shift ($\\Delta Z$)",
        color="#d7191c",
        alpha=0.85,
        capsize=4,
        edgecolor="#8c1113",
    )

    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.7)

    xticklabels = [
        f"Group {g}\n{CONDITION_SHORT_MAP.get(g, '')}" for g in sorted_groups
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=9.5)
    ax.set_ylabel(
        "Mean Affect Shift ($Z_{post} - Z_{pre} \\pm$ SEM)",
        fontsize=11,
        fontweight="semibold",
    )
    ax.set_title(
        "Pre-to-Post Musical Exposure PANAS Affective Shifts",
        fontsize=12.5,
        fontweight="bold",
        pad=15,
    )

    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] PANAS affective shifts plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PANAS pre-vs-post affect paired analysis."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/experiment_data_clean.csv"),
        help="Path to cleaned dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to save generated figures",
    )
    args = parser.parse_args()

    analysis_df, s1, s2 = load_and_preprocess_panas(args.data)
    print(f"Loaded {len(analysis_df)} participants with paired PANAS ratings.")

    print("\n" + "=" * 60)
    print(" PANAS PRE- VS. POST-EXPOSURE AFFECT SHIFT (PAIRED TESTS)")
    print("=" * 60)
    results = run_paired_affect_tests(analysis_df)

    for grp, res in sorted(results.items()):
        print(f"\nGroup {grp} (n={res['size']}):")
        print(
            f"  Positive Affect: Diff={res['pos_diff_mean']:+.4f} | "
            f"{res['pos_test']}: {res['pos_stat']}, p={res['pos_p']:.4f} "
            f"({'Significant' if res['pos_sig'] else 'Not significant'})"
        )
        print(
            f"  Negative Affect: Diff={res['neg_diff_mean']:+.4f} | "
            f"{res['neg_test']}: {res['neg_stat']}, p={res['neg_p']:.4f} "
            f"({'Significant' if res['neg_sig'] else 'Not significant'})"
        )

    plot_path = args.output_dir / "panas_affect_shifts.png"
    plot_affect_shifts(results, plot_path)


if __name__ == "__main__":
    main()
