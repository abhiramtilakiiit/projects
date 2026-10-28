"""
Word Recall Task Analysis.

Analyzes implicit mood-congruent memory bias by measuring the number of positive
and negative adjectives recalled immediately following the music listening phase.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, levene, shapiro, wilcoxon
import statsmodels.api as sm
from statsmodels.formula.api import ols

from src.config import (
    ADJECTIVE_RECALL_COLUMN_NAME,
    CONDITION_SHORT_MAP,
    IP_COLUMN_NAME,
    NEGATIVE_ADJECTIVES,
    POSITIVE_ADJECTIVES,
    extract_group_port,
)


def load_and_preprocess_data(data_path: Path) -> pd.DataFrame:
    """Load survey data and calculate positive and negative recall counts."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df = pd.read_csv(data_path)

    df["positive_count"] = 0
    df["negative_count"] = 0

    # Count recalled positive and negative adjectives
    for index, row in df.iterrows():
        raw_text = str(row[ADJECTIVE_RECALL_COLUMN_NAME]).lower()
        cleaned_text = re.sub(r"[,\s]+", " ", raw_text).strip()
        words = cleaned_text.split()

        pos_matches = [w for w in words if w in POSITIVE_ADJECTIVES]
        neg_matches = [w for w in words if w in NEGATIVE_ADJECTIVES]

        df.at[index, "positive_count"] = len(pos_matches)
        df.at[index, "negative_count"] = len(neg_matches)

    df["group"] = df[IP_COLUMN_NAME].apply(extract_group_port)
    filtered_df = df[df["group"] != "Other"].copy()

    filtered_df["difference_score"] = (
        filtered_df["positive_count"] - filtered_df["negative_count"]
    )
    filtered_df["total_count"] = (
        filtered_df["positive_count"] + filtered_df["negative_count"]
    )

    return filtered_df


def test_normality(data: pd.DataFrame, variable: str) -> Tuple[bool, Dict[str, Any]]:
    """Test normality within each group using the Shapiro-Wilk test."""
    results = {}
    all_normal = True

    for group_name, group_df in data.groupby("group"):
        vals = group_df[variable].dropna()
        if len(vals) < 3:
            results[group_name] = {"stat": None, "p": None, "normal": True}
            continue
        stat, p_val = shapiro(vals)
        normal = p_val > 0.05
        results[group_name] = {"stat": stat, "p": p_val, "normal": normal}
        if not normal:
            all_normal = False

    return all_normal, results


def test_homogeneity(data: pd.DataFrame, variable: str) -> Tuple[bool, float, float]:
    """Test homogeneity of variance across groups using Levene's test."""
    groups_data = [
        gdf[variable].dropna().values
        for _, gdf in data.groupby("group")
        if len(gdf[variable].dropna()) >= 2
    ]
    if len(groups_data) < 2:
        return False, float("nan"), float("nan")

    stat, p_val = levene(*groups_data)
    return p_val > 0.05, stat, p_val


def run_between_group_analysis(
    df: pd.DataFrame, variable: str, label: str
) -> Dict[str, Any]:
    """Perform between-group hypothesis testing (ANOVA or Kruskal-Wallis)."""
    alpha = 0.05
    normal, norm_res = test_normality(df, variable)
    homogeneous, lev_stat, lev_p = test_homogeneity(df, variable)

    if normal and homogeneous:
        test_type = "ANOVA"
        model = ols(f"{variable} ~ C(group)", data=df).fit()
        table = sm.stats.anova_lm(model, typ=2)
        p_value = table["PR(>F)"].iloc[0]
        stat_val = table["F"].iloc[0]
    else:
        test_type = "Kruskal-Wallis"
        groups = [
            gdf[variable].dropna().values
            for _, gdf in df.groupby("group")
            if len(gdf[variable].dropna()) > 0
        ]
        stat_val, p_value = kruskal(*groups)
        table = None

    means = df.groupby("group")[variable].mean().to_dict()

    return {
        "variable": variable,
        "label": label,
        "normal": normal,
        "homogeneous": homogeneous,
        "test_type": test_type,
        "stat": stat_val,
        "p_value": p_value,
        "significant": p_value < alpha,
        "table": table,
        "means": means,
    }


def run_within_group_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Test whether positive word recall exceeds negative word recall within each group."""
    results = {}
    alpha = 0.05

    for group_name, group_df in df.groupby("group"):
        if len(group_df) < 2:
            continue

        pos = group_df["positive_count"].tolist()
        neg = group_df["negative_count"].tolist()
        pos_mean = float(np.mean(pos))
        neg_mean = float(np.mean(neg))
        pos_sem = float(stats.sem(pos)) if len(pos) > 1 else 0.0
        neg_sem = float(stats.sem(neg)) if len(neg) > 1 else 0.0

        diffs = [p - n for p, n in zip(pos, neg)]
        if len(diffs) >= 3:
            s_stat, s_p = shapiro(diffs)
            normal = s_p > 0.05
        else:
            normal = False

        if normal:
            t_stat, p_val = stats.ttest_rel(pos, neg, alternative="greater")
            test_name = "Paired t-test"
            stat_str = f"t={t_stat:.4f}"
        else:
            w_stat, p_val = wilcoxon(pos, neg, alternative="greater")
            test_name = "Wilcoxon"
            stat_str = f"W={w_stat:.4f}"

        results[group_name] = {
            "pos_mean": pos_mean,
            "pos_sem": pos_sem,
            "neg_mean": neg_mean,
            "neg_sem": neg_sem,
            "diff_normal": normal,
            "test_name": test_name,
            "stat_str": stat_str,
            "p_val": p_val,
            "significant": p_val < alpha,
        }

    return results


def plot_word_recall(
    within_results: Dict[str, Dict[str, Any]], output_path: Path
) -> None:
    """Generate high-resolution publication-quality word recall errorbar plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_groups = sorted(within_results.keys())
    x_pos = np.arange(len(sorted_groups))
    offset = 0.15

    pos_means = [within_results[g]["pos_mean"] for g in sorted_groups]
    pos_sems = [within_results[g]["pos_sem"] for g in sorted_groups]
    neg_means = [within_results[g]["neg_mean"] for g in sorted_groups]
    neg_sems = [within_results[g]["neg_sem"] for g in sorted_groups]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.errorbar(
        x_pos - offset,
        pos_means,
        yerr=pos_sems,
        fmt="s",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        markersize=7,
        label="Positive Adjectives",
    )
    ax.errorbar(
        x_pos + offset,
        neg_means,
        yerr=neg_sems,
        fmt="o",
        color="#d62728",
        ecolor="#d62728",
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        markersize=7,
        label="Negative Adjectives",
    )

    # Annotations for significance
    y_offset = 0.35
    for i, group in enumerate(sorted_groups):
        res = within_results[group]
        p_val = res["p_val"]
        t_label = "t" if res["test_name"] == "Paired t-test" else "W"
        sig_text = f"p < .05 ({t_label})" if res["significant"] else "ns"

        max_y = max(pos_means[i] + pos_sems[i], neg_means[i] + neg_sems[i])
        ax.text(
            x_pos[i],
            max_y + y_offset,
            sig_text,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold" if res["significant"] else "normal",
            color="#222222",
        )

    xticklabels = [
        f"Group {g}\n{CONDITION_SHORT_MAP.get(g, '')}" for g in sorted_groups
    ]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xticklabels, fontsize=9.5)
    ax.set_ylabel(
        "Average Adjectives Recalled ($\\pm$ SEM)",
        fontsize=11,
        fontweight="semibold",
    )
    ax.set_title(
        "Implicit Word Recall as a Function of Music Valence and Contextual Narrative",
        fontsize=12.5,
        fontweight="bold",
        pad=15,
    )

    max_val = max(
        [
            max(p + ps, n + ns)
            for p, ps, n, ns in zip(pos_means, pos_sems, neg_means, neg_sems)
        ]
    )
    ax.set_ylim(0, max_val + y_offset + 0.8)

    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Word recall plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run word recall implicit affect analysis."
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

    df = load_and_preprocess_data(args.data)
    print(f"Loaded {len(df)} participants across groups:")
    print(df["group"].value_counts().sort_index())

    print("\n" + "=" * 60)
    print(" 1. BETWEEN-GROUP DIFFERENCE SCORES (Positive - Negative)")
    print("=" * 60)
    diff_res = run_between_group_analysis(
        df, "difference_score", "Difference Scores"
    )
    print(
        f"Assumptions: Normal={diff_res['normal']}, Homogeneous={diff_res['homogeneous']}"
    )
    print(
        f"Test: {diff_res['test_type']} | Stat={diff_res['stat']:.4f}, p={diff_res['p_value']:.4f}"
    )
    print(f"Significant difference across groups: {diff_res['significant']}")

    print("\n" + "=" * 60)
    print(" 2. BETWEEN-GROUP TOTAL ADJECTIVES RECALLED")
    print("=" * 60)
    total_res = run_between_group_analysis(df, "total_count", "Total Counts")
    print(
        f"Assumptions: Normal={total_res['normal']}, Homogeneous={total_res['homogeneous']}"
    )
    print(
        f"Test: {total_res['test_type']} | Stat={total_res['stat']:.4f}, p={total_res['p_value']:.4f}"
    )
    print(f"Significant difference across groups: {total_res['significant']}")

    print("\n" + "=" * 60)
    print(" 3. WITHIN-GROUP PAIRED COMPARISONS (Positive vs. Negative Recall)")
    print("=" * 60)
    within_res = run_within_group_analysis(df)
    for grp, res in sorted(within_res.items()):
        print(f"Group {grp}:")
        print(
            f"  Pos Mean={res['pos_mean']:.2f} (SEM={res['pos_sem']:.2f}) | "
            f"Neg Mean={res['neg_mean']:.2f} (SEM={res['neg_sem']:.2f})"
        )
        print(
            f"  {res['test_name']}: {res['stat_str']}, p={res['p_val']:.4f} "
            f"({'Significant' if res['significant'] else 'Not significant'})"
        )

    plot_path = args.output_dir / "word_recall_scores.png"
    plot_word_recall(within_res, plot_path)


if __name__ == "__main__":
    main()
