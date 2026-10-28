"""
Picture Judgment Facial Rating Analysis.

Analyzes implicit affective bias by examining participants' ratings of 25 prototypical
facial expressions (Happy and Upset) on a 1-7 Likert scale. 

Following standard psychological methodology (Vuoskoski & Eerola, 2012), ratings are
standardized to within-participant z-scores across all facial stimuli to measure relative
interpretive bias towards positive vs negative expressions.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import levene, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols

from src.config import (
    CONDITION_SHORT_MAP,
    IP_COLUMN_NAME,
    LIKERT_7PT_MAP,
    extract_group_port,
)


def load_and_standardize_ratings(
    data_path: Path, standardize_across_all: bool = True
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load survey data, map Likert responses, and compute within-subject z-scores."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df = pd.read_csv(data_path)

    # Identify face rating columns
    happy_cols = [
        col for col in df.columns if col.startswith("Happy") and not col.endswith("_zscore")
    ]
    upset_cols = [
        col for col in df.columns if col.startswith("Upset") and not col.endswith("_zscore")
    ]
    all_face_cols = happy_cols + upset_cols

    # Map Likert scale values
    for col in all_face_cols:
        df[col] = df[col].astype(str).str.lower().map(LIKERT_7PT_MAP)

    df["group"] = df[IP_COLUMN_NAME].apply(extract_group_port)
    filtered_df = df[df["group"] != "Other"].copy()

    if standardize_across_all:
        # Standardize across all 50 items per participant (relative judgment bias)
        def compute_overall_zscores(row: pd.Series) -> pd.Series:
            ratings = row[all_face_cols].dropna().astype(float)
            if len(ratings) < 2:
                return pd.Series([np.nan] * len(all_face_cols), index=[f"{c}_zscore" for c in all_face_cols])
            mean = ratings.mean()
            std = ratings.std()
            if std == 0:
                z = pd.Series(0.0, index=ratings.index)
            else:
                z = (ratings - mean) / std
            return pd.Series({f"{c}_zscore": z.get(c, np.nan) for c in all_face_cols})

        z_df = filtered_df.apply(compute_overall_zscores, axis=1)
    else:
        # Standardize separately within each emotion (script2 legacy mode)
        def compute_emotion_zscores(row: pd.Series, cols: List[str]) -> pd.Series:
            ratings = row[cols].dropna().astype(float)
            if len(ratings) < 2:
                return pd.Series([np.nan] * len(cols), index=[f"{c}_zscore" for c in cols])
            mean = ratings.mean()
            std = ratings.std()
            z = (ratings - mean) / std if std != 0 else pd.Series(0.0, index=ratings.index)
            return pd.Series({f"{c}_zscore": z.get(c, np.nan) for c in cols})

        happy_z = filtered_df.apply(lambda r: compute_emotion_zscores(r, happy_cols), axis=1)
        upset_z = filtered_df.apply(lambda r: compute_emotion_zscores(r, upset_cols), axis=1)
        z_df = pd.concat([happy_z, upset_z], axis=1)

    analysis_df = pd.concat([filtered_df[["ID", "group"]], z_df], axis=1)

    happy_z_cols = [f"{c}_zscore" for c in happy_cols]
    upset_z_cols = [f"{c}_zscore" for c in upset_cols]

    analysis_df["mean_happy_zscore"] = analysis_df[happy_z_cols].mean(axis=1)
    analysis_df["mean_upset_zscore"] = analysis_df[upset_z_cols].mean(axis=1)

    return analysis_df, happy_cols, upset_cols


def run_anova_with_assumptions(
    data: pd.DataFrame, dv_col: str, group_col: str = "group"
) -> Dict[str, Any]:
    """Perform one-way ANOVA with normality and homoscedasticity assumption checks."""
    sub_df = data.dropna(subset=[dv_col, group_col]).copy()
    alpha = 0.05

    # 1. Fit model to check residual normality
    model = ols(f"{dv_col} ~ C({group_col})", data=sub_df).fit()
    residuals = model.resid

    resid_stat, resid_p = shapiro(residuals) if len(residuals) >= 3 else (None, None)
    resid_normal = resid_p is not None and resid_p > alpha

    # 2. Within-group normality
    group_normality = {}
    all_groups_normal = True
    for name, grp in sub_df.groupby(group_col):
        vals = grp[dv_col].dropna()
        if len(vals) >= 3:
            s, p = shapiro(vals)
            norm = p > alpha
            group_normality[name] = {"stat": s, "p": p, "normal": norm}
            if not norm:
                all_groups_normal = False
        else:
            group_normality[name] = {"stat": None, "p": None, "normal": True}

    # 3. Levene's test for homogeneity of variance
    group_samples = [
        grp[dv_col].dropna().values
        for _, grp in sub_df.groupby(group_col)
        if len(grp[dv_col].dropna()) > 0
    ]
    if len(group_samples) >= 2:
        lev_stat, lev_p = levene(*group_samples)
        homogeneous = lev_p > alpha
    else:
        lev_stat, lev_p, homogeneous = None, None, False

    # 4. ANOVA Table
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_val_anova = anova_table["PR(>F)"].iloc[0]
    f_stat = anova_table["F"].iloc[0]

    descriptives = sub_df.groupby(group_col)[dv_col].agg(
        ["mean", "std", "count", "min", "max"]
    )

    return {
        "dv": dv_col,
        "model": model,
        "anova_table": anova_table,
        "f_stat": f_stat,
        "p_val": p_val_anova,
        "significant": p_val_anova < alpha,
        "resid_stat": resid_stat,
        "resid_p": resid_p,
        "resid_normal": resid_normal,
        "all_groups_normal": all_groups_normal,
        "group_normality": group_normality,
        "levene_stat": lev_stat,
        "levene_p": lev_p,
        "homogeneous": homogeneous,
        "descriptives": descriptives,
    }


def plot_facial_ratings(
    happy_res: Dict[str, Any], upset_res: Dict[str, Any], output_path: Path
) -> None:
    """Generate high-resolution comparative errorbar plot of facial expression z-scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    happy_desc = happy_res["descriptives"]
    upset_desc = upset_res["descriptives"]

    sorted_groups = sorted(happy_desc.index.tolist())
    x_pos = np.arange(len(sorted_groups))
    offset = 0.15

    happy_means = happy_desc.loc[sorted_groups, "mean"].values
    happy_sems = (
        happy_desc.loc[sorted_groups, "std"] / np.sqrt(happy_desc.loc[sorted_groups, "count"])
    ).values

    upset_means = upset_desc.loc[sorted_groups, "mean"].values
    upset_sems = (
        upset_desc.loc[sorted_groups, "std"] / np.sqrt(upset_desc.loc[sorted_groups, "count"])
    ).values

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.errorbar(
        x_pos - offset,
        happy_means,
        yerr=happy_sems,
        fmt="^",
        color="#2ca02c",
        ecolor="#2ca02c",
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        markersize=8,
        label="Happy Facial Rating (Z-score)",
    )
    ax.errorbar(
        x_pos + offset,
        upset_means,
        yerr=upset_sems,
        fmt="v",
        color="#ff7f0e",
        ecolor="#ff7f0e",
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        markersize=8,
        label="Upset Facial Rating (Z-score)",
    )

    ax.axhline(0, color="grey", linestyle=":", linewidth=1, alpha=0.8)

    xticklabels = [
        f"Group {g}\n{CONDITION_SHORT_MAP.get(g, '')}" for g in sorted_groups
    ]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xticklabels, fontsize=9.5)
    ax.set_ylabel(
        "Standardized Emotion Rating (Z-score $\\pm$ SEM)",
        fontsize=11,
        fontweight="semibold",
    )
    ax.set_title(
        "Picture Judgment Task: Facial Emotion Interpretation Bias Across Conditions",
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
    print(f"[+] Picture judgment plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run picture judgment facial rating ANOVA analysis."
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
    parser.add_argument(
        "--per-emotion-standardization",
        action="store_true",
        help="Standardize within emotion subset only (legacy mode)",
    )
    args = parser.parse_args()

    standardize_all = not args.per_emotion_standardization
    analysis_df, happy_cols, upset_cols = load_and_standardize_ratings(
        args.data, standardize_across_all=standardize_all
    )
    mode_str = "All Face Stimuli (Recommended)" if standardize_all else "Within Emotion Subset"
    print(f"Loaded {len(analysis_df)} participants | Standardization: {mode_str}")

    print("\n" + "=" * 60)
    print(" 1. ANOVA: MEAN HAPPY FACIAL RATINGS (Z-SCORES)")
    print("=" * 60)
    happy_res = run_anova_with_assumptions(analysis_df, "mean_happy_zscore")
    print(f"Normality of Residuals: W={happy_res['resid_stat']:.4f}, p={happy_res['resid_p']:.4f} (Normal={happy_res['resid_normal']})")
    print(f"Homogeneity of Variances: W={happy_res['levene_stat']:.4f}, p={happy_res['levene_p']:.4f} (Homogeneous={happy_res['homogeneous']})")
    print("\nANOVA Table:")
    print(happy_res["anova_table"])
    print(f"\nResult: F={happy_res['f_stat']:.4f}, p={happy_res['p_val']:.4f} ({'Significant' if happy_res['significant'] else 'Not significant'})")
    print("\nDescriptive Statistics per Group:")
    print(happy_res["descriptives"])

    print("\n" + "=" * 60)
    print(" 2. ANOVA: MEAN UPSET FACIAL RATINGS (Z-SCORES)")
    print("=" * 60)
    upset_res = run_anova_with_assumptions(analysis_df, "mean_upset_zscore")
    print(f"Normality of Residuals: W={upset_res['resid_stat']:.4f}, p={upset_res['resid_p']:.4f} (Normal={upset_res['resid_normal']})")
    print(f"Homogeneity of Variances: W={upset_res['levene_stat']:.4f}, p={upset_res['levene_p']:.4f} (Homogeneous={upset_res['homogeneous']})")
    print("\nANOVA Table:")
    print(upset_res["anova_table"])
    print(f"\nResult: F={upset_res['f_stat']:.4f}, p={upset_res['p_val']:.4f} ({'Significant' if upset_res['significant'] else 'Not significant'})")
    print("\nDescriptive Statistics per Group:")
    print(upset_res["descriptives"])

    plot_path = args.output_dir / "facial_emotion_ratings.png"
    plot_facial_ratings(happy_res, upset_res, plot_path)


if __name__ == "__main__":
    main()
