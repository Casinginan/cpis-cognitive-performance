# 05_stats_and_charts.py

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

NUMERIC_COLS = ["sleep_hours", "screen_time_hours", "physical_activity_minutes",
                 "stress_level", "diet_quality_score", "caffeine_intake_mg",
                 "sleep_quality_score", "age"]
TARGET = "cognitive_performance_score"

def plot_correlation_matrix(df):
    cols = NUMERIC_COLS + [TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig("reports/figures/correlation_matrix.png", dpi=130)
    plt.close(fig)
    return corr

def plot_histograms(df):
    cols = NUMERIC_COLS + [TARGET]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col], bins=25, color="#C97D44", edgecolor="#FAF6EF")
        axes[i].set_title(col, fontsize=10)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig("reports/figures/distributions.png", dpi=130)
    plt.close(fig)

def run_hypothesis_tests(df):
    low_quality = df[df["sleep_quality_score"] <= 5][TARGET]
    high_quality = df[df["sleep_quality_score"] >= 8][TARGET]
    t_stat, p_val = stats.ttest_ind(high_quality, low_quality, equal_var=False)
    print(f"t-test: high sleep quality (n={len(high_quality)}, mean={high_quality.mean():.1f}) vs low (n={len(low_quality)}, mean={low_quality.mean():.1f})")
    print(f"  t={t_stat:.3f}, p={p_val:.6f} -> {'significant' if p_val < 0.05 else 'not significant'}")
    r, p2 = stats.pearsonr(df["sleep_hours"], df[TARGET])
    print(f"\nsleep hours vs target: r={r:.3f}, p={p2:.6f}")

def feature_ranking(df):
    print("\nfeature significance (correlation with target):")
    rows = []
    for col in NUMERIC_COLS:
        r, p = stats.pearsonr(df[col], df[TARGET])
        rows.append((col, r, p))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, r, p in rows:
        print(f"  {col:30s} r={r:6.3f}  p={p:.4f}  ({'significant' if p < 0.05 else 'not significant'})")

if __name__ == "__main__":
    df = pd.read_csv("data/cpis_clean.csv")
    print("making charts...")
    plot_correlation_matrix(df)
    plot_histograms(df)
    print("saved to reports/figures/")
    print("\n--- hypothesis testing ---")
    run_hypothesis_tests(df)
    feature_ranking(df)
