# analysis.py
# Graphical representation, Descriptive Statistics, Measures of Position,
# Confidence Intervals, Frequency Tables — for Student Performance dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t
import warnings
warnings.filterwarnings("ignore")

from data_preprocessing import get_clean_data, NUM_COLS, CAT_COLS

# ─────────────────────────────────────────────
#  Helper: styled figure
# ─────────────────────────────────────────────
def _fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#444444")
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    ax.title.set_color("#222222")
    return fig, ax


# ═══════════════════════════════════════════════════════
#  DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════

def descriptive_stats(df):
    """Full descriptive stats: mean, median, mode, midrange, weighted mean,
       range, deviations, variance, std, CV, quartiles, deciles,
       IQR, skewness, kurtosis."""
    cols = [c for c in NUM_COLS if c in df.columns]
    rows = []
    for col in cols:
        s  = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        rows.append({
            "Variable"    : col,
            "N"           : len(s),
            "Mean"        : round(s.mean(), 3),
            "Median"      : round(s.median(), 3),
            "Mode"        : round(s.mode()[0], 3),
            "Mid-Range"   : round((s.max() + s.min()) / 2, 3),
            "Std Dev"     : round(s.std(), 3),
            "Variance"    : round(s.var(), 3),
            "Range"       : round(s.max() - s.min(), 3),
            "Min"         : round(s.min(), 3),
            "Max"         : round(s.max(), 3),
            "Q1 (25%)"    : round(q1, 3),
            "Q3 (75%)"    : round(q3, 3),
            "IQR"         : round(q3 - q1, 3),
            "Skewness"    : round(s.skew(), 3),
            "Kurtosis"    : round(s.kurtosis(), 3),
            "CV%"         : round(s.std() / s.mean() * 100, 2),
            "Coeff QD"    : round((q3 - q1) / (q3 + q1), 4),
        })
    return pd.DataFrame(rows).set_index("Variable")


def weighted_mean(df, value_col="Exam_Score", weight_col="Attendance"):
    """Compute weighted mean of Exam_Score weighted by Attendance."""
    w  = df[weight_col]
    v  = df[value_col]
    wm = (v * w).sum() / w.sum()
    return round(wm, 4)


# ─── Percentiles / Deciles / Quartiles table ─────────────────

def percentile_table(df, col="Exam_Score"):
    s = df[col].dropna()
    percentiles = list(range(10, 100, 10))
    rows = []
    for p in percentiles:
        rows.append({"Percentile": f"P{p}", "Value": round(np.percentile(s, p), 2)})
    return pd.DataFrame(rows)


def decile_table(df, col="Exam_Score"):
    s = df[col].dropna()
    rows = []
    for i in range(1, 11):
        rows.append({"Decile": f"D{i}", "Value": round(np.percentile(s, i * 10), 2)})
    return pd.DataFrame(rows)


def quartile_table(df, col="Exam_Score"):
    s = df[col].dropna()
    rows = [
        {"Quartile": "Q1 (25%)", "Value": round(s.quantile(0.25), 2)},
        {"Quartile": "Q2 (50%)", "Value": round(s.quantile(0.50), 2)},
        {"Quartile": "Q3 (75%)", "Value": round(s.quantile(0.75), 2)},
    ]
    return pd.DataFrame(rows)


# ─── Absolute & Relative Dispersion ──────────────────────────

def dispersion_table(df):
    cols = [c for c in NUM_COLS if c in df.columns]
    rows = []
    for col in cols:
        s  = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        mad = (s - s.mean()).abs().mean()
        rows.append({
            "Variable"          : col,
            "Range"             : round(s.max() - s.min(), 3),
            "Mean Abs Deviation": round(mad, 3),
            "Std Dev"           : round(s.std(), 3),
            "Variance"          : round(s.var(), 3),
            "IQR"               : round(q3 - q1, 3),
            "CV (%)"            : round(s.std() / s.mean() * 100, 2),
            "Coeff QD"          : round((q3 - q1) / (q3 + q1), 4),
        })
    return pd.DataFrame(rows).set_index("Variable")


# ═══════════════════════════════════════════════════════
#  CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════

def confidence_interval(series, conf=0.95):
    s     = series.dropna()
    n     = len(s)
    mu    = s.mean()
    se    = stats.sem(s)
    alpha = 1 - conf
    t_crit = t.ppf(1 - alpha / 2, df=n - 1)
    z_crit = norm.ppf(1 - alpha / 2)
    return {
        "n"      : n,
        "mean"   : round(mu, 4),
        "std"    : round(s.std(), 4),
        "se"     : round(se, 4),
        "conf"   : conf,
        "t_crit" : round(t_crit, 4),
        "t_lo"   : round(mu - t_crit * se, 4),
        "t_hi"   : round(mu + t_crit * se, 4),
        "z_crit" : round(z_crit, 4),
        "z_lo"   : round(mu - z_crit * se, 4),
        "z_hi"   : round(mu + z_crit * se, 4),
    }


def all_confidence_intervals(df, conf=0.95):
    cols = [c for c in NUM_COLS if c in df.columns]
    rows = []
    for col in cols:
        ci = confidence_interval(df[col], conf)
        rows.append({
            "Variable": col,
            "Mean"    : ci["mean"],
            "t-Lower" : ci["t_lo"],
            "t-Upper" : ci["t_hi"],
            "z-Lower" : ci["z_lo"],
            "z-Upper" : ci["z_hi"],
        })
    return pd.DataFrame(rows).set_index("Variable")


# ═══════════════════════════════════════════════════════
#  FREQUENCY TABLE
# ═══════════════════════════════════════════════════════

def frequency_table(df):
    bins   = [54, 60, 65, 70, 75, 80, 85, 90, 101]
    labels = ["55-60", "61-65", "66-70", "71-75", "76-80", "81-85", "86-90", "91-100"]
    df2    = df.copy()
    df2["band"] = pd.cut(df2["Exam_Score"], bins=bins, labels=labels,
                          right=True, include_lowest=True)
    freq  = df2["band"].value_counts().sort_index()
    total = freq.sum()
    return pd.DataFrame({
        "Grade Band"    : freq.index.astype(str),
        "Frequency"     : freq.values,
        "Relative (%)"  : (freq.values / total * 100).round(2),
        "Cumulative (%)": np.cumsum(freq.values / total * 100).round(2),
    })


# ═══════════════════════════════════════════════════════
#  GRAPHS — QUALITATIVE (Pie, Bar)
# ═══════════════════════════════════════════════════════

BLUE_PALETTE = ["#1a3a5c", "#4C72B0", "#AEC6E8", "#7bafd4",
                "#2c5f8a", "#6094c4", "#c8ddf0", "#3a6fa0"]

def plot_pie_gender(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#f8f9fa")
    counts = df["Gender"].value_counts()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           colors=["#4C72B0", "#e05c5c"], startangle=90,
           wedgeprops=dict(linewidth=2, edgecolor="white"))
    ax.set_title("Gender Distribution", color="#222222")
    fig.tight_layout()
    return fig


def plot_pie_school(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#f8f9fa")
    counts = df["School_Type"].value_counts()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           colors=["#4C72B0", "#AEC6E8"], startangle=90,
           wedgeprops=dict(linewidth=2, edgecolor="white"))
    ax.set_title("School Type", color="#222222")
    fig.tight_layout()
    return fig


def plot_bar_parental(df):
    fig, ax = _fig(7, 4)
    order = ["Low", "Medium", "High"]
    counts = df["Parental_Involvement"].value_counts().reindex(order, fill_value=0)
    bars = ax.bar(counts.index, counts.values,
                  color=["#AEC6E8", "#4C72B0", "#1a3a5c"], edgecolor="white")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 20,
                str(int(b.get_height())), ha="center", fontsize=10, color="#333")
    ax.set_xlabel("Parental Involvement")
    ax.set_ylabel("Count")
    ax.set_title("Parental Involvement Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bar_motivation(df):
    fig, ax = _fig(7, 4)
    order  = ["Low", "Medium", "High"]
    counts = df["Motivation_Level"].value_counts().reindex(order, fill_value=0)
    bars   = ax.bar(counts.index, counts.values,
                    color=["#AEC6E8", "#4C72B0", "#1a3a5c"], edgecolor="white")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 20,
                str(int(b.get_height())), ha="center", fontsize=10, color="#333")
    ax.set_xlabel("Motivation Level")
    ax.set_ylabel("Count")
    ax.set_title("Motivation Level Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bar_avg_by_cat(df, cat_col, title=None):
    """Average Exam Score by a categorical column."""
    fig, ax = _fig(7, 4)
    avg = df.groupby(cat_col)["Exam_Score"].mean().sort_values(ascending=False)
    colors = BLUE_PALETTE[:len(avg)]
    bars = ax.bar(avg.index.astype(str), avg.values,
                  color=colors, edgecolor="white")
    for b, v in zip(bars, avg.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{v:.1f}", ha="center", fontsize=9, color="#333")
    ax.set_xlabel(cat_col)
    ax.set_ylabel("Average Exam Score")
    ax.set_title(title or f"Avg Exam Score by {cat_col}")
    ax.set_ylim(0, avg.max() + 5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════
#  GRAPHS — QUANTITATIVE (Histogram, Scatter, Heatmap, Box)
# ═══════════════════════════════════════════════════════

def plot_histogram(df, col="Exam_Score"):
    fig, ax = _fig()
    ax.hist(df[col], bins=20, color="#4C72B0", edgecolor="white", linewidth=0.6)
    ax.axvline(df[col].mean(),   color="#e05c5c", linewidth=1.8,
               linestyle="--", label=f"Mean = {df[col].mean():.1f}")
    ax.axvline(df[col].median(), color="#2ca02c", linewidth=1.8,
               linestyle=":",  label=f"Median = {df[col].median():.1f}")
    ax.set_xlabel(col)
    ax.set_ylabel("Number of Students")
    ax.set_title(f"Distribution of {col}")
    ax.legend(framealpha=0.6, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shape_of_distribution(df):
    """Shows histogram + KDE to illustrate skewness / shape."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#f8f9fa")
    cols  = ["Exam_Score", "Hours_Studied", "Attendance"]
    names = ["Exam Score", "Hours Studied", "Attendance"]
    for ax, col, name in zip(axes, cols, names):
        s   = df[col].dropna()
        ax.set_facecolor("#ffffff")
        ax.hist(s, bins=20, density=True, color="#AEC6E8",
                edgecolor="white", linewidth=0.5)
        xs  = np.linspace(s.min(), s.max(), 300)
        kde = stats.gaussian_kde(s)
        ax.plot(xs, kde(xs), color="#1a3a5c", linewidth=2)
        skew = s.skew()
        shape = ("Symmetric / Normal" if abs(skew) < 0.3
                 else "Right Skewed" if skew > 0 else "Left Skewed")
        ax.set_title(f"{name}\nSkewness = {skew:.3f}\n({shape})",
                     fontsize=9, color="#222222")
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Shape of Distribution", fontsize=12, color="#1a3a5c", y=1.02)
    fig.tight_layout()
    return fig


def plot_scatter(df, x_col="Hours_Studied"):
    fig, ax = _fig()
    ax.scatter(df[x_col], df["Exam_Score"], color="#4C72B0",
               alpha=0.4, s=25, edgecolors="none")
    z  = np.polyfit(df[x_col], df["Exam_Score"], 1)
    xs = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    ax.plot(xs, np.poly1d(z)(xs), color="#e05c5c", linewidth=2,
            label=f"y = {z[0]:.2f}x + {z[1]:.2f}")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Exam Score")
    ax.set_title(f"{x_col} vs Exam Score")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#f8f9fa")
    num_df = df.select_dtypes(include=[np.number])
    corr   = num_df.corr()
    mask   = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="Blues", linewidths=0.5, ax=ax,
                annot_kws={"size": 9},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap", pad=12, color="#222222", fontsize=12)
    fig.tight_layout()
    return fig


def plot_boxplot(df):
    fig, ax = _fig(8, 4)
    num_df = df.select_dtypes(include=[np.number])
    cols   = [c for c in num_df.columns if c != "Exam_Score"]
    data   = [df[c].dropna().values for c in cols]
    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor="#AEC6E8", color="#4C72B0"),
                    medianprops=dict(color="#e05c5c", linewidth=2),
                    whiskerprops=dict(color="#4C72B0"),
                    capprops=dict(color="#4C72B0"),
                    flierprops=dict(marker="o", markerfacecolor="#4C72B0",
                                   markersize=3, alpha=0.4))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=7)
    ax.set_ylabel("Value")
    ax.set_title("Boxplot — All Numeric Variables")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_boxplot_score_by(df, cat_col):
    """Boxplot of Exam_Score grouped by a categorical column."""
    fig, ax = _fig(8, 4)
    groups  = df[cat_col].unique()
    data    = [df[df[cat_col] == g]["Exam_Score"].dropna().values for g in groups]
    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor="#AEC6E8", color="#4C72B0"),
                    medianprops=dict(color="#e05c5c", linewidth=2),
                    whiskerprops=dict(color="#4C72B0"),
                    capprops=dict(color="#4C72B0"),
                    flierprops=dict(marker="o", markerfacecolor="#4C72B0",
                                   markersize=3, alpha=0.4))
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_xlabel(cat_col)
    ax.set_ylabel("Exam Score")
    ax.set_title(f"Exam Score by {cat_col}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pie_pass_fail(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#f8f9fa")
    passed = (df["Exam_Score"] >= 67).sum()
    failed = len(df) - passed
    ax.pie([passed, failed],
           labels=["Above Average", "Below Average"],
           autopct="%1.1f%%",
           colors=["#4C72B0", "#e05c5c"],
           startangle=90,
           wedgeprops=dict(linewidth=2, edgecolor="white"))
    ax.set_title("Above / Below Average (67)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df = get_clean_data()
    print(descriptive_stats(df))
    print(dispersion_table(df))
    print(frequency_table(df))
    print("Weighted Mean:", weighted_mean(df))
