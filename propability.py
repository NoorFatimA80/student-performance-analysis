# probability.py
# Classical / Objective / Subjective probability, Permutation, Combination,
# Conditional Probability, Bayes' Theorem, Poisson, Hypergeometric, Binomial,
# Uniform, Normal distributions, Normality tests, Hypothesis testing
# — all tied to the Student Performance dataset

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, binom, poisson, hypergeom, uniform as sp_uniform
from scipy.special import comb as sp_comb
import warnings
warnings.filterwarnings("ignore")

from data_preprocessing import get_clean_data


# ─────────────────────────────────────────────
#  Styled figure helper
# ─────────────────────────────────────────────
def _fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#444444")
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    ax.title.set_color("#222222")
    return fig, ax


# ═══════════════════════════════════════════════════════
#  1. CLASSICAL / OBJECTIVE / SUBJECTIVE PROBABILITY
# ═══════════════════════════════════════════════════════

def classical_probability(df):
    """
    Classical: all outcomes equally likely.
    P(score in grade band) = favorable / total
    """
    total = len(df)
    bands = {
        "55-65": ((df["Exam_Score"] >= 55) & (df["Exam_Score"] <= 65)).sum(),
        "66-70": ((df["Exam_Score"] >= 66) & (df["Exam_Score"] <= 70)).sum(),
        "71-75": ((df["Exam_Score"] >= 71) & (df["Exam_Score"] <= 75)).sum(),
        "76-100": (df["Exam_Score"] >= 76).sum(),
    }
    rows = []
    for band, fav in bands.items():
        rows.append({
            "Event (Score Band)": band,
            "Favorable Outcomes": fav,
            "Total Outcomes"    : total,
            "Classical P"       : round(fav / total, 4),
        })
    return pd.DataFrame(rows)


def empirical_probability(df):
    """Objective / Relative-frequency probability."""
    total = len(df)
    rows  = []
    for col in ["Internet_Access", "Extracurricular_Activities",
                "Learning_Disabilities"]:
        for val in df[col].unique():
            cnt = (df[col] == val).sum()
            rows.append({
                "Variable": col,
                "Category": val,
                "Count"   : cnt,
                "Empirical P": round(cnt / total, 4),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════
#  2. PERMUTATION & COMBINATION
# ═══════════════════════════════════════════════════════

def permutation(n, r):
    """nPr = n! / (n-r)!"""
    if r > n:
        return 0
    return math.factorial(n) // math.factorial(n - r)


def combination(n, r):
    """nCr = n! / (r! * (n-r)!)"""
    if r > n:
        return 0
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))


def perm_comb_examples(df):
    n = len(df)
    r = 5
    return {
        "n (dataset size)"   : n,
        "r (choose)"         : r,
        "nPr"                : permutation(n, r),
        "nCr"                : combination(n, r),
        "nPr_10_3"           : permutation(10, 3),
        "nCr_10_3"           : combination(10, 3),
        "Explanation"        : (
            f"Choosing {r} students from {n}: "
            f"Ordered = {permutation(n, r):,}  |  Unordered = {combination(n, r):,}"
        ),
    }


# ═══════════════════════════════════════════════════════
#  3. CONDITIONAL PROBABILITY & BAYES' THEOREM
# ═══════════════════════════════════════════════════════

def conditional_probability(df):
    """
    P(High Score | High Study Hours)
    and P(High Study | High Score) via Bayes
    """
    high_score = df["Exam_Score"]   >= df["Exam_Score"].quantile(0.75)
    high_study = df["Hours_Studied"] >= df["Hours_Studied"].quantile(0.75)

    p_hs      = high_score.mean()          # P(High Score)
    p_hst     = high_study.mean()          # P(High Study)
    p_hs_and_hst = (high_score & high_study).mean()  # P(both)

    # P(High Score | High Study) = P(HS ∩ HST) / P(HST)
    p_hs_given_hst = p_hs_and_hst / p_hst if p_hst > 0 else 0

    # Bayes: P(High Study | High Score) = P(HS|HST)*P(HST) / P(HS)
    p_hst_given_hs = (p_hs_given_hst * p_hst / p_hs) if p_hs > 0 else 0

    return {
        "P(High_Score)"              : round(p_hs, 4),
        "P(High_Study)"              : round(p_hst, 4),
        "P(HighScore ∩ HighStudy)"   : round(p_hs_and_hst, 4),
        "P(High_Score | High_Study)" : round(p_hs_given_hst, 4),
        "P(High_Study | High_Score) [Bayes]": round(p_hst_given_hs, 4),
    }


def bayes_theorem(df, cat_col="Internet_Access", high_thresh=None):
    """
    Generic Bayes table for P(High Score | category).
    Returns a DataFrame for display.
    """
    if high_thresh is None:
        high_thresh = df["Exam_Score"].mean()
    high  = df["Exam_Score"] >= high_thresh
    total = len(df)
    rows  = []
    for val in df[cat_col].unique():
        mask  = df[cat_col] == val
        p_cat = mask.mean()
        p_hs_given_cat = (mask & high).sum() / mask.sum() if mask.sum() > 0 else 0
        p_joint = p_hs_given_cat * p_cat
        rows.append({
            "Category"              : val,
            "P(Category)"          : round(p_cat, 4),
            "P(HighScore|Category)": round(p_hs_given_cat, 4),
            "Joint P"              : round(p_joint, 4),
        })
    df_out = pd.DataFrame(rows)
    total_joint = df_out["Joint P"].sum()
    df_out["Posterior P(Cat|HighScore)"] = (
        df_out["Joint P"] / total_joint
    ).round(4)
    return df_out


# ═══════════════════════════════════════════════════════
#  4. NORMAL DISTRIBUTION
# ═══════════════════════════════════════════════════════

def get_normal_params(df):
    mu    = round(df["Exam_Score"].mean(), 4)
    sigma = round(df["Exam_Score"].std(), 4)
    return mu, sigma


def calc_probability(df, x, prob_type):
    mu, sigma = get_normal_params(df)
    if prob_type == "P(X <= x)":
        return round(norm.cdf(x, mu, sigma), 6)
    elif prob_type == "P(X > x)":
        return round(1 - norm.cdf(x, mu, sigma), 6)
    return None


def calc_probability_between(df, a, b):
    mu, sigma = get_normal_params(df)
    return round(norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma), 6)


def normality_tests(df):
    marks     = df["Exam_Score"].dropna()
    mu, sigma = get_normal_params(df)
    ks_stat, ks_p = stats.kstest(marks, "norm", args=(mu, sigma))
    sw_stat, sw_p = stats.shapiro(marks[:300])
    return {
        "ks_stat"  : round(float(ks_stat), 4),
        "ks_p"     : round(float(ks_p), 4),
        "sw_stat"  : round(float(sw_stat), 4),
        "sw_p"     : round(float(sw_p), 4),
        "normal_ks": float(ks_p) > 0.05,
        "normal_sw": float(sw_p) > 0.05,
    }


# ═══════════════════════════════════════════════════════
#  5. BINOMIAL DISTRIBUTION
# ═══════════════════════════════════════════════════════

def binomial_stats(df, n, k, threshold=None):
    if threshold is None:
        threshold = df["Exam_Score"].mean()
    p = float((df["Exam_Score"] >= threshold).mean())
    return {
        "p"        : round(p, 4),
        "n"        : n,
        "k"        : k,
        "pmf"      : round(float(binom.pmf(k, n, p)), 6),
        "cdf"      : round(float(binom.cdf(k, n, p)), 6),
        "expected" : round(n * p, 2),
        "variance" : round(n * p * (1 - p), 4),
        "std"      : round(np.sqrt(n * p * (1 - p)), 4),
    }


# ═══════════════════════════════════════════════════════
#  6. POISSON DISTRIBUTION
# ═══════════════════════════════════════════════════════

def poisson_stats(df, k):
    """
    Model: number of tutoring sessions per student follows Poisson.
    Lambda = mean tutoring sessions.
    """
    lam = df["Tutoring_Sessions"].mean()
    pmf = poisson.pmf(k, lam)
    cdf = poisson.cdf(k, lam)
    return {
        "lambda (mean sessions)": round(lam, 4),
        "k"                     : k,
        "PMF P(X=k)"            : round(float(pmf), 6),
        "CDF P(X<=k)"           : round(float(cdf), 6),
        "P(X>k)"                : round(1 - float(cdf), 6),
    }


# ═══════════════════════════════════════════════════════
#  7. HYPERGEOMETRIC DISTRIBUTION
# ═══════════════════════════════════════════════════════

def hypergeometric_stats(df, n_draw, k_success, cat_col="Internet_Access",
                          success_val="Yes"):
    """
    Population N = total students
    K = students with Internet Access = Yes (success states in population)
    n = drawn students
    k = expected successes in draw
    """
    N = len(df)
    K = int((df[cat_col] == success_val).sum())
    pmf = hypergeom.pmf(k_success, N, K, n_draw)
    cdf = hypergeom.cdf(k_success, N, K, n_draw)
    mean_hg = n_draw * K / N
    return {
        "N (Population)"        : N,
        "K (Success in Pop)"    : K,
        "n (Sample drawn)"      : n_draw,
        "k (Expected successes)": k_success,
        "PMF P(X=k)"            : round(float(pmf), 6),
        "CDF P(X<=k)"           : round(float(cdf), 6),
        "Expected value"        : round(mean_hg, 4),
    }


# ═══════════════════════════════════════════════════════
#  8. UNIFORM DISTRIBUTION
# ═══════════════════════════════════════════════════════

def uniform_stats(df, a=None, b=None):
    """Continuous uniform distribution over [a, b] of Exam_Score range."""
    if a is None:
        a = float(df["Exam_Score"].min())
    if b is None:
        b = float(df["Exam_Score"].max())
    mean_u    = (a + b) / 2
    var_u     = (b - a) ** 2 / 12
    std_u     = np.sqrt(var_u)
    return {
        "a (min)": a,
        "b (max)": b,
        "Mean"   : round(mean_u, 4),
        "Variance": round(var_u, 4),
        "Std Dev": round(std_u, 4),
        "P(X between a and b)": 1.0,
    }


def uniform_prob(x1, x2, a, b):
    """P(x1 <= X <= x2) for uniform distribution."""
    if x1 < a: x1 = a
    if x2 > b: x2 = b
    return round((x2 - x1) / (b - a), 6)


# ═══════════════════════════════════════════════════════
#  9. HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════

def hypothesis_test(df, mu0=67, alpha=0.05):
    marks      = df["Exam_Score"].dropna()
    t_stat, p  = stats.ttest_1samp(marks, mu0)
    reject     = float(p) < alpha
    return {
        "t_stat"  : round(float(t_stat), 4),
        "p_value" : round(float(p), 6),
        "alpha"   : alpha,
        "mu0"     : mu0,
        "reject"  : reject,
        "decision": ("Reject H0 (significant difference)" if reject
                     else "Fail to Reject H0 (no significant difference)"),
    }


def passing_probability(predicted_score, df=None):
    if df is None:
        df = get_clean_data()
    mean_score = df["Exam_Score"].mean()
    total      = len(df)
    above_mean = (df["Exam_Score"] >= mean_score).sum()
    base       = above_mean / total * 100
    if predicted_score >= 75:
        return round(min(base + 10, 99.9), 2)
    elif predicted_score >= mean_score:
        return round(base, 2)
    else:
        return round(max(base - 20, 1.0), 2)


# ═══════════════════════════════════════════════════════
#  GRAPHS
# ═══════════════════════════════════════════════════════

def plot_normal_fit(df):
    marks     = df["Exam_Score"].dropna()
    mu, sigma = get_normal_params(df)
    fig, ax   = _fig()
    ax.hist(marks, bins=20, density=True, color="#AEC6E8",
            edgecolor="white", linewidth=0.5, label="Observed data")
    x = np.linspace(marks.min() - 5, marks.max() + 5, 400)
    ax.plot(x, norm.pdf(x, mu, sigma), color="#4C72B0", linewidth=2.5,
            label=f"Normal fit  μ={mu:.1f}, σ={sigma:.1f}")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Density")
    ax.set_title("Normal Distribution Fit")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_qq(df):
    marks   = df["Exam_Score"].dropna()
    fig, ax = _fig()
    (osm, osr), (slope, intercept, r) = stats.probplot(marks, dist="norm")
    ax.scatter(osm, osr, color="#4C72B0", alpha=0.6, s=22)
    ax.plot(osm, slope * np.array(osm) + intercept,
            color="#e05c5c", linewidth=2, label=f"r = {r:.4f}")
    ax.set_title("Q-Q Plot (Normality Check)")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_binomial(df, n_students=30, threshold=None):
    if threshold is None:
        threshold = df["Exam_Score"].mean()
    p       = float((df["Exam_Score"] >= threshold).mean())
    k_range = np.arange(0, n_students + 1)
    pmf_all = binom.pmf(k_range, n_students, p)
    exp_k   = int(round(n_students * p))
    fig, ax = _fig(9, 4)
    ax.bar(k_range, pmf_all, color="#AEC6E8", edgecolor="white", linewidth=0.4)
    ax.bar(exp_k, pmf_all[exp_k], color="#4C72B0",
           label=f"Expected = {exp_k}")
    ax.set_xlabel("Number of Students (Above Avg Score)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Binomial Distribution  (n={n_students}, p={p:.2f})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_poisson(df, max_k=10):
    lam     = df["Tutoring_Sessions"].mean()
    k_range = np.arange(0, max_k + 1)
    pmf_all = poisson.pmf(k_range, lam)
    fig, ax = _fig(8, 4)
    ax.bar(k_range, pmf_all, color="#AEC6E8", edgecolor="white", linewidth=0.4)
    exp_k = int(round(lam))
    ax.bar(exp_k, pmf_all[exp_k], color="#4C72B0",
           label=f"λ = {lam:.2f}  (Expected = {lam:.2f})")
    ax.set_xlabel("Number of Tutoring Sessions")
    ax.set_ylabel("Probability")
    ax.set_title(f"Poisson Distribution  (λ = {lam:.2f})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_uniform(df):
    a    = float(df["Exam_Score"].min())
    b    = float(df["Exam_Score"].max())
    fig, ax = _fig(8, 4)
    x    = np.array([a, a, b, b])
    y    = np.array([0, 1 / (b - a), 1 / (b - a), 0])
    ax.fill_between([a, b], [1 / (b - a), 1 / (b - a)],
                    alpha=0.3, color="#4C72B0", label="Uniform PDF")
    ax.plot([a, a, b, b], [0, 1 / (b - a), 1 / (b - a), 0],
            color="#1a3a5c", linewidth=2)
    actual = df["Exam_Score"].dropna()
    ax.hist(actual, bins=20, density=True, color="#AEC6E8",
            edgecolor="white", linewidth=0.5, alpha=0.5, label="Actual data")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Uniform Distribution  U({a:.0f}, {b:.0f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_hypergeometric(df, n_draw=50):
    N       = len(df)
    K       = int((df["Internet_Access"] == "Yes").sum())
    k_range = np.arange(0, n_draw + 1)
    pmf_all = hypergeom.pmf(k_range, N, K, n_draw)
    exp_k   = int(round(n_draw * K / N))
    fig, ax = _fig(9, 4)
    ax.bar(k_range, pmf_all, color="#AEC6E8", edgecolor="white", linewidth=0.3)
    ax.bar(exp_k, pmf_all[exp_k], color="#4C72B0",
           label=f"Expected = {exp_k}")
    ax.set_xlabel("Students with Internet Access (in sample)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Hypergeometric Distribution  (N={N}, K={K}, n={n_draw})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_classical_prob(df):
    cp  = classical_probability(df)
    fig, ax = _fig(7, 4)
    bars = ax.bar(cp["Event (Score Band)"], cp["Classical P"],
                  color=["#1a3a5c", "#4C72B0", "#7bafd4", "#AEC6E8"],
                  edgecolor="white")
    for b, v in zip(bars, cp["Classical P"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=10, color="#333")
    ax.set_xlabel("Score Band")
    ax.set_ylabel("Classical Probability")
    ax.set_title("Classical Probability by Score Band")
    ax.set_ylim(0, cp["Classical P"].max() + 0.08)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df = get_clean_data()
    print("Normal params:", get_normal_params(df))
    print("Normality:", normality_tests(df))
    print("Hypothesis test:", hypothesis_test(df, 67))
    print("Binomial:", binomial_stats(df, 30, 15))
    print("Poisson:", poisson_stats(df, 2))
    print("Hypergeometric:", hypergeometric_stats(df, 50, 35))
    print("Uniform:", uniform_stats(df))
    print("Conditional:", conditional_probability(df))
    print("Perm/Comb:", perm_comb_examples(df))
