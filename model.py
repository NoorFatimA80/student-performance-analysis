# model.py
# Covariance, Correlation, Simple & Multiple Linear Regression
# Model evaluation: R2, MSE, RMSE
# Prediction function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from data_preprocessing import get_clean_data, FEATURES

# Train model once at import time
_df    = get_clean_data()
_model = LinearRegression().fit(_df[FEATURES].values, _df["Exam_Score"].values)


# ─────────────────────────────────────────────
def _fig(w=7, h=4):
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
#  COVARIANCE & CORRELATION
# ═══════════════════════════════════════════════════════

def covariance_matrix(df):
    num_df = df[FEATURES + ["Exam_Score"]]
    return num_df.cov().round(4)


def correlation_matrix(df):
    num_df = df[FEATURES + ["Exam_Score"]]
    return num_df.corr().round(4)


def pairwise_covariance_correlation(df):
    rows = []
    for col in FEATURES:
        cov  = df[col].cov(df["Exam_Score"])
        corr = df[col].corr(df["Exam_Score"])
        rows.append({
            "Variable"          : col,
            "Covariance w/ Exam": round(cov, 4),
            "Pearson r"         : round(corr, 4),
            "r²"                : round(corr ** 2, 4),
            "Interpretation"    : (
                "Strong positive" if corr > 0.5 else
                "Moderate positive" if corr > 0.2 else
                "Weak positive" if corr > 0 else
                "Weak negative" if corr > -0.2 else
                "Moderate negative" if corr > -0.5 else
                "Strong negative"
            ),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════
#  SIMPLE LINEAR REGRESSION
# ═══════════════════════════════════════════════════════

def simple_regression(df, x_col):
    X      = df[[x_col]].values
    Y      = df["Exam_Score"].values
    m      = LinearRegression().fit(X, Y)
    y_pred = m.predict(X)
    r2     = r2_score(Y, y_pred)
    mse    = mean_squared_error(Y, y_pred)
    corr   = float(np.corrcoef(X.ravel(), Y)[0, 1])
    return {
        "slope"    : round(float(m.coef_[0]), 4),
        "intercept": round(float(m.intercept_), 4),
        "r2"       : round(r2, 4),
        "mse"      : round(mse, 4),
        "rmse"     : round(float(np.sqrt(mse)), 4),
        "pearson_r": round(corr, 4),
        "equation" : f"Exam_Score = {m.intercept_:.3f} + {m.coef_[0]:.3f} × {x_col}",
    }


# ═══════════════════════════════════════════════════════
#  MULTIPLE LINEAR REGRESSION
# ═══════════════════════════════════════════════════════

def multiple_regression(df, features=None):
    if features is None:
        features = FEATURES
    X = df[features].values
    Y = df["Exam_Score"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    m      = LinearRegression().fit(X_train, y_train)
    y_pred = m.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    mse    = mean_squared_error(y_test, y_pred)
    return {
        "intercept": round(float(m.intercept_), 4),
        "coef_df"  : pd.DataFrame({
            "Variable"   : features,
            "Coefficient": m.coef_.round(4),
        }),
        "r2"    : round(r2, 4),
        "mse"   : round(mse, 4),
        "rmse"  : round(float(np.sqrt(mse)), 4),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict_marks(hours, attendance, sleep, prev_score,
                  tutoring=1, physical=3):
    data = np.array([[hours, attendance, sleep, prev_score,
                      tutoring, physical]])
    pred = _model.predict(data)[0]
    return round(float(np.clip(pred, 55, 100)), 2)


# ═══════════════════════════════════════════════════════
#  GRAPHS
# ═══════════════════════════════════════════════════════

def plot_scatter_corr(df, x_col="Hours_Studied"):
    """Scatter + regression line with Pearson r."""
    res = simple_regression(df, x_col)
    fig, ax = _fig()
    ax.scatter(df[x_col], df["Exam_Score"], color="#4C72B0",
               alpha=0.4, s=25, edgecolors="none", label="Data points")
    xs = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    ax.plot(xs, res["slope"] * xs + res["intercept"],
            color="#e05c5c", linewidth=2, label="Regression line")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Exam Score")
    ax.set_title(f"{x_col} vs Exam Score  (r = {res['pearson_r']}, R² = {res['r2']})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_slr(df, x_col):
    return plot_scatter_corr(df, x_col)


def plot_actual_vs_pred(mlr):
    y_test = mlr["y_test"]
    y_pred = mlr["y_pred"]
    fig, ax = _fig()
    ax.scatter(y_test, y_pred, color="#4C72B0", alpha=0.5, s=25, edgecolors="none")
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color="#e05c5c", linewidth=2,
            linestyle="--", label="Perfect fit")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title(f"Actual vs Predicted  (R² = {mlr['r2']})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(mlr):
    residuals = mlr["y_test"] - mlr["y_pred"]
    fig, ax   = _fig()
    ax.scatter(mlr["y_pred"], residuals, color="#4C72B0",
               alpha=0.5, s=25, edgecolors="none")
    ax.axhline(0, color="#e05c5c", linewidth=2, linestyle="--")
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_coefficients(mlr):
    coef   = mlr["coef_df"].sort_values("Coefficient")
    colors = ["#e05c5c" if v < 0 else "#4C72B0" for v in coef["Coefficient"]]
    fig, ax = _fig(7, 3.5)
    bars = ax.barh(coef["Variable"], coef["Coefficient"],
                   color=colors, edgecolor="white")
    ax.axvline(0, color="#888888", linewidth=1)
    for bar, val in zip(bars, coef["Coefficient"]):
        ax.text(val + (0.02 if val >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9, color="#333333")
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Feature Coefficients (MLR)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_covariance_heatmap(df):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#f8f9fa")
    cov = covariance_matrix(df)
    sns.heatmap(cov, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax, annot_kws={"size": 8},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Covariance Matrix", pad=12, color="#222222", fontsize=12)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#f8f9fa")
    corr = correlation_matrix(df)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="Blues", linewidths=0.5, ax=ax,
                annot_kws={"size": 9},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap", pad=12, color="#222222", fontsize=12)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df = get_clean_data()
    print("SLR:", simple_regression(df, "Hours_Studied"))
    mlr = multiple_regression(df)
    print(f"MLR R²={mlr['r2']}  RMSE={mlr['rmse']}")
    print("Predict:", predict_marks(25, 85, 7, 78))
    print(pairwise_covariance_correlation(df))
