# app.py
# Student Performance Analysis System
# Probability & Statistics Final Project — Spring 2026
# Built with Python + Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data_preprocessing import get_clean_data, NUM_COLS, CAT_COLS, FEATURES
from analysis import (
    descriptive_stats, all_confidence_intervals, frequency_table,
    confidence_interval, dispersion_table, weighted_mean,
    percentile_table, decile_table, quartile_table,
    plot_histogram, plot_scatter, plot_heatmap,
    plot_boxplot, plot_boxplot_score_by,
    plot_bar_parental, plot_bar_motivation, plot_bar_avg_by_cat,
    plot_pie_gender, plot_pie_school, plot_pie_pass_fail,
    plot_shape_of_distribution,
)
from probability import (
    get_normal_params, calc_probability, calc_probability_between,
    normality_tests, binomial_stats, hypothesis_test,
    poisson_stats, hypergeometric_stats, uniform_stats, uniform_prob,
    conditional_probability, bayes_theorem, classical_probability,
    empirical_probability, perm_comb_examples,
    permutation, combination,
    plot_normal_fit, plot_qq, plot_binomial, plot_poisson,
    plot_uniform, plot_hypergeometric, plot_classical_prob,
    passing_probability,
)
from model import (
    simple_regression, multiple_regression, predict_marks,
    covariance_matrix, correlation_matrix, pairwise_covariance_correlation,
    plot_slr, plot_actual_vs_pred, plot_residuals, plot_coefficients,
    plot_covariance_heatmap, plot_correlation_heatmap,
)


# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #f0f4f8; }
[data-testid="stSidebar"] { background-color: #1a3a5c; }
[data-testid="stSidebar"] * { color: #e8edf2 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 13px !important; }
.page-title {
    background-color: #1a3a5c; color: white;
    padding: 18px 24px; border-radius: 8px; margin-bottom: 20px;
}
.page-title h2 { margin: 0; font-size: 1.5rem; font-weight: 600; color: white !important; }
.page-title p  { margin: 4px 0 0; font-size: 0.85rem; color: #a8c4e0 !important; }
.stat-card {
    background-color: white; border: 1px solid #d0dce8;
    border-left: 4px solid #1a3a5c; border-radius: 6px;
    padding: 14px 16px; text-align: center;
}
.stat-card .val { font-size: 1.6rem; font-weight: 700; color: #1a3a5c; font-family: monospace; }
.stat-card .lbl { font-size: 0.78rem; color: #666; margin-top: 2px;
    text-transform: uppercase; letter-spacing: 0.04em; }
.sec-head {
    font-size: 1.05rem; font-weight: 600; color: #1a3a5c;
    border-bottom: 2px solid #1a3a5c; padding-bottom: 4px; margin: 20px 0 12px;
}
.info-box {
    background-color: #eaf0f8; border-left: 4px solid #1a3a5c;
    border-radius: 4px; padding: 12px 16px;
    font-size: 0.88rem; color: #333; line-height: 1.7; margin: 8px 0;
}
.result-pass {
    background-color: #e8f5e9; border-left: 4px solid #2e7d32;
    border-radius: 4px; padding: 14px 18px; color: #1b5e20; margin-top: 12px;
}
.result-fail {
    background-color: #fce4ec; border-left: 4px solid #c62828;
    border-radius: 4px; padding: 14px 18px; color: #b71c1c; margin-top: 12px;
}
p, label { color: #333 !important; }
h3, h4   { color: #1a3a5c !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load():
    return get_clean_data()

df_full = load()


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Go to", [
        "Home / Data Overview",
        "Graphical Analysis",
        "Descriptive Statistics",
        "Probability",
        "Distributions",
        "Regression & Prediction",
    ])

    st.markdown("---")
    st.markdown("**Filter Data**")
    att_range = st.slider("Attendance %",
                           int(df_full["Attendance"].min()),
                           int(df_full["Attendance"].max()),
                           (int(df_full["Attendance"].min()),
                            int(df_full["Attendance"].max())))
    sh_range = st.slider("Hours Studied", 1, 44, (1, 44))

    gender_opts = ["All"] + sorted(df_full["Gender"].unique().tolist())
    sel_gender  = st.selectbox("Gender", gender_opts)

    df = df_full[
        df_full["Attendance"].between(*att_range) &
        df_full["Hours_Studied"].between(*sh_range)
    ].copy()
    if sel_gender != "All":
        df = df[df["Gender"] == sel_gender]

    st.markdown("---")
    st.markdown(f"Showing **{len(df)}** of {len(df_full)} records")


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "Home / Data Overview":

    st.markdown("""
    <div class="page-title">
      <h2>Student Performance Analysis System</h2>
      <p>Probability &amp; Statistics Final Project | Spring 2026 | Python + Streamlit</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (f"{df['Exam_Score'].mean():.1f}", "Average Exam Score"),
        (f"{(df['Exam_Score'] >= df['Exam_Score'].mean()).mean()*100:.1f}%", "Above Average Rate"),
        (f"{df['Hours_Studied'].mean():.1f} hrs", "Avg Study Hours"),
        (f"{df['Attendance'].mean():.1f}%", "Avg Attendance"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f'<div class="stat-card"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<div class="sec-head">Variable Descriptions</div>', unsafe_allow_html=True)
    vdesc = pd.DataFrame({
        "Variable": [
            "Hours_Studied", "Attendance", "Parental_Involvement",
            "Access_to_Resources", "Extracurricular_Activities", "Sleep_Hours",
            "Previous_Scores", "Motivation_Level", "Internet_Access",
            "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
            "School_Type", "Peer_Influence", "Physical_Activity",
            "Learning_Disabilities", "Parental_Education_Level",
            "Distance_from_Home", "Gender", "Exam_Score",
        ],
        "Type": ["Numeric","Numeric","Categorical","Categorical","Categorical",
                 "Numeric","Numeric","Categorical","Categorical","Numeric",
                 "Categorical","Categorical","Categorical","Categorical",
                 "Numeric","Categorical","Categorical","Categorical",
                 "Categorical","Numeric (Target)"],
        "Description": [
            "Hours studied per week",
            "Attendance percentage (%)",
            "Level of parental involvement (Low/Medium/High)",
            "Access to educational resources (Low/Medium/High)",
            "Whether student participates in extracurricular (Yes/No)",
            "Average hours of sleep per night",
            "Score in previous exam",
            "Student motivation level (Low/Medium/High)",
            "Whether student has internet access (Yes/No)",
            "Number of tutoring sessions attended",
            "Family income level (Low/Medium/High)",
            "Teacher quality rating (Low/Medium/High)",
            "Type of school (Public/Private)",
            "Peer influence (Negative/Neutral/Positive)",
            "Hours of physical activity per week",
            "Whether student has learning disabilities (Yes/No)",
            "Parental education level",
            "Distance from home to school",
            "Gender (Male/Female)",
            "Final exam score (target variable)",
        ],
    })
    st.dataframe(vdesc, use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-head">Basic Info</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(f"""
    <div class="info-box">
    Total records: <b>{len(df)}</b><br>
    Number of variables: <b>{df.shape[1]}</b><br>
    Numeric variables: <b>{len(NUM_COLS)}</b><br>
    Categorical variables: <b>{len(CAT_COLS)}</b><br>
    Missing values: <b>{df.isnull().sum().sum()}</b>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="info-box">
    Highest score: <b>{df['Exam_Score'].max()}</b><br>
    Lowest score: <b>{df['Exam_Score'].min()}</b><br>
    Avg tutoring sessions: <b>{df['Tutoring_Sessions'].mean():.2f}</b><br>
    Avg sleep hours: <b>{df['Sleep_Hours'].mean():.2f}</b><br>
    Avg previous score: <b>{df['Previous_Scores'].mean():.1f}</b>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — GRAPHICAL ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "Graphical Analysis":

    st.markdown("""
    <div class="page-title">
      <h2>Graphical Analysis</h2>
      <p>Visual representation of student performance data</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Qualitative Graphs - Pie Charts</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Gender Distribution**")
        st.pyplot(plot_pie_gender(df))
    with c2:
        st.markdown("**School Type**")
        st.pyplot(plot_pie_school(df))
    with c3:
        st.markdown("**Above / Below Average Score**")
        st.pyplot(plot_pie_pass_fail(df))

    st.markdown('<div class="sec-head">Qualitative Graphs - Bar Charts</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_bar_parental(df))
    with c2:
        st.pyplot(plot_bar_motivation(df))

    cat_sel = st.selectbox("Average Exam Score by:", CAT_COLS, index=0)
    st.pyplot(plot_bar_avg_by_cat(df, cat_sel))
    st.markdown(f"""
    <div class="info-box">
    Bar chart shows the average exam score across categories of <b>{cat_sel}</b>.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sec-head">Quantitative Graphs - Histogram & Distribution Shape</div>',
                unsafe_allow_html=True)
    hist_col = st.selectbox("Select variable for histogram:", NUM_COLS)
    st.pyplot(plot_histogram(df, hist_col))
    st.markdown("""
    <div class="info-box">
    Histogram shows frequency distribution. Red dashed = mean, green dotted = median.
    </div>""", unsafe_allow_html=True)

    st.markdown("**Shape of Distribution - Skewness Analysis**")
    st.pyplot(plot_shape_of_distribution(df))
    st.markdown("""
    <div class="info-box">
    <b>Normal:</b> Symmetric, skewness near 0 |
    <b>Right Skewed:</b> Tail on right, skewness > 0 |
    <b>Left Skewed:</b> Tail on left, skewness < 0
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sec-head">Scatter Plot</div>', unsafe_allow_html=True)
    x_scatter = st.selectbox("X-axis variable:", FEATURES)
    st.pyplot(plot_scatter(df, x_scatter))

    st.markdown("---")

    st.markdown('<div class="sec-head">Correlation Heatmap</div>', unsafe_allow_html=True)
    st.pyplot(plot_heatmap(df))

    st.markdown("---")

    st.markdown('<div class="sec-head">Box Plots</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**All Numeric Variables**")
        st.pyplot(plot_boxplot(df))
    with c2:
        box_cat = st.selectbox("Exam Score grouped by:", CAT_COLS, key="boxcat")
        st.pyplot(plot_boxplot_score_by(df, box_cat))

    st.markdown("""
    <div class="info-box">
    Box plots show the five-number summary: Min, Q1, Median, Q3, Max.
    Dots outside whiskers are outliers.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sec-head">Frequency Table - Exam Score</div>',
                unsafe_allow_html=True)
    st.dataframe(frequency_table(df), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════
elif page == "Descriptive Statistics":

    st.markdown("""
    <div class="page-title">
      <h2>Descriptive Statistics</h2>
      <p>Central tendency, dispersion, position, variability, and confidence intervals</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">1. Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(descriptive_stats(df), use_container_width=True)
    st.markdown(f"""
    <div class="info-box">
    <b>Weighted Mean</b> (Exam Score weighted by Attendance) = <b>{weighted_mean(df):.4f}</b><br>
    Measures: Mean, Median, Mode, Mid-Range, Std Dev, Variance, Range, Q1, Q3, IQR,
    Skewness, Kurtosis, CV%, Coefficient of Quartile Deviation.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sec-head">2. Measures of Dispersion</div>', unsafe_allow_html=True)
    st.dataframe(dispersion_table(df), use_container_width=True)
    st.markdown("""
    <div class="info-box">
    <b>Range</b> = Max - Min |
    <b>Mean Abs Deviation</b> = Avg absolute deviation from mean |
    <b>IQR</b> = Q3 - Q1 |
    <b>CV%</b> = std/mean x 100 |
    <b>Coeff QD</b> = (Q3-Q1)/(Q3+Q1)
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sec-head">3. Measures of Position</div>', unsafe_allow_html=True)
    pos_col = st.selectbox("Select variable:", NUM_COLS, index=NUM_COLS.index("Exam_Score"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Quartiles**")
        st.dataframe(quartile_table(df, pos_col), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Deciles**")
        st.dataframe(decile_table(df, pos_col), hide_index=True, use_container_width=True)
    with c3:
        st.markdown("**Percentiles**")
        st.dataframe(percentile_table(df, pos_col), hide_index=True, use_container_width=True)

    pct_val = st.slider("Find percentile rank of a score:",
                         int(df[pos_col].min()), int(df[pos_col].max()),
                         int(df[pos_col].mean()))
    rank = round(float(np.mean(df[pos_col] <= pct_val)) * 100, 2)
    st.markdown(f"""
    <div class="info-box">
    A score of <b>{pct_val}</b> in <b>{pos_col}</b> is at the
    <b>{rank:.1f}th percentile</b> — {rank:.1f}% of students scored at or below this.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sec-head">4. Confidence Intervals</div>', unsafe_allow_html=True)
    conf = st.select_slider("Confidence level", [0.90, 0.95, 0.99], value=0.95,
                             format_func=lambda x: f"{int(x*100)}%")
    st.dataframe(all_confidence_intervals(df, conf), use_container_width=True)

    ci = confidence_interval(df["Exam_Score"], conf)
    c1, c2, c3 = st.columns(3)
    for col, val, lbl in zip(
        [c1, c2, c3],
        [f"{ci['mean']:.2f}",
         f"[{ci['t_lo']:.2f}, {ci['t_hi']:.2f}]",
         f"[{ci['z_lo']:.2f}, {ci['z_hi']:.2f}]"],
        ["Sample Mean", f"t-Interval ({int(conf*100)}%)", f"z-Interval ({int(conf*100)}%)"],
    ):
        col.markdown(
            f'<div class="stat-card"><div class="val" style="font-size:1.2rem">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    We are {int(conf*100)}% confident the true mean of Exam_Score lies between
    <b>{ci['t_lo']:.2f}</b> and <b>{ci['t_hi']:.2f}</b>
    (n={ci['n']}, SE={ci['se']}, t-crit={ci['t_crit']}, z-crit={ci['z_crit']}).
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — PROBABILITY
# ══════════════════════════════════════════════════════════════
elif page == "Probability":

    st.markdown("""
    <div class="page-title">
      <h2>Probability</h2>
      <p>Classical, Empirical, Conditional, Bayes Theorem, Permutation and Combination</p>
    </div>""", unsafe_allow_html=True)

    # 1. Classical
    st.markdown('<div class="sec-head">1. Classical Probability</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Formula:</b> P(E) = Favorable Outcomes / Total Outcomes<br>
    Assumption: all outcomes are equally likely.
    </div>""", unsafe_allow_html=True)
    st.pyplot(plot_classical_prob(df))
    st.dataframe(classical_probability(df), use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Empirical
    st.markdown('<div class="sec-head">2. Empirical / Objective Probability</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Formula:</b> P(E) = Frequency of event / Total observations<br>
    Based on actual data — no assumption of equally likely outcomes.
    </div>""", unsafe_allow_html=True)
    st.dataframe(empirical_probability(df), use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="info-box">
    <b>Subjective Probability</b> is based on personal judgment or expert opinion,
    not computed from data. Example: a teacher estimating a students chances of passing
    based on observation. This project uses Classical and Empirical probability computationally.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 3. Permutation & Combination
    st.markdown('<div class="sec-head">3. Permutation and Combination</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Permutation nPr</b> = n! / (n-r)! — ordered selection<br>
    <b>Combination nCr</b> = n! / (r! x (n-r)!) — unordered selection
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    n_in = c1.number_input("n (total)", min_value=2, max_value=50, value=10)
    r_in = c2.number_input("r (choose)", min_value=1, max_value=int(n_in), value=3)

    npr = permutation(int(n_in), int(r_in))
    ncr = combination(int(n_in), int(r_in))

    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="stat-card"><div class="val">{npr:,}</div>'
                f'<div class="lbl">nPr  ({int(n_in)}P{int(r_in)})</div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><div class="val">{ncr:,}</div>'
                f'<div class="lbl">nCr  ({int(n_in)}C{int(r_in)})</div></div>',
                unsafe_allow_html=True)

    # ── fixed: use correct key names ──
    pc = perm_comb_examples(df)
    ds_n = pc["n (dataset size)"]
    ds_r = pc["r (choose)"]
    ds_npr = pc["nPr"]
    ds_ncr = pc["nCr"]
    st.markdown(f"""
    <div class="info-box">
    Dataset size n = {ds_n}. Choosing r = {ds_r} students:<br>
    <b>Ordered (Permutation):</b> {ds_npr:,} ways |
    <b>Unordered (Combination):</b> {ds_ncr:,} ways
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 4. Conditional Probability
    st.markdown('<div class="sec-head">4. Conditional Probability</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Formula:</b> P(A|B) = P(A and B) / P(B)<br>
    We compute P(High Score | High Study Hours) and related probabilities.
    </div>""", unsafe_allow_html=True)

    cp = conditional_probability(df)
    c1, c2, c3 = st.columns(3)
    metrics = [
        ("P(High Score)", cp["P(High_Score)"]),
        ("P(High Study)", cp["P(High_Study)"]),
        ("P(High Score | High Study)", cp["P(High_Score | High_Study)"]),
    ]
    for col, (lbl, val) in zip([c1, c2, c3], metrics):
        col.markdown(f'<div class="stat-card"><div class="val">{val:.4f}</div>'
                     f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    P(High Score AND High Study) = <b>{cp["P(HighScore \u2229 HighStudy)"]:.4f}</b><br>
    P(High Score | High Study) = <b>{cp["P(High_Score | High_Study)"]:.4f}</b><br>
    A student who studies heavily has a {cp["P(High_Score | High_Study)"]*100:.1f}%
    chance of scoring above the 75th percentile.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 5. Bayes Theorem
    st.markdown('<div class="sec-head">5. Bayes Theorem</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Formula:</b> P(B|A) = P(A|B) x P(B) / P(A)<br>
    We update prior probabilities based on evidence from the data.
    </div>""", unsafe_allow_html=True)

    bayes_cat = st.selectbox("Select categorical variable for Bayes analysis:",
                              ["Internet_Access", "Parental_Involvement",
                               "Motivation_Level", "School_Type",
                               "Family_Income", "Teacher_Quality"])
    threshold = df["Exam_Score"].mean()
    bt = bayes_theorem(df, bayes_cat, threshold)
    st.dataframe(bt, use_container_width=True, hide_index=True)
    st.markdown(f"""
    <div class="info-box">
    High Score threshold = mean ({threshold:.1f}).
    The posterior column shows P(Category | High Score) via Bayes theorem.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 6. Normal Probability Calculator
    st.markdown('<div class="sec-head">6. Normal Probability Calculator</div>',
                unsafe_allow_html=True)
    mu, sigma = get_normal_params(df)
    ptype = st.selectbox("Probability type",
                          ["P(X <= x)", "P(X > x)", "P(a <= X <= b)"])
    if ptype == "P(a <= X <= b)":
        c1, c2 = st.columns(2)
        a_val = c1.slider("Lower bound (a)", 55, 99, 60)
        b_val = c2.slider("Upper bound (b)", 56, 100, 75)
        prob  = calc_probability_between(df, a_val, b_val)
        label = f"P({a_val} <= Score <= {b_val})"
    else:
        x_val = st.slider("Score value (x)", 55, 100, 67)
        prob  = calc_probability(df, x_val, ptype)
        label = ptype.replace("x", str(x_val))

    st.markdown(
        f'<div class="stat-card" style="max-width:300px;">'
        f'<div class="val">{prob:.4f}</div>'
        f'<div class="lbl">{label} = {prob*100:.2f}%</div></div>',
        unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    Using Normal distribution N(mu={mu}, sigma={sigma}).
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════
elif page == "Distributions":

    st.markdown("""
    <div class="page-title">
      <h2>Probability Distributions</h2>
      <p>Normal, Binomial, Poisson, Hypergeometric, Uniform — with normality testing</p>
    </div>""", unsafe_allow_html=True)

    dist_tab = st.tabs([
        "Normal", "Binomial", "Poisson", "Hypergeometric", "Uniform", "Hypothesis Test"
    ])

    # Normal
    with dist_tab[0]:
        st.markdown('<div class="sec-head">Normal Distribution</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_normal_fit(df))
        with c2:
            st.pyplot(plot_qq(df))
        mu, sigma = get_normal_params(df)
        nt = normality_tests(df)
        st.markdown(f"""
        <div class="info-box">
        Fitted Normal: N(mu={mu}, sigma={sigma})<br><br>
        <b>Kolmogorov-Smirnov:</b> D={nt['ks_stat']}, p={nt['ks_p']}
        -- {"Follows Normal" if nt['normal_ks'] else "Slight deviation"}<br>
        <b>Shapiro-Wilk:</b> W={nt['sw_stat']}, p={nt['sw_p']}
        -- {"Follows Normal" if nt['normal_sw'] else "Slight deviation"}<br>
        Q-Q plot: points close to the line = normality.
        </div>""", unsafe_allow_html=True)

    # Binomial
    with dist_tab[1]:
        st.markdown('<div class="sec-head">Binomial Distribution</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Model:</b> Out of n students, how many score above average?<br>
        <b>Formula:</b> P(X=k) = C(n,k) x p^k x (1-p)^(n-k)
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        n_val     = c1.slider("Number of students (n)", 5, 100, 30)
        threshold = df["Exam_Score"].mean()
        p_binom   = float((df["Exam_Score"] >= threshold).mean())
        k_val     = c2.slider("Number above average (k)", 0, n_val, int(n_val * p_binom))
        st.pyplot(plot_binomial(df, n_val, threshold))
        bres = binomial_stats(df, n_val, k_val, threshold)
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip(
            [c1, c2, c3, c4],
            [f"{bres['p']:.4f}", f"{bres['pmf']:.4f}",
             f"{bres['cdf']:.4f}", f"{bres['expected']:.2f}"],
            ["Pass Prob (p)", f"P(X={k_val})", f"P(X<={k_val})", "Expected"],
        ):
            col.markdown(f'<div class="stat-card"><div class="val">{val}</div>'
                         f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        Variance = {bres['variance']} | Std Dev = {bres['std']}
        </div>""", unsafe_allow_html=True)

    # Poisson
    with dist_tab[2]:
        st.markdown('<div class="sec-head">Poisson Distribution</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Model:</b> Number of tutoring sessions per student.<br>
        <b>Formula:</b> P(X=k) = (lambda^k x e^-lambda) / k!
        </div>""", unsafe_allow_html=True)
        k_pois = st.slider("Find probability for k sessions:", 0, 10, 2)
        st.pyplot(plot_poisson(df))
        pres = poisson_stats(df, k_pois)
        c1, c2, c3 = st.columns(3)
        for col, key, lbl in zip(
            [c1, c2, c3],
            ["PMF P(X=k)", "CDF P(X<=k)", "P(X>k)"],
            [f"P(X={k_pois})", f"P(X<={k_pois})", f"P(X>{k_pois})"],
        ):
            col.markdown(f'<div class="stat-card"><div class="val">{pres[key]:.5f}</div>'
                         f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        lambda (mean sessions) = <b>{pres['lambda (mean sessions)']:.4f}</b>
        </div>""", unsafe_allow_html=True)

    # Hypergeometric
    with dist_tab[3]:
        st.markdown('<div class="sec-head">Hypergeometric Distribution</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Model:</b> From N students (K have Internet Access), draw n without replacement.<br>
        <b>Formula:</b> P(X=k) = C(K,k) x C(N-K, n-k) / C(N,n)
        </div>""", unsafe_allow_html=True)
        N_total = len(df)
        K_inet  = int((df["Internet_Access"] == "Yes").sum())
        c1, c2  = st.columns(2)
        n_draw  = c1.slider("Sample drawn (n)", 10, 100, 50)
        k_succ  = c2.slider("Expected successes (k)", 0, n_draw,
                             int(n_draw * K_inet / N_total))
        st.pyplot(plot_hypergeometric(df, n_draw))
        hres = hypergeometric_stats(df, n_draw, k_succ)
        c1, c2, c3 = st.columns(3)
        for col, key, lbl in zip(
            [c1, c2, c3],
            ["PMF P(X=k)", "CDF P(X<=k)", "Expected value"],
            [f"P(X={k_succ})", f"P(X<={k_succ})", "Expected"],
        ):
            col.markdown(f'<div class="stat-card"><div class="val">{hres[key]}</div>'
                         f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        N = {N_total} | K (with internet) = {K_inet} | n drawn = {n_draw}
        </div>""", unsafe_allow_html=True)

    # Uniform
    with dist_tab[4]:
        st.markdown('<div class="sec-head">Uniform Distribution</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Model:</b> If exam scores were uniformly distributed over [min, max].<br>
        <b>PDF:</b> f(x) = 1/(b-a) | <b>Mean:</b> (a+b)/2 | <b>Variance:</b> (b-a)^2/12
        </div>""", unsafe_allow_html=True)
        st.pyplot(plot_uniform(df))
        ures = uniform_stats(df)
        c1, c2, c3 = st.columns(3)
        for col, key, lbl in zip(
            [c1, c2, c3], ["Mean", "Variance", "Std Dev"], ["Mean", "Variance", "Std Dev"],
        ):
            col.markdown(f'<div class="stat-card"><div class="val">{ures[key]}</div>'
                         f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown("**Probability between two values:**")
        c1, c2 = st.columns(2)
        a_u = float(ures["a (min)"])
        b_u = float(ures["b (max)"])
        u1  = c1.slider("Lower bound", int(a_u), int(b_u) - 1, int(a_u) + 5)
        u2  = c2.slider("Upper bound", int(u1) + 1, int(b_u), int(b_u) - 5)
        up  = uniform_prob(u1, u2, a_u, b_u)
        st.markdown(f"""
        <div class="info-box">
        P({u1} &lt;= X &lt;= {u2}) under U({a_u:.0f}, {b_u:.0f}) = <b>{up:.4f} ({up*100:.2f}%)</b>
        </div>""", unsafe_allow_html=True)

    # Hypothesis Test
    with dist_tab[5]:
        st.markdown('<div class="sec-head">One-Sample t-Test</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>H0:</b> mu = mu0 | <b>H1:</b> mu != mu0 (two-tailed test)
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        mu0   = c1.slider("Null hypothesis mean (mu0)", 55, 85, 67)
        alpha = c2.selectbox("Significance level", [0.01, 0.05, 0.10], index=1)
        ht    = hypothesis_test(df, mu0=mu0, alpha=alpha)
        c1, c2, c3 = st.columns(3)
        for col, val, lbl in zip(
            [c1, c2, c3],
            [f"{ht['t_stat']}", f"{ht['p_value']}", ht["decision"].split("(")[0].strip()],
            ["t-Statistic", "p-Value", "Decision"],
        ):
            col.markdown(
                f'<div class="stat-card"><div class="val" style="font-size:1.1rem">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
        box_cls = "result-pass" if ht["reject"] else "result-fail"
        st.markdown(f"""
        <div class="{box_cls}">
        <b>Result:</b> {ht['decision']}<br>
        At alpha={alpha}, p-value={ht['p_value']}.
        {"Since p &lt; alpha, we reject H0." if ht['reject'] else "Since p >= alpha, we fail to reject H0."}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 6 — REGRESSION & PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "Regression & Prediction":

    st.markdown("""
    <div class="page-title">
      <h2>Regression and Prediction</h2>
      <p>Covariance, Correlation, Simple and Multiple Linear Regression, Prediction</p>
    </div>""", unsafe_allow_html=True)

    reg_tab = st.tabs(["Covariance & Correlation", "Simple Regression",
                        "Multiple Regression", "Predict Score"])

    # Covariance & Correlation
    with reg_tab[0]:
        st.markdown('<div class="sec-head">Covariance and Correlation</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Covariance</b> = direction of linear relationship.<br>
        <b>Pearson r</b> = direction + strength (-1 to +1).
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Covariance Matrix**")
            st.pyplot(plot_covariance_heatmap(df))
        with c2:
            st.markdown("**Correlation Matrix**")
            st.pyplot(plot_correlation_heatmap(df))
        st.markdown('<div class="sec-head">Pairwise Summary</div>', unsafe_allow_html=True)
        st.dataframe(pairwise_covariance_correlation(df),
                     use_container_width=True, hide_index=True)

    # Simple Regression
    with reg_tab[1]:
        st.markdown('<div class="sec-head">Simple Linear Regression</div>',
                    unsafe_allow_html=True)
        x_var = st.selectbox("Independent variable (X):", FEATURES)
        slr   = simple_regression(df, x_var)
        st.pyplot(plot_slr(df, x_var))
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip(
            [c1, c2, c3, c4],
            [slr["slope"], slr["intercept"], slr["r2"], slr["pearson_r"]],
            ["Slope (b1)", "Intercept (b0)", "R-squared", "Pearson r"],
        ):
            col.markdown(f'<div class="stat-card"><div class="val">{val}</div>'
                         f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        <b>Equation:</b> {slr['equation']}<br>
        MSE = {slr['mse']} | RMSE = {slr['rmse']}<br>
        R^2 = {slr['r2']} means the model explains {slr['r2']*100:.1f}% of variance using {x_var}.
        </div>""", unsafe_allow_html=True)

    # Multiple Regression
    with reg_tab[2]:
        st.markdown('<div class="sec-head">Multiple Linear Regression</div>',
                    unsafe_allow_html=True)
        sel_feat = st.multiselect("Select predictor variables:", FEATURES, default=FEATURES)
        if len(sel_feat) >= 2:
            mlr = multiple_regression(df, sel_feat)
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_actual_vs_pred(mlr))
            with c2:
                st.pyplot(plot_residuals(mlr))
            st.pyplot(plot_coefficients(mlr))
            c1, c2, c3 = st.columns(3)
            for col, val, lbl in zip(
                [c1, c2, c3], [mlr["r2"], mlr["mse"], mlr["rmse"]],
                ["R-squared", "MSE", "RMSE"],
            ):
                col.markdown(f'<div class="stat-card"><div class="val">{val}</div>'
                             f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
            Intercept = {mlr['intercept']} |
            R^2 = {mlr['r2']} ({mlr['r2']*100:.1f}% variance explained) |
            RMSE = {mlr['rmse']}
            </div>""", unsafe_allow_html=True)
            st.write("**Coefficient Table:**")
            st.dataframe(mlr["coef_df"], use_container_width=True, hide_index=True)
        else:
            st.warning("Please select at least 2 variables.")

    # Predict Score
    with reg_tab[3]:
        st.markdown('<div class="sec-head">Predict Student Exam Score</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Enter student details to predict exam score using the trained MLR model.
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            hours_in  = st.number_input("Hours Studied (per week)", 1, 44, 20)
            attend_in = st.number_input("Attendance (%)", 60, 100, 80)
            sleep_in  = st.number_input("Sleep Hours", 4, 10, 7)
        with c2:
            prev_in   = st.number_input("Previous Score", 40, 100, 70)
            tutor_in  = st.number_input("Tutoring Sessions", 0, 8, 1)
            phys_in   = st.number_input("Physical Activity (hrs/week)", 0, 6, 3)

        if st.button("Predict Exam Score", use_container_width=True):
            pred = predict_marks(hours_in, attend_in, sleep_in, prev_in, tutor_in, phys_in)
            prob = passing_probability(pred, df)
            avg  = df["Exam_Score"].mean()

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="stat-card"><div class="val">{pred}/100</div>'
                        f'<div class="lbl">Predicted Score</div></div>',
                        unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-card"><div class="val">{prob}%</div>'
                        f'<div class="lbl">Above-Avg Probability</div></div>',
                        unsafe_allow_html=True)
            status = "ABOVE AVG" if pred >= avg else "BELOW AVG"
            c3.markdown(f'<div class="stat-card"><div class="val">{status}</div>'
                        f'<div class="lbl">Result (Avg={avg:.1f})</div></div>',
                        unsafe_allow_html=True)

            box = "result-pass" if pred >= avg else "result-fail"
            st.markdown(f"""
            <div class="{box}">
            <b>Prediction Summary:</b><br>
            Hours={hours_in} | Attendance={attend_in}% | Sleep={sleep_in}hrs |
            Prev Score={prev_in} | Tutoring={tutor_in} | Physical={phys_in}<br><br>
            Predicted Score = <b>{pred}/100</b> -- <b>{status}</b> |
            Above-average probability = <b>{prob}%</b>
            </div>""", unsafe_allow_html=True)
