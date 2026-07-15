import os, json, joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

BASE = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE, "models", "cpis_model.pkl")
DATA_PATH = os.path.join(BASE, "data", "cpis_clean.csv")
FIG_DIR = os.path.join(BASE, "reports", "figures")

NICE_NAMES = {
    "sleep_hours": "Sleep duration", "screen_time_hours": "Screen time",
    "physical_activity_minutes": "Physical activity", "stress_level": "Stress level",
    "diet_quality_score": "Diet quality", "caffeine_intake_mg": "Caffeine intake",
    "sleep_quality_score": "Sleep quality", "age": "Age",
}

NUMERIC_COLS = list(NICE_NAMES.keys())
CATEGORICAL_COLS = ["occupation_type"]

def nice_name(raw_name):
    for prefix in ("scale__", "onehot__"):
        if raw_name.startswith(prefix):
            raw_name = raw_name[len(prefix):]
    for key, label in NICE_NAMES.items():
        if raw_name.startswith(key):
            return label
    if raw_name.startswith("occupation_type_"):
        return f"Occupation ({raw_name.replace('occupation_type_', '')})"
    return raw_name.replace("_", " ").title()


def explain_one_prediction(model, row):
    preprocessing = model.named_steps["preprocessing"]
    rf = model.named_steps["model"]
    feature_names = [nice_name(n) for n in preprocessing.get_feature_names_out()]
    X_t = preprocessing.transform(row)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_t)[0]
    total = np.abs(shap_values).sum()
    order = np.argsort(-np.abs(shap_values))
    explanations = []
    for i in order[:5]:
        pct = abs(shap_values[i]) / total * 100 if total > 0 else 0
        direction = "increases" if shap_values[i] > 0 else "reduces"
        sign = "+" if shap_values[i] > 0 else "-"
        explanations.append(f"{feature_names[i]} contributes {sign}{pct:.0f}% to the score ({direction} it)")
    return explanations


PROJECT_START = datetime(2026, 5, 18)
PROJECT_PHASES = [
    {"phase": "Data sourcing", "script": "get_data.py", "start_day": 0, "duration": 2,
     "status": "Done", "note": "Real Kaggle sleep/lifestyle dataset (374 people)"},
    {"phase": "Feature engineering", "script": "build_features.py", "start_day": 2, "duration": 3,
     "status": "Done", "note": "Mapped real columns to schema, simulated 3 missing features, built target formula"},
    {"phase": "Validation & bias audit", "script": "clean_and_validate.py", "start_day": 5, "duration": 2,
     "status": "Done", "note": "Pydantic schema enforcement, cleaning, occupation/age balance checks"},
    {"phase": "Model training", "script": "train_model.py", "start_day": 7, "duration": 2,
     "status": "Done", "note": "Linear/RF/GB comparison, 5-fold CV, picked Random Forest"},
    {"phase": "Statistical analysis", "script": "stats_and_charts.py", "start_day": 9, "duration": 2,
     "status": "Done", "note": "Correlation matrix, distributions, hypothesis tests"},
    {"phase": "Explainability (SHAP)", "script": "explainability.py", "start_day": 11, "duration": 2,
     "status": "Done", "note": "Feature importance + SHAP + plain-english explanations"},
    {"phase": "API + dashboard", "script": "api/main.py, app/dashboard.py", "start_day": 13, "duration": 4,
     "status": "In progress", "note": "FastAPI endpoints + Streamlit UI, 5-tab design"},
]
STATUS_COLORS = {"Done": "#5C7A52", "In progress": "#C97D44", "Blocked": "#B85C5C", "Planned": "#B7A98C"}

PLOTLY_LAYOUT = dict(
    plot_bgcolor="#FAF6EF", paper_bgcolor="#FAF6EF",
    font=dict(family="IBM Plex Mono, monospace", size=12, color="#1B2430"),
    margin=dict(l=10, r=10, t=30, b=10),
)
CHART_COLORS = ["#C97D44", "#5C7A52", "#7A6F5E", "#B85C5C", "#8C9BB0"]


def build_gantt(phases):
    rows = []
    for p in phases:
        start = PROJECT_START + timedelta(days=p["start_day"])
        end = start + timedelta(days=p["duration"])
        rows.append({"Phase": p["phase"], "Start": start, "Finish": end,
                     "Status": p["status"], "Script": p["script"], "Note": p["note"]})
    gdf = pd.DataFrame(rows)
    fig = px.timeline(gdf, x_start="Start", x_end="Finish", y="Phase",
                      color="Status", color_discrete_map=STATUS_COLORS,
                      custom_data=["Script", "Note"])
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=None, gridcolor="#E5DCC8")
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        marker_line_width=0)
    fig.update_layout(**{**PLOTLY_LAYOUT, "height": 320, "margin": dict(l=10, r=10, t=10, b=10)},
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    return fig


def occupation_pie(df):
    counts = df["occupation_type"].value_counts()
    fig = px.pie(names=counts.index, values=counts.values, hole=0.45,
                 color_discrete_sequence=CHART_COLORS)
    fig.update_traces(textfont=dict(family="IBM Plex Mono, monospace"))
    fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.15))
    return fig


def sleep_quality_bar(df):
    bins = [0, 4, 7, 10]
    labels = ["Poor (1-4)", "Okay (5-7)", "Good (8-10)"]
    bands = pd.cut(df["sleep_quality_score"], bins=bins, labels=labels, include_lowest=True)
    counts = bands.value_counts().reindex(labels)
    fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                 color_discrete_sequence=["#B85C5C", "#C97D44", "#5C7A52"])
    fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False,
                      xaxis_title=None, yaxis_title="people")
    fig.update_xaxes(gridcolor="#E5DCC8")
    fig.update_yaxes(gridcolor="#E5DCC8")
    return fig


def histogram_with_marker(df, column, label, user_value=None):
    fig = px.histogram(df, x=column, nbins=25, color_discrete_sequence=["#C97D44"])
    fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False,
                      xaxis_title=label, yaxis_title="people", bargap=0.05)
    fig.update_xaxes(gridcolor="#E5DCC8")
    fig.update_yaxes(gridcolor="#E5DCC8")
    avg = df[column].mean()
    fig.add_vline(x=avg, line_dash="dot", line_color="#7A6F5E",
                  annotation_text="avg",
                  annotation_font=dict(family="IBM Plex Mono, monospace", size=10))
    if user_value is not None:
        fig.add_vline(x=user_value, line_color="#1B2430", line_width=2,
                      annotation_text="you", annotation_position="top right",
                      annotation_font=dict(family="IBM Plex Mono, monospace", size=11, color="#1B2430"))
    return fig


st.set_page_config(page_title="Sleep & Cognition Field Notes", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { max-width: 1100px; padding-top: 2rem; }
h1, h2, h3 { font-family: 'Source Serif 4', serif !important; letter-spacing: -0.01em; }
h1 { font-weight: 700 !important; border-bottom: 3px solid #1B2430; padding-bottom: 0.4rem; margin-bottom: 0.2rem !important; }

.field-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: #7A6F5E;
    margin-bottom: 0.4rem; margin-top: 0.8rem;
}
div[data-testid="stSlider"] { padding-top: 0.6rem; }

.lab-card {
    background: #F0E9DC; border: 1px solid #DCD0B8;
    border-left: 5px solid #C97D44; padding: 1.4rem 1.6rem;
    border-radius: 2px; margin-top: 0.5rem;
}
.lab-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.2rem; font-weight: 500; color: #1B2430; line-height: 1;
}
.lab-score-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: #7A6F5E;
}
.band-good { color: #5C7A52; font-weight: 600; }
.band-mid  { color: #B07A2E; font-weight: 600; }
.band-low  { color: #B85C5C; font-weight: 600; }

.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }
.stButton > button {
    border-radius: 2px; border: none;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.04em; text-transform: uppercase; font-size: 0.8rem;
}
hr { border-color: #DCD0B8 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None


model = load_model()
df = load_data()

st.markdown("""
    <h1 style="margin-bottom:0;">Sleep &amp; Cognition Field Notes</h1>
    <p style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#7A6F5E; letter-spacing:0.03em;">
    A LIFESTYLE-BASED ESTIMATE, NOT A DIAGNOSIS — BUILT FROM REAL SLEEP &amp; LIFESTYLE DATA (KAGGLE, 374 PEOPLE)
    </p>
""", unsafe_allow_html=True)

if model is None:
    st.error("No trained model found. Run scripts 01–06 first.")
    st.stop()

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "▸ THE POPULATION", "▸ NEW ENTRY", "▸ MODEL LOGIC", "▸ SAMPLE & BIAS", "▸ PROJECT PLAN"
])


with tab0:
    st.markdown("##### What everyone else looks like")
    st.markdown(
        f'<p class="field-label">{len(df) if df is not None else 0} people in this sample ',
        unsafe_allow_html=True)

    if df is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="field-label">Occupation mix</p>', unsafe_allow_html=True)
            st.plotly_chart(occupation_pie(df), use_container_width=True)
        with c2:
            st.markdown('<p class="field-label">Sleep quality bands</p>', unsafe_allow_html=True)
            st.plotly_chart(sleep_quality_bar(df), use_container_width=True)

        st.markdown('<p class="field-label">Sleep duration across the sample</p>', unsafe_allow_html=True)
        st.plotly_chart(histogram_with_marker(df, "sleep_hours", "sleep hours"), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<p class="field-label">Stress level across the sample</p>', unsafe_allow_html=True)
            st.plotly_chart(histogram_with_marker(df, "stress_level", "stress (1-10)"), use_container_width=True)
        with c4:
            st.markdown('<p class="field-label">Cognitive score across the sample</p>', unsafe_allow_html=True)
            st.plotly_chart(histogram_with_marker(df, "cognitive_performance_score", "score"), use_container_width=True)

        avg_score = df["cognitive_performance_score"].mean()
        st.markdown(
            f'<p style="font-size:0.9rem; color:#4A4438; margin-top:0.5rem;">'
            f'Average score in this sample: <b>{avg_score:.1f}</b>. '
            f'Go to <b>New Entry</b> to see where your lifestyle would land.</p>',
            unsafe_allow_html=True)


with tab1:
    st.markdown("##### Today's intake")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<p class="field-label">Sleep duration (hrs)</p>', unsafe_allow_html=True)
        sleep_hours = st.slider("sleep_hours", 3.0, 12.0, 7.5, 0.1, label_visibility="collapsed")
        st.markdown('<p class="field-label">Screen time (hrs/day)</p>', unsafe_allow_html=True)
        screen_time = st.slider("screen_time", 0.0, 16.0, 6.0, 0.5, label_visibility="collapsed")
        st.markdown('<p class="field-label">Physical activity (min/day)</p>', unsafe_allow_html=True)
        activity = st.slider("activity", 0, 180, 45, 5, label_visibility="collapsed")

    with c2:
        st.markdown('<p class="field-label">Stress level (1–10)</p>', unsafe_allow_html=True)
        stress = st.slider("stress", 1, 10, 5, label_visibility="collapsed")
        st.markdown('<p class="field-label">Diet quality (1–10)</p>', unsafe_allow_html=True)
        diet = st.slider("diet", 1.0, 10.0, 6.0, 0.5, label_visibility="collapsed")
        st.markdown('<p class="field-label">Caffeine (mg/day)</p>', unsafe_allow_html=True)
        caffeine = st.slider("caffeine", 0, 600, 150, 10, label_visibility="collapsed")

    with c3:
        st.markdown('<p class="field-label">Sleep quality (1–10)</p>', unsafe_allow_html=True)
        sleep_quality = st.slider("sleep_quality", 1.0, 10.0, 7.0, 0.5, label_visibility="collapsed")
        st.markdown('<p class="field-label">Age</p>', unsafe_allow_html=True)
        age = st.slider("age", 18, 60, 30, label_visibility="collapsed")
        st.markdown('<p class="field-label">Occupation</p>', unsafe_allow_html=True)
        occupation = st.selectbox("occupation", ["student", "worker", "athlete", "unemployed"],
                                   label_visibility="collapsed")

    run = st.button("Log entry & predict", type="primary")

    if run:
        row = pd.DataFrame([{
            "sleep_hours": sleep_hours, "screen_time_hours": screen_time,
            "physical_activity_minutes": activity, "stress_level": stress,
            "diet_quality_score": diet, "caffeine_intake_mg": caffeine,
            "sleep_quality_score": sleep_quality, "age": age, "occupation_type": occupation,
        }])
        score = float(np.clip(model.predict(row)[0], 0, 100))
        explanations = explain_one_prediction(model, row)

        if score >= 70:
            band_label, band_class = "STRONG RANGE", "band-good"
        elif score >= 45:
            band_label, band_class = "MODERATE RANGE", "band-mid"
        else:
            band_label, band_class = "LOWER RANGE", "band-low"

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1, 2])
        with left:
            st.markdown(f"""
                <div class="lab-card">
                    <div class="lab-score-label">Estimated score</div>
                    <div class="lab-score">{score:.1f}</div>
                    <div class="{band_class}" style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; margin-top:0.3rem;">
                        {band_label}
                    </div>
                </div>""", unsafe_allow_html=True)
        with right:
            st.markdown('<p class="field-label">What drove this estimate</p>', unsafe_allow_html=True)
            for line in explanations:
                st.markdown(f"— {line}")

        if df is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="field-label">Where you land vs. the sample</p>', unsafe_allow_html=True)
            st.plotly_chart(
                histogram_with_marker(df, "cognitive_performance_score", "score", user_value=score),
                use_container_width=True)


with tab2:
    st.markdown("##### What the model actually pays attention to")
    c1, c2 = st.columns(2)
    fi_path = os.path.join(FIG_DIR, "feature_importance.png")
    shap_path = os.path.join(FIG_DIR, "shap_summary.png")

    if os.path.exists(fi_path):
        c1.image(fi_path, use_container_width=True)
    else:
        c1.warning("Feature importance chart not found. Run `python 06_explainability.py` to generate it.")

    if os.path.exists(shap_path):
        c2.image(shap_path, use_container_width=True)
    else:
        c2.warning("SHAP chart not found. Run `python 06_explainability.py` to generate it.")

    results_path = os.path.join(BASE, "models", "model_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        st.markdown('<p class="field-label">Model comparison</p>', unsafe_allow_html=True)
        comp = pd.DataFrame(results["all_results"]).T
        comp.index.name = "Model"
        comp.columns = ["CV R²", "CV std", "Test MAE", "Test RMSE", "Test R²", "Train time (s)"]
        st.dataframe(comp, use_container_width=True)
        trained_on = results.get("trained_on", "unknown")
        st.markdown(
            f'<p style="font-size:0.85rem; color:#7A6F5E;">Trained on: <b>{trained_on.replace("_", " ")}</b> — '
            f'deployed model: <b>{results["chosen_model"].replace("_", " ").title()}</b></p>',
            unsafe_allow_html=True)
    else:
        st.warning("Model results not found. Run `python 04_train_model.py` first, then `05` and `06`.")


with tab3:
    st.markdown("##### Sample log")
    if df is not None:
        st.markdown(f'<p class="field-label">{len(df)} entries · {len(df.columns)} fields</p>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("##### Correlation matrix & distributions")
    corr_path = os.path.join(FIG_DIR, "correlation_matrix.png")
    dist_path = os.path.join(FIG_DIR, "distributions.png")
    c1, c2 = st.columns(2)
    if os.path.exists(corr_path):
        c1.image(corr_path, use_container_width=True)
    else:
        c1.warning("Correlation matrix not found. Run `python 05_stats_and_charts.py`.")
    if os.path.exists(dist_path):
        c2.image(dist_path, use_container_width=True)
    else:
        c2.warning("Distributions chart not found. Run `python 05_stats_and_charts.py`.")

    if df is not None:
        st.markdown("##### Bias check")
        occ_dist = df["occupation_type"].value_counts(normalize=True).round(3)
        st.markdown('<p class="field-label">Occupation balance</p>', unsafe_allow_html=True)
        st.bar_chart(occ_dist, color="#C97D44")
        if occ_dist.max() > 0.6:
            st.warning(
                f"'{occ_dist.idxmax()}' makes up {occ_dist.max()*100:.0f}% of this sample. "
                f"The real Kaggle dataset only contains working professionals "
                f"(nurses, doctors, engineers, etc.) — no students or athletes — "
                f"so predictions for those groups are extrapolations, not interpolations.")


with tab4:
    st.markdown("##### How this project was actually built")
    st.markdown(
        '<p class="field-label">Phase-by-phase build order — not the dataset, the project itself</p>',
        unsafe_allow_html=True)

    st.plotly_chart(build_gantt(PROJECT_PHASES), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="field-label">Phase notes</p>', unsafe_allow_html=True)
    for p in PROJECT_PHASES:
        dot_color = STATUS_COLORS.get(p["status"], "#B7A98C")
        st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:0.6rem; margin-bottom:0.7rem;">
                <div style="width:10px; height:10px; border-radius:50%; background:{dot_color}; margin-top:0.3rem; flex-shrink:0;"></div>
                <div>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; font-weight:500;">{p['phase']}</span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#9C9282;"> — {p['script']}</span>
                    <br><span style="font-size:0.85rem; color:#4A4438;">{p['note']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p class="field-label">Why this order</p>'
        '<p style="font-size:0.9rem; color:#4A4438;">Validation has to come before training, or the model '
        'learns from bad rows. Stats and SHAP both need a fitted model and cleaned dataset. '
        'API/dashboard is last — it just wraps what the pipeline already produced.</p>',
        unsafe_allow_html=True)


st.markdown("""
    <p style="
        font-family:'IBM Plex Mono',monospace;
        font-size:0.75rem;
        color:#1B2430;
        opacity:0.5;
        font-style:italic;
        margin-top:2rem;
        border-top:1px solid #DCD0B8;
        padding-top:0.8rem;
        letter-spacing:0.02em;
    ">
    Caution — This is a predictive insight tool only and is <em>not</em> a medical diagnosis.
    Results should not be used to assess, treat, or make decisions about any individual's health or cognitive condition.
    Always consult a qualified healthcare professional for medical advice.
    </p>""", unsafe_allow_html=True)
