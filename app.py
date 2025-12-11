import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Heart Disease Risk Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');
    :root {
        --bg: #0f172a;
        --panel: #111827;
        --accent: #22d3ee;
        --accent-2: #a78bfa;
        --text: #e2e8f0;
        --muted: #94a3b8;
    }
    .stApp {
        background: radial-gradient(120% 120% at 20% 20%, #1f2937 0%, #0f172a 40%, #0b1628 100%);
        color: var(--text);
        font-family: 'Manrope', sans-serif;
    }
    .block-container {
        padding: 2rem 3rem 3.5rem;
        max-width: 1400px;
    }
    .card {
        background: linear-gradient(145deg, rgba(34,211,238,0.08), rgba(167,139,250,0.05));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 18px 60px rgba(0,0,0,0.35);
    }
    .section-title {
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.8rem;
        color: var(--muted);
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text);
    }
    .badge {
        background: rgba(34, 211, 238, 0.12);
        color: #67e8f9;
        border: 1px solid rgba(34, 211, 238, 0.35);
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")


df = load_data()
categorical_cols = df.select_dtypes(include="object").columns
encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
df_encoded = df.copy()
for col, encoder in encoders.items():
    df_encoded[col] = encoder.transform(df_encoded[col])

X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]
model = LogisticRegression(max_iter=500)
model.fit(X, y)

# --- Sidebar filters ------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")
    sex_filter = st.multiselect("Sex", df["Sex"].unique().tolist(), default=df["Sex"].unique().tolist())
    cp_filter = st.multiselect(
        "Chest Pain Type", df["ChestPainType"].unique().tolist(), default=df["ChestPainType"].unique().tolist()
    )
    age_range = st.slider("Age range", int(df["Age"].min()), int(df["Age"].max()), (40, 65))
    risk_view = st.toggle("Show only high-risk patients", value=False)

filtered_df = df[
    df["Sex"].isin(sex_filter)
    & df["ChestPainType"].isin(cp_filter)
    & df["Age"].between(age_range[0], age_range[1])
]
if risk_view:
    filtered_df = filtered_df[filtered_df["HeartDisease"] == 1]
active_df = filtered_df if not filtered_df.empty else df

# --- Hero -----------------------------------------------------------------
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;margin-bottom:1.3rem;">
        <div>
            <div class="section-title">Health Informatics • Cohort Explorer</div>
            <h1 style="margin:0.2rem 0 0.5rem 0;font-size:2.3rem;color:var(--text);">Heart Disease Risk Dashboard</h1>
            <p style="color:var(--muted);max-width:680px;font-size:1rem;">
                Explore risk signals across demographic and clinical features. Use the filters to slice the cohort and run live risk predictions.
            </p>
        </div>
        <div style="display:flex;flex-direction:column;gap:0.35rem;align-items:flex-end;">
            <div class="badge">Model: Logistic Regression</div>
            <div class="badge" style="background:rgba(167,139,250,0.12);color:#c4b5fd;border-color:rgba(167,139,250,0.35);">Dataset: heart.csv</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Metrics row ----------------------------------------------------------
total_patients = len(active_df)
positive_cases = int(active_df["HeartDisease"].sum())
positive_rate = (positive_cases / total_patients * 100) if total_patients else 0
avg_age = active_df["Age"].mean()
median_chol = active_df["Cholesterol"].median()

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown('<div class="card"><div class="metric-label">Patients in view</div>'
                f'<div class="metric-value">{total_patients:,}</div></div>', unsafe_allow_html=True)
with col_b:
    st.markdown('<div class="card"><div class="metric-label">High-risk count</div>'
                f'<div class="metric-value">{positive_cases:,}</div></div>', unsafe_allow_html=True)
with col_c:
    st.markdown('<div class="card"><div class="metric-label">High-risk rate</div>'
                f'<div class="metric-value">{positive_rate:,.1f}%</div></div>', unsafe_allow_html=True)
with col_d:
    st.markdown('<div class="card"><div class="metric-label">Avg age / Median cholesterol</div>'
                f'<div class="metric-value">{avg_age:,.1f} yrs • {median_chol:,.0f} mg/dL</div></div>',
                unsafe_allow_html=True)

st.markdown("---")

# --- Visuals --------------------------------------------------------------
viz_col1, viz_col2 = st.columns([1.2, 1])

with viz_col1:
    st.markdown("### Chest Pain Distribution")
    cp_chart = px.bar(
        active_df,
        x="ChestPainType",
        color="ChestPainType",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="",
    )
    cp_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(cp_chart, use_container_width=True)

with viz_col2:
    st.markdown("### Sex Distribution")
    sex_chart = px.pie(
        active_df,
        names="Sex",
        hole=0.55,
        color_discrete_sequence=["#22d3ee", "#a78bfa", "#fbbf24"],
    )
    sex_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(sex_chart, use_container_width=True)

st.markdown("### Correlation Snapshot")
corr_ax = sns.heatmap(df_encoded.corr(), cmap="mako", center=0, cbar=False)
corr_fig = corr_ax.get_figure()
st.pyplot(corr_fig, use_container_width=True)
plt.close(corr_fig)

st.markdown("### Dataset Preview")
st.dataframe(active_df.head(50), use_container_width=True, height=320)

# --- Prediction form ------------------------------------------------------
st.markdown("### Live Risk Prediction")
with st.form("prediction_form"):
    form_col1, form_col2, form_col3 = st.columns(3)
    with form_col1:
        age = st.slider("Age", 20, 90, 45)
        sex = st.selectbox("Sex", df["Sex"].unique())
        cp = st.selectbox("Chest Pain Type", df["ChestPainType"].unique())
        st_slope = st.selectbox("ST Slope", df["ST_Slope"].unique())
    with form_col2:
        restingbp = st.slider("Resting BP", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 500, 220)
        fasting_bs = st.selectbox("FastingBS (1 if >120 mg/dL)", [0, 1])
        rest_ecg = st.selectbox("Resting ECG", df["RestingECG"].unique())
    with form_col3:
        maxhr = st.slider("Max Heart Rate", 60, 202, 150)
        exercise_angina = st.selectbox("Exercise Angina", df["ExerciseAngina"].unique())
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 5.0, 1.0, step=0.1)

    submit_btn = st.form_submit_button("Predict risk", use_container_width=True)

if submit_btn:
    input_df = pd.DataFrame(
        {
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [cp],
            "RestingBP": [restingbp],
            "Cholesterol": [chol],
            "FastingBS": [fasting_bs],
            "RestingECG": [rest_ecg],
            "MaxHR": [maxhr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_slope],
        }
    )

    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    proba = float(model.predict_proba(input_df)[0][1])
    prediction = int(model.predict(input_df)[0])

    st.markdown(
        f"""
        <div class="card" style="margin-top:0.8rem; background: linear-gradient(120deg, rgba(34,211,238,0.14), rgba(167,139,250,0.12));">
            <div class="section-title">Prediction Result</div>
            <div style="display:flex;justify-content:space-between;align-items:center;gap:1.5rem;">
                <div style="font-size:1.7rem;font-weight:700;color:{'#f87171' if prediction == 1 else '#34d399'};">
                    {"High Risk" if prediction == 1 else "Low Risk"}
                </div>
                <div style="flex:1;">
                    <div style="height:12px;background:rgba(255,255,255,0.08);border-radius:999px;overflow:hidden;">
                        <div style="height:12px;width:{proba*100:.1f}%;background:linear-gradient(90deg,#22d3ee,#a78bfa);"></div>
                    </div>
                    <div style="margin-top:0.4rem;color:var(--muted);font-size:0.95rem;">
                        Probability of heart disease: <strong style="color:var(--text);">{proba*100:.1f}%</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
