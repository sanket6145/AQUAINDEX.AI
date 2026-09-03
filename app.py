"""
==================================================================
AquaIndex - Water Quality Intelligence Platform (Streamlit App)
==================================================================
Premium milky-white themed dashboard for Water Quality Index (WQI)
prediction, with prediction history and CSV export.

RUN:
    pip install -r requirements.txt
    streamlit run app.py

On first run, the app looks for wqi_model.pkl and wqi_features.pkl
next to app.py. If they are missing, it automatically trains a
model from wqi.csv (must be in the same folder) and saves those
.pkl files for instant loading next time.
==================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from datetime import datetime
import plotly.graph_objects as go
# Load trained model
model = joblib.load("model.pkl")

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AquaIndex | Water Quality Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# GLOBAL THEME (CSS is flush-left on purpose: any indented line here
# gets treated as a Markdown code block by Streamlit instead of being
# rendered as real HTML, which was the bug in the previous version)
# ------------------------------------------------------------------
CSS = """
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
--bg-milk: #FAF9F5;
--card-white: #FFFFFF;
--ink: #1C2B2D;
--ink-soft: #5B6B6D;
--teal-deep: #0E4F52;
--teal: #14767C;
--gold: #B78A2E;
--line: #E7E3D9;
--excellent: #2E7D5B;
--good: #14767C;
--poor: #C48A1D;
--verypoor: #C06B2C;
--unsuitable: #B23A3A;
}
html, body, [class*="css"] {
font-family: 'Inter', sans-serif;
color: var(--ink);
}
.stApp {
background: var(--bg-milk);
}
#MainMenu, footer, header {visibility: hidden;}
.brand-wrap {
display: flex;
align-items: center;
gap: 18px;
padding: 8px 0 28px 0;
border-bottom: 1px solid var(--line);
margin-bottom: 32px;
}
.brand-title {
font-family: 'Fraunces', serif;
font-size: 2.4rem;
font-weight: 600;
color: var(--teal-deep);
letter-spacing: -0.5px;
margin: 0;
line-height: 1.1;
}
.brand-title span {
color: var(--gold);
font-style: italic;
}
.brand-sub {
font-family: 'Inter', sans-serif;
font-size: 0.82rem;
letter-spacing: 3px;
text-transform: uppercase;
color: var(--ink-soft);
margin-top: 2px;
}
.card {
background: var(--card-white);
border: 1px solid var(--line);
border-radius: 18px;
padding: 28px 30px;
box-shadow: 0 4px 24px rgba(20, 40, 40, 0.04);
}
.card-title {
font-family: 'Fraunces', serif;
font-size: 1.25rem;
font-weight: 600;
color: var(--teal-deep);
margin-bottom: 4px;
}
.card-eyebrow {
font-size: 0.72rem;
letter-spacing: 2.5px;
text-transform: uppercase;
color: var(--gold);
font-weight: 600;
margin-bottom: 6px;
}
.stNumberInput input, .stTextInput input {
border-radius: 10px !important;
border: 1px solid var(--line) !important;
background-color: #FCFBF8 !important;
color: var(--ink) !important;
-webkit-text-fill-color: var(--ink) !important;
}
.stNumberInput label, .stTextInput label, .stSelectbox label {
font-size: 0.85rem !important;
font-weight: 500 !important;
color: var(--ink-soft) !important;
}
.stNumberInput button svg, .stNumberInput button {
color: var(--ink) !important;
fill: var(--ink) !important;
}
.stButton > button {
background: var(--teal-deep);
color: #fff;
border: none;
border-radius: 10px;
padding: 0.6rem 1.6rem;
font-weight: 600;
letter-spacing: 0.5px;
transition: all 0.2s ease;
width: 100%;
}
.stButton > button:hover {
background: var(--teal);
color: #fff;
box-shadow: 0 4px 14px rgba(14, 79, 82, 0.25);
}
.stDownloadButton > button {
background: var(--gold);
color: #fff;
border: none;
border-radius: 10px;
font-weight: 600;
width: 100%;
}
.stDownloadButton > button:hover {
background: #A17A28;
color: #fff;
}
.result-badge {
display: inline-block;
padding: 6px 18px;
border-radius: 999px;
font-weight: 600;
font-size: 0.95rem;
letter-spacing: 0.3px;
margin-top: 6px;
}
section[data-testid="stSidebar"] {
background: #F3F1EA;
border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h3 {
color: var(--ink) !important;
}
section[data-testid="stSidebar"] strong {
color: var(--teal-deep) !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
color: var(--ink-soft) !important;
}
.stDataFrame {
border-radius: 12px;
overflow: hidden;
border: 1px solid var(--line);
}
hr {
border: none;
border-top: 1px solid var(--line);
margin: 24px 0;
}
.footer-note {
text-align: center;
color: var(--ink-soft);
font-size: 0.78rem;
padding: 30px 0 10px 0;
letter-spacing: 0.5px;
}
[data-testid="stStatusWidget"], .stSpinner {
background: var(--card-white) !important;
border: 1px solid var(--line) !important;
border-radius: 14px !important;
box-shadow: 0 4px 24px rgba(20, 40, 40, 0.06) !important;
}
[data-testid="stStatusWidget"] p, .stSpinner p,
[data-testid="stStatusWidget"] span, .stSpinner span {
color: var(--ink) !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# LOGO - water drop SVG, teal gradient with a small gold droplet
# ------------------------------------------------------------------
DROP_LOGO = '<svg width="52" height="52" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="dropGrad" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stop-color="#14767C"/><stop offset="100%" stop-color="#0E4F52"/></linearGradient></defs><path d="M50 6 C50 6 82 46 82 66 C82 85 68 96 50 96 C32 96 18 85 18 66 C18 46 50 6 50 6 Z" fill="url(#dropGrad)"/><ellipse cx="38" cy="60" rx="7" ry="11" fill="#DCEEEE" opacity="0.55"/><circle cx="63" cy="72" r="3.2" fill="#B78A2E" opacity="0.9"/></svg>'

# ------------------------------------------------------------------
# FEATURE ENGINEERING (shared by training and prediction)
# ------------------------------------------------------------------
def safe_col(possible_names, columns):
    for name in possible_names:
        for c in columns:
            if name.lower().replace(" ", "") in c.lower().replace(" ", ""):
                return c
    return None


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = list(df.columns)

    c_do = safe_col(["Dissolved O2", "Dissolved Oxygen", "DO"], cols)
    c_bod = safe_col(["Bio-Chemical", "BOD"], cols)
    c_temp = safe_col(["Temperature"], cols)
    c_ph = safe_col(["pH"], cols)
    c_fc = safe_col(["Faecal Coliform"], cols)
    c_tc = safe_col(["Total Coliform"], cols)
    c_fs = safe_col(["Faecal Strep"], cols)
    c_nit = safe_col(["Nitrate"], cols)

    if c_fc and c_tc:
        df["FC_TC_ratio"] = df[c_fc] / (df[c_tc] + 1)
    if c_bod and c_do:
        df["BOD_DO_ratio"] = df[c_bod] / (df[c_do] + 1)
    if c_fs and c_tc:
        df["FS_TC_ratio"] = df[c_fs] / (df[c_tc] + 1)
    if c_temp and c_do:
        df["Temp_DO_interaction"] = df[c_temp] * df[c_do]
    if c_ph and c_nit:
        df["pH_Nitrate_interaction"] = df[c_ph] * df[c_nit]
    for c in [c_fc, c_tc, c_fs]:
        if c:
            df[f"{c}_log"] = np.log1p(df[c])

    return df


def engineer_features(raw: dict, feature_cols: list) -> pd.DataFrame:
    row = add_engineered_columns(pd.DataFrame([raw]))
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    return row[feature_cols]


# ------------------------------------------------------------------
# TRAIN MODEL DIRECTLY FROM wqi.csv (used only if no .pkl files exist)
# ------------------------------------------------------------------
def train_from_csv(csv_path="wqi.csv", target_col="WQI", random_state=42):
    from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    feature_cols_raw = [c for c in df.columns if c != target_col]
    for col in feature_cols_raw:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    df = add_engineered_columns(df)
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [None, 10, 16, 20],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        param_distributions=param_dist, n_iter=8, cv=kf,
        scoring="r2", n_jobs=-1, random_state=random_state,
    )
    search.fit(X_train.values, y_train)
    model = search.best_estimator_

    r2 = r2_score(y_test, model.predict(X_test.values))

    meta = {"use_scaled": False, "feature_cols": feature_cols, "test_r2": r2}
    joblib.dump(model, "wqi_model.pkl")
    joblib.dump(meta, "wqi_features.pkl")

    return model, meta


# ------------------------------------------------------------------
# LOAD MODEL ARTIFACTS (cached so this runs once per session).
# If the .pkl files already exist, they are loaded instantly.
# If not, the app trains a model directly from wqi.csv and saves
# the .pkl files for next time.
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    # 1) Preferred file names (matches this app's own training output)
    try:
        model = joblib.load("wqi_model.pkl")
        meta = joblib.load("wqi_features.pkl")
        # meta may be a plain list of feature names (older format) —
        # normalise it into the dict shape the rest of the app expects
        if isinstance(meta, list):
            meta = {"use_scaled": False, "feature_cols": meta}
        scaler = None
        try:
            scaler = joblib.load("wqi_scaler.pkl")
        except Exception:
            pass
        return model, meta, scaler, None
    except Exception:
        pass

    # 2) Older naming used in an earlier version of this project
    try:
        model = joblib.load("wqi_best_model.pkl")
        meta = joblib.load("wqi_meta.pkl")
        scaler = None
        try:
            scaler = joblib.load("wqi_scaler.pkl")
        except Exception:
            pass
        return model, meta, scaler, None
    except Exception:
        pass

    # 3) Nothing found — train fresh from wqi.csv and save as
    #    wqi_model.pkl / wqi_features.pkl for next time
    try:
        model, meta = train_from_csv()
        return model, meta, None, None
    except Exception as e:
        return None, None, None, str(e)

with st.spinner("Preparing model..."):
    model, meta, scaler, load_error = load_artifacts()
MODEL_READY = model is not None


def get_water_quality(wqi):
    if wqi <= 25:
        return "Excellent", "var(--excellent)", "#E7F4EE"
    elif wqi <= 50:
        return "Good", "var(--good)", "#E1F1F1"
    elif wqi <= 75:
        return "Poor", "var(--poor)", "#FBF0DC"
    elif wqi <= 100:
        return "Very Poor", "var(--verypoor)", "#FBEADD"
    else:
        return "Unsuitable for Drinking", "var(--unsuitable)", "#F9E3E3"


def make_gauge(wqi_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=wqi_value,
        number={"font": {"size": 40, "color": "#0E4F52", "family": "Fraunces"}},
        gauge={
            "axis": {"range": [0, 130], "tickcolor": "#5B6B6D"},
            "bar": {"color": "#0E4F52", "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "#CDEBDC"},
                {"range": [25, 50], "color": "#CDE9EA"},
                {"range": [50, 75], "color": "#F5E3B8"},
                {"range": [75, 100], "color": "#F5D3B8"},
                {"range": [100, 130], "color": "#F3C4C4"},
            ],
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
header_html = (
    '<div class="brand-wrap">' + DROP_LOGO +
    '<div><p class="brand-title">Aqua<span>Index</span></p>'
    '<p class="brand-sub">Water Quality Intelligence Platform</p></div></div>'
)
st.markdown(header_html, unsafe_allow_html=True)

if not MODEL_READY:
    error_html = (
        '<div style="background:#F9E3E3; border:1px solid var(--unsuitable); '
        'border-radius:14px; padding:18px 22px; color:var(--ink);">'
        '<p style="font-weight:600; color:var(--unsuitable); margin-bottom:6px;">'
        'Could not load or train a model</p>'
        '<p style="font-size:0.9rem; color:var(--ink); margin-bottom:6px;">'
        "Place <code>wqi.csv</code> in the same folder as app.py and rerun — "
        "the app will train and save the model automatically.</p>"
        f'<p style="font-size:0.8rem; color:var(--ink-soft);">Error: {load_error}</p>'
        '</div>'
    )
    st.markdown(error_html, unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    sidebar_logo_html = (
        '<div style="text-align:center; padding: 10px 0 20px 0;">' + DROP_LOGO +
        '<p style="font-family:\'Fraunces\',serif; font-size:1.1rem; '
        'color:#0E4F52; margin-top:8px;">AquaIndex</p></div>'
    )
    st.markdown(sidebar_logo_html, unsafe_allow_html=True)
    st.markdown("### About")
    st.markdown(
        "AquaIndex uses a trained machine-learning model to estimate the "
        "**Water Quality Index (WQI)** from physico-chemical and "
        "bacteriological parameters, and classifies the result into a "
        "standard quality band."
    )
    st.markdown("---")
    st.markdown("### Quality Bands")
    st.markdown(
        "- 🟢 **0-25** Excellent\n"
        "- 🔵 **26-50** Good\n"
        "- 🟡 **51-75** Poor\n"
        "- 🟠 **76-100** Very Poor\n"
        "- 🔴 **100+** Unsuitable for Drinking"
    )
    st.markdown("---")
    st.caption(f"Session started · {datetime.now().strftime('%d %b %Y')}")

# ------------------------------------------------------------------
# MAIN LAYOUT - INPUT + RESULT
# ------------------------------------------------------------------
col_input, col_result = st.columns([1.1, 1], gap="large")

with col_input:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-eyebrow">Sample Parameters</p>', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Enter Water Sample Data</p>', unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        temperature = st.number_input("Temperature (°C)", value=15.0, step=0.1, format="%.2f")
        do = st.number_input("Dissolved Oxygen (mg/L)", value=8.5, step=0.1, format="%.2f")
        ph = st.number_input("pH", value=7.8, step=0.01, format="%.2f")
        bod = st.number_input("BOD (mg/L)", value=0.3, step=0.01, format="%.3f")
        fs = st.number_input("Faecal Streptococci (MPN/100mL)", value=150.0, step=1.0, format="%.2f")
    with c2:
        nitrate = st.number_input("Nitrate (mg/L)", value=0.3, step=0.01, format="%.3f")
        fc = st.number_input("Faecal Coliform (MPN/100mL)", value=50.0, step=1.0, format="%.2f")
        tc = st.number_input("Total Coliform (MPN/100mL)", value=250.0, step=1.0, format="%.2f")
        conductivity = st.number_input("Conductivity (mho/cm)", value=120.0, step=1.0, format="%.2f")

    predict_clicked = st.button("⟶  Predict Water Quality")
    st.markdown('</div>', unsafe_allow_html=True)

raw_input = {
    "Temperature": temperature,
    "Dissolved Oxygen": do,
    "pH": ph,
    "Bio-Chemical Oxygen Demand (mg/L)": bod,
    "Faecal Streptococci (MPN/100 mL)": fs,
    "Nitrate (mg/L)": nitrate,
    "Faecal Coliform (MPN/100 mL)": fc,
    "Total Coliform (MPN/100 mL)": tc,
    "Conductivity (mho/Cm)": conductivity,
}

with col_result:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-eyebrow">Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Result</p>', unsafe_allow_html=True)

    if predict_clicked:
        try:
            X_row = engineer_features(raw_input, meta["feature_cols"])
            X_input = scaler.transform(X_row) if (meta.get("use_scaled") and scaler is not None) else X_row.values
            predicted_wqi = float(model.predict(X_input)[0])
            category, color, bg = get_water_quality(predicted_wqi)

            st.plotly_chart(make_gauge(predicted_wqi), use_container_width=True)

            result_html = (
                f'<div style="background:{bg}; border-radius:14px; padding:14px 18px; margin-top:-10px;">'
                f'<span class="result-badge" style="background:{bg}; color:{color}; border:1px solid {color};">'
                f'{category}</span>'
                f'<p style="color:var(--ink-soft); font-size:0.85rem; margin-top:8px;">'
                f'Predicted WQI Score: <b style="color:var(--ink);">{predicted_wqi:.2f}</b></p></div>'
            )
            st.markdown(result_html, unsafe_allow_html=True)

            record = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            record.update(raw_input)
            record["Predicted WQI"] = round(predicted_wqi, 2)
            record["Quality Category"] = category
            st.session_state.history.append(record)

        except Exception as e:
            fail_html = (
                '<div style="background:#F9E3E3; border:1px solid var(--unsuitable); '
                'border-radius:14px; padding:14px 18px; color:var(--ink);">'
                f'<p style="font-size:0.9rem;">Prediction failed: {e}</p></div>'
            )
            st.markdown(fail_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="text-align:center; padding: 50px 10px; color: var(--ink-soft);">'
            '<p style="font-size:0.95rem;">Enter sample data and click '
            '<b>Predict</b> - the result will appear here.</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# PREDICTION HISTORY
# ------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

hist_col1, hist_col2 = st.columns([3, 1])
with hist_col1:
    st.markdown('<p class="card-eyebrow">Session Log</p>', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Prediction History</p>', unsafe_allow_html=True)
with hist_col2:
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        csv_buffer = io.StringIO()
        hist_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇  Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"aquaindex_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
    if st.button("🗑  Clear History", key="clear_hist"):
        st.session_state.history = []
        st.rerun()
else:
    st.markdown(
        '<p style="color:var(--ink-soft); font-size:0.9rem;">'
        'No predictions yet. Run your first prediction to see the history here.</p>',
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="footer-note">AquaIndex © 2026 — Machine-Learning Powered Water Quality Intelligence</p>',
    unsafe_allow_html=True,
)
