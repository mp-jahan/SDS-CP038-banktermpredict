# Streamlit Web App for Bank Term Deposit Prediction

import streamlit as st
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('catboost').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)

# --- Page Config ---
st.set_page_config(page_title="Bank Term Deposit Predictor", page_icon=":bank:", layout="centered")

# --- Load Model ---
MODEL_FILE = "beginner/submissions/team-members/jackiecwv/bankterm_pipeline.pkl" 
# MODEL_FILE = "bankterm_pipeline.pkl"
# Locked threshold from your notebook (recall-first)
LOCKED_THRESHOLD = 0.45

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

model = load_model(MODEL_FILE)

# --- Custom CSS ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f6f8fc 0%, #e5f0ff 100%);
            font-family: 'Segoe UI', sans-serif;
        }
        .big-title {
            font-size: 2.6em;
            color: #293241;
            font-weight: bold;
            letter-spacing: 1px;
            text-align: center;
            margin-bottom: 0.2em;
        }
        .subtitle {
            color: #3a5a97;
            font-size: 1.1em;
            text-align: center;
            margin-bottom: 1.2em;
        }
        .input-card {
            background: #eaf4ff;
            border-radius: 14px;
            box-shadow: 0 2px 8px rgba(60, 120, 180, 0.07);
            padding: 1.5em 1em 0.3em 1em;
            margin-bottom: 1em;
            margin-top: 1em;
        }
        .result-card {
            background: #fff;
            border-radius: 18px;
            box-shadow: 0 3px 18px rgba(60, 120, 180, 0.08);
            padding: 2em;
            margin: 2em auto;
            max-width: 480px;
        }
        .prob-meter {
            font-size: 1.5em;
            font-weight: bold;
            color: #294E80;
            margin: 0.5em 0;
        }
        .success-label {
            color: #0a8754;
            font-weight: bold;
        }
        .fail-label {
            color: #ff6633;
            font-weight: bold;
        }
        .byline {
            font-size: 0.95em;
            color: #7e88a2;
            margin-top: 2em;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown('<div class="big-title">💳 Bank Term Deposit Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by LightGBM (Balanced, Untuned) — Recall-first threshold</div>', unsafe_allow_html=True)
st.info(f"Model: LightGBM (Balanced, Untuned) • Threshold: {LOCKED_THRESHOLD:.2f}")

# --- Input Section Card ---
st.markdown("""
<div class="input-card">
  <span style="font-size:1.2em; color:#294E80; font-weight:600;">🔽 Enter Client Features</span>
  <div style="font-size:0.96em; color:#576887; margin-bottom: 0.5em;">
    Set features below to customize your prediction. All fields are required.
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔧 Show/Hide All Feature Inputs", expanded=True):
    age = st.number_input("Age", min_value=18, max_value=100, value=35, help="Client's age (18-100)")
    job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"], help="Client's profession")
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has Credit in Default?", ["no", "yes"])
    balance = st.number_input("Account Balance", value=1000, help="Current balance in euros")
    housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])
    day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
    month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    campaign = st.number_input("Number of Contacts During Campaign", value=1)
    pdays = st.number_input("Days Since Last Contact (-1 means never)", value=-1)
    previous = st.number_input("Number of Contacts Before This Campaign", value=0)
    poutcome = st.selectbox("Previous Outcome", ["unknown", "other", "failure", "success"])

X_input = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])

# --- Predict Button ---
st.markdown("""<div style="margin-top: -1em; font-size:1.1em; color:#0a8754;">⬇️ Click "Predict" to see results</div>""", unsafe_allow_html=True)

if st.button("🎯 Predict"):
    y_pred_proba = model.predict_proba(X_input)[:, 1][0]
    y_pred_label = "yes" if y_pred_proba >= LOCKED_THRESHOLD else "no"

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    st.markdown(f"<div class='prob-meter'>Probability of Subscription: <b>{y_pred_proba:.1%}</b></div>", unsafe_allow_html=True)
    st.progress(y_pred_proba)
    
    if y_pred_label == "yes":
        st.markdown("<div class='success-label'>✅ Likely to Subscribe!</div>", unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("<div class='fail-label'>⚠️ Unlikely to Subscribe</div>", unsafe_allow_html=True)

    st.write(f"**Threshold used:** {LOCKED_THRESHOLD:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
**How does this help?**  
This tool helps bank teams identify clients most likely to subscribe to a term deposit, making marketing efforts more efficient and focused.
""")

st.markdown('<div class="byline">Demo app for BankTermPredict | Designed by Jackie CWV 🤖</div>', unsafe_allow_html=True)

