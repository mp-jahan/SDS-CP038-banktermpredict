import streamlit as st
import joblib
import pandas as pd
import datetime as dt

# Load full pipeline (preprocessor + CatBoost model)
pipeline = joblib.load("../models/catboost_best_pipeline.pkl")

st.set_page_config(page_title="Bank Term Deposit Predictor", page_icon="💰")
st.title("💰 Bank Term Deposit Predictor")
st.write("Estimate whether a customer is likely to subscribe to a term deposit based on campaign data.")

# --- Sidebar for categorical features ---
st.sidebar.header("Customer Details")

job = st.sidebar.selectbox(
    "Job Type",
    [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
        "self-employed", "services", "student", "technician", "unemployed", "unknown",
    ],
    index=0,
)

marital = st.sidebar.selectbox("Marital Status", ["single", "married", "divorced", "unknown"], index=0)
education = st.sidebar.selectbox(
    "Education Level",
    ["unknown", "primary", "secondary", "tertiary"],
    index=2,
)
contact = st.sidebar.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"], index=0)
month = st.sidebar.selectbox(
    "Last Contact Month",
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    index=4,
)
poutcome = st.sidebar.selectbox(
    "Previous Campaign Outcome",
    ["unknown", "failure", "other", "success"],
    index=0,
)

default = st.sidebar.radio("Default Credit?", ["no", "yes"], index=0)
housing = st.sidebar.radio("Has Housing Loan?", ["no", "yes"], index=0)
loan = st.sidebar.radio("Has Personal Loan?", ["no", "yes"], index=0)

# --- Main panel for numeric inputs ---
st.header("Campaign Information")

# New or returning customer logic
is_new_customer = st.checkbox("New customer (never contacted before)", value=False)

# Get current date
today = dt.date.today()
current_month_str = today.strftime("%b").lower()  # e.g., 'oct'
current_day = today.day

if is_new_customer:
    pdays = -1
    month = current_month_str
    day = current_day
    st.info(f"New customer: 'Days since last contact' set to 0 and 'month/day' set to {month.title()} {day}.")
else:
    pdays = st.number_input("Days since last contact", 0, 999, 30)
    month = st.selectbox(
    "Last Contact Month",
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    index=4,
    )
    day = st.slider("Day of Month", 1, 31, 15)


age = st.slider("Customer Age", 18, 100, 35)

# balance = st.number_input("Account Balance ($)", -10000, 100000, 1000, step=500)
balance = st.slider("Account Balance ($)", -20000, 200000, 5000)
# st.caption("💡 Note: Values above $40,000 are automatically capped for model consistency.")

campaign = st.number_input("Contacts during this campaign", 1, 50, 2)
previous = st.number_input("Contacts during previous campaign", 0, 50, 0)


# --- Build input DataFrame ---
input_df = pd.DataFrame({
    "age": [age],
    "balance": [balance],
    "campaign": [campaign],
    "previous": [previous],
    "pdays": [pdays],
    "default": [default],
    "housing": [housing],
    "loan": [loan],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "contact": [contact],
    "month": [month],
    "day": [day],
    "poutcome": [poutcome],
})

# --- Display summary ---
st.subheader("🧾 Input Summary")
st.dataframe(
    input_df.drop(columns=["day"]),  # hide technical placeholder
    use_container_width=True,
)

# --- Predict ---
if st.button("Predict Subscription Likelihood"):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.metric("Likelihood of Subscription", f"{probability*100:.1f}%")

    if prediction == 1:
        st.success("✅ The customer is **likely** to subscribe to a term deposit.")
    else:
        st.error("❌ The customer is **unlikely** to subscribe to a term deposit.")