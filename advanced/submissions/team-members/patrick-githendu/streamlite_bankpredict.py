import streamlit as st
import pandas as pd
import joblib
import os

# --- Load preprocessing objects only (not model) ---
@st.cache_resource
def load_preprocessing():
    scaler = joblib.load('advanced/submissions/team-members/patrick-githendu/scaler.joblib')
    label_encoders = joblib.load('advanced/submissions/team-members/patrick-githendu/label_encoders.joblib')
    return scaler, label_encoders

scaler, label_encoders = load_preprocessing()
# Remove 'y' from scaler if present
if hasattr(scaler, 'feature_names_in_'):
    scaler.feature_names_in_ = [f for f in scaler.feature_names_in_ if f != 'y']

# Remove 'y' from label_encoders if present
if 'y' in label_encoders:
    del label_encoders['y']


# --- Feature list (original features only, no engineered features) ---
feature_list = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
]

# --- Streamlit UI ---
st.title("Bank Term Deposit Prediction (Focal Loss Model)")
st.write("Enter customer details to predict the likelihood of subscribing to a term deposit.")

# --- Input form for all original features only ---
user_input = {}
for feature in feature_list:
    if feature in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
        options = label_encoders[feature].classes_
        user_input[feature] = st.selectbox(f"{feature.capitalize()}", options)
    else:
        user_input[feature] = st.number_input(f"{feature.capitalize()}", value=0)

# --- Feature engineering (must match notebook) ---
def engineer_features(df):
    # Ensure bins are strictly increasing for pd.cut
    max_campaign = df['campaign'].max()
    if max_campaign <= 2:
        bins = [-1, max_campaign]
        labels = ['low']
    elif max_campaign <= 5:
        bins = [-1, 2, max_campaign]
        labels = ['low', 'medium']
    else:
        bins = [-1, 2, 5, max_campaign]
        labels = ['low', 'medium', 'high']
    df['campaign_intensity'] = pd.cut(df['campaign'], bins=bins, labels=labels)
    df['job_education'] = df['job'] + '_' + df['education']
    df['married_with_loan'] = ((df['marital'] == 'married') & (df['loan'] == 'yes')).astype(int)
    df['single_with_housing'] = ((df['marital'] == 'single') & (df['housing'] == 'yes')).astype(int)
    df['recent_contact'] = (df['pdays'] < 30).astype(int)
    return df

# --- Predict button ---
if st.button("Predict"):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    # --- Focal loss function (must match training) ---
    def focal_loss(gamma=2., alpha=0.25):
        def focal_loss_fixed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            loss = alpha * tf.pow(1. - pt, gamma) * bce
            return tf.reduce_mean(loss)
        return focal_loss_fixed

    # Load model only when needed
    model = load_model('advanced/submissions/team-members/patrick-githendu/focal_model.h5', custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)})

    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    # Feature engineering
    input_df = engineer_features(input_df)
    # Encode categorical features
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])
    # Scale numeric features
    numeric_cols = scaler.feature_names_in_
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    # Predict
    pred_prob = model.predict(input_df)[0][0]
    pred = int(pred_prob > 0.5)
    st.write(f"**Prediction:** {'Subscribed' if pred else 'Not Subscribed'} (Probability: {pred_prob:.2f})")
    
