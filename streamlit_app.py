import streamlit as st
import pandas as pd
import joblib
import json
import subprocess

st.set_page_config(page_title="Parkinson’s Predictor", page_icon="🧠")

# ==============================
# Load dataset just for feature names
# ==============================
df = pd.read_csv("data/parkinsons.csv")
X = df.drop("status", axis=1)

# ==============================
# Helper to load model + metrics
# ==============================
def load_model_and_metrics():
    best_model = joblib.load("models/best_model.joblib")
    with open("assets/metrics.json", "r") as f:
        metrics = json.load(f)
    return best_model, metrics

# Initialize session_state
if "best_model" not in st.session_state or "metrics" not in st.session_state:
    st.session_state.best_model, st.session_state.metrics = load_model_and_metrics()

st.title("🧠 Parkinson’s Disease Prediction App")
st.write("This app predicts the likelihood of Parkinson’s Disease based on voice features.")

# --- Model Comparison ---
st.header("📊 Model Comparison")
df_metrics = pd.DataFrame(st.session_state.metrics.items(), columns=["Model", "ROC-AUC"])
st.dataframe(df_metrics)
st.bar_chart(df_metrics.set_index("Model"))

# --- Prediction ---
st.header("🔍 New Prediction")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))
sample = pd.DataFrame([input_data])

if st.button("Predict"):
    pred_prob = st.session_state.best_model.predict_proba(sample)[0,1]
    pred = int(pred_prob >= 0.5)
    if pred == 1:
        st.error(f"❌ Parkinson’s (probability: {pred_prob:.2f})")
    else:
        st.success(f"✅ Healthy (probability: {1-pred_prob:.2f})")

# --- Promote Button ---
st.header("⚡ Promote (Re-train Best Model)")
if st.button("Promote Model (Re-train)"):
    result = subprocess.run(["python", "model_pipeline.py"], capture_output=True, text=True)
    st.success("✔️ Model retrained and promoted successfully!")
    st.text(result.stdout)

    # Reload after retrain into session_state
    st.session_state.best_model, st.session_state.metrics = load_model_and_metrics()

    # Refresh UI with new values
    df_metrics = pd.DataFrame(st.session_state.metrics.items(), columns=["Model", "ROC-AUC"])
    st.dataframe(df_metrics)
    st.bar_chart(df_metrics.set_index("Model"))
