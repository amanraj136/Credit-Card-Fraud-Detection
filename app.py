import streamlit as st
import pandas as pd
import joblib
import io

st.title("Credit Card Fraud Detection App")

# Load model

model = joblib.load("best_model_gradient_boosting.pkl")
model = model['model']

features = [
"Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11",
"V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22",
"V23","V24","V25","V26","V27","V28","Amount"
]

# ---------------- Single Input Prediction ----------------

st.header("Single Transaction Prediction (Comma-separated input)")

with st.form("single_input_form"):
    user_input_str = st.text_input(f"Enter all features as comma-separated values")
    submitted = st.form_submit_button("Predict")


if submitted:
    try:
        # Convert input string to list of floats
        user_input_list = [float(x.strip()) for x in user_input_str.split(",")]
        
        if len(user_input_list) != len(features):
            st.error(f"Please enter exactly {len(features)} values.")
        else:
            input_df = pd.DataFrame([user_input_list], columns=features)
            pred = model.predict(input_df)[0]
            # prob = model.predict_proba(input_df)[0][1]
            if pred == 1:
                st.error(f"⚠️ Fraudulent Transaction Detected")
            else:
                st.success(f"✅ Normal Transaction")
    except ValueError:
        st.error("Please enter valid numeric values separated by commas.")


# ---------------- File Upload Prediction ----------------

st.header("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if set(features).issubset(df.columns):
        predictions = model.predict(df[features])
        df['Prediction'] = ["Fraudulent" if p==1 else "Normal Transaction" for p in predictions]
        st.dataframe(df)

        # Provide download option
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv_buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv"
        )
    else:
        st.error(f"The CSV must contain all features: {features}")

