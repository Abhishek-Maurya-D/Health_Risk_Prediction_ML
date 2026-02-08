import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="NovaGen Health Predictor", layout="centered")

st.title("🩺 NovaGen Status Predictor")
st.markdown("Enter the required details below to check the status result.")

# 1. Load and Train Model (Hidden from user)
@st.cache_resource
def train_model():
    df = pd.read_csv("novagen_dataset.csv")
    X = df.drop("Target", axis=1)
    y = df["Target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X_scaled, y)
    
    return rf, scaler, X.columns

try:
    model, scaler, feature_names = train_model()

    # 2. User Input Form
    st.subheader("👤 User Information")
    
    # We dynamically create inputs based on your CSV columns
    user_inputs = {}
    
    # Splitting inputs into two columns for a better UI
    cols = st.columns(2)
    for i, col_name in enumerate(feature_names):
        with cols[i % 2]:
            # Adjust min/max values based on your typical data ranges
            user_inputs[col_name] = st.number_input(f"Enter {col_name}", value=0.0)

    st.markdown("---")

    # 3. Prediction Logic
    if st.button("Analyze Results", type="primary"):
        # Convert inputs to dataframe
        input_df = pd.DataFrame([user_inputs])
        
        # Scale the input using the saved scaler
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # 4. Display Result
        st.subheader("Results:")
        
        if prediction == 1: # Assuming 1 = 'Good'
            st.success("### Result: **GOOD**")
            st.balloons()
        else:
            st.warning("### Result: **NOT GOOD**")
        
        # Show confidence levels
        st.write(f"Confidence: {max(prediction_proba):.2%}")

except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'novagen_dataset.csv' is in the folder to train the model.")
