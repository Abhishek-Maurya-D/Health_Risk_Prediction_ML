import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="NovaGen Model Dashboard", layout="wide")

st.title("🚀 NovaGen Classifier")
st.markdown("This app trains a Random Forest model and displays performance metrics.")

# 1. Load Data
@st.cache_data
def load_data():
    # Ensure the CSV is in the same directory
    df = pd.read_csv("novagen_dataset.csv")
    return df

try:
    df = load_data()
    
    with st.expander("👀 View Raw Dataset"):
        st.dataframe(df.head())

    # 2. Preprocessing
    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Training
    with st.spinner('Training Random Forest Model...'):
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=42
        )
        rf.fit(X_train_scaled, y_train) # Note: Used scaled data for consistency
        y_pred_rf = rf.predict(X_test_scaled)

    # 4. Display Results
    st.subheader("📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    acc = accuracy_score(y_test, y_pred_rf)
    rec = recall_score(y_test, y_pred_rf)
    
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Recall", f"{rec:.2%}")

    st.write("**Classification Report:**")
    report_dict = classification_report(y_test, y_pred_rf, output_dict=True)
    st.dataframe(pd.DataFrame(report_dict).transpose())

except FileNotFoundError:
    st.error("Error: 'novagen_dataset.csv' not found. Please ensure the file is in the project folder.")
