import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("📊 Telecom Churn Prediction")

# Upload file
file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file, sep=';')

    st.write("Dataset Preview")
    st.write(df.head())

    # Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    # Prediction
    df['Risk Score'] = model.predict_proba(X_scaled)[:,1]

    st.write("Prediction Output")
    st.write(df[['Risk Score']].head())
