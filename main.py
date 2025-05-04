import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model and preprocessing objects
model = joblib.load('laptop_price_prediction_model.sav')
label_encoders = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Load original dataset (for feature reference)
df_original = pd.read_csv('laptop-price-prediction-dataset/data.csv')

df_original = df_original.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')

# Drop unnecessary columns (like index columns or unnamed columns)
df_original = df_original.loc[:, ~df_original.columns.str.contains('^Unnamed', case=False)]

st.title('💻 Laptop Price Prediction')

st.header("Input Laptop Specifications")

# Prepare input data
input_data = {}
for col in df_original.columns:
    if col.lower() == 'price':
        continue  # skip target
    elif df_original[col].dtype == 'object':
        input_data[col] = st.selectbox(f"{col}", df_original[col].unique())
    elif 'ram' in col.lower() or 'rom' in col.lower():
        input_data[col] = st.slider(f"{col}", 4, 1024, int(df_original[col].median()))
    elif 'weight' in col.lower():
        input_data[col] = st.slider(f"{col}", 0.5, 5.0, float(df_original[col].median()))
    elif 'screen' in col.lower():
        input_data[col] = st.slider(f"{col}", 10.0, 20.0, float(df_original[col].median()))
    else:
        input_data[col] = st.number_input(f"{col}", value=float(df_original[col].median()))

# Convert input data into DataFrame
input_df = pd.DataFrame([input_data])

# Apply label encoding where needed
for col in label_encoders:
    if col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Align strictly to model features
model_features = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else [c for c in df_original.columns if c != 'price']

# Drop extra columns not used by model
input_df = input_df[[col for col in input_df.columns if col in model_features]]

# Add missing columns with default 0
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure column order matches model
input_df = input_df[model_features]

# Add button for prediction
if st.button('Predict Price'):
    # Apply scaler if present
    
    # Make prediction
    predicted_price = model.predict(input_df)[0]

    # Display result
    st.subheader('Predicted Price:')
    st.write(f"${predicted_price:,.2f}")
