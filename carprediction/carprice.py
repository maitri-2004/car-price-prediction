# Created on May 2, 2025
# Author: MAITRI

import streamlit as st
import numpy as np
import pickle


MODEL_PATH = "D:/machine learning/carprediction/car_price.sav"  # Adjust if needed
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


st.set_page_config(page_title="🚗 Car Price Predictor", layout="centered")

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #2980b9;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #1c5980;
        transition: 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">🚗 Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter car details to predict its selling price (in lakhs ₹)</div>', unsafe_allow_html=True)


with st.form("car_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        year = st.slider("Year of Purchase", 2000, 2023, 2015)
        present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1, format="%.2f")
        kms_driven = st.number_input("Kilometers Driven", min_value=0)
        owner = st.selectbox("Number of Previous Owners", options=[0, 1, 2, 3])

    with col2:
        fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
        seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
        transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])

    submit = st.form_submit_button("🔮 Predict")

fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
seller_map = {'Dealer': 0, 'Individual': 1}
trans_map = {'Manual': 0, 'Automatic': 1}

if submit:
    try:
        features = np.array([[year, present_price, kms_driven, owner,
                              fuel_map[fuel_type],
                              seller_map[seller_type],
                              trans_map[transmission]]])
        
        prediction = model.predict(features)[0]
        st.success(f" Estimated Selling Price: ₹ {prediction:.2f} lakhs")
    except Exception as e:
        st.error(f" Error: {e}")
