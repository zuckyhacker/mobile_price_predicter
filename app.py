import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
le = pickle.load(open("brand_encoder.pkl", "rb"))

st.title("📱 AI Mobile Price Predictor")

brand = st.selectbox("Brand", ["Samsung", "Apple", "Xiaomi", "Realme", "OnePlus"])
ram = st.slider("RAM (GB)", 1, 16)
battery = st.slider("Battery Power", 1000, 7000)
memory = st.slider("Internal Memory", 8, 512)
camera = st.slider("Primary Camera", 2, 108)
weight = st.slider("Weight", 100, 300)

brand_encoded = le.transform([brand])[0]

input_data = pd.DataFrame([[ram, battery, memory, camera, weight, brand_encoded]], columns=features)

if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Price: ₹{int(prediction[0])}")

data = pd.read_csv("mobile_data.csv")
if 'price_range' in data.columns:
    data['price'] = data['price_range'] * 10000

st.subheader("📊 RAM vs Price")
plt.scatter(data['ram'], data['price'])
st.pyplot(plt)

st.subheader("🔍 Feature Importance")
importances = model.feature_importances_
st.bar_chart(importances)
