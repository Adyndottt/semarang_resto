import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Prediksi Rating Restoran")

# Load model yang sudah dilatih di Colab
model = joblib.load("models/model.pkl")

st.write("Masukkan nilai-nilai fitur berikut untuk prediksi:")

def get_input():
    features = {}
    features["rating_numbers"] = st.number_input("Jumlah Rating", 0, 10000)
    features["average_operation_hours"] = st.number_input("Jam Operasional", 0.0, 24.0)
    features["cash_payment_only"] = st.selectbox("Hanya Terima Tunai?", [0, 1])
    features["wifi_facility"] = st.selectbox("Ada WiFi?", [0, 1])
    features["open_space"] = st.selectbox("Ada Ruang Terbuka?", [0, 1])
    return pd.DataFrame([features])

input_df = get_input()

st.subheader("Data yang Anda masukkan:")
st.write(input_df)

if st.button("Prediksi"):
    if input_df.isnull().values.any():
        st.warning("Mohon lengkapi semua input sebelum memprediksi.")
    else:
        prediction = model.predict(input_df)[0]
        st.success(f"Prediksi rating: {prediction:.2f}")
        st.caption("Rating diprediksi dalam skala 1.0 (buruk) hingga 5.0 (sangat baik)")
