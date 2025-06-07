import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.title("🤖 Model Pelatihan - Prediksi Rating Restoran")

df = pd.read_csv("data/semarang_resto_dataset.csv")

# Pra-pemrosesan
df = df.drop(columns=["resto_id", "resto_name", "resto_address"])
df = df.dropna()

X = df.drop(columns=["resto_rating", "resto_type"])
y = df["resto_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

joblib.dump(model, "models/model.pkl")

st.success(f"Model dilatih dengan MSE: {mse:.4f}")
