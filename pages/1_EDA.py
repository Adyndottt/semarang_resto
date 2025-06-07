import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 EDA - Eksplorasi Data Restoran")

# Load dataset
df = pd.read_csv("data/semarang_resto_dataset.csv")

st.subheader("Preview Data")
st.dataframe(df.head())

st.subheader("Statistik Deskriptif")
st.write(df.describe())

st.subheader("Distribusi Rating Restoran")
fig, ax = plt.subplots()
sns.histplot(df["resto_rating"], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Top 10 Jenis Restoran")
top_types = df["resto_type"].value_counts().head(10)
st.bar_chart(top_types)
