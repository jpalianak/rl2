import streamlit as st
from recommender import load_model, get_latest_observation, recommend_action
from data_utils import download_data
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import os

os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(layout="wide")

# Parámetros
SYMBOL = st.sidebar.text_input("Símbolo", "NVDA")
MODEL_PATH = f"dqn_model_{SYMBOL}.pth"
WINDOW_SIZE = 10

# Carga de datos
end_date = datetime.today()
start_date = end_date - timedelta(days=60)
df = download_data(SYMBOL, start=start_date.strftime(
    "%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

# Mostrar gráfico de precio
st.title(f"Dashboard de Trading - {SYMBOL}")
st.subheader("Precio de Cierre")
st.line_chart(df)

# Obtener recomendación
obs, n_obs, n_act = get_latest_observation(SYMBOL, WINDOW_SIZE)
model = load_model(MODEL_PATH, n_obs, n_act)
action = recommend_action(model, obs)

acciones = ["Mantener", "Comprar", "Vender"]
st.subheader("Recomendación de hoy")
st.write(f"👉 Acción recomendada: **{acciones[action]}**")

# Mostrar datos más recientes
st.subheader("Últimos datos")
st.dataframe(df.tail(WINDOW_SIZE))

# Footer
st.caption("Modelo DQN entrenado previamente | Basado en datos de Yahoo Finance")
