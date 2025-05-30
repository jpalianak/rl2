# 📈 Double DQN Trading Agent
Este proyecto implementa un agente de trading algorítmico basado en Double Deep Q-Learning (Double DQN), entrenado para tomar decisiones de inversión (comprar, vender o mantener) utilizando datos históricos de precios bursátiles. El sistema está compuesto por un entorno de simulación personalizado y un modelo de aprendizaje profundo construido con PyTorch.

## 🚀 Características
- Entrenamiento de un agente con algoritmo Double DQN para evitar sobreestimaciones de valor.

- Descarga automática de datos históricos con yfinance.

- Simulación de operaciones financieras con portafolio y balance.

- Recomendaciones diarias de acción.

- Evaluación visual y cuantitativa del desempeño del agente.

- Arquitectura modular y extensible.

## 🗂️ Estructura del Proyecto
```text
.
├── data_utils.py            # Descarga y preprocesamiento de datos
├── model.py                 # Arquitectura de la red (Q-Network)
├── trading_env.py          # Entorno de trading personalizado (Gym)
├── train.py                # Entrenamiento del agente Double DQN
├── evaluate.py             # Evaluación del modelo entrenado
├── recommend.py            # Recomendación diaria (uso en tiempo real)
├── main.py                 # Punto de entrada principal del proyecto
├── dqn_model_<SYMBOL>.pth  # Modelo entrenado (por ticker)
├── dashboard.py            # Dashboar de visualización en Streamlit
└── README.md
```

## ▶️ Ejecución
El script principal es main.py, el cual ejecuta el entrenamiento y evaluación.

## 🧠 Entrenamiento
El agente aprende a operar maximizando la recompensa total a lo largo del tiempo. Se utiliza Double DQN, con una red objetivo actualizada periódicamente.

Parámetros típicos:

- symbol: ticker del activo (ej. "AAPL")

- window_size: número de días observados

- initial_balance: capital inicial

- episodes: número de episodios de entrenamiento

## 📊 Evaluación
El script evaluate.py permite analizar el rendimiento del modelo en un conjunto de datos separado. Se informa:

- Recompensa acumulada

- Días operados

- Balance final

- Acciones restantes

- Valor total final del portafolio

- Gráfico de evolución

## 📈 Métricas de desempeño
- Ganancia/Pérdida neta

- Recompensa total acumulada

- Porcentaje de ganancia

- Evolución del portafolio

## 🎛️ Dashboard con Streamlit
Este proyecto incluye una interfaz interactiva construida con Streamlit (dashboard.py) que permite:

- Ingresar el símbolo de un activo (ej: AAPL, NVDA, BTC-USD). En este momento solo incluye NVDA.

- Cargar el modelo entrenado correspondiente.

- Obtener una recomendación de inversión en tiempo real.

- Visualizar la evolución del precio del activo.