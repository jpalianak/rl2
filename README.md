# 📈 Double DQN Trading Agent
Este proyecto implementa un agente de trading algorítmico basado en Double Deep Q-Learning (Double DQN), entrenado para tomar decisiones de inversión (comprar, vender o mantener) utilizando datos históricos de precios bursátiles. El sistema está compuesto por un entorno de simulación personalizado y un modelo de aprendizaje profundo construido con PyTorch.

## 🚀 Características
- Entrenamiento de un agente con algoritmo Double DQN para evitar sobreestimaciones de valor.

- Descarga automática de datos históricos con yfinance.

- Simulación de operaciones financieras con portafolio y balance.

- Recomendaciones diarias de acción.

- Arquitectura modular y extensible.

## 🗂️ Estructura del Proyecto
```text
.
├── data_utils.py           # Descarga y preprocesamiento de datos
├── model.py                # Arquitectura de la red (Q-Network)
├── trading_env.py          # Entorno de trading personalizado (Gym)
├── train.py                # Entrenamiento del agente Double DQN
├── evaluate.py             # Evaluación del modelo entrenado
├── main.py                 # Punto de entrada principal del proyecto
├── dqn_model_<SYMBOL>.pth  # Modelo entrenado (por ticker)
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

- Profit Factor


## 📈 Salida del entrenamiento y evaluación

Procesando NVDA...<br>

[*********************100%***********************]  1 of 1 completed<br>
[NVDA] Episodio: 50/1000, Recompensa promedio: 272.92, Epsilon: 0.918<br>
[NVDA] Episodio: 100/1000, Recompensa promedio: 77.59, Epsilon: 0.835<br>
[NVDA] Episodio: 150/1000, Recompensa promedio: 69.76, Epsilon: 0.752<br>
[NVDA] Episodio: 200/1000, Recompensa promedio: 49.12, Epsilon: 0.669<br>
[NVDA] Episodio: 250/1000, Recompensa promedio: 81.01, Epsilon: 0.585<br>
[NVDA] Episodio: 300/1000, Recompensa promedio: 141.29, Epsilon: 0.502<br>
[NVDA] Episodio: 350/1000, Recompensa promedio: 176.86, Epsilon: 0.419<br>
[NVDA] Episodio: 400/1000, Recompensa promedio: 373.89, Epsilon: 0.336<br>
[NVDA] Episodio: 450/1000, Recompensa promedio: 344.34, Epsilon: 0.252<br>
[NVDA] Episodio: 500/1000, Recompensa promedio: 996.67, Epsilon: 0.169<br>
[NVDA] Episodio: 550/1000, Recompensa promedio: 633.73, Epsilon: 0.086<br>
[NVDA] Episodio: 600/1000, Recompensa promedio: 2093.12, Epsilon: 0.003<br>
[NVDA] Episodio: 650/1000, Recompensa promedio: 1620.05, Epsilon: 0.001<br>
[NVDA] Episodio: 700/1000, Recompensa promedio: 761.22, Epsilon: 0.001<br>
[NVDA] Episodio: 750/1000, Recompensa promedio: 1440.82, Epsilon: 0.001<br>
[NVDA] Episodio: 800/1000, Recompensa promedio: 2180.80, Epsilon: 0.001<br>
[NVDA] Episodio: 850/1000, Recompensa promedio: 1330.71, Epsilon: 0.001<br>
[NVDA] Episodio: 900/1000, Recompensa promedio: 1736.95, Epsilon: 0.001<br>
[NVDA] Episodio: 950/1000, Recompensa promedio: 1727.35, Epsilon: 0.001<br>
[NVDA] Episodio: 1000/1000, Recompensa promedio: 1270.90, Epsilon: 0.001<br>

[*********************100%***********************]  1 of 1 completed<br>
[NVDA] Evaluaci�n:<br>
Recompensa total: 4657.90<br>
D�as operados: 231<br>
Balance final: $85.30<br>
Acciones restantes: 106<br>
Valor total final: $14657.90<br>
Ganancias totales: $30398.09<br>
Perdidas totales: $25740.19<br>
Ganancia/P�rdida neta: $4657.90 (46.58%)<br>
Profit Factor: 1.18