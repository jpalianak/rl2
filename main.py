from data_utils import download_data
from train import train_dqn_for_ticker, save_model, load_model
from evaluate import evaluate_model
from trading_env import TradingEnv
from tqdm import tqdm
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import numpy as np
import os

tickers = ["NVDA"]

for ticker in tqdm(tickers, desc="Entrenando el agente"):

    print(f"Procesando {ticker}...")

    df_train = download_data(ticker, "2020-01-01", "2023-12-31")
    model, rewards_hist = train_dqn_for_ticker(
        ticker, df_train, window_size=20, initial_balance=10_000)
    save_model(model, ticker)

    # ----- Guardar y graficar curva de convergencia -----
    os.makedirs("resultados", exist_ok=True)
    np.save(f"resultados/{ticker}_rewards.npy", np.array(rewards_hist))

    def plot_convergence_curve(rewards, smoothing_window=50, ticker="Ticker"):
        plt.figure(figsize=(10, 5))
        if len(rewards) >= smoothing_window:
            smoothed = np.convolve(rewards, np.ones(
                smoothing_window)/smoothing_window, mode='valid')
            plt.plot(smoothed, label=f"Media móvil ({smoothing_window})")
        else:
            plt.plot(rewards, label="Reward por episodio")

        plt.title(f"Curva de convergencia del agente ({ticker})")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"resultados/{ticker}_convergencia.png")
        plt.show()

    plot_convergence_curve(rewards_hist, smoothing_window=50, ticker=ticker)
    # -----------------------------------------------------

    df_eval = download_data(ticker, "2024-01-01", "2024-12-31")
    env_tmp = TradingEnv(df=df_eval, window_size=20, initial_balance=10_000)
    obs, _ = env_tmp.reset()
    input_dim = 5  # la cantidad de features
    n_actions = env_tmp.action_space.n  # type: ignore
    model = load_model(ticker, input_dim, n_actions)
    evaluate_model(ticker, model, df_eval)
