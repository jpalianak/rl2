import torch
import matplotlib.pyplot as plt
from trading_env import TradingEnv
from data_utils import download_data
from train import load_model


def evaluate_model(ticker, model, df_eval, initial_balance=10_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ticker}] Evaluando modelo en dispositivo: {device}")

    model.to(device)
    model.eval()

    env = TradingEnv(df=df_eval, window_size=20,
                     initial_balance=initial_balance)
    obs, info = env.reset()
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    done = False
    portfolio_values = [env.total_asset]
    daily_actions = []
    rewards = []
    episode_reward = 0
    steps = 0

    while not done:
        with torch.no_grad():
            q_values = model(state)
            action = q_values.max(1)[1].item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        rewards.append(reward)
        steps += 1
        daily_actions.append(action)
        portfolio_values.append(env.total_asset)

        if not done:
            state = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(device)

    final_balance = env.balance
    final_asset = env.total_asset
    shares_final = env.shares_held

    ganancia_neta = final_asset - initial_balance
    porcentaje_ganancia = (ganancia_neta / initial_balance) * 100

    # Calcular Profit Factor
    total_ganancias = sum(r for r in rewards if r > 0)
    total_perdidas = -sum(r for r in rewards if r < 0)
    profit_factor = total_ganancias / \
        total_perdidas if total_perdidas > 0 else float('inf')

    print(f"\n[{ticker}] Evaluación:")
    print(f"Recompensa total: {episode_reward:.2f}")
    print(f"Días operados: {steps}")
    print(f"Balance final: ${final_balance:.2f}")
    print(f"Acciones restantes: {shares_final}")
    print(f"Valor total final: ${final_asset:.2f}")
    print(f"Ganancias totales: ${total_ganancias:.2f}")
    print(f"Perdidas totales: ${total_perdidas:.2f}")
    print(
        f"Ganancia/Pérdida neta: ${ganancia_neta:.2f} ({porcentaje_ganancia:.2f}%)")
    print(f"Profit Factor: {profit_factor:.2f}")

    # Gráfico evolución portafolio
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_values)
    plt.title(f"Evolución del portafolio {ticker}")
    plt.xlabel("Día")
    plt.ylabel("Valor total ($)")
    plt.grid()
    plt.show()

    env.close()
    return daily_actions, portfolio_values


if __name__ == "__main__":
    # Parámetros
    ticker = "NVDA"
    model_path = f"dqn_model_{ticker}.pth"
    df_eval = download_data(ticker, "2024-01-01", "2024-12-31")

    # Crear red y cargar pesos
    env_tmp = TradingEnv(df=df_eval, window_size=20, initial_balance=10_000)
    input_dim = 5
    n_actions = env_tmp.action_space.n  # type: ignore
    model = load_model(ticker, input_dim, n_actions)
    evaluate_model(ticker, model, df_eval)
