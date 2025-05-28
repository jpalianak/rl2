import torch
import matplotlib.pyplot as plt
from trading_env import TradingEnv


def evaluate_model(ticker, model, df_eval, initial_balance=10_000):
    env = TradingEnv(df=df_eval, window_size=10,
                     initial_balance=initial_balance)
    obs, info = env.reset()
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    done = False
    portfolio_values = [env.total_asset]
    daily_actions = []
    episode_reward = 0
    steps = 0

    while not done:
        with torch.no_grad():
            q_values = model(state)
            action = q_values.max(1)[1].item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        steps += 1
        daily_actions.append(action)
        portfolio_values.append(env.total_asset)

        if not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    final_balance = env.balance
    final_asset = env.total_asset
    shares_final = env.shares_held

    ganancia_neta = final_asset - initial_balance
    porcentaje_ganancia = (ganancia_neta / initial_balance) * 100

    print(f"\n[{ticker}] Evaluación:")
    print(f"Recompensa total: {episode_reward:.2f}")
    print(f"Días operados: {steps}")
    print(f"Balance final: ${final_balance:.2f}")
    print(f"Acciones restantes: {shares_final}")
    print(f"Valor total final: ${final_asset:.2f}")
    print(
        f"Ganancia/Pérdida neta: ${ganancia_neta:.2f} ({porcentaje_ganancia:.2f}%)")

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
