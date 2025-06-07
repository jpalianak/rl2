from data_utils import download_data
from train import train_dqn_for_ticker, save_model, load_model
from evaluate import evaluate_model
from trading_env import TradingEnv

tickers = ["NVDA"]

for ticker in tickers:
    print(f"Procesando {ticker}...")

    df_train = download_data(ticker, "2020-01-01", "2023-12-31")
    model, rewards_hist = train_dqn_for_ticker(
        ticker, df_train, window_size=20, initial_balance=10_000)
    save_model(model, ticker)

    df_eval = download_data(ticker, "2024-01-01", "2024-12-31")
    env_tmp = TradingEnv(df=df_eval, window_size=20, initial_balance=10_000)
    obs, _ = env_tmp.reset()
    input_dim = 5  # la cantidad de features
    n_actions = env_tmp.action_space.n  # type: ignore
    model = load_model(ticker, input_dim, n_actions)
    evaluate_model(ticker, model, df_eval)
