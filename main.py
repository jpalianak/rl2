from data_utils import download_data
from train import train_dqn_for_ticker, save_model, load_model
from evaluate import evaluate_model

tickers = ["NVDA"]  # Podés modificar aquí

for ticker in tickers:
    print(f"Procesando {ticker}...")

    df_train = download_data(ticker, "2020-01-01", "2023-12-31")
    model, rewards_hist = train_dqn_for_ticker(ticker, df_train)
    save_model(model, ticker)

    df_eval = download_data(ticker, "2024-01-01", "2024-12-31")

    n_obs = model[0].in_features
    n_act = model[-1].out_features
    loaded_model = load_model(ticker, n_obs, n_act)

    evaluate_model(ticker, loaded_model, df_eval)
