import torch
from model import create_q_network
from trading_env import TradingEnv
from data_utils import download_data
from datetime import datetime, timedelta


def load_model(model_path, n_observations, n_actions):
    model = create_q_network(n_observations, n_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_latest_observation(symbol, window_size=10):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)

    df_close = download_data(symbol, start=start_date.strftime(
        "%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    env = TradingEnv(df=df_close, window_size=window_size)
    obs, _ = env.reset()
    return obs, env.observation_space.shape[0], env.action_space.n


def recommend_action(model, observation):
    state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
        action = q_values.max(1)[1].item()
    return action


if __name__ == "__main__":
    SYMBOL = "NVDA"
    MODEL_PATH = "dqn_model.pth"
    WINDOW_SIZE = 10

    obs, n_obs, n_act = get_latest_observation(SYMBOL, WINDOW_SIZE)
    model = load_model(MODEL_PATH, n_obs, n_act)
    action = recommend_action(model, obs)

    acciones = ["Mantener", "Comprar", "Vender"]
    print(f"Recomendacion para {SYMBOL}: {acciones[action]} (accion {action})")
