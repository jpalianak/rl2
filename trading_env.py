import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}  # opcional

    def __init__(self, df, window_size, initial_balance):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.num_features = 5  # ['Close', 'High', 'Low', 'Open', 'Volume']
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )

        # Tres acciones posibles: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_asset = self.initial_balance
        self.purchase_prices = []
        self.done = False
        return self._get_state(), {}

    def step(self, action):
        if self.current_step >= len(self.df):
            self.done = True
            return self._get_state(), 0.0, True, False, {}

        current_price = self.df['Close'].iloc[self.current_step]
        prev_total_asset = self.total_asset

        reward = 0.0

        # Comprar
        if action == 1 and self.balance >= current_price:
            self.shares_held += 1
            self.balance -= current_price
            self.purchase_prices.append(current_price)

        # Vender
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += current_price
            if self.purchase_prices:
                buy_price = self.purchase_prices.pop(0)
                realized_profit = current_price - buy_price
            else:
                realized_profit = -0.01 * current_price  # penalización
            reward += realized_profit

        # Penalización leve por no operar
        if action == 0:
            reward -= 0.001 * current_price

        self.current_step += 1

        terminated = self.current_step >= len(self.df)
        truncated = False

        self.total_asset = self.balance + self.shares_held * current_price

        # Cambio total en la riqueza: principal fuente de recompensa
        reward += self.total_asset - prev_total_asset

        obs = self._get_state()
        return obs, reward, terminated, truncated, {}

    def _get_state(self):
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        seq_data = self.df.iloc[start_idx:end_idx]
        seq_features = seq_data[['Close', 'High', 'Low', 'Open', 'Volume']]
        state = seq_features.values

        # Rellenar con ceros si la secuencia es más corta que window_size
        if len(state) < self.window_size:
            padding = np.zeros((self.window_size - len(state), state.shape[1]))
            state = np.vstack((padding, state))

        return state.astype(np.float32)
