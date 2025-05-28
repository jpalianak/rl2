import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}  # opcional

    def __init__(self, df, window_size=10, initial_balance=10_000):
        super().__init__()  # buena práctica para entornos personalizados
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size + 2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # para compatibilidad con Gymnasium
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_asset = self.initial_balance
        self.done = False
        return self._get_state(), {}

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        reward = 0

        if action == 1 and self.balance >= current_price:
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += current_price
            reward = 1

        self.current_step += 1
        terminated = self.current_step >= len(self.df)
        truncated = False
        self.total_asset = self.balance + self.shares_held * current_price
        reward += (self.total_asset - self.initial_balance) / \
            self.initial_balance

        obs = self._get_state()
        return obs, reward, terminated, truncated, {}

    def _get_state(self):
        window_prices = self.df['Close'].iloc[self.current_step -
                                              self.window_size:self.current_step].values
        window_prices = window_prices.reshape(-1)  # asegura 1D

        state = np.concatenate([
            window_prices,
            np.array([self.balance, self.shares_held], dtype=np.float32)
        ])

        return state.astype(np.float32)
