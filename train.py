import torch
import torch.optim as optim
import random
import collections
import numpy as np
from model import LSTM_QNetwork
from trading_env import TradingEnv
import config

Transition = collections.namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done')
)


def train_dqn_for_ticker(
    ticker, df_train, window_size, initial_balance, max_episodes=config.MAX_EPISODES
):
    env = TradingEnv(df=df_train, window_size=window_size,
                     initial_balance=initial_balance)
    n_actions = env.action_space.n  # type: ignore
    input_dim = 5

    q_network = LSTM_QNetwork(input_dim=input_dim, output_dim=n_actions)
    target_network = LSTM_QNetwork(input_dim=input_dim, output_dim=n_actions)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=config.LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    replay_buffer = collections.deque(maxlen=config.BUFFER_SIZE)
    global_step = 0
    epsilon = config.EPSILON_START
    episode_rewards_history = []

    def update_target_network():
        target_network.load_state_dict(q_network.state_dict())

    for episode in range(max_episodes):
        obs, info = env.reset()
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        episode_reward = 0

        for step in range(config.MAX_STEPS_PER_EPISODE):
            global_step += 1

            if random.random() < epsilon:
                action = torch.tensor(
                    [[env.action_space.sample()]], dtype=torch.long
                )
            else:
                with torch.no_grad():
                    q_values = q_network(state)
                    action = q_values.max(1)[1].view(1, 1)

            next_obs, reward, terminated, truncated, info = env.step(
                action.item()
            )
            done = terminated or truncated
            episode_reward += reward

            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            next_state = (
                None
                if done
                else torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            )
            done_tensor = torch.tensor([done], dtype=torch.float32)

            replay_buffer.append(
                Transition(state, action, reward_tensor,
                           next_state, done_tensor)
            )
            state = next_state

            if len(replay_buffer) >= config.BATCH_SIZE:
                transitions = random.sample(replay_buffer, config.BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(
                    tuple(s is not None for s in batch.next_state),
                    dtype=torch.bool,
                )
                non_final_next_states = torch.cat(
                    [s for s in batch.next_state if s is not None], dim=0
                )

                state_batch = torch.cat(batch.state, dim=0)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                state_action_values = q_network(
                    state_batch).gather(1, action_batch).squeeze(1)

                next_state_values = torch.zeros(config.BATCH_SIZE)
                with torch.no_grad():
                    online_next_q_values = q_network(non_final_next_states)
                    online_best_next_actions = online_next_q_values.max(1)[
                        1].unsqueeze(1)
                    target_next_q_values = target_network(
                        non_final_next_states)
                    selected_target_next_q_values = target_next_q_values.gather(
                        1, online_best_next_actions
                    ).squeeze(1)
                    next_state_values[non_final_mask] = selected_target_next_q_values

                expected_state_action_values = reward_batch + (
                    config.GAMMA * next_state_values
                )

                loss = loss_fn(state_action_values,
                               expected_state_action_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % config.TARGET_UPDATE_FREQ == 0:
                update_target_network()

            if done:
                break

        episode_rewards_history.append(episode_reward)
        epsilon = max(
            config.EPSILON_END,
            config.EPSILON_START
            - (episode / config.EPSILON_DECAY_EPISODES)
            * (config.EPSILON_START - config.EPSILON_END),
        )

        if (episode + 1) % config.PRINT_EVERY == 0:
            avg_reward = np.mean(episode_rewards_history[-config.PRINT_EVERY:])
            print(
                f'[{ticker}] Episodio: {episode + 1}/{max_episodes}, '
                f'Recompensa promedio: {avg_reward:.2f}, Epsilon: {epsilon:.3f}'
            )

    env.close()
    return q_network, episode_rewards_history


def save_model(model, ticker):
    torch.save(model.state_dict(), f"dqn_model_{ticker}.pth")


def load_model(ticker, input_dim, output_dim):
    model = LSTM_QNetwork(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(f"dqn_model_{ticker}.pth"))
    model.eval()
    return model
