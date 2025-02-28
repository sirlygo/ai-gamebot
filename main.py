import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Monkey patch NumPy to avoid np.bool8 issues
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import random
from collections import deque
import gym
from dueling_dqn import DuelingDQN


class GameBot:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.update_target_every = 10

        self.model = DuelingDQN(state_size, action_size)
        self.target_model = DuelingDQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.train_step = 0

    def remember(self, state, action, reward, next_state, done):
        priority = abs(reward) + 1e-5
        self.memory.append((priority, state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = sorted(random.sample(self.memory, self.batch_size), key=lambda x: x[0], reverse=True)
        states, targets = [], []

        for _, state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state).clone().detach()

            # Ensure proper indexing
            if target_f.numel() > action:
                target_f.view(-1)[action] = target

            states.append(state.squeeze(0))
            targets.append(target_f.squeeze(0))

        states = torch.stack(states)
        targets = torch.stack(targets)

        self.optimizer.zero_grad()
        loss = self.criterion(self.model(states), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def train(self, env, episodes=500):
        for e in range(episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state = np.reshape(state, [1, self.state_size])
            done = False
            total_reward = 0

            while not done:
                action = self.act(state)
                step_result = env.step(action)
                if len(step_result) == 5:  # Handle gym versions that return (obs, reward, terminated, truncated, info)
                    next_state, reward, terminated, truncated, _ = step_result
                    done = bool(terminated) or bool(truncated)
                else:
                    next_state, reward, done, _ = step_result

                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.replay()

            print(f"Episode {e + 1}/{episodes}, Score: {total_reward}, Epsilon: {self.epsilon:.4f}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    bot = GameBot(state_size, action_size)
    bot.train(env, episodes=500)
