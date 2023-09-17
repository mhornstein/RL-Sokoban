import torch
from collections import namedtuple, deque
import random
import numpy as np
import copy
import torch.nn as nn
from torch.optim import SGD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 10000

class QNN(nn.Module):
  "DNN to approximate the Q-function"
  def __init__(self):
    super(QNN, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(3, 32, 5),
      torch.nn.MaxPool2d((2,2)),
      torch.nn.ReLU(),
      torch.nn.Conv2d(32,64, 5),
      torch.nn.MaxPool2d((2,2)),
      torch.nn.ReLU(),
      torch.nn.Conv2d(64,128, 5),
      torch.nn.MaxPool2d((2,2)),
      torch.nn.ReLU(),
      torch.nn.Flatten(),
      torch.nn.Linear(12800, 512),
      torch.nn.ReLU(),
      torch.nn.Linear(512, 13))

  def forward(self, x):
    return self.model(x)

class ReplayMemory(object):
  "stores past experiences as fifo"
  def __init__(self, capacity, batch_size):
    self.batch_size = batch_size
    self.memory_baffer = deque([], maxlen=capacity)
    self.transition = namedtuple('T', ('s', 'a', 'ns', 'r'))

  def __len__(self):
        return len(self.memory_baffer)

  def push(self, state, action, next_state, reward):
    transition = (state, action, next_state, reward)
    self.memory_baffer.append(transition)

  def sample(self):
    batch = random.sample(self.memory_baffer, self.batch_size)
    return self.transition(*zip(*batch))

class DQN():

  def __init__(self, env, epsilon, decay_rate, discount_factor, change_reward=True):
    self.original_env = env
    self.epsilon = epsilon
    self.decay = decay_rate
    self.discount_factor = discount_factor
    self.change_reward = change_reward
    self.policy = QNN().to(device)
    self.target = QNN().to(device)
    self.buffer= ReplayMemory(REPLAY_MEMORY_SIZE, BATCH_SIZE)
    self.optimizer = SGD(self.policy.parameters(), lr=0.0001)
    self.update_target(self.target)

  def update_target(self, target):
    target.load_state_dict(self.policy.state_dict())

  def get_state_tensor(self, state, done=False):
    if done:
      return None
    return torch.tensor(state, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

  def update_epsilon(self):
    self.epsilon *= self.decay
    if self.epsilon < 0.05:
      self.epsilon = 0.05

  def epsilon_greedy_action(self, state):
    if random.random() < self.epsilon:
      return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)
    else:
      with torch.no_grad():
        return self.policy(state).max(1)[1].view(1, 1)

  def calculate_reward(self):
    reward = 0
    if self.change_reward:
      room_state = self.env.room_state
      room_fixed = self.env.room_fixed
      box_x, box_y= np.where(room_state == 4)
      target_x, target_y = np.where(room_fixed == 2)
      for i in range(len(box_x)):
        for j in range(len(target_x)):
          # Calculate distance between a box and a target
          reward -= abs(box_x[i] - target_x[j]) + abs(box_y[i] - target_y[j])
    return float(reward)

  def update_model(self):
    "The heart of DQN Algorithm"
    batch = self.buffer.sample()
    state_batch = torch.cat(batch.s)
    action_batch = torch.cat(batch.a)
    reward_batch = torch.cat(batch.r)
    next_state_batch = torch.cat([ns for ns in batch.ns if ns is not None])
    non_final_mask = torch.tensor(tuple(map(lambda ns: ns is not None, batch.ns)), device=device, dtype=torch.bool)

    # Compute Q-values for the current state and selected actions
    q_values = self.policy(state_batch).gather(1, action_batch)

    # Compute the maximum Q-values for the next states
    # use target network to estimate these Q-values to avoid biasing
    # the estimates with the values produced by the policy network
    next_q_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
      next_q_values[non_final_mask] = self.target(next_state_batch).max(1)[0]

    # Compute the target Q-values
    target_q_values = (next_q_values * self.discount_factor) + reward_batch

    # Compute MSE Loss between predicted and target Q-values
    criterion = torch.nn.MSELoss()
    loss = criterion(q_values, target_q_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()

  def train(self, num_episodes):
    losses, rewards = [], []
    for i in range(num_episodes):
      print("\nEpisode: ", i)
      self.env = copy.deepcopy(self.original_env)
      state = self.get_state_tensor(self.env.get_image(mode='rgb_array'))
      done = False
      episode_reward = 0
      episode_loss = 0
      num_steps = 0
      while not done:
        print(num_steps, end = " ")
        action = self.epsilon_greedy_action(state)
        self.update_epsilon()
        next_state, reward, done, _ = self.env.step(action.item())
        next_state = self.get_state_tensor(next_state, done)
        reward += self.calculate_reward()
        episode_reward += reward
        reward = torch.tensor([reward], device=device)
        self.buffer.push(state, action, next_state, reward)
        state = next_state
        if len(self.buffer) >= BATCH_SIZE:
          loss = self.update_model()
          episode_loss += loss
          num_steps += 1

      rewards.append(episode_reward)
      losses.append(episode_loss / num_steps)

      # Once in 10 episodes - update target network
      if i % 10 == 9:
        self.update_target(self.target)

      # save the MidWay parameters to show
      if num_episodes / 2 == i:
        self.MidWay = QNN().to(device)
        self.update_target(self.MidWay)

    return losses, rewards

import matplotlib.pyplot as plt
from soko_pap import PushAndPullSokobanEnv
random.seed(2)
sok = PushAndPullSokobanEnv(dim_room=(7, 7),num_boxes=1 ,max_steps=500)
dqn = DQN(sok, 1.0, 0.999, 0.999, change_reward=True)
train_losses, train_rewards = dqn.train(100)
plt.plot(range(len(train_losses)), train_losses, label=f"with reward addition")
# plt.plot(range(len(train_rewards)), train_rewards, label=f"with reward addition")
