import torch
import random
import torch.nn as nn
from torch.optim import SGD
from collections import namedtuple, deque

class QNN(nn.Module):
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
            torch.nn.Linear(512, 13)
        )

    def forward(self, x):
        return self.model(x)

def get_state_tensor(state, done=False):
    if done:
        return None
    return torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

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

def epsilon_greedy_action(epsilon, state, env, policy_net):
    if random.random() < epsilon:
        return torch.tensor([[env.sample_action()]], dtype=torch.long)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)

def update_target(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())

def update_epsilon(epsilon, decay):
    epsilon *= decay
    if epsilon < 0.05:
        epsilon = 0.05
    return epsilon

def update_model(buffer, policy_net, target_net, batch_size, discount_factor, optimizer):
    "The heart of DQN Algorithm"
    batch = buffer.sample()
    state_batch = torch.cat(batch.s)
    action_batch = torch.cat(batch.a)
    reward_batch = torch.cat(batch.r)
    next_state_batch = torch.cat([ns for ns in batch.ns if ns is not None])
    non_final_mask = torch.tensor(tuple(map(lambda ns: ns is not None, batch.ns)), dtype=torch.bool)

    # Compute Q-values for the current state and selected actions
    q_values = policy_net(state_batch).gather(1, action_batch)

    # Compute the maximum Q-values for the next states
    # use target network to estimate these Q-values to avoid biasing
    # the estimates with the values produced by the policy network
    next_q_values = torch.zeros(batch_size)
    with torch.no_grad():
        next_q_values[non_final_mask] = target_net(next_state_batch).max(1)[0]

    # Compute the target Q-values
    target_q_values = (next_q_values * discount_factor) + reward_batch

    # Compute MSE Loss between predicted and target Q-values
    criterion = torch.nn.MSELoss()
    loss = criterion(q_values, target_q_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def dqn(env, num_episodes, batch_size, gamma, ep_decay, epsilon,
        target_freq_update, memory_buffer_size, learning_rate, steps_cutoff, fixed_board,
        layers_sizes, train_action_value_freq_update):
    policy_net = QNN()
    target_net = QNN()
    update_target(policy_net, target_net)

    buffer= ReplayMemory(memory_buffer_size, batch_size)
    optimizer = SGD(policy_net.parameters(), lr=learning_rate)

    done_count = 0
    episodes_loss, episodes_rewards, episodes_steps = [], [], []

    for i in range(1, num_episodes+1):
        print("\nEpisode: ", i)
        state = get_state_tensor(env.reset())
        done = False
        episode_reward = 0
        episode_loss = 0
        num_steps = 1
        while not done and num_steps <= steps_cutoff:
            print(num_steps, end = " ")
            action = epsilon_greedy_action(epsilon, state, env, policy_net)
            epsilon = update_epsilon(epsilon, ep_decay)
            next_state, reward, done, info = env.step(action.item())
            next_state = get_state_tensor(next_state, done)
            episode_reward += reward
            reward = torch.tensor([reward])
            buffer.push(state, action, next_state, reward)
            state = next_state
            if len(buffer) >= batch_size and num_steps % train_action_value_freq_update == 0:
                loss = update_model(buffer, policy_net, target_net, batch_size, gamma, optimizer)
                episode_loss += loss
            num_steps += 1

        episodes_rewards.append(episode_reward)
        episodes_loss.append(episode_loss / num_steps)
        episodes_steps.append(num_steps)

        if i % target_freq_update == 0:
            update_target(policy_net, target_net)

        if done:
            done_count += 1

      # save the MidWay parameters to show
      # if num_episodes / 2 == i:
      #   self.MidWay = QNN().to(device)
      #   self.update_target(self.MidWay)

    def policy(s):
        '''
        This function gets a state and returns the preferable action.
        It does it by providing the state to the network and returning the action with maximal q-value
        (i.e. this is a greedy policy_net)
        '''
        state_tensor = get_state_tensor(s)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        a = q_values.max(1)[1].view(1, 1)
        return a.item()

    return policy, done_count, episodes_steps, episodes_rewards, episodes_loss