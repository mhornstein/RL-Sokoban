import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

class DQN_Net(nn.Module):
    def __init__(self, input_shape, output_shape, layers_sizes):
        super(DQN_Net, self).__init__()
        input_size = input_shape[0] * input_shape[1]
        layer_sizes = [input_size] + layers_sizes + [output_shape]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2: # Do not add relu layer after the last linear layer - this is where the softmax is
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        size = x.size()
        flattened_size = size[0], -1
        return self.model(x.view(flattened_size))

def train_action_value_network(action_value_net, target_net, batch, gamma, criterion, optimizer):
    states, actions, rewards, next_states, dones = batch

    '''
    In this part, we Convert all the matrices
    Only this part was inspired by the ATARI DQN implamantation paper: https://github.com/BY571/DQN-Atari-Agents/blob/master/Agents/dqn_agent.py
    '''
    states = torch.stack([torch.tensor(arr, dtype=torch.float32) for arr in states], dim=0)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.stack([torch.tensor(arr, dtype=torch.float32) for arr in next_states], dim=0)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    q_values = action_value_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)

    '''
    This is the matrix-way to write the following for a single state:
    if done:
            target = r
        else:
            target = r + gamma * np.amax(target_net.predict(np.array([s_tag,]))
    '''
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    loss = criterion(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def dqn(env, num_episodes, batch_size, gamma, ep_decay, epsilon,
        target_freq_update, memory_buffer_size, learning_rate, steps_cutoff, fixed_board,
        layers_sizes, train_action_value_freq_update):
    net_performance = []

    done_count = 0
    episodes_steps = []
    episodes_rewards = []

    # create a memory buffer
    memory_buffer = deque(maxlen=memory_buffer_size)

    # create both agent and target nets
    states_dim = env.get_states_dim()
    actions_dim = env.action_space().n

    action_value_net = DQN_Net(states_dim, actions_dim, layers_sizes)
    target_net = DQN_Net(states_dim, actions_dim, layers_sizes)
    target_net.load_state_dict(action_value_net.state_dict())

    optimizer = optim.Adam(action_value_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Start running episodes
    for ep in range(1, num_episodes+1):
        # print(f'running ep: {ep}. Steps: ', end ='')
        if not fixed_board:
            env.change_board()
        s = env.reset()
        env.render()

        done = False
        steps_count = 1
        reward_sum = 0

        while not done and steps_count < steps_cutoff:
            # print(f'{steps_count}', end= ' ')
            # Step 1: Choose an action a based on current policy (e.g. 𝜀 − 𝑔𝑟𝑒𝑒𝑑𝑦))
            if np.random.rand() <= epsilon:
                a = env.sample_action()
            else:
                state = torch.FloatTensor(s).unsqueeze(0)
                with torch.no_grad():
                    q_values = action_value_net(state)
                a = torch.argmax(q_values).item()

            # Step 2: You get a reward r. You are now in state s’
            s_tag, r, done, info = env.step(a)

            # Step 3: save the result of the step in memory
            memory_buffer.append((s, a, r, s_tag, done))

            # Step 4: train the agent network
            if len(memory_buffer) > batch_size and steps_count % train_action_value_freq_update == 0:
                batch = zip(*random.sample(memory_buffer, batch_size))
                loss = train_action_value_network(action_value_net, target_net, batch, gamma, criterion, optimizer)
                net_performance.append({
                    'ep': ep,
                    'steps_count': steps_count,
                    'loss': loss
                })

            # Step 5: update to the new state
            env.render()
            s = s_tag
            reward_sum += r
            steps_count += 1

        epsilon = max(epsilon*ep_decay, 0.05)  # Decay exploration rate

        if ep % target_freq_update == 0:
            target_net.load_state_dict(action_value_net.state_dict())

        if done:
            done_count += 1
        episodes_steps.append(steps_count)
        episodes_rewards.append(reward_sum)

    # Last - create a policy and return it along with all other measurements
    def policy(s):
        '''
        This function gets a state and returns the preferable action.
        It does it by providing the state to the network and returning the action with maximal q-value
        (i.e. this is a greedy policy)
        '''
        state = torch.FloatTensor(s).unsqueeze(0)
        with torch.no_grad():
            q_values = action_value_net(state)
        a = torch.argmax(q_values).item()
        return a

    return policy, done_count, episodes_steps, episodes_rewards, net_performance
