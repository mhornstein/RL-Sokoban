import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_dqn(input_shape, output_shape, learning_rate=0.001):
    '''
    Creates and return a dqn network with the given input and output shapes, and learning-rate
    '''
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

def train_action_value_network(action_value_net, target_net, batch, gamma):
    states = []
    targets = []

    for s, a, r, s_tag, done in batch:
        if done:
            target = r
        else:
            target = r + gamma * np.amax(target_net.predict(s_tag, verbose = 0))

        target_q_values = action_value_net.predict(s, verbose = 0)[0]
        target_q_values[a] = target

        states.append(s.T)
        targets.append(np.expand_dims(target_q_values, axis=0).T)

    action_value_net.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

def dqn(env, num_episodes, batch_size, gamma, ep_decay, epsilon,
        target_freq_update, memory_buffer_size, learning_rate, steps_cutoff):

    done_count = 0
    episodes_steps = []
    episodes_rewards = []

    # create a memory buffer
    memory_buffer = deque(maxlen=memory_buffer_size)

    # create both agent and target nets
    states_dim = env.get_states_dim()
    network_input_shape = (np.prod(states_dim),)
    actions_dim = env.action_space().n
    network_output_shape = actions_dim
    action_value_net = build_dqn(network_input_shape, network_output_shape, learning_rate)
    target_net = build_dqn(network_input_shape, network_output_shape, learning_rate)
    target_net.set_weights(action_value_net.get_weights())

    # Start running episodes
    for ep in range(1, num_episodes+1):
        print(f'running ep: {ep}. Steps: ', end ='')
        s = env.reset()

        done = False
        steps_count = 0
        reward_sum = 0

        while not done and steps_count < steps_cutoff:
            print(f'{steps_count}', end= ' ')
            # Step 1: Choose an action a based on current policy (e.g. 𝜀 − 𝑔𝑟𝑒𝑒𝑑𝑦))
            if np.random.rand() <= epsilon:
                a = env.sample_action()
            else:
                q_values = action_value_net.predict(s)
                a = np.argmax(q_values)

            # Step 2: You get a reward r. You are now in state s’
            s_tag, r, done, info = env.step(a)

            # Step 3: save the result of the step in memory
            memory_buffer.append((np.reshape(s, (1, -1)),
                                  a,
                                  r,
                                  np.reshape(s_tag, (1, -1)),
                                  done))

            # Step 4: train the agent network
            if len(memory_buffer) > batch_size:
                batch = random.sample(memory_buffer, batch_size)
                train_action_value_network(action_value_net, target_net, batch, gamma)

            # Step 5: Every target_update_freq update the target net
            if ep % target_freq_update == 0:
                target_net.set_weights(action_value_net.get_weights())

            # Step 6: update to the new state
            s = s_tag
            reward_sum += r
            steps_count += 1

        epsilon *= ep_decay  # Decay exploration rate

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
        q_values = action_value_net.predict(np.reshape(s, (1, -1)), verbose=0)
        action = np.argmax(q_values)
        return action

    return policy, done_count, episodes_steps, episodes_rewards
