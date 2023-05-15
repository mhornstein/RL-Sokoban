import numpy as np
import random
from collections import deque
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Define the Deep Q-Network (DQN) model
def build_dqn(input_shape, output_size):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Initialize the DQN agent
def initialize_agent(input_shape, output_size):
    return build_dqn(input_shape, output_size)

# Define the DQN algorithm with replay experience
def dqn_algorithm(env, agent, target_agent, episodes, batch_size, gamma, epsilon_decay, target_update_freq, replay_memory_size):
    replay_memory = deque(maxlen=replay_memory_size)

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, -1))
        done = False
        total_reward = 0

        while not done:
            # Exploration-exploitation trade-off
            if np.random.rand() <= epsilon_decay:
                action = env.action_space.sample()
            else:
                q_values = agent.predict(state)
                action = np.argmax(q_values)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, (1, -1))
            total_reward += reward

            # Store the current transition in replay memory
            replay_memory.append((state, action, reward, next_state, done))

            # Sample a random minibatch from replay memory
            minibatch = random.sample(replay_memory, batch_size)

            # Update the Q-values using the minibatch
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target += gamma * np.amax(target_agent.predict(next_state))

                target_q_values = agent.predict(state)[0]
                target_q_values[action] = target

                # Train the agent using the target Q-values
                agent.fit(state, np.expand_dims(target_q_values, axis=0), batch_size=batch_size, verbose=0)

            state = next_state

            # Update the target network every c steps
            if episode % target_update_freq == 0:
                target_agent.set_weights(agent.get_weights())

        # Print the total reward achieved in each episode
        print("Episode:", episode+1, "Total Reward:", total_reward)
