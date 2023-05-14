import gym
import random
import numpy as np
import matplotlib.pyplot as plt

class EnvWrapper(gym.Env):
    '''
    This wrapper enables environment customization:
    * simulating stochastic environment.
    * Enabling resetting the board to any initial state wanted.
    * reseting the reward.

    The implementation is based on the notebook:
    https://colab.research.google.com/drive/1ohrs6k0m17tPoQehdInUwwdj0g3H0vra?usp=sharing#scrollTo=XXqIDAd1SeMX

    but makes use of the Sokoban env
    '''

    def __init__(self, sok, compliance=0.9):
        '''
        :param sok: Sokoban warehouse env
        :param compliance: when the agent takes a certain action, this is the probability that the environment will
        '''
        self.env = sok
        plt.imshow(self.render())

        # creating stochactic transition mapping
        n = self.env.action_space.n
        self.action_list = list(range(n))

        self.action_dist = {}
        for action in self.action_list:
            dist = [(1 - compliance) / (n - 1) for i in range(n)]
            dist[action] = compliance
            self.action_dist[action] = dist

        # creating a dictionary for custom rewards
        self.rewards_dict = {}

    def step(self, action):
        dist = self.action_dist[action]
        chosen_action = random.choices(self.action_list, dist)[0]

        state, reward, done, info = self.env.step(chosen_action)
        state = tuple(state)

        if done:
            if self.goal_reward is not None:
                reward = self.goal_reward
        else:
            if self.step_reward is not None:
                reward = self.step_reward

        if state in self.states_rewards: # this is a state with custom reward
            reward = self.states_rewards[state]

        return state, reward, done, info

    def reset(self, state=None):
        if state is None:
            state = self.env.reset()
            return tuple(map(int, state))
        else:
            _ = self.env.reset()
            self.env.state = np.array(state)
            self.env.maze_view._MazeView2D__robot = np.array(state) # Note: I do not know why it is not enabled by platform, but as it is needed I added it
            self.env.maze_view._MazeView2D__draw_robot(transparency=255)
            return state

    def render(self, mode='rgb_array'):
        return self.env.render(mode)

# for testing
if __name__ == '__main__':
    from soko_pap import PushAndPullSokobanEnv

    sok = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1, max_steps=500)
    env = EnvWrapper(sok)
    print()

