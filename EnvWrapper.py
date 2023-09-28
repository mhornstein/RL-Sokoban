import gym
import matplotlib.pyplot as plt
import numpy as np
import copy

class EnvWrapper(gym.Env):
    '''
    This wrapper enables environment customization (e.g. simulating stochastic environment).

    The implementation is based on the notebook:
    https://colab.research.google.com/drive/1ohrs6k0m17tPoQehdInUwwdj0g3H0vra?usp=sharing#scrollTo=XXqIDAd1SeMX

    but makes use of the Sokoban env instead.
    '''

    def __init__(self, sok, compliance=1, use_distance_reward=True):
        '''
        :param sok: the sokoban env
        :param compliance: when the agent takes a certain action, this is the probability that the environment will comply.
        :param use_distance_reward: set to True to use distance-based reward. Set to False to use the default sokoban setting
        '''
        self.compliance = compliance
        self.use_distance_reward = use_distance_reward

        self.source_env = sok
        self.reset()

        # creating stochactic transition mapping
        n = self.env.action_space.n
        self.action_list = list(range(n))

        self.action_dist = {}
        for action in self.action_list:
            dist = [(1 - compliance) / (n - 1) for i in range(n)]
            dist[action] = compliance
            self.action_dist[action] = dist

    def step(self, action):
        dist = self.action_dist[action]
        chosen_action = random.choices(self.action_list, dist)[0]

        state, reward, done, info = self.env.step(chosen_action)
        if self.use_distance_reward:
            reward += self.calc_reward()
        state = self.get_current_state()

        return state, reward, done, info

    def reset(self): # open gym reset changes the board. override, as we do not want to change the board upon reset
        self.env = copy.deepcopy(self.source_env)
        return self.get_current_state()

    def change_board(self):
        self.source_env.reset()
        self.reset()

    def get_current_state(self):
        rgb_display = self.env.get_image(mode='rgb_array')
        return rgb_display

    def render(self, mode='rgb_array'):
        plt.imshow(self.env.render(mode))

    def sample_action(self):
        return self.env.action_space.sample()

    def get_states_dim(self):
        current_state = self.get_current_state()
        return current_state.shape

    def action_space(self):
        return self.env.action_space

    def calc_reward(self):
        '''
        Calculate the Manhattan distance between the positions of boxes (4) in room_state
        and their corresponding target positions (2) in room_fixed.

        The reward is the negative sum of these Manhattan distances. It is negative
        because the agent is penalized more when the boxes are farther from their targets.
        '''
        reward = 0

        room_state = self.env.room_state
        box_x, box_y = np.where(room_state == 4)

        room_fixed = self.env.room_fixed
        box_target_x, box_target_y = np.where(room_fixed == 2)

        for i in range(len(box_x)):
            for j in range(len(box_target_x)):
                dist = abs(box_x[i] - box_target_x[j]) + abs(box_y[i] - box_target_y[j])
                reward -= dist

        return float(reward)