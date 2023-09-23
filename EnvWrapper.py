import gym
import matplotlib.pyplot as plt
import random
import numpy as np
from soko_pap import PushAndPullSokobanEnv
import copy

class EnvWrapper(gym.Env):
    '''
    This wrapper enables environment customization:
    * simulating stochastic environment.

    The implementation is based on the notebook:
    https://colab.research.google.com/drive/1ohrs6k0m17tPoQehdInUwwdj0g3H0vra?usp=sharing#scrollTo=XXqIDAd1SeMX

    but makes use of the Sokoban env instead.
    '''

    def __init__(self, num_boxes=1, compliance=0.9):
        '''
        :param num_boxes: number of boxes in the sokoban env
        :param compliance: when the agent takes a certain action, this is the probability that the environment will
        '''
        self.num_boxes = num_boxes
        self.compliance = compliance

        self.source_env = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=num_boxes, max_steps=10000) # todo UPDATE
        self.reset()

        # creating stochactic transition mapping
        n = self.env.action_space.n
        self.action_list = list(range(n))

        self.action_dist = {}
        for action in self.action_list:
            dist = [(1 - compliance) / (n - 1) for i in range(n)]
            dist[action] = compliance
            self.action_dist[action] = dist

        self.render()

    def step(self, action):
        dist = self.action_dist[action]
        chosen_action = random.choices(self.action_list, dist)[0]

        state, reward, done, info = self.env.step(chosen_action)
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

    def render(self, mode='tiny_rgb_array'):
        plt.imshow(self.env.render(mode))

    def sample_action(self):
        return self.env.action_space.sample()

    def get_states_dim(self):
        current_state = self.get_current_state()
        return current_state.shape

    def action_space(self):
        return self.env.action_space

    def calc_reward2(self):
        '''
        Calculate the Manhattan distance between the positions of boxes (4) in room_state
        and their corresponding target positions (2) in room_fixed.

        The reward is the negative sum of these Manhattan distances. It is negative
        because the agent is penalized more when the boxes are farther from their targets.
        '''
        reward = 0
        room_state = self.env.room_state
        room_fixed = self.env.room_fixed
        box_positions = np.argwhere(room_state == 4)
        target_positions = np.argwhere(room_fixed == 2)
        for box_position in box_positions:
            for target_position in target_positions:
                manhattan_distance = np.abs(box_position - target_position).sum()
                reward -= manhattan_distance
        return float(reward)

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
        target_x, target_y = np.where(room_fixed == 2)

        for i in range(len(box_x)):
            for j in range(len(target_x)):
                dist = abs(box_x[i] - target_x[j]) + abs(box_y[i] - target_y[j])
                reward -= dist

        reward2 = self.calc_reward2()
        assert reward2 == reward

        return float(reward)

    def __str__(self):
        return f'<num of boxes: {self.num_boxes}, compliance: {self.compliance}>'

# for testing
if __name__ == '__main__': # TODO finish here
    from soko_pap import PushAndPullSokobanEnv
    import random

    random.seed(2)

    env = EnvWrapper(num_boxes=1, compliance=1)
    env.render()

    action = 8 # Move right
    state, reward, done, info = env.step(action)
    # print(state, reward, done, info)
    env.render()

    action = env.sample_action()
    state, reward, done, info = env.step(action)
    # print(state, reward, done, info)
    env.render()
    env.reset()
    env.render()

    env.change_board()
    env.render()

    input()

