import gym
import matplotlib.pyplot as plt
import numpy as np
import copy

class EnvWrapper(gym.Env):
    '''
    This wrapper enables environment customization.

    The implementation is based on the notebook:
    https://colab.research.google.com/drive/1ohrs6k0m17tPoQehdInUwwdj0g3H0vra?usp=sharing#scrollTo=XXqIDAd1SeMX

    but makes use of the Sokoban env instead.
    '''

    def __init__(self, sok, use_distance_reward=True):
        '''
        :param sok: the sokoban env
        :param use_distance_reward: set to True to use distance-based reward. Set to False to use the default sokoban setting
        '''
        self.source_env = sok
        self.use_distance_reward = use_distance_reward
        self.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        box_target_distance = self.calc_box_target_distance()

        # override state with rgb representation
        state = self.get_current_state()

        # override reward with target distance metric if required
        if self.use_distance_reward:
            reward -= box_target_distance

        # override done to be True iff all targets in place (and not also in case steps reached maximum limit as employed by the sokoban env)
        done = True if box_target_distance == 0 else False

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

    def calc_box_target_distance(self):
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
                reward += dist

        return float(reward)