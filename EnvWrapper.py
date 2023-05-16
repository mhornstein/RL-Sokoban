import gym
import matplotlib.pyplot as plt
import random
import numpy as np

class EnvWrapper(gym.Env):
    '''
    This wrapper enables environment customization:
    * simulating stochastic environment.

    The implementation is based on the notebook:
    https://colab.research.google.com/drive/1ohrs6k0m17tPoQehdInUwwdj0g3H0vra?usp=sharing#scrollTo=XXqIDAd1SeMX

    but makes use of the Sokoban env instead.
    '''

    def __init__(self, sok, compliance=0.9):
        '''
        :param sok: Sokoban warehouse env
        :param compliance: when the agent takes a certain action, this is the probability that the environment will
        '''
        self.env = sok

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
        state = self.get_current_state()

        return state, reward, done, info

    def reset(self):
        self.env.reset()
        return self.get_current_state()

    def get_current_state(self): # Returns a simplified version of the state.
        rgb_display = self.env.render(mode='tiny_rgb_array')
        grayscale_display = np.dot(rgb_display[..., :3], [0.2989, 0.5870, 0.1140]) # refernece: https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
        return grayscale_display

    def render(self, mode='tiny_rgb_array'):
        plt.imshow(self.env.render(mode))

    def sample_action(self):
        return self.env.action_space.sample()

    def get_states_dim(self):
        current_state = self.get_current_state()
        return current_state.shape

    def action_space(self):
        return self.env.action_space

# for testing
if __name__ == '__main__': # TODO finish here
    from soko_pap import PushAndPullSokobanEnv
    import random

    random.seed(2)

    sok = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1, max_steps=500)
    env = EnvWrapper(sok, 1)

    action = 5 # Move up
    state, reward, done, info = env.step(action)
    # print(state, reward, done, info)
    env.render()

    action = env.sample_action()
    state, reward, done, info = env.step(action)
    # print(state, reward, done, info)
    env.render()

    input()

