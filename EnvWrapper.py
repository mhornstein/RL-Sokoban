import gym
import matplotlib.pyplot as plt

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

        return state, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='rgb_array'):
        plt.imshow(self.env.render(mode))

    def sample_action(self):
        return self.env.action_space.sample()

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

