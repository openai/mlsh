"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class MovementBanditsConv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # new action space = [left, right]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-10000000, 10000000, shape=(768,))

        self.realgoal = np.random.randint(0,2)

        self._seed()
        self.viewer = None
        self.reset()


        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def randomizeCorrect(self):
        self.realgoal = np.random.randint(0,2)

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if action == 1:
            self.state[0] += 20
        if action == 2:
            self.state[0] -= 20
        if action == 3:
            self.state[1] += 20
        if action == 4:
            self.state[1] -= 20


        distance = np.mean(abs(self.state[0] - self.goals[self.realgoal][0])**2 + abs(self.state[1] - self.goals[self.realgoal][1])**2)
        # reward = -distance / 5000
        if distance < 2500:
            reward = 1
        else:
            reward = 0

        return self.obs(), reward, False, {}

    def obs(self):
        self.state[0] = max(min(self.state[0], 350), 27)
        self.state[1] = max(min(self.state[1], 350), 27)

        blanks = np.zeros((16,16,3), dtype=np.float32)
        blanks[int(self.state[0]/25), int(self.state[1]/25)][0] = 1
        blanks[int(self.goals[0][0]/25), int(self.goals[0][1]/25)][1] = 1
        blanks[int(self.goals[1][0]/25), int(self.goals[1][1]/25)][2] = 1
        return np.reshape(blanks, (-1,))

    def _reset(self):
        self.state = [200, 200]
        self.goals = []
        for x in range(2):
            self.goals.append(np.random.randint(26, 350, size=(2,)))
        return self.obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 400
        screen_height = 400


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            self.man_trans = rendering.Transform()
            self.man = rendering.make_circle(10)
            self.man.add_attr(self.man_trans)
            self.man.set_color(.5,.5,.8)
            self.viewer.add_geom(self.man)

            self.goal_trans = []
            for g in range(len(self.goals)):
                self.goal_trans.append(rendering.Transform())
                self.goal = rendering.make_circle(20)
                self.goal.add_attr(self.goal_trans[g])
                self.viewer.add_geom(self.goal)
                self.goal.set_color(.5,.5,g*0.8)


        self.man_trans.set_translation(self.state[0], self.state[1])
        for g in range(len(self.goals)):
            self.goal_trans[g].set_translation(self.goals[g][0], self.goals[g][1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
