import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SwimmerBanditsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'swimmer_bandits.xml', 4)
        self.realgoal = self.np_random.uniform(low=0, high=5, size=2)
        # self.realgoal = np.array([5, 0]) if np.random.uniform() < 0.5 else np.array([0, 5])
        # self.realgoal = np.array([0, 5])

    def _step(self, a):
        vec = self.get_body_com("mid")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum() * 0.0001
        reward = (reward_dist + reward_ctrl) * 0.001
        # reward = 0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, reward, False, {}

    def randomizeCorrect(self):
        self.realgoal = self.np_random.uniform(low=0, high=5, size=2)
        # self.realgoal = np.array([5, 0]) if np.random.uniform() < 0.5 else np.array([0, 5])
        # self.realgoal = np.array([5, 0])
        pass

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos.flat[:-2], qvel.flat[:-2]])

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[-2:] = self.realgoal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()
