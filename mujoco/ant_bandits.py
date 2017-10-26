import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntBanditsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'ant_bandits.xml', 5)
        # self.realgoal = self.np_random.uniform(low=0, high=5, size=2)
        self.realgoal = np.array([5, 0]) if np.random.uniform() < 0.5 else np.array([0, 5])
        # self.realgoal = np.array([5, 0])
        # self.realgoal = np.array([3, 3])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        vec = self.get_body_com("torso")-self.get_body_com("target")
        reward_dist = -np.sqrt(np.linalg.norm(vec)) / 3000
        # reward_dist = -np.linalg.norm(vec)
        forward_reward = reward_dist
        # ctrl_cost = .5 * np.square(a).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        # survive_reward = 1.0
        # reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        reward = forward_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, False, {}

    def randomizeCorrect(self):
        # self.realgoal = self.np_random.uniform(low=0, high=5, size=2)
        self.realgoal = np.array([5, 0]) if np.random.uniform() < 0.5 else np.array([0, 5])
        # self.realgoal = np.array([0, 5])
        pass

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos.flat[:-2], qvel.flat[:-2], np.array([0])])

    def reset_model(self):
        # self.randomizeCorrect()
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[-2:] = self.realgoal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()
